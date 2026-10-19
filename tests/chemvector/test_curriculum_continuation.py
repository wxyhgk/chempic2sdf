"""Synthetic CPU loss/gradient tests; no deployment checkpoint or real corpus."""

import torch
import pytest

from chemvector.model import ChemVectorConfig, ChemVisionEncoder
from chemvector.spatial_atom_object import SpatialAtomObjectConfig, SpatialAtomObjectDetector
from chemvector.dense_bond_object import DenseBondObjectConfig, MultiScaleDenseBondObjectDetector
from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig
from chemvector.direct_atom_role import DirectAtomRoleConfig, DirectAtomRoleGate
from chemvector.direct_atom_text_bbox import DirectAtomTextBBox, DirectAtomTextBBoxConfig
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.training.curriculum.continuation import (
    ContinuationComponents, continuation_losses, gradient_report, structure_forward, train_step,
)
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.runtime import StepLimit


@pytest.fixture
def tiny():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ContinuationComponents(
        vision=ChemVisionEncoder(ChemVectorConfig(d_model=32, decoder_heads=4,
            vision_dims=(8, 16, 32, 64), vision_depths=(1, 1, 1, 1), dropout=0)),
        atom=SpatialAtomObjectDetector(SpatialAtomObjectConfig(input_channels=8, hidden_channels=8, maximum_atoms=8)),
        bond=MultiScaleDenseBondObjectDetector(DenseBondObjectConfig(input_channels=8, hidden_channels=8),
                                                stride8_channels=16, stride16_channels=32),
        set_head=ProposalObjectSetHead(ProposalObjectSetConfig(incidence=SpatialIncidenceConfig(
            atom_feature_dim=8, bond_feature_dim=8, hidden_dim=16, pair_hidden_dim=8), hidden_dim=16)),
        pair=DirectAtomPairBond(DirectAtomPairBondConfig(feature_channels=8, atom_feature_dim=8,
            longitudinal_samples=4, transverse_samples=3, hidden_dim=16, pair_chunk_size=16)),
        role=DirectAtomRoleGate(DirectAtomRoleConfig(radii_pixels=(12,), patch_size=8, hidden_dim=16)),
        bbox=DirectAtomTextBBox(DirectAtomTextBBoxConfig(radii_pixels=(16,), patch_size=8, hidden_dim=16)),
    )
    config = PredictedProposalConfig(maximum_atoms=8, maximum_bonds=8, atom_threshold=0, bond_threshold=0)
    pixels = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        atoms = structure_forward(model, pixels, config, -1e6)[3]
    # Test targets coincide with three independently predicted cells, enabling
    # positive-label plumbing tests without changing the proposal algorithm.
    anchors = atoms["atom_geometry"][:, :3, :2].detach().clone()
    assert len(torch.unique((anchors[0] * 16).long(), dim=0)) == 3
    refs = torch.tensor([[[0, 1], [1, 2]]])
    batch = {"pixel_values": pixels, "atom_anchors": anchors,
             "atom_mask": torch.ones(1, 3, dtype=torch.bool), "atom_roles": torch.tensor([[0, 1, 0]]),
             "atom_text_mask": torch.tensor([[False, True, False]]), "atom_text_center": anchors.clone(),
             "atom_text_bbox": torch.cat((anchors, torch.full_like(anchors, .04)), -1),
             "bond_atom_refs": refs, "bond_ports": anchors[:, refs[0]],
             "bond_mask": torch.ones(1, 2, dtype=torch.bool), "bond_chemical": torch.tensor([[0, 1]]),
             "bond_display": torch.tensor([[0, 1]]), "bond_direction": torch.zeros(1, 2, dtype=torch.long)}
    try:
        yield model, config, batch
    finally:
        torch.set_num_threads(previous)


def test_full_loss_reaches_all_existing_groups_without_mutating_weights(tiny, monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("GPU is forbidden")
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    model, config, batch = tiny
    before = state_digest(model)
    losses, coverage = continuation_losses(model, batch, config, atom_threshold=-1e6)
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["loss"].backward()
    gradients = gradient_report(model)
    assert set(gradients) == {"vision", "atom", "bond", "set_head", "pair", "role", "bbox"}
    assert all(r["grad_l2"] > 0 and r["finite"] for r in gradients.values())
    assert coverage["reachable_positive_pairs"] == 2
    assert coverage["role_rows"] == 3 and coverage["bbox_rows"] == 1
    assert not coverage["oracle_candidates_injected"]
    assert state_digest(model) == before


def test_parameter_groups_cover_each_trainable_once_and_unused_vision_stays_frozen(tiny):
    model, config, batch = tiny
    groups = model.parameter_groups()
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen and all(n.startswith(("vision.stages.5.", "vision.stages.6.", "vision.output_norm.", "vision.projection.")) for n in frozen)
    model.eval().train()
    assert all(not module.training for module in model.vision.stages[5:])
    # One synthetic optimizer update verifies the grouping, not real training.
    optimizer = torch.optim.AdamW(groups)
    losses, _ = continuation_losses(model, batch, config, atom_threshold=-1e6)
    losses["loss"].backward()
    optimizer.step()
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)


def test_rejected_atom_set_has_no_fabricated_pair_or_text_gradients(tiny):
    model, config, batch = tiny
    losses, coverage = continuation_losses(model, batch, config, atom_threshold=1e6)
    losses["loss"].backward()
    gradients = gradient_report(model)
    assert coverage["target_bonds"] == 2
    assert coverage["selected_atoms"] == coverage["reachable_positive_pairs"] == 0
    assert coverage["pair_rows"] == coverage["role_rows"] == coverage["bbox_rows"] == 0
    assert all(gradients[g]["with_grad"] == 0 for g in ("pair", "role", "bbox"))
    assert gradients["atom"]["grad_l2"] > 0


def test_single_atom_no_bond_batch_is_finite(tiny):
    model, config, batch = tiny
    for key in ("atom_anchors", "atom_mask", "atom_roles", "atom_text_mask", "atom_text_center", "atom_text_bbox"):
        batch[key] = batch[key][:, :1]
    for key in ("bond_atom_refs", "bond_ports", "bond_mask", "bond_chemical", "bond_display", "bond_direction"):
        batch[key] = batch[key][:, :0]
    losses, coverage = continuation_losses(model, batch, config, atom_threshold=-1e6)
    losses["loss"].backward()
    assert coverage["target_bonds"] == coverage["reachable_positive_pairs"] == coverage["bbox_rows"] == 0
    assert torch.isfinite(losses["loss"])


def test_proposal_forward_has_no_target_input_and_does_not_use_ground_truth(tiny):
    model, config, batch = tiny
    with torch.no_grad():
        first = structure_forward(model, batch["pixel_values"], config, 0)[-1]
        batch["atom_anchors"] = 1 - batch["atom_anchors"]
        second = structure_forward(model, batch["pixel_values"], config, 0)[-1]
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_nonfinite_gradients_fail_before_an_optimizer_step(tiny):
    model, _, _ = tiny
    p = next(model.atom.parameters())
    p.grad = torch.full_like(p, float("nan"))
    with pytest.raises(ValueError, match="Non-finite gradient"):
        gradient_report(model)


def synthetic_admission():
    return {"schema_version": "chemvector-curriculum-admission-0.1", "training_authorized": True,
            "external_overlap_status": "checked", "coverage_gaps": []}


def test_actual_update_primitive_checks_admission_before_forward(tiny, monkeypatch):
    model, config, batch = tiny
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not forward unadmitted data")
    monkeypatch.setattr("chemvector.training.curriculum.continuation.continuation_losses", forbidden)
    with pytest.raises(ValueError, match="must not start"):
        train_step(model, batch, config, None, admission={}, step_limit=StepLimit(1))
    with pytest.raises(ValueError, match="Diagnostic/quarantine"):
        train_step(model, batch, config, None, admission=synthetic_admission(), step_limit=StepLimit(1))


def test_synthetic_authorized_step_stops_at_exact_step_limit(tiny):
    model, config, batch = tiny
    batch.update(purpose="training", training_authorized=True)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=0)
    limit = StepLimit(1)
    before = state_digest(model)
    result = train_step(model, batch, config, optimizer, admission=synthetic_admission(),
                        step_limit=limit, atom_threshold=-1e6)
    assert result["optimizer_steps"] == 1 and result["coverage"]["reachable_positive_pairs"] == 2
    after = state_digest(model)
    assert after != before
    with pytest.raises(RuntimeError, match="budget exhausted"):
        train_step(model, batch, config, optimizer, admission=synthetic_admission(), step_limit=limit)
    assert state_digest(model) == after


def test_optimizer_cannot_accidentally_include_frozen_vision(tiny):
    model, config, batch = tiny
    batch.update(purpose="training", training_authorized=True)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="frozen parameters"):
        train_step(model, batch, config, optimizer, admission=synthetic_admission(), step_limit=StepLimit(1))
