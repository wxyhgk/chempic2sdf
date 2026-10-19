"""Controlled predicted candidates exercise the actual continuation loss wiring."""

import pytest
import torch

from chemvector.training.curriculum import continuation
from chemvector.training.curriculum.gradient_preflight import state_digest
from tests.chemvector.test_curriculum_continuation import tiny


@pytest.fixture
def candidates(tiny, monkeypatch):
    model, config, batch = tiny
    forward = list(continuation.structure_forward(model, batch["pixel_values"], config, 0))
    anchors = torch.tensor([[[.25, .25], [.5, .5], [.75, .25]]])
    batch.update(atom_anchors=anchors, atom_text_center=anchors.clone(),
                 atom_text_bbox=torch.cat((anchors, torch.full_like(anchors, .04)), -1),
                 bond_ports=anchors[:, batch["bond_atom_refs"][0]])
    raw = {k: v.clone() for k, v in forward[3].items()}
    # Three retained candidates are 4 px from GT. Three rejected candidates
    # coincide with GT and would win an assignment over ALL raw candidates.
    raw["atom_geometry"] = raw["atom_geometry"].detach().clone()
    raw["atom_geometry"][0, :3, :2] = anchors[0] + torch.tensor([4 / config.image_size, 0])
    raw["atom_geometry"][0, 3:6, :2] = anchors[0]
    raw["atom_mask"].zero_()
    raw["atom_mask"][:, :6] = True
    selected = {**raw, "atom_mask": torch.zeros_like(raw["atom_mask"])}
    selected["atom_mask"][:, :3] = True
    forward[3], forward[-1] = raw, selected
    # Only structure predictions are controlled. Both matching stages, pair
    # targets/sampling and every downstream model/loss remain real functions.
    monkeypatch.setattr(continuation, "structure_forward", lambda *a, **k: tuple(forward))
    captured = {}
    for name in ("associate_predicted_proposals", "proposal_object_set_loss",
                 "build_pair_targets", "direct_atom_pair_loss"):
        original = getattr(continuation, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            captured[_name] = (args, result)
            return result

        monkeypatch.setattr(continuation, name, spy)
    return model, config, batch, raw, selected, captured


@pytest.mark.parametrize("recipe", ["legacy-all-v1", "freeze-role-v1"])
def test_rejected_closer_candidates_cannot_steal_selected_pair_or_text_labels(candidates, recipe):
    model, config, batch, raw, selected, captured = candidates
    model.configure_update_recipe(recipe)
    before = state_digest(model)
    raw["atom_mask"][:, 3:6] = False
    baseline, baseline_coverage = continuation.continuation_losses(model, batch, config)
    baseline_mapping = captured["build_pair_targets"][0][2].clone()

    raw["atom_mask"][:, 3:6] = True
    losses, coverage = continuation.continuation_losses(model, batch, config)
    association = captured["associate_predicted_proposals"][1]
    # Raw/set labels must still use raw matching (the rejected candidates win).
    assert association["atom_target_indices"].tolist() == [[-1, -1, -1, 0, 1, 2, -1, -1]]
    assert torch.equal(captured["proposal_object_set_loss"][0][1], association["atom_presence_targets"])
    pair_args, pair_targets = captured["build_pair_targets"]
    assert torch.equal(pair_args[2], baseline_mapping)
    assert pair_args[2].tolist() == [[0, 1, 2, -1, -1, -1, -1, -1]]
    assert pair_args[0].tolist() == [[0, 1], [0, 2], [1, 2]]
    assert pair_targets["presence"].tolist() == [True, False, True]
    assert pair_targets["chemical"][[0, 2]].tolist() == [0, 1]
    assert pair_targets["structured_negative"].tolist() == [False, True, False]
    for name in ("pair", "role", "bbox"):
        torch.testing.assert_close(losses[name], baseline[name], rtol=0, atol=0)
    for name in ("matched_selected_atoms", "reachable_positive_pairs", "role_rows", "bbox_rows"):
        assert coverage[name] == baseline_coverage[name]
    assert coverage["matched_selected_atoms"] == coverage["role_rows"] == 3
    assert coverage["reachable_positive_pairs"] == 2 and coverage["bbox_rows"] == 1
    assert coverage["role_update_rows"] == (0 if recipe == "freeze-role-v1" else 3)
    assert not coverage["oracle_candidates_injected"]

    # Check the real sampled pair loss: positives raise presence, the true
    # non-edge lowers it. No optimizer or real checkpoint is involved.
    pair_output, labels, _ = captured["direct_atom_pair_loss"][0]
    pair_output["presence_logits"].retain_grad()
    losses["loss"].backward()
    grad = pair_output["presence_logits"].grad
    assert bool((grad[labels] < 0).all()) and bool((grad[~labels] > 0).all())
    assert state_digest(model) == before
    assert any(p.grad is not None for p in model.bbox.parameters())
    assert any(p.grad is not None for p in model.role.parameters()) == (recipe == "legacy-all-v1")


@pytest.mark.parametrize("retained,matched,positives,text_rows", [
    ([], 0, 0, 0),
    ([0], 1, 0, 0),
    ([1], 1, 0, 1),
    ([0, 1], 2, 1, 1),
    ([0, 2], 2, 0, 0),
])
def test_partial_selected_sets_do_not_inject_atoms_or_bonds(candidates, retained, matched, positives, text_rows):
    model, config, batch, raw, selected, captured = candidates
    selected["atom_mask"].zero_()
    selected["atom_mask"][0, retained] = True
    losses, coverage = continuation.continuation_losses(model, batch, config)
    assert coverage["selected_atoms"] == coverage["matched_selected_atoms"] == matched
    assert coverage["reachable_positive_pairs"] == positives
    assert coverage["pair_rows"] == len(retained) * (len(retained) - 1) // 2
    assert coverage["bbox_rows"] == text_rows
    assert coverage["role_rows"] == matched
    assert int((captured["build_pair_targets"][0][2] >= 0).sum()) == matched
    assert all(bool(torch.isfinite(v)) for v in losses.values())


@pytest.mark.parametrize("distance,matched,positives", [(6.0, 3, 2), (6.25, 2, 1)])
def test_rematching_keeps_existing_six_pixel_gate(candidates, distance, matched, positives):
    model, config, batch, raw, selected, captured = candidates
    assert config.atom_match_radius_px == 6
    selected["atom_geometry"][0, 0, 0] = batch["atom_anchors"][0, 0, 0] + distance / config.image_size
    _, coverage = continuation.continuation_losses(model, batch, config)
    assert coverage["matched_raw_atoms"] == 3
    assert coverage["matched_selected_atoms"] == matched
    assert coverage["reachable_positive_pairs"] == positives


def test_selected_assignment_and_text_rows_stay_within_each_batch_sample(candidates, monkeypatch):
    model, config, batch, raw, selected, captured = candidates
    forward = continuation.structure_forward(model, batch["pixel_values"], config, 0)

    def duplicate(value):
        if isinstance(value, dict):
            return {k: duplicate(v) for k, v in value.items()}
        return torch.cat((value, value), dim=0)

    batch = duplicate(batch)
    forward = [duplicate(v) for v in forward]
    # Same coordinates, but sample 1 has different connectivity and no text.
    # The absent second bond remains padding, not a negative target atom.
    batch["bond_atom_refs"][1, 0] = torch.tensor([0, 2])
    batch["bond_ports"][1, 0] = batch["atom_anchors"][1, [0, 2]]
    batch["bond_mask"][1, 1] = False
    batch["atom_roles"][1].zero_()
    batch["atom_text_mask"][1].zero_()
    forward[-1]["atom_mask"][1, 1] = False
    monkeypatch.setattr(continuation, "structure_forward", lambda *a, **k: tuple(forward))
    _, coverage = continuation.continuation_losses(model, batch, config)
    args, targets = captured["build_pair_targets"]
    assert args[2].tolist() == [[0, 1, 2, -1, -1, -1, -1, -1], [0, -1, 2, -1, -1, -1, -1, -1]]
    assert targets["presence"][args[1] == 0].tolist() == [True, False, True]
    assert targets["presence"][args[1] == 1].tolist() == [True]
    assert coverage["matched_selected_atoms"] == coverage["role_rows"] == 5
    assert coverage["bbox_rows"] == 1 and coverage["reachable_positive_pairs"] == 3
