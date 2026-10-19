"""Synthetic selector gradient contrast; no real checkpoint or optimizer."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from chemvector.predicted_incidence import associate_predicted_proposals
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.gradient_preflight import state_digest
from tests.chemvector.test_curriculum_continuation import tiny
from tools import diagnose_curriculum_v02_selector_gradients as audit


@pytest.mark.parametrize("values", [[1., 1., 1., 1.], [-4., -4., -4., -4.]])
def test_count_only_detaches_positive_gradients_and_keeps_values(values):
    z = torch.tensor([values], requires_grad=True)
    positive = torch.tensor([[True, True, False, False]])
    mask = torch.tensor([[True, True, True, False]])
    original = F.smooth_l1_loss((z.sigmoid() * mask).sum(-1), positive.sum(-1).float())
    delta = audit.count_contrast(z, positive, mask, original)
    changed = .1 * original + delta
    assert changed.detach() == .1 * original.detach() and z.grad is None
    before = torch.autograd.grad(.1 * original, z, retain_graph=True)[0]
    after = torch.autograd.grad(changed, z)[0]
    assert torch.equal(after[positive], torch.zeros_like(after[positive]))
    assert torch.equal(after[~positive], before[~positive])
    assert after[0, -1] == 0


@pytest.mark.parametrize("positive", [[[True, True]], [[False, False]]])
def test_count_contrast_homogeneous_labels(positive):
    z = torch.tensor([[.2, .4]], requires_grad=True)
    labels = torch.tensor(positive)
    mask = torch.ones_like(labels)
    count = F.smooth_l1_loss(z.sigmoid().sum(-1), labels.sum(-1).float())
    delta = audit.count_contrast(z, labels, mask, count)
    assert float(delta.detach()) == 0
    grad = torch.autograd.grad(.1 * count + delta, z)[0]
    if labels.all():
        assert torch.equal(grad, torch.zeros_like(grad))


def test_count_rejects_mismatched_original():
    z = torch.zeros(1, 2, requires_grad=True)
    mask = torch.ones_like(z, dtype=torch.bool)
    with pytest.raises(ValueError, match="Original count"):
        audit.count_contrast(z, mask, mask, torch.tensor(99.))
    with pytest.raises(ValueError, match="Malformed"):
        audit.count_contrast(z, mask.float(), mask, torch.tensor(1.))


def test_shared_parameter_direction_not_same_as_independent_logit_gradient():
    theta = torch.nn.Parameter(torch.tensor([0.]))
    unused = torch.nn.Parameter(torch.tensor([8.]))
    # dL/dz0 > 0, but shared parameters make its actual -grad(L) direction up.
    logits = torch.stack((theta[0], 2 * theta[0]))
    loss = logits[0] - 2 * logits[1]
    gradient = audit.flat_gradient(loss, (theta, unused))
    jacobian = audit.flat_gradient(logits[0], (theta, unused))
    result = audit.direction_record(jacobian, {"shared": gradient})
    assert result["shared"]["unit_negative_gradient_derivative"] == pytest.approx(1.)
    assert gradient.tolist() == [-3., 0.]
    assert theta.grad is None and unused.grad is None and float(theta.detach()) == 0


def test_zero_gradient_has_no_unit_direction():
    result = audit.direction_record(torch.ones(2), {"zero": torch.zeros(2)})
    assert result["zero"]["unit_negative_gradient_derivative"] is None


def test_full_selector_contrast_preserves_state_predictions_and_frozen_front(tiny):
    model, config, batch = tiny
    model.eval().requires_grad_(False)
    model.set_head.requires_grad_(True)
    before = state_digest(model)
    with torch.no_grad():
        front = structure_forward(model, batch["pixel_values"], config, -.25)
    raw, bonds = front[3:5]
    output = audit.selector_call(model.set_head, raw, bonds)
    assert torch.equal(output["atom_presence_logits"].detach(), front[5]["atom_presence_logits"])
    labels = associate_predicted_proposals(raw, bonds, batch, config)
    existing = proposal_object_set_loss(output, labels["atom_presence_targets"], labels["bond_presence_targets"],
        raw["atom_mask"], bonds["bond_mask"], labels["target_atom_refs"], labels["bond_supervision_mask"])
    partial = existing["atom_presence_loss"] + .5 * existing["atom_ranking_loss"] + .1 * existing["atom_count_loss"]
    derivatives = torch.autograd.grad(partial, output["atom_presence_logits"], retain_graph=True)[0]
    rows = [dict(candidate=i, selected=bool(front[6]["atom_mask"][0, i]),
        raw_positive=bool(labels["atom_presence_targets"][0, i]), raw_target=int(labels["atom_target_indices"][0, i]),
        atom_only_logit_derivatives={"atom_only_total": float(derivatives[0, i])}) for i in range(8)]
    prior = dict(relevant_candidates=rows, target_details=[{"near_raw_ids": [0, 1, 2]}])
    old = deepcopy(prior)
    result = audit.parameter_probe(output, raw, bonds, labels, tuple(model.set_head.parameters()), prior,
        atom_threshold=-.25, bond_threshold=-.5)
    assert state_digest(model) == before and prior == old
    assert all(p.grad is None for p in model.parameters())
    assert not raw["atom_features"].requires_grad and not bonds["bond_features"].requires_grad
    assert result["objectives"][audit.ARMS[0]] == result["objectives"][audit.ARMS[1]]
    assert result["gradient_sum_relative_error"] < 1e-3
    assert sum(r["count"] for r in result["groups"].values()) == 8
    for group, record in result["groups"].items():
        members = [r for r in result["candidates"] if r["category"] == group]
        if not members:
            continue
        assert len(members) == record["count"]
        for arm in audit.ARMS:
            mean = sum(r["directions"][arm]["unit_negative_gradient_derivative"] for r in members) / len(members)
            assert record["directions"][arm]["unit_negative_gradient_derivative"] == pytest.approx(mean, abs=1e-4, rel=1e-4)


def test_no_unbudgeted_gpu_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        audit.audit(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="fresh budgeted"):
        audit.audit(tmp_path / "not-created")
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh budgeted"):
        audit.audit(tmp_path)
