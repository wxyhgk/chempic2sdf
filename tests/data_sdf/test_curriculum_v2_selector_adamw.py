"""Synthetic checks only; real-data runner never constructs an optimizer."""

import pytest
import torch

from tools import diagnose_curriculum_v02_selector_adamw as audit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cold_formula_matches_synthetic_adamw_first_step(dtype):
    gradient = torch.tensor([0., 2e-10, -.03, 7., -4.], dtype=dtype)
    before = gradient.clone()
    delta, info = audit.reference_direction(gradient.double())
    # Synthetic zero parameters avoid subtraction cancellation; real model is never stepped.
    p = torch.nn.Parameter(torch.zeros_like(gradient))
    p.grad = gradient.clone()
    torch.nn.utils.clip_grad_norm_([p], 1.)
    optimizer = torch.optim.AdamW([p], lr=audit.LEARNING_RATE, eps=1e-8, weight_decay=0, foreach=False)
    optimizer.step()
    assert torch.allclose(p.double(), delta, rtol=2e-6, atol=1e-12)
    assert torch.equal(gradient, before)
    assert info["selector_clip_coefficient"] < 1
    assert delta[0] == 0 and delta.abs().max() <= audit.LEARNING_RATE


@pytest.mark.parametrize("value", [torch.zeros(2), torch.tensor([float("nan")]), torch.ones(2, 2)])
def test_invalid_gradient_fails(value):
    with pytest.raises(ValueError):
        audit.reference_direction(value)


def test_old_probe_rule_includes_correct_boundary_and_retained_wrong_candidates():
    z = torch.tensor([1., -.249, -.6, .1, -.26, -.26, 9.])
    pos = torch.tensor([True, True, True, False, False, False, False])
    mask = torch.tensor([True, True, True, True, True, True, False])
    selected = audit.probe_selection(z, pos, mask, -.25)
    assert selected["positive_slots"] == [0, 1, 2]
    assert selected["hardest_negative"] == 3
    assert selected["nearest_threshold_negative"] == 4
    assert selected["probe_slots"] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        audit.probe_selection(z, pos, mask.float(), -.25)


def test_absent_negative_not_replaced():
    result = audit.probe_selection(torch.ones(2), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), -.25)
    assert result["hardest_negative"] is None and result["probe_slots"] == [0, 1]


def test_headroom_not_blanket_negative_decrease_and_exact_threshold_semantics():
    safe = audit.score_effect(-1.1, .01, False, -.25)
    assert safe["correct_side_headroom_change"] < 0
    assert safe["correct_side_headroom_linearized"] > 0
    assert not safe["linearized_correct_to_wrong"]
    lost = audit.score_effect(-.249, -.002, True, -.25)
    assert lost["linearized_correct_to_wrong"]
    rescued = audit.score_effect(-.5, .25, True, -.25)
    assert rescued["linearized_wrong_to_correct"] and rescued["selected_linearized"]
    bad = audit.score_effect(-.5, .25, False, -.25)
    assert bad["linearized_correct_to_wrong"]


def test_old_effect_uses_new_direction_not_own_loss_and_never_sets_grad():
    p = torch.nn.Parameter(torch.zeros(2))
    logits = (torch.tensor([-.249, -.7]) + torch.tensor([[1., 2.], [-1., 3.]]) @ p).unsqueeze(0)
    labels = dict(atom_presence_targets=torch.tensor([[True, False]]), atom_target_indices=torch.tensor([[0, -1]]))
    directions = {"new_a": torch.tensor([-.01, 0.], dtype=torch.float64),
                  "new_b": torch.tensor([0., .01], dtype=torch.float64)}
    snapshot = {key: value.clone() for key, value in directions.items()}
    rows = audit.measure_probes(logits, labels, (p,), {"probe_slots": [0, 1]}, directions, -.25)
    assert rows[0]["effects"]["new_a"]["logit_change"] == pytest.approx(-.01)
    assert rows[1]["effects"]["new_b"]["logit_change"] == pytest.approx(.03)
    assert rows[0]["effects"]["new_a"]["linearized_correct_to_wrong"]
    assert p.grad is None and torch.equal(p, torch.zeros(2))
    assert all(torch.equal(directions[k], v) for k, v in snapshot.items())


def test_budget_cpu_and_fresh_directory_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        audit.audit(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="Fresh budgeted"):
        audit.audit(tmp_path / "absent")
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="Fresh budgeted"):
        audit.audit(tmp_path)
