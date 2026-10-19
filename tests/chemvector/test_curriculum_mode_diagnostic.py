"""CPU controls for the read-only train/eval probe; no real data or weights."""

import pytest
import torch

from chemvector.training.curriculum.mode_diagnostic import MODES, measure, temporary_mode
from chemvector.training.curriculum.gradient_preflight import state_digest
from tests.chemvector.test_curriculum_continuation import tiny


@pytest.mark.parametrize("mode", MODES)
def test_temporary_modes_restore_mixed_flags_and_never_unfreeze_role(tiny, mode):
    model, _, _ = tiny
    model.configure_update_recipe("freeze-role-v1")
    before = [m.training for m in model.modules()]
    gradient_flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(RuntimeError, match="intentional"):
        with temporary_mode(model, mode):
            assert model.training == mode.startswith("train")
            assert not model.role.training
            assert all(not m.training for m in model.vision.stages[5:])
            assert gradient_flags == [p.requires_grad for p in model.parameters()]
            dropout = model.vision.stages[0][0].dropout
            assert dropout.training == (mode in ("train", "eval_vision_dropout_on"))
            raise RuntimeError("intentional")
    assert before == [m.training for m in model.modules()]
    assert gradient_flags == [p.requires_grad for p in model.parameters()]


def test_unknown_mode_fails_without_state_changes(tiny):
    model, _, _ = tiny
    before = [m.training for m in model.modules()]
    with pytest.raises(ValueError, match="Unknown"):
        with temporary_mode(model, "arbitrary"):
            pytest.fail("Invalid mode entered")
    assert before == [m.training for m in model.modules()]


def test_dropout_cross_controls_replay_without_rng_or_weight_mutation(tiny):
    model, config, batch = tiny
    # Only this synthetic fixture's dropout p changes; the real audit never
    # edits p or any other saved model configuration.
    for module in model.vision.modules():
        if isinstance(module, torch.nn.Dropout2d):
            module.p = .1
    model.configure_update_recipe("freeze-role-v1")
    before, rng = state_digest(model), torch.get_rng_state().clone()
    results = {mode: measure(model, batch, config, -1e6, mode, 17) for mode in MODES}
    assert results["eval"]["prediction_sha256"] == results["train_vision_dropout_off"]["prediction_sha256"]
    assert results["train"]["prediction_sha256"] == results["eval_vision_dropout_on"]["prediction_sha256"]
    assert results["train"]["prediction_sha256"] != results["eval"]["prediction_sha256"]
    assert measure(model, batch, config, -1e6, "train", 17) == results["train"]
    assert measure(model, batch, config, -1e6, "train", 18)["prediction_sha256"] != results["train"]["prediction_sha256"]
    assert torch.equal(rng, torch.get_rng_state()) and state_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_no_selected_candidates_are_counted_not_fabricated(tiny):
    model, config, batch = tiny
    result = measure(model, batch, config, 1e6, "eval", 17)
    assert result["target_atoms"] == 3 and result["target_bonds"] == 2
    assert result["selected_atoms"] == result["selected_matched"] == result["pair_candidates"] == 0
    assert result["reachable_positive_pairs"] == result["bbox_rows"] == 0
    assert result["matched_target_ids"] == result["reachable_target_bond_ids"] == []
