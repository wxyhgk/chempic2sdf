"""Freeze is opt-in and survives train/eval, steps and self-contained bundles."""

import pytest
import torch

from tests.chemvector.test_curriculum_continuation import tiny, synthetic_admission
from tests.chemvector.test_curriculum_bundle import configure
from chemvector.training.curriculum.bundle import save_bundle, load_bundle
from chemvector.training.curriculum.continuation import train_step
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.protocol import update_recipe, validate_protocol
from chemvector.training.curriculum.runtime import StepLimit


def test_freeze_role_leaves_six_groups_and_preserves_state_over_real_synthetic_update(tiny):
    model, config, batch = tiny
    role_before = state_digest(model.role)
    atom_before = state_digest(model.atom)
    model.configure_update_recipe("freeze-role-v1")
    model.eval().train()
    assert not model.role.training and model.bbox.training
    assert not any(p.requires_grad for p in model.role.parameters())
    assert {g["name"] for g in model.parameter_groups()} == {"vision", "atom", "bond", "set_head", "pair", "bbox"}
    batch.update(purpose="training", training_authorized=True)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=0)
    result = train_step(model, batch, config, optimizer, admission=synthetic_admission(),
                        step_limit=StepLimit(1), atom_threshold=-1e6)
    assert state_digest(model.role) == role_before and state_digest(model.atom) != atom_before
    assert result["losses"]["role"] == 0 and result["coverage"]["role_rows"] > 0
    assert result["coverage"]["role_update_rows"] == 0 and result["coverage"]["role_frozen"]
    assert all(p.grad is None for p in model.role.parameters())


def test_role_freeze_roundtrips_even_without_caller_recipe_metadata(tiny, tmp_path):
    model, config, _ = tiny
    configure(model)
    model.configure_update_recipe("freeze-role-v1")
    saved = save_bundle(tmp_path / "frozen.pt", model, config, provenance={}, progress={})
    restored, _, _ = load_bundle(saved["path"], expected_sha256=saved["sha256"])
    assert restored.update_recipe == "freeze-role-v1"
    restored.train()
    assert not restored.role.training and not any(p.requires_grad for p in restored.role.parameters())
    assert state_digest(model) == state_digest(restored)


def test_recipe_cannot_be_switched_after_gradients_or_lied_about_in_bundle(tiny, tmp_path):
    model, config, _ = tiny
    configure(model)
    model.configure_update_recipe("freeze-role-v1")
    with pytest.raises(ValueError, match="disagree"):
        save_bundle(tmp_path / "bad.pt", model, config, provenance={},
                    progress={"recipe": {"update_recipe": "legacy-all-v1"}})
    assert not (tmp_path / "bad.pt").exists()
    next(model.atom.parameters()).grad = torch.ones_like(next(model.atom.parameters()))
    with pytest.raises(ValueError, match="before gradients"):
        model.configure_update_recipe("legacy-all-v1")


def test_protocol_keeps_legacy_compatibility_and_requires_explicit_freeze_recipe():
    legacy = dict(schema_version="chemvector-fixed-continuation-protocol-0.1", max_steps=1200, batch_size=1,
                  validate_every=400, patience=2, wall_seconds=1800, max_gpus=1,
                  selection_domains=["full_molecule"], automatic_resume=False)
    assert update_recipe({}) == update_recipe({"training_protocol": legacy}) == "legacy-all-v1"
    frozen = dict(legacy, schema_version="chemvector-fixed-continuation-protocol-0.2", update_recipe="freeze-role-v1")
    validate_protocol(frozen)
    assert update_recipe({"training_protocol": frozen}) == "freeze-role-v1"
    for value in ({**frozen, "update_recipe": "legacy-all-v1"}, {k: v for k, v in frozen.items() if k != "update_recipe"},
                  dict(legacy, update_recipe="freeze-role-v1")):
        with pytest.raises(ValueError):
            validate_protocol(value)


@pytest.mark.parametrize("recipe", ["legacy-all-v1", "freeze-role-v1"])
def test_gradient_preflight_records_recipe_without_updating_weights(tiny, tmp_path, monkeypatch, recipe):
    from chemvector.training.curriculum import checkpoints, data, gradient_preflight

    model, config, batch = tiny
    provenance = {"atom_threshold": -1e6, "bond_threshold": -.5, "checkpoints": {}}
    monkeypatch.setattr(checkpoints, "load_current_components", lambda: (model, config, provenance))
    monkeypatch.setattr(checkpoints, "file_digest", lambda path: "synthetic")
    monkeypatch.setattr(data, "load_sample", lambda *a, **k: batch)
    monkeypatch.setattr(data, "CurriculumCollator", lambda **k: lambda samples: samples[0])
    report = gradient_preflight.audit(tmp_path, ["synthetic"], tmp_path, update_recipe=recipe)
    assert report["state_before_sha256"] == report["state_after_sha256"]
    assert report["role_state_before_sha256"] == report["role_state_after_sha256"]
    assert report["update_recipe"] == recipe and report["optimizer_steps"] == 0
    assert report["group_gradient_coverage_passed"]
    if recipe == "freeze-role-v1":
        assert len(report["parameter_groups"]) == 6
        assert report["samples"][0]["coverage"]["role_update_rows"] == 0
