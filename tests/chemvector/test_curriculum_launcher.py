"""CPU-only synthetic job integration; no real weights or real admission."""

import json
from pathlib import Path

import pytest
from rdkit import Chem
import torch

from tests.chemvector.test_curriculum_continuation import tiny
from tests.chemvector.test_curriculum_bundle import configure
from tests.chemvector.test_curriculum_release import released
from chemvector.training.curriculum.bundle import load_bundle
from chemvector.training.curriculum.evaluation import canonical_molblock, summarize_predictions
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.launcher import main, run_training


def test_training_entry_does_not_shadow_historical_train_module():
    import chemvector.train
    assert Path(chemvector.train.__file__).parts[-3:] == ("training", "general", "train.py")


def test_partial_real_style_report_cannot_launch_or_create_outputs(tmp_path, monkeypatch):
    admission = tmp_path / "partial.json"
    admission.write_text(json.dumps({"schema_version": "chemvector-curriculum-admission-0.1",
        "training_authorized": False, "external_overlap_status": "partially_checked", "coverage_gaps": ["missing index"]}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before launching a worker")
    monkeypatch.setattr("subprocess.Popen", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="must not start"):
        main(["--admission", str(admission), "--manifest", str(tmp_path / "absent.json"),
              "--output", str(output), "--gpu-device", "0", "--max-steps", "1", "--wall-seconds", "5"])
    assert not output.exists()


@pytest.mark.parametrize("recipe", ["legacy-all-v1", "freeze-role-v1"])
def test_synthetic_job_trains_validates_stops_and_restores_complete_bundle(tiny, released, tmp_path, monkeypatch, recipe):
    model, config, _ = tiny
    configure(model)
    manifest, admission, mp, ap = released
    if recipe == "freeze-role-v1":
        from tests.chemvector.test_curriculum_release import bind
        manifest["training_protocol"] = dict(schema_version="chemvector-fixed-continuation-protocol-0.2",
            update_recipe=recipe, max_steps=5, batch_size=1, validate_every=1, patience=2,
            wall_seconds=60, max_gpus=1, selection_domains=["full_molecule"], automatic_resume=False)
        bind(manifest, admission, mp, ap)
    output = tmp_path / "run"
    output.mkdir()
    provenance = {"checkpoints": {}, "atom_threshold": -1e6, "bond_threshold": -.5,
                  "pair_threshold": .5, "role_threshold": .5, "synthetic_test_only": True}
    initial = state_digest(model)
    initial_role = state_digest(model.role)
    calls = []
    good = Chem.MolToMolBlock(Chem.MolFromSmiles("CCl"))
    class FakePredictor:
        def __call__(self, model, config, pixels, image):
            assert not model.training and not torch.is_grad_enabled()
            calls.append(True)
            return {"molblock": None if len(calls) == 1 else good, "fallback": False, "error": None}
    def no_cuda(*args, **kwargs):
        raise AssertionError("GPU access is forbidden")
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    report = run_training(mp, ap, output, max_steps=5, validate_every=1, patience=2,
        component_loader=lambda: (model, config, provenance), predictor_factory=lambda *args: FakePredictor())
    assert report["optimizer_steps"] == 3 and report["stop_reason"] == "validation_patience"
    assert report["domain_seen"] == {"full_molecule": 2, "curriculum": 1}
    assert report["selected_metrics"]["rescued"] == 1 and report["selected_metrics"]["broken"] == 0
    assert report["selected"]["path"].endswith("step-000001.pt")
    assert not report["gpu_used"] and not report["deployed"]
    assert state_digest(model) != initial
    restored, _, payload = load_bundle(report["last"]["path"], expected_sha256=report["last"]["sha256"])
    assert state_digest(restored) == state_digest(model)
    assert payload["progress"]["optimizer_steps"] == 3
    assert report["update_recipe"] == restored.update_recipe == recipe
    if recipe == "freeze-role-v1":
        assert report["frozen_initial_states"] == report["frozen_final_states"] == {"role": initial_role}
        assert state_digest(restored.role) == initial_role
        assert not restored.role.training and not any(p.requires_grad for p in restored.role.parameters())
    assert len((output / "steps.jsonl").read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="fresh supervised"):
        run_training(mp, ap, output, max_steps=5, validate_every=1, patience=2)


def test_check_only_verifies_payloads_without_loading_a_model(released, monkeypatch):
    _, _, mp, ap = released
    def forbidden(*args, **kwargs):
        raise AssertionError("Check-only must not load a model")
    monkeypatch.setattr("chemvector.training.curriculum.checkpoints.load_current_components", forbidden)
    assert main(["--manifest", str(mp), "--admission", str(ap), "--check-only"]) == 0


def test_no_improvement_keeps_initial_weights_selected(tiny, released, tmp_path):
    model, config, _ = tiny
    configure(model)
    _, _, mp, ap = released
    output = tmp_path / "no-improvement"
    output.mkdir()
    provenance = {"checkpoints": {}, "atom_threshold": -1e6, "bond_threshold": -.5}
    before = state_digest(model)
    def invalid(*args):
        return {"molblock": None, "fallback": False, "error": "synthetic invalid prediction"}
    report = run_training(mp, ap, output, max_steps=3, validate_every=1, patience=1,
        component_loader=lambda: (model, config, provenance), predictor_factory=lambda *args: invalid)
    assert report["optimizer_steps"] == 1
    assert report["selected"]["path"].endswith("initial.pt")
    assert report["selected_metrics"]["rescued"] == report["selected_metrics"]["broken"] == 0
    restored, _, _ = load_bundle(report["selected"]["path"], expected_sha256=report["selected"]["sha256"])
    assert state_digest(restored) == before


def test_validation_keeps_invalid_outputs_and_tracks_regressions():
    baseline = [{"key": "a", "valid": True, "exact": True, "fallback": False},
                {"key": "b", "valid": False, "exact": False, "fallback": False}]
    candidate = [{"key": "b", "valid": True, "exact": True, "fallback": True},
                 {"key": "a", "valid": False, "exact": False, "fallback": False}]
    result = summarize_predictions(candidate, baseline)
    assert result == {"images": 2, "exact": 1, "valid": 1, "fallback": 1,
                      "sdf_exact_rate": .5, "rescued": 1, "broken": 1}
    with pytest.raises(ValueError, match="identities changed"):
        summarize_predictions(candidate[:1], baseline)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_predictions(candidate + candidate[:1])


@pytest.mark.parametrize("smiles", ["CCO", "[13CH3]CO", "C[O-]", "F[C@H](Cl)Br", "[Na+].[Cl-]"])
def test_sdf_comparison_preserves_chemical_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert canonical_molblock(Chem.MolToMolBlock(mol)) == Chem.MolToSmiles(mol, isomericSmiles=True)
    assert canonical_molblock(None) is None
    assert canonical_molblock("invalid molblock") is None


def test_fixed_run_does_not_select_weights_that_only_improve_curriculum(tiny, released, tmp_path):
    from copy import deepcopy
    from tests.chemvector.test_curriculum_release import bind
    model, config, _ = tiny
    configure(model)
    m, a, mp, ap = released
    # Two synthetic views of one held-out molecule. Fake predictions deliberately
    # improve the teaching view while breaking the full-molecule view.
    m["sources"]["synthetic-teaching-view"] = m["sources"]["synthetic"]
    extra = deepcopy(m["samples"][-1])
    extra.update(source="synthetic-teaching-view", domain="curriculum")
    m["samples"].append(extra)
    m["training_protocol"] = {"schema_version": "chemvector-fixed-continuation-protocol-0.1",
        "max_steps": 1, "batch_size": 1, "validate_every": 1, "patience": 1,
        "wall_seconds": 60, "max_gpus": 1, "selection_domains": ["full_molecule"], "automatic_resume": False}
    a["checked_images"] = 4
    bind(m, a, mp, ap)
    output = tmp_path / "fixed-selection"
    output.mkdir()
    good, calls = Chem.MolToMolBlock(Chem.MolFromSmiles("CCl")), []
    def predict(*args):
        calls.append(True)
        return {"molblock": good if len(calls) in {1, 4} else None, "fallback": False, "error": None}
    provenance = {"checkpoints": {}, "atom_threshold": -1e6, "bond_threshold": -.5}
    report = run_training(mp, ap, output, max_steps=1, validate_every=1, patience=1,
        component_loader=lambda: (model, config, provenance), predictor_factory=lambda *args: predict)
    assert report["selected"]["path"].endswith("initial.pt")
    assert report["baseline_selection"]["images"] == 1
    last = json.loads((output / "validation-000001.json").read_text())
    assert last["metrics"]["exact"] == report["baseline"]["exact"] == 1
    assert last["selection_metrics"]["exact"] == 0
    assert last["selection_metrics"]["broken"] == 1


def test_changed_protocol_fails_before_model_or_output(released, tmp_path):
    from tests.chemvector.test_curriculum_release import bind
    m, a, mp, ap = released
    m["training_protocol"] = {"schema_version": "chemvector-fixed-continuation-protocol-0.1",
        "max_steps": 1, "batch_size": 1, "validate_every": 1, "patience": 1,
        "wall_seconds": 60, "max_gpus": 1, "selection_domains": ["full_molecule"], "automatic_resume": False}
    bind(m, a, mp, ap)
    def forbidden():
        raise AssertionError("Protocol mismatch must precede model load")
    with pytest.raises(ValueError, match="hash-bound training protocol"):
        run_training(mp, ap, tmp_path / "not-created", max_steps=2, component_loader=forbidden)
