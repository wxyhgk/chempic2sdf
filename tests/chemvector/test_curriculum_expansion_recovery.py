"""Small synthetic recovery tests; no protected corpora or GPU."""

from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from Data.sdf.chemvector_curriculum_v0_1.build import _render_job, _write, finish_build
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import render
from chemvector.training.curriculum.expansion import write_json
from chemvector.training.curriculum.expansion_plan import selected_profiles
from chemvector.training.curriculum.expansion_recovery import recovery_inputs, run
from chemvector.training.curriculum.expansion_recovery_checks import vector_batch, verified_bundle
from chemvector.training.curriculum.full_molecule_data import file_hash


def entry(smiles="CCO"):
    return merged_records([dict(record_id="test", name="test", stage="full_molecule", concepts=[], smiles=smiles)])[0]


def save_bundle(root, molecule, profile):
    (root / "samples").mkdir(parents=True, exist_ok=True)
    files, meta = render(molecule, profile, typography="stereo-aware-v1")
    for suffix, data in files.items():
        _write(root / "samples" / (meta["sample_key"] + suffix), data)
    return files, meta


def test_vector_gate_known_failure_excludes_all_profiles_but_default_builder_stays_strict():
    bad = entry("CC(=O)N(C)[C@@H](Cc1c[nH]c2ccccc12)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(N)=O")
    before = deepcopy(bad)
    rows = vector_batch(([entry(), bad], selected_profiles(), "stereo-aware-v1"))
    assert rows[0]["supported"] and not rows[1]["supported"]
    assert rows[1]["failures"][0]["reason"] == "Auxiliary junction has no unique incident-bond geometric witness"
    assert len(rows[1]["checked_profiles"]) == 6 and not rows[1]["chemical_labels_modified"]
    assert bad == before
    with pytest.raises(ValueError, match="unique incident-bond"):
        _render_job((bad, selected_profiles()[0], "stereo-aware-v1"))


def test_vector_gate_unknown_error_is_fatal(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unknown new invariant")
    monkeypatch.setattr("chemvector.training.curriculum.expansion_recovery_checks.prepare_drawing", fail)
    with pytest.raises(ValueError, match="unknown new invariant"):
        vector_batch(([entry()], selected_profiles(), "stereo-aware-v1"))


def test_verified_bundle_preserves_bytes_and_checks_extra_masks_and_entry(tmp_path):
    molecule, profile = entry(), selected_profiles()[0]
    files, meta = save_bundle(tmp_path, molecule, profile)
    assert verified_bundle(tmp_path, molecule, profile, "stereo-aware-v1") == (files, meta)
    changed = {**molecule, "names": ["changed source metadata"]}
    with pytest.raises(ValueError, match="fixed entry"):
        verified_bundle(tmp_path, changed, profile, "stereo-aware-v1")
    mask = tmp_path / "samples" / (meta["sample_key"] + ".bond.png")
    mask.write_bytes(b"corrupt mask not among the adapter's six main payloads")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verified_bundle(tmp_path, molecule, profile, "stereo-aware-v1")


def test_finish_build_cannot_publish_incomplete_or_duplicate_jobs(tmp_path):
    with pytest.raises(ValueError, match="job accounting"):
        finish_build(tmp_path, [entry()], selected_profiles(), [], [], {},
                     typography="stereo-aware-v1", task_timeout=60)
    assert not (tmp_path / "summary.json").exists()


def failed_fixture(tmp_path):
    previous = tmp_path / "failed"
    previous.mkdir()
    molecule = entry()
    row = dict(molecule_id="test", canonical_isomeric_smiles="CCO", split="train")
    molecule["source_provenance"] = dict(source_molecule=row, commitment_sha256="fixture")
    plan = dict(schema_version="chemvector-expansion-plan-0.1", selected=[row],
        profiles=[asdict(p) for p in selected_profiles()], typography="stereo-aware-v1",
        training_authorized=False, planned_images=6, minimum_qualified_image_target=1, input_sha256={})
    write_json(previous / "plan.json", plan)
    chemistry = dict(plan_sha256=file_hash(previous / "plan.json"), replacement_molecules_added=0,
                     supported_molecules=1, failures=[])
    write_json(previous / "chemistry-preflight.json", chemistry)
    provenance = dict(expansion_plan_sha256=chemistry["plan_sha256"], commitment_sha256="fixture",
                      chemistry_preflight_sha256=file_hash(previous / "chemistry-preflight.json"))
    files, meta = save_bundle(previous / "dataset", molecule, selected_profiles()[0])
    for name, value in (("molecules.json", [molecule]), ("profiles.json", plan["profiles"]),
                        ("source-provenance.json", provenance)):
        write_json(previous / "dataset" / name, value)
    write_json(previous / "runtime.json", dict(gpu_devices=[], training=False,
        status="worker_failed", elapsed_seconds=10))
    return previous, files, meta


def test_recovery_budget_and_selection_bindings_cannot_be_reset(tmp_path):
    previous, _, _ = failed_fixture(tmp_path)
    recovery_inputs(previous, [previous], 20, 30)
    with pytest.raises(ValueError, match="budget"):
        recovery_inputs(previous, [previous], 21, 30)
    with pytest.raises(ValueError, match="Distinct"):
        recovery_inputs(previous, [previous, previous], 10, 30)
    path = previous / "dataset/molecules.json"
    values = json.loads(path.read_text())
    values[0]["canonical_smiles"] = "CCN"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="original train-only selection"):
        recovery_inputs(previous, [previous], 20, 30)


def test_recovery_only_renders_missing_and_publishes_exact_accounting(tmp_path, monkeypatch):
    previous, files, meta = failed_fixture(tmp_path)
    called = []
    def sequential(function, jobs, **kwargs):
        for job in jobs:
            if function is _render_job:
                called.append(job[1].profile_id)
            yield function(job)
    monkeypatch.setattr("Data.sdf.chemvector_curriculum_v0_1.bounded.bounded_map", sequential)
    output = tmp_path / "recovered"
    output.mkdir()
    report = run(Namespace(previous=previous, prior_runs=[previous], output=output,
                           wall_seconds=20, total_wall_seconds=30, workers=1))
    assert called == [p.profile_id for p in selected_profiles()[1:]]
    assert report["reused_images"] == report["verified_old_complete_images"] == 1
    assert report["newly_rendered_images"] == 5 and report["render_accepted_images"] == 6
    assert not report["training_authorized"] and not report["gpu_used"]
    for suffix, data in files.items():
        assert (output / "dataset/samples" / (meta["sample_key"] + suffix)).read_bytes() == data
        assert (previous / "dataset/samples" / (meta["sample_key"] + suffix)).read_bytes() == data
    assert not (previous / "dataset/summary.json").exists()
    journal = [json.loads(line) for line in (output / "sample-journal.jsonl").read_text().splitlines()]
    assert len(journal) == 6 and journal[0]["status"] == "reused"
    summary = json.loads((output / "dataset/summary.json").read_text())
    assert summary["planned_images"] == summary["accepted_images"] == 6
