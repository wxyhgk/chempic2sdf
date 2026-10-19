"""Independent auditor tests built only from tiny synthetic molecules."""

from dataclasses import replace
from hashlib import sha256
import json
import shutil

import pytest

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from Data.sdf.chemvector_curriculum_v0_2 import build as builder
from tools.audit_curriculum_expansion_v02 import (
    AuditError, audit, check_dataset, decode, source_path, validate_sdf,
)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    output = tmp_path_factory.mktemp("audit-v02-fixture") / "dataset"
    examples = ["CCO", "F[C@H](Cl)Br", "F[C@@H](Cl)Br", "C/C=C/C", "CC=CC", "[13CH3]CO"]
    entries = merged_records([dict(record_id=f"test_{i}", name=f"example {i}", stage="atoms_bonds",
                                   concepts=[], smiles=smiles) for i, smiles in enumerate(examples)])
    patch = pytest.MonkeyPatch()
    patch.setenv("CUDA_VISIBLE_DEVICES", "")
    patch.setattr(builder, "entries", lambda: entries)
    patch.setattr(builder, "profiles", lambda: [Profile(), replace(Profile(), profile_id="font_small", font_size=14)])
    patch.setattr(builder, "bounded_map", lambda function, jobs, **kwargs: (function(job) for job in jobs))
    original_job = builder._render_job
    def inject_one_known_rejection(job):
        entry, profile, _ = job
        if entry["molecule_id"] == entries[0]["molecule_id"] and profile.profile_id == "font_small":
            return None, None, {"molecule_id": entry["molecule_id"], "profile_id": profile.profile_id,
                                "canonical_smiles": entry["canonical_smiles"],
                                "reason": "Typography clearance would erase a bond segment"}
        return original_job(job)
    patch.setattr(builder, "_render_job", inject_one_known_rejection)
    try:
        builder.build(output, workers=1)
    finally:
        patch.undo()
    return output


@pytest.fixture
def copy_dataset(dataset, tmp_path):
    copy = tmp_path / "dataset"
    shutil.copytree(dataset, copy)
    return copy


def quiet(*args, **kwargs):
    pass


def test_complete_synthetic_audit_preserves_quarantine_and_stereo_groups(dataset, tmp_path):
    report = audit(dataset, tmp_path / "audit.json")
    assert report["status"] == "passed_artifact_integrity_only"
    assert report["checked"]["planned_images"] == 12
    assert report["checked"]["accepted_images"] + report["checked"]["rejected_images"] == 12
    assert report["checked"]["unique_molecules"] == 6
    assert report["checked"]["rejected_images"] == 1
    # Enantiomers merge; specified/unspecified butene merge; isotopic ethanol does not.
    assert report["stereo_removed_isotope_preserved_identities"] == 4
    assert len(report["multiple_stereo_variants_per_connectivity"]) == 2
    assert report["training_authorized"] is False
    assert report["formal_training_admission"] == "not_granted"
    assert report["renderer_exit_status"] == "not_inferred_from_artifacts"
    assert report["auditor_source_sha256"]
    assert report["manifest_sha256"]["plan.json"]
    assert report["gpu_used"] is False
    with pytest.raises(FileExistsError):
        audit(dataset, tmp_path / "audit.json")


def test_missing_or_incomplete_summary_fails_and_cannot_create_success_receipt(copy_dataset, tmp_path):
    (copy_dataset / "summary.json").unlink()
    output = tmp_path / "failed.json"
    with pytest.raises(AuditError, match="Missing"):
        audit(copy_dataset, output)
    assert json.loads(output.read_text())["status"] == "failed"


@pytest.mark.parametrize("field,value", [("generation_complete", False), ("accepted_images", 12345),
                                        ("training_authorized", True), ("plan_sha256", "0" * 64)])
def test_summary_contradictions_fail(copy_dataset, field, value):
    path = copy_dataset / "summary.json"
    data = json.loads(path.read_text())
    data[field] = value
    write_json(path, data)
    with pytest.raises(AuditError):
        check_dataset(copy_dataset, progress=quiet)


def test_duplicate_sample_denominator_fails(copy_dataset):
    path = copy_dataset / "samples.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join([*rows, rows[0]]) + "\n")
    with pytest.raises(AuditError, match="denominator"):
        check_dataset(copy_dataset, progress=quiet)


def test_tampered_artifact_bytes_fail_before_contract_checks(copy_dataset):
    rows = [json.loads(line) for line in (copy_dataset / "samples.jsonl").read_text().splitlines()]
    path = copy_dataset / "samples" / (rows[0]["sample_key"] + ".png")
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(AuditError, match="artifact SHA"):
        check_dataset(copy_dataset, progress=quiet)


def test_internal_meta_hash_checked_even_when_outer_manifest_is_resigned(copy_dataset):
    rows = [json.loads(line) for line in (copy_dataset / "samples.jsonl").read_text().splitlines()]
    relative = "samples/" + rows[0]["sample_key"] + ".meta.json"
    path = copy_dataset / relative
    meta = json.loads(path.read_text())
    meta["sha256"][".sdf"] = "0" * 64
    write_json(path, meta)
    manifest = json.loads((copy_dataset / "sample-sha256.json").read_text())
    manifest[relative] = sha256(path.read_bytes()).hexdigest()
    write_json(copy_dataset / "sample-sha256.json", manifest)
    with pytest.raises(AuditError, match="Internal metadata SHA"):
        check_dataset(copy_dataset, progress=quiet)


def test_unmanifested_file_is_not_ignored(copy_dataset):
    (copy_dataset / "samples" / "foreign.png").write_bytes(b"unlisted")
    with pytest.raises(AuditError, match="unmanifested"):
        check_dataset(copy_dataset, progress=quiet)


def test_changed_duplicate_ledger_is_recomputed(copy_dataset):
    path = copy_dataset / "image-duplicates.json"
    ledger = json.loads(path.read_text())
    ledger["duplicate_png_groups"]["fake"] = []
    write_json(path, ledger)
    with pytest.raises(AuditError, match="duplicate ledger"):
        check_dataset(copy_dataset, progress=quiet)


def test_rejected_rows_must_be_preserved_in_journal(copy_dataset):
    path = copy_dataset / "rejected.jsonl"
    row = json.loads(path.read_text())
    row["reason"] = "silently rewritten"
    write_json(path, row)
    with pytest.raises(AuditError, match="Rejected journal"):
        check_dataset(copy_dataset, progress=quiet)


def test_sample_symlink_is_rejected_even_with_same_filename(copy_dataset):
    row = json.loads((copy_dataset / "samples.jsonl").read_text().splitlines()[0])
    png = copy_dataset / "samples" / (row["sample_key"] + ".png")
    target = copy_dataset / "saved-original.png"
    png.rename(target)
    png.symlink_to(target)
    with pytest.raises(AuditError, match="symlinked artifact"):
        check_dataset(copy_dataset, progress=quiet)


def test_manifest_cannot_authorize_arbitrary_external_reads():
    for value in ("/etc/passwd", "Data/sdf/chemvector_joint_v1/fresh_eval_12k/sealed_test/identities.json",
                  "Data/sdf/chemvector_curriculum_v0_2/../../../secret.py"):
        with pytest.raises(AuditError, match="outside allowed"):
            source_path(value)


def test_duplicate_json_keys_and_nonfinite_numbers_fail():
    with pytest.raises(AuditError, match="Duplicate JSON"):
        decode('{"a": 1, "a": 2}')
    with pytest.raises(AuditError, match="Nonfinite"):
        decode('{"a": NaN}')


@pytest.mark.parametrize("smiles", ["F[C@H](Cl)Br", "C/C=C/C", "CC=CC", "[13CH3]CO", "c1ccccc1"])
def test_sdf_storage_retains_identity_and_graph_ids(smiles):
    entry = merged_records([dict(record_id="case", name="case", stage="atoms_bonds", concepts=[], smiles=smiles)])[0]
    files, meta = render(entry, Profile(), typography="stereo-aware-v1")
    graph = json.loads(files[".graph.json"])
    assert validate_sdf(files[".sdf"], graph, meta["canonical_smiles"]) == (len(graph["atoms"]), len(graph["bonds"]))
    graph["atoms"][0]["formal_charge"] = 99
    with pytest.raises(AuditError, match="Graph chemistry"):
        validate_sdf(files[".sdf"], graph, meta["canonical_smiles"])


def test_sdf_opposite_specified_stereo_is_rejected():
    entries = merged_records([dict(record_id=str(i), name="case", stage="atoms_bonds", concepts=[], smiles=smiles)
                              for i, smiles in enumerate(("F[C@H](Cl)Br", "F[C@@H](Cl)Br"))])
    files, meta = render(entries[0], Profile(), typography="stereo-aware-v1")
    other, _ = render(entries[1], Profile(), typography="stereo-aware-v1")
    with pytest.raises(AuditError, match="strict isomeric identity"):
        validate_sdf(other[".sdf"], json.loads(files[".graph.json"]), meta["canonical_smiles"])
