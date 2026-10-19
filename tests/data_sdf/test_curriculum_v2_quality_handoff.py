"""Fast synthetic contract tests, plus a tiny real producer/auditor integration.

The colored-PNG fixture is deliberately synthetic, not a claim that it would
pass the full chemical artifact auditor; it exercises independent handoff
binding, pixel-conflict recomputation, whole-identity exclusion and accounting.
"""

from collections import defaultdict
from dataclasses import replace
import io
import json

from PIL import Image
import pytest

from tools.prepare_curriculum_v02_quality_handoff import (
    AUDITOR, AUDITOR_DEPENDENCIES, MANIFESTS, REPO, HandoffError,
    conflict_groups, decode, digest, main, prepare_handoff,
)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def png(color, compression=6):
    stream = io.BytesIO()
    Image.new("RGB", (768, 768), color).save(stream, format="PNG", compress_level=compression)
    return stream.getvalue()


@pytest.fixture
def fixture(tmp_path, request):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "samples").mkdir()
    smiles = ["C" * size for size in range(2, 8)]
    ids = ["curr_" + digest(value.encode())[:16] for value in smiles]
    profiles = [{"profile_id": f"style_{i}"} for i in range(6)]
    scope = {"training_authorized": False, "external_overlap_status": "unchecked",
             "external_identity_queries": 0, "external_corpora_read": False,
             "model_forwards": 0, "optimizer_steps": 0}
    molecules = [{"molecule_id": identity, "canonical_smiles": value, "split": "quarantine", **scope}
                 for identity, value in zip(ids, smiles)]
    plan = {"schema_version": "chemvector-curriculum-v0.2-plan", "profiles": profiles,
            "selected_molecules": len(molecules), "planned_images": len(molecules) * 6, **scope}
    samples, rejected, file_hashes = [], [], {}
    by_bytes, by_rgb = defaultdict(list), defaultdict(list)
    for index, (identity, value) in enumerate(zip(ids, smiles)):
        for profile_index in reversed(range(6)):  # Deliberately not lexical order.
            profile = f"style_{profile_index}"
            key = f"{identity}__{profile}"
            if index == 5 or (getattr(request, "param", None) == "conflict_rejection"
                              and index == 0 and profile_index == 5):
                rejected.append({"molecule_id": identity, "canonical_smiles": value,
                                 "profile_id": profile, "reason": "synthetic_known_render_rejection"})
                continue
            color, compression = (20 + index * 25, 30 + profile_index * 20, 60), 6
            if (index, profile_index) in {(0, 0), (1, 0)}:
                color = (255, 0, 0)  # Two identities, identical PNG bytes.
            if (index, profile_index) in {(1, 1), (2, 0)}:
                color, compression = (0, 0, 255), (1 if index == 1 else 9)
            if index == 3:
                color = (0, 255, 0)  # Six same-identity exact duplicate styles.
            if index == 4 and profile_index in {0, 1}:
                color, compression = (255, 255, 0), (1 if profile_index == 0 else 9)
            payload = png(color, compression)
            path = root / "samples" / f"{key}.png"
            path.write_bytes(payload)
            file_hashes[f"samples/{key}.png"] = digest(payload)
            samples.append({"sample_key": key, "molecule_id": identity, "profile_id": profile})
            description = {"sample_key": key, "molecule_id": identity, "canonical_smiles": value}
            by_bytes[digest(payload)].append(description)
            with Image.open(io.BytesIO(payload)) as image:
                by_rgb[digest(image.tobytes())].append(description)
    byte_conflicts = conflict_groups(by_bytes)
    rgb_conflicts = conflict_groups(by_rgb)
    counts = {"unique_molecules": len(molecules), "planned_images": 36, "accepted_images": len(samples),
              "rejected_images": len(rejected), "distinct_png_files": len(by_bytes),
              "same_png_extra_rows": len(samples) - len(by_bytes),
              "cross_identity_same_png_groups": len(byte_conflicts),
              "cross_identity_same_png_rows": sum(map(len, byte_conflicts.values()))}
    for name in MANIFESTS:
        write_json(root / name, {})
    write_json(root / "plan.json", plan)
    summary = {"schema_version": "chemvector-curriculum-v0.2-result", "generation_complete": True,
               "status": "quarantine_preview_only", "gpu_used": False, **scope, **counts,
               "plan_sha256": digest((root / "plan.json").read_bytes())}
    write_json(root / "summary.json", summary)
    write_json(root / "molecules.json", molecules)
    write_json(root / "profiles.json", profiles)
    write_json(root / "sample-sha256.json", file_hashes)
    for name, rows in (("samples.jsonl", samples), ("rejected.jsonl", rejected)):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"schema_version": "chemvector-curriculum-v0.2-independent-audit",
              "status": "passed_artifact_integrity_only", "dataset": str(root),
              "training_authorized": False, "gpu_used": False, "external_identity_queries": 0,
              "model_forwards": 0, "formal_training_admission": "not_granted",
              "auditor_source_sha256": digest(AUDITOR.read_bytes()),
              "auditor_dependency_sha256": {name: digest((REPO / name).read_bytes())
                                            for name in AUDITOR_DEPENDENCIES},
              "manifest_sha256": {name: digest((root / name).read_bytes()) for name in MANIFESTS},
              "decoded_rgb_cross_identity_conflicts": rgb_conflicts,
              "byte_identical_cross_identity_conflicts": byte_conflicts,
              "blocking_cross_identity_pixel_conflicts": True,
              "decoded_rgb_distinct_images": len(by_rgb), "checked": counts}
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, report)
    return root, audit_path, ids


def test_whole_identity_exclusion_lexical_dedup_and_conserved_denominators(fixture, tmp_path):
    root, audit_path, ids = fixture
    original = {str(path.relative_to(root)): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()}
    result = prepare_handoff(root, audit_path, tmp_path / "handoff.json")
    assert result["status"] == "proposed_internal_quality_handoff_only"
    stats = result["statistics"]
    assert stats["planned_source_samples"] == 36
    assert stats["accepted_source_samples"] == 30
    assert stats["generation_rejected_source_samples"] == 6
    assert stats["direct_cross_identity_conflict_samples"] == 4
    assert stats["quarantined_identities"] == 3
    assert stats["quarantined_accepted_samples"] == stats["quarantined_planned_samples"] == 18
    assert stats["retained_candidate_samples"] == stats["retained_distinct_png_bytes"] == 7
    assert stats["retained_distinct_decoded_rgb"] == 6
    assert stats["retained_distinct_molecules"] == 2
    assert stats["singleton_png_candidate_samples"] == 6
    assert stats["same_identity_duplicate_groups"] == 1
    assert stats["same_identity_duplicate_extra_samples"] == 5
    assert stats["byte_identical_cross_identity_conflict_groups"] == 1
    assert stats["decoded_rgb_cross_identity_conflict_groups"] == 2
    assert {row["molecule_id"] for row in result["quarantined_identities"]} == set(ids[:3])
    assert all(len(row["accepted_sample_keys"]) == len(row["all_planned_profile_ids"]) == 6
               for row in result["quarantined_identities"])
    group = result["same_identity_duplicate_groups"][0]
    assert group["retain_sample_key"] == f"{ids[3]}__style_0"
    assert group["exclude_sample_keys"] == [f"{ids[3]}__style_{i}" for i in range(1, 6)]
    assert result["training_authorized"] is False
    assert result["external_overlap_status"] == "unchecked"
    assert result["formal_training_admission"] == "not_granted"
    assert result["split_created"] is result["train_manifest_published"] is False
    assert result["source_files_modified"] == result["source_files_deleted"] == 0
    assert '"split":' not in json.dumps(result)
    after = {str(path.relative_to(root)): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()}
    assert original == after


@pytest.mark.parametrize("fixture", ["conflict_rejection"], indirect=True)
def test_whole_identity_exclusion_accounts_for_preexisting_render_reject(fixture, tmp_path):
    root, audit_path, ids = fixture
    result = prepare_handoff(root, audit_path, tmp_path / "handoff.json")
    stats = result["statistics"]
    assert stats["accepted_source_samples"] == 29
    assert stats["generation_rejected_source_samples"] == 7
    assert stats["quarantined_accepted_samples"] == 17
    assert stats["quarantined_generation_rejected_samples"] == 1
    assert stats["quarantined_planned_samples"] == 18
    first = next(row for row in result["quarantined_identities"] if row["molecule_id"] == ids[0])
    assert len(first["all_planned_profile_ids"]) == 6
    assert len(first["accepted_sample_keys"]) == 5
    assert first["generation_rejected_profile_ids"] == ["style_5"]


def test_proposal_membership_is_deterministic(fixture, tmp_path):
    root, audit_path, _ = fixture
    first = prepare_handoff(root, audit_path, tmp_path / "first.json")
    second = prepare_handoff(root, audit_path, tmp_path / "second.json")
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second


@pytest.mark.parametrize("mutation", [
    "failed_status", "other_batch", "auditor_sha", "dependency_sha", "manifest_sha",
    "omit_rgb_group", "omit_conflict_identity", "false_blocker", "wrong_counts", "unsafe_dependency",
])
def test_tampered_or_unbound_audit_fails_closed(fixture, tmp_path, mutation):
    root, audit_path, _ = fixture
    report = json.loads(audit_path.read_text())
    if mutation == "failed_status":
        report["status"] = "failed"
    elif mutation == "other_batch":
        report["dataset"] = str(root.with_name("another_batch"))
    elif mutation == "auditor_sha":
        report["auditor_source_sha256"] = "0" * 64
    elif mutation == "dependency_sha":
        report["auditor_dependency_sha256"][AUDITOR_DEPENDENCIES[0]] = "0" * 64
    elif mutation == "manifest_sha":
        report["manifest_sha256"]["samples.jsonl"] = "0" * 64
    elif mutation == "omit_rgb_group":
        report["decoded_rgb_cross_identity_conflicts"].pop(next(iter(report["decoded_rgb_cross_identity_conflicts"])))
    elif mutation == "omit_conflict_identity":
        group = next(iter(report["byte_identical_cross_identity_conflicts"].values()))
        group.pop()
    elif mutation == "false_blocker":
        report["blocking_cross_identity_pixel_conflicts"] = False
    elif mutation == "wrong_counts":
        report["checked"]["accepted_images"] += 1
    elif mutation == "unsafe_dependency":
        report["auditor_dependency_sha256"]["/etc/passwd"] = "0" * 64
    write_json(audit_path, report)
    output = tmp_path / "failure.json"
    with pytest.raises(HandoffError):
        prepare_handoff(root, audit_path, output)
    failure = json.loads(output.read_text())
    assert failure["status"] == "failed"
    assert failure["training_authorized"] is False


@pytest.mark.parametrize("artifact", ["samples.jsonl", "png", "symlink"])
def test_changed_dataset_artifact_fails(fixture, tmp_path, artifact):
    root, audit_path, _ = fixture
    if artifact == "samples.jsonl":
        path = root / artifact
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = next((root / "samples").glob("*.png"))
        if artifact == "png":
            path.write_bytes(path.read_bytes() + b"tamper")
        else:
            target = tmp_path / "target.png"
            path.rename(target)
            path.symlink_to(target)
    with pytest.raises(HandoffError):
        prepare_handoff(root, audit_path, tmp_path / "failed.json")


def test_output_is_exclusive_external_and_parent_must_exist(fixture, tmp_path):
    root, audit_path, _ = fixture
    output = tmp_path / "existing.json"
    output.write_text("original sentinel")
    with pytest.raises(FileExistsError):
        prepare_handoff(root, audit_path, output)
    assert output.read_text() == "original sentinel"
    for invalid in (root / "new.json", tmp_path / "missing" / "new.json", audit_path):
        with pytest.raises(HandoffError):
            prepare_handoff(root, audit_path, invalid)


def test_strict_json_and_cli_schema(fixture, tmp_path, capsys):
    root, audit_path, _ = fixture
    assert main(["--dataset", str(root), "--audit", str(audit_path),
                 "--output", str(tmp_path / "cli.json")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "proposed_internal_quality_handoff_only"
    assert printed["statistics"]["retained_candidate_samples"] == 7
    for text in ('{"x": 1, "x": 2}', '{"x": NaN}'):
        with pytest.raises(HandoffError):
            decode(text)


def test_tiny_real_builder_and_independent_auditor_integration(tmp_path, monkeypatch):
    from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
    from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile
    from Data.sdf.chemvector_curriculum_v0_2 import build as builder
    from tools.audit_curriculum_expansion_v02 import audit

    entries = merged_records([dict(record_id=f"tiny_{i}", stage="atoms_bonds", name="tiny",
                                   concepts=[], smiles=value) for i, value in enumerate(("CCO", "CCN"))])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(builder, "entries", lambda: entries)
    monkeypatch.setattr(builder, "profiles", lambda: [Profile(), replace(Profile(), profile_id="font_small")])
    monkeypatch.setattr(builder, "bounded_map", lambda function, jobs, **kwargs: (function(job) for job in jobs))
    root, receipt_path = tmp_path / "real_tiny", tmp_path / "real_audit.json"
    builder.build(root, workers=1)
    receipt = audit(root, receipt_path)
    assert receipt["status"] == "passed_artifact_integrity_only"
    result = prepare_handoff(root, receipt_path, tmp_path / "real_handoff.json")
    assert result["statistics"]["accepted_source_samples"] == 4
    assert result["statistics"]["quarantined_identities"] == 0
    assert result["statistics"]["retained_candidate_samples"] == 2
    assert result["statistics"]["same_identity_duplicate_extra_samples"] == 2
