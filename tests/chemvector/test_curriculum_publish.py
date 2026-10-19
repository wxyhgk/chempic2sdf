"""Only temporary synthetic publications; no real data approvals in tests."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from chemvector.training.curriculum.publish import SUFFIXES, publish, transition
from chemvector.training.curriculum.release_evidence import encoded, inputs, unify_splits
from chemvector.training.curriculum.release import CHECKS, load_record, verify_release
from tests.chemvector.test_curriculum_fresh_dev import original, mixed
from tests.chemvector.test_curriculum_release import released


@pytest.fixture
def planned(mixed, tmp_path, monkeypatch):
    manifest, _, _, _ = mixed
    manifest = deepcopy(manifest)
    artifacts = {}
    source = Path(manifest["sources"].pop("synthetic"))
    for row in manifest["samples"]:
        if row["source"] != "synthetic":
            continue
        row["source"] = row["domain"]
        manifest["sources"][row["source"]] = str(source)
        key = row["sample_key"]
        payloads = {s: (source / "samples" / (key + s)).read_bytes() for s in SUFFIXES}
        for suffix in (".graph.json", ".layers.json", ".meta.json"):
            value = json.loads(payloads[suffix])
            value.update(split="quarantine", training_authorized=False, external_overlap_status="unchecked")
            if suffix == ".graph.json":
                value.pop("split")
                value.pop("external_overlap_status")
            if suffix == ".meta.json":
                value["sha256"] = {s: sha256(v).hexdigest() for s, v in payloads.items() if s != suffix}
            payloads[suffix] = encoded(value)
        for suffix, value in payloads.items():
            (source / "samples" / (key + suffix)).write_bytes(value)
        row["meta_sha256"] = sha256(payloads[".meta.json"]).hexdigest()
        artifacts[row["source"] + ":" + key] = {"samples/" + key + s: sha256(v).hexdigest() for s, v in payloads.items()}
    plan = {"schema_version": "chemvector-promotion-plan-0.1", "training_authorized": False,
        "manifest_template": manifest, "source_artifacts": artifacts,
        "counts": {"train:curriculum": 1, "train:full_molecule": 1, "validation:full_molecule": 4}, "limitations": []}
    rights = tmp_path / "rights.json"
    rights.write_text('{"synthetic_test_only": true}')
    spec = {"roots": {d: str(source) for d in ("curriculum", "full_molecule")},
            "inputs": {"source_rights": {"path": str(rights), "sha256": sha256(rights.read_bytes()).hexdigest()}}}
    review = {"schema_version": "chemvector-promotion-review-0.1", "decision": "approve_local_research",
        "approved_plan_sha256": sha256(encoded(plan)).hexdigest(), "reviewer": "synthetic-test-only",
        "rationale": "temporary generated test fixtures only", "release_checks": {k: "checked" for k in CHECKS},
        "source_use_review_sha256": spec["inputs"]["source_rights"]["sha256"]}
    paths = [tmp_path / name for name in ("spec.json", "plan.json", "review.json")]
    for path, value in zip(paths, (spec, plan, review)):
        path.write_bytes(encoded(value))
    # Evidence-consumer tests are separate; this fixture focuses on the actual
    # disk transition and launcher preflight, with synthetic review input only.
    monkeypatch.setattr("chemvector.training.curriculum.publish.build_plan", lambda s: deepcopy(plan))
    return paths, plan, source


def test_independent_publication_preserves_originals_and_passes_launcher(planned, tmp_path):
    paths, plan, source = planned
    before = {p: p.read_bytes() for p in (source / "samples").iterdir()}
    output = tmp_path / "published"
    report = publish(*paths, output)
    assert report["copied_bundles"] == 2 and report["byte_identical_non_lifecycle_artifacts"] == 16
    assert all(p.read_bytes() == data for p, data in before.items())
    manifest, approval, _ = verify_release(output / "manifest.json", output / "admission.json")
    assert approval["training_authorized"] and approval["checked_images"] == 6
    for row in manifest["samples"]:
        sample = load_record(manifest, row)
        assert sample["purpose"] == ("training" if row["split"] == "train" else "diagnostic")
        if row["split"] == "train":
            dest = Path(manifest["sources"][row["source"]]) / "samples"
            for suffix in SUFFIXES - {".meta.json", ".graph.json", ".layers.json"}:
                original = source / "samples" / (row["sample_key"] + suffix)
                assert (dest / original.name).read_bytes() == before[original]
                assert (dest / original.name).stat().st_ino != original.stat().st_ino
    with pytest.raises(FileExistsError):
        publish(*paths, output)


@pytest.mark.parametrize("mutation", ["review_hash", "review_decision", "review_check", "plan", "source", "preflight", "final_verify"])
def test_publication_failure_never_emits_admission(planned, tmp_path, monkeypatch, mutation):
    paths, plan, source = planned
    spec_path, plan_path, review_path = paths
    if mutation.startswith("review"):
        review = json.loads(review_path.read_bytes())
        if mutation == "review_hash":
            review["approved_plan_sha256"] = "0" * 64
        elif mutation == "review_decision":
            review["decision"] = "pending"
        else:
            review["release_checks"].pop("public_benchmark_identity")
        review_path.write_bytes(encoded(review))
    elif mutation == "plan":
        value = json.loads(plan_path.read_bytes())
        value["manifest_template"]["seed"] += 1
        plan_path.write_bytes(encoded(value))
    elif mutation == "source":
        row = plan["manifest_template"]["samples"][0]
        (source / "samples" / (row["sample_key"] + ".png")).write_bytes(b"bad")
    else:
        def fail(*args):
            raise ValueError("synthetic failed preflight")
        target = ("chemvector.training.curriculum.publish.verify_release" if mutation == "final_verify"
                  else "chemvector.training.curriculum.launcher.preflight_records")
        monkeypatch.setattr(target, fail)
    output = tmp_path / "failed-publication"
    with pytest.raises(ValueError):
        publish(*paths, output)
    assert not (output / "admission.json").exists()


def test_transition_cannot_change_chemistry_or_accept_already_admitted_payloads(planned):
    _, plan, source = planned
    row = plan["manifest_template"]["samples"][0]
    payloads = {s: (source / "samples" / (row["sample_key"] + s)).read_bytes() for s in SUFFIXES}
    changed = transition(payloads, row, plan_sha256="a" * 64, original_root=str(source))
    flags = {"split", "training_authorized", "external_overlap_status"}
    for suffix in (".graph.json", ".layers.json"):
        before, after = json.loads(payloads[suffix]), json.loads(changed[suffix])
        assert {k: v for k, v in before.items() if k not in flags} == {k: v for k, v in after.items() if k not in flags}
    next_row = dict(row, meta_sha256=sha256(changed[".meta.json"]).hexdigest())
    with pytest.raises(ValueError, match="quarantined"):
        transition(changed, next_row, plan_sha256="a" * 64, original_root=str(source))


def test_evidence_requires_complete_pinned_inputs(tmp_path):
    with pytest.raises(ValueError, match="Incomplete"):
        inputs({"inputs": {}})
    names = {"request", "request_candidates", "public_report", "remaining", "private_receipt",
             "validation_candidates", "curriculum_split", "source_rights", "curriculum_targets", "full_molecule_targets"}
    path = tmp_path / "evidence.json"
    path.write_text("{}")
    spec = {"inputs": {name: {"path": str(path), "sha256": sha256(b"{}").hexdigest()} for name in names}}
    assert set(inputs(spec)[0]) == names
    path.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        inputs(spec)


def test_charged_connectivity_siblings_keep_all_variants_in_holdout():
    rows = [dict(source="curriculum", sample_key=smi + str(i), molecule_id=smi,
                 canonical_smiles=smi, split=split)
            for smi, split in (("CN", "train"), ("C[NH3+]", "validation"), ("CS", "validation"), ("C[S-]", "train"))
            for i in range(2)]
    changes = unify_splits(rows)
    assert len(changes) == 4 and {r["split"] for r in rows} == {"validation"}
    assert all(c["from"] == "train" and c["to"] == "validation" for c in changes)


def test_global_split_union_is_transitive_and_cannot_relabel_old_train():
    rows = [dict(source="curriculum", sample_key=str(i), molecule_id=str(i), canonical_smiles=smi, split=split)
            for i, (smi, split) in enumerate((("Cc1ccccc1", "validation"), ("Nc1ccccc1", "train"), ("Oc1ccccc1", "train")))]
    assert len(unify_splits(rows)) == 2
    rows[-1].update(source="full_molecule", split="train")
    with pytest.raises(ValueError, match="Old-train replay"):
        unify_splits(rows)
