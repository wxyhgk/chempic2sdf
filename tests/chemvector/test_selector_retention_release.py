"""Synthetic release and authorization-boundary tests; no real publication."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from chemvector.training.curriculum import selector_retention_release as stage
from tests.chemvector.test_curriculum_release import released


@pytest.fixture
def local(released, tmp_path):
    old, admission, _, _ = released
    manifest = dict(schema_version=stage.SCHEMA, use=stage.USE, sources=old["sources"],
        training_run_authorized=False, training_samples=old["samples"][:2],
        validation_only_samples=[{**old["samples"][2], "purpose": "diagnostic"}])
    root = tmp_path / "local"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    admission = {**admission, "use": stage.USE, "training_run_authorized": False,
        "approved_manifest_sha256": sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "checked_training_images": 2, "checked_validation_references": 1}
    (root / "admission.json").write_text(json.dumps(admission))
    return root, manifest, admission


def test_data_release_loads_only_its_declared_purpose(local):
    root, expected, _ = local
    manifest, admission = stage.verify_release(root)
    assert manifest == expected and admission["training_run_authorized"] is False
    for row in manifest["training_samples"]:
        assert stage.load_row(manifest, row, purpose="training")["purpose"] == "training"
    row = manifest["validation_only_samples"][0]
    assert stage.load_row(manifest, row, purpose="diagnostic")["purpose"] == "diagnostic"
    with pytest.raises(ValueError, match="validation cannot"):
        stage.load_row(manifest, row, purpose="training")
    with pytest.raises(ValueError, match="outside exact"):
        stage.load_row(manifest, manifest["training_samples"][0], purpose="diagnostic")


@pytest.mark.parametrize("fault", ["manifest_hash", "run_authorized", "counts", "not_admitted"])
def test_admission_tampering_rejected(local, fault):
    root, _, admission = local
    if fault == "manifest_hash":
        admission["approved_manifest_sha256"] = "0" * 64
    elif fault == "run_authorized":
        admission["training_run_authorized"] = True
    elif fault == "counts":
        admission["checked_training_images"] += 1
    else:
        admission["training_authorized"] = False
    (root / "admission.json").write_text(json.dumps(admission))
    with pytest.raises(ValueError):
        stage.verify_release(root)


@pytest.mark.parametrize("fault", ["duplicate", "overlap", "group", "validation_purpose", "schema"])
def test_manifest_scope_and_identity_guard(local, fault):
    _, manifest, _ = local
    manifest = deepcopy(manifest)
    if fault == "duplicate":
        manifest["training_samples"].append(manifest["training_samples"][0])
    elif fault == "overlap":
        original = manifest["training_samples"][0]
        for field in ("canonical_smiles", "group_key"):
            manifest["validation_only_samples"][0][field] = original[field]
    elif fault == "group":
        manifest["training_samples"][0]["group_key"] = "fake"
    elif fault == "validation_purpose":
        manifest["validation_only_samples"][0]["purpose"] = "training"
    else:
        manifest["schema_version"] = "chemvector-continuation-release-0.1"
    with pytest.raises(ValueError):
        stage.validate_manifest(manifest)


def test_real_review_is_data_only_exact_scope():
    # Review document is a source input, not a model/sample or publication.
    review = json.loads(stage.REVIEW.read_text())
    stage.validate_review(review)
    for field, value in (("training_run_authorized", True), ("copied_new_training_images", 17),
                         ("approved_trial_proposal_sha256", "0" * 64), ("full_curriculum_publication_authorized", True)):
        changed = {**review, field: value}
        with pytest.raises(ValueError, match="data-only review"):
            stage.validate_review(changed)


def test_publication_guards_before_any_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        stage.publish(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="Fresh independent"):
        stage.publish(tmp_path / "absent")
    (tmp_path / "samples").mkdir()
    with pytest.raises(ValueError, match="Fresh independent"):
        stage.publish(tmp_path)
