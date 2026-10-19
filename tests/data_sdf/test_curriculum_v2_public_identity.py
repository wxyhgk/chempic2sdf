"""Synthetic public-only review integration; never opens evaluation corpora."""

from collections import defaultdict
import json

import pytest
from rdkit import rdBase

from chemvector.training.curriculum.admission import identities
from tools import review_curriculum_v02_public_identity as review


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    request_dir, benchmark = tmp_path / "request", tmp_path / "public"
    request_dir.mkdir()
    (benchmark / "manifests").mkdir(parents=True)
    for name in review.public_review.SUBSETS:
        (benchmark / "manifests" / (name + ".csv")).write_text("synthetic\n")
    candidates = [dict(source="curriculum", molecule_id=f"m{i}", canonical_smiles=smiles,
                       **identities(smiles), sample_keys=[f"m{i}__a", f"m{i}__b"], image_count=2)
                  for i, smiles in enumerate(("CCO", "CCN", "CC#N"))]
    review.write(request_dir / "candidate-identities.json", candidates)
    monkeypatch.setattr(review.fixed, "CANDIDATES_SHA", review.fixed.digest(request_dir / "candidate-identities.json"))
    monkeypatch.setattr(review.fixed, "COUNTS", (3, 6))
    request = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
                   candidate_molecules=3, candidate_images=6, input_sha256={}, rdkit_version=rdBase.rdkitVersion,
                   identity_implementation_sha256=review.fixed.digest(review.private_review.IDENTITIES),
                   candidate_identities_sha256=review.fixed.CANDIDATES_SHA)
    review.write(request_dir / "request.json", request)
    monkeypatch.setattr(review.fixed, "REQUEST_SHA", review.fixed.digest(request_dir / "request.json"))
    cache = tmp_path / "cache" / "index.json"
    cache.parent.mkdir()
    review.write(cache, dict(schema_version="chemvector-joint-overlap-index-0.1", benchmark_inchikeys=[candidates[0]["inchikey"]]))
    receipt = tmp_path / "private_receipt" / "receipt.json"
    receipt.parent.mkdir()
    review.write(receipt, {"synthetic_saved_receipt": True})
    monkeypatch.setattr(review.private_review, "validate_receipt", lambda *args: {
        "status": "checked", "evaluation_commitment_sha256": "synthetic", "excluded_candidates": [
            {"source": "curriculum", "molecule_id": "m1", "rules": ["stereo_isotope_independent_group_key"]}]})
    def forbidden(*args, **kwargs):
        raise AssertionError("No private invocation allowed")
    monkeypatch.setattr(review.private_review, "run", forbidden)
    def scan(root, by_ref, inventories):
        assert root == benchmark and len(by_ref) == len(inventories) == 3
        return ([{"sample_id": "synthetic"}], [], [], defaultdict(list), {
            name: review.fixed.digest(benchmark / "manifests" / (name + ".csv")) for name in review.public_review.SUBSETS})
    monkeypatch.setattr(review.public_review, "scan_public", scan)
    return request_dir, benchmark, cache, receipt, tmp_path / "output"


def test_separate_public_plan_does_not_rewrite_request_or_requery(fixed):
    original = (fixed[0] / "request.json").read_bytes()
    result = review.run(*fixed)
    assert result["complete"] and result["status"] == "screened_with_conservative_exclusions"
    assert result["candidate_molecules"] == 3 and result["candidate_images"] == 6
    assert result["remaining_molecules"] == 1 and result["remaining_images"] == 2
    assert result["conservatively_excluded_molecules"] == 2 and result["conservatively_excluded_images"] == 4
    assert result["private_identity_queries"] == 0 and result["training_authorized"] is False
    assert (fixed[0] / "request.json").read_bytes() == original
    plan = json.loads((fixed[4] / "plan.json").read_text())
    assert plan["input_sha256"][str(fixed[2])] == review.fixed.digest(fixed[2])
    assert "separate_public_sidecar" in plan["cache_binding"]
    assert json.loads((fixed[4] / "remaining-candidates.json").read_text())[0]["molecule_id"] == "m2"
    for name, digest in result["artifact_sha256"].items():
        assert review.fixed.digest(fixed[4] / name) == digest
    with pytest.raises(FileExistsError):
        review.run(*fixed)


def test_changed_cache_fails_closed_after_scan(fixed, monkeypatch):
    original = review.public_review.scan_public
    def changed(*args):
        values = original(*args)
        fixed[2].write_bytes(fixed[2].read_bytes() + b" ")
        return values
    monkeypatch.setattr(review.public_review, "scan_public", changed)
    with pytest.raises(ValueError, match="input changed"):
        review.run(*fixed)
    report = json.loads((fixed[4] / "report.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert not (fixed[4] / "remaining-candidates.json").exists()


def test_changed_request_cannot_scan(fixed):
    path = fixed[0] / "request.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="request mismatch"):
        review.run(*fixed)
    assert not fixed[4].exists()


def test_cpu_only(fixed, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        review.run(*fixed)


def test_union_whole_identity_all_images_and_reasons():
    candidates = {f"curriculum:{i}": {"source": "curriculum", "molecule_id": i, "connectivity_prefix": i,
                                      "image_count": 6, "sample_keys": [i + str(j) for j in range(6)]}
                  for i in "abcd"}
    private = {"excluded_candidates": [{"source": "curriculum", "molecule_id": "a"}]}
    public = {"curriculum:b": [{"reason": "cannot_rule_out_unresolved_identity"}],
              "curriculum:a": [{"reason": "public_key_or_connectivity_hit"}]}
    remaining, excluded, _ = review.combine(candidates, public, {"c"}, private)
    assert remaining == [candidates["curriculum:d"]]
    assert len(excluded) == 3 and sum(row["image_count"] for row in excluded) == 18
    assert len(excluded[0]["exclusion_reasons"]) == 2
    assert len(candidates) == 4 and len(public["curriculum:a"]) == 1
    with pytest.raises(ValueError, match="Foreign"):
        review.combine(candidates, {"curriculum:foreign": []}, set(), private)
