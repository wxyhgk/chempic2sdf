"""Synthetic public inputs; private calls forbidden."""

from collections import defaultdict
import json

import pytest

from chemvector.training.curriculum.admission import identities
from tools import review_structure_expansion_public as review


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OUTPUT", "CACHE", "BENCHMARK"):
        monkeypatch.setattr(review, name, tmp_path / name.lower())
    (review.BENCHMARK / "manifests").mkdir(parents=True)
    for name in review.public_review.SUBSETS:
        (review.BENCHMARK / "manifests" / (name + ".csv")).write_text("synthetic\n")
    candidates = [dict(source="curriculum", molecule_id=f"m{i}", canonical_smiles=smi, **identities(smi),
                       sample_keys=[f"m{i}__oa00", f"m{i}__oa02"], image_count=2)
                  for i, smi in enumerate(("CCO", "CCN", "CC#N"))]
    review.fixed.write_new(review.CACHE, dict(schema_version="chemvector-joint-overlap-index-0.1",
                                           benchmark_inchikeys=[candidates[0]["inchikey"]]))
    receipt = dict(status="checked", evaluation_commitment_sha256="synthetic", excluded_candidates=[
        dict(source="curriculum", molecule_id="m1", rules=["full_inchikey"])])
    dependency = tmp_path / "frozen-request.json"
    dependency.write_text("synthetic fixed bytes\n")
    monkeypatch.setattr(review, "bound_inputs", lambda: (candidates, receipt, {str(dependency):review.fixed.digest(dependency)}))
    def forbidden(*args, **kwargs): raise AssertionError("No private call")
    monkeypatch.setattr(review.private_review, "run", forbidden)
    def scan(root, by_ref, inventories):
        assert len(by_ref) == len(inventories) == 3
        return ([{"sample_id":"synthetic"}], [], [], defaultdict(list),
                {name:review.fixed.digest(root / "manifests" / (name + ".csv")) for name in review.public_review.SUBSETS})
    monkeypatch.setattr(review.public_review, "scan_public", scan)
    return dependency


def test_sidecar_preserves_fixed_bytes_and_unions_whole_candidates(fixed):
    before = fixed.read_bytes()
    report = review.run()
    assert report["complete"] and report["remaining_molecules"] == 1 and report["remaining_images"] == 2
    assert report["conservatively_excluded_molecules"] == 2 and report["conservatively_excluded_images"] == 4
    assert report["private_identity_queries"] == 0 and report["training_authorized"] is False
    assert fixed.read_bytes() == before
    for name, digest in report["artifact_sha256"].items():
        assert review.fixed.digest(review.OUTPUT / name) == digest
    with pytest.raises(ValueError, match="Fresh"):
        review.run()


def test_changed_cache_cannot_publish_remaining_candidates(fixed, monkeypatch):
    original = review.public_review.scan_public
    def scan(*args):
        result = original(*args)
        review.CACHE.write_bytes(review.CACHE.read_bytes() + b" ")
        return result
    monkeypatch.setattr(review.public_review, "scan_public", scan)
    with pytest.raises(ValueError, match="input changed"): review.run()
    report = json.loads((review.OUTPUT / "report.json").read_bytes())
    assert not report["complete"] and report["status"] == "failed"
    assert not (review.OUTPUT / "remaining-candidates.json").exists()


def test_cpu_only(fixed, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"): review.run()


def test_unresolved_risk_excludes_every_style(fixed, monkeypatch):
    original = review.public_review.scan_public
    def scan(*args):
        index, _, _, excluded, manifests = original(*args)
        excluded["curriculum:m2"] = [{"reason":"cannot_rule_out_unresolved_identity"}]
        unresolved = [dict(native_probe={"key":None}, unsupported_inventory=True)]
        evidence = [dict(candidate_ref="curriculum:m2", deficiency_witness=None, excluded_conservatively=True)]
        return index, unresolved, evidence, excluded, manifests
    monkeypatch.setattr(review.public_review, "scan_public", scan)
    report = review.run()
    assert report["remaining_molecules"] == report["remaining_images"] == report["remaining_ambiguous_pairs"] == 0
    assert report["conservatively_excluded_images"] == 6
