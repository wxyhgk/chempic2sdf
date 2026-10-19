"""Synthetic incremental publication; never real review authority or private data."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from Data.sdf.chemvector_curriculum_v0_1.build import build
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import profiles
from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.expansion_publish import publish
from chemvector.training.curriculum.expansion_release_plan import build_plan, choose_additions
from chemvector.training.curriculum.preflight import run as preflight
from chemvector.training.curriculum.release import CHECKS, verify_release
from chemvector.training.curriculum.release_evidence import digest, encoded
from tests.chemvector.test_curriculum_release import released


def pixel_row(key, mid, pixel, split="train", smiles="CCO"):
    return dict(sample_key=key, molecule_id=mid, pixel_sha256=pixel, split=split, canonical_smiles=smiles)


def test_new_pixels_deduplicate_against_old_without_changing_old_rows():
    old = [pixel_row("old", "a", "p"), pixel_row("old-duplicate", "a", "p")]
    before = deepcopy(old)
    new = [pixel_row("new-repeat", "a", "p"), pixel_row("new", "b", "q", smiles="CCN")]
    kept, report = choose_additions(old, new)
    assert kept == new[1:] and old == before
    assert report["new_duplicate_keys"] == ["new-repeat"]
    assert report["old_train_images"] == 2 and report["old_train_unique_pixels"] == 1


def test_holdout_pixel_match_excludes_all_new_variants_not_just_the_matching_image():
    old = [pixel_row("val", "val", "p", split="validation")]
    new = [pixel_row("a", "same-mol", "p"), pixel_row("b", "same-mol", "q"), pixel_row("c", "other", "r")]
    kept, report = choose_additions(old, new)
    assert kept == new[2:] and report["heldout_pixel_excluded_keys"] == ["a", "b"]


def test_conflicting_chemistry_duplicate_new_pixels_and_existing_pixel_leakage_fail_closed():
    old = [pixel_row("old", "a", "p")]
    with pytest.raises(ValueError, match="conflicting chemistry"):
        choose_additions(old, [pixel_row("new", "b", "p", smiles="CCN")])
    with pytest.raises(ValueError, match="unique"):
        choose_additions([], [pixel_row("a", "a", "p"), pixel_row("b", "a", "p")])
    with pytest.raises(ValueError, match="Existing train/validation"):
        choose_additions(old + [pixel_row("val", "b", "p", split="validation")], [])


@pytest.fixture
def incremental(released, tmp_path, monkeypatch):
    old, _, mp, ap = released
    monkeypatch.setattr("chemvector.training.curriculum.expansion_release_plan.MINIMUM_IMAGES", 1)
    # Pure evidence verifier has real integration coverage separately. These
    # intentionally fake documents exercise source/target/copy gates only.
    monkeypatch.setattr("chemvector.training.curriculum.expansion_release_plan.verify_reviews", lambda *args: None)
    root = tmp_path / "new-quarantine"
    entry = merged_records([dict(record_id="new", name="new", stage="full_molecule", concepts=[], smiles="CCBr")])[0]
    entry["source_provenance"] = dict(source_molecule=dict(source="chembl", split="train", molecule_id="chembl:fixture"))
    build(root, entries=[entry], selected_profiles=profiles()[:2], workers=1, gallery=False, typography="stereo-aware-v1")
    checked = tmp_path / "targets"
    preflight(root, checked)
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    unique = [dict(sample_key=r["sample_key"], molecule_id=entry["molecule_id"], canonical_smiles="CCBr",
        profile_id=r["profile_id"], target_compatible=True,
        pixel_sha256=sha256(load_sample(root, r["sample_key"], purpose="diagnostic")["image"].tobytes()).hexdigest()) for r in rows]
    candidate = dict(source="full_molecule", molecule_id=entry["molecule_id"], canonical_smiles="CCBr", **identities("CCBr"),
        image_count=len(rows), sample_keys=[r["sample_key"] for r in rows], source_molecule_id="chembl:fixture")
    paths = {"previous_manifest": mp, "previous_admission": ap, "targets": checked / "report.json", "target_samples": checked / "samples.json"}
    docs = {"request": {"input_sha256": {str(root / n): digest(root / n) for n in
            ("summary.json", "molecules.json", "samples.jsonl", "sample-sha256.json")}},
        "request_candidates": [candidate], "private_receipt": {}, "public_report": {}, "remaining": [candidate],
        "inventory_candidates": unique,
        "inventory": dict(training_authorized=False, compatible_unique_candidate_images=len(unique),
            input_sha256={str(paths[n]): digest(paths[n]) for n in ("targets", "target_samples")}),
        "source_rights": dict(schema_version="chemvector-source-use-review-0.1", use="local_research_only", redistribution_authorized=False,
            remaining_candidates_sha256=sha256(encoded([candidate])).hexdigest(), inherited_release_admission_sha256=digest(ap),
            sources={n: dict(basis="synthetic only", attribution="synthetic only") for n in ("hand_authored", "chembl", "coconut", "dejavu")})}
    for name, value in docs.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(encoded(value))
    spec = dict(schema_version="chemvector-expanded-release-spec-0.1", data_root=str(root), new_source="full_molecule_expansion",
        minimum_unique_train_images=1, seed=13, inputs={n: dict(path=str(p), sha256=digest(p)) for n, p in paths.items()},
        training_protocol=dict(schema_version="chemvector-fixed-continuation-protocol-0.2", update_recipe="freeze-role-v1", max_steps=1200,
            batch_size=1, validate_every=400, patience=2, wall_seconds=1800, max_gpus=1, selection_domains=["full_molecule"], automatic_resume=False))
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(encoded(spec))
    return spec, spec_path, old, root


def test_incremental_plan_preserves_all_existing_validation_and_train_rows(incremental):
    spec, _, old, _ = incremental
    plan = build_plan(spec)
    assert plan["manifest_template"]["samples"][:len(old["samples"])] == old["samples"]
    assert plan["added_train_images"] == 2 and plan["unique_train_images"] == 4
    assert plan["counts"] == {"train:curriculum": 1, "train:full_molecule": 3, "validation:full_molecule": 1}
    assert not plan["validation_changed"] and not plan["training_authorized"]
    assert not plan["new_private_query_performed"]


def test_incremental_publication_only_copies_new_bundles_without_changing_originals(incremental, tmp_path):
    spec, sp, old, root = incremental
    plan = build_plan(spec)
    pp = tmp_path / "plan.json"
    pp.write_bytes(encoded(plan))
    rp = tmp_path / "review.json"
    rp.write_bytes(encoded(dict(schema_version="chemvector-expanded-promotion-review-0.1", decision="approve_local_research_data_release",
        approved_plan_sha256=digest(pp), reviewer="synthetic only", rationale=["temporary test fixtures only"],
        release_checks={k: "checked" for k in CHECKS}, source_use_review_sha256=spec["inputs"]["source_rights"]["sha256"])))
    before = {p: p.read_bytes() for p in (root / "samples").iterdir()}
    output = tmp_path / "release"
    report = publish(sp, pp, rp, output)
    manifest, _, _ = verify_release(output / "manifest.json", output / "admission.json")
    assert report["added_train_images"] == 2 and not report["training_performed"]
    assert manifest["samples"][:len(old["samples"])] == old["samples"]
    assert all(p.read_bytes() == data for p, data in before.items())
    for row in plan["addition_rows"]:
        old_png = root / "samples" / (row["sample_key"] + ".png")
        new_png = output / "sources/full_molecule_expansion/samples" / old_png.name
        assert new_png.read_bytes() == before[old_png] and new_png.stat().st_ino != old_png.stat().st_ino


def test_bad_review_or_changed_source_never_creates_admission(incremental, tmp_path):
    spec, sp, _, root = incremental
    plan = build_plan(spec)
    pp, rp = tmp_path / "plan.json", tmp_path / "review.json"
    pp.write_bytes(encoded(plan)); rp.write_bytes(encoded({"decision": "pending"}))
    with pytest.raises(ValueError, match="exact-plan"):
        publish(sp, pp, rp, tmp_path / "denied")
    assert not (tmp_path / "denied").exists()
    row = plan["addition_rows"][0]
    (root / "samples" / (row["sample_key"] + ".png")).write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_plan(spec)
