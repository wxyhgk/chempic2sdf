"""Synthetic releases only: never promote or edit the real quarantine corpus."""

from copy import deepcopy
from hashlib import sha256
from itertools import islice
import json

import pytest

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.release import CHECKS, SCHEMA, load_record, training_order, verify_release


@pytest.fixture
def released(tmp_path):
    root = tmp_path / "synthetic-data"
    (root / "samples").mkdir(parents=True)
    rows = []
    for smiles, split, domain in (("CCO", "train", "curriculum"), ("CCN", "train", "full_molecule"),
                                   ("CCl", "validation", "full_molecule")):
        entry = merged_records([dict(record_id="test", name="test", stage="atoms_bonds", concepts=[], smiles=smiles)])[0]
        files, meta = render(entry, Profile(), typography="ink-aware-v1")
        # Authorize ONLY these temporary synthetic test fixtures. Do not expose
        # a promotion helper in production code or touch an existing dataset.
        for suffix in (".graph.json", ".layers.json", ".meta.json"):
            record = json.loads(files[suffix])
            record.update(split=split, training_authorized=split == "train", external_overlap_status="checked")
            files[suffix] = json.dumps(record).encode()
        meta = json.loads(files[".meta.json"])
        for suffix in meta["sha256"]:
            meta["sha256"][suffix] = sha256(files[suffix]).hexdigest()
        files[".meta.json"] = json.dumps(meta).encode()
        for suffix, content in files.items():
            (root / "samples" / (meta["sample_key"] + suffix)).write_bytes(content)
        rows.append({"source": "synthetic", "sample_key": meta["sample_key"], "molecule_id": entry["molecule_id"],
            "canonical_smiles": smiles, "group_key": identities(smiles)["group_key"], "split": split, "domain": domain,
            "meta_sha256": sha256(files[".meta.json"]).hexdigest()})
    manifest = {"schema_version": SCHEMA, "target_format": "curriculum-same-svg", "sources": {"synthetic": str(root)},
                "samples": rows, "seed": 12, "train_cycle": ["full_molecule", "full_molecule", "curriculum"]}
    admission = {"schema_version": "chemvector-curriculum-admission-0.1", "training_authorized": True,
                 "external_overlap_status": "checked", "coverage_gaps": [], "checked_images": 3,
                 "release_checks": {k: "checked" for k in CHECKS}}
    manifest_path, admission_path = tmp_path / "manifest.json", tmp_path / "admission.json"
    bind(manifest, admission, manifest_path, admission_path)
    return manifest, admission, manifest_path, admission_path


def bind(manifest, admission, manifest_path, admission_path):
    manifest_path.write_text(json.dumps(manifest))
    admission["approved_manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    admission_path.write_text(json.dumps(admission))


def test_release_is_bound_and_can_only_load_its_declared_splits(released):
    m, a, mp, ap = released
    checked, _, bindings = verify_release(mp, ap)
    assert checked == m
    assert bindings["manifest_sha256"] == a["approved_manifest_sha256"]
    for row in m["samples"]:
        sample = load_record(m, row)
        assert sample["purpose"] == ("training" if row["split"] == "train" else "diagnostic")


def test_unapproved_report_fails_before_opening_manifest(tmp_path):
    path = tmp_path / "admission.json"
    path.write_text(json.dumps({"training_authorized": False}))
    with pytest.raises(ValueError, match="must not start"):
        verify_release(tmp_path / "does-not-exist.json", path)


def test_toggled_flags_alone_do_not_admit_a_manifest(released):
    _, a, mp, ap = released
    a.pop("approved_manifest_sha256")
    ap.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="bound"):
        verify_release(mp, ap)


def test_manifest_mutation_requires_new_authorization(released):
    m, _, mp, ap = released
    m["seed"] += 1
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="bound"):
        verify_release(mp, ap)


@pytest.mark.parametrize("mutation", ["duplicate", "same_molecule", "isotope", "missing_full", "missing_check", "wrong_group"])
def test_release_cannot_hide_split_or_source_gaps(released, mutation):
    m, a, mp, ap = released
    if mutation == "duplicate":
        m["samples"].append(deepcopy(m["samples"][0]))
    elif mutation in {"same_molecule", "isotope"}:
        row = m["samples"][-1]
        row["canonical_smiles"] = "CCO" if mutation == "same_molecule" else "[13CH3]CO"
        row["group_key"] = identities(row["canonical_smiles"])["group_key"]
    elif mutation == "missing_full":
        m["samples"][1]["domain"] = "curriculum"
    elif mutation == "missing_check":
        a["release_checks"].pop("fresh_eval_identity")
    else:
        m["samples"][0]["group_key"] = "not-the-molecular-core"
    bind(m, a, mp, ap)
    with pytest.raises(ValueError):
        verify_release(mp, ap)


def test_metadata_changes_fail_on_access(released):
    m, _, _, _ = released
    row = m["samples"][0]
    from pathlib import Path
    path = Path(m["sources"][row["source"]]) / "samples" / (row["sample_key"] + ".meta.json")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_record(m, row)


def test_reviewed_mixture_order_is_deterministic_and_never_uses_validation(released):
    m, _, _, _ = released
    first = list(islice(training_order(m), 9))
    assert first == list(islice(training_order(m), 9))
    assert [r["domain"] for r in first] == m["train_cycle"] * 3
    assert all(r["split"] == "train" for r in first)
