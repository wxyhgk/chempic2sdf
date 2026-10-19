"""Synthetic identities only: no benchmark payloads or real registry required."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from chemvector.training.curriculum.admission import identities, require_training_admission, run, screen


def fixtures():
    records = [dict(record_id=str(i), stage="atoms_bonds", name=str(i), smiles=s, concepts=[])
               for i, s in enumerate(["CC", "CCO", "[13CH3]CO", "CO", "c1ccccc1", "Cc1ccccc1"])]
    molecules = merged_records(records)
    samples = [{"molecule_id": r["molecule_id"], "sample_key": r["molecule_id"] + "__" + p}
               for r in molecules for p in ("reference", "serif")]
    ethane = Chem.MolToInchiKey(Chem.MolFromSmiles("CC"))
    index = {"schema_version": "chemvector-joint-overlap-index-0.1", "benchmark_inchikeys": [ethane],
             "benchmark_scaffolds": ["ACYCLIC:CC"], "inchikeys": [ethane], "audit": {"synthetic": {"unresolved": 1}}}
    return molecules, samples, index


def test_known_overlap_excludes_every_variant_but_does_not_delete_inputs():
    molecules, samples, index = fixtures()
    before = deepcopy((molecules, samples, index))
    report, rows, proposed = screen(molecules, samples, index)
    assert before == (molecules, samples, index)
    assert report["known_benchmark_exact_molecules"] == 1
    assert report["images_by_proposal"]["excluded_known_benchmark"] == 2
    assert len(proposed) == len(samples)
    assert all(not r["training_authorized"] for r in rows + proposed)


def test_core_and_scaffold_grouping_keeps_related_variants_together():
    assert identities("CCO")["group_key"] == identities("[13CH3]CO")["group_key"]
    assert identities("c1ccccc1")["group_key"] == identities("Cc1ccccc1")["group_key"]
    assert identities("CC")["group_key"] != identities("CO")["group_key"]
    report, rows, _ = screen(*fixtures())
    by_smiles = {r["canonical_smiles"]: r for r in rows}
    assert by_smiles["CCO"]["proposed_split"] == by_smiles["[13CH3]CO"]["proposed_split"]
    assert report["proposed_train_validation_group_overlap"] == 0


def test_split_proposal_is_order_invariant_and_not_admission():
    molecules, samples, index = fixtures()
    first = screen(molecules, samples, index)
    second = screen(list(reversed(molecules)), list(reversed(samples)), index)
    assert first[0] == second[0]
    assert sorted(first[2], key=lambda r: r["sample_key"]) == sorted(second[2], key=lambda r: r["sample_key"])
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(first[0])
    changed = dict(first[0], training_authorized=True, external_overlap_status="checked")
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(changed)


@pytest.mark.parametrize("invalid", ["", "not-an-inchikey", None, {"unresolved": True}])
def test_malformed_cached_keys_are_coverage_gaps_not_negative_identity_evidence(invalid):
    molecules, samples, index = fixtures()
    index["benchmark_inchikeys"].append(invalid)
    index["inchikeys"].append(invalid)
    before = deepcopy(index)
    report, rows, proposed = screen(molecules, samples, index)
    assert index == before
    assert report["known_benchmark_exact_molecules"] == 1
    assert report["images_by_proposal"]["excluded_known_benchmark"] == 2
    for name in ("benchmark_inchikeys", "inchikeys"):
        assert report["cached_identity_key_audit"][name] == {
            "entries": 2, "valid_unique_keys": 1, "invalid_entries": 1}
        assert any(name in gap and "invalid identity" in gap for gap in report["coverage_gaps"])
    assert all(not row["training_authorized"] for row in rows + proposed)
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(report)


@pytest.mark.parametrize("entries", [[], [""], None, "not-a-list"])
def test_index_without_valid_benchmark_keys_fails(entries):
    molecules, samples, index = fixtures()
    index["benchmark_inchikeys"] = entries
    with pytest.raises(ValueError):
        screen(molecules, samples, index)


def test_missing_coverage_declaration_cannot_authorize_training():
    report = {"schema_version": "chemvector-curriculum-admission-0.1",
              "training_authorized": True, "external_overlap_status": "checked"}
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(report)


def test_run_writes_separate_auditable_proposals_without_rewriting_inputs(tmp_path):
    molecules, samples, index = fixtures()
    index["benchmark_inchikeys"].append("")
    root = tmp_path / "data"
    root.mkdir()
    paths = {"molecules": root / "molecules.json", "samples": root / "samples.jsonl",
             "index": tmp_path / "cached-index.json"}
    paths["molecules"].write_text(json.dumps(molecules))
    paths["samples"].write_text("\n".join(json.dumps(r) for r in samples))
    paths["index"].write_text(json.dumps(index))
    before = {name: path.read_bytes() for name, path in paths.items()}
    output = tmp_path / "audit"
    report = run(root, paths["index"], output)
    assert json.loads((output / "report.json").read_text()) == report
    assert len(json.loads((output / "split-proposal.json").read_text())) == len(samples)
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert {name: row["sha256"] for name, row in report["inputs"].items()} == {
        name: sha256(data).hexdigest() for name, data in before.items()}
    with pytest.raises(FileExistsError):
        run(root, paths["index"], output)
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(report)


@pytest.mark.parametrize("mutation", ["molecule", "sample", "unknown", "path", "index"])
def test_bad_inputs_fail_closed(mutation):
    molecules, samples, index = fixtures()
    if mutation == "molecule": molecules.append(molecules[0])
    elif mutation == "sample": samples.append(samples[0])
    elif mutation == "unknown": samples[0]["molecule_id"] = "unknown"
    elif mutation == "path": samples[0]["sample_key"] = "../bad"
    elif mutation == "index": index["benchmark_inchikeys"] = []
    with pytest.raises(ValueError): screen(molecules, samples, index)
