from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
from pathlib import Path

import pytest
from rdkit import Chem, rdBase

from chemvector.training.curriculum import public_review
from chemvector.training.curriculum.admission import identities, require_training_admission


def candidates():
    return [dict(source="curriculum", molecule_id=f"candidate_{i}", canonical_smiles=smi,
                 image_count=2, sample_keys=[f"candidate_{i}__small", f"candidate_{i}__large"], **identities(smi))
            for i, smi in enumerate(("CCO", "CCN", "CCCO"))]


def fixture(tmp_path):
    root = tmp_path / "public"
    (root / "manifests").mkdir(parents=True)
    for subset, smi in zip(public_review.SUBSETS, ("CCO", "CCN", "c1ccccc1")):
        sid = subset + "_synthetic"
        name = f"prepared/{subset}/source_structures/{sid}.mol"
        path = root / name
        path.parent.mkdir(parents=True)
        block = Chem.MolToMolBlock(Chem.MolFromSmiles(smi))
        if subset == "uspto_10k_l":
            block = "force_unresolved" + block
        path.write_text(block)
        row = dict(sample_id=sid, source_structure_file=name, source_structure_sha256=sha256(path.read_bytes()).hexdigest())
        with (root / "manifests" / (subset + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    return root


def force_one_unresolved(monkeypatch):
    original = public_review.source_identity
    monkeypatch.setattr(public_review, "source_identity", lambda block: (None, "simulated_error") if block.startswith("force_unresolved") else original(block))
    monkeypatch.setattr(public_review, "native_probe", lambda block: {"key": None, "return_code": 2, "message": "synthetic", "use": "additional_positive_exclusion_only"})


def test_actual_native_inchi_probe_does_not_edit_molfile_or_drop_warnings():
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
    before = block
    record = public_review.native_probe(block)
    assert record["return_code"] == 0
    assert record["key"] == identities("CCO")["inchikey"]
    assert record["use"] == "additional_positive_exclusion_only"
    assert block == before


def test_opaque_sup_atoms_cannot_supply_false_negative_evidence(monkeypatch):
    rows = candidates(); by_ref, inventories = public_review.candidate_inventory(rows)
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CCN")).replace("M  END", "M  STY  1   1 SUP\nM  SAL   1  1   3\nM  SMT   1 N\nH2\nM  END")
    monkeypatch.setattr(public_review, "native_probe", lambda block: {"key": None})
    result = public_review.screen_unresolved(block, by_ref, inventories)
    assert result["inventory"]["guaranteed_non_hydrogen_inventory"] == {"C": 2}
    assert all(row["excluded_conservatively"] for row in result["pairs"])


def test_unsupported_syntax_blocks_all_and_native_negative_never_clears(monkeypatch):
    by_ref, inventories = public_review.candidate_inventory(candidates())
    monkeypatch.setattr(public_review, "native_probe", lambda block: {"key": identities("C")["inchikey"]})
    result = public_review.screen_unresolved("unparseable source", by_ref, inventories)
    assert result["unsupported_inventory"]
    assert all(p["excluded_conservatively"] and p["deficiency_witness"] is None for p in result["pairs"])


def test_native_positive_is_retained_even_if_inventory_disagrees(monkeypatch):
    by_ref, inventories = public_review.candidate_inventory(candidates())
    monkeypatch.setattr(public_review, "native_probe", lambda block: {"key": identities("CCO")["inchikey"]})
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("[Na+]"))
    result = public_review.screen_unresolved(block, by_ref, inventories)
    first = result["pairs"][0]
    assert first["deficiency_witness"] is not None
    assert first["native_key_positive_hit"] and first["excluded_conservatively"]


def test_scan_full_coverage_witnesses_and_all_variants(tmp_path, monkeypatch):
    root = fixture(tmp_path); force_one_unresolved(monkeypatch)
    rows = candidates(); before = deepcopy(rows)
    by_ref, inventories = public_review.candidate_inventory(rows)
    index, unresolved, pairs, excluded, bindings = public_review.scan_public(root, by_ref, inventories, expected_per_subset=1)
    assert len(index) == 3 and len(unresolved) == 1 and len(pairs) == 3
    assert set(excluded) == {"curriculum:candidate_0", "curriculum:candidate_1"}
    assert sum(r["deficiency_witness"] is not None for r in pairs) == 2
    assert len(bindings) == 3
    assert rows == before
    assert all(row["image_count"] == len(row["sample_keys"]) == 2 for row in rows)


@pytest.mark.parametrize("mutation", ["hash", "path", "symlink", "missing", "count", "duplicate"])
def test_missing_or_unbound_public_source_fails_closed(tmp_path, mutation):
    root = fixture(tmp_path)
    manifest = root / "manifests/uspto_10k.csv"
    row = next(csv.DictReader(io.StringIO(manifest.read_text())))
    source = root / row["source_structure_file"]
    if mutation == "hash":
        source.write_text(source.read_text() + "changed")
    elif mutation == "path":
        row["source_structure_file"] = "../images/private.png"
    elif mutation == "symlink":
        target = source.with_suffix(".copy"); source.rename(target); source.symlink_to(target)
    elif mutation == "missing":
        source.rename(source.with_suffix(".missing"))
    with manifest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
        if mutation != "count": writer.writerow(row)
        if mutation == "duplicate": writer.writerow(row)
    by_ref, inventories = public_review.candidate_inventory(candidates())
    with pytest.raises((ValueError, FileNotFoundError)):
        public_review.scan_public(root, by_ref, inventories, expected_per_subset=1)


def test_run_produces_hash_bound_subset_without_modifying_sources(tmp_path, monkeypatch):
    root = fixture(tmp_path); force_one_unresolved(monkeypatch)
    rows = candidates(); request_dir = tmp_path / "request"; request_dir.mkdir()
    candidate_bytes = json.dumps(rows).encode()
    (request_dir / "candidate-identities.json").write_bytes(candidate_bytes)
    request = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
                   candidate_identities_sha256=sha256(candidate_bytes).hexdigest(),
                   identity_implementation_sha256=public_review.digest(Path(public_review.__file__).with_name("admission.py")),
                   rdkit_version=rdBase.rdkitVersion, candidate_molecules=3, candidate_images=6)
    (request_dir / "request.json").write_text(json.dumps(request))
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(dict(schema_version="chemvector-joint-overlap-index-0.1",
                                     benchmark_inchikeys=[identities("CF")["inchikey"], ""])))
    request["input_sha256"] = {str(cache.resolve()): public_review.digest(cache)}
    (request_dir / "request.json").write_text(json.dumps(request))
    original = public_review.scan_public
    monkeypatch.setattr(public_review, "scan_public", lambda root, rows, inv: original(root, rows, inv, expected_per_subset=1))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "output"
    report = public_review.run(request_dir, root, cache, output)
    assert report["remaining_molecules"] == 1 and report["remaining_images"] == 2
    assert report["conservatively_excluded_images"] == 4
    assert report["remaining_ambiguous_pairs"] == 0
    assert report["unresolved_global_identity_records"] == 1
    assert report["remaining_candidates_sha256"] == public_review.digest(output / "remaining-candidates.json")
    assert json.loads((output / "remaining-candidates.json").read_text()) == [rows[2]]
    assert all(path.read_bytes() == payload for path, payload in before.items())
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(report)
    with pytest.raises(FileExistsError):
        public_review.run(request_dir, root, cache, output)


def test_inchi_exception_is_an_unresolved_record_not_silently_lost(monkeypatch):
    monkeypatch.setattr(Chem, "MolFromMolBlock", lambda *a, **kw: (_ for _ in ()).throw(ValueError("parse failed")))
    assert public_review.source_identity("input") == (None, "molblock_exception")
