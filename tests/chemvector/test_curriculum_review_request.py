"""Review handoff uses synthetic public/candidate inputs; no private payloads."""

from hashlib import sha256
import json
from pathlib import Path

import pytest
from rdkit import Chem

from chemvector.training.curriculum.admission import identities, require_training_admission
from chemvector.training.curriculum.review_request import build_request, run


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    roots = {name: tmp_path / name for name in ("curriculum", "full_molecule")}
    for name, root in roots.items():
        root.mkdir()
        smiles = ["CC", "Cc1ccccc1"] if name == "curriculum" else ["Oc1ccccc1", "CCO"]
        molecules, samples = [], []
        for i, smi in enumerate(smiles):
            mid = f"m{i}"
            molecule = dict(molecule_id=mid, canonical_smiles=smi, split="quarantine", training_authorized=False)
            if name == "full_molecule":
                molecule["source_provenance"] = dict(source_split="train", source_molecule={
                    "split": "train", "molecule_id": "old_" + mid, "inchikey": identities(smi)["inchikey"]})
            molecules.append(molecule)
            if not (name == "full_molecule" and i == 1):
                samples.extend(dict(molecule_id=mid, sample_key=mid + suffix) for suffix in ("__small", "__large"))
        write(root / "summary.json", dict(schema_version="curriculum-pilot-summary-0.1", status="quarantine_preview_only",
            training_authorized=False, unique_molecules=len(molecules), accepted_images=len(samples)))
        write(root / "molecules.json", molecules)
        (root / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in samples))
        write(root / "sample-sha256.json", {})
    index = tmp_path / "cache.json"
    write(index, dict(schema_version="chemvector-joint-overlap-index-0.1",
                     benchmark_inchikeys=[Chem.MolToInchiKey(Chem.MolFromSmiles("[13CH3]C")), ""]))
    commitment = tmp_path / "commitment.json"
    # These paths must never be followed (they intentionally do not exist).
    value = dict(schema_version="chemvector-joint-commitment-1.0", dataset="fresh_eval_12k",
                 sealed_access="locked_until_checkpoint_gate", split_counts={"fresh_dev": 8000, "sealed_test": 4000},
                 samples_csv=[{"path": "sealed_test/samples.csv"}], tars=[{"path": "sealed_test/test.tar"}])
    value["commitment_sha256"] = sha256((json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")) + "\n").encode()).hexdigest()
    write(commitment, value)
    return roots, index, commitment


def test_request_excludes_connectivity_hits_and_reports_group_collision(tmp_path):
    args = fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report, candidates, excluded, absent = build_request(*args)
    assert report["candidate_molecules"] == 2
    assert report["candidate_images"] == 4
    assert report["known_public_excluded_images"] == 2
    assert report["molecules_without_images"] == 1
    assert excluded[0]["canonical_smiles"] == "CC"
    assert len(excluded[0]["sample_keys"]) == 2
    assert absent[0]["image_count"] == 0
    assert len(report["cross_source_group_collisions"]["scaffold:c1ccccc1"]) == 2
    assert candidates[1]["intended_use"] == "old_train_identity_rerender_not_unseen_validation"
    assert all("split" not in row for row in candidates)
    assert report["cached_public_key_audit"]["invalid_entries"] == 1
    assert all(p.read_bytes() == payload for p, payload in before.items())
    with pytest.raises(ValueError, match="must not start"):
        require_training_admission(report)


def test_exact_request_binding_no_payload_reads_and_no_overwrite(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    paths = set()
    original = Path.read_bytes

    def tracked_read(path):
        paths.add(path)
        assert "sealed_test" not in path.parts
        assert path.suffix not in {".tar", ".png", ".sdf", ".pt"}
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    output = tmp_path / "request"
    first = run(*args, output)
    assert sha256((output / "candidate-identities.json").read_bytes()).hexdigest() == first["candidate_identities_sha256"]
    assert first == build_request(*args)[0]
    assert not first["payloads_reverified"]
    assert not first["sealed_test_sample_data_read"]
    assert all(path.exists() for path in paths)
    with pytest.raises(FileExistsError):
        run(*args, output)


def test_bad_commitment_fails_before_candidate_reads_or_output(tmp_path):
    roots, index, commitment = fixture(tmp_path)
    data = json.loads(commitment.read_text())
    data["split_counts"]["sealed_test"] = 0
    write(commitment, data)
    with pytest.raises(ValueError, match="aggregate"):
        run({key: tmp_path / "missing" for key in roots}, index, commitment, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("mutation", ["count", "duplicate", "unknown", "unsafe", "not_quarantined", "eval_provenance", "wrong_key"])
def test_invalid_source_is_not_packaged(tmp_path, mutation):
    roots, index, commitment = fixture(tmp_path)
    root = roots["full_molecule"]
    molecules = json.loads((root / "molecules.json").read_text())
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    if mutation == "count":
        molecules.pop()
    elif mutation == "duplicate":
        samples[1] = samples[0]
    elif mutation == "unknown":
        samples[0]["molecule_id"] = "unknown"
    elif mutation == "unsafe":
        samples[0]["sample_key"] = "../sealed_test/samples"
    elif mutation == "not_quarantined":
        molecules[0]["training_authorized"] = True
    elif mutation == "eval_provenance":
        molecules[0]["source_provenance"]["source_split"] = "validation"
    else:
        molecules[0]["source_provenance"]["source_molecule"]["inchikey"] = "wrong"
    write(root / "molecules.json", molecules)
    (root / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in samples))
    with pytest.raises(ValueError):
        run(roots, index, commitment, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_symlinked_manifest_and_missing_cache_keys_are_rejected(tmp_path):
    roots, index, commitment = fixture(tmp_path)
    link = tmp_path / "linked.json"
    link.symlink_to(commitment)
    with pytest.raises(ValueError, match="non-symlinked"):
        build_request(roots, index, link)
    write(index, dict(schema_version="chemvector-joint-overlap-index-0.1", benchmark_inchikeys=[""]))
    with pytest.raises(ValueError, match="valid public"):
        build_request(roots, index, commitment)
