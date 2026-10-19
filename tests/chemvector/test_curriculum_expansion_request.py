"""Expansion handoff does not open hidden data, run reviews or admit training."""

from hashlib import sha256
import json
from pathlib import Path

import pytest
from rdkit import rdBase

from Data.sdf.chemvector_curriculum_v0_1.build import build
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.expansion import write_json
from chemvector.training.curriculum.expansion_inventory import run as inventory
from chemvector.training.curriculum.expansion_plan import selected_profiles
from chemvector.training.curriculum.expansion_request import run
from chemvector.training.curriculum.full_molecule_data import file_hash
from chemvector.training.curriculum.preflight import run as preflight
from tests.chemvector.test_curriculum_review_request import fixture as review_fixture


def fixture(tmp_path):
    _, cache, commitment = review_fixture(tmp_path)
    value = json.loads(commitment.read_text())
    value["rdkit_version"] = rdBase.rdkitVersion
    value.pop("commitment_sha256")
    value["commitment_sha256"] = sha256((json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")) + "\n").encode()).hexdigest()
    commitment.write_text(json.dumps(value))
    expanded = tmp_path / "expanded"
    expanded.mkdir()
    molecule = merged_records([dict(record_id="test", name="test", stage="full_molecule", concepts=[], smiles="CCO")])[0]
    molecule["source_provenance"] = dict(source_split="train", source_molecule=dict(
        source="chembl", split="train", inchikey=identities("CCO")["inchikey"], molecule_id="original-test"))
    build(expanded / "dataset", entries=[molecule], selected_profiles=selected_profiles()[:2],
          workers=1, typography="stereo-aware-v1", gallery=False)
    write_json(expanded / "plan.json", dict(planned_images=2, input_sha256={str(cache.resolve()): file_hash(cache)}))
    write_json(expanded / "report.json", dict(plan_sha256=file_hash(expanded / "plan.json"), chemistry_excluded_images=0))
    checked, counted = tmp_path / "checked", tmp_path / "counted"
    preflight(expanded / "dataset", checked)
    inventory(expanded, checked, counted)
    return expanded, counted, cache, commitment


def test_packet_binds_exact_unique_candidates_without_private_access_or_admission(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = Path.read_bytes
    def safe_read(path):
        assert "sealed_test" not in path.parts and "fresh_dev" not in path.parts
        assert path.suffix not in {".png", ".sdf", ".tar", ".pt"}
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", safe_read)
    output = tmp_path / "request"
    report = run(*args, output)
    candidates = json.loads((output / "candidate-identities.json").read_text())
    assert report["candidate_molecules"] == 1 and report["candidate_images"] == 2
    assert report["candidate_identities_sha256"] == file_hash(output / "candidate-identities.json")
    assert candidates[0]["image_count"] == len(candidates[0]["sample_keys"]) == 2
    assert not report["training_authorized"] and not report["private_identity_review_executed"]
    assert not report["payloads_reverified"] and not report["sealed_test_sample_data_read"]
    with pytest.raises(FileExistsError):
        run(*args, output)


@pytest.mark.parametrize("mutation", ["chosen", "plan", "commitment"])
def test_changed_evidence_cannot_be_packaged(tmp_path, mutation):
    args = fixture(tmp_path)
    expanded, counted, _, commitment = args
    if mutation == "chosen":
        path = counted / "unique-candidates.json"
        rows = json.loads(path.read_text())
        rows.append(rows[0])
        path.write_text(json.dumps(rows))
    elif mutation == "plan":
        path = expanded / "plan.json"
        path.write_text(path.read_text() + "\n")
    else:
        value = json.loads(commitment.read_text())
        value["sealed_access"] = "open"
        commitment.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run(*args, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
