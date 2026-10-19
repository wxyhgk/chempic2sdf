"""Fixed synthetic matrix, honest denominators, and isolated runner verification."""

import copy
import json
from pathlib import Path
import subprocess

import pytest
from rdkit import Chem

from Data.sdf.text_source_coverage.matrix import build_matrix
from Data.sdf.text_source_coverage.outcomes import rejection_reason, summarize
from Data.sdf.text_source_coverage.run import launch, run, source_hashes, verify_result
from Data.sdf.text_source_coverage.worker import render_sample
from Data.sdf.text_source_renderer.layers import json_bytes, sha256


@pytest.fixture
def protocol():
    return {"matrix": build_matrix()}


def test_matrix_frozen_denominators_and_chemical_replay(protocol):
    matrix = protocol["matrix"]
    assert matrix == build_matrix()
    assert len(matrix["molecules"]) == 35
    assert len(matrix["samples"]) == 140
    assert sum(s["expected_rejection"] is None for s in matrix["samples"]) == 128
    assert len({s["sample_key"] for s in matrix["samples"]}) == 140
    assert len({s["group"] for s in matrix["samples"]}) == 9
    for row in matrix["molecules"]:
        mol = Chem.MolFromMolBlock(row["molblock"], removeHs=False)
        assert mol.GetNumAtoms() == row["atom_count"]
        assert mol.GetNumBonds() == row["bond_count"]
        assert Chem.MolToSmiles(mol) == row["canonical_smiles"]


@pytest.mark.parametrize("message,reason", [
    ("Overlapping glyph ink is outside the reference contract", "glyph_overlap"),
    ("Overlapping text/geometry ink is outside the reference contract", "text_geometry_overlap"),
    ("Auxiliary junction has no unique incident-bond geometric witness", "auxiliary_witness_missing_or_ambiguous"),
    ("Recorded glyph touches canvas edge; possible clipping", "canvas_clipping"),
    ("Recorded bond produced no raster pixels", "empty_raster_object"),
    ("Required reference font is unavailable; no silent fallback", None),
    ("Random code error", None),
])
def test_error_classification_does_not_hide_unexpected_errors(message, reason):
    assert rejection_reason(message) == reason


def results_for(protocol):
    return [{"sample_key": s["sample_key"], "status": "rejected", "reason": s["expected_rejection"]}
            if s["expected_rejection"] else {"sample_key": s["sample_key"], "status": "rendered"}
            for s in protocol["matrix"]["samples"]]


def test_controls_are_not_in_primary_denominator(protocol):
    rows = results_for(protocol)
    rows[0] = {"sample_key": rows[0]["sample_key"], "status": "rejected", "reason": "glyph_overlap"}
    result = summarize(protocol, rows)
    assert result["experiment_valid"]
    assert result["primary"]["attempted"] == 128
    assert result["primary"]["rendered"] == 127
    assert result["primary"]["observed_acceptance_fraction"] == 127 / 128
    assert result["negative_controls"]["attempted"] == 12
    assert result["molecules"]["all_configs_rendered"] == 31
    assert len(result["review_sample_keys"]) == 8


@pytest.mark.parametrize("mutation", ["error", "wrong_control", "accepted_control"])
def test_errors_and_failed_controls_invalidate_experiment(protocol, mutation):
    rows = results_for(protocol)
    if mutation == "error":
        rows[0].update(status="error", reason="worker_process_failed")
    elif mutation == "wrong_control":
        rows[-1]["reason"] = "glyph_overlap"
    else:
        rows[-1] = {"sample_key": rows[-1]["sample_key"], "status": "rendered"}
    result = summarize(protocol, rows)
    assert not result["experiment_valid"]
    assert result["primary"]["attempted"] == 128


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "unknown_status"])
def test_incomplete_or_invalid_result_matrix_is_rejected(protocol, mutation):
    rows = results_for(protocol)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "extra":
        rows.append({"sample_key": "unregistered", "status": "rendered"})
    else:
        rows[0]["status"] = "skipped"
    with pytest.raises(ValueError):
        summarize(protocol, rows)


def test_worker_roundtrip_and_bundle_integrity(protocol, tmp_path):
    sample = protocol["matrix"]["samples"][0]
    result = render_sample(protocol, sample["sample_key"], tmp_path)
    assert result["status"] == "rendered"
    assert verify_result(result, sample, tmp_path) == result
    bad = copy.deepcopy(result)
    bad["files"][sample["sample_key"] + ".png"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_result(bad, sample, tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        verify_result({"sample_key": sample["sample_key"], "status": "rendered", "files": {}}, sample, tmp_path)
    with pytest.raises(FileExistsError):
        render_sample(protocol, sample["sample_key"], tmp_path)


def test_worker_controls_do_not_write_success_bundles(protocol, tmp_path):
    controls = [s for s in protocol["matrix"]["samples"] if s["expected_rejection"]]
    for sample in controls:
        result = render_sample(protocol, sample["sample_key"], tmp_path)
        assert result["status"] == "rejected"
        assert result["reason"] == sample["expected_rejection"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["timeout", "crash", "bad_json", "wrong_identity"])
def test_subprocess_failures_stay_errors(protocol, tmp_path, monkeypatch, mode):
    def fake(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], 30)
        if mode == "crash":
            return subprocess.CompletedProcess(args[0], -11, "", "crash")
        if mode == "bad_json":
            return subprocess.CompletedProcess(args[0], 0, "invalid", "")
        return subprocess.CompletedProcess(args[0], 0, '{"sample_key":"other","status":"rendered"}', "")
    monkeypatch.setattr(subprocess, "run", fake)
    result = launch(protocol["matrix"]["samples"][0], tmp_path, "hash")
    assert result["status"] == "error"


def test_runner_small_matrix_freezes_sources_and_refuses_overwrite(protocol, tmp_path, monkeypatch):
    import importlib
    runner = importlib.import_module("Data.sdf.text_source_coverage.run")
    matrix = copy.deepcopy(protocol["matrix"])
    matrix["samples"] = [matrix["samples"][0], matrix["samples"][-1]]
    monkeypatch.setattr(runner, "build_matrix", lambda: matrix)
    output = tmp_path / "run"
    assert run(output, workers=2) == 0
    frozen = json.loads((output / "protocol.json").read_bytes())
    report = json.loads((output / "report.json").read_bytes())
    assert frozen["sources"] == source_hashes()
    assert report["protocol_sha256"] == sha256((output / "protocol.json").read_bytes())
    assert report["primary"]["attempted"] == 1
    assert report["negative_controls"]["attempted"] == 1
    with pytest.raises(FileExistsError):
        run(output)


def test_protocol_tampering_refused_before_render(protocol, tmp_path):
    import sys

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(json_bytes(protocol))
    process = subprocess.run([sys.executable, "-m", "Data.sdf.text_source_coverage.worker",
                              "--protocol", str(protocol_path), "--protocol-sha256", "0" * 64,
                              "--sample-key", "butane__baseline", "--output", str(tmp_path)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert "Protocol changed" in process.stderr
    assert list(tmp_path.iterdir()) == [protocol_path]


def test_new_coverage_package_is_outside_frozen_renderer_tree():
    paths = source_hashes()
    assert any(name.startswith("Data/sdf/text_source_coverage/") for name in paths)
    assert not any("chemvector" in name for name in paths)
    assert Path("Data/sdf/text_source_coverage").parent == Path("Data/sdf/text_source_renderer").parent
