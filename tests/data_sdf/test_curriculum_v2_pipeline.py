"""Synthetic-only integration of the new quarantine generator."""

import json
from types import SimpleNamespace

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_2.catalog import entries
from Data.sdf.chemvector_curriculum_v0_2.coverage import chemical_features, summarize, visible_coverage
from Data.sdf.chemvector_curriculum_v0_2 import build as builder


def records():
    return [
        {"record_id": "a", "stage": "atoms_bonds", "name": "nitrile", "smiles": "CC#N", "concepts": ["triple_bond"]},
        {"record_id": "b", "stage": "functional_groups", "name": "nitrile context", "smiles": "N#CC", "concepts": ["nitrile"]},
    ]


def test_cross_agent_merge_preserves_provenance_and_quarantine():
    merged = entries(records())
    assert len(merged) == 1
    assert merged[0]["source_record_ids"] == ["a", "b"]
    assert len(merged[0]["provenance"]["source_records"]) == 2
    assert merged[0]["training_authorized"] is False
    assert merged[0]["external_overlap_status"] == "unchecked"
    assert merged[0]["split"] == "quarantine"
    assert entries(list(reversed(records()))) == merged


def test_coverage_does_not_count_styles_as_distinct_molecules():
    merged = entries(records())
    planned = summarize(merged)
    accepted = summarize(merged, [{"molecule_id": merged[0]["molecule_id"]}] * 6)
    assert planned["instances"]["bond_types"]["TRIPLE"] == 1
    assert accepted["instances"]["bond_types"]["TRIPLE"] == 6
    assert accepted["examples"] == 6 and accepted["distinct_molecules"] == 1
    assert accepted["predicted_candidate_reachability"].startswith("not_measured")


def test_stereochemistry_is_separate_from_chemical_bond_order():
    alkene = chemical_features("C/C=C/C")
    chiral = chemical_features("F[C@H](Cl)Br")
    assert alkene["bond_stereo"] == {"STEREOE": 1}
    assert alkene["bond_types"] == {"SINGLE": 2, "DOUBLE": 1}
    assert chiral["specified_tetrahedral_centers"] == 1
    assert chiral["bond_types"] == {"SINGLE": 3}


def test_visible_coverage_reports_drawing_and_non_edges_separately():
    layers = {
        "atoms": [
            {"atom_id": 0, "center": [0, 0], "element": "C", "visible_text": False},
            {"atom_id": 1, "center": [1, 0], "element": "C", "visible_text": False},
            {"atom_id": 2, "center": [0, 1], "element": "O", "visible_text": True},
        ],
        "bonds": [
            {"begin_atom": 0, "end_atom": 1, "chemical_type": "SINGLE",
             "drawing": {"bond_type": "SINGLE", "direction": "BEGINWEDGE"}},
            {"begin_atom": 0, "end_atom": 2, "chemical_type": "SINGLE",
             "drawing": {"bond_type": "SINGLE", "direction": "NONE"}},
        ],
    }
    result = visible_coverage(layers)
    assert result["chemical_bond_types"] == {"SINGLE": 2}
    assert result["drawing_directions"]["BEGINWEDGE"] == 1
    assert result["nearby_target_non_edges"] == 1
    assert result["not_predicted_pair_training_labels"] is True


def test_accounting_requires_every_planned_image_including_rejections():
    molecules, profiles = [{"molecule_id": "m"}], [SimpleNamespace(profile_id="a"), SimpleNamespace(profile_id="b")]
    accepted = [{"molecule_id": "m", "profile_id": "a"}]
    rejected = [{"molecule_id": "m", "profile_id": "b"}]
    builder.validate_accounting(molecules, profiles, accepted, rejected)
    with pytest.raises(ValueError):
        builder.validate_accounting(molecules, profiles, accepted, [])
    with pytest.raises(ValueError):
        builder.validate_accounting(molecules, profiles, accepted * 2, rejected)


def test_real_six_profile_build_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(builder, "entries", lambda: entries(records()))
    monkeypatch.setattr(builder, "source_bindings", lambda: {})
    monkeypatch.setattr(builder, "bounded_map", lambda function, jobs, **kwargs: (function(job) for job in jobs))
    output = tmp_path / "dataset"
    summary = builder.build(output, workers=1)
    assert summary["planned_images"] == 6
    assert summary["accepted_images"] == 6
    assert summary["training_authorized"] is False
    assert summary["model_forwards"] == 0
    assert (output / "index.html").is_file()
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(row["status"] == "accepted" for row in rows)
    meta = json.loads(next((output / "samples").glob("*.meta.json")).read_text())
    assert meta["split"] == "quarantine"
    assert meta["training_authorized"] is False
    assert meta["external_overlap_status"] == "unchecked"
    with pytest.raises(FileExistsError):
        builder.build(output, workers=1)


def test_refuses_non_cpu_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        builder.build(tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
