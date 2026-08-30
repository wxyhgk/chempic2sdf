from __future__ import annotations

import pytest
import torch

import chemvector.hybrid_junction_bonds as junction_bonds
from chemvector.bond_endpoint_junction import BondEndpointJunctionResult
from chemvector.hybrid_graph_refinement import SemanticBondRuntime
from chemvector.junction_aware_bond_selector import JunctionAwareSelectionDiagnostics


def _semantic(**overrides) -> SemanticBondRuntime:
    values = {
        "device": torch.device("cpu"),
        "relation": None,
        "relation_mode": "chemical_all",
        "aromatic_mean_threshold": 0.72,
        "aromatic_minimum_threshold": 0.25,
        "existence": None,
        "existence_threshold": 0.6,
        "existence_chunk_size": 256,
    }
    values.update(overrides)
    return SemanticBondRuntime(**values)


def _runtime(**overrides) -> junction_bonds.HybridJunctionBondRuntime:
    values = {
        "bond_set_selector": object(),
        "bond_set_selector_payload": object(),
        "endpoint_grouper": object(),
        "endpoint_grouper_payload": object(),
        "line_band_reader": object(),
        "projection_score_source": "detector",
        "bond_type_source": "detector",
        "semantic": _semantic(),
    }
    values.update(overrides)
    return junction_bonds.HybridJunctionBondRuntime(**values)


def test_partial_junction_aware_runtime_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires all v0.8.21 checkpoints"):
        junction_bonds.junction_aware_bonds(
            torch.zeros((1, 3, 16, 16)),
            [],
            runtime=_runtime(line_band_reader=None),
        )


def test_junction_aware_stage_returns_explicit_diagnostics(monkeypatch) -> None:
    bond = {
        "score": 0.9,
        "endpoints": torch.tensor(((0.2, 0.5), (0.8, 0.5))),
        "junction_pair": (0, 1),
        "display": 0,
        "chemical": 0,
        "direction": 0,
    }
    result = BondEndpointJunctionResult(
        points=torch.tensor(((0.2, 0.5), (0.8, 0.5))),
        bond_objects=(bond,),
        raw_clusters=2,
        self_loops=0,
        duplicate_pairs=0,
    )
    selection_diagnostics = JunctionAwareSelectionDiagnostics(
        raw_candidates=1,
        threshold_selected=1,
        grouped_bonds=1,
        predicted_junctions=2,
        keep_threshold=0.5,
        projection_score_source="detector",
        bond_type_source="detector",
        self_loop_objects=0,
        duplicate_pair_objects=0,
    )
    monkeypatch.setattr(
        junction_bonds,
        "read_line_band_features",
        lambda reader, pixels, batches: [torch.zeros((1, 4))],
    )
    monkeypatch.setattr(
        junction_bonds,
        "select_and_group_bonds",
        lambda *args, **kwargs: (result, selection_diagnostics),
    )
    monkeypatch.setattr(
        junction_bonds,
        "apply_semantic_bond_types",
        lambda *args, **kwargs: ([{**bond, "chemical": 1}], {"semantic_test": 1}),
    )
    runtime = _runtime(semantic=_semantic(relation=object()))
    updated, diagnostics = junction_bonds.junction_aware_bonds(
        torch.zeros((1, 3, 16, 16)), [bond], runtime=runtime
    )
    assert updated.bond_objects[0]["chemical"] == 1
    assert diagnostics["raw_candidates"] == 1
    assert diagnostics["semantic_test"] == 1
