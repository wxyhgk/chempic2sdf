from __future__ import annotations

import pytest
import torch

import chemvector.hybrid_graph_refinement as refinement


def _graph_runtime(**overrides) -> refinement.GraphCorrectionRuntime:
    values = {
        "device": torch.device("cpu"),
        "corrector": object(),
        "default_keep_threshold": 0.3,
        "uses_elements": False,
    }
    values.update(overrides)
    return refinement.GraphCorrectionRuntime(**values)


def _semantic_runtime(**overrides) -> refinement.SemanticBondRuntime:
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
    return refinement.SemanticBondRuntime(**values)


def test_empty_graph_correction_uses_effective_threshold() -> None:
    corrected, diagnostics = refinement.correct_bond_objects(
        torch.empty((0, 2)),
        torch.empty((0,), dtype=torch.long),
        [],
        [],
        mode="full",
        graph_keep_threshold=None,
        runtime=_graph_runtime(),
    )
    assert corrected == []
    assert diagnostics["graph_keep_threshold"] == 0.3
    assert diagnostics["graph_removed_bond_objects"] == 0


def test_graph_correction_requires_loaded_model_even_for_empty_graph() -> None:
    with pytest.raises(ValueError, match="requires a graph corrector"):
        refinement.correct_bond_objects(
            torch.empty((0, 2)),
            torch.empty((0,), dtype=torch.long),
            [],
            [],
            mode="full",
            graph_keep_threshold=None,
            runtime=_graph_runtime(corrector=None),
        )


def test_disabled_semantic_relation_preserves_bonds() -> None:
    bond = {"junction_pair": (0, 1), "chemical": 0}
    bonds = [bond]
    updated, diagnostics = refinement.apply_semantic_bond_types(
        torch.zeros((1, 3, 16, 16)),
        torch.tensor(((0.2, 0.5), (0.8, 0.5))),
        bonds,
        runtime=_semantic_runtime(),
    )
    assert updated == [bond]
    assert updated is not bonds
    assert diagnostics["semantic_bond_relation_enabled"] is False


def test_semantic_relation_updates_only_chemical_type() -> None:
    class Relation:
        def __call__(self, pixels, points, pair_indices, pair_mask):
            return {"chemical_logits": torch.tensor(((0.0, 4.0, 0.0, 0.0),))}

    bond = {"junction_pair": (0, 1), "chemical": 0, "display": 0}
    updated, diagnostics = refinement.apply_semantic_bond_types(
        torch.zeros((1, 3, 16, 16)),
        torch.tensor(((0.2, 0.5), (0.8, 0.5))),
        [bond],
        runtime=_semantic_runtime(relation=Relation()),
    )
    assert updated[0]["chemical"] == 1
    assert updated[0]["display"] == 0
    assert diagnostics["semantic_chemical_type_changes"] == 1


def test_semantic_existence_scores_each_pair_once(monkeypatch) -> None:
    class Existence:
        def __call__(self, pixels, points, pair_indices, pair_mask):
            count = pair_indices.size(1)
            return {
                "edge_logits": torch.full((count,), 10.0),
                "display_logits": torch.tensor(((4.0, 0.0, 0.0, 0.0),)).repeat(count, 1),
                "chemical_logits": torch.tensor(((4.0, 0.0, 0.0, 0.0),)).repeat(count, 1),
            }

    monkeypatch.setattr(
        refinement,
        "apply_aromatic_ring_semantics",
        lambda objects, logits, **kwargs: (list(objects), {"aromatic_ring_changes": 0}),
    )
    bonds, diagnostics = refinement.semantic_relation_bonds(
        torch.zeros((1, 3, 16, 16)),
        torch.tensor(((0.1, 0.1), (0.5, 0.8), (0.9, 0.1))),
        runtime=_semantic_runtime(existence=Existence(), existence_chunk_size=2),
    )
    assert len(bonds) == 3
    assert diagnostics["semantic_bond_pairs_scored"] == 3
    assert diagnostics["semantic_bond_pairs_selected"] == 3
