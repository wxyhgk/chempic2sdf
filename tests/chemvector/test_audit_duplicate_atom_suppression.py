import torch

from chemvector.audit_duplicate_atom_suppression import (
    evaluate_sample,
    graph_duplicate_candidates,
    suppress_graph_duplicates,
)


def duplicate_sample() -> dict:
    return {
        "key": "duplicate",
        "anchors": torch.tensor(
            [[0.5, 0.5], [0.501, 0.501], [0.4, 0.4], [0.6, 0.4], [0.5, 0.65]]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.7, 0.7, 0.7]),
        "pairs": torch.tensor(
            [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3]]
        ),
        "probabilities": torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.01]),
        "chemical": torch.zeros(7, dtype=torch.long),
        "target_anchors": torch.tensor([[0.5, 0.5], [0.4, 0.4], [0.6, 0.4], [0.5, 0.65]]),
        "target_bonds": {(0, 1), (0, 2), (0, 3)},
    }


def test_graph_duplicate_suppression_keeps_higher_score_twin() -> None:
    sample = duplicate_sample()
    candidates = graph_duplicate_candidates(
        sample, presence_threshold=0.85, distance_px=2.0, minimum_degree=2
    )
    assert [(left, right) for _, left, right in candidates] == [(0, 1)]
    kept, changes = suppress_graph_duplicates(
        sample, presence_threshold=0.85, distance_px=2.0, minimum_degree=2
    )
    assert kept.tolist() == [True, False, True, True, True]
    assert changes[0]["retained"] == 0
    result = evaluate_sample(sample, kept, 0.85)
    assert result["atom_exact"]
    assert result["skeleton_exact"]


def test_graph_duplicate_suppression_preserves_close_distinct_neighborhoods() -> None:
    sample = duplicate_sample()
    sample["probabilities"][5] = 0.01
    kept, changes = suppress_graph_duplicates(
        sample, presence_threshold=0.85, distance_px=2.0, minimum_degree=2
    )
    assert kept.all()
    assert changes == []
