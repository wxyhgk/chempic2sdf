from collections import Counter

import torch

from chemvector.training.bonds.objects.train_v1p2d2e6_candidate_graph_capacity64 import (
    CACHE_SHA256,
    _metrics,
    _selection_result,
    capacity_checks,
)


def test_capacity_cache_hash_is_frozen() -> None:
    assert CACHE_SHA256 == "6d80911067a60dc91ad4cf5f7716e7549122bdef8d2f46015a562ae896b2bcd8"


def test_capacity_selection_requires_one_correct_edge_per_bond() -> None:
    result = _selection_result(
        torch.tensor((4.0, 3.0, 2.0, 1.0)),
        torch.tensor((0, 0, 1, -1)),
        selected_count=2,
        bond_count=2,
    )
    assert result == {"selected": 2, "correct": 2, "recovered": 1, "exact": 0}


def test_capacity_metrics_keep_count_and_set_exact_separate() -> None:
    metrics = _metrics(
        Counter(
            samples=2,
            bonds=4,
            nodes=10,
            count_correct=2,
            known_selected=4,
            known_correct=4,
            known_recovered=4,
            known_exact=2,
            predicted_selected=4,
            predicted_correct=3,
            predicted_recovered=3,
            predicted_exact=1,
            node_keep_correct=9,
            node_keep_exact=1,
            node_degree_correct=8,
        )
    )
    assert metrics["predicted_count_accuracy"] == 1.0
    assert metrics["known_count_image_exact"] == 1.0
    assert metrics["predicted_image_exact"] == 0.5


def test_capacity_checks_require_all_variants_and_predicted_exact() -> None:
    metrics = {
        "samples": 64,
        "known_count_image_exact": 1.0,
        "predicted_count_accuracy": 1.0,
        "predicted_image_exact": 1.0,
        "by_variant": {
            name: {"predicted_image_exact": 1.0}
            for name in ("coordgen", "original", "rdkit", "stress")
        },
    }
    assert all(capacity_checks(metrics).values())
    metrics["by_variant"]["stress"]["predicted_image_exact"] = 0.99
    assert not all(capacity_checks(metrics).values())
