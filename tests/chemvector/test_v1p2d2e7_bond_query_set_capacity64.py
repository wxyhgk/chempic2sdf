from collections import Counter

import torch

from chemvector.training.bonds.objects.train_v1p2d2e7_bond_query_set_capacity64 import (
    _metrics,
    capacity_checks,
    query_selection_result,
)


def test_query_selection_counts_duplicate_bond_only_once() -> None:
    result = query_selection_result(
        torch.tensor((4.0, 3.0, 2.0)),
        torch.tensor(((5.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 5.0, 0.0))),
        torch.tensor((0, 1, -1)),
        selected_queries=torch.tensor((0, 1)),
        bond_count=2,
    )
    assert result == {"selected": 2, "correct": 2, "recovered": 1, "exact": 0}


def test_query_capacity_metrics_separate_count_and_exact() -> None:
    metrics = _metrics(
        Counter(
            samples=2,
            bonds=4,
            count_correct=2,
            known_selected=4,
            known_correct=4,
            known_recovered=4,
            known_exact=2,
            predicted_selected=4,
            predicted_correct=3,
            predicted_recovered=3,
            predicted_exact=1,
        )
    )
    assert metrics["predicted_count_accuracy"] == 1.0
    assert metrics["known_count_image_exact"] == 1.0
    assert metrics["predicted_image_exact"] == 0.5


def test_query_capacity_checks_require_all_variant_exact() -> None:
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
    metrics["predicted_count_accuracy"] = 0.99
    assert not all(capacity_checks(metrics).values())
