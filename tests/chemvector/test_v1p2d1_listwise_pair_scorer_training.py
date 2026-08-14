import torch

from chemvector.training.bonds.objects.train_v1p2d1_listwise_pair_scorer import (
    split_pair_scorer_rows,
    summarize_records,
)
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    load_safe_manifest,
)


def test_pair_scorer_split_keeps_p2b_holdout_disjoint() -> None:
    fit, calibration, holdout = split_pair_scorer_rows(load_safe_manifest())
    assert (len(fit), len(calibration), len(holdout)) == (7_656, 400, 400)
    molecule_sets = [
        {str(row["molecule_id"]) for row in rows}
        for rows in (fit, calibration, holdout)
    ]
    assert [len(values) for values in molecule_sets] == [1_914, 100, 100]
    assert not molecule_sets[0] & molecule_sets[1]
    assert not molecule_sets[0] & molecule_sets[2]
    assert not molecule_sets[1] & molecule_sets[2]


def test_pair_scorer_metrics_count_unique_recovered_bonds() -> None:
    records = [
        {
            "key": "sample",
            "variant": "stress",
            "stress_operation": "crossed_bonds",
            "scores": torch.tensor((0.9, 0.8, 0.7, 0.1)),
            "target_bond_ids": torch.tensor((0, 0, 1, -1)),
            "bond_count": 2,
            "candidate_count": 8,
            "deduplicated_count": 4,
        }
    ]
    metrics = summarize_records(records, threshold=0.5)
    assert metrics["known_pair_recall"] == 0.5
    assert metrics["known_image_exact"] == 0.0
    assert metrics["predicted_pair_recall"] == 1.0
    assert metrics["predicted_pair_precision"] == 2 / 3
    assert metrics["predicted_count_accuracy"] == 0.0
    assert metrics["by_stress_operation"]["crossed_bonds"][
        "predicted_pair_recall"
    ] == 1.0
