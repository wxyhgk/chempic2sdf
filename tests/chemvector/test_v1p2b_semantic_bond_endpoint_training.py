import torch

from chemvector.training.bonds.objects.train_v1p2b_semantic_bond_endpoint import (
    _unique_port_data,
    split_train_endpoint_rows,
    summarize_predictions,
)
from chemvector.training.vision.train_v1p1_semantic_ink_separator import (
    load_safe_manifest,
)


def test_endpoint_split_is_molecule_disjoint_and_keeps_validation_unread() -> None:
    fit, calibration, holdout = split_train_endpoint_rows(load_safe_manifest())
    assert (len(fit), len(calibration), len(holdout)) == (8_056, 400, 400)
    groups = [
        {str(row["molecule_id"]) for row in rows}
        for rows in (fit, calibration, holdout)
    ]
    assert [len(group) for group in groups] == [2_014, 100, 100]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert all(row["split"] == "train" for rows in (fit, calibration, holdout) for row in rows)


def test_unique_port_data_preserves_bond_to_shared_port_mapping() -> None:
    endpoints = torch.tensor(
        [
            [[0.1, 0.2], [0.5, 0.5]],
            [[0.5, 0.5], [0.8, 0.7]],
        ]
    )
    unique, indices = _unique_port_data(endpoints)
    assert len(unique) == 3
    assert indices.shape == (2, 2)
    assert int(indices[0, 1]) == int(indices[1, 0])


def test_endpoint_metrics_distinguish_port_recall_and_pairable_bonds() -> None:
    record = {
        "key": "sample",
        "variant": "stress",
        "stress_operation": "crossed_bonds",
        "objects": [
            {"score": 0.9, "point": torch.tensor((0.1, 0.2))},
            {"score": 0.8, "point": torch.tensor((0.5, 0.5))},
        ],
        "points": torch.tensor(((0.1, 0.2), (0.5, 0.5), (0.8, 0.7))),
        "port_indices": torch.tensor(((0, 1), (1, 2))),
        "text_adjacent": torch.tensor((False, True)),
    }
    metrics = summarize_predictions([record], threshold=0.5)
    assert metrics["known_unique_port_recall"] == 2 / 3
    assert metrics["known_pairable_bond_recall"] == 0.5
    assert metrics["text_adjacent_pairable_bond_recall"] == 0.0
    assert metrics["implicit_pairable_bond_recall"] == 1.0
    assert metrics["predicted_count_accuracy"] == 0.0
