import torch

from chemvector.training.graph.correction import (
    train_g3_p19e8_output_route_train4k as p19e8,
)


def _rows(molecule, scaffold):
    return [
        {
            "sample_key": f"{molecule}_{variant}",
            "molecule_id": molecule,
            "scaffold": scaffold,
            "variant": variant,
        }
        for variant in p19e8.VARIANTS
    ]


def test_split_selection_excludes_and_isolates_scaffolds(monkeypatch):
    monkeypatch.setattr(p19e8, "TRAIN_MOLECULES", 1)
    monkeypatch.setattr(p19e8, "CALIBRATION_MOLECULES", 1)
    monkeypatch.setattr(p19e8, "HOLDOUT_MOLECULES", 1)
    monkeypatch.setattr(p19e8, "TRAIN_IMAGES", 4)
    monkeypatch.setattr(p19e8, "CALIBRATION_IMAGES", 4)
    monkeypatch.setattr(p19e8, "HOLDOUT_IMAGES", 4)
    excluded = _rows("x", "excluded")
    full = (
        excluded
        + _rows("a", "sa")
        + _rows("b", "sb")
        + _rows("c", "sc")
    )
    splits, audit = p19e8.select_router_splits(full, excluded)
    assert all(len(rows) == 4 for rows in splits.values())
    assert not any(audit["excluded_scaffold_overlap"].values())
    assert not any(audit["cross_split_scaffold_overlap"].values())


def test_action_threshold_defaults_low_confidence_to_keep():
    logits = torch.tensor([[2.0, 1.9, 0.0], [0.0, 2.0, 1.0]])
    actions = p19e8._actions_from_logits(logits, 0.7)
    assert actions.tolist() == [0, 0]
    actions = p19e8._actions_from_logits(logits, 0.6)
    assert actions.tolist() == [0, 1]


def test_tensor_record_round_trip():
    cache = {
        "labels": torch.tensor([1]),
        "route_exact": torch.tensor([[False, True, False]]),
        "observable_disagreement": torch.tensor([True]),
        "sample_keys": ["a"],
        "variants": ["stress"],
        "stress_operations": ["crossed_bonds"],
    }
    record = p19e8._tensor_records(cache)[0]
    assert record["target_action"] == 1
    assert record["routes_observably_disagree"]
    assert record["sample_key"] == "a"
