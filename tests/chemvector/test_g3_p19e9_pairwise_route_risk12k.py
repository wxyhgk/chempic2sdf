import torch

from chemvector.training.graph.correction import (
    train_g3_p19e9_pairwise_route_risk12k as p19e9,
)


def _rows(molecule, scaffold):
    return [
        {
            "sample_key": f"{molecule}_{variant}",
            "molecule_id": molecule,
            "scaffold": scaffold,
            "variant": variant,
        }
        for variant in p19e9.VARIANTS
    ]


def test_route_outcome_labels_cover_all_four_states():
    exact = torch.tensor(
        [
            [False, False, True],
            [True, True, False],
            [False, True, False],
            [True, False, True],
        ]
    )
    labels = p19e9.route_outcome_labels(exact)
    assert labels[:, 0].tolist() == [0, 1, 2, 3]
    assert labels[:, 1].tolist() == [2, 3, 0, 1]


def test_risk_action_requires_rescue_and_harmful_thresholds():
    probabilities = torch.tensor(
        [
            [[0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.70, 0.20]],
            [[0.05, 0.05, 0.75, 0.15], [0.05, 0.05, 0.40, 0.50]],
            [[0.05, 0.05, 0.65, 0.25], [0.05, 0.05, 0.80, 0.10]],
        ]
    )
    actions = p19e9.actions_from_risk_logits(
        probabilities.log(), rescue_threshold=0.7, harmful_threshold=0.1
    )
    assert actions.tolist() == [1, 0, 2]


def test_zero_initialized_risk_router_defaults_to_fusion():
    logits = torch.zeros(5, 2, 4)
    actions = p19e9.actions_from_risk_logits(logits, 0.5, 0.2)
    assert actions.tolist() == [0] * 5


def test_rule_key_prioritizes_zero_harmful_over_rescue():
    safe = {
        "overall": {
            "images": 20,
            "fusion_exact": 10,
            "selected_exact": 11,
            "rescue": 1,
            "harmful": 0,
            "actions": {"KEEP_FUSION": 19, "USE_NEW": 1},
        }
    }
    unsafe = {
        "overall": {
            "images": 20,
            "fusion_exact": 10,
            "selected_exact": 19,
            "rescue": 10,
            "harmful": 1,
            "actions": {"KEEP_FUSION": 9, "USE_NEW": 11},
        }
    }
    assert p19e9.rule_key(safe, 0.9, 0.01) > p19e9.rule_key(
        unsafe, 0.9, 0.01
    )


def test_split_selection_excludes_and_isolates_scaffolds(monkeypatch):
    monkeypatch.setattr(p19e9, "TRAIN_MOLECULES", 1)
    monkeypatch.setattr(p19e9, "CALIBRATION_MOLECULES", 1)
    monkeypatch.setattr(p19e9, "HOLDOUT_MOLECULES", 1)
    monkeypatch.setattr(p19e9, "TRAIN_IMAGES", 4)
    monkeypatch.setattr(p19e9, "CALIBRATION_IMAGES", 4)
    monkeypatch.setattr(p19e9, "HOLDOUT_IMAGES", 4)
    excluded = _rows("x", "excluded")
    full = excluded + _rows("a", "sa") + _rows("b", "sb") + _rows("c", "sc")
    splits, audit = p19e9.select_risk_splits(full, excluded)
    assert all(len(rows) == 4 for rows in splits.values())
    assert not any(audit["excluded_scaffold_overlap"].values())
    assert not any(audit["cross_split_scaffold_overlap"].values())
