import copy

import torch

from chemvector.safe_atom_action_model import (
    ACTION_FEATURE_DIM,
    SafeAtomActionModel,
    delete_action_features,
    safe_atom_action_loss,
)
from chemvector.train_safe_atom_action import _summarize_threshold, utility_class


def _sample() -> dict:
    return {
        "key": "sample",
        "anchors": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]]) / 768.0,
        "atom_features": torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128) / 512.0,
        "atom_geometry": torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) / 32.0,
        "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
        "pairs": torch.tensor(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long
        ),
        "probabilities": torch.tensor([0.01, 0.01, 0.99, 0.01, 0.99, 0.99]),
        "chemical": torch.ones(6, dtype=torch.long),
        "target_anchors": torch.zeros((2, 2)),
        "target_bonds": {(0, 1): 1},
        "action_labels": {"winner_removed_nodes": [1]},
    }


def test_action_features_are_unique_per_removed_node_and_target_free() -> None:
    sample = _sample()
    features, removed = delete_action_features(sample, presence_threshold=0.5)
    assert features.shape == (3, ACTION_FEATURE_DIM)
    assert removed.tolist() == [0, 1, 2]
    corrupted = copy.deepcopy(sample)
    corrupted["target_anchors"] = torch.randn(100, 2)
    corrupted["target_bonds"] = {(40, 90): 4}
    corrupted["action_labels"] = {"winner_removed_nodes": [0, 1, 2]}
    other, other_removed = delete_action_features(corrupted, presence_threshold=0.5)
    assert torch.equal(removed, other_removed)
    assert torch.equal(features, other)


def test_pair_order_does_not_change_directional_delete_features() -> None:
    sample = _sample()
    expected, removed = delete_action_features(sample, presence_threshold=0.5)
    sample["pairs"] = sample["pairs"].flip((0, 1))
    sample["probabilities"] = sample["probabilities"].flip(0)
    sample["chemical"] = sample["chemical"].flip(0)
    actual, actual_removed = delete_action_features(sample, presence_threshold=0.5)
    assert torch.equal(removed, actual_removed)
    assert torch.equal(expected, actual)


def test_action_feature_distance_matches_float64_candidate_enumeration() -> None:
    anchors = torch.tensor(
        [
            [0.2768707275390625, 0.5091146230697632],
            [0.2587483823299408, 0.5193888545036316],
            *[[2.0 + index, 2.0] for index in range(28)],
        ],
        dtype=torch.float32,
    )
    pairs = torch.tensor(
        [[left, right] for left in range(30) for right in range(left + 1, 30)],
        dtype=torch.long,
    )
    sample = {
        "anchors": anchors,
        "atom_features": torch.zeros((30, 128)),
        "atom_geometry": torch.zeros((30, 8)),
        "scores": torch.full((30,), 0.5),
        "pairs": pairs,
        "probabilities": torch.full((len(pairs),), 0.01),
        "chemical": torch.zeros(len(pairs), dtype=torch.long),
    }
    float_distance = float(
        torch.cdist(sample["anchors"], sample["anchors"])[0, 1] * 768.0
    )
    double_distance = float(
        torch.cdist(sample["anchors"].double(), sample["anchors"].double())[
            0, 1
        ]
        * 768.0
    )
    assert float_distance > 16.0
    assert double_distance < 16.0
    features, removed = delete_action_features(sample, presence_threshold=0.5)
    assert features.shape == (2, ACTION_FEATURE_DIM)
    assert features.dtype == torch.float32
    assert removed.tolist() == [0, 1]


def test_small_action_model_backpropagates_all_losses() -> None:
    model = SafeAtomActionModel()
    assert model.parameter_count() < 100_000
    output = model(torch.randn(4, ACTION_FEATURE_DIM))
    losses = safe_atom_action_loss(
        output,
        torch.tensor([True, False, True, True]),
        torch.tensor([1, 0, 3, 0]),
        torch.tensor([True, False, False, False]),
        torch.tensor([0, 0, 1, 1]),
    )
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(value) for value in losses.values())


def test_utility_class_and_cached_validation_use_strict_directional_action() -> None:
    baseline = (0, 0, -1, -1)
    assert utility_class(baseline, {"safe": True, "utility": (0, 0, 0, -1)}) == 1
    assert utility_class(baseline, {"safe": True, "utility": (0, 1, 0, 0)}) == 2
    assert utility_class(baseline, {"safe": True, "utility": (1, 1, 0, 0)}) == 3
    assert utility_class(baseline, {"safe": False, "utility": (1, 1, 0, 0)}) == 0

    sample = _sample()
    sample["action_labels"] = {
        "baseline_utility": baseline,
        "winner_removed_nodes": [0],
        "actions": [
            {"removed": 0, "safe": True, "strict_gain": True, "utility": (1, 1, 0, 0)},
            {"removed": 1, "safe": False, "strict_gain": False, "utility": baseline},
            {"removed": 2, "safe": False, "strict_gain": False, "utility": baseline},
        ],
    }
    output = {
        "sample_indices": torch.tensor([0, 0, 0]),
        "removed_nodes": torch.tensor([0, 1, 2]),
        "safe_targets": torch.tensor([True, False, False]),
        "winner_targets": torch.tensor([True, False, False]),
        "safety_logits": torch.tensor([20.0, -20.0, -20.0]),
        "utility_logits": torch.tensor(
            [[-10.0, -10.0, -10.0, 10.0], [10.0, -10.0, -10.0, -10.0], [10.0, -10.0, -10.0, -10.0]]
        ),
        "rank_logits": torch.tensor([10.0, -10.0, -10.0]),
    }
    metrics = _summarize_threshold(
        [sample],
        output,
        threshold=0.999,
        presence_threshold=0.85,
        forced_failures=0,
    )
    assert metrics["executed_actions"] == 1
    assert metrics["harmful_actions"] == 0
    assert metrics["atom_exact_delta"] == 1.0
    assert metrics["skeleton_exact_delta"] == 1.0

    output["rank_logits"] = torch.full((3,), -20.0)
    no_action = _summarize_threshold(
        [sample],
        output,
        threshold=0.999,
        presence_threshold=0.85,
        forced_failures=0,
    )
    assert no_action["executed_actions"] == 0
    assert no_action["harmful_actions"] == 0
    assert no_action["atom_break"] == 0
    assert no_action["skeleton_break"] == 0
