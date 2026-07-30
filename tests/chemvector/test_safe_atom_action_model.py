import copy

import torch

from chemvector.safe_atom_action_model import (
    ACTION_FEATURE_DIM,
    SafeAtomActionModel,
    delete_action_features,
    safe_atom_action_loss,
)


def _sample() -> dict:
    return {
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
