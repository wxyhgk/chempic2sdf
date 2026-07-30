import torch

from chemvector.duplicate_atom_affinity import (
    DuplicateAtomAffinity,
    DuplicateAtomAffinityConfig,
    duplicate_atom_affinity_loss,
    enumerate_close_atom_pairs,
)
from chemvector.train_duplicate_atom_affinity import select_threshold


def test_close_pair_enumeration_has_no_neighbor_limit() -> None:
    mask = torch.tensor([[True, True, True]])
    anchors = torch.tensor([[[0.5, 0.5], [0.501, 0.501], [0.8, 0.8]]])
    pairs, samples = enumerate_close_atom_pairs(
        mask, anchors, maximum_distance_px=2.0
    )
    assert pairs.tolist() == [[0, 1]]
    assert samples.tolist() == [0]


def test_duplicate_affinity_is_symmetric_and_backpropagates() -> None:
    config = DuplicateAtomAffinityConfig(atom_feature_dim=8, hidden_dim=16)
    model = DuplicateAtomAffinity(config)
    features = torch.randn(1, 2, 8)
    geometry = torch.randn(1, 2, 8)
    context = torch.randn(1, 10)
    forward = model(features, geometry, torch.tensor([[0, 1]]), torch.tensor([0]), context)
    reverse = model(features, geometry, torch.tensor([[1, 0]]), torch.tensor([0]), context)
    assert torch.allclose(forward, reverse)
    loss = duplicate_atom_affinity_loss(forward, torch.tensor([True]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_threshold_selection_prefers_safe_candidate_over_more_aggressive_merge() -> None:
    true_duplicate = {
        "key": "true",
        "anchors": torch.tensor([[0.5, 0.5], [0.501, 0.501], [0.6, 0.5]]),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "pairs": torch.tensor([[0, 2], [1, 2]]),
        "probabilities": torch.tensor([0.99, 0.99]),
        "chemical": torch.zeros(2, dtype=torch.long),
        "target_anchors": torch.tensor([[0.5, 0.5], [0.6, 0.5]]),
        "target_bonds": {(0, 1)},
        "duplicate_pairs": torch.tensor([[0, 1]]),
        "duplicate_probabilities": torch.tensor([0.9998]),
        "duplicate_targets": torch.tensor([True]),
    }
    protected_close = {
        "key": "protected",
        "anchors": torch.tensor([[0.5, 0.5], [0.51, 0.5]]),
        "scores": torch.tensor([0.9, 0.8]),
        "pairs": torch.tensor([[0, 1]]),
        "probabilities": torch.tensor([0.01]),
        "chemical": torch.zeros(1, dtype=torch.long),
        "target_anchors": torch.tensor([[0.5, 0.5], [0.51, 0.5]]),
        "target_bonds": set(),
        "duplicate_pairs": torch.tensor([[0, 1]]),
        "duplicate_probabilities": torch.tensor([0.99]),
        "duplicate_targets": torch.tensor([False]),
    }
    selected, _ = select_threshold([true_duplicate, protected_close], 0.85)
    assert selected["safety_eligible"]
    assert selected["threshold"] == 0.9995
    assert selected["protected_close_merges"] == 0
    assert selected["atom_break"] == 0
