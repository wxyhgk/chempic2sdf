import torch

from chemvector.duplicate_atom_affinity import (
    DuplicateAtomAffinity,
    DuplicateAtomAffinityConfig,
    duplicate_atom_affinity_loss,
    enumerate_close_atom_pairs,
)


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
