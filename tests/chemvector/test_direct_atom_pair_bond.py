import torch

from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    enumerate_unordered_atom_pairs,
)
from chemvector.train_direct_atom_pair_bond import build_pair_targets


def tiny_config() -> DirectAtomPairBondConfig:
    return DirectAtomPairBondConfig(
        feature_channels=4,
        atom_feature_dim=3,
        longitudinal_samples=8,
        transverse_samples=3,
        ribbon_radius_pixels=3.0,
        image_size=32.0,
        hidden_dim=16,
        pair_chunk_size=2,
    )


def test_enumerates_every_unordered_pair_without_distance_filter() -> None:
    mask = torch.tensor([[True, True, False, True], [True, False, True, False]])

    pairs, samples = enumerate_unordered_atom_pairs(mask)

    assert pairs.tolist() == [[0, 1], [0, 3], [1, 3], [0, 2]]
    assert samples.tolist() == [0, 0, 0, 1]


def test_pair_prediction_is_invariant_to_endpoint_order() -> None:
    torch.manual_seed(7)
    model = DirectAtomPairBond(tiny_config()).eval()
    feature_maps = torch.randn(1, 4, 8, 8)
    atom_features = torch.randn(1, 2, 3)
    anchors = torch.tensor([[[0.12, 0.23], [0.86, 0.71]]])
    scores = torch.tensor([[0.8, 0.6]])
    samples = torch.tensor([0])

    forward = model(
        feature_maps,
        atom_features,
        anchors,
        scores,
        torch.tensor([[0, 1]]),
        samples,
    )
    reverse = model(
        feature_maps,
        atom_features,
        anchors,
        scores,
        torch.tensor([[1, 0]]),
        samples,
    )

    assert torch.allclose(forward["presence_logits"], reverse["presence_logits"], atol=1e-6)
    assert torch.allclose(forward["chemical_logits"], reverse["chemical_logits"], atol=1e-6)


def test_build_pair_targets_labels_bonds_and_two_hop_hard_negatives() -> None:
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
    samples = torch.zeros(3, dtype=torch.long)
    mapping = torch.tensor([[0, 1, 2]])
    targets = {
        "atom_mask": torch.tensor([[True, True, True]]),
        "bond_mask": torch.tensor([[True, True]]),
        "bond_atom_refs": torch.tensor([[[0, 1], [1, 2]]]),
        "bond_chemical": torch.tensor([[1, 2]]),
    }
    atoms = {"atom_mask": torch.tensor([[True, True, True]])}

    result = build_pair_targets(pairs, samples, mapping, targets, atoms)

    assert result["presence"].tolist() == [True, False, True]
    assert result["chemical"].tolist() == [1, 0, 2]
    assert result["structured_negative"].tolist() == [False, True, False]
    assert result["matched_atoms"].tolist() == [3]


def test_ribbon_encoder_receives_gradients_through_chunk_reassembly() -> None:
    torch.manual_seed(11)
    model = DirectAtomPairBond(tiny_config())
    output = model(
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 3, 3),
        torch.tensor([[[0.1, 0.1], [0.5, 0.8], [0.9, 0.2]]]),
        torch.tensor([[0.9, 0.8, 0.7]]),
        torch.tensor([[0, 1], [0, 2], [1, 2]]),
        torch.zeros(3, dtype=torch.long),
    )

    output["presence_logits"].sum().backward()

    gradients = [parameter.grad for parameter in model.ribbon_encoder.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
