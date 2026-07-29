import torch

from chemvector.direct_atom_role import (
    DirectAtomRoleConfig,
    DirectAtomRoleGate,
    enumerate_selected_atoms,
)


def test_enumerates_selected_atoms_without_geometry_filter() -> None:
    atoms, samples = enumerate_selected_atoms(
        torch.tensor([[True, False, True], [False, True, True]])
    )
    assert atoms.tolist() == [0, 2, 1, 2]
    assert samples.tolist() == [0, 0, 1, 1]


def test_role_gate_preserves_requested_atom_order_and_gradients() -> None:
    torch.manual_seed(3)
    model = DirectAtomRoleGate(
        DirectAtomRoleConfig(radii_pixels=(2.0, 4.0), patch_size=8, image_size=16, hidden_dim=8)
    )
    logits = model(
        torch.randn(2, 3, 16, 16),
        torch.tensor([[[0.2, 0.2], [0.8, 0.8]], [[0.3, 0.7], [0.7, 0.3]]]),
        torch.tensor([1, 0, 1]),
        torch.tensor([1, 0, 1]),
    )
    assert logits.shape == (3,)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
