import torch

from chemvector.infer_direct_atom_pair_bond import direct_scene_tensors


def test_direct_scene_tensors_encode_pair_incidence_without_distance_filter() -> None:
    atoms = {
        "atom_mask": torch.tensor([[True, True, False, True]]),
        "atom_scores": torch.tensor([[0.9, 0.8, -1.0, 0.7]]),
    }
    pairs = torch.tensor([[0, 1], [0, 3], [1, 3]])
    output = {
        "presence_logits": torch.tensor([2.0, -1.0, 3.0]),
        "chemical_logits": torch.tensor(
            [[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0], [0.0, 1.0, 4.0, 0.0]]
        ),
    }

    result = direct_scene_tensors(atoms, pairs, output)

    assert result["bond_mask"].tolist() == [True, True, True]
    assert result["atom_presence_logits"].tolist() == [1e4, 1e4, -1e4, 1e4]
    assert result["incidence_logits"].argmax(-1).tolist() == [[0, 1], [0, 3], [1, 3]]
    assert result["bond_types"].shape == (3, 8)
    assert torch.allclose(result["bond_types"][:, :4], result["bond_types"][:, 4:])


def test_direct_scene_tensors_only_overrides_changed_chemical_predictions() -> None:
    atoms = {
        "atom_mask": torch.tensor([[True, True, True]]),
        "atom_scores": torch.tensor([[0.9, 0.8, 0.7]]),
    }
    pairs = torch.tensor([[0, 1], [1, 2]])
    output = {
        "presence_logits": torch.tensor([2.0, 2.0]),
        "chemical_logits": torch.tensor([[4.0, 1.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]]),
    }

    result = direct_scene_tensors(atoms, pairs, output, torch.tensor([0, 3]))

    assert result["bond_types"][0, :4].argmax().item() == 0
    assert result["bond_types"][1, :4].tolist() == [0.0, 0.0, 0.0, 1.0]
