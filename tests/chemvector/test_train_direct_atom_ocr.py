import torch
from PIL import Image

from chemvector.train_direct_atom_ocr import training_crops


def test_training_crops_assign_blank_to_unmatched_text_proposal() -> None:
    sample = {
        "image": Image.new("RGB", (768, 768), "white"),
        "layers": {"atoms": [{"atom_id": 0, "visible_text": True, "text": "N"}]},
        "chemmol": {"atoms": []},
    }
    atoms = {
        "atom_geometry": torch.tensor(
            [[[0.5, 0.5, 0, 0, 0, 0, 0.02, 0.02], [0.2, 0.2, 0, 0, 0, 0, 0.02, 0.02]]]
        )
    }
    mapping = torch.tensor([[0, -1]])
    scenes = [{"atom_objects": [{"text": {"value": "N"}}]}]
    crops, labels = training_crops(
        [sample], scenes, atoms, mapping, torch.tensor([0, 1]), torch.tensor([0, 0]),
        torch.tensor([0, 1]), 1.5
    )
    assert len(crops) == 2
    assert labels == ["N", ""]


def test_training_crops_use_chemscene_order_after_atom_reordering() -> None:
    sample = {
        "image": Image.new("RGB", (768, 768), "white"),
        "layers": {
            "atoms": [
                {"atom_id": 0, "visible_text": True, "text": "N"},
                {"atom_id": 1, "visible_text": True, "text": "O"},
            ]
        },
        "chemmol": {"atoms": []},
    }
    scenes = [
        {
            "atom_objects": [
                {"text": {"value": "O"}},
                {"text": {"value": "N"}},
            ]
        }
    ]
    atoms = {
        "atom_geometry": torch.tensor(
            [[[0.2, 0.2, 0, 0, 0, 0, 0.02, 0.02], [0.8, 0.8, 0, 0, 0, 0, 0.02, 0.02]]]
        )
    }
    crops, labels = training_crops(
        [sample], scenes, atoms, torch.tensor([[0, 1]]), torch.tensor([0, 1]),
        torch.tensor([0, 0]), torch.tensor([0, 1]), 1.5
    )
    assert len(crops) == 2
    assert labels == ["O", "N"]


def test_training_crops_accept_learned_bbox_geometry() -> None:
    sample = {"image": Image.new("RGB", (768, 768), "white")}
    scenes = [{"atom_objects": [{"text": {"value": "Cl"}}]}]
    atoms = {"atom_geometry": torch.zeros(1, 1, 8)}
    learned = {
        "centers": torch.tensor([[0.5, 0.5]]),
        "half_extents": torch.tensor([[20.0 / 768.0, 10.0 / 768.0]]),
    }
    crops, labels = training_crops(
        [sample], scenes, atoms, torch.tensor([[0]]), torch.tensor([0]),
        torch.tensor([0]), torch.tensor([0]), 1.0, learned
    )
    assert len(crops) == 1
    assert crops[0].shape == (3, 48, 192)
    assert labels == ["Cl"]
