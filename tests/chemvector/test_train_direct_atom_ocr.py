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
    crops, labels = training_crops(
        [sample], atoms, mapping, torch.tensor([0, 1]), torch.tensor([0, 0]),
        torch.tensor([0, 1]), 1.5
    )
    assert len(crops) == 2
    assert labels == ["N", ""]
