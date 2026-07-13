"""PyTorch data adapters for the ChemMol WebDataset package."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from Data.sdf.chemmol_webdataset import ChemMolWebDataset


class ChemVectorIterableDataset(IterableDataset):
    def __init__(
        self,
        shards: str | list[str],
        *,
        shardshuffle: bool = False,
        sample_shuffle_buffer: int = 0,
        sequence_version: str = "0.2",
        reject_keys: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.shards = shards
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer
        self.sequence_version = sequence_version
        self.reject_keys = reject_keys or set()

    def __iter__(self) -> Iterator[dict]:
        yield from ChemMolWebDataset(
            self.shards,
            validate_legacy_hash=False,
            shardshuffle=self.shardshuffle,
            sample_shuffle_buffer=self.sample_shuffle_buffer,
            sequence_version=self.sequence_version,
            reject_keys=self.reject_keys,
        )


class ChemVectorCollator:
    ELEMENTS = ("B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si")
    BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

    def __init__(
        self,
        tokenizer,
        *,
        image_size: int = 768,
        max_length: int = 3072,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5

    def _visual_targets(self, sample: dict) -> torch.Tensor:
        grid_size = self.image_size // 32
        channels = len(self.ELEMENTS) + len(self.BOND_TYPES)
        targets = np.zeros((grid_size, grid_size, channels), dtype=np.float32)
        render = sample["render"]
        chemmol = sample["chemmol"]
        render_width = render["width"]
        render_height = render["height"]
        element_channels = {element: index for index, element in enumerate(self.ELEMENTS)}
        bond_channels = {
            bond_type: len(self.ELEMENTS) + index
            for index, bond_type in enumerate(self.BOND_TYPES)
        }

        for position in render["atoms"]:
            atom = chemmol["atoms"][position["atom_id"]]
            x = min(grid_size - 1, int(position["image_x"] / render_width * grid_size))
            y = min(grid_size - 1, int(position["image_y"] / render_height * grid_size))
            targets[y, x, element_channels[atom["element"]]] = 1.0

        for bond in render["bonds"]:
            channel = bond_channels[bond["display_type"]]
            begin_x = bond["begin_image_x"] / render_width * grid_size
            begin_y = bond["begin_image_y"] / render_height * grid_size
            end_x = bond["end_image_x"] / render_width * grid_size
            end_y = bond["end_image_y"] / render_height * grid_size
            steps = max(2, int(max(abs(end_x - begin_x), abs(end_y - begin_y)) * 4) + 1)
            for x, y in zip(
                np.linspace(begin_x, end_x, steps),
                np.linspace(begin_y, end_y, steps),
            ):
                grid_x = min(grid_size - 1, max(0, int(x)))
                grid_y = min(grid_size - 1, max(0, int(y)))
                targets[grid_y, grid_x, channel] = 1.0
        return torch.from_numpy(targets.reshape(-1, channels))

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        encoded = self.tokenizer.encode_batch(
            (sample["target_sequence"] for sample in samples),
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "pixel_values": torch.stack([self._image_tensor(sample["image"]) for sample in samples]),
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "keys": [sample["key"] for sample in samples],
            "target_sequences": [sample["target_sequence"] for sample in samples],
            "expected_smiles": [
                sample["chemmol"]["canonical_isomeric_smiles"] for sample in samples
            ],
            "expected_inchikeys": [sample["chemmol"]["inchikey"] for sample in samples],
            "visual_targets": torch.stack(
                [self._visual_targets(sample) for sample in samples]
            ),
        }
