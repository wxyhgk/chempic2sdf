"""PyTorch data adapters for the ChemMol WebDataset package."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.ocr import OCRTextCodec, atom_display_text


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
        ocr_codec: OCRTextCodec | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.ocr_codec = ocr_codec

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

    def _ocr_batch(self, samples: list[dict]) -> dict[str, torch.Tensor | list[list[str]]]:
        if self.ocr_codec is None:
            return {}
        maximum_atoms = max(len(sample["render"]["atoms"]) for sample in samples)
        encoded_samples = []
        maximum_text_length = 2
        for sample in samples:
            positions = sorted(sample["render"]["atoms"], key=lambda item: item["atom_id"])
            encoded_atoms = []
            for position in positions:
                atom = sample["chemmol"]["atoms"][position["atom_id"]]
                text = atom_display_text(atom)
                encoded = self.ocr_codec.encode(text)
                maximum_text_length = max(maximum_text_length, len(encoded))
                encoded_atoms.append((position, text, encoded))
            encoded_samples.append(encoded_atoms)
        anchors = torch.zeros((len(samples), maximum_atoms, 2), dtype=torch.float32)
        anchor_mask = torch.zeros((len(samples), maximum_atoms), dtype=torch.bool)
        token_ids = torch.full(
            (len(samples), maximum_atoms, maximum_text_length),
            self.ocr_codec.PAD_TOKEN_ID,
            dtype=torch.long,
        )
        batch_texts: list[list[str]] = []
        for sample_index, (sample, encoded_atoms) in enumerate(
            zip(samples, encoded_samples)
        ):
            render = sample["render"]
            texts = []
            for anchor_index, (position, text, encoded) in enumerate(encoded_atoms):
                anchors[sample_index, anchor_index] = torch.tensor(
                    (
                        position["image_x"] / render["width"],
                        position["image_y"] / render["height"],
                    ),
                    dtype=torch.float32,
                )
                anchor_mask[sample_index, anchor_index] = True
                token_ids[sample_index, anchor_index, : len(encoded)] = torch.tensor(
                    encoded,
                    dtype=torch.long,
                )
                texts.append(text)
            batch_texts.append(texts)
        return {
            "ocr_anchor_points": anchors,
            "ocr_anchor_mask": anchor_mask,
            "ocr_token_ids": token_ids,
            "ocr_texts": batch_texts,
        }

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        encoded = self.tokenizer.encode_batch(
            (sample["target_sequence"] for sample in samples),
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch = {
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
        batch.update(self._ocr_batch(samples))
        return batch
