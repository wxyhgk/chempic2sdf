"""WebDataset adapter for raster-to-Visual-SVG training."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
import webdataset as wds


class VisualSVGWebDataset(IterableDataset):
    def __init__(
        self,
        shards: str | list[str],
        *,
        shardshuffle: bool = False,
        sample_shuffle_buffer: int = 0,
        validate_hashes: bool = True,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.shards = shards
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer
        self.validate_hashes = validate_hashes
        self.limit = limit

    def __iter__(self):
        dataset = wds.WebDataset(
            self.shards,
            shardshuffle=100 if self.shardshuffle else False,
            empty_check=False,
        )
        if self.sample_shuffle_buffer:
            dataset = dataset.shuffle(self.sample_shuffle_buffer)
        for sample_index, sample in enumerate(dataset, 1):
            if self.limit is not None and sample_index > self.limit:
                break
            sequence_bytes = sample["visualsvg.txt"]
            primitives_bytes = sample["primitives.json"]
            image_bytes = sample["png"]
            meta = json.loads(sample["meta.json"])
            if self.validate_hashes:
                expected = meta["sha256"]
                actual = {
                    "png": hashlib.sha256(image_bytes).hexdigest(),
                    "primitives.json": hashlib.sha256(primitives_bytes).hexdigest(),
                    "visualsvg.txt": hashlib.sha256(sequence_bytes).hexdigest(),
                }
                for name, digest in actual.items():
                    if digest != expected[name]:
                        raise ValueError(f"Hash mismatch for {sample['__key__']}.{name}")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.load()
            bond_layer = Image.open(io.BytesIO(sample["bond.png"])).convert("L")
            bond_layer.load()
            yield {
                "key": sample["__key__"],
                "image": image,
                "bond_layer": bond_layer,
                "target_sequence": sequence_bytes.decode("utf-8"),
                "primitives": json.loads(primitives_bytes),
                "meta": meta,
            }


class VisualSVGCollator:
    PRIMITIVE_COUNT_ORDER = ("line", "triangle", "quad", "text_box")

    def __init__(
        self,
        tokenizer,
        *,
        image_size: int = 768,
        max_length: int = 2048,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5

    def __call__(self, samples: list[dict]) -> dict:
        encoded = self.tokenizer.encode_batch(
            (sample["target_sequence"] for sample in samples),
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoded["attention_mask"], dtype=torch.long
            ),
            "keys": [sample["key"] for sample in samples],
            "target_sequences": [sample["target_sequence"] for sample in samples],
            "primitives": [sample["primitives"] for sample in samples],
            "primitive_counts": torch.tensor(
                [
                    [
                        sample["meta"].get("primitive_counts", {}).get(kind, 0)
                        for kind in self.PRIMITIVE_COUNT_ORDER
                    ]
                    for sample in samples
                ],
                dtype=torch.float32,
            ),
            "bond_layers": [sample["bond_layer"] for sample in samples],
            "meta": [sample["meta"] for sample in samples],
        }
