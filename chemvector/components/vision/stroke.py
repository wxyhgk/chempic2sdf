"""Pair supervision and line-feature scoring for visible chemical strokes."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from PIL import Image


STROKE_CLASSES = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
STROKE_CLASS_TO_ID = {name: index for index, name in enumerate(STROKE_CLASSES)}


def all_pair_indices(atom_count: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(atom_count)
        for right in range(left + 1, atom_count)
    ]


class StrokePairCollator:
    def __init__(
        self,
        *,
        image_size: int = 768,
        negative_ratio: int = 4,
        anchor_jitter_pixels: float = 3.0,
        seed: int = 20260714,
        include_all_pairs: bool = False,
    ) -> None:
        if negative_ratio < 1:
            raise ValueError("negative_ratio must be positive")
        if anchor_jitter_pixels < 0:
            raise ValueError("anchor_jitter_pixels must be nonnegative")
        self.image_size = image_size
        self.negative_ratio = negative_ratio
        self.anchor_jitter_pixels = anchor_jitter_pixels
        self.seed = seed
        self.include_all_pairs = include_all_pairs

    def _rng(self, key: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5

    def _sample_pairs(self, sample: dict) -> tuple[list[tuple[int, int]], list[int]]:
        atom_count = len(sample["render"]["atoms"])
        display_types = {
            bond["bond_id"]: bond["display_type"] for bond in sample["render"]["bonds"]
        }
        positives: dict[tuple[int, int], int] = {}
        adjacency = [set() for _ in range(atom_count)]
        for bond in sample["chemmol"]["bonds"]:
            pair = tuple(sorted((bond["begin_atom"], bond["end_atom"])))
            positives[pair] = STROKE_CLASS_TO_ID[display_types[bond["id"]]]
            adjacency[pair[0]].add(pair[1])
            adjacency[pair[1]].add(pair[0])

        all_negatives = [pair for pair in all_pair_indices(atom_count) if pair not in positives]
        if self.include_all_pairs:
            pairs = all_pair_indices(atom_count)
            return pairs, [
                positives.get(pair, STROKE_CLASS_TO_ID["NONE"]) for pair in pairs
            ]
        target_negatives = min(
            len(all_negatives),
            max(len(positives) * self.negative_ratio, 16),
        )
        two_hop = set()
        for center, neighbors in enumerate(adjacency):
            ordered = sorted(neighbors)
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    pair = tuple(sorted((left, right)))
                    if pair not in positives:
                        two_hop.add(pair)

        positions = {
            item["atom_id"]: (item["image_x"], item["image_y"])
            for item in sample["render"]["atoms"]
        }
        near = sorted(
            all_negatives,
            key=lambda pair: (
                (positions[pair[0]][0] - positions[pair[1]][0]) ** 2
                + (positions[pair[0]][1] - positions[pair[1]][1]) ** 2
            ),
        )
        rng = self._rng(sample["key"])
        random_negatives = list(all_negatives)
        rng.shuffle(random_negatives)

        selected: list[tuple[int, int]] = []
        selected_set = set()

        def add(candidates, limit: int) -> None:
            for pair in candidates:
                if len(selected) >= limit:
                    return
                if pair not in selected_set:
                    selected.append(pair)
                    selected_set.add(pair)

        first_limit = target_negatives // 3
        add(sorted(two_hop), first_limit)
        add(near, first_limit * 2)
        add(random_negatives, target_negatives)

        pairs = [*positives, *selected]
        labels = [positives.get(pair, STROKE_CLASS_TO_ID["NONE"]) for pair in pairs]
        order = list(range(len(pairs)))
        rng.shuffle(order)
        return [pairs[index] for index in order], [labels[index] for index in order]

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        maximum_atoms = max(len(sample["render"]["atoms"]) for sample in samples)
        sampled = [self._sample_pairs(sample) for sample in samples]
        maximum_pairs = max(len(pairs) for pairs, _ in sampled)
        anchors = torch.zeros((len(samples), maximum_atoms, 2), dtype=torch.float32)
        anchor_mask = torch.zeros((len(samples), maximum_atoms), dtype=torch.bool)
        pair_indices = torch.zeros((len(samples), maximum_pairs, 2), dtype=torch.long)
        pair_mask = torch.zeros((len(samples), maximum_pairs), dtype=torch.bool)
        pair_labels = torch.full((len(samples), maximum_pairs), -100, dtype=torch.long)

        for sample_index, (sample, (pairs, labels)) in enumerate(zip(samples, sampled)):
            render = sample["render"]
            rng = self._rng(sample["key"] + ":jitter")
            for position in render["atoms"]:
                atom_id = position["atom_id"]
                jitter_x = rng.uniform(-self.anchor_jitter_pixels, self.anchor_jitter_pixels)
                jitter_y = rng.uniform(-self.anchor_jitter_pixels, self.anchor_jitter_pixels)
                anchors[sample_index, atom_id] = torch.tensor(
                    (
                        min(1.0, max(0.0, (position["image_x"] + jitter_x) / render["width"])),
                        min(1.0, max(0.0, (position["image_y"] + jitter_y) / render["height"])),
                    )
                )
                anchor_mask[sample_index, atom_id] = True
            pair_indices[sample_index, : len(pairs)] = torch.tensor(pairs)
            pair_mask[sample_index, : len(pairs)] = True
            pair_labels[sample_index, : len(labels)] = torch.tensor(labels)

        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "anchor_points": anchors,
            "anchor_mask": anchor_mask,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "pair_labels": pair_labels,
            "keys": [sample["key"] for sample in samples],
        }
