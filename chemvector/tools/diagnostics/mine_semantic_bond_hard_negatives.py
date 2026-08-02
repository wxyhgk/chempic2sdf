#!/usr/bin/env python3
"""Mine high-scoring visual non-bond relations without using pair length."""

from __future__ import annotations

import argparse
from itertools import combinations, islice
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate_semantic_bond_existence import file_sha256
from chemvector.semantic_bond_relation import (
    SemanticBondExistence,
    SemanticBondRelationConfig,
)
from chemvector.train import expand_shards


def image_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(values).permute(2, 0, 1) - 0.5) / 0.5


@torch.inference_mode()
def score_pairs(
    model: SemanticBondExistence,
    pixels: torch.Tensor,
    points: torch.Tensor,
    pairs: list[tuple[int, int]],
    device: torch.device,
    *,
    chunk_size: int,
) -> torch.Tensor:
    scores = []
    for start in range(0, len(pairs), chunk_size):
        current = pairs[start : start + chunk_size]
        pair_indices = torch.tensor([current], device=device, dtype=torch.long)
        pair_mask = torch.ones(
            pair_indices.shape[:2], device=device, dtype=torch.bool
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(pixels, points, pair_indices, pair_mask)
        scores.append(output["edge_logits"].float().cpu().sigmoid())
    return torch.cat(scores) if scores else torch.empty(0)


def sample_points(record: dict, image_size: int) -> torch.Tensor:
    width = float(record["layers"]["width"])
    height = float(record["layers"]["height"])
    atom_count = max(atom["atom_id"] for atom in record["layers"]["atoms"]) + 1
    points = torch.zeros((atom_count, 2))
    for atom in record["layers"]["atoms"]:
        points[atom["atom_id"]] = torch.tensor(
            (
                float(atom["center"][0]) / width,
                float(atom["center"][1]) / height,
            )
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.samples, args.chunk_size, args.image_size) < 1:
        raise ValueError("mining counts must be positive")
    if args.negative_ratio <= 0.0:
        raise ValueError("negative ratio must be positive")

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SemanticBondExistence(
        SemanticBondRelationConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["semantic_bond_existence"])
    model.eval().requires_grad_(False)
    records = islice(
        iter(
            ChemVectorIterableDataset(
                expand_shards(args.train_shards), sequence_version="0.4"
            )
        ),
        args.samples,
    )
    mined = []
    total_pairs = total_negatives = 0
    for index, record in enumerate(records, 1):
        points = sample_points(record, args.image_size)
        positives = {
            tuple(sorted((bond["begin_atom"], bond["end_atom"])))
            for bond in record["layers"]["bonds"]
        }
        pairs = list(combinations(range(len(points)), 2))
        negatives = [pair for pair in pairs if pair not in positives]
        pixels = image_tensor(record["image"], args.image_size)[None].to(device)
        scores = score_pairs(
            model,
            pixels,
            points[None].to(device),
            negatives,
            device,
            chunk_size=args.chunk_size,
        )
        count = min(len(negatives), max(1, round(len(positives) * args.negative_ratio)))
        selected = scores.topk(count).indices.tolist() if count else []
        mined.append(
            {
                "key": record["key"],
                "positive_count": len(positives),
                "non_bond_count": len(negatives),
                "pairs": [list(negatives[offset]) for offset in selected],
                "scores": [float(scores[offset]) for offset in selected],
            }
        )
        total_pairs += len(pairs)
        total_negatives += len(negatives)
        if index % 100 == 0:
            print(f"mined={index}", flush=True)
    result = {
        "schema_version": "semantic-bond-hard-negatives-0.1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "protocol": "top-scoring visual non-bonds from every unordered GT Junction pair; no length feature or filter",
        "requested_samples": args.samples,
        "samples": mined,
        "total_pairs": total_pairs,
        "total_non_bonds": total_negatives,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote={output} samples={len(mined)}", flush=True)


if __name__ == "__main__":
    main()
