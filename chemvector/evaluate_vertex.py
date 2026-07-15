#!/usr/bin/env python3
"""Evaluate one vertex checkpoint at several decoding thresholds in one pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator, decode_vertices


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in (0, 1)")
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return thresholds


def parse_positive_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be comma-separated positive numbers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be comma-separated positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    thresholds: tuple[float, ...],
    tolerance_pixels_values: tuple[float, ...],
    maximum_vertices_values: tuple[int, ...],
    maximum_batches: int,
) -> dict:
    model.eval()
    totals = {
        (maximum_vertices, tolerance_pixels, threshold): new_detection_totals()
        for maximum_vertices in maximum_vertices_values
        for tolerance_pixels in tolerance_pixels_values
        for threshold in thresholds
    }
    batches = 0
    samples = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = model.vision_encoder.forward_stroke_resolution(pixels)
            output = model.vertex_head(features)
        for maximum_vertices in maximum_vertices_values:
            for threshold in thresholds:
                decoded = decode_vertices(
                    output,
                    threshold=threshold,
                    maximum_vertices=maximum_vertices,
                )
                for tolerance_pixels in tolerance_pixels_values:
                    update_detection_metrics(
                        totals[(maximum_vertices, tolerance_pixels, threshold)],
                        decoded,
                        batch["anchor_points"],
                        batch["anchor_mask"],
                        batch["aromatic_targets"],
                        batch["pixel_values"].shape[-1],
                        tolerance_pixels,
                    )
        batches += 1
        samples += int(pixels.size(0))
        if maximum_batches > 0 and batches >= maximum_batches:
            break
    grid = {
        str(maximum_vertices): {
            f"{tolerance_pixels:.4f}": {
                f"{threshold:.4f}": summarize_detection_metrics(
                    totals[(maximum_vertices, tolerance_pixels, threshold)]
                )
                for threshold in thresholds
            }
            for tolerance_pixels in tolerance_pixels_values
        }
        for maximum_vertices in maximum_vertices_values
    }
    result = {
        "samples": samples,
        "batches": batches,
        "tolerance_pixels_values": tolerance_pixels_values,
        "maximum_vertices_values": maximum_vertices_values,
        "grid": grid,
    }
    if len(tolerance_pixels_values) == len(maximum_vertices_values) == 1:
        result["tolerance_pixels"] = tolerance_pixels_values[0]
        result["maximum_vertices"] = maximum_vertices_values[0]
        result["thresholds"] = grid[str(maximum_vertices_values[0])][
            f"{tolerance_pixels_values[0]:.4f}"
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.15,0.20,0.25,0.30,0.35"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--maximum-vertices", type=int, default=128)
    parser.add_argument("--tolerances", type=parse_positive_floats)
    parser.add_argument("--maximum-vertices-values", type=parse_positive_ints)
    parser.add_argument("--maximum-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    if args.tolerance_pixels <= 0 or args.maximum_vertices < 1:
        raise ValueError("tolerance and maximum vertices must be positive")
    tolerance_pixels_values = args.tolerances or (args.tolerance_pixels,)
    maximum_vertices_values = args.maximum_vertices_values or (args.maximum_vertices,)
    model = load_vertex_model(Path(args.checkpoint), device)
    loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.shards),
            sequence_version="0.4",
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=VertexCollator(),
    )
    result = evaluate(
        model,
        loader,
        device,
        thresholds=args.thresholds,
        tolerance_pixels_values=tolerance_pixels_values,
        maximum_vertices_values=maximum_vertices_values,
        maximum_batches=args.maximum_batches,
    )
    result.update(
        {
            "checkpoint": args.checkpoint,
            "shards": expand_shards(args.shards),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
