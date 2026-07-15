#!/usr/bin/env python3
"""Evaluate an Atom Query checkpoint over threshold and tolerance grids."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    decode_atom_queries,
    snap_queries_to_dense_candidates,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate_vertex import parse_positive_floats, parse_thresholds
from chemvector.pretrain_atom_query import query_features
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator
from chemvector.vertex import decode_vertices


@torch.no_grad()
def evaluate(
    base_model,
    detector,
    loader,
    device: torch.device,
    *,
    feature_level: str,
    thresholds: tuple[float, ...],
    tolerances: tuple[float, ...],
    dense_model=None,
    dense_threshold: float = 0.2,
    snap_distance_pixels: float = 64.0,
) -> dict:
    base_model.eval()
    detector.eval()
    modes = ("raw", "dense_snap") if dense_model is not None else ("raw",)
    totals = {
        (mode, threshold, tolerance): new_detection_totals()
        for mode in modes
        for threshold in thresholds
        for tolerance in tolerances
    }
    samples = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = query_features(
                base_model,
                pixels,
                feature_level=feature_level,
            )
            output = detector(features)
            dense_output = None
            if dense_model is not None:
                dense_features = dense_model.vision_encoder.forward_stroke_resolution(pixels)
                dense_output = dense_model.vertex_head(dense_features)
        dense_decoded = (
            decode_vertices(
                dense_output,
                threshold=dense_threshold,
                maximum_vertices=128,
            )
            if dense_output is not None
            else None
        )
        for threshold in thresholds:
            decoded = decode_atom_queries(output, threshold=threshold)
            decoded_modes = {"raw": decoded}
            if dense_decoded is not None:
                decoded_modes["dense_snap"] = [
                    snap_queries_to_dense_candidates(
                        query_items,
                        candidate_items,
                        image_size=batch["pixel_values"].shape[-1],
                        maximum_distance_pixels=snap_distance_pixels,
                    )
                    for query_items, candidate_items in zip(decoded, dense_decoded)
                ]
            for mode, mode_decoded in decoded_modes.items():
                for tolerance in tolerances:
                    update_detection_metrics(
                        totals[(mode, threshold, tolerance)],
                        mode_decoded,
                        batch["anchor_points"],
                        batch["anchor_mask"],
                        batch["aromatic_targets"],
                        batch["pixel_values"].shape[-1],
                        tolerance,
                    )
        samples += int(pixels.size(0))
    return {
        "samples": samples,
        "modes": {
            mode: {
                f"{tolerance:.4f}": {
                    f"{threshold:.4f}": summarize_detection_metrics(
                        totals[(mode, threshold, tolerance)]
                    )
                    for threshold in thresholds
                }
                for tolerance in tolerances
            }
            for mode in modes
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-level", choices=("stroke", "high"), default="stroke")
    parser.add_argument("--dense-checkpoint")
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--snap-distance-pixels", type=float, default=64.0)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.3,0.5,0.7"),
    )
    parser.add_argument(
        "--tolerances",
        type=parse_positive_floats,
        default=parse_positive_floats("4,6,8,10,16"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    base_model = load_vertex_model(Path(args.base_checkpoint), device).eval()
    payload = torch.load(args.query_checkpoint, map_location="cpu", weights_only=False)
    config = AtomQueryConfig(**payload["atom_query_config"])
    detector = AtomQueryDetector(config).to(device)
    detector.load_state_dict(payload["atom_query"], strict=True)
    dense_model = None
    if args.dense_checkpoint:
        dense_model = load_vertex_model(Path(args.dense_checkpoint), device).eval()
    expected_channels = base_model.config.vision_dims[
        0 if args.feature_level == "stroke" else 1
    ]
    if config.input_channels != expected_channels:
        raise ValueError("Atom Query checkpoint input channels do not match feature level")
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
        base_model,
        detector,
        loader,
        device,
        feature_level=args.feature_level,
        thresholds=args.thresholds,
        tolerances=args.tolerances,
        dense_model=dense_model,
        dense_threshold=args.dense_threshold,
        snap_distance_pixels=args.snap_distance_pixels,
    )
    result.update(
        {
            "base_checkpoint": args.base_checkpoint,
            "query_checkpoint": args.query_checkpoint,
            "query_config": asdict(config),
            "feature_level": args.feature_level,
            "dense_checkpoint": args.dense_checkpoint,
            "dense_threshold": args.dense_threshold,
            "snap_distance_pixels": args.snap_distance_pixels,
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
