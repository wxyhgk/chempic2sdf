#!/usr/bin/env python3
"""Scan decoding thresholds for an implicit-only vertex checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.pretrain_implicit_vertex import ImplicitVertexCollator
from chemvector.pretrain_vertex import new_detection_totals, summarize_detection_metrics, update_detection_metrics
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, graph_payload = load_grounded_graph_checkpoint(payload["grounded_graph_checkpoint"], device)
    model, _, _, _ = load_frozen_grounded_pipeline(Path(graph_payload["grounded_selector_checkpoint"]), device)
    head = deepcopy(model.vertex_head).to(device)
    head.load_state_dict(payload["implicit_vertex_head"])
    head.eval()
    loader = DataLoader(
        ChemVectorIterableDataset(expand_shards(args.validation_shards), sequence_version="0.4"),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=ImplicitVertexCollator(),
    )
    totals = {threshold: new_detection_totals() for threshold in thresholds}
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = head(model.vision_encoder.forward_stroke_resolution(pixels))
            for threshold in thresholds:
                decoded = decode_vertices(output, threshold=threshold, maximum_vertices=256, nms_kernel_size=1)
                update_detection_metrics(
                    totals[threshold], decoded, batch["anchor_points"], batch["anchor_mask"],
                    batch["aromatic_targets"], 768, 6.0,
                )
    result = {
        str(threshold): summarize_detection_metrics(totals[threshold])
        for threshold in thresholds
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
