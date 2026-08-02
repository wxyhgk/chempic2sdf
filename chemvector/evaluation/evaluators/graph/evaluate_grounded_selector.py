#!/usr/bin/env python3
"""Evaluate a frozen dense-candidate/query-fusion selector checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.grounded_selector import GroundedQuerySelector, GroundedSelectorConfig
from chemvector.pretrain_grounded_selector import (
    load_frozen_pipeline,
    prepare_cache,
    validate_cached,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--reject-keys-file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    args = parser.parse_args()
    if min(args.samples, args.batch_size) < 1:
        raise ValueError("Evaluation sample and batch counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grounded selector evaluation")

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GroundedSelectorConfig(**payload["grounded_selector_config"])
    selector = GroundedQuerySelector(config).to(device)
    selector.load_state_dict(payload["grounded_selector"])
    selector.eval()
    base_model, detector, _ = load_frozen_pipeline(
        Path(payload["query_checkpoint"]),
        device,
    )
    reject_keys = load_reject_keys(args.reject_keys_file)
    loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.shards),
            sequence_version="0.4",
            reject_keys=reject_keys,
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    records = prepare_cache(
        base_model,
        detector,
        loader,
        device,
        samples=args.samples,
        dense_threshold=float(payload["dense_threshold"]),
        snap_distance_pixels=float(payload["snap_distance_pixels"]),
        cache_name="evaluation",
        candidate_centric=bool(payload.get("candidate_centric", False)),
    )
    training_args_path = Path(args.checkpoint).parent / "training_args.json"
    training_args = (
        json.loads(training_args_path.read_text(encoding="utf-8"))
        if training_args_path.is_file()
        else {}
    )
    result = validate_cached(
        selector,
        records,
        device,
        batch_size=args.batch_size,
        tolerance_pixels=args.tolerance_pixels,
        loss_kwargs={
            "ranking_weight": float(training_args.get("ranking_weight", 1.0)),
            "set_count_weight": float(training_args.get("set_count_weight", 1.0)),
            "cardinality_weight": float(training_args.get("cardinality_weight", 0.1)),
        },
    )
    result.update(
        {
            "checkpoint": args.checkpoint,
            "query_checkpoint": payload["query_checkpoint"],
            "selector_config": asdict(config),
            "candidate_centric": bool(payload.get("candidate_centric", False)),
            "shards": args.shards,
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

