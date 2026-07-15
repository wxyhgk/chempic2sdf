#!/usr/bin/env python3
"""Evaluate an OCR checkpoint on ChemVision layer shards."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.layer_ocr import ChemVisionLayerOCRDataset
from chemvector.ocr import OCRTextCodec
from chemvector.pretrain_ocr import load_ocr_model, validate
from chemvector.replacement_ocr import collate_replacement_ocr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.samples, args.batch_size) < 1 or args.workers < 0:
        raise ValueError("Evaluation sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    model = load_ocr_model(Path(args.checkpoint), device)
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    dataset = ChemVisionLayerOCRDataset(
        args.shards,
        seed=20260715 + 1_000_000_007,
        sample_limit=args.samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=partial(collate_replacement_ocr, codec=codec),
    )
    metrics = validate(model, loader, device, codec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
