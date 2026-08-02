#!/usr/bin/env python3
"""Build the train-only frozen-P0 pair augmentation cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chemvector.evidence_carrier_p0_pair_cache import (
    DEFAULT_BASE_CACHE,
    DEFAULT_OUTPUT,
    DEFAULT_P0_CHECKPOINT,
    build_evidence_carrier_p0_pair_cache,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", default=DEFAULT_BASE_CACHE)
    parser.add_argument("--p0-checkpoint", default=DEFAULT_P0_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_evidence_carrier_p0_pair_cache(
        args.base_cache,
        args.p0_checkpoint,
        args.output,
        device=_device(args.device),
        batch_size=args.batch_size,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        {
            "output": str(Path(args.output).resolve()),
            "records": payload["record_count"],
            "complete": payload["complete"],
            "p0_global_step": payload["p0_global_step"],
            "anchor_feature_dtype": payload["anchor_feature_dtype"],
            "autocast_enabled": payload["p0_extraction_provenance"][
                "autocast_enabled"
            ],
            "uses_validation": payload["uses_validation"],
        }
    )


if __name__ == "__main__":
    main()
