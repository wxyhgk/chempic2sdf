#!/usr/bin/env python3
"""Build the train-only v0.9.10 Evidence Carrier pair feature cache."""

from __future__ import annotations

import argparse
import json

from chemvector.evidence_carrier_pair_cache import (
    build_evidence_carrier_pair_train_cache,
)


DEFAULT_TRAIN_SHARD = "Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-000000.tar"
DEFAULT_OUTPUT = (
    "outputs/cache/"
    "chemvector-v0910-evidence-carrier-pair-training-wide-all-variants.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=[DEFAULT_TRAIN_SHARD])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--partition-seed", default="chemvector-v0.9.10")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_evidence_carrier_pair_train_cache(
        args.train_shards,
        args.output,
        calibration_fraction=args.calibration_fraction,
        partition_seed=args.partition_seed,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    summary = {
        "cache_schema_version": payload["cache_schema_version"],
        "output": args.output,
        "record_count": payload["record_count"],
        "fit_records": len(payload["fit_records"]),
        "calibration_records": len(payload["calibration_records"]),
        "pairing_profile": payload["pairing_profile"],
        "uses_validation": payload["uses_validation"],
        "uses_internal_test": payload["uses_internal_test"],
    }
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
