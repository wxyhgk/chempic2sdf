#!/usr/bin/env python3
"""Verify canonical bytes and recompute the v0.6 E1 witness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .produce_constructive_witness_v0_6 import (
        ConstructiveWitnessError,
        build_witness,
        canonical_bytes,
    )
except ImportError:
    from produce_constructive_witness_v0_6 import (  # type: ignore
        ConstructiveWitnessError,
        build_witness,
        canonical_bytes,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ConstructiveWitnessError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def verify(path: Path, v0_6_contract_sha256: str) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConstructiveWitnessError(f"invalid_json:{error}") from error
    if canonical_bytes(parsed) != payload:
        raise ConstructiveWitnessError("witness_is_not_canonical_json")
    expected = build_witness(v0_6_contract_sha256)
    if parsed != expected:
        raise ConstructiveWitnessError("witness_recomputation_mismatch")
    return {
        "schema_version": "chemgraph-constructive-witness-verification-v0.6",
        "status": "passed",
        "training_authorized": False,
        "witness_path": path.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v0-6-contract-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = canonical_bytes(verify(args.input, args.v0_6_contract_sha256))
    if args.output is not None:
        args.output.write_bytes(encoded)
    print(encoded.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
