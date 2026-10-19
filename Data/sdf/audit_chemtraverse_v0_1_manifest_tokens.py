#!/usr/bin/env python3
"""Audit token lengths for every sample in a frozen ChemTraverse manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
from typing import Any

try:
    from .chemscene_v0_1 import build_chemscene_v0_1
    from .chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
    from .chemtraverse_v0_1 import serialize_chemtraverse_v0_1
except ImportError:
    from chemscene_v0_1 import build_chemscene_v0_1
    from chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
    from chemtraverse_v0_1 import serialize_chemtraverse_v0_1


SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def audit_manifest(manifest: Path, max_length: int) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for record in records:
        by_shard[Path(record["shard"])].append(record["sample_key"])
    tokenizer = ChemTraverseTokenizerV01()
    lengths: list[int] = []
    longest: list[dict[str, Any]] = []
    over_limit: list[dict[str, Any]] = []
    for shard in sorted(by_shard):
        with tarfile.open(shard, "r") as archive:
            for key in by_shard[shard]:
                payloads = []
                for suffix in SUFFIXES:
                    handle = archive.extractfile(key + suffix)
                    if handle is None:
                        raise ValueError(f"Missing {key + suffix}")
                    payloads.append(json.load(handle))
                scene = build_chemscene_v0_1(*payloads)
                sequence = serialize_chemtraverse_v0_1(scene)
                length = len(tokenizer.encode(sequence))
                lengths.append(length)
                item = {"sample_key": key, "shard": str(shard), "tokens": length}
                longest.append(item)
                if length > max_length:
                    over_limit.append(item)
    lengths.sort()
    longest.sort(key=lambda item: (-item["tokens"], item["sample_key"]))
    return {
        "schema_version": "chemtraverse-token-length-audit-0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "status": "passed" if len(lengths) == len(records) and not over_limit else "failed",
        "counts": {
            "manifest": len(records),
            "audited": len(lengths),
            "over_limit": len(over_limit),
        },
        "tokenizer": {
            "schema_version": tokenizer.sequence_schema_version,
            "vocabulary_size": tokenizer.vocab_size,
            "max_length": max_length,
        },
        "lengths": {
            "minimum": min(lengths, default=0),
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "maximum": max(lengths, default=0),
        },
        "longest_10": longest[:10],
        "over_limit": over_limit[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_manifest(args.manifest.resolve(), args.max_length)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
