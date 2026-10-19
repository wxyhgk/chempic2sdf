#!/usr/bin/env python3
"""Audit ChemScene predictions for ChemTraverse parse/canonical identity."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from .chemtraverse_v0_1 import (
        parse_chemtraverse_v0_1,
        serialize_chemtraverse_v0_1,
    )
except ImportError:
    from chemtraverse_v0_1 import (
        parse_chemtraverse_v0_1,
        serialize_chemtraverse_v0_1,
    )


def audit_predictions(root: Path) -> dict[str, Any]:
    paths = sorted(root.rglob("*.chemscene.json"))
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    total_bytes = 0
    maximum_bytes = 0
    for path in paths:
        counts["scenes"] += 1
        try:
            scene = json.loads(path.read_text(encoding="utf-8"))
            sequence = serialize_chemtraverse_v0_1(scene)
            restored = parse_chemtraverse_v0_1(sequence)
            repeated = serialize_chemtraverse_v0_1(restored)
            if repeated != sequence:
                raise ValueError("ChemTraverse sequence is not byte-canonical")
            size = len(sequence.encode("utf-8"))
            total_bytes += size
            maximum_bytes = max(maximum_bytes, size)
            counts["canonical"] += 1
        except Exception as error:
            counts["failed"] += 1
            if len(failures) < 100:
                failures.append(
                    {
                        "path": str(path),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    return {
        "schema_version": "chemtraverse-prediction-audit-0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": "passed" if paths and counts["failed"] == 0 else "failed",
        "counts": dict(counts),
        "sequence": {
            "mean_bytes": total_bytes / max(counts["canonical"], 1),
            "maximum_bytes": maximum_bytes,
        },
        "sdf_sanitization_required": False,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_predictions(args.root.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
