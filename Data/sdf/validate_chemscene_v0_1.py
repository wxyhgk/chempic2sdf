#!/usr/bin/env python3
"""Batch validation for ChemScene v0.1 WebDataset targets."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .chemscene_v0_1 import (
        ChemSceneDepictionConflictError,
        build_chemscene_v0_1,
        chemsvg_from_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
        validate_chemscene_v0_1_roundtrip,
    )
    from .chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError
except ImportError:
    from chemscene_v0_1 import (
        ChemSceneDepictionConflictError,
        build_chemscene_v0_1,
        chemsvg_from_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
        validate_chemscene_v0_1_roundtrip,
    )
    from chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError


SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _load_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Tar member cannot be read: {name}")
    return json.load(handle)


def validate_archive(path: Path, limit: int | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    failures = []
    scene_bytes = 0
    chemsvg_bytes = 0
    sdf_bytes = 0
    with tarfile.open(path) as archive:
        names = set(archive.getnames())
        keys = sorted(name[: -len(SUFFIXES[0])] for name in names if name.endswith(SUFFIXES[0]))
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            counts["total"] += 1
            try:
                required = [key + suffix for suffix in SUFFIXES]
                missing = [name for name in required if name not in names]
                if missing:
                    raise ValueError(f"Missing members: {missing}")
                chemmol, render, layers = (
                    _load_json(archive, name) for name in required
                )
                scene = build_chemscene_v0_1(chemmol, render, layers)
                chemsvg = chemsvg_from_chemscene_v0_1(scene)
                sdf = sdf_from_chemscene_v0_1(scene)
                validate_chemscene_v0_1_roundtrip(
                    scene,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )
                scene_bytes += len(json.dumps(scene, separators=(",", ":")))
                chemsvg_bytes += len(chemsvg)
                sdf_bytes += len(sdf)
                counts["passed"] += 1
            except ChemSVGCoordinateCollisionError:
                counts["skipped_coordinate_collision"] += 1
            except ChemSceneDepictionConflictError:
                counts["skipped_depiction_conflict"] += 1
            except Exception as error:
                counts["failed"] += 1
                if len(failures) < 50:
                    failures.append(
                        {"key": key, "error_type": type(error).__name__, "error": str(error)}
                    )
    passed = counts["passed"]
    return {
        "schema_version": "chemscene-validation-0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(path),
        "status": "passed" if counts["failed"] == 0 else "failed",
        "counts": dict(counts),
        "mean_bytes": {
            "chemscene_json": scene_bytes / max(passed, 1),
            "chemsvg": chemsvg_bytes / max(passed, 1),
            "sdf": sdf_bytes / max(passed, 1),
        },
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_archive(args.archive, args.limit)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
