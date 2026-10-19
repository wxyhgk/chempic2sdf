#!/usr/bin/env python3
"""Validate canonical ChemTraverse v0.1 targets from a WebDataset shard."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
from typing import Any

from rdkit import Chem, RDLogger

try:
    from .chemscene_v0_1 import (
        build_chemscene_v0_1,
        mol_from_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
    )
    from .chemtraverse_v0_1 import (
        mol_from_chemtraverse_v0_1,
        validate_chemtraverse_v0_1_roundtrip,
    )
except ImportError:
    from chemscene_v0_1 import (
        build_chemscene_v0_1,
        mol_from_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
    )
    from chemtraverse_v0_1 import (
        mol_from_chemtraverse_v0_1,
        validate_chemtraverse_v0_1_roundtrip,
    )


SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Cannot read tar member: {name}")
    return json.load(handle)


def validate_archive(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    source_rejections: list[dict[str, str]] = []
    total_bytes = 0
    total_records = 0
    maximum_bytes = 0
    maximum_records = 0
    with tarfile.open(path, "r") as archive:
        names = set(archive.getnames())
        keys = sorted(
            name[: -len(SUFFIXES[0])]
            for name in names
            if name.endswith(SUFFIXES[0])
        )
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            counts["samples"] += 1
            required = [key + suffix for suffix in SUFFIXES]
            missing = [name for name in required if name not in names]
            if missing:
                counts["source_rejected"] += 1
                source_rejections.append(
                    {"key": key, "error_type": "MissingMembers", "error": str(missing)}
                )
                continue
            chemmol, render, layers = (
                _json_member(archive, name) for name in required
            )
            try:
                scene = build_chemscene_v0_1(chemmol, render, layers)
                source_mol = mol_from_chemscene_v0_1(scene)
                source_inchikey = Chem.MolToInchiKey(source_mol)
                if source_inchikey != chemmol["inchikey"]:
                    raise ValueError(
                        "Source ChemScene InChIKey mismatch: "
                        f"{source_inchikey} != {chemmol['inchikey']}"
                    )
                sdf_from_chemscene_v0_1(scene)
            except Exception as error:
                counts["source_rejected"] += 1
                if len(source_rejections) < 50:
                    source_rejections.append(
                        {
                            "key": key,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                continue

            counts["eligible"] += 1
            try:
                sequence = validate_chemtraverse_v0_1_roundtrip(scene)
                restored = mol_from_chemtraverse_v0_1(sequence)
                actual_inchikey = Chem.MolToInchiKey(restored)
                if actual_inchikey != chemmol["inchikey"]:
                    raise ValueError(
                        f"InChIKey mismatch: {actual_inchikey} != {chemmol['inchikey']}"
                    )
                sequence_bytes = len(sequence.encode("utf-8"))
                sequence_records = len(sequence.splitlines())
                total_bytes += sequence_bytes
                total_records += sequence_records
                maximum_bytes = max(maximum_bytes, sequence_bytes)
                maximum_records = max(maximum_records, sequence_records)
                counts["passed"] += 1
            except Exception as error:
                counts["failed"] += 1
                if len(failures) < 50:
                    failures.append(
                        {
                            "key": key,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    passed = counts["passed"]
    return {
        "schema_version": "chemtraverse-validation-0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(path),
        "status": "passed" if counts["failed"] == 0 else "failed",
        "counts": dict(counts),
        "sequence": {
            "mean_bytes": total_bytes / max(passed, 1),
            "maximum_bytes": maximum_bytes,
            "mean_records": total_records / max(passed, 1),
            "maximum_records": maximum_records,
        },
        "source_rejections": source_rejections,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    result = validate_archive(args.archive.resolve(), limit=args.limit)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
