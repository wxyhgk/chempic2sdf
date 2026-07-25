#!/usr/bin/env python3
"""批量验收 WebDataset 中动态构建的 ChemScene v0.2 标签。"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .chemscene_v0_1 import ChemSceneDepictionConflictError
    from .chemscene_v0_2 import (
        build_chemscene_v0_2,
        parse_chemscene_v0_2,
        sdf_from_chemscene_v0_2,
        validate_chemscene_v0_2_roundtrip,
    )
    from .chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError
except ImportError:
    from chemscene_v0_1 import ChemSceneDepictionConflictError
    from chemscene_v0_2 import (
        build_chemscene_v0_2,
        parse_chemscene_v0_2,
        sdf_from_chemscene_v0_2,
        validate_chemscene_v0_2_roundtrip,
    )
    from chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError


SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _load_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Tar member cannot be read: {name}")
    return json.load(handle)


def _mean(total: int, count: int) -> float:
    return total / max(count, 1)


def validate_archive(path: Path, limit: int | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    atom_roles: Counter[str] = Counter()
    port_sources: Counter[str] = Counter()
    display_bond_types: Counter[str] = Counter()
    chemical_bond_types: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    object_totals: Counter[str] = Counter()
    object_minima = {"atom_objects": None, "bond_objects": None}
    object_maxima = {"atom_objects": 0, "bond_objects": 0}
    scene_bytes = 0
    sdf_bytes = 0

    with tarfile.open(path) as archive:
        names = set(archive.getnames())
        keys = sorted(
            name[: -len(SUFFIXES[0])]
            for name in names
            if name.endswith(SUFFIXES[0])
        )
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
                scene = build_chemscene_v0_2(chemmol, render, layers)
                parsed = parse_chemscene_v0_2(scene)
                sdf = sdf_from_chemscene_v0_2(scene)
                validate_chemscene_v0_2_roundtrip(
                    scene,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )

                atom_count = len(parsed.atom_objects)
                bond_count = len(parsed.bond_objects)
                object_totals["atom_objects"] += atom_count
                object_totals["bond_objects"] += bond_count
                for name, value in (
                    ("atom_objects", atom_count),
                    ("bond_objects", bond_count),
                ):
                    current_minimum = object_minima[name]
                    object_minima[name] = (
                        value if current_minimum is None else min(current_minimum, value)
                    )
                    object_maxima[name] = max(object_maxima[name], value)
                for atom in parsed.atom_objects:
                    atom_roles[atom.role] += 1
                    for port in atom.incident_ports:
                        port_sources[port.source] += 1
                for bond in parsed.bond_objects:
                    display_bond_types[bond.display_type] += 1
                    chemical_bond_types[bond.chemical_type] += 1

                scene_bytes += len(
                    json.dumps(scene, ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                sdf_bytes += len(sdf.encode("utf-8"))
                counts["passed"] += 1
            except ChemSVGCoordinateCollisionError:
                counts["skipped_coordinate_collision"] += 1
            except ChemSceneDepictionConflictError:
                counts["skipped_depiction_conflict"] += 1
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
        "schema_version": "chemscene-v0.2-validation-0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(path),
        "limit": limit,
        "status": "passed" if counts["failed"] == 0 else "failed",
        "counts": dict(counts),
        "atom_roles": dict(sorted(atom_roles.items())),
        "port_sources": dict(sorted(port_sources.items())),
        "display_bond_types": dict(sorted(display_bond_types.items())),
        "chemical_bond_types": dict(sorted(chemical_bond_types.items())),
        "objects": {
            name: {
                "total": object_totals[name],
                "mean_per_passed_sample": _mean(object_totals[name], passed),
                "min": object_minima[name],
                "max": object_maxima[name],
            }
            for name in ("atom_objects", "bond_objects")
        },
        "mean_bytes": {
            "chemscene_json": _mean(scene_bytes, passed),
            "sdf": _mean(sdf_bytes, passed),
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
