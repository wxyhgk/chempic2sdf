#!/usr/bin/env python3
"""Build one frozen eligible-image manifest for the ChemTraverse 50K A/B."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def build_manifest(
    shards: list[Path], count: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for shard in shards:
        if count is not None and len(accepted) >= count:
            break
        with tarfile.open(shard, "r") as archive:
            names = set(archive.getnames())
            keys = sorted(
                name[: -len(SUFFIXES[0])]
                for name in names
                if name.endswith(SUFFIXES[0])
            )
            for key in keys:
                if count is not None and len(accepted) >= count:
                    break
                counts["examined"] += 1
                try:
                    required = [key + suffix for suffix in SUFFIXES]
                    missing = [name for name in required if name not in names]
                    if missing:
                        raise ValueError(f"Missing members: {missing}")
                    chemmol, render, layers = (
                        _json_member(archive, name) for name in required
                    )
                    scene = build_chemscene_v0_1(chemmol, render, layers)
                    source = mol_from_chemscene_v0_1(scene)
                    source_key = Chem.MolToInchiKey(source)
                    if source_key != chemmol["inchikey"]:
                        raise ValueError(
                            f"Source ChemScene mismatch: {source_key} != {chemmol['inchikey']}"
                        )
                    sdf_from_chemscene_v0_1(scene)
                    sequence = validate_chemtraverse_v0_1_roundtrip(scene)
                    restored_key = Chem.MolToInchiKey(
                        mol_from_chemtraverse_v0_1(sequence)
                    )
                    if restored_key != chemmol["inchikey"]:
                        raise ValueError(
                            f"ChemTraverse mismatch: {restored_key} != {chemmol['inchikey']}"
                        )
                    accepted.append(
                        {
                            "sample_key": key,
                            "shard": str(shard),
                            "molecule_id": chemmol["molecule_id"],
                            "inchikey": chemmol["inchikey"],
                            "variant": render.get("variant"),
                        }
                    )
                    counts["accepted"] += 1
                except Exception as error:
                    rejected.append(
                        {
                            "sample_key": key,
                            "shard": str(shard),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    counts["rejected"] += 1
    if count is not None and len(accepted) != count:
        raise RuntimeError(f"Only found {len(accepted)} eligible samples; required {count}")
    return accepted, rejected, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help="Scan every sample instead of stopping after --count accepted samples",
    )
    parser.add_argument("--shard-pattern", default="train-*.tar")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    shard_dir = args.shard_dir.resolve()
    shards = sorted(shard_dir.glob(args.shard_pattern))
    if not shards:
        raise SystemExit(f"No train shards found in {shard_dir}")
    target_count = None if args.all_eligible else args.count
    accepted, rejected, counts = build_manifest(shards, target_count)
    accepted_bytes = _jsonl_bytes(accepted)
    rejected_bytes = _jsonl_bytes(rejected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "samples.jsonl").write_bytes(accepted_bytes)
    (args.output_dir / "source_rejections.jsonl").write_bytes(rejected_bytes)
    summary = {
        "schema_version": "chemtraverse-ab-manifest-0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_shard_dir": str(shard_dir),
        "selection": (
            "lexical shard then lexical sample key; all eligible samples"
            if target_count is None
            else "lexical shard then lexical sample key; replace rejected targets from subsequent keys"
        ),
        "shard_pattern": args.shard_pattern,
        "counts": dict(counts),
        "unique_molecules": len({record["molecule_id"] for record in accepted}),
        "samples_sha256": hashlib.sha256(accepted_bytes).hexdigest(),
        "source_rejections_sha256": hashlib.sha256(rejected_bytes).hexdigest(),
        "shared_sample_manifest": "samples.jsonl",
    }
    if target_count is not None:
        summary["groups"] = [
            "A0_atom_bond_set",
            "A1_chemtraverse",
            "A2_chemtraverse_aux",
        ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
