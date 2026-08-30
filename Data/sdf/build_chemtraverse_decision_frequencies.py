#!/usr/bin/env python3
"""Build frozen train-only ChemTraverse decision-token frequencies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from typing import Any

import torch

try:
    from .chemscene_v0_1 import build_chemscene_v0_1
    from .chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
    from .chemtraverse_v0_1 import serialize_chemtraverse_v0_1
except ImportError:
    from chemscene_v0_1 import build_chemscene_v0_1
    from chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
    from chemtraverse_v0_1 import serialize_chemtraverse_v0_1

try:
    from chemvector.chemtraverse_decision_loss import (
        build_chemtraverse_decision_plan,
        decision_token_frequencies,
    )
except ModuleNotFoundError as error:
    if error.name != "chemvector":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from chemvector.chemtraverse_decision_loss import (
        build_chemtraverse_decision_plan,
        decision_token_frequencies,
    )


FREQUENCY_SCHEMA_VERSION = "chemtraverse-decision-frequencies-0.1"
SOURCE_SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str | int]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _expand_shards(patterns: list[str | Path]) -> list[Path]:
    paths = []
    for pattern in patterns:
        value = str(pattern)
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match).resolve() for match in matches)
        elif Path(value).is_file():
            paths.append(Path(value).resolve())
    result = list(dict.fromkeys(paths))
    if not result:
        raise ValueError(f"No shards matched: {patterns}")
    return result


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Sample manifest is empty")
    sample_keys = [record.get("sample_key") for record in records]
    if any(not isinstance(key, str) or not key for key in sample_keys):
        raise ValueError("Every manifest record must contain a non-empty sample_key")
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Sample manifest contains duplicate sample keys")
    return records


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Missing tar member: {name}")
    return json.load(handle)


def build_decision_frequencies(
    shards: list[str | Path],
    sample_manifest: str | Path,
    target_molecule_id: str,
) -> dict[str, Any]:
    """Audit exactly the manifest samples without loading raster members."""

    if not target_molecule_id:
        raise ValueError("target_molecule_id must be non-empty")
    manifest_path = Path(sample_manifest).resolve()
    records = _manifest_records(manifest_path)
    shard_paths = _expand_shards(shards)
    allowed_shards = set(shard_paths)
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for record in records:
        shard_value = record.get("shard")
        if not isinstance(shard_value, str) or not shard_value:
            raise ValueError("Every manifest record must contain a shard path")
        shard_path = Path(shard_value)
        shard = (
            shard_path
            if shard_path.is_absolute()
            else manifest_path.parent / shard_path
        ).resolve()
        if shard not in allowed_shards:
            raise ValueError(
                f"Manifest shard is outside the declared --shards set: {shard}"
            )
        by_shard[shard].append(record["sample_key"])

    tokenizer = ChemTraverseTokenizerV01()
    frequencies: Counter[int] = Counter()
    valid_positions = 0
    decision_positions = 0
    forced_positions = 0
    audited = 0
    for shard in shard_paths:
        keys = by_shard.get(shard)
        if not keys:
            continue
        with tarfile.open(shard, "r") as archive:
            for key in keys:
                chemmol, render, layers = (
                    _json_member(archive, key + suffix) for suffix in SOURCE_SUFFIXES
                )
                scene = build_chemscene_v0_1(chemmol, render, layers)
                scene = {**scene, "molecule_id": target_molecule_id}
                sequence = serialize_chemtraverse_v0_1(scene)
                input_ids = torch.tensor(
                    [tokenizer.encode(sequence)],
                    dtype=torch.long,
                )
                plan = build_chemtraverse_decision_plan(input_ids, tokenizer)
                frequencies.update(decision_token_frequencies(plan))
                valid_positions += plan.valid_positions
                decision_positions += plan.decision_positions
                forced_positions += plan.forced_positions
                audited += 1

    if audited != len(records):
        raise ValueError(
            f"Audited {audited} samples but the manifest contains {len(records)}"
        )
    return {
        "schema_version": FREQUENCY_SCHEMA_VERSION,
        "sequence_schema_version": tokenizer.sequence_schema_version,
        "target_molecule_id": target_molecule_id,
        "sample_manifest": _file_record(manifest_path),
        "declared_shards": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in shard_paths
        ],
        "samples": {
            "manifest": len(records),
            "audited": audited,
        },
        "positions": {
            "valid": valid_positions,
            "decision": decision_positions,
            "forced": forced_positions,
        },
        "decision_frequencies": {
            tokenizer.id_to_token[token_id]: frequencies[token_id]
            for token_id in range(tokenizer.vocab_size)
            if frequencies[token_id]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--target-molecule-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_decision_frequencies(
        args.shards,
        args.sample_manifest,
        args.target_molecule_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
