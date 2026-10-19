#!/usr/bin/env python3
"""CPU/read-only validation for a visible O/N pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.core.data import ChemVectorIterableDataset  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("training_performed") is not False or manifest.get("sealed_test_read") is not False:
        raise ValueError("pilot manifest is not read-only")
    train_keys = {
        row["sample_key"]
        for group in manifest["train"]["molecules"]
        for row in group["rows"]
    }
    holdout_keys = {
        row["sample_key"]
        for group in manifest["holdout"]["molecules"]
        for row in group["rows"]
    }
    if train_keys & holdout_keys:
        raise ValueError("sample-key overlap between train and holdout")
    shards = sorted(str(path) for path in (args.data / "shards").glob("train-*.tar"))
    if not shards:
        raise FileNotFoundError("no public train shards found")
    loaded: dict[str, str] = {}
    for sample in ChemVectorIterableDataset(
        shards,
        include_keys=train_keys | holdout_keys,
        sequence_version="0.4",
        load_raster_layers=False,
    ):
        key = sample["key"]
        if key in loaded:
            raise ValueError(f"duplicate loaded key: {key}")
        if sample["meta"].get("split") != "train":
            raise ValueError(f"non-train source leaked into pilot: {key}")
        loaded[key] = str(sample["meta"]["molecule_id"])
    expected = train_keys | holdout_keys
    missing = sorted(expected - set(loaded))
    if missing:
        raise ValueError(f"unrepresentable or missing keys: {missing[:8]}")
    train_ids = {group["molecule_id"] for group in manifest["train"]["molecules"]}
    holdout_ids = {group["molecule_id"] for group in manifest["holdout"]["molecules"]}
    if train_ids & holdout_ids:
        raise ValueError("molecule identity overlap")
    result = {
        "schema_version": "chemvector-visible-on-data-first-pilot-audit-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "source_data": str(args.data),
        "manifest_sha256": sha256(args.manifest),
        "train_keys": len(train_keys),
        "holdout_keys": len(holdout_keys),
        "loaded_keys": len(loaded),
        "missing_keys": len(missing),
        "identity_overlap": len(train_ids & holdout_ids),
        "train_summary": manifest["train"]["summary"],
        "holdout_summary": manifest["holdout"]["summary"],
        "gate": {
            "all_keys_v04_representable": True,
            "identity_disjoint": True,
            "sealed_test_read": False,
            "training_admissible": False,
            "next": "frozen initial reachability audit; no GPU training",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
