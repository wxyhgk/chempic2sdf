#!/usr/bin/env python3
"""Merge disjoint quarantine sources into one metadata-bound cohort.

The inputs are public quarantine manifests only.  Sample payloads are copied
byte-for-byte into a new source directory; no model, sealed artifact, or
production checkpoint is opened.  The script fails closed on identity,
sample-key, image-byte, or split overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=2, metavar=("MANIFEST", "SOURCE"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows: list[dict] = []
    molecules: set[str] = set()
    samples: set[str] = set()
    images: set[str] = set()
    split_molecules: dict[str, set[str]] = {"train": set(), "holdout": set()}
    sources = []
    for manifest_arg, source_arg in args.input:
        manifest = Path(manifest_arg)
        source = Path(source_arg)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        source_rows = payload.get("rows", payload)
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"empty source: {manifest}")
        sources.append({"manifest": str(manifest), "manifest_sha256": digest(manifest), "rows": len(source_rows)})
        source_molecules: set[str] = set()
        source_split_molecules: dict[str, set[str]] = {"train": set(), "holdout": set()}
        for row in source_rows:
            molecule = str(row.get("molecule_id", "")); sample = str(row.get("sample_key", ""))
            image = str(row.get("image_sha256") or row.get("png_sha256") or "")
            if not molecule or not sample or not image:
                raise ValueError(f"missing identity/provenance in {manifest}: {row}")
            if molecule in molecules and molecule not in source_molecules:
                raise ValueError(f"molecule identity overlap: {molecule}")
            if sample in samples:
                raise ValueError(f"sample key overlap: {sample}")
            if image in images:
                raise ValueError(f"image hash overlap: {image}")
            molecules.add(molecule); source_molecules.add(molecule); samples.add(sample); images.add(image)
            split = str(row.get("split", ""))
            if split not in split_molecules:
                raise ValueError(f"unsupported split {split}")
            split_molecules[split].add(molecule)
            source_split_molecules[split].add(molecule)
            prefix = source / ""  # resolve below without trusting row paths
            sample_files = list(source.glob(f"{sample}.*"))
            required = {f"{sample}.png", f"{sample}.layers.json", f"{sample}.graph.json"}
            if not required.issubset({path.name for path in sample_files}):
                raise FileNotFoundError(f"incomplete sample payload for {sample}")
            rows.append(dict(row))
        if source_split_molecules["train"] & source_split_molecules["holdout"]:
            raise ValueError(f"train/holdout molecule overlap in {manifest}")
    if split_molecules["train"] & split_molecules["holdout"]:
        raise ValueError("train/holdout molecule overlap")
    args.out.mkdir(parents=True)
    out_source = args.out / "samples"
    out_source.mkdir()
    for manifest_arg, source_arg in args.input:
        source = Path(source_arg)
        for path in source.iterdir():
            if path.is_file():
                target = out_source / path.name
                if target.exists():
                    raise FileExistsError(target)
                shutil.copy2(path, target)
    manifest_payload = {
        "schema_version": "chemvector-r17-merged-quarantine-source-manifest-0.1",
        "source_id": "chemvector-r17-merged-quarantine-20261024-v1",
        "source_version": "20261024.v1",
        "source_seed": "merge:" + ",".join(item["manifest_sha256"][:12] for item in sources),
        "metadata_only": True,
        "training_authorized": False,
        "sealed_test_read": False,
        "rows": rows,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ledger_path = args.out / "prior-use-ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({**row, "event": "reserved_before_model_use", "source_namespace": manifest_payload["source_id"]}, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "chemvector-r17-merged-quarantine-source-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "rows": len(rows),
        "molecules": len(molecules),
        "split_molecules": {key: len(value) for key, value in split_molecules.items()},
        "source_inputs": sources,
        "manifest_sha256": digest(manifest_path),
        "ledger_sha256": digest(ledger_path),
        "training_admissible": False,
        "gpu_training_go": False,
        "production_go": False,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
