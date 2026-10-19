#!/usr/bin/env python3
"""Audit a composed quarantine cohort without pretending it is a new source.

Inputs must already have their own metadata admission receipts.  Composition
is allowed to reuse those quarantined identities, but the resulting report
explicitly records that it is a union, not an all-history-fresh source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("rows", payload)
    if not isinstance(result, list):
        raise ValueError(f"rows missing in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input-admission", action="append", type=Path, required=True)
    parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    merged = rows(args.manifest)
    admission = [json.loads(path.read_text(encoding="utf-8")) for path in args.input_admission]
    complexity = json.loads(args.complexity.read_text(encoding="utf-8"))
    sample_keys = [str(row.get("sample_key", "")) for row in merged]
    molecule_ids = [str(row.get("molecule_id", "")) for row in merged]
    image_hashes = [str(row.get("image_sha256") or row.get("png_sha256") or "") for row in merged]
    split_ids = {split: {row["molecule_id"] for row in merged if row.get("split") == split} for split in ("train", "holdout")}
    required = all(all(row.get(field) for field in ("sample_key", "molecule_id", "profile_id", "split", "image_sha256", "png_sha256", "rgb_sha256")) for row in merged)
    payload_complete = True
    for key in sample_keys:
        payload_complete &= all((args.source / f"{key}{suffix}").exists() for suffix in (".png", ".layers.json", ".graph.json"))
    report = {
        "schema_version": "chemvector-r17-merged-quarantine-source-audit-0.1",
        "complete": True,
        "read_only": True,
        "composed_source": True,
        "all_history_freshness_claim": False,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "manifest": str(args.manifest),
        "manifest_sha256": digest(args.manifest),
        "rows": len(merged),
        "molecules": len(set(molecule_ids)),
        "profiles": dict(Counter(row.get("profile_id") for row in merged)),
        "split_molecules": {split: len(values) for split, values in split_ids.items()},
        "input_admission_receipts": [
            {"path": str(path), "sha256": digest(path), "all_gates_except_training": all(value for key, value in payload.get("gate", {}).items() if key not in {"training_admissible", "gpu_training_go", "production_go"})}
            for path, payload in zip(args.input_admission, admission)
        ],
        "complexity_receipt": {"path": str(args.complexity), "sha256": digest(args.complexity), "gate": complexity.get("gate", {})},
        "gate": {
            "input_admission_receipts_present": len(admission) == len(args.input_admission),
            "input_metadata_gates_pass": all(all(value for key, value in payload.get("gate", {}).items() if key not in {"training_admissible", "gpu_training_go", "production_go"}) for payload in admission),
            "required_metadata_complete": required,
            "payload_files_complete": payload_complete,
            "unique_sample_keys": len(sample_keys) == len(set(sample_keys)),
            "unique_image_hashes": len(image_hashes) == len(set(image_hashes)),
            "train_holdout_molecule_disjoint": not (split_ids["train"] & split_ids["holdout"]),
            "complexity_matched": bool(complexity.get("gate", {}).get("complexity_matched_for_training")),
            "sealed_test_read": False,
            "training_admissible": False,
            "gpu_training_go": False,
            "production_go": False,
        },
    }
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out / "report.json"), "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
