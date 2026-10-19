#!/usr/bin/env python3
"""Metadata-only admission audit for a genuinely new r17 source batch.

This intentionally does not manufacture a holdout from an old release train
split.  A new source manifest and an append-only prior-use ledger are required
before the cohort can be called fresh.  Image/graph payloads and model weights
are never opened; the ledger binds source/version/seed and image-byte SHA
before any model inference or training.

Usage::

    python tools/audit_r17_new_source_prior_use_ledger.py \
      --manifest path/to/new_source_manifest.json \
      --ledger path/to/prior_use_ledger.jsonl \
      --out outputs/audits/chemvector-r17-new-source-admission-v1

The report is an admission receipt only.  It never authorizes GPU training or
production merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "outputs" / "audits"
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.r17_manifest_rows import rows_from_payload
IDENTITY_FIELDS = ("molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold")
PROVENANCE_FIELDS = ("source_id", "source_version", "source_seed", "image_sha256")
REQUIRED_ROW_FIELDS = (
    "sample_key", "molecule_id", "canonical_smiles", "connection_identity",
    "legacy_scaffold", "profile_id", "split", "source_id", "source_version",
    "source_seed", "image_sha256", "png_sha256", "rgb_sha256",
)
SKIP_PARTS = ("sealed", "internal-test", "test996")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text(value: Any) -> str:
    return "" if value is None or isinstance(value, (dict, list, tuple, set)) else str(value).strip()


def identities(row: dict[str, Any]) -> dict[str, str]:
    return {field: text(row.get(field)) for field in IDENTITY_FIELDS}


def image_hash(row: dict[str, Any]) -> str:
    return text(row.get("image_sha256") or row.get("png_sha256") or row.get("rgb_sha256"))


def rows_from(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return rows_from_payload(payload)


def history_manifest_rows(candidate: Path) -> tuple[dict[str, set[str]], set[str], set[str], list[str]]:
    """Collect visible metadata-only identity inventories.

    Older releases stored their molecule identity ledger as ``molecules.json``
    under ``outputs/datasets`` rather than as an audit ``manifest.json``.  A
    manifest-only scan can therefore falsely admit a canonical identity that
    was already rendered in one of those releases.  Include both public
    metadata forms while retaining the explicit sealed/test exclusions.  Image
    payloads are never opened.
    """
    seen = {field: set() for field in IDENTITY_FIELDS}
    sample_keys: set[str] = set()
    image_shas: set[str] = set()
    files: list[str] = []
    paths = set(HISTORY_ROOT.glob("**/manifest.json"))
    paths.update((ROOT / "outputs" / "datasets").glob("**/molecules.json"))
    paths.update((ROOT / "outputs" / "datasets").glob("**/manifest.json"))
    for path in sorted(paths):
        if path.resolve() == candidate.resolve() or any(part in str(path).lower() for part in SKIP_PARTS):
            continue
        try:
            rows = rows_from(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        files.append(str(path.relative_to(ROOT)))
        for row in rows:
            identity = identities(row)
            # A few older metadata-only plans used the explicit alias rather
            # than ``connection_identity``.  Treat it as the same identity
            # dimension for prior-use purposes.
            if not identity["connection_identity"]:
                identity["connection_identity"] = text(row.get("connection_identity_without_isotope_stereo"))
            for field, value in identity.items():
                if value:
                    seen[field].add(value)
            key = text(row.get("sample_key"))
            sha = image_hash(row)
            if key:
                sample_keys.add(key)
            if sha:
                image_shas.add(sha)
    return seen, sample_keys, image_shas, files


def ledger_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"ledger line {line_number} is not an object")
            out.append(row)
    return out


def run(manifest_path: Path, ledger_path: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    rows = rows_from(manifest_path)
    ledger = ledger_rows(ledger_path)
    history, history_samples, history_images, history_files = history_manifest_rows(manifest_path)
    overlaps = {field: sorted({identities(row)[field] for row in rows if identities(row)[field]} & history[field]) for field in IDENTITY_FIELDS}
    duplicate_samples = [key for key, n in Counter(text(row.get("sample_key")) for row in rows if text(row.get("sample_key"))).items() if n > 1]
    duplicate_images = [key for key, n in Counter(image_hash(row) for row in rows if image_hash(row)).items() if n > 1]
    source_values = {field: {text(row.get(field)) for row in rows if text(row.get(field))} for field in PROVENANCE_FIELDS}
    ledger_keys = {text(row.get("sample_key")) for row in ledger if text(row.get("sample_key"))}
    manifest_keys = {text(row.get("sample_key")) for row in rows if text(row.get("sample_key"))}
    ledger_molecules = {text(row.get("molecule_id")) for row in ledger if text(row.get("molecule_id"))}
    manifest_molecules = {text(row.get("molecule_id")) for row in rows if text(row.get("molecule_id"))}
    missing_provenance = {field: sum(not (image_hash(row) if field == "image_sha256" else text(row.get(field))) for row in rows) for field in PROVENANCE_FIELDS}
    missing_required_fields = {field: sum(not text(row.get(field)) for row in rows) for field in REQUIRED_ROW_FIELDS}
    manifest_by_key = {text(row.get("sample_key")): row for row in rows if text(row.get("sample_key"))}
    ledger_by_key = {text(row.get("sample_key")): row for row in ledger if text(row.get("sample_key"))}
    exact_fields = (
        "molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold",
        "profile_id", "split", "source_id", "source_version", "source_seed",
        "image_sha256", "png_sha256", "rgb_sha256",
    )
    ledger_exact_mismatch = [
        key for key, row in manifest_by_key.items()
        if key not in ledger_by_key or any(text(ledger_by_key[key].get(field)) != text(row.get(field)) for field in exact_fields)
    ]
    split_ids = defaultdict(set)
    for row in rows:
        split = text(row.get("split"))
        if split:
            split_ids[split].add(text(row.get("molecule_id")))
    split_overlap = len(split_ids.get("train", set()) & split_ids.get("holdout", set()))
    report = {
        "schema_version": "chemvector-r17-new-source-prior-use-ledger-admission-0.1",
        "complete": True,
        "read_only": True,
        "metadata_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": digest(manifest_path),
        "prior_use_ledger": str(ledger_path.resolve()),
        "prior_use_ledger_sha256": digest(ledger_path),
        "history_manifest_count": len(history_files),
        "history_manifest_files_sha256": hashlib.sha256("\n".join(history_files).encode()).hexdigest(),
        "rows": len(rows),
        "molecules": len(manifest_molecules),
        "profiles": dict(Counter(text(row.get("profile_id")) for row in rows)),
        "split_molecules": {split: len(ids) for split, ids in sorted(split_ids.items())},
        "missing_provenance": missing_provenance,
        "missing_required_fields": missing_required_fields,
        "duplicate_sample_keys": duplicate_samples,
        "duplicate_image_shas": duplicate_images,
        "history_overlap_counts": {field: len(values) for field, values in overlaps.items()},
        "history_overlap_examples": {field: values[:10] for field, values in overlaps.items()},
        "ledger_manifest_sample_overlap": len(ledger_keys & manifest_keys),
        "ledger_manifest_molecule_overlap": len(ledger_molecules & manifest_molecules),
        "ledger_exact_row_mismatch": len(ledger_exact_mismatch),
        "split_molecule_overlap": split_overlap,
        "ledger_rows": len(ledger),
        "gate": {
            "metadata_only": True,
            "ledger_present": True,
            "ledger_covers_manifest_samples": bool(manifest_keys) and manifest_keys <= ledger_keys,
            "ledger_covers_manifest_molecules": bool(manifest_molecules) and manifest_molecules <= ledger_molecules,
            "provenance_complete": all(count == 0 for count in missing_provenance.values()),
            "required_row_fields_complete": all(count == 0 for count in missing_required_fields.values()),
            "ledger_exact_row_match": not ledger_exact_mismatch,
            "unique_sample_keys": not duplicate_samples,
            "unique_image_bytes": not duplicate_images,
            "train_holdout_molecule_disjoint": split_overlap == 0,
            "all_history_identity_disjoint": all(not values for values in overlaps.values()),
            "source_ids_nonempty": all(source_values[field] for field in PROVENANCE_FIELDS),
            "training_admissible": False,
            "gpu_training_go": False,
            "production_go": False,
        },
        "next": "only after every metadata gate passes: frozen-initial target-free replay; still no GPU or production merge",
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out / "report.json"), "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.ledger, args.out)


if __name__ == "__main__":
    main()
