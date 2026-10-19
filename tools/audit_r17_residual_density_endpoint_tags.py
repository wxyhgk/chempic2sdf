#!/usr/bin/env python3
"""Metadata-only endpoint-tag preflight for residual-density sources.

The ordinary r17 strata audit verifies graph rows, but cannot enforce the
endpoint-conditioned tags used to select residual-positive cohorts.  This
small audit reads only a quarantine manifest (not PNG/SVG/model/sealed data),
counts independent molecules per split/profile, and fails closed when a
requested tag quota is absent or train/holdout identities overlap.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_HOLDOUT = {
    "EP:O:DOUBLE:D1": 2, "EP:O:SINGLE:D2": 2,
    "EP:N:DOUBLE:D1": 2, "EP:N:TRIPLE:D1": 2,
    "GRAPH:TRIPLE": 3, "GRAPH:AROMATIC": 1, "RING": 1,
}


def run(manifest: Path, out: Path, minimum: dict[str, int] | None = None) -> dict:
    if out.exists():
        raise FileExistsError(out)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    minimum = DEFAULT_HOLDOUT if minimum is None else minimum
    by = defaultdict(lambda: defaultdict(set))
    split_ids = defaultdict(set)
    duplicate_keys = []
    seen_keys = set()
    missing_tags = 0
    for row in rows:
        key = str(row.get("sample_key", "")); split = str(row.get("split", "")); profile = str(row.get("profile_id", ""))
        molecule = str(row.get("molecule_id", ""))
        if not key or key in seen_keys: duplicate_keys.append(key)
        seen_keys.add(key)
        split_ids[split].add(molecule)
        tags = row.get("endpoint_tags")
        if not isinstance(tags, list): missing_tags += 1; tags = []
        for tag in set(str(tag) for tag in tags):
            by[(split, profile)][tag].add(molecule)
    profiles = sorted({str(row.get("profile_id", "")) for row in rows})
    deficits = {}
    for split in ("train", "holdout"):
        for profile in profiles:
            counts = by[(split, profile)]
            for tag, quota in minimum.items():
                # Holdout quotas are the contract; train gets a conservative
                # one-molecule floor unless a caller supplies another policy.
                goal = quota if split == "holdout" else min(1, quota)
                if len(counts.get(tag, set())) < goal:
                    deficits[f"{split}/{profile}/{tag}"] = {"count": len(counts.get(tag, set())), "required": goal}
    overlap = len(split_ids.get("train", set()) & split_ids.get("holdout", set()))
    report = {
        "schema_version": "chemvector-r17-residual-density-endpoint-tags-0.1",
        "complete": True, "read_only": True, "metadata_only": True,
        "training_performed": False, "gpu_used": False, "sealed_test_read": False,
        "manifest": str(manifest.resolve()), "rows": len(rows),
        "profiles": profiles,
        "split_molecules": {s: len(v) for s, v in sorted(split_ids.items())},
        "tag_molecule_counts": {f"{s}/{p}": {tag: len(vals) for tag, vals in sorted(tags.items())}
                                for (s, p), tags in sorted(by.items())},
        "duplicate_sample_keys": duplicate_keys, "missing_endpoint_tags": missing_tags,
        "train_holdout_molecule_overlap": overlap,
        "deficits": deficits,
        "gate": {
            "manifest_rows_present": bool(rows), "unique_sample_keys": not duplicate_keys,
            "endpoint_tags_present": missing_tags == 0,
            "train_holdout_molecule_disjoint": overlap == 0,
            "requested_quotas_met": not deficits,
            "training_admissible": False, "gpu_training_go": False, "production_go": False,
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out / "report.json"), "gate": report["gate"], "deficits": deficits}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-tag", action="append", default=[], metavar="TAG=N",
                        help="override/add holdout quota; may be repeated")
    args = parser.parse_args()
    minimum = dict(DEFAULT_HOLDOUT)
    for item in args.min_tag:
        tag, value = item.rsplit("=", 1); minimum[tag] = int(value)
    run(args.manifest, args.out, minimum)
