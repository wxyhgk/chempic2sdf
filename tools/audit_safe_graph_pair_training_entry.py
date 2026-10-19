#!/usr/bin/env python3
"""Audit the isolated vertex-candidate -> graph/pair scorer training entry.

This audit is intentionally target-blind with respect to the candidate
artifact.  It checks that the scored vertex cache contains inference-only
fields, that it has an exact one-to-one join with the accepted render rows,
and that train/holdout identity keys are disjoint.  Ground-truth files are
not opened by this script; labels remain available only to the separate
post-hoc CPU preflight after this contract has passed.

The report is quarantine-only and never writes a checkpoint or modifies the
production model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
FORBIDDEN = {
    "target", "targets", "target_indices", "matched", "match", "label",
    "labels", "element", "visible_element", "atom_id", "graph", "bonds",
    "ground_truth", "groundtruth", "truth", "identity", "atom_refs",
}
ALLOWED_CANDIDATE_ROW = {"sample_key", "source", "candidates"}
ALLOWED_CANDIDATE = {"x", "y", "score"}
IDENTITY_KEYS = ("molecule_id", "canonical_smiles", "connection_identity")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recursive_keys(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            out.add(str(key).lower())
            out |= recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            out |= recursive_keys(item)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = load_jsonl(args.candidate)
    sample_rows = load_jsonl(args.samples)
    errors: list[str] = []
    if sha256(args.checkpoint) != INITIAL_SHA:
        errors.append("production checkpoint digest differs from fixed initial SHA")

    by_key: dict[str, dict[str, Any]] = {}
    candidate_nested_forbidden = 0
    for idx, row in enumerate(candidate_rows, 1):
        keys = set(row)
        if keys != ALLOWED_CANDIDATE_ROW:
            errors.append(f"candidate row {idx} fields={sorted(keys)}")
        if recursive_keys(row) & FORBIDDEN:
            candidate_nested_forbidden += 1
            errors.append(f"candidate row {idx} contains forbidden target-like field")
        key = row.get("sample_key")
        if not isinstance(key, str) or not key:
            errors.append(f"candidate row {idx} has invalid sample_key")
            continue
        if key in by_key:
            errors.append(f"duplicate candidate sample_key={key}")
        by_key[key] = row
        if row.get("source") != "shadow":
            errors.append(f"candidate row {idx} source is not shadow")
        values = row.get("candidates")
        if not isinstance(values, list):
            errors.append(f"candidate row {idx} candidates is not a list")
            continue
        for j, value in enumerate(values):
            if not isinstance(value, dict) or set(value) != ALLOWED_CANDIDATE:
                errors.append(f"candidate row {idx} candidate {j} has non-inference fields")
                continue
            try:
                x, y, score = float(value["x"]), float(value["y"]), float(value["score"])
            except (TypeError, ValueError):
                errors.append(f"candidate row {idx} candidate {j} is nonnumeric")
                continue
            if not all(map(lambda z: z == z and abs(z) != float("inf"), (x, y, score))):
                errors.append(f"candidate row {idx} candidate {j} is nonfinite")
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                errors.append(f"candidate row {idx} candidate {j} coordinate outside [0,1]")

    accepted = [row for row in sample_rows if row.get("render_status") == "accepted"]
    sample_by_key = {row.get("sample_key"): row for row in accepted}
    if len(sample_by_key) != len(accepted):
        errors.append("accepted sample rows contain duplicate sample_key")
    missing = sorted(set(sample_by_key) - set(by_key))
    extra = sorted(set(by_key) - set(sample_by_key))
    if missing:
        errors.append(f"candidate artifact missing {len(missing)} accepted sample keys")
    if extra:
        errors.append(f"candidate artifact has {len(extra)} keys absent from accepted samples")

    split_identity: dict[str, dict[str, set[str]]] = {"train": {}, "holdout": {}}
    for row in accepted:
        split = row.get("split")
        if split not in split_identity:
            errors.append(f"accepted row {row.get('sample_key')} has invalid split={split}")
            continue
        for name in IDENTITY_KEYS:
            value = row.get(name)
            if not isinstance(value, str) or not value:
                errors.append(f"accepted row {row.get('sample_key')} missing {name}")
                continue
            split_identity[split].setdefault(name, set()).add(value)
    overlap = {
        name: sorted(split_identity["train"].get(name, set()) & split_identity["holdout"].get(name, set()))
        for name in IDENTITY_KEYS
    }
    for name, values in overlap.items():
        if values:
            errors.append(f"train/holdout overlap for {name}: {len(values)}")

    report = {
        "schema_version": "chemvector-safe-graph-pair-training-entry-audit-0.1",
        "complete": not errors,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "checkpoint_sha256": sha256(args.checkpoint),
        "candidate_sha256": sha256(args.candidate),
        "candidate_rows": len(candidate_rows),
        "accepted_sample_rows": len(accepted),
        "candidate_nested_forbidden_rows": candidate_nested_forbidden,
        "candidate_keys_match_accepted": not missing and not extra,
        "split_counts": {
            split: sum(1 for row in accepted if row.get("split") == split)
            for split in ("train", "holdout")
        },
        "identity_counts": {
            split: {name: len(values) for name, values in split_identity[split].items()}
            for split in ("train", "holdout")
        },
        "identity_overlap_counts": {name: len(values) for name, values in overlap.items()},
        "errors": errors,
        "training_admissible": not errors,
        "next": "CPU pair scorer may run only from this fixed split; no production integration",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
