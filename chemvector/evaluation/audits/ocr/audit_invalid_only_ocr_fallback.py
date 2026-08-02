#!/usr/bin/env python3
"""Audit a deterministic invalid-primary/valid-expert OCR fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_FIELDS = (
    "sdf_valid",
    "formula_match",
    "atom_count_match",
    "bond_count_match",
    "nonstereo_structure_match",
    "chemical_match",
)


def load_records(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = str(row["molecule_id"])
        if key in records:
            raise ValueError(f"Duplicate molecule_id in {path}: {key}")
        records[key] = row
    if not records:
        raise ValueError(f"Prediction records are empty: {path}")
    return records


def audit_fallback(primary: dict[str, dict], expert: dict[str, dict]) -> dict:
    if set(primary) != set(expert):
        raise ValueError("Primary and expert prediction IDs differ")
    keys = sorted(primary)
    routed = [
        key
        for key in keys
        if not bool(primary[key]["sdf_valid"])
        and bool(expert[key]["sdf_valid"])
    ]
    routed_set = set(routed)
    chosen = {
        key: expert[key] if key in routed_set else primary[key]
        for key in keys
    }
    metrics = {}
    for field in METRIC_FIELDS:
        primary_count = sum(bool(primary[key][field]) for key in keys)
        expert_count = sum(bool(expert[key][field]) for key in keys)
        hybrid_count = sum(bool(chosen[key][field]) for key in keys)
        metrics[field] = {
            "primary": primary_count,
            "expert": expert_count,
            "hybrid": hybrid_count,
            "rescue": sum(
                not bool(primary[key][field]) and bool(chosen[key][field])
                for key in keys
            ),
            "break": sum(
                bool(primary[key][field]) and not bool(chosen[key][field])
                for key in keys
            ),
        }
    gate = {
        "routes_positive": len(routed) > 0,
        "sdf_valid_break_zero": metrics["sdf_valid"]["break"] == 0,
        "chemical_break_zero": metrics["chemical_match"]["break"] == 0,
        "chemical_rescue_positive": metrics["chemical_match"]["rescue"] > 0,
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "use_expert_iff_primary_invalid_and_expert_valid",
        "samples": len(keys),
        "routes": len(routed),
        "routed_molecule_ids": routed,
        "metrics": metrics,
        "gate": gate,
        "routing_uses_ground_truth": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-records", required=True)
    parser.add_argument("--expert-records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_fallback(
        load_records(Path(args.primary_records)),
        load_records(Path(args.expert_records)),
    )
    result["primary_records"] = args.primary_records
    result["expert_records"] = args.expert_records
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
