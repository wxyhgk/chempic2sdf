#!/usr/bin/env python3
"""Audit the exact-image union ceiling of old, new, and fused P19 routes."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from chemvector.evaluation.audits.graph import (
    audit_g3_p19e4_presence_threshold_calibration as p19e4,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e5_old_new_logit_fusion as p19e5,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p19e6-three-route-oracle"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
NEW_ROWS = p19e4.OUTPUT / "eval-rows.json"
NEW_ROWS_SHA256 = "702f55b151631047700969f41272313101be4c43d49c4a4cffdc37aa4739cd2d"
FUSION_ROWS = p19e5.OUTPUT / "fused-eval-rows.json"
FUSION_ROWS_SHA256 = "e7c610df1c3aa4c16c49f24b53bb0cb767b423a3cf0a2e37645d2163c26f8c3d"
OLD_PAIRED_ROWS = p19e5.OUTPUT / "paired-vs-old-rows.json"
OLD_PAIRED_ROWS_SHA256 = "0e9f7589c2cd41c24a388e920937c1f2489d116a870da32b417a88154960075f"
GATES = {
    "overall_min": 553,
    "fusion_headroom_min": 10,
    "nonstress_min": 456,
    "stress_min": 82,
    "crossed_min": 7,
    "near_nonbonded_min": 9,
}


def build_oracle_rows(
    new_rows: list[dict], fusion_rows: list[dict], old_rows: list[dict]
) -> list[dict]:
    new_by_key = {str(row["sample_key"]): row for row in new_rows}
    fusion_by_key = {str(row["sample_key"]): row for row in fusion_rows}
    old_by_key = {str(row["sample_key"]): row for row in old_rows}
    if (
        len(new_by_key) != len(new_rows)
        or len(fusion_by_key) != len(fusion_rows)
        or len(old_by_key) != len(old_rows)
    ):
        raise ValueError("P19-E6 input rows contain duplicate keys")
    if set(new_by_key) != set(fusion_by_key) or set(new_by_key) != set(old_by_key):
        raise ValueError("P19-E6 route sample keys differ")
    rows = []
    for key in sorted(new_by_key):
        new = new_by_key[key]
        fusion = fusion_by_key[key]
        old = old_by_key[key]
        if (
            new["variant"] != fusion["variant"]
            or new["variant"] != old["variant"]
            or new.get("stress_operation") != fusion.get("stress_operation")
            or new.get("stress_operation") != old.get("stress_operation")
        ):
            raise ValueError("P19-E6 route metadata differs")
        exact = {
            "new": bool(new["skeleton_exact"]),
            "old": bool(old["old_skeleton_exact"]),
            "fusion": bool(fusion["skeleton_exact"]),
        }
        rows.append(
            {
                "sample_key": key,
                "variant": str(new["variant"]),
                "stress_operation": new.get("stress_operation"),
                **{f"{route}_exact": value for route, value in exact.items()},
                "oracle_exact": any(exact.values()),
                "routes_disagree": len(set(exact.values())) > 1,
                "exact_routes": [route for route, value in exact.items() if value],
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "images": len(rows),
        "new_exact": sum(int(row["new_exact"]) for row in rows),
        "old_exact": sum(int(row["old_exact"]) for row in rows),
        "fusion_exact": sum(int(row["fusion_exact"]) for row in rows),
        "oracle_exact": sum(int(row["oracle_exact"]) for row in rows),
        "routes_disagree": sum(int(row["routes_disagree"]) for row in rows),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E6 output is not empty: {OUTPUT}")
    for path, expected in (
        (NEW_ROWS, NEW_ROWS_SHA256),
        (FUSION_ROWS, FUSION_ROWS_SHA256),
        (OLD_PAIRED_ROWS, OLD_PAIRED_ROWS_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E6 input SHA256 differs: {path}")
    rows = build_oracle_rows(
        json.loads(NEW_ROWS.read_text(encoding="utf-8")),
        json.loads(FUSION_ROWS.read_text(encoding="utf-8")),
        json.loads(OLD_PAIRED_ROWS.read_text(encoding="utf-8")),
    )
    overall = summarize(rows)
    by_variant = {
        variant: summarize([row for row in rows if row["variant"] == variant])
        for variant in ("coordgen", "original", "rdkit", "stress")
    }
    operations = sorted(
        {
            str(row["stress_operation"])
            for row in rows
            if row["stress_operation"] is not None
        }
    )
    by_operation = {
        operation: summarize(
            [row for row in rows if row["stress_operation"] == operation]
        )
        for operation in operations
    }
    route_patterns = dict(
        Counter(
            "".join(
                route[0].upper()
                for route in ("new", "old", "fusion")
                if row[f"{route}_exact"]
            )
            or "none"
            for row in rows
        )
    )
    nonstress = sum(
        by_variant[variant]["oracle_exact"]
        for variant in ("coordgen", "original", "rdkit")
    )
    checks = {
        "overall_oracle_min553": overall["oracle_exact"] >= GATES["overall_min"],
        "fusion_headroom_min10": (
            overall["oracle_exact"] - overall["fusion_exact"]
            >= GATES["fusion_headroom_min"]
        ),
        "nonstress_oracle_min456": nonstress >= GATES["nonstress_min"],
        "stress_oracle_min82": (
            by_variant["stress"]["oracle_exact"] >= GATES["stress_min"]
        ),
        "crossed_oracle_min7": (
            by_operation["crossed_bonds"]["oracle_exact"]
            >= GATES["crossed_min"]
        ),
        "near_oracle_min9": (
            by_operation["near_nonbonded"]["oracle_exact"]
            >= GATES["near_nonbonded_min"]
        ),
        "only_existing_routes_used": all(
            set(row["exact_routes"]) <= {"new", "old", "fusion"} for row in rows
        ),
    }
    decision = (
        "three_route_oracle_go" if all(checks.values()) else "three_route_oracle_no_go"
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "rows.json", rows)
    report = {
        "schema_version": "g3-p19e6-three-route-oracle-0.1",
        "decision": decision,
        "checks": checks,
        "gates": GATES,
        "overall": overall,
        "by_variant": by_variant,
        "stress_by_operation": by_operation,
        "nonstress_oracle_exact": nonstress,
        "route_patterns": route_patterns,
        "training": False,
        "oracle_uses_eval_labels": True,
        "deployment_result": False,
        "fresh640_read": False,
        "confirm_read": False,
        "sealed_test_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
