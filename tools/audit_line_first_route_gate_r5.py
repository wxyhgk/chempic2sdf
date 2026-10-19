#!/usr/bin/env python3
"""Train-only CPU gate for the endpoint-free line-first route.

This is deliberately a proposal feasibility audit, not a production decoder.
Raw raster line segments are clustered into provisional nodes and emitted as
unordered edges.  The target graph is read only after proposal construction
to score reachability; no target coordinate, atom, bond, or bond type enters
the proposal path.  A cluster radius is selected from train images only and
then evaluated once on the fixed holdout split.

The gate is intended to answer one question: is this route worth a small
isolated line/bond-type shadow, given that residual endpoint ownership has a
low reachable-edge ceiling?  It never writes a checkpoint or calls the
production graph/selector path.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_line_first_graph_traversal_r5 import DATA, audit_sample

OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-route-gate-r5"


def _rows(split: str) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("split") == split and row.get("render_status", "accepted") == "accepted"]
    rows.sort(key=lambda row: row["sample_key"])
    return rows


def _summarize(results: list[dict]) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in results])) if results else 0.0

    by_profile: dict[str, list[dict]] = defaultdict(list)
    metadata = {json.loads(line)["sample_key"]: json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()}
    for row in results:
        by_profile[metadata[row["sample_key"]]["profile_id"]].append(row)
    kinds = sorted({kind for row in results for kind in row.get("by_bond_type", {})})
    by_bond_type = {}
    for kind in kinds:
        gt = sum(int(row.get("by_bond_type", {}).get(kind, {}).get("gt", 0)) for row in results)
        recovered = sum(int(row.get("by_bond_type", {}).get(kind, {}).get("recovered", 0)) for row in results)
        by_bond_type[kind] = {
            "gt": gt,
            "recovered": recovered,
            "recall": float(recovered / gt) if gt else 1.0,
        }
    return {
        "images": len(results),
        "mean_bond_recall": mean("bond_recall"),
        "mean_edge_precision": mean("edge_precision"),
        "images_all_bonds_recovered": int(sum(bool(row["all_bonds_recovered"]) for row in results)),
        "mean_gt_bonds": mean("gt_bonds"),
        "mean_provisional_nodes": mean("provisional_nodes"),
        "mean_provisional_edges": mean("provisional_edges"),
        "by_bond_type": by_bond_type,
        "by_profile": {
            profile: {
                "images": len(group),
                "mean_bond_recall": float(np.mean([row["bond_recall"] for row in group])),
                "mean_edge_precision": float(np.mean([row["edge_precision"] for row in group])),
                "images_all_bonds_recovered": int(sum(bool(row["all_bonds_recovered"]) for row in group)),
            }
            for profile, group in sorted(by_profile.items())
        },
    }


def _evaluate(rows: list[dict], *, cluster_radius: float, target_radius: float, max_segments: int) -> list[dict]:
    return [
        audit_sample(
            row["sample_key"],
            row["split"],
            cluster_radius=cluster_radius,
            target_radius=target_radius,
            max_segments=max_segments,
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", default="8,10,12,16", help="train-only candidate cluster radii in pixels")
    parser.add_argument("--target-radius", type=float, default=24.0, help="post-hoc scoring radius; never used to form proposals")
    parser.add_argument("--max-segments", type=int, default=512)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    radii = tuple(float(item) for item in args.radii.split(",") if item.strip())
    if not radii or any(value <= 0 for value in radii):
        raise ValueError("radii must contain positive values")

    train_rows = _rows("train")
    holdout_rows = _rows("holdout")
    if not train_rows or not holdout_rows:
        raise RuntimeError("fixed train/holdout split is empty")

    # Radius selection is made exclusively from train metrics.  Precision is
    # required first; recall breaks ties.  Holdout is evaluated once below.
    train_sweeps = []
    for radius in radii:
        result = _evaluate(train_rows, cluster_radius=radius, target_radius=args.target_radius, max_segments=args.max_segments)
        summary = _summarize(result)
        train_sweeps.append({"cluster_radius_px": radius, **summary})
    eligible = [item for item in train_sweeps if item["mean_edge_precision"] >= 0.85]
    if not eligible:
        selected = max(train_sweeps, key=lambda item: (item["mean_edge_precision"], item["mean_bond_recall"]))
    else:
        selected = max(eligible, key=lambda item: (item["mean_bond_recall"], item["mean_edge_precision"], -item["cluster_radius_px"]))
    radius = float(selected["cluster_radius_px"])

    holdout_result = _evaluate(holdout_rows, cluster_radius=radius, target_radius=args.target_radius, max_segments=args.max_segments)
    holdout = _summarize(holdout_result)
    train = selected
    gate = {
        "train_precision_ge_085": bool(train["mean_edge_precision"] >= 0.85),
        "holdout_precision_ge_080": bool(holdout["mean_edge_precision"] >= 0.80),
        "holdout_recall_ge_050": bool(holdout["mean_bond_recall"] >= 0.50),
        "proposal_target_independent": True,
        "sealed_test_read": False,
        "training_performed": False,
        "production_checkpoint_written": False,
        "deployment_authorized": False,
        "cpu_gate": bool(
            train["mean_edge_precision"] >= 0.85
            and holdout["mean_edge_precision"] >= 0.80
            and holdout["mean_bond_recall"] >= 0.50
        ),
        "shadow_recommendation": "line/bond-type shadow" if holdout["mean_bond_recall"] >= 0.50 and holdout["mean_edge_precision"] >= 0.80 else "NO-GO",
    }
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-route-gate-r5-0.1",
        "complete": True,
        "read_only": True,
        "proposal_contract": "raster LSD segments -> endpoint clustering -> provisional unordered edges; target graph is consulted only after proposal construction",
        "split": {"train_images": len(train_rows), "holdout_images": len(holdout_rows)},
        "config": {"selected_cluster_radius_px": radius, "target_radius_px": args.target_radius, "max_segments": args.max_segments},
        "selection": {"source": "train-only", "candidates": train_sweeps, "selected": train},
        "train": train,
        "holdout": holdout,
        "gate": gate,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output / "report.json"), "selected_cluster_radius_px": radius, "train": train, "holdout": holdout, "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
