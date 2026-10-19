"""Deterministic graph-assignment veto for residual atom candidates.

This probe does not train a pointwise gate. For each image, protected raw
matches form the immutable baseline. All residual candidates are connected
into coordinate components at a fixed 12 px radius; a protected duplicate
within 6 px vetoes the residual, and at most one residual is assigned per
component. The highest frozen presence logit above a fixed zero gate wins.
Labels are consulted only after assignment for an audit, with near-unknown
and selected-unmatched reported separately. This is a graph-constraint probe,
not a bond-port prediction result; explicit predicted-port attachment remains
the next extension requiring frozen model replay.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-graph-assignment-veto-cpu-20261015-r1/report.json"
COMPONENT_RADIUS = 12.0
DUPLICATE_RADIUS = 6.0
SCORE_GATE = 0.0
CORE = ("coordgen", "original", "rdkit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(rows: list[dict]) -> list[list[dict]]:
    n = len(rows); parent = list(range(n))
    xy = np.asarray([[r["x_px"], r["y_px"]] for r in rows], dtype=np.float64)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(n):
        for j in range(i):
            if float(np.linalg.norm(xy[i] - xy[j])) <= COMPONENT_RADIUS:
                ri, rj = root(i), root(j)
                if ri != rj: parent[rj] = ri
    grouped: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows): grouped[root(i)].append(row)
    return list(grouped.values())


def assign_image(rows: list[dict]) -> list[dict]:
    baseline = [r for r in rows if r["label"] == "protected"]
    residual = [r for r in rows if r["label"] != "protected"]
    accepted = []
    for group in components(residual):
        # Existing baseline ownership is immutable: a residual duplicate is
        # vetoed before graph assignment.
        eligible = [r for r in group if not baseline or min(
            float(np.hypot(r["x_px"] - b["x_px"], r["y_px"] - b["y_px"])) for b in baseline
        ) > DUPLICATE_RADIUS]
        eligible = [r for r in eligible if float(r["presence_logit"]) >= SCORE_GATE]
        if eligible:
            accepted.append(max(eligible, key=lambda r: (float(r["presence_logit"]), -int(r["slot"]))))
    return accepted


def run() -> dict:
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    images: dict[str, list[dict]] = defaultdict(list)
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            images[row["sample_key"]].append(row)
    totals = defaultdict(lambda: Counter())
    per_image = []
    for sample_key, rows in images.items():
        accepted = assign_image(rows)
        split, variant = rows[0]["replay_split"], rows[0]["variant"]
        key = f"{split}/{variant}"
        c = totals[key]; c["images"] += 1; c["residual_candidates"] += sum(r["label"] != "protected" for r in rows)
        c["protected_baseline"] += sum(r["label"] == "protected" for r in rows); c["accepted"] += len(accepted)
        c["strict_selected"] += sum(r["label"] == "strict_positive" for r in accepted)
        c["hard_negative_selected"] += sum(r["label"] == "hard_negative" for r in accepted)
        c["near_unknown_selected"] += sum(r["label"] == "near_unknown" for r in accepted)
        c["selected_unmatched"] += sum(r["label"] == "selected_unmatched" for r in accepted)
        per_image.append({"sample_key": sample_key, "split": split, "variant": variant,
                          "residual": c["residual_candidates"], "accepted": len(accepted)})
    metrics = {}
    for key, c in totals.items():
        strict = c["strict_selected"]; accepted = c["accepted"]
        metrics[key] = {**dict(c), "strict_precision_conservative": strict / max(accepted, 1),
                        "assignment_recall_not_available": True,
                        "assignment": "one residual per coordinate component; baseline duplicate veto"}
    core = [metrics[f"dev_core/{v}"] for v in CORE]
    result = {"schema_version": "chemvector-style-factorial-graph-assignment-veto-cpu-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False,
              "cache_sha256": digest(CACHE), "component_radius_px": COMPONENT_RADIUS,
              "duplicate_veto_radius_px": DUPLICATE_RADIUS, "score_gate": SCORE_GATE,
              "baseline_policy": "protected immutable; residual merge cannot steal duplicate",
              "metrics": metrics, "per_image_count": len(per_image),
              "labels_not_used_for_assignment": True,
              "gate": {"core_styles_reported": all(f"dev_core/{v}" in metrics for v in CORE),
                       "stress_separate": "dev_stress/stress" in metrics,
                       "pointwise_gate_replaced": True, "training_admissible": False,
                       "next": "if this assignment has useful precision, replay frozen predicted bond_endpoints and replace coordinate components with attach_residual_candidates(radius=12px); keep protected occupancy and one-to-one endpoint ownership"}}
    result["gate"]["core_precision_ge_085"] = all(x["strict_precision_conservative"] >= .85 for x in core)
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "metrics": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False)); return result


if __name__ == "__main__": run()
