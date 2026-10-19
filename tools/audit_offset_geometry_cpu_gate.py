"""CPU-only gate for a frozen offset-aware residual candidate cache.

The cache is produced by a separate read-only replay.  This command never
loads a model and never searches thresholds.  It applies one registered
geometry policy, preserves protected baseline rows, and reports post-hoc
labels only for evaluation.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

from chemvector.components.graph.proposals.offset_geometry_gate import (
    OffsetGeometryPolicy,
    audit_offset_gate,
    select_offset_residuals,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/audits/chemvector-offset-geometry-feature-cache-20261017-r1/candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-offset-geometry-cpu-gate-20261017-r1/report.json"
REQUIRED = {
    "candidate_id", "sample_key", "split", "profile_id", "protected",
    "role_text_prob", "peak_prominence", "ink_fraction",
    "text_extent_x_px", "text_extent_y_px", "offset_norm",
    "heat_score", "anchor_x_px", "anchor_y_px",
}
FORBIDDEN = {"target_id", "nearest_distance_px", "nearest_target_distance_px", "layers", "graph", "gt_atom_id"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"offset feature cache is not present: {input_path}; run the frozen replay exporter first")
    rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            forbidden = FORBIDDEN & row.keys()
            if missing or forbidden:
                raise ValueError(f"line {line_no}: missing={sorted(missing)} forbidden={sorted(forbidden)}")
            rows.append(row)
    if not rows:
        raise ValueError("empty feature cache")
    policy = OffsetGeometryPolicy()  # one pre-registered policy; no sweep
    selected = select_offset_residuals(rows, policy=policy)
    groups = defaultdict(list)
    selected_groups = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["profile_id"]))].append(row)
    for row in selected:
        selected_groups[(str(row["split"]), str(row["profile_id"]))].append(row)
    by_group = {}
    for key, values in sorted(groups.items()):
        metrics = audit_offset_gate(values, selected_groups.get(key, []))
        metrics["split"], metrics["profile_id"] = key
        by_group["/".join(key)] = metrics
    core = [m for key, m in by_group.items() if m["split"] in {"train", "holdout"}]
    gate = {
        "single_fixed_policy": True,
        "protected_retention_exact": all(m["protected_retention"] == 1.0 for m in core),
        "conservative_precision_ge_085": all(m["strict_precision_conservative"] >= 0.85 for m in core),
        "hard_negative_survival_le_015": all(m["hard_negative_survival"] <= 0.15 for m in core),
        "holdout_strict_positive_present": any(m["split"] == "holdout" and m["strict_positive_total"] > 0 for m in core),
        "holdout_strict_positive_selected": any(m["split"] == "holdout" and m["strict_positive_selected"] > 0 for m in core),
        "nonempty_residual_admission": any(m["selected"] > m["protected_total"] for m in core),
        "training_admissible": False,
        "reason": "CPU geometry admission only; no GPU or production authorization",
    }
    result = {
        "schema_version": "chemvector-offset-geometry-cpu-gate-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False,
        "feature_cache": str(input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path),
        "feature_cache_sha256": digest(input_path), "candidate_count": len(rows),
        "selected_count": len(selected), "policy": policy.__dict__ if hasattr(policy, "__dict__") else {
            name: getattr(policy, name) for name in policy.__dataclass_fields__
        },
        "forbidden_inputs": sorted(FORBIDDEN), "by_group": by_group, "gate": gate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "candidate_count": len(rows), "selected_count": len(selected), "gate": gate}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
