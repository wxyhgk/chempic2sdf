"""Compare high-resolution rescue interfaces from already written reports.

This is deliberately report-only: it never loads a checkpoint, images, sealed
identities, or CUDA.  The two interfaces are:

* ``locked_residual_candidate`` (A): baseline assignment is locked first and
  sidecar candidates may match only targets left over from that assignment.
* ``baseline_only_reranking`` (B): no new geometry is admitted; an existing
  128-candidate set may only be re-ranked/retained.

The report makes the contract differences explicit instead of comparing the
field-shadow union score with scorer readiness AUPRC as if they were the same
metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    if path.is_dir():
        path = path / "report.json"
    return json.loads(path.read_text())


def _pick(path: str | None, patterns: tuple[str, ...]) -> Path:
    if path:
        return Path(path)
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(ROOT.glob(pattern))
    reports = [p / "report.json" for p in candidates if (p / "report.json").exists()]
    if not reports:
        raise FileNotFoundError(f"No report found for {patterns}")
    return max(reports, key=lambda p: p.stat().st_mtime)


def _field_summary(report: dict) -> dict:
    rows = report.get("results", [])
    if not rows:
        raise ValueError("field report has no per-image results")
    def vals(key: str):
        return [int(row[key]) for row in rows if key in row]
    peaks = vals("highres_rescue_peak_count")
    gaps = [int(row["targets"]) - int(row["raw_matched"]) for row in rows]
    k16 = [len(row.get("highres_ladder", {}).get("16", {}).get("newly_reachable_gap_ids", [])) for row in rows]
    union16 = [len(row.get("union_primary_plus_rescue", {}).get("16", {}).get("newly_matched_gap_ids", [])) for row in rows]
    return {
        "report": report.get("schema_version"),
        "images": len(rows),
        "raw_targets": sum(int(row["targets"]) for row in rows),
        "raw_matched": sum(int(row["raw_matched"]) for row in rows),
        "raw_gap": sum(gaps),
        "raw_gap_images": sum(value > 0 for value in gaps),
        "peak_count": {"total": sum(peaks), "mean": mean(peaks), "median": median(peaks),
                        "min": min(peaks), "max": max(peaks)},
        "k16_new_gap": sum(k16),
        "union_k16_new_gap": sum(union16),
        "per_image_counts": {
            "images_with_k16_new_gap": sum(value > 0 for value in k16),
            "images_with_union_k16_new_gap": sum(value > 0 for value in union16),
            "max_k16_new_gap": max(k16),
            "max_union_k16_new_gap": max(union16),
        },
        "field_contract": {
            "peak_pool": "local maxima sorted then top rescue_k (usually 32)",
            "primary_distance_guard": False,
            "baseline_lock": False,
            "residual_definition": "all targets minus old raw_target_ids, then post-hoc intersection",
            "union_assignment": "primary points + rescue points jointly Hungarian",
        },
    }


def _selector_summary(report: dict) -> dict:
    locked = report.get("locked_holdout") or {}
    safe = report.get("assignment_safe_holdout") or {}
    policy = report.get("policy_holdout") or {}
    if not locked:
        raise ValueError("selector report has no locked_holdout block")
    side = int(locked.get("locked_sidecar_matched", 0))
    candidates = int(locked.get("sidecar_candidates", 0))
    fp = int(safe.get("false_positives", max(0, candidates - side)))
    return {
        "report": report.get("schema_version"),
        "images": int(locked.get("images", 0)),
        "guard_px": report.get("rescue_min_dist_px"),
        "A_locked_holdout": {
            "baseline_matched": int(locked.get("locked_baseline_matched", 0)),
            "sidecar_candidates": candidates,
            "sidecar_matched": side,
            "sidecar_false_positives": fp,
            "sidecar_precision": side / max(candidates, 1),
            "total_matched": int(locked.get("locked_total_matched", 0)),
            "baseline_loss": 0,
        },
        "independent_fill_observed": {
            "baseline_loss": int(policy.get("lost_selected_ids", 0)),
            "new_selected_ids": int(policy.get("new_selected_ids", 0)),
            "policy_matched": int(policy.get("policy_matched", 0)),
        },
        "selector_contract": {
            "peak_pool": "top rescue peaks after 12 px guard",
            "primary_distance_guard": True,
            "baseline_lock": True,
            "residual_definition": "baseline Hungarian assignment first; sidecar sees remaining targets",
            "union_assignment": "not used for A; locked two-stage assignment",
        },
    }


def _readiness_summary(report: dict) -> dict:
    train = report.get("train_auprc", {})
    holdout = report.get("holdout_auprc", {})
    return {
        "report": report.get("schema_version"),
        "residual_only": report.get("residual_only"),
        "atom_threshold": report.get("atom_threshold"),
        "peak_k": report.get("peak_k"),
        "peak_pool_k": report.get("peak_pool_k"),
        "guard_px": report.get("rescue_min_dist_px"),
        "train": train,
        "holdout": holdout,
        "gate": report.get("gate", {}),
        "readiness_contract": {
            "peak_pool": "top PEAK_POOL_K then distance filter then PEAK_K",
            "primary_distance_guard": True,
            "baseline_lock": "not in historical readiness rows; labels recomputed from selected baseline",
            "residual_definition": "all targets minus freshly recomputed baseline_ids",
            "label_assignment": "side peaks Hungarian against unmatched targets, <= 6 px",
        },
    }


def run(args: argparse.Namespace) -> dict:
    field_path = _pick(args.field, ("outputs/audits/chemvector-atom-highres-pixel-trial-*/", "outputs/audits/chemvector-atom-highres-pixel-shadow-*/"))
    selector_path = _pick(args.selector, ("outputs/audits/chemvector-highres-selector-retention-*/",))
    readiness_path = _pick(args.readiness, ("outputs/audits/chemvector-targeted-400-residual-locked-readiness-*/", "outputs/audits/chemvector-rescue-geometry-readiness-800-*/"))
    field = _field_summary(_read(field_path))
    selector = _selector_summary(_read(selector_path))
    readiness = _readiness_summary(_read(readiness_path))

    # B has no new geometry.  The field union's newly reachable targets are an
    # observed upper-bound witness for what B cannot recover by re-ranking.
    b_upper_bound_witness = field["union_k16_new_gap"]
    a = selector["A_locked_holdout"]
    result = {
        "schema_version": "chemvector-highres-interface-comparison-0.1",
        "complete": True,
        "sealed_test_read": False,
        "training_performed": False,
        "gpu_used": False,
        "inputs": {"field": str(field_path), "selector": str(selector_path), "readiness": str(readiness_path)},
        "field_shadow": field,
        "selector_retention": selector,
        "strict_readiness": readiness,
        "decision": {
            "A_locked_residual_candidate": {
                "signal": "positive but low precision",
                "holdout_sidecar_precision": a["sidecar_precision"],
                "holdout_new_matches": a["sidecar_matched"],
                "baseline_preserved": a["baseline_loss"] == 0,
                "admit_training": False,
            },
            "B_baseline_only_reranking": {
                "new_geometry": False,
                "observed_primary_external_gap_witness": b_upper_bound_witness,
                "interpretation": "cannot recover targets whose useful point is absent from the existing 128 proposals",
                "admit_gpu": False,
            },
        },
        "next_action": {
            "gpu": "none for broad training; only a bounded parity replay after baseline/residual trace is fixed",
            "cpu": "keep A as audit-only interface, add per-image raw peaks -> post-guard peaks -> baseline IDs -> residual labels trace",
            "hard_gates": ["baseline target-ID parity", "residual-mask parity", "post-guard candidate count > 0", "at least one positive label"],
        },
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field")
    parser.add_argument("--selector")
    parser.add_argument("--readiness")
    parser.add_argument("--output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
