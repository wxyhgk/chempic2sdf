"""Dual-reference CPU gate for the bounded hybrid ownership route."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-hybrid-merge-roundtrip-r5/report.json"
RAW = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-ownership-aromatic-critic-r5/report.json"
GUARD = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-topology-guard-r5/report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-hybrid-merge-gate-r5/report.json"


def main() -> dict:
    probe = json.loads(PROBE.read_text()); raw = json.loads(RAW.read_text()); guard = json.loads(GUARD.read_text())
    hold = probe["sweep"]["20.0"]["by_split"]["holdout"]
    profiles = {k: v for k, v in probe["sweep"]["20.0"]["by_profile"].items() if k.startswith("holdout:")}
    raw_hold = raw["splits"]["holdout"]
    guard_hold = guard["sweeps"]["0.4"]["by_split"]["holdout"]
    gate = {
        "checkpoint_unchanged": probe["gate"]["checkpoint_unchanged"], "sealed_test_read": False,
        "train_selected_T20": True,
        "precision_above_frozen_raw_floor": hold["edge_precision"] >= raw_hold["edge_precision"],
        "recall_above_frozen_raw_floor": hold["edge_recall"] >= raw_hold["edge_recall"],
        "sanitize_above_frozen_raw_floor": hold["oracle_sanitize_rate"] >= raw_hold["sanitize_critic"],
        "all_profiles_sanitize_ge_0_90": all(v["oracle_sanitize_rate"] >= .90 for v in profiles.values()),
        "all_profiles_recall_ge_0_75": all(v["edge_recall"] >= .75 for v in profiles.values()),
        "dangling_le_0_05": hold["dangling_rate"] <= .05,
        "relative_to_current_guard_precision_regression": hold["edge_precision"] < guard_hold["edge_precision"],
        "production_go": False,
    }
    gate["bounded_shadow_admissible"] = all(gate[k] for k in (
        "checkpoint_unchanged", "precision_above_frozen_raw_floor", "recall_above_frozen_raw_floor",
        "sanitize_above_frozen_raw_floor", "all_profiles_sanitize_ge_0_90", "all_profiles_recall_ge_0_75", "dangling_le_0_05"))
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-hybrid-merge-gate-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "selected": {"merge_px": 20.0, "guard_tau": .4}, "holdout": hold,
        "raw_reference": raw_hold, "current_guard_reference": guard_hold,
        "profiles": profiles, "gate": gate,
        "next": "bounded shadow only; precision regression versus current guard must remain explicit and blocks production",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False)); return report


if __name__ == "__main__": main()
