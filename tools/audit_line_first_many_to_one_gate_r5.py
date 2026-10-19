"""Strict CPU admission gate for the many-to-one ownership route."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-many-to-one-roundtrip-r5/report.json"
BASE = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-topology-guard-r5/report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-many-to-one-gate-r5/report.json"


def main() -> dict:
    report = json.loads(PROBE.read_text())
    hold = report["results"]["guard"]["by_split"]["holdout"]
    profiles = {k: v for k, v in report["results"]["guard"]["by_profile"].items() if k.startswith("holdout:")}
    baseline = json.loads(BASE.read_text())["sweeps"]["0.4"]["by_split"]["holdout"]
    gate = {
        "checkpoint_unchanged": report["gate"]["checkpoint_unchanged"],
        "sealed_test_read": False,
        "strict_precision_ge_hungarian_guard": hold["edge_precision"] >= baseline["edge_precision"],
        "strict_recall_ge_hungarian_guard": hold["edge_recall"] >= baseline["edge_recall"],
        "sanitize_ge_hungarian_guard": hold["oracle_sanitize_rate"] >= baseline["oracle_sanitize_rate"],
        "all_profiles_sanitize_ge_0_90": all(v["oracle_sanitize_rate"] >= .90 for v in profiles.values()),
        "all_profiles_recall_ge_0_75": all(v["edge_recall"] >= .75 for v in profiles.values()),
        "dangling_le_0_05": hold["dangling_rate"] <= .05,
        "gpu_shadow_admissible": False,
        "production_go": False,
    }
    gate["cpu_route_pass"] = all(value for key, value in gate.items()
                                  if key not in {"sealed_test_read", "gpu_shadow_admissible", "production_go"})
    output = {
        "schema_version": "chemvector-identity-aligned-line-first-many-to-one-gate-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "candidate_probe": str(PROBE), "baseline_guard": str(BASE),
        "holdout": hold, "baseline": baseline, "profiles": profiles, "gate": gate,
        "next": "if cpu_route_pass, run isolated <=800-step many-to-one edge/type shadow; never merge production",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    main()
