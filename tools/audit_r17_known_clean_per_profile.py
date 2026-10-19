#!/usr/bin/env python3
"""Post-hoc per-profile strict audit of the frozen known-clean replay.

The graph replay already computed target-free decisions and strict identity
attribution for every row.  This audit only groups those persisted rows by
profile, with no model invocation, target rerouting, or compiler rerun.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/audits/chemvector-r17-selector-graph-replay-known-clean-v1/report.json"
OUT = ROOT / "outputs/audits/chemvector-r17-known-clean-per-profile-strict-v1/report.json"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric(rows: list[dict]) -> dict:
    def agg(arm: str) -> dict:
        identity = [row[f"{arm}_strict_identity"] for row in rows]
        pred = sum(x["predicted_edges"] for x in identity)
        mapped = sum(x["predicted_edges"] - x["unmapped_edges"] for x in identity)
        true = sum(x["true_edges"] for x in identity)
        typed = sum(x["typed_true_edges"] for x in identity)
        target = sum(x["target_edges"] for x in identity)
        atoms = sum(x["predicted_atoms"] for x in identity)
        mapped_atoms = sum(x["mapped_atoms"] for x in identity)
        target_atoms = sum(x["target_atoms"] for x in identity)
        elem = sum(x["element_correct"] for x in identity)
        return {
            "images": len(rows),
            "new_atoms": sum(row["new_atoms"] for row in rows) if arm == "augmented" else 0,
            "sanitize_rate": sum(bool(row[f"{arm}_sanitize"]) for row in rows) / max(len(rows), 1),
            "predicted_atoms": atoms,
            "mapped_atoms": mapped_atoms,
            "target_atoms": target_atoms,
            "atom_recall": mapped_atoms / max(target_atoms, 1),
            "element_accuracy": elem / max(mapped_atoms, 1),
            "predicted_edges": pred,
            "mapped_edges": mapped,
            "unmapped_edges": pred - mapped,
            "target_edges": target,
            "true_edges": true,
            "typed_true_edges": typed,
            "edge_precision": true / max(mapped, 1),
            "typed_edge_precision": typed / max(mapped, 1),
            "typed_edge_recall": typed / max(target, 1),
            "exact_typed_topology_rate": sum(bool(x["exact_typed_topology"]) for x in identity) / max(len(rows), 1),
            "exact_elements_and_typed_topology_rate": sum(bool(x["exact_elements_and_typed_topology"]) for x in identity) / max(len(rows), 1),
        }

    return {"baseline": agg("baseline"), "augmented": agg("augmented"),
            "delta": {"edge_precision": agg("augmented")["edge_precision"] - agg("baseline")["edge_precision"],
                      "typed_edge_precision": agg("augmented")["typed_edge_precision"] - agg("baseline")["typed_edge_precision"],
                      "typed_edge_recall": agg("augmented")["typed_edge_recall"] - agg("baseline")["typed_edge_recall"],
                      "sanitize_rate": agg("augmented")["sanitize_rate"] - agg("baseline")["sanitize_rate"],
                      "atom_recall": agg("augmented")["atom_recall"] - agg("baseline")["atom_recall"]}}


def run() -> dict:
    if OUT.parent.exists() and OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads(SOURCE.read_text())
    if report.get("initial_checkpoint_sha256") != SHA or digest(INITIAL) != SHA:
        raise RuntimeError("checkpoint binding mismatch")
    rows = report["rows"]
    by = {}
    for split in ("train", "fresh_holdout"):
        for profile in sorted({r["sample_key"].rsplit("__", 1)[-1] for r in rows if r["split"] == split}):
            by[f"{split}/{profile}"] = metric([r for r in rows if r["split"] == split and r["sample_key"].rsplit("__", 1)[-1] == profile])
    fresh = {p: by[f"fresh_holdout/{p}"] for p in sorted({k.split("/", 1)[1] for k in by if k.startswith("fresh_holdout/")})}
    # A profile must not lose strict typed recall or precision by >1 point;
    # this is an audit gate, not a deployment authorization.
    profile_gate = {p: {"typed_recall_no_drop_gt_0.01": v["delta"]["typed_edge_recall"] >= -0.01,
                         "typed_precision_no_drop_gt_0.01": v["delta"]["typed_edge_precision"] >= -0.01,
                         "sanitize_no_drop": v["delta"]["sanitize_rate"] >= 0.0,
                         "edge_precision_no_drop_gt_0.01": v["delta"]["edge_precision"] >= -0.01}
                    for p, v in fresh.items()}
    gate = {"checkpoint_unchanged": digest(INITIAL) == SHA, "rows_from_frozen_replay_only": True,
            "fresh_each_profile_strict_typed_recall_no_drop_gt_0.01": all(x["typed_recall_no_drop_gt_0.01"] for x in profile_gate.values()),
            "fresh_each_profile_strict_typed_precision_no_drop_gt_0.01": all(x["typed_precision_no_drop_gt_0.01"] for x in profile_gate.values()),
            "fresh_each_profile_sanitize_no_drop": all(x["sanitize_no_drop"] for x in profile_gate.values()),
            "fresh_each_profile_edge_precision_no_drop_gt_0.01": all(x["edge_precision_no_drop_gt_0.01"] for x in profile_gate.values()),
            "oa13_oa33_reported_separately": all(p in fresh for p in ("oa13", "oa33")),
            "production_go": False}
    result = {"schema_version": "chemvector-r17-known-clean-per-profile-strict-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "source_report": str(SOURCE.relative_to(ROOT)), "source_decision_digest": report.get("decision_digest"), "profiles": by, "fresh_profile_gate": profile_gate, "gate": gate}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "fresh": fresh, "gate": gate}, indent=2))
    return result


if __name__ == "__main__":
    run()
