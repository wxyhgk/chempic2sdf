#!/usr/bin/env python3
"""Read-only m128 abstain-only shadow receipt.

This audit consumes the target-free transaction journal produced by
``audit_r17_target_free_rollback.py`` but never commits a graph edit.  It
reports coverage of the proposed single-edge repair and treats every invalid
image as abstained in the output contract.  Target-bearing metric fields in
the source report are deliberately ignored.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / os.environ.get(
    "R17_ABSTAIN_INPUT",
    "outputs/audits/chemvector-r17-target-free-rollback-m128-panel128/report.json",
)
OUT = ROOT / os.environ.get(
    "R17_ABSTAIN_OUT",
    "outputs/audits/chemvector-r17-abstain-only-shadow-m128-panel128",
)
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    report = json.loads(INPUT.read_text())
    rows = report.get("rows", [])
    # The journal's routing fields are target-free.  Do not inspect any
    # predicted/target edge fields here, even though the input report stores
    # them for its separate post-hoc evaluation.
    by_split: dict[str, dict[str, int]] = {}
    decisions = []
    for row in rows:
        split = str(row["split"])
        c = by_split.setdefault(split, {"images": 0, "baseline_valid": 0,
                                        "baseline_invalid": 0, "repair_proposed": 0,
                                        "would_abstain": 0, "proposed_removals": 0,
                                        "attempts": 0})
        c["images"] += 1
        valid = bool(row["baseline_sanitize"])
        c["baseline_valid"] += int(valid)
        c["baseline_invalid"] += int(not valid)
        tx = row.get("transactions", [])
        proposed = bool(tx)
        c["repair_proposed"] += int(proposed)
        c["would_abstain"] += int((not valid) and not proposed)
        c["proposed_removals"] += len(tx)
        c["attempts"] += int(row.get("attempts", 0))
        decisions.append({"split": split, "sample_key": row["sample_key"],
                          "baseline_valid": valid,
                          "shadow_action": "keep" if valid else ("abstain_with_candidate" if proposed else "abstain"),
                          "proposed_removal_count": len(tx)})
    for c in by_split.values():
        n = max(c["images"], 1)
        c["baseline_valid_rate"] = c["baseline_valid"] / n
        c["abstain_rate"] = c["baseline_invalid"] / n
        c["repair_coverage_on_invalid"] = c["repair_proposed"] / max(c["baseline_invalid"], 1)
    initial_after = digest(INITIAL)
    result = {
        "schema_version": "chemvector-r17-abstain-only-shadow-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False,
        "input": str(INPUT.relative_to(ROOT)),
        "target_free_source_journal": True,
        "policy": {"invalid_graph": "abstain; no edge is deleted or committed",
                   "valid_graph": "keep unchanged", "posthoc_targets_used": False},
        "by_split": by_split,
        "decision_count": len(decisions),
        "decisions": decisions,
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": initial_after,
        "gate": {"checkpoint_unchanged": initial_after == SHA,
                 "sealed_test_read": False,
                 "no_graph_edit": True,
                 "target_free_routing": True,
                 "production_go": False},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "by_split": by_split,
                      "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
