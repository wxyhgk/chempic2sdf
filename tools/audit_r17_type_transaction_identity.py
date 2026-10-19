#!/usr/bin/env python3
"""Post-hoc identity/type audit for the target-free type transaction shadow."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / os.environ.get("R17_TYPE_TX_ID_REPORT", "outputs/audits/chemvector-r17-type-transaction-m128-v1/report.json")
OUT = ROOT / os.environ.get("R17_TYPE_TX_ID_OUT", "outputs/audits/chemvector-r17-type-transaction-m128-v1-identity/report.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads(REPORT.read_text())
    by_split = {}
    total = Counter()
    details = []
    for split in ("train", "dev", "fresh_holdout"):
        rows = [row for row in report["rows_detail"] if row["split"] == split]
        counts = Counter()
        for row in rows:
            baseline = row["baseline_types"]
            transaction = row["transaction_types"]
            truth = row["target_types"]
            for key, before in baseline.items():
                after = transaction.get(key)
                if after == before:
                    continue
                counts["changed"] += 1
                counts["before_correct"] += int(truth.get(key) == before)
                counts["after_correct"] += int(truth.get(key) == after)
                counts["changed_from_to"] += 1
                total["changed"] += 1
                total["before_correct"] += int(truth.get(key) == before)
                total["after_correct"] += int(truth.get(key) == after)
                details.append({"split": split, "sample_key": row["sample_key"], "edge": key,
                                "before": before, "after": after, "truth": truth.get(key)})
        by_split[split] = {
            "changed": counts["changed"],
            "before_correct": counts["before_correct"],
            "after_correct": counts["after_correct"],
            "changed_accuracy_before": counts["before_correct"] / max(counts["changed"], 1),
            "changed_accuracy_after": counts["after_correct"] / max(counts["changed"], 1),
        }
    result = {
        "schema_version": "chemvector-r17-type-transaction-identity-0.1",
        "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "input": str(REPORT.relative_to(ROOT)), "input_sha256": sha256(REPORT),
        "by_split": by_split, "changed_details": details,
        "gate": {
            "source_report_complete": bool(report.get("complete")),
            "edge_set_unchanged": bool(report.get("gate", {}).get("edge_set_unchanged")),
            "no_aromatic_edits": bool(report.get("gate", {}).get("aromatic_edges_untouched")),
            "fresh_changed_type_accuracy_non_decrease": by_split["fresh_holdout"]["changed_accuracy_after"] >= by_split["fresh_holdout"]["changed_accuracy_before"],
            "fresh_overall_type_accuracy_non_decrease": report["summary"]["fresh_holdout"]["transaction_type_accuracy"] >= report["summary"]["fresh_holdout"]["baseline_type_accuracy"],
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "by_split": by_split, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
