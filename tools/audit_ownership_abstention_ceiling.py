"""Prove the recall ceiling of an abstention-only route.

This is a read-only CPU audit.  It does not fit a calibrator and does not load
images, models, checkpoints, sealed identities, or target geometry.  Since an
abstention policy may only remove candidates from the frozen selected prefix,
the baseline selected strict recall is an absolute upper bound.
"""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-ownership-abstention-ceiling-20261017/report.json"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    grouped = defaultdict(list)
    with CACHE.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["replay_split"] in {"train_core", "dev_core", "dev_stress"}:
                grouped[(row["replay_split"], row["variant"])].append(row)
    results = {}
    for key, rows in sorted(grouped.items()):
        selected = [row for row in rows if row["selected_kept"]]
        nonprotected_selected = [row for row in selected if row["label"] != "protected"]
        strict = [row for row in rows if row["label"] == "strict_positive"]
        strict_selected = [row for row in nonprotected_selected if row["label"] == "strict_positive"]
        protected = [row for row in rows if row["label"] == "protected"]
        protected_selected = [row for row in selected if row["label"] == "protected"]
        results[f"{key[0]}/{key[1]}"] = {
            "rows": len(rows),
            "baseline_selected_nonprotected": len(nonprotected_selected),
            "baseline_strict_selected": len(strict_selected),
            "all_strict_positive": len(strict),
            "abstention_only_recall_ceiling": len(strict_selected) / max(len(strict), 1),
            "protected_retention": len(protected_selected) / max(len(protected), 1),
            "hard_negative_selected": sum(row["label"] == "hard_negative" for row in nonprotected_selected),
        }
    core = {key: value for key, value in results.items() if key.startswith("dev_core/")}
    report = {
        "schema_version": "chemvector-ownership-abstention-ceiling-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "model_forwards": 0, "sealed_test_read": False,
        "candidate_count_changed": False, "cache_sha256": digest(CACHE),
        "results": results,
        "gate": {
            "required_strict_recall": 0.50,
            "each_dev_core_ceiling_ge_required": all(
                value["abstention_only_recall_ceiling"] >= 0.50 for value in core.values()),
            "decision": "PASS" if all(value["abstention_only_recall_ceiling"] >= 0.50 for value in core.values()) else "NO-GO",
            "reason": "abstention cannot recover strict positives absent from the frozen selected prefix",
            "training_authorized": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"gate": report["gate"], "dev_core": core}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
