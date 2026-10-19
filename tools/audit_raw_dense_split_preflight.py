"""Read-only admission check for a strict raw-gap dense-field trial.

This consumes only the already-published dense reachability report and the
public 1,200-row cohort metadata.  It does not load images or a model, train,
write a checkpoint, or access sealed identities.  The purpose is to prevent a
new GPU trial when the identity-disjoint split has too few raw-gap positives.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

from chemvector.training.curriculum.pair_scale_training import source_checks
from tools.current_initial_cohort import split_group_disjoint


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
DENSE = ROOT / "outputs/audits/chemvector-dense-anchor-reachability-20261006/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_RAW_DENSE_PREFLIGHT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-raw-dense-split-preflight-20261006-r2"),
))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    dense = json.loads(DENSE.read_text(encoding="utf-8"))
    if fixed.get("sealed_test_read") or dense.get("sealed_test_read"):
        raise ValueError("Input evidence is not training-only")
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    manifest = source_checks()["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in keys]
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 cohort rows, found {len(rows)}")
    train, holdout, split = split_group_disjoint(rows, 480)
    split_by_key = {row["sample_key"]: "train" for row in train}
    split_by_key.update({row["sample_key"]: "holdout" for row in holdout})
    records = dense["records"]
    if any(row["sample_key"] not in split_by_key for row in records):
        raise ValueError("Dense report contains rows outside the fixed cohort")

    # A positive is a raw-missing target for which some dense cell is already
    # within the strict 6px ownership radius.  This is a capacity/readout
    # admission signal only; it is not a deployment metric.
    by_partition: dict[str, list[dict]] = {"train": [], "holdout": []}
    for record in records:
        by_partition[split_by_key[record["sample_key"]]].append(record)

    partitions = {}
    for name, values in by_partition.items():
        positives = [row for row in values if bool(row["all_cell_within_6px"])]
        partitions[name] = {
            "images": len(train) if name == "train" else len(holdout),
            "raw_gap_targets": len(values),
            "dense_reachable_positives": len(positives),
            "styles": dict(Counter(row["style"] for row in positives)),
            "roles": dict(Counter(row["role"] for row in positives)),
            "elements": dict(Counter(row["element"] for row in positives)),
        }
    gate = {
        "train_min50_positives": partitions["train"]["dense_reachable_positives"] >= 50,
        "holdout_min50_positives": partitions["holdout"]["dense_reachable_positives"] >= 50,
        "identity_disjoint": all(split[key] == 0 for key in (
            "group_overlap", "molecule_overlap", "canonical_smiles_overlap")),
        "sealed_test_read": False,
    }
    # ``sealed_test_read`` is a negative safety assertion: False is the
    # passing value, unlike the positive-count/identity predicates above.
    gate["bounded_trial_admissible"] = (
        gate["train_min50_positives"]
        and gate["holdout_min50_positives"]
        and gate["identity_disjoint"]
        and not gate["sealed_test_read"]
    )
    result = {
        "schema_version": "chemvector-raw-dense-split-preflight-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "dense_report": str(DENSE.relative_to(ROOT)),
        "split": split,
        "partitions": partitions,
        "gate": gate,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "gate": gate, "partitions": partitions},
                     indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
