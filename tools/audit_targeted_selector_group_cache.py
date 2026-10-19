"""Read-only audit of the frozen group-selector cache.

The cache contains model candidates and post-forward labels.  This audit never
loads a checkpoint, reads sealed data, changes a label, or authorizes training.
It checks ownership/partition invariants and reports whether image/model
features carry a usable signal without treating that as a trained result.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r5/report.json"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-r3/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-group-cache-audit-20261006-r3/report.json"
FEATURE_NAMES = ("atom_score", "presence_logit", "pre_presence_logit", "x", "y", "port_distance_norm")


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = sum(a > b for a in positive for b in negative)
    ties = sum(a == b for a in positive for b in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    groups = contract.get("groups", [])
    rows = plan.get("rows", [])
    if len(rows) != 130 or len(groups) == 0:
        raise ValueError("Fixed r5 plan and nonempty r2 cache are required")
    row_by_key = {row["sample_key"]: row for row in rows}
    if len(row_by_key) != len(rows):
        raise ValueError("Duplicate planned sample key")
    if contract.get("sealed_test_read") is not False or contract.get("training_performed"):
        raise ValueError("Contract is not diagnostic-only")

    seen_groups: set[str] = set()
    by_split = {"train": [], "holdout": []}
    feature_rows = {"train": {name: {"positive": [], "negative": []} for name in FEATURE_NAMES},
                    "holdout": {name: {"positive": [], "negative": []} for name in FEATURE_NAMES}}
    large_groups = []
    partition = Counter()
    for group in groups:
        gid = group["group_id"]
        if gid in seen_groups:
            raise ValueError("Duplicate group identity")
        seen_groups.add(gid)
        key = group["sample_key"]
        row = row_by_key.get(key)
        if row is None or row["split"] != group["split"] or row["molecule_id"] != group["molecule_id"]:
            raise ValueError("Cache group is not bound to the fixed split plan")
        split = group["split"]
        members = group["members"]
        if not members:
            raise ValueError("Empty candidate group")
        if len(members) > 32:
            large_groups.append({"group_id": gid, "sample_key": key, "split": split,
                                 "size": len(members), "strict_positive": bool(group["strict_positive"])})
        strict_members = [m for m in members if m["raw_target"] >= 0 and m["selected_target"] < 0]
        protected_members = [m for m in members if m["selected_target"] >= 0]
        unknown_members = [m for m in members if m["raw_target"] < 0 and m["selected_target"] < 0]
        declared = ("strict_positive" if strict_members else "protected" if protected_members else "confirmed_negative")
        if bool(group["strict_positive"]) != bool(strict_members):
            raise ValueError("Strict-positive flag disagrees with member labels")
        if bool(group["protected"]) != bool(protected_members):
            raise ValueError("Protected flag disagrees with member labels")
        if bool(group["confirmed_negative"]) != bool(unknown_members and not strict_members and not protected_members):
            raise ValueError("Confirmed-negative flag disagrees with member labels")
        # Strict positives are the only fitting positives.  Protected groups
        # are hard-locked and all their members are excluded from negatives.
        if strict_members:
            partition["strict_positive_groups"] += 1
        elif protected_members:
            partition["protected_groups"] += 1
        else:
            partition["confirmed_negative_groups"] += 1
        partition[declared + "_members"] += len(members)
        for member in members:
            values = member.get("feature")
            if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
                raise ValueError("Feature vector schema changed")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise ValueError("Non-finite image/model feature")
            if not 0.0 <= values[3] <= 1.0 or not 0.0 <= values[4] <= 1.0 or values[5] < 0.0:
                raise ValueError("Geometry feature is outside normalized bounds")
            if member["raw_target"] >= 0 and member["selected_target"] < 0:
                destination = "positive"
            elif member["raw_target"] < 0 and member["selected_target"] < 0 and not strict_members and not protected_members:
                destination = "negative"
            else:
                destination = None
            if destination:
                for index, name in enumerate(FEATURE_NAMES):
                    feature_rows[split][name][destination].append(float(values[index]))
        by_split[split].append(group)

    if partition["strict_positive_groups"] != contract["totals"].get("train_strict_positive", 0) + contract["totals"].get("holdout_strict_positive", 0):
        raise ValueError("Strict-positive total does not match contract receipt")
    if {row["molecule_id"] for row in rows if row["split"] == "train"} & {row["molecule_id"] for row in rows if row["split"] == "holdout"}:
        raise ValueError("Train/holdout molecule overlap")

    auc = {}
    for split in by_split:
        auc[split] = {}
        for name in FEATURE_NAMES:
            values = feature_rows[split][name]
            auc[split][name] = {"positive_count": len(values["positive"]),
                                "negative_count": len(values["negative"]),
                                "auc_positive_higher": _auc(values["positive"], values["negative"])}

    result = {
        "schema_version": "chemvector-targeted-selector-group-cache-audit-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "optimizer_steps": 0, "gpu_used": False,
        "sealed_test_read": False, "plan": str(PLAN.relative_to(ROOT)),
        "contract": str(CONTRACT.relative_to(ROOT)), "plan_images": len(rows),
        "groups": len(groups), "partition": dict(partition), "feature_auc": auc,
        "large_groups_gt32": large_groups,
        "gate": {
            "identity_split": True,
            "feature_schema_image_only": True,
            "strict_positive_counts": {
                "train": contract["totals"].get("train_strict_positive", 0),
                "holdout": contract["totals"].get("holdout_strict_positive", 0),
            },
            "strict_positive_minimums": True,
            "large_group_review_required": bool(large_groups),
            "training_ready": False,
            "reason": "cache audit only; no group scorer or release admission",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "partition": dict(partition), "large_groups": len(large_groups), "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
