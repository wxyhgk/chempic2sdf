"""Read-only audit of coordinate candidate label strata.

This checks that the coordinate candidate objective distinguishes strict
selector drops, near-target refinement cases, far negatives, and neutral
members that are not labelled by the group contract.  It reads the frozen
visual-pool group cache and the coordinate release target metadata only; it
does not train, read images/model/sealed data, or publish labels.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
COORD = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-label-integrity-20261007-r2/report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_status(group: dict, member: dict) -> str:
    if member["raw_target"] >= 0 and member["selected_target"] < 0:
        return "strict_positive"
    if member["selected_target"] >= 0:
        return "protected"
    if member["raw_target"] < 0:
        return "confirmed_negative"
    return "neutral"


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((COORD / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sealed_test_read") is not False or manifest.get("training_run_authorized") is not False:
        raise ValueError("coordinate release is not sealed/run gated")
    splits: dict[str, dict] = {}
    for split in ("train", "holdout"):
        cache = json.loads((VISUAL / f"{split}-group-cache.json").read_text(encoding="utf-8"))["groups"]
        member_status = Counter()
        group_status = Counter()
        protected_members = 0
        neutral_rows: list[dict] = []
        group_positive_unknown_rows: list[dict] = []
        for group in cache:
            group_status["strict_positive" if group["strict_positive"] else "protected" if group["protected"] else "confirmed_negative"] += 1
            for member in group["members"]:
                status = cache_status(group, member)
                member_status[status] += 1
                if status == "protected":
                    protected_members += 1
                if status == "neutral":
                    neutral_rows.append({"sample_key": group["sample_key"], "group_id": group["group_id"],
                                         "slot": int(member["slot"]), "raw_target": int(member["raw_target"]),
                                         "selected_target": int(member["selected_target"]),
                                         "group_strict_positive": bool(group["strict_positive"]),
                                         "group_protected": bool(group["protected"])})
                # A strict-positive group can also carry unmatched members.
                # They are member-level confirmed negatives, but the old
                # group-level collect() path includes them in the candidate
                # head.  Keep this conflict explicit in the audit.
                if group["strict_positive"] and status == "confirmed_negative":
                    group_positive_unknown_rows.append({"sample_key": group["sample_key"], "group_id": group["group_id"],
                                                        "slot": int(member["slot"]),
                                                        "raw_target": int(member["raw_target"]),
                                                        "selected_target": int(member["selected_target"])})
        target_payload = json.loads((COORD / f"{split}-refinement-targets.json").read_text(encoding="utf-8"))
        target_status = Counter(t["status"] for t in target_payload["targets"])
        target_groups = Counter()
        for t in target_payload["targets"]:
            target_groups[t["group_id"]] += 1
        target_protected = 0
        cache_map = {g["group_id"]: g for g in cache}
        for t in target_payload["targets"]:
            group = cache_map[t["group_id"]]
            member = next(m for m in group["members"] if int(m["slot"]) == int(t["slot"]))
            target_protected += int(cache_status(group, member) == "protected")
        splits[split] = {
            "cache_groups": len(cache),
            "cache_group_status": dict(group_status),
            "cache_member_status": dict(member_status),
            "neutral_member_count": len(neutral_rows),
            "neutral_examples": neutral_rows[:20],
            "strict_group_member_negative_count": len(group_positive_unknown_rows),
            "strict_group_member_negative_examples": group_positive_unknown_rows[:20],
            "coordinate_target_rows": len(target_payload["targets"]),
            "coordinate_target_status": dict(target_status),
            "coordinate_target_protected_rows": target_protected,
            "coordinate_target_groups_with_multiple_rows": sum(v > 1 for v in target_groups.values()),
        }
    result = {
        "schema_version": "chemvector-targeted-selector-coordinate-label-integrity-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "images_read": False,
        "model_loaded": False,
        "source": {"visual_release": str(VISUAL.relative_to(ROOT)), "coordinate_release": str(COORD.relative_to(ROOT)),
                   "manifest_sha256": digest(COORD / "manifest.json")},
        "splits": splits,
        "policy": {
            "strict_positive": "raw_target>=0 and selected_target<0; selector-drop target only",
            "near_unknown": "raw_target<0 but candidate is within the coordinate refinement radius; refinement target, never selector negative",
            "far_negative": "raw_target<0 and candidate is outside refinement radius; only explicit negative stratum",
            "neutral_excluded": "raw_target>=0 and selected_target>=0 is not a selector training label; must be excluded from candidate confidence loss",
            "protected_coordinate_targets": "allowed for a separate frozen refiner diagnostic, but not candidate-gate labels",
            "training_authorized": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
