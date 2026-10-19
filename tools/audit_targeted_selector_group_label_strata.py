"""Read-only audit of raw-unmatched candidate distance strata.

This diagnostic checks whether ``confirmed_negative`` members in the frozen
group cache are actually close to a ground-truth atom center.  It reads only
the published cache and the corresponding ``layers.json`` atom centers from
the quarantine visual pool; it never reads sealed data, images, text records,
model checkpoints, or writes any training artifact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-group-label-strata-20261006-r1/report.json"
BANDS = ("le_6px", "gt_6_le_8px", "gt_8_le_10px", "gt_10px")


def _layers_centers(sample_key: str) -> list[tuple[float, float]]:
    payload = json.loads((DATA / f"{sample_key}.layers.json").read_text(encoding="utf-8"))
    if payload.get("training_authorized") is not False or payload.get("split") != "quarantine":
        raise ValueError(f"unexpected non-quarantine layers payload: {sample_key}")
    return [(float(atom["center"][0]), float(atom["center"][1])) for atom in payload["atoms"]]


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    q = statistics.quantiles(values, n=4) if len(values) >= 2 else [values[0]] * 3
    return {"count": len(values), "min": min(values), "q25": q[0], "median": q[1],
            "q75": q[2], "max": max(values)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    centers: dict[str, list[tuple[float, float]]] = {}
    report_splits: dict[str, dict] = {}
    for split in ("train", "holdout"):
        cache = json.loads((RELEASE / f"{split}-group-cache.json").read_text(encoding="utf-8"))["groups"]
        by_status: dict[str, list[float]] = defaultdict(list)
        by_profile: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        strata: Counter[tuple[str, str]] = Counter()
        group_sizes: Counter[tuple[str, int]] = Counter()
        for group in cache:
            sample_key = group["sample_key"]
            if sample_key not in centers:
                centers[sample_key] = _layers_centers(sample_key)
            target_centers = centers[sample_key]
            group_sizes[("strict_positive" if group["strict_positive"] else "protected" if group["protected"] else "confirmed_negative", len(group["members"]))] += 1
            for member in group["members"]:
                if member["raw_target"] >= 0 and member["selected_target"] < 0:
                    status = "strict_positive"
                elif member["raw_target"] < 0:
                    status = "confirmed_negative"
                else:
                    status = "protected"
                x, y = float(member["feature"][3]) * 768.0, float(member["feature"][4]) * 768.0
                distance = min(math.hypot(x - tx, y - ty) for tx, ty in target_centers)
                by_status[status].append(distance)
                by_profile[group["profile_id"]][status].append(distance)
                if status == "confirmed_negative":
                    if distance <= 6.0:
                        band = "le_6px"
                    elif distance <= 8.0:
                        band = "gt_6_le_8px"
                    elif distance <= 10.0:
                        band = "gt_8_le_10px"
                    else:
                        band = "gt_10px"
                    strata[(group["profile_id"], band)] += 1
        report_splits[split] = {
            "groups": len(cache),
            "group_size_by_status": {f"{status}:size_{size}": count for (status, size), count in sorted(group_sizes.items())},
            "member_distance_quantiles_px": {status: _quantiles(values) for status, values in by_status.items()},
            "confirmed_negative_bands_by_profile": {
                profile: {band: strata[(profile, band)] for band in BANDS}
                for profile in sorted(by_profile)
            },
        }
    result = {
        "schema_version": "chemvector-targeted-selector-group-label-strata-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "images_read": False,
        "text_records_read": False,
        "source": {"release": str(RELEASE.relative_to(ROOT)), "layers_atom_centers_only": True, "bands_px": list(BANDS)},
        "splits": report_splits,
        "interpretation": "confirmed negatives are partitioned by nearest layer atom-center distance; this is a label-strata audit, not training data publication",
        "policy": {
            "near_target_bands_not_confirmed_negative": ["le_6px", "gt_6_le_8px", "gt_8_le_10px"],
            "reason": "a raw-unmatched candidate near a layer atom center may be a coordinate near-miss; this audit must not turn that GT-derived distance into a training label",
            "training_authorized": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
