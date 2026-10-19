"""Read-only group-level readiness audit for the local PNG crop branch."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-group-release-20261006-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-group-readiness-20261006-r1/report.json"


def _auc(positive: list[float], negative: list[float]) -> float | None:
    values = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    pos_rank, index = 0.0, 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        pos_rank += ((index + 1 + end) / 2.0) * sum(label for _, label in values[index:end])
        index = end
    if not positive or not negative:
        return None
    return (pos_rank - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    groups_by_split = {
        "train": json.loads((RELEASE / "train-group-cache.json").read_text(encoding="utf-8"))["groups"],
        "holdout": json.loads((RELEASE / "holdout-group-cache.json").read_text(encoding="utf-8"))["groups"],
    }
    image_cache: dict[str, Image.Image] = {}
    rows = {}
    for split, groups in groups_by_split.items():
        records = []
        for group in groups:
            if group["protected"]:
                continue
            image = image_cache.get(group["sample_key"])
            if image is None:
                image = Image.open(RELEASE / "samples" / (group["sample_key"] + ".png")).convert("L")
                image_cache[group["sample_key"]] = image
            member_features = [extract(image, member["feature"][3], member["feature"][4]) for member in group["members"]]
            # The group feature is computed without labels.  The label is
            # attached only after the crop feature has been extracted.
            score = max(feature["largest_component_fraction"] for feature in member_features)
            records.append({"score": score, "positive": bool(group["strict_positive"]),
                            "profile_id": group["profile_id"]})
        rows[split] = records
    metrics = {}
    for split, records in rows.items():
        positive = [r["score"] for r in records if r["positive"]]
        negative = [r["score"] for r in records if not r["positive"]]
        ordered = sorted(records, key=lambda r: r["score"], reverse=True)
        total = len(positive)
        topk = {}
        for k in (16, 32, 64, 128, 256, 512):
            chosen = ordered[:k]
            recovered = sum(r["positive"] for r in chosen)
            topk[str(k)] = {"selected": len(chosen), "recovered": recovered,
                            "conservative_precision": recovered / max(len(chosen), 1),
                            "strict_recall": recovered / max(total, 1)}
        metrics[split] = {"groups": len(records), "strict_positive_groups": total,
                          "confirmed_negative_groups": len(negative),
                          "auc_positive_higher": _auc(positive, negative), "topk": topk}
    result = {
        "schema_version": "chemvector-targeted-selector-visual-group-readiness-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "layers_or_graph_read": False,
        "source": str(RELEASE.relative_to(ROOT)), "crop_feature": "largest_component_fraction",
        "splits": metrics,
        "gate": {
            "image_only": True,
            "holdout_auc_minimum_080": bool(metrics["holdout"]["auc_positive_higher"] >= 0.80),
            "holdout_top128_precision_minimum_085": bool(metrics["holdout"]["topk"]["128"]["conservative_precision"] >= 0.85),
            "training_ready": False,
            "reason": "AUC/readiness diagnostic only; existence/ranking scorer and fixed abstention calibration are still untrained",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "splits": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
