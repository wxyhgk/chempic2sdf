"""Image-only crop readiness audit for the larger quarantine pool."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PLAN = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-plan-20261006-r1/report.json"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-pool-r1/group-cache.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-features-20261006-r1/report.json"


def auc(positive: list[float], negative: list[float]) -> float:
    values = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank, index = 0.0, 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        rank += ((index + 1 + end) / 2.0) * sum(label for _, label in values[index:end])
        index = end
    return (rank - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    groups = json.loads(CONTRACT.read_text(encoding="utf-8"))["groups"]
    images: dict[str, Image.Image] = {}
    by_split = {"train": [], "holdout": []}
    for group in groups:
        if group["protected"]:
            continue
        image = images.get(group["sample_key"])
        if image is None:
            path = DATA / "samples" / (group["sample_key"] + ".png")
            image = Image.open(path).convert("L")
            images[group["sample_key"]] = image
        features = [extract(image, member["feature"][3], member["feature"][4]) for member in group["members"]]
        score = max(feature["largest_component_fraction"] for feature in features)
        by_split[group["split"]].append({"score": score, "positive": bool(group["strict_positive"]),
                                         "profile_id": group["profile_id"]})
    metrics = {}
    for split, rows in by_split.items():
        positive = [r["score"] for r in rows if r["positive"]]
        negative = [r["score"] for r in rows if not r["positive"]]
        ordered = sorted(rows, key=lambda r: r["score"], reverse=True)
        topk = {}
        for k in (128, 256, 512, 1024, 2048):
            selected = ordered[:k]
            recovered = sum(r["positive"] for r in selected)
            topk[str(k)] = {"selected": len(selected), "recovered": recovered,
                            "conservative_precision": recovered / max(len(selected), 1),
                            "strict_recall": recovered / max(len(positive), 1)}
        metrics[split] = {"groups": len(rows), "strict_positive_groups": len(positive),
                          "confirmed_negative_groups": len(negative),
                          "auc_positive_higher": auc(positive, negative), "topk": topk,
                          "positive_by_profile": {profile: sum(r["positive"] and r["profile_id"] == profile for r in rows)
                                                   for profile in ("oa11", "oa13")}}
    result = {
        "schema_version": "chemvector-targeted-selector-visual-pool-features-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "layers_or_graph_read": False,
        "source": {"dataset": str(DATA.relative_to(ROOT)), "plan": str(PLAN.relative_to(ROOT)),
                    "contract": str(CONTRACT.relative_to(ROOT)), "images": len(images)},
        "crop_feature": "largest_component_fraction", "splits": metrics,
        "gate": {"holdout_auc_minimum_080": metrics["holdout"]["auc_positive_higher"] >= 0.80,
                 "holdout_top512_precision_minimum_085": metrics["holdout"]["topk"]["512"]["conservative_precision"] >= 0.85,
                 "oa11_positive_support": metrics["holdout"]["positive_by_profile"]["oa11"] >= 50,
                 "training_ready": False,
                 "reason": "image-only readiness; existence/abstention scorer still requires a new fixed run receipt"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "splits": metrics, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
