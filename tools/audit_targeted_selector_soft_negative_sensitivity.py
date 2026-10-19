"""Label-strata sensitivity audit for near-target raw candidates.

This is a data-quality diagnostic only.  It reads released cache candidates
and quarantined layer atom centers to identify raw-unmatched candidates within
10px of a target.  Such candidates are reported as unknown/near-miss, never as
training negatives.  No model, image, sealed payload, or optimizer is used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-soft-negative-sensitivity-20261006-r1/report.json"
NEAR_THRESHOLD_PX = 10.0


def auc(pos: list[float], neg: list[float]) -> float:
    values = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        rank += ((i + 1 + j) / 2.0) * sum(label for _, label in values[i:j])
        i = j
    return (rank - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    rows = {}
    images: dict[str, Image.Image] = {}
    for split, filename in (("train", "train-group-cache.json"), ("holdout", "holdout-group-cache.json")):
        groups = json.loads((RELEASE / filename).read_text(encoding="utf-8"))["groups"]
        atom_centers = {}
        values = []
        for group in groups:
            if group["protected"]:
                continue
            key = group["sample_key"]
            if key not in atom_centers:
                layers = json.loads((RELEASE / "samples" / (key + ".layers.json")).read_text(encoding="utf-8"))
                atom_centers[key] = [(float(a["center"][0]), float(a["center"][1])) for a in layers["atoms"]]
            image = images.get(key)
            if image is None:
                image = Image.open(RELEASE / "samples" / (key + ".png")).convert("L")
                images[key] = image
            for member in group["members"]:
                raw = member["raw_target"] >= 0 and member["selected_target"] < 0
                confirmed = group["confirmed_negative"] and member["raw_target"] < 0 and member["selected_target"] < 0
                if not (raw or confirmed):
                    continue
                x, y = member["feature"][3] * 768.0, member["feature"][4] * 768.0
                distance = min(math.hypot(x - ax, y - ay) for ax, ay in atom_centers[key])
                crop = extract(image, member["feature"][3], member["feature"][4])
                # Fixed readiness score; normalization is fit from train later.
                values.append({"positive": raw, "near_unknown": confirmed and distance <= NEAR_THRESHOLD_PX,
                               "hard_negative": confirmed and distance > NEAR_THRESHOLD_PX,
                               "atom_score": float(member["feature"][0]),
                               "largest_component": crop["largest_component_fraction"],
                               "distance_px": distance})
        rows[split] = values
    # Train-only normalization, then one frozen equal-weight combined score.
    normalizers = {}
    for name in ("atom_score", "largest_component"):
        vals = [r[name] for r in rows["train"]]
        normalizers[name] = (min(vals), max(vals))
    metrics = {}
    for split, values in rows.items():
        for row in values:
            score = 0.0
            for name in normalizers:
                lo, hi = normalizers[name]
                score += (row[name] - lo) / max(hi - lo, 1e-9)
            row["score"] = score / 2.0
        positives = [r for r in values if r["positive"]]
        hard = [r for r in values if r["hard_negative"]]
        unknown = [r for r in values if r["near_unknown"]]
        ordered = sorted(values, key=lambda r: r["score"], reverse=True)
        topk = {}
        for k in (32, 64, 128, 512):
            chosen = ordered[:k]
            topk[str(k)] = {"selected": len(chosen),
                            "strict_positive": sum(r["positive"] for r in chosen),
                            "near_unknown": sum(r["near_unknown"] for r in chosen),
                            "hard_negative": sum(r["hard_negative"] for r in chosen),
                            "strict_precision_all": sum(r["positive"] for r in chosen) / max(len(chosen), 1),
                            "strict_precision_excluding_near_unknown": sum(r["positive"] for r in chosen) / max(sum(r["positive"] or r["hard_negative"] for r in chosen), 1)}
        metrics[split] = {"candidates": len(values), "strict_positive": len(positives),
                          "near_unknown": len(unknown), "hard_negative": len(hard),
                          "near_unknown_fraction": len(unknown) / max(len(values), 1),
                          "score_auc_vs_hard_negative": auc([r["score"] for r in positives], [r["score"] for r in hard]),
                          "topk": topk}
    result = {
        "schema_version": "chemvector-targeted-selector-soft-negative-sensitivity-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "images_read": True, "layers_atom_centers_only": True,
        "near_threshold_px": NEAR_THRESHOLD_PX, "normalizers_fit_from_train_only": True,
        "splits": metrics,
        "policy": {"near_unknown_excluded_from_negative_loss": True,
                   "gt_distance_not_a_model_feature": True,
                   "training_authorized": False,
                   "reason": "sensitivity audit only; does not publish a relabeled training cache"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "splits": metrics, "policy": result["policy"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
