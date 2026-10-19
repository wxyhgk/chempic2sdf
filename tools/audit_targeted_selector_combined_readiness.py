"""Read-only combined model-logit + PNG-crop separability audit."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
CACHE = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-pool-r1/group-cache.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-combined-readiness-20261006-r1/report.json"
FEATURES = ("atom_score", "presence_logit", "pre_presence_logit", "largest_component", "ink_center")


def auc(positive: list[float], negative: list[float]) -> float:
    values = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank, index = 0.0, 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        rank += ((index + 1 + end) / 2.0) * sum(label for _, label in values[index:end])
        index = end
    return (rank - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    groups = json.loads(CACHE.read_text(encoding="utf-8"))["groups"]
    images: dict[str, Image.Image] = {}
    rows = {"train": [], "holdout": []}
    for group in groups:
        if group["protected"]:
            continue
        image = images.get(group["sample_key"])
        if image is None:
            image = Image.open(DATA / "samples" / (group["sample_key"] + ".png")).convert("L")
            images[group["sample_key"]] = image
        crops = [extract(image, member["feature"][3], member["feature"][4]) for member in group["members"]]
        rows[group["split"]].append({
            "positive": bool(group["strict_positive"]),
            "atom_score": max(member["feature"][0] for member in group["members"]),
            "presence_logit": max(member["feature"][1] for member in group["members"]),
            "pre_presence_logit": max(member["feature"][2] for member in group["members"]),
            "largest_component": max(crop["largest_component_fraction"] for crop in crops),
            "ink_center": max(crop["ink_center_t192"] for crop in crops),
        })
    train = rows["train"]
    normalizers = {}
    for name in FEATURES:
        values = [row[name] for row in train]
        normalizers[name] = (min(values), max(values))
    metrics = {}
    for split, values in rows.items():
        split_metrics = {}
        for name in FEATURES:
            split_metrics[name] = auc([r[name] for r in values if r["positive"]],
                                      [r[name] for r in values if not r["positive"]])
        # Fixed, predeclared equal-weight combinations; no fitted weights or
        # threshold selection is performed in this audit.
        for pair in (("atom_score", "largest_component"), ("presence_logit", "largest_component"),
                     ("atom_score", "ink_center"), ("presence_logit", "ink_center"),
                     ("atom_score", "presence_logit", "largest_component")):
            scores = []
            for row in values:
                total = 0.0
                for name in pair:
                    lo, hi = normalizers[name]
                    total += (row[name] - lo) / max(hi - lo, 1e-9)
                scores.append((total / len(pair), row["positive"]))
            split_metrics["+".join(pair)] = auc([s for s, positive in scores if positive],
                                                 [s for s, positive in scores if not positive])
        metrics[split] = {"groups": len(values), "strict_positive": sum(r["positive"] for r in values),
                          "auc": split_metrics}
    result = {
        "schema_version": "chemvector-targeted-selector-combined-readiness-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "layers_or_graph_read": False,
        "fixed_equal_weight_only": True, "normalizers_fit_from_train_for_diagnostic_only": True,
        "metrics": metrics,
        "gate": {"holdout_combined_auc_minimum_080": any(value >= 0.80 for key, value in metrics["holdout"]["auc"].items() if "+" in key),
                 "training_ready": False,
                 "reason": "separability only; no existence head, abstention calibration, or run authorization"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
