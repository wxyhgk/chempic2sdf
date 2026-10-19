"""Read-only readiness audit for richer raw-PNG crop features.

This diagnostic never trains or loads a checkpoint.  It extracts the existing
image-only raster descriptors around frozen candidate coordinates, reports
per-feature AUC, and evaluates fixed, train-normalized combinations with the
frozen atom score.  Labels are attached only after extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-rich-visual-readiness-20261006-r1/report.json"


def auc(positive: list[float], negative: list[float]) -> float:
    values = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        rank += ((i + 1 + j) / 2.0) * sum(label for _, label in values[i:j])
        i = j
    return (rank - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    rows = {}
    images: dict[str, Image.Image] = {}
    for split, filename in (("train", "train-group-cache.json"), ("holdout", "holdout-group-cache.json")):
        cache = json.loads((RELEASE / filename).read_text(encoding="utf-8"))["groups"]
        values = []
        for group in cache:
            if group["protected"]:
                continue
            image = images.get(group["sample_key"])
            if image is None:
                image = Image.open(RELEASE / "samples" / (group["sample_key"] + ".png")).convert("L")
                images[group["sample_key"]] = image
            crops = [extract(image, m["feature"][3], m["feature"][4]) for m in group["members"]]
            # Group-level max preserves the candidate existence interpretation.
            row = {"positive": bool(group["strict_positive"]),
                   "atom_score": max(m["feature"][0] for m in group["members"])}
            for name in crops[0]:
                row[name] = max(c[name] for c in crops)
            values.append(row)
        rows[split] = values
    names = [name for name in rows["train"][0] if name != "positive"]
    individual = {}
    for name in names:
        individual[name] = {split: auc([r[name] for r in rows[split] if r["positive"]],
                                       [r[name] for r in rows[split] if not r["positive"]])
                             for split in rows}
    # Fixed, predeclared combinations. Normalizers are fit on train only for
    # this diagnostic; signs are chosen from train AUC and then frozen.
    combos = {
        "atom_plus_largest": ("atom_score", "largest_component_fraction"),
        "atom_plus_total_ink": ("atom_score", "total_ink_fraction"),
        "atom_plus_center192": ("atom_score", "ink_center_t192"),
        "atom_plus_quadrants": ("atom_score", "quadrant_nw", "quadrant_ne", "quadrant_sw", "quadrant_se"),
        "atom_plus_shape": ("atom_score", "ink_anisotropy", "central_horizontal", "central_vertical"),
    }
    combo_metrics = {}
    normalizers = {}
    for name in names:
        vals = [r[name] for r in rows["train"]]
        normalizers[name] = (min(vals), max(vals))
    for combo_name, combo in combos.items():
        signs = {name: 1.0 if individual[name]["train"] >= 0.5 else -1.0 for name in combo}
        scores = {}
        for split in rows:
            scores[split] = []
            for row in rows[split]:
                total = 0.0
                for name in combo:
                    lo, hi = normalizers[name]
                    total += signs[name] * ((row[name] - lo) / max(hi - lo, 1e-9))
                scores[split].append((total / len(combo), row["positive"]))
        combo_metrics[combo_name] = {
            "features": combo,
            "train_signs": signs,
            "auc": {split: auc([s for s, p in scores[split] if p],
                                [s for s, p in scores[split] if not p]) for split in rows},
        }
    result = {
        "schema_version": "chemvector-targeted-selector-rich-visual-readiness-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "layers_or_graph_read": False,
        "groups": {split: len(values) for split, values in rows.items()},
        "strict_positive": {split: sum(r["positive"] for r in values) for split, values in rows.items()},
        "individual_auc": individual, "fixed_combo_auc": combo_metrics,
        "gate": {"holdout_combo_auc_minimum_095": any(v["auc"]["holdout"] >= 0.95 for v in combo_metrics.values()),
                 "training_ready": False,
                 "reason": "separability only; no scorer or threshold was trained"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"], "fixed_combo_auc": combo_metrics}, indent=2))
    return result


if __name__ == "__main__":
    run()
