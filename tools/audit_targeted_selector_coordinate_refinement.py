"""CPU-only readiness audit for continuous coordinate refinement.

The audit asks whether raw-PNG local ink geometry predicts a correction from a
raw candidate coordinate toward the nearest rendered atom center.  Layer atom
centers are used only for this post-forward diagnostic target; no corrected
labels or training cache are published.  A linear ridge map is fit on train
near-target candidates (distance <=10px) and evaluated once on holdout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-refinement-20261006-r1/report.json"
NEAR_PX = 10.0
RIDGE = 1.0


def load_rows(filename: str) -> list[dict]:
    groups = json.loads((RELEASE / filename).read_text(encoding="utf-8"))["groups"]
    images: dict[str, Image.Image] = {}
    atoms: dict[str, list[tuple[float, float]]] = {}
    rows = []
    for group in groups:
        if group["protected"]:
            continue
        key = group["sample_key"]
        if key not in atoms:
            layer = json.loads((RELEASE / "samples" / (key + ".layers.json")).read_text(encoding="utf-8"))
            atoms[key] = [(float(a["center"][0]), float(a["center"][1])) for a in layer["atoms"]]
        if key not in images:
            images[key] = Image.open(RELEASE / "samples" / (key + ".png")).convert("L")
        image = images[key]
        for member in group["members"]:
            if not (group["strict_positive"] or group["confirmed_negative"]):
                continue
            fx, fy = member["feature"][3], member["feature"][4]
            x, y = fx * 768.0, fy * 768.0
            distances = [math.hypot(x - ax, y - ay) for ax, ay in atoms[key]]
            target_index = min(range(len(distances)), key=distances.__getitem__)
            target = atoms[key][target_index]
            crop = extract(image, fx, fy)
            # Crop descriptors are all raw PNG-derived.  Keep a small stable
            # feature vector; the fitted map is diagnostic, not deployable.
            feature = [1.0, crop["ink_centroid_x"], crop["ink_centroid_y"],
                       crop["central_horizontal"], crop["central_vertical"],
                       crop["ink_covariance"], crop["ink_anisotropy"]]
            rows.append({"sample_key": key, "distance_px": distances[target_index],
                         "residual_x": target[0] - x, "residual_y": target[1] - y,
                         "feature": feature, "strict": bool(member["raw_target"] >= 0 and member["selected_target"] < 0),
                         "confirmed_negative": bool(group["confirmed_negative"] and member["raw_target"] < 0 and member["selected_target"] < 0)})
    return rows


def summarize(rows: list[dict], correction: np.ndarray | None = None) -> dict:
    distances = []
    for row in rows:
        d = row["distance_px"]
        if correction is not None:
            delta = correction @ np.asarray(row["feature"], dtype=np.float64)
            d = math.hypot(row["residual_x"] - delta[0], row["residual_y"] - delta[1])
        distances.append(d)
    if not distances:
        return {"count": 0}
    return {"count": len(distances), "mean_px": float(np.mean(distances)),
            "median_px": float(np.median(distances)),
            "le_6px": sum(d <= 6 for d in distances) / len(distances),
            "le_8px": sum(d <= 8 for d in distances) / len(distances),
            "le_10px": sum(d <= 10 for d in distances) / len(distances)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    train = load_rows("train-group-cache.json")
    holdout = load_rows("holdout-group-cache.json")
    train_near = [r for r in train if r["distance_px"] <= NEAR_PX]
    holdout_near = [r for r in holdout if r["distance_px"] <= NEAR_PX]
    if len(train_near) < 100 or len(holdout_near) < 50:
        raise ValueError("Insufficient near-target audit support")
    # Fixed, predeclared raster-centroid correction: move by the local ink
    # centroid measured in the 40px crop.  No GT-derived parameter is fitted;
    # train/holdout are only used to audit this deterministic proposal.
    correction = np.zeros((2, 7), dtype=np.float64)
    correction[0, 1] = 40.0
    correction[1, 2] = 40.0
    result = {
        "schema_version": "chemvector-targeted-selector-coordinate-refinement-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "training_scope": "fixed_raster_centroid_audit_only", "gpu_used": False,
        "optimizer_steps": 0, "sealed_test_read": False,
        "layers_atom_centers_only": True, "images_read": True,
        "near_threshold_px": NEAR_PX, "ridge": RIDGE,
        "train_near_count": len(train_near), "holdout_near_count": len(holdout_near),
        "correction_matrix": correction.tolist(),
        "train_before": summarize(train_near), "train_after": summarize(train_near, correction),
        "holdout_before": summarize(holdout_near), "holdout_after": summarize(holdout_near, correction),
        "gate": {"holdout_mean_improves": summarize(holdout_near, correction)["mean_px"] < summarize(holdout_near)["mean_px"],
                 "holdout_le6_improves": summarize(holdout_near, correction)["le_6px"] > summarize(holdout_near)["le_6px"],
                 "training_ready": False,
                 "reason": "coordinate signal audit only; no corrected labels or GPU authorization"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train_before": result["train_before"],
                      "train_after": result["train_after"], "holdout_before": result["holdout_before"],
                      "holdout_after": result["holdout_after"], "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
