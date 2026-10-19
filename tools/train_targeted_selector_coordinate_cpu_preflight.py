"""CPU-only coordinate-refinement fit on the admitted coordinate release."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_targeted_selector_group_image_features import extract

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-release-audit-20261007-r2/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-cpu-preflight-20261007-r2/report.json"
RIDGE = 10.0


def build_rows(split: str) -> list[dict]:
    cache_name = "train-group-cache.json" if split == "train" else "holdout-group-cache.json"
    cache = json.loads((RELEASE.parent / "chemvector-targeted-selector-visual-pool-release-20261006-r1" / cache_name).read_text(encoding="utf-8"))["groups"]
    targets = json.loads((RELEASE / ("train-refinement-targets.json" if split == "train" else "holdout-refinement-targets.json")).read_text(encoding="utf-8"))["targets"]
    member_by_id = {}
    for group in cache:
        for member in group["members"]:
            member_by_id[(group["group_id"], int(member["slot"]))] = member
    images: dict[str, Image.Image] = {}
    rows = []
    for target in targets:
        member = member_by_id[(target["group_id"], target["slot"])]
        image = images.get(target["sample_key"])
        if image is None:
            image = Image.open(RELEASE / "samples" / (target["sample_key"] + ".png")).convert("L")
            images[target["sample_key"]] = image
        crop = extract(image, member["feature"][3], member["feature"][4])
        crop_names = sorted(crop)
        # Keep a deterministic, raw-PNG-only descriptor plus frozen candidate
        # scalars.  Target coordinates never enter the input features.
        vector = list(member["feature"]) + [crop[name] for name in crop_names]
        rows.append({"vector": vector, "residual": target["residual_px"],
                     "distance_px": target["distance_px"], "status": target["status"],
                     "crop_names": crop_names})
    return rows


def summarize(rows: list[dict], residuals: np.ndarray) -> dict:
    distances = np.sqrt(((np.asarray([r["residual"] for r in rows]) - residuals) ** 2).sum(1))
    return {"count": len(rows), "mean_px": float(distances.mean()), "median_px": float(np.median(distances)),
            "le6": float((distances <= 6).mean()), "le8": float((distances <= 8).mean()),
            "strict_count": sum(r["status"] == "strict_positive" for r in rows),
            "strict_le6": float((distances[[r["status"] == "strict_positive" for r in rows]] <= 6).mean()),
            "near_count": sum(r["status"] == "near_unknown" for r in rows)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Coordinate release audit is not ready")
    train = build_rows("train")
    holdout = build_rows("holdout")
    names = train[0]["crop_names"]
    if any(row["crop_names"] != names for row in train + holdout):
        raise ValueError("Crop feature schema changed")
    x_train = np.asarray([r["vector"] for r in train], dtype=np.float64)
    y_train = np.asarray([r["residual"] for r in train], dtype=np.float64)
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    xn = (x_train - mean) / std
    # Closed-form CPU ridge; only admitted train targets enter the fit.
    gram = xn.T @ xn + RIDGE * np.eye(xn.shape[1])
    beta = np.linalg.solve(gram, xn.T @ y_train)
    x_holdout = np.asarray([r["vector"] for r in holdout], dtype=np.float64)
    baseline_train = np.zeros_like(y_train)
    pred_train = ((x_train - mean) / std) @ beta
    pred_holdout = ((x_holdout - mean) / std) @ beta
    baseline_holdout = np.zeros((len(holdout), 2), dtype=np.float64)
    result = {
        "schema_version": "chemvector-targeted-selector-coordinate-cpu-preflight-0.1",
        "complete": True, "read_only": False, "training_performed": True,
        "training_scope": "cpu_closed_form_ridge_train_targets_only", "gpu_used": False,
        "optimizer_steps": 0, "checkpoint_written": False, "sealed_test_read": False,
        "release_manifest_sha256": hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest(),
        "feature_dim": int(x_train.shape[1]), "ridge": RIDGE,
        "crop_features": names, "train_targets": len(train), "holdout_targets": len(holdout),
        "train_before": summarize(train, baseline_train), "train_after": summarize(train, pred_train),
        "holdout_before": summarize(holdout, baseline_holdout), "holdout_after": summarize(holdout, pred_holdout),
        "gate": {"train_mean_improves": summarize(train, pred_train)["mean_px"] < summarize(train, baseline_train)["mean_px"],
                 "holdout_mean_improves": summarize(holdout, pred_holdout)["mean_px"] < summarize(holdout, baseline_holdout)["mean_px"],
                 "holdout_le6_improves": summarize(holdout, pred_holdout)["le6"] > summarize(holdout, baseline_holdout)["le6"],
                 "holdout_strict_le6_improves": summarize(holdout, pred_holdout)["strict_le6"] > summarize(holdout, baseline_holdout)["strict_le6"],
                 "training_ready": False,
                 "reason": "CPU preflight only; GPU shadow requires a separate coordinate-head receipt"},
    }
    result["gate"]["training_ready"] = all(result["gate"][k] for k in (
        "train_mean_improves", "holdout_mean_improves", "holdout_le6_improves", "holdout_strict_le6_improves"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train_before": result["train_before"],
                      "train_after": result["train_after"], "holdout_before": result["holdout_before"],
                      "holdout_after": result["holdout_after"], "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
