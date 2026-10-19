"""CPU-only evaluation of coordinate-refined one-to-one assignment.

Fits the same train-only ridge coordinate map as the refinement preflight,
then evaluates a fixed <=6px refined-distance gate with greedy one-to-one
assignment against rendered atom centers.  Ground truth is used only for this
audit; no labels are published or fed to a selector loss.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_targeted_selector_coordinate_release import RELEASE
from tools.audit_targeted_selector_group_image_features import extract

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-assignment-20261007-r2/report.json"
RIDGE = 10.0
DISTANCE_GATE = 6.0


def collect(split: str) -> list[dict]:
    source = RELEASE.parent / "chemvector-targeted-selector-visual-pool-release-20261006-r1"
    cache_name = "train-group-cache.json" if split == "train" else "holdout-group-cache.json"
    groups = json.loads((source / cache_name).read_text(encoding="utf-8"))["groups"]
    images: dict[str, Image.Image] = {}
    atoms: dict[str, list[tuple[float, float]]] = {}
    rows = []
    for group in groups:
        if group["protected"]:
            continue
        key = group["sample_key"]
        if key not in atoms:
            layer = json.loads((source / "samples" / (key + ".layers.json")).read_text(encoding="utf-8"))
            atoms[key] = [(float(a["center"][0]), float(a["center"][1])) for a in layer["atoms"]]
        if key not in images:
            images[key] = Image.open(source / "samples" / (key + ".png")).convert("L")
        for member in group["members"]:
            explicit_strict = member["raw_target"] >= 0 and member["selected_target"] < 0
            explicit_negative = group["confirmed_negative"] and member["raw_target"] < 0 and member["selected_target"] < 0
            # Strict-positive groups may contain additional unknown members;
            # they are neither refinement positives nor negatives and must be
            # excluded from both fitting and evaluation.
            if not (explicit_strict or explicit_negative):
                continue
            x, y = float(member["feature"][3]) * 768.0, float(member["feature"][4]) * 768.0
            distances = [math.hypot(x - ax, y - ay) for ax, ay in atoms[key]]
            index = min(range(len(distances)), key=distances.__getitem__)
            crop = extract(images[key], member["feature"][3], member["feature"][4])
            crop_names = sorted(crop)
            vector = list(member["feature"]) + [crop[name] for name in crop_names]
            rows.append({"sample_key": key, "candidate": [x, y], "atoms": atoms[key],
                         "vector": vector, "nearest_index": index,
                         "strict": bool(explicit_strict),
                         "near_unknown": bool(explicit_negative and distances[index] <= 10.0),
                         "hard_negative": bool(explicit_negative and distances[index] > 10.0),
                         "raw_distance": distances[index]})
    return rows


def fit(train: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    near = [r for r in train if r["raw_distance"] <= 10.0]
    x = np.asarray([r["vector"] for r in near], dtype=np.float64)
    y = np.asarray([[r["atoms"][r["nearest_index"]][0] - r["candidate"][0],
                     r["atoms"][r["nearest_index"]][1] - r["candidate"][1]] for r in near], dtype=np.float64)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    xn = (x - mean) / std
    beta = np.linalg.solve(xn.T @ xn + RIDGE * np.eye(xn.shape[1]), xn.T @ y)
    return mean, std, beta


def evaluate(rows: list[dict], mean: np.ndarray, std: np.ndarray, beta: np.ndarray) -> dict:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        correction = ((np.asarray(row["vector"], dtype=np.float64) - mean) / std) @ beta
        refined = np.asarray(row["candidate"], dtype=np.float64) + correction
        distances = [float(np.linalg.norm(refined - np.asarray(atom))) for atom in row["atoms"]]
        row = {**row, "refined": refined.tolist(), "refined_distance": min(distances),
               "refined_index": int(np.argmin(distances))}
        by_image[row["sample_key"]].append(row)
    assigned = []
    for image_rows in by_image.values():
        used: set[int] = set()
        for row in sorted(image_rows, key=lambda r: r["refined_distance"]):
            idx = row["refined_index"]
            if row["refined_distance"] <= DISTANCE_GATE and idx not in used:
                used.add(idx)
                assigned.append(row)
    true = sum(r["strict"] or r["near_unknown"] for r in assigned)
    strict = sum(r["strict"] for r in assigned)
    near = sum(r["near_unknown"] for r in assigned)
    false = sum(r["hard_negative"] for r in assigned)
    return {"images": len(by_image), "assigned": len(assigned), "true_strict_or_near": true,
            "strict_recovered": strict, "near_unknown_assigned": near, "hard_false_assigned": false,
            "precision_if_near_is_recoverable": true / max(len(assigned), 1),
            "strict_precision_only": strict / max(len(assigned), 1),
            "assignment_cap": "one candidate per rendered atom", "distance_gate_px": DISTANCE_GATE}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train = collect("train")
    holdout = collect("holdout")
    mean, std, beta = fit(train)
    train_result = evaluate(train, mean, std, beta)
    holdout_result = evaluate(holdout, mean, std, beta)
    result = {"schema_version": "chemvector-targeted-selector-coordinate-assignment-0.1",
              "complete": True, "read_only": False, "training_performed": True,
              "training_scope": "cpu_train_only_coordinate_ridge_then_fixed_assignment_audit",
              "gpu_used": False, "optimizer_steps": 0, "sealed_test_read": False,
              "checkpoint_written": False, "ridge": RIDGE, "distance_gate_px": DISTANCE_GATE,
              "train": train_result, "holdout": holdout_result,
              "gate": {"holdout_precision_if_near_recoverable_ge_085": holdout_result["precision_if_near_is_recoverable"] >= 0.85,
                       "holdout_strict_recovery_ge_50": holdout_result["strict_recovered"] >= 50,
                       "training_ready": False,
                       "reason": "assignment audit only; GPU coordinate-head receipt still required"}}
    result["gate"]["training_ready"] = all(result["gate"][k] for k in (
        "holdout_precision_if_near_recoverable_ge_085", "holdout_strict_recovery_ge_50"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "holdout": holdout_result,
                      "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
