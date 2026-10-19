"""CPU-only train-calibrated conformal abstention diagnostic.

This is deliberately a *post*-gate diagnostic: the frozen candidate gate is
loaded unchanged, and the conformal rule may only abstain from a selected
candidate after coordinate refinement.  It is not a selector and never adds
candidates.  A deterministic image-level split of the released training
images is used to fit a ridge refiner and calibrate a leverage-adjusted
residual quantile; holdout labels are consulted only for the audit table.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.audit_targeted_selector_coordinate_assignment import collect
from tools.prepare_targeted_selector_frozen_coordinate_gate import CandidateHead

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "outputs/audits/chemvector-targeted-selector-frozen-coordinate-gate-20261007-r2"
SIDECAR = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-sidecar-20261007-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-conformal-abstention-20261007-r2/report.json"
RIDGE = 10.0
NEAR_PX = 10.0
STRICT_PX = 6.0
CALIBRATION = 0.90
SPLIT_MOD = 5
SPLIT_REMAINDER = 0  # image hash remainder 0 is calibration; the rest fit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % SPLIT_MOD


def residuals(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[r["atoms"][r["nearest_index"]][0] - r["candidate"][0],
          r["atoms"][r["nearest_index"]][1] - r["candidate"][1]] for r in rows],
        dtype=np.float64,
    )


def fit_ridge(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    near = [r for r in rows if r["raw_distance"] <= NEAR_PX]
    x = np.asarray([r["vector"] for r in near], dtype=np.float64)
    y = residuals(near)
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-5)
    xn = (x - mean) / std
    precision = np.linalg.inv(xn.T @ xn + RIDGE * np.eye(xn.shape[1]))
    beta = precision @ xn.T @ y
    return mean, std, beta, precision


def predictions(rows: list[dict], mean: np.ndarray, std: np.ndarray,
                beta: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (np.asarray([r["vector"] for r in rows], dtype=np.float64) - mean) / std
    correction = x @ beta
    # Ridge leverage is available at inference and adjusts the conformal
    # radius for extrapolation away from the fitting images.
    leverage = np.einsum("ij,jk,ik->i", x, precision, x)
    return correction, np.maximum(1.0 + leverage, 1.0)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    if len(scores) < 20:
        raise ValueError(f"Too few calibration rows: {len(scores)}")
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    # Conservative finite-sample split-conformal quantile.
    rank = int(np.ceil((len(ordered) + 1) * coverage)) - 1
    return float(ordered[min(max(rank, 0), len(ordered) - 1)])


def evaluate(rows: list[dict], selected: np.ndarray, accepted: np.ndarray,
             corrections: np.ndarray, radii: np.ndarray) -> dict:
    labels = np.asarray([r["raw_distance"] <= NEAR_PX for r in rows], dtype=bool)
    strict = np.asarray([r["strict"] for r in rows], dtype=bool)
    strict_distance = np.asarray([r["raw_distance"] <= STRICT_PX for r in rows], dtype=bool)
    refined = np.asarray([r["candidate"] for r in rows], dtype=np.float64) + corrections
    true_nearest = np.asarray([r["atoms"][r["nearest_index"]] for r in rows], dtype=np.float64)
    refined_dist = np.linalg.norm(refined - true_nearest, axis=1)

    def stats(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        return {"selected": n,
                "near_true": int((mask & labels).sum()),
                "strict_true": int((mask & strict).sum()),
                "strict_distance_le6": int((mask & strict_distance).sum()),
                "near_precision": float((mask & labels).sum() / max(n, 1)),
                "near_recall": float((mask & labels).sum() / max(int(labels.sum()), 1)),
                "strict_precision": float((mask & strict).sum() / max(n, 1)),
                "refined_le6": float((mask & (refined_dist <= STRICT_PX)).sum() / max(n, 1)),
                "mean_radius_px": float(radii[mask].mean()) if n else None}

    return {"candidates": len(rows), "available_near": int(labels.sum()),
            "gate": stats(selected), "gate_then_conformal": stats(accepted),
            "abstained_after_gate": int((selected & ~accepted).sum()),
            "accepted_fraction_of_gate": float(accepted.sum() / max(selected.sum(), 1)),
            "radius_le6_fraction_of_gate": float((selected & (radii <= STRICT_PX)).sum() / max(selected.sum(), 1)),
            "calibration_is_post_gate_only": True}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = torch.load(GATE / "candidate-head.pt", map_location="cpu", weights_only=False)
    candidate = CandidateHead(int(payload["feature_dim"]))
    candidate.load_state_dict(payload["state_dict"])
    candidate.eval()
    mean_gate, std_gate = np.asarray(payload["mean"]), np.asarray(payload["std"])

    all_train = collect("train")
    holdout = collect("holdout")
    fit_rows = [r for r in all_train if image_bucket(r["sample_key"]) != SPLIT_REMAINDER]
    cal_rows = [r for r in all_train if image_bucket(r["sample_key"]) == SPLIT_REMAINDER
                and r["raw_distance"] <= NEAR_PX]
    mean, std, beta, precision = fit_ridge(fit_rows)
    cal_corr, cal_lev = predictions(cal_rows, mean, std, beta, precision)
    cal_error = np.linalg.norm(residuals(cal_rows) - cal_corr, axis=1)
    # Calibration is conditional on the frozen gate: this is a post-gate
    # veto, so unselected candidates must not determine its error quantile.
    cal_gate_x = (np.asarray([r["vector"] for r in cal_rows], dtype=np.float64)
                  - mean_gate) / std_gate
    with torch.no_grad():
        cal_gate_selected = candidate(torch.tensor(cal_gate_x, dtype=torch.float32)).numpy() >= 0.0
    scores_all = cal_error / np.sqrt(cal_lev)
    scores = scores_all[cal_gate_selected]
    q = conformal_quantile(scores, CALIBRATION)

    def apply(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = (np.asarray([r["vector"] for r in rows], dtype=np.float64) - mean_gate) / std_gate
        with torch.no_grad():
            logits = candidate(torch.tensor(x, dtype=torch.float32)).numpy()
        selected = logits >= 0.0
        corr, lev = predictions(rows, mean, std, beta, precision)
        radii = q * np.sqrt(lev)
        # The rule only vetoes selected candidates; it never creates a new one.
        accepted = selected & (radii <= STRICT_PX)
        return selected, accepted, corr, radii

    train_selected, train_accepted, train_corr, train_radii = apply(all_train)
    hold_selected, hold_accepted, hold_corr, hold_radii = apply(holdout)
    result = {
        "schema_version": "chemvector-targeted-selector-coordinate-conformal-abstention-0.1",
        "complete": True, "training_performed": True,
        "training_scope": "cpu_train_image_split_ridge_plus_split_conformal_diagnostic",
        "gpu_used": False, "optimizer_steps": 0, "sealed_test_read": False,
        "production_checkpoint_written": False, "deployment_authorized": False,
        "gate_weights_sha256": digest(GATE / "candidate-head.pt"),
        "sidecar_sha256": digest(SIDECAR / "coordinate-refiner.npz"),
        "fit_images": len({r["sample_key"] for r in fit_rows}),
        "calibration_images": len({r["sample_key"] for r in cal_rows}),
        "fit_rows_near": len([r for r in fit_rows if r["raw_distance"] <= NEAR_PX]),
        "calibration_rows_near": len(cal_rows),
        "calibration_rows_selected_near": int(cal_gate_selected.sum()), "ridge": RIDGE,
        "coverage_target": CALIBRATION, "calibration_scope": "frozen_gate_selected_near_train_only",
        "conformal_q_px": q,
        "calibration_score": {"count": len(scores), "mean": float(scores.mean()),
                               "median": float(np.median(scores)), "max": float(scores.max())},
        "train": evaluate(all_train, train_selected, train_accepted, train_corr, train_radii),
        "holdout": evaluate(holdout, hold_selected, hold_accepted, hold_corr, hold_radii),
        "interpretation": {
            "post_gate_only": True,
            "does_not_replace_selector": True,
            "labels_used_for_calibration_only": True,
            "strict_only_is_not_near_recoverable": True,
            "deployment_status": "diagnostic_only_no_production_integration",
            "next": "If acceptance improves strict-only precision at tolerable recall, repeat with a pre-registered image-level calibration split; otherwise discard.",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "q": q, "train": result["train"], "holdout": result["holdout"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
