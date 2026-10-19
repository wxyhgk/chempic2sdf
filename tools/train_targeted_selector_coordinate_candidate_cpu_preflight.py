"""CPU preflight for a learnable coordinate-refinable-candidate head.

Near-target candidates (nearest atom <=10px) are positive *refinement* cases;
far candidates are negative.  This target is distinct from strict selector
matching and avoids treating coordinate near-misses as confirmed negatives.
The classifier is fit on the admitted train release only, then evaluated once
on holdout.  A separate train-only ridge map supplies coordinate corrections.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_coordinate_assignment import collect, fit as fit_refiner

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-release-audit-20261007-r2/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-candidate-cpu-preflight-20261007-r2/report.json"
NEAR_PX = 10.0
STEPS = 300
SEED = 20261006


class CandidateHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def evaluate(rows: list[dict], logits: np.ndarray, correction: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    labels = np.asarray([r["raw_distance"] <= NEAR_PX for r in rows])
    selected = logits >= 0.0  # fixed posterior 0.5 gate; no threshold sweep
    refinable = int((selected & labels).sum())
    far = int((selected & ~labels).sum())
    strict = np.asarray([r["strict"] for r in rows])
    strict_recovered = int((selected & strict).sum())
    x = np.asarray([r["vector"] for r in rows], dtype=np.float64)
    predicted = ((x - mean) / std) @ correction
    residuals = np.asarray([[r["atoms"][r["nearest_index"]][0] - r["candidate"][0],
                             r["atoms"][r["nearest_index"]][1] - r["candidate"][1]] for r in rows])
    corrected = residuals - predicted
    refined_dist = np.linalg.norm(corrected, axis=1)
    selected_dist = refined_dist[selected]
    return {"candidates": len(rows), "refinable_available": int(labels.sum()),
            "selected": int(selected.sum()), "refinable_selected": refinable,
            "far_selected": far, "strict_recovered": strict_recovered,
            "refinable_precision": refinable / max(int(selected.sum()), 1),
            "refinable_recall": refinable / max(int(labels.sum()), 1),
            "strict_recall": strict_recovered / max(int(strict.sum()), 1),
            "selected_refined_mean_distance_px": float(selected_dist.mean()) if len(selected_dist) else None,
            "selected_refined_le6": float((selected_dist <= 6).mean()) if len(selected_dist) else None}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Coordinate release audit is not ready")
    torch.manual_seed(SEED)
    train = collect("train")
    holdout = collect("holdout")
    x_train = np.asarray([r["vector"] for r in train], dtype=np.float32)
    y_train = torch.tensor([float(r["raw_distance"] <= NEAR_PX) for r in train], dtype=torch.float32)
    mean = x_train.mean(0)
    std = np.maximum(x_train.std(0), 1e-5)
    x_norm = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    head = CandidateHead(x_norm.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        logits = head(x_norm)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    with torch.no_grad():
        train_logits = head(x_norm).numpy()
        x_holdout = np.asarray([r["vector"] for r in holdout], dtype=np.float32)
        holdout_logits = head(torch.tensor((x_holdout - mean) / std, dtype=torch.float32)).numpy()
    _, _, correction = fit_refiner(train)
    train_result = evaluate(train, train_logits, correction, mean, std)
    holdout_result = evaluate(holdout, holdout_logits, correction, mean, std)
    result = {"schema_version": "chemvector-targeted-selector-coordinate-candidate-cpu-preflight-0.1",
              "complete": True, "read_only": False, "training_performed": True,
              "training_scope": "cpu_refinable_candidate_head_train_only", "gpu_used": False,
              "optimizer_steps": STEPS, "checkpoint_written": False, "sealed_test_read": False,
              "release_manifest_sha256": hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest(),
              "near_threshold_px": NEAR_PX, "fixed_logit_gate": 0.0,
              "feature_dim": int(x_norm.shape[1]), "events": events,
              "train": train_result, "holdout": holdout_result,
              "gate": {"train_precision_ge_085": train_result["refinable_precision"] >= 0.85,
                       "train_recall_ge_050": train_result["refinable_recall"] >= 0.50,
                       "holdout_precision_ge_085": holdout_result["refinable_precision"] >= 0.85,
                       "holdout_recall_ge_050": holdout_result["refinable_recall"] >= 0.50,
                       "training_ready": False,
                       "reason": "CPU preflight only; coordinate-head GPU receipt remains separate"}}
    result["gate"]["training_ready"] = all(result["gate"][k] for k in (
        "train_precision_ge_085", "train_recall_ge_050", "holdout_precision_ge_085", "holdout_recall_ge_050"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "holdout": holdout_result,
                      "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
