"""Fit and freeze the CPU candidate gate used by the coordinate branch."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_coordinate_assignment import collect

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-release-audit-20261007-r2/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-frozen-coordinate-gate-20261007-r2"
# Must match the independent CPU candidate preflight.  This is part of the
# frozen artifact contract; changing it would make the gate non-reproducible.
SEED = 20261006
STEPS = 300
NEAR_PX = 10.0


class CandidateHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    labels = np.asarray([r["raw_distance"] <= NEAR_PX for r in rows])
    selected = logits >= 0.0
    good = int((selected & labels).sum())
    false = int((selected & ~labels).sum())
    strict = np.asarray([r["strict"] for r in rows])
    return {"candidates": len(rows), "available": int(labels.sum()), "selected": int(selected.sum()),
            "good": good, "false": false, "precision": good / max(int(selected.sum()), 1),
            "recall": good / max(int(labels.sum()), 1), "strict_recovered": int((selected & strict).sum())}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Coordinate release audit is not ready")
    torch.manual_seed(SEED)
    train = collect("train")
    holdout = collect("holdout")
    x_train_raw = np.asarray([r["vector"] for r in train], dtype=np.float32)
    mean = x_train_raw.mean(0)
    std = np.maximum(x_train_raw.std(0), 1e-5)
    x_train = torch.tensor((x_train_raw - mean) / std, dtype=torch.float32)
    y_train = torch.tensor([float(r["raw_distance"] <= NEAR_PX) for r in train], dtype=torch.float32)
    head = CandidateHead(x_train.shape[1]).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        logits = head(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    head.eval()
    with torch.no_grad():
        train_logits = head(x_train).numpy()
        x_holdout = torch.tensor((np.asarray([r["vector"] for r in holdout], dtype=np.float32) - mean) / std)
        holdout_logits = head(x_holdout).numpy()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    weights_path = OUTPUT / "candidate-head.pt"
    torch.save({"state_dict": head.state_dict(), "mean": mean, "std": std,
                "feature_dim": x_train.shape[1], "steps": STEPS, "seed": SEED}, weights_path)
    report = {"schema_version": "chemvector-targeted-selector-frozen-coordinate-gate-0.1",
              "complete": True, "training_performed": True, "training_scope": "cpu_candidate_gate_only",
              "gpu_used": False, "optimizer_steps": STEPS, "sealed_test_read": False,
              "checkpoint_written": False, "production_checkpoint_written": False,
              "release_manifest_sha256": digest(RELEASE / "manifest.json"),
              "weights_sha256": digest(weights_path), "feature_dim": int(x_train.shape[1]),
              "near_threshold_px": NEAR_PX, "fixed_logit_gate": 0.0,
              "train": evaluate(train, train_logits), "holdout": evaluate(holdout, holdout_logits),
              "events": events,
              "gate": {"train_precision_ge_085": evaluate(train, train_logits)["precision"] >= 0.85,
                       "train_recall_ge_050": evaluate(train, train_logits)["recall"] >= 0.50,
                       "holdout_precision_ge_085": evaluate(holdout, holdout_logits)["precision"] >= 0.85,
                       "holdout_recall_ge_050": evaluate(holdout, holdout_logits)["recall"] >= 0.50,
                       "candidate_gate_frozen": True, "gpu_refiner_ready": False}}
    report["gate"]["gpu_refiner_ready"] = all(report["gate"][k] for k in (
        "train_precision_ge_085", "train_recall_ge_050", "holdout_precision_ge_085", "holdout_recall_ge_050"))
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "weights_sha256": report["weights_sha256"],
                      "train": report["train"], "holdout": report["holdout"], "gate": report["gate"]}, indent=2))
    return report


if __name__ == "__main__":
    run()
