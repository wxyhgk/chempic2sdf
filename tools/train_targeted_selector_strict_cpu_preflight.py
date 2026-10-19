"""CPU-only strict selector preflight.

Only explicit strict members and confirmed far negatives enter the loss.
Near-unknown members are masked from training and counted as conservative
false selections at evaluation.  This is a diagnostic preflight, not a
production checkpoint or a threshold sweep.
"""

from __future__ import annotations

import hashlib
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
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-strict-cpu-preflight-20261007-r1/report.json"
SEED = 20261009
STEPS = 300


class StrictHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    strict = np.asarray([r["strict"] for r in rows], dtype=bool)
    near = np.asarray([r["near_unknown"] for r in rows], dtype=bool)
    hard = np.asarray([r["hard_negative"] for r in rows], dtype=bool)
    selected = logits >= 0.0
    true = int((selected & strict).sum())
    # Conservative denominator includes near-unknowns: they are not accepted.
    conservative_false = int((selected & ~strict).sum())
    return {
        "candidates": len(rows), "strict_positive": int(strict.sum()),
        "near_unknown": int(near.sum()), "hard_negative": int(hard.sum()),
        "selected": int(selected.sum()), "strict_selected": true,
        "near_unknown_selected": int((selected & near).sum()),
        "hard_negative_selected": int((selected & hard).sum()),
        "conservative_false_selected": conservative_false,
        "strict_precision_conservative": true / max(int(selected.sum()), 1),
        "strict_recall": true / max(int(strict.sum()), 1),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Release audit is not ready")
    torch.manual_seed(SEED)
    train_all, holdout = collect("train"), collect("holdout")
    train = [r for r in train_all if r["strict"] or r["hard_negative"]]
    x_raw = np.asarray([r["vector"] for r in train], dtype=np.float32)
    mean, std = x_raw.mean(0), np.maximum(x_raw.std(0), 1e-5)
    x = torch.tensor((x_raw - mean) / std, dtype=torch.float32)
    y = torch.tensor([float(r["strict"]) for r in train], dtype=torch.float32)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    weights = torch.where(y > 0.5, torch.tensor(len(y) / (2 * n_pos)),
                         torch.tensor(len(y) / (2 * n_neg)))
    head = StrictHead(x.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        logits = head(x)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    head.eval()
    with torch.no_grad():
        train_logits = head(torch.tensor((np.asarray([r["vector"] for r in train_all], dtype=np.float32) - mean) / std)).numpy()
        holdout_logits = head(torch.tensor((np.asarray([r["vector"] for r in holdout], dtype=np.float32) - mean) / std)).numpy()
    train_result, holdout_result = evaluate(train_all, train_logits), evaluate(holdout, holdout_logits)
    result = {
        "schema_version": "chemvector-targeted-selector-strict-cpu-preflight-0.1",
        "complete": True, "training_performed": True, "training_scope": "strict_vs_confirmed_far_negative_cpu_only",
        "gpu_used": False, "optimizer_steps": STEPS, "checkpoint_written": False,
        "sealed_test_read": False, "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "seed": SEED, "fixed_logit_gate": 0.0, "feature_dim": int(x.shape[1]),
        "train_loss_rows": len(train), "train_strict_rows": n_pos, "train_hard_negative_rows": n_neg,
        "near_unknown_training_rows": 0, "events": events,
        "train": train_result, "holdout": holdout_result,
        "gate": {
            "train_precision_ge_085": train_result["strict_precision_conservative"] >= 0.85,
            "train_recall_ge_050": train_result["strict_recall"] >= 0.50,
            "holdout_precision_ge_085": holdout_result["strict_precision_conservative"] >= 0.85,
            "holdout_recall_ge_050": holdout_result["strict_recall"] >= 0.50,
            "strict_training_ready": False,
            "reason": "CPU diagnostic only; no GPU or production authorization",
        },
    }
    result["gate"]["strict_training_ready"] = all(result["gate"][key] for key in (
        "train_precision_ge_085", "train_recall_ge_050", "holdout_precision_ge_085", "holdout_recall_ge_050"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "holdout": holdout_result, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
