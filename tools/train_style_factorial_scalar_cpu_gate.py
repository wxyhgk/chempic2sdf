"""Fixed CPU preflight for the style-factorial candidate cache.

This is intentionally a small diagnostic gate, not a deployment model.  It
uses only frozen-model scalar/geometry features (raw score, set presence logit,
and candidate x/y), trains on strict positives plus confirmed hard negatives,
and masks near-unknown rows.  A single zero-logit decision is evaluated once;
there is no threshold sweep and no GPU/checkpoint write.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
AUDIT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/cache-audit.json"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-scalar-cpu-gate-20261010-r1/report.json"
SEED = 20261012
STEPS = 300
FEATURES = ("raw_score", "presence_logit", "x_norm", "y_norm")


class Head(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[list[dict], list[dict]]:
    train, dev = [], []
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["replay_split"] == "train_core":
                # near_unknown and protected/unknown rows never enter loss.
                if row["label"] in ("strict_positive", "hard_negative"):
                    train.append(row)
            elif row["replay_split"] in ("dev_core", "dev_stress"):
                dev.append(row)
    return train, dev


def evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    labels = np.asarray([r["label"] == "strict_positive" for r in rows], dtype=bool)
    near = np.asarray([r["label"] == "near_unknown" for r in rows], dtype=bool)
    hard = np.asarray([r["label"] == "hard_negative" for r in rows], dtype=bool)
    selected = logits >= 0.0
    strict_selected = int((selected & labels).sum())
    return {
        "candidates": len(rows), "strict_positive": int(labels.sum()),
        "near_unknown": int(near.sum()), "hard_negative": int(hard.sum()),
        "selected": int(selected.sum()), "strict_selected": strict_selected,
        "near_unknown_selected": int((selected & near).sum()),
        "hard_negative_selected": int((selected & hard).sum()),
        "strict_precision_conservative": strict_selected / max(int(selected.sum()), 1),
        "strict_recall": strict_selected / max(int(labels.sum()), 1),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not (audit["gate"]["jsonl_complete_and_valid"] and audit["gate"]["matches_forward_aggregate"]):
        raise ValueError("candidate cache audit is not complete")
    torch.manual_seed(SEED)
    train_loss, dev = load()
    if not train_loss or not dev:
        raise ValueError("cache has no train/dev rows")
    x_train = np.asarray([[float(r[name]) for name in FEATURES] for r in train_loss], dtype=np.float32)
    y_train = np.asarray([r["label"] == "strict_positive" for r in train_loss], dtype=np.float32)
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    x = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    model = Head(len(FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    with torch.no_grad():
        train_logits = model(x).numpy()
        dev_x = np.asarray([[float(r[name]) for name in FEATURES] for r in dev], dtype=np.float32)
        dev_logits = model(torch.tensor((dev_x - mean) / std)).numpy()
    train_result = evaluate(train_loss, train_logits)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row, logit in zip(dev, dev_logits):
        by_variant[row["variant"]].append({"row": row, "logit": float(logit)})
    dev_variant = {variant: evaluate([x["row"] for x in values], np.asarray([x["logit"] for x in values]))
                   for variant, values in sorted(by_variant.items())}
    result = {
        "schema_version": "chemvector-style-factorial-scalar-cpu-gate-0.1",
        "complete": True, "training_performed": True, "training_scope": "frozen_scalar_candidate_cpu_only",
        "gpu_used": False, "optimizer_steps": STEPS, "checkpoint_written": False,
        "sealed_test_read": False, "cache_sha256": digest(CACHE), "seed": SEED,
        "features": list(FEATURES), "fixed_logit_gate": 0.0,
        "train_loss_rows": len(train_loss), "near_unknown_training_rows": 0,
        "train": train_result, "dev_by_variant": dev_variant, "events": events,
    }
    core = [dev_variant[v] for v in ("coordgen", "original", "rdkit")]
    result["gate"] = {
        "each_core_precision_ge_085": all(x["strict_precision_conservative"] >= .85 for x in core),
        "each_core_recall_ge_050": all(x["strict_recall"] >= .50 for x in core),
        "stress_reported_separately": "stress" in dev_variant,
        "training_ready": False,
        "reason": "CPU scalar gate only; no GPU authorization",
    }
    result["gate"]["training_ready"] = (result["gate"]["each_core_precision_ge_085"] and
                                          result["gate"]["each_core_recall_ge_050"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result,
                      "dev_by_variant": dev_variant, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
