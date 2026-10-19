"""CPU-only monotonic candidate scorer preflight.

The scorer is deliberately constrained to four interpretable, nonnegative
features.  This tests whether the fixed combined-readiness signal survives a
prior-corrected objective and a fixed top-tail negative penalty.  It is a
diagnostic fit on train only; no GPU authorization or checkpoint promotion is
possible from this report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_hard_negative_preflight import load_candidates

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-monotonic-preflight-20261006-r1/report.json"
FEATURE_INDEX = (0, 1, 6, 7)  # atom, presence, largest-component, ink-center
TAIL_K = 24
STEPS = 200
SEED = 20261006


class MonotonicScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ F.softplus(self.raw_weight) + self.bias


def evaluate(candidates: list[dict], scorer: MonotonicScorer, mean: torch.Tensor, std: torch.Tensor) -> dict:
    scorer.eval()
    with torch.no_grad():
        x = torch.tensor([[c["values"][i] for i in FEATURE_INDEX] for c in candidates])
        scores = scorer((x - mean) / std).tolist()
    rows = [{**c, "score": float(s)} for c, s in zip(candidates, scores)]
    pos = [r for r in rows if r["positive"]]
    emitted = [r for r in rows if r["score"] >= 0.0]
    recovered = sum(r["positive"] for r in emitted)
    false = sum(r["negative"] for r in emitted)
    ordered = sorted(rows, key=lambda r: r["score"], reverse=True)
    topk = {}
    for k in (32, 64, 128):
        chosen = ordered[:k]
        hit = sum(r["positive"] for r in chosen)
        topk[str(k)] = {"selected": len(chosen), "recovered": hit,
                        "precision": hit / max(len(chosen), 1),
                        "recall": hit / max(len(pos), 1)}
    return {"candidates": len(rows), "positive": len(pos), "emitted": len(emitted),
            "strict_recovered": recovered, "confirmed_false": false,
            "conservative_precision": recovered / max(len(emitted), 1),
            "strict_recall": recovered / max(len(pos), 1), "topk": topk}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    torch.manual_seed(SEED)
    train = load_candidates("train-group-cache.json", "train")
    holdout = load_candidates("holdout-group-cache.json", "holdout")
    train_x = torch.tensor([[c["values"][i] for i in FEATURE_INDEX] for c in train], dtype=torch.float32)
    train_y = torch.tensor([float(c["positive"]) for c in train], dtype=torch.float32)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    x = (train_x - mean) / std
    scorer = MonotonicScorer(len(FEATURE_INDEX))
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-2, weight_decay=1e-3)
    positives = train_y > 0.5
    negatives = ~positives
    events = []
    for step in range(1, STEPS + 1):
        logits = scorer(x)
        # Proper BCE uses the observed train prior; there is no artificial
        # pos_weight/intercept shift.  The two tail terms explicitly control
        # the deployment-relevant extreme negatives.
        bce = F.binary_cross_entropy_with_logits(logits, train_y)
        positive_margin = F.softplus(1.0 - logits[positives]).mean()
        tail = torch.topk(logits[negatives], min(TAIL_K, int(negatives.sum()))).values
        tail_loss = F.softplus(tail + 0.5).mean()
        loss = bce + positive_margin + tail_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(scorer.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "bce": float(bce.detach()),
                       "positive_margin": float(positive_margin.detach()), "tail": float(tail_loss.detach()),
                       "gradient_norm": float(norm)})
    train_result = evaluate(train, scorer, mean, std)
    holdout_result = evaluate(holdout, scorer, mean, std)
    result = {
        "schema_version": "chemvector-targeted-selector-monotonic-preflight-0.1",
        "complete": True, "read_only": False, "training_performed": True,
        "training_scope": "cpu_monotonic_four_feature_train_only", "gpu_used": False,
        "optimizer_steps": STEPS, "sealed_test_read": False, "checkpoint_written": False,
        "release_manifest_sha256": hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest(),
        "feature_names": ["atom_score", "presence_logit", "largest_component_fraction", "ink_center_t192"],
        "tail_k": TAIL_K, "events": events, "train": train_result, "holdout": holdout_result,
        "gate": {"train_precision_ge_085": train_result["conservative_precision"] >= 0.85,
                 "train_recall_ge_050": train_result["strict_recall"] >= 0.50,
                 "holdout_precision_ge_085": holdout_result["conservative_precision"] >= 0.85,
                 "holdout_recall_ge_050": holdout_result["strict_recall"] >= 0.50,
                 "training_ready": False,
                 "reason": "CPU-only monotonic preflight; GPU remains unauthorized"},
    }
    result["gate"]["training_ready"] = all(result["gate"][k] for k in (
        "train_precision_ge_085", "train_recall_ge_050", "holdout_precision_ge_085", "holdout_recall_ge_050"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train": train_result, "holdout": holdout_result,
                      "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
