"""CPU-only CVaR/top-tail rank preflight for the next selector objective.

This is a fresh objective diagnostic, not a retry of the failed GPU shadow.
It directly penalizes the highest-scoring confirmed negatives rather than
relying on a class-weighted BCE intercept.  Train-only fitting uses fixed
same-image hard pairs and a fixed negative-tail budget; holdout is evaluated
once with score>=0 and top-8 winner abstention left outside production.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_hard_negative_preflight import (
    HARD_NEG_K, MARGIN, LinearRanker, load_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-cvar-rank-preflight-20261006-r1/report.json"
TAIL_K = 32
STEPS = 200
SEED = 20261006


def evaluate(candidates: list[dict], ranker: LinearRanker, mean: torch.Tensor, std: torch.Tensor) -> dict:
    ranker.eval()
    with torch.no_grad():
        scores = ranker((torch.tensor([c["values"] for c in candidates]) - mean) / std).tolist()
    rows = [{**c, "score": float(s)} for c, s in zip(candidates, scores)]
    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if r["negative"]]
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
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["sample_key"]].append(row)
    rank8 = []
    for image_rows in by_image.values():
        p = [r for r in image_rows if r["positive"]]
        if p:
            rank8.append(any(r in sorted(image_rows, key=lambda x: x["score"], reverse=True)[:8] for r in p))
    return {"candidates": len(rows), "positive": len(pos), "negative": len(neg),
            "emitted": len(emitted), "strict_recovered": recovered,
            "confirmed_false": false, "conservative_precision": recovered / max(len(emitted), 1),
            "strict_recall": recovered / max(len(pos), 1),
            "topk": topk, "positive_image_rank_top8": sum(rank8) / max(len(rank8), 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    torch.manual_seed(SEED)
    train = load_candidates("train-group-cache.json", "train")
    holdout = load_candidates("holdout-group-cache.json", "holdout")
    train_values = torch.tensor([c["values"] for c in train], dtype=torch.float32)
    mean, std = train_values.mean(0), train_values.std(0).clamp_min(1e-5)
    pos = [c for c in train if c["positive"]]
    neg = [c for c in train if c["negative"]]
    negatives_by_image: dict[str, list[dict]] = defaultdict(list)
    for c in neg:
        negatives_by_image[c["sample_key"]].append(c)
    pair_pos, pair_neg = [], []
    for p in pos:
        pool = sorted(negatives_by_image[p["sample_key"]], key=lambda c: c["values"][0], reverse=True)[:HARD_NEG_K]
        for n in pool:
            pair_pos.append(p["values"])
            pair_neg.append(n["values"])
    pair_pos = (torch.tensor(pair_pos) - mean) / std
    pair_neg = (torch.tensor(pair_neg) - mean) / std
    x_pos = (torch.tensor([c["values"] for c in pos]) - mean) / std
    x_neg = (torch.tensor([c["values"] for c in neg]) - mean) / std
    ranker = LinearRanker(x_pos.shape[1])
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=1e-2, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        score_pos = ranker(x_pos)
        score_neg_pair = ranker(pair_neg)
        score_pair_pos = ranker(pair_pos)
        # The tail is recomputed from current scores; it is a fixed CVaR
        # budget, not a threshold sweep.
        score_neg_all = ranker(x_neg)
        tail = torch.topk(score_neg_all, min(TAIL_K, score_neg_all.numel())).values
        loss_pos = F.softplus(MARGIN - score_pos).mean()
        loss_pair = F.softplus(MARGIN + score_neg_pair - score_pair_pos).mean()
        loss_tail = F.softplus(tail + 0.5).mean()
        loss = loss_pos + loss_pair + loss_tail
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(ranker.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "positive": float(loss_pos.detach()),
                       "pair": float(loss_pair.detach()), "tail": float(loss_tail.detach()),
                       "gradient_norm": float(norm)})
    train_result = evaluate(train, ranker, mean, std)
    holdout_result = evaluate(holdout, ranker, mean, std)
    result = {
        "schema_version": "chemvector-targeted-selector-cvar-rank-preflight-0.1",
        "complete": True, "read_only": False, "training_performed": True,
        "training_scope": "cpu_linear_candidate_ranker_train_only", "gpu_used": False,
        "optimizer_steps": STEPS, "sealed_test_read": False, "checkpoint_written": False,
        "release_manifest_sha256": hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest(),
        "hard_negative_per_positive": HARD_NEG_K, "tail_k": TAIL_K, "margin": MARGIN,
        "pair_count": len(pair_pos), "events": events, "train": train_result, "holdout": holdout_result,
        "gate": {"train_precision_ge_085": train_result["conservative_precision"] >= 0.85,
                 "train_recall_ge_050": train_result["strict_recall"] >= 0.50,
                 "holdout_precision_ge_085": holdout_result["conservative_precision"] >= 0.85,
                 "holdout_recall_ge_050": holdout_result["strict_recall"] >= 0.50,
                 "training_ready": False,
                 "reason": "CPU-only CVaR objective preflight; GPU requires every fixed gate"},
    }
    result["gate"]["training_ready"] = all(result["gate"][k] for k in (
        "train_precision_ge_085", "train_recall_ge_050", "holdout_precision_ge_085", "holdout_recall_ge_050"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "pair_count": len(pair_pos),
                      "train": train_result, "holdout": holdout_result, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
