#!/usr/bin/env python3
"""CPU-only probe for visual information in the frozen candidate union.

This is an audit, not a production path.  It consumes the already materialised
r17 pair cache and asks a narrow question: once the frozen atom/pair candidate
union is fixed, do the pair hidden features contain enough information to rank
true molecular edges?  Three frozen-feature groups are compared:

* ``visual_hidden``: the 256-dimensional pair hidden output;
* ``geometry_aux``: presence logit, midpoint, distance and endpoint scores;
* ``full``: their concatenation.

A tiny linear readout is fit on train molecules only.  Holdout is evaluated at
the per-image candidate budget used by the frozen presence head (top-k), so the
comparison does not quietly trade precision for an arbitrarily larger edge
set.  No checkpoint, sealed test, GPU, or production artifact is touched.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_r17_pair_incidence_known_clean_cpu import SHA

CACHE = ROOT / os.environ.get(
    "R17_VISUAL_PROBE_CACHE",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt",
)
OUT = ROOT / os.environ.get(
    "R17_VISUAL_PROBE_OUT",
    "outputs/audits/chemvector-r17-candidate-union-visual-probe-cpu-20261018-t2",
)
STEPS = int(os.environ.get("R17_VISUAL_PROBE_STEPS", "300"))
BASE_THRESHOLD = 0.72
SEED = 20261018


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out = nn.Linear(dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x).squeeze(-1)


def groups(features: torch.Tensor) -> dict[str, torch.Tensor]:
    if features.ndim != 2 or features.size(1) < 262:
        raise ValueError(f"unexpected pair feature shape: {tuple(features.shape)}")
    # train_r17_pair_incidence_known_clean_cpu.py concatenates:
    # hidden(256), presence(1), midpoint(2), distance(1), endpoint scores(2).
    return {
        "visual_hidden": features[:, :256],
        "geometry_aux": features[:, 256:],
        "full": features[:, :262],
    }


def rows(items: list[dict], group: str, train: bool):
    xs, ys, keys = [], [], []
    for item in items:
        g = groups(item["features"])[group]
        label = (item["labels"] > 0.5).float()
        # Keep every candidate for evaluation.  Training is balanced per image
        # to avoid the many non-edge pairs dominating the readout.
        if train:
            pos = torch.where(label > .5)[0]
            neg = torch.where(label <= .5)[0]
            if not len(pos) or not len(neg):
                continue
            n = min(len(pos), len(neg))
            # Deterministic hard-negative subset: closest negatives by the
            # frozen presence score, not target information.
            order = torch.argsort(item["base_logits"][neg], descending=True)[:n]
            take = torch.cat((pos, neg[order]))
        else:
            take = torch.arange(len(label))
        xs.append(g[take].float()); ys.append(label[take]); keys.extend([item["key"]] * len(take))
    if not xs:
        raise RuntimeError(f"no rows for {group}")
    return torch.cat(xs), torch.cat(ys), keys


def standardize(train_x: torch.Tensor, x: torch.Tensor):
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    return (train_x - mean) / scale, (x - mean) / scale, mean, scale


def auc_ap(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    pos = int(y.sum()); neg = int(len(y) - pos)
    if not pos or not neg:
        return 0.0, 0.0
    # Mann–Whitney AUC; ties receive their average rank.
    ranks = torch.empty_like(order, dtype=torch.float32)
    sorted_scores = scores[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    rank_pos = ranks[y > .5].sum().item()
    # Ranks are descending (rank 1 is the highest score), hence invert the
    # usual ascending-rank Mann–Whitney expression.
    auc = (pos * neg - (rank_pos - pos * (pos + 1) / 2)) / max(pos * neg, 1)
    tp = torch.cumsum(y, 0)
    precision = tp / torch.arange(1, len(y) + 1, dtype=torch.float32)
    ap = float((precision * y).sum() / max(pos, 1))
    return float(auc), ap


def topk_metrics(items: list[dict], scores_by_key: dict[str, torch.Tensor]):
    total = defaultdict(float); by = defaultdict(lambda: defaultdict(float))
    for item in items:
        score = scores_by_key[item["key"]]
        # Frozen candidate budget: retain exactly as many candidates as the
        # baseline presence threshold, capped by available candidates.
        base_logit = math.log(BASE_THRESHOLD / (1 - BASE_THRESHOLD))
        k = int((item["base_logits"] >= base_logit).sum())
        k = min(max(k, 0), len(score))
        chosen = torch.topk(score, k=k).indices if k else torch.empty(0, dtype=torch.long)
        labels = item["labels"]
        true = float(labels[chosen].sum())
        target = float(labels.sum())
        rec = true / max(target, 1.0); prec = true / max(k, 1)
        bucket = by[item["profile"]]
        for name, value in (("images", 1), ("predicted", k), ("true", true), ("target", target)):
            total[name] += value; bucket[name] += value
        bucket["recall_sum"] += rec; bucket["precision_sum"] += prec
    def finish(c):
        return {
            "images": int(c["images"]), "predicted_edges": int(c["predicted"]),
            "true_edges": int(c["true"]), "target_edges": int(c["target"]),
            "edge_precision": c["true"] / max(c["predicted"], 1),
            "edge_recall": c["true"] / max(c["target"], 1),
            "macro_precision": c["precision_sum"] / max(c["images"], 1),
            "macro_recall": c["recall_sum"] / max(c["images"], 1),
        }
    # Add aggregate macro values for total from per-profile accumulators.
    total["precision_sum"] = sum(c["precision_sum"] for c in by.values())
    total["recall_sum"] = sum(c["recall_sum"] for c in by.values())
    return finish(total), {p: finish(c) for p, c in sorted(by.items())}


def residual_metrics(items: list[dict], scores_by_key: dict[str, torch.Tensor]):
    """Measure the hard subset rejected by the frozen presence threshold.

    ``oracle_topk`` uses the number of true residual edges as k; it is not a
    deployable operating point, but cleanly answers whether visual features
    can rank the missed edges at all.  The threshold-free AUC/AP are the more
    conservative readout of separability.
    """
    base_logit = math.log(BASE_THRESHOLD / (1 - BASE_THRESHOLD))
    all_scores, all_labels = [], []
    total = defaultdict(float); by = defaultdict(lambda: defaultdict(float))
    for item in items:
        mask = item["base_logits"] < base_logit
        score = scores_by_key[item["key"]][mask]
        label = item["labels"][mask]
        all_scores.append(score); all_labels.append(label)
        pos = int(label.sum()); k = min(pos, len(label))
        chosen = torch.topk(score, k=k).indices if k else torch.empty(0, dtype=torch.long)
        true = float(label[chosen].sum())
        c = by[item["profile"]]
        for name, value in (("rows", len(label)), ("positive", pos), ("oracle_true", true), ("oracle_k", k)):
            total[name] += value; c[name] += value
    if not all_scores:
        return {"rows": 0, "positive": 0, "auc": 0.0, "average_precision": 0.0}, {}
    scores = torch.cat(all_scores); labels = torch.cat(all_labels)
    auc, ap = auc_ap(scores, labels)
    def finish(c):
        return {"rows": int(c["rows"]), "positive": int(c["positive"]),
                "oracle_topk_recall": c["oracle_true"] / max(c["positive"], 1),
                "oracle_topk_precision": c["oracle_true"] / max(c["oracle_k"], 1)}
    return {"rows": int(len(labels)), "positive": int(labels.sum()), "auc": auc,
            "average_precision": ap, "oracle_topk_recall": total["oracle_true"] / max(total["positive"], 1),
            "oracle_topk_precision": total["oracle_true"] / max(total["oracle_k"], 1)}, {p: finish(c) for p, c in sorted(by.items())}


def run():
    if (OUT / "report.json").exists():
        raise FileExistsError(OUT / "report.json")
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint changed")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    cache = torch.load(CACHE, map_location="cpu")
    train, hold = cache["train"], cache["hold"]
    report = {
        "schema_version": "chemvector-r17-candidate-union-visual-probe-cpu-0.1",
        "complete": True, "training_performed": True,
        "training_scope": "frozen_pair_feature_linear_probe_cpu",
        "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)), "cache_sha256": digest(CACHE),
        "initial_checkpoint_sha256": SHA,
        "candidate_union": {}, "groups": {}, "gate": {},
    }
    # Candidate ceiling is independent of any readout: if this is low, no
    # downstream scorer can recover the missing edge from this union.
    for split_name, items in (("train", train), ("holdout", hold)):
        target = sum(float(i["labels"].sum()) for i in items)
        target_total = sum(len(i.get("target_edges", {})) for i in items)
        report["candidate_union"][split_name] = {
            "images": len(items),
            "candidate_pairs": sum(len(i["labels"]) for i in items),
            "reachable_true_edges": int(target),
            "target_edges": int(target_total),
            "edge_recall_ceiling_against_target_edges": target / max(target_total, 1),
            "edge_recall_ceiling_against_mapped_labels": 1.0 if target else 0.0,
        }
    for name in ("visual_hidden", "geometry_aux", "full"):
        tx, ty, _ = rows(train, name, True)
        hx, hy, hkeys = rows(hold, name, False)
        tx, hx, mean, scale = standardize(tx, hx)
        probe = LinearProbe(tx.size(1))
        opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
        pos_weight = ((len(ty) - ty.sum()) / ty.sum().clamp_min(1)).clamp(1, 32)
        events = []
        for step in range(1, STEPS + 1):
            loss = F.binary_cross_entropy_with_logits(probe(tx), ty, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step in (1, 100, STEPS):
                events.append({"step": step, "loss": float(loss.detach())})
        probe.eval()
        with torch.no_grad():
            hold_scores = probe(hx)
        auc, ap = auc_ap(hold_scores, hy)
        # Map concatenated hold rows back to each image for fixed-budget scores.
        scores_by_key = {}
        offset = 0
        for item in hold:
            n = len(item["labels"]); scores_by_key[item["key"]] = hold_scores[offset:offset+n]; offset += n
        top, profiles = topk_metrics(hold, scores_by_key)
        residual, residual_profiles = residual_metrics(hold, scores_by_key)
        report["groups"][name] = {
            "dimension": int(tx.size(1)), "train_rows": int(len(ty)),
            "holdout_rows": int(len(hy)), "holdout_positive_rows": int(hy.sum()),
            "holdout_auc": auc, "holdout_average_precision": ap,
            "holdout_fixed_baseline_budget": top, "holdout_profiles": profiles,
            "holdout_nonbaseline_residual": residual,
            "holdout_nonbaseline_residual_profiles": residual_profiles,
            "events": events,
        }
    vh = report["groups"]["visual_hidden"]
    geom = report["groups"]["geometry_aux"]
    full = report["groups"]["full"]
    report["gate"] = {
        "checkpoint_unchanged": digest(INITIAL) == SHA,
        "sealed_test_read": False, "gpu_used": False,
        "production_checkpoint_written": False,
        "visual_auc_gt_geometry": vh["holdout_auc"] > geom["holdout_auc"],
        "visual_ap_gt_geometry": vh["holdout_average_precision"] > geom["holdout_average_precision"],
        "full_fixed_budget_recall_ge_baseline": full["holdout_fixed_baseline_budget"]["edge_recall"] >= geom["holdout_fixed_baseline_budget"]["edge_recall"],
        "candidate_union_nonempty": report["candidate_union"]["holdout"]["reachable_true_edges"] > 0,
        "production_go": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "groups": report["groups"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
