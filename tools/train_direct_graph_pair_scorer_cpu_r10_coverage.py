#!/usr/bin/env python3
"""CPU-only r10 coverage/hard-negative trial for the direct AtomPair route.

This is an audit-local experiment.  It uses a deterministic, profile-balanced
384-image train subset (including all available hetero-N/O examples) and
retains all positive pairs while sampling same-image geometric hard negatives
at a bounded 2:1 negative/positive ratio.  The frozen production model and
the fresh holdout are never updated; the holdout is evaluated once after a
train-only threshold selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import (
    DATA, SHA, BASE_PAIR_THRESHOLD, DirectPairResidual, _extract, _finish,
    _score_rows, rows,
)

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"
STATE = OUT / "scorer-state.pt"
REPORT = OUT / "report.json"
IMAGE_SIZE = 768.0
STEPS = 300
NEGATIVE_RATIO = 2


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def choose_train(all_rows: list[dict]) -> list[dict]:
    """Select 96 images/profile with deterministic category quotas."""
    categories = ["DOUBLE-only", "AROMATIC", "TRIPLE-only", "mixed_multiple_bond", "hetero_N_O_multiple"]
    quotas = {"DOUBLE-only": 30, "AROMATIC": 20, "TRIPLE-only": 20, "mixed_multiple_bond": 20, "hetero_N_O_multiple": 6}
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in sorted((r for r in all_rows if r.get("split") == "train"), key=lambda x: x["sample_key"]):
        by_profile[row["profile_id"]].append(row)
    chosen = []
    for profile in sorted(by_profile):
        pool = by_profile[profile]
        for category in categories:
            part = [r for r in pool if r.get("category") == category]
            chosen.extend(part[:quotas[category]])
        # The final profile may have only six hetero rows; fill one slot from
        # the largest non-hetero category while keeping the profile balanced.
        current = [r for r in chosen if r["profile_id"] == profile]
        if len(current) < 96:
            used = {r["sample_key"] for r in current}
            for r in pool:
                if r["sample_key"] not in used:
                    chosen.append(r)
                    current.append(r)
                    if len(current) == 96:
                        break
        if len(current) != 96:
            raise RuntimeError(f"profile {profile} selected {len(current)} rows")
    if len(chosen) != 384:
        raise RuntimeError(f"expected 384 train rows, got {len(chosen)}")
    return sorted(chosen, key=lambda x: x["sample_key"])


def sample_hard_negatives(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    selected = []
    total_pos = total_neg = hard_neg = 0
    by_category = Counter()
    for item in items:
        labels = item["labels"]
        features = item["features"]
        positives = torch.where(labels > 0.5)[0].tolist()
        negatives = torch.where(labels <= 0.5)[0].tolist()
        # Feature layout ends with midpoint(2), distance(1), score(2).
        # Distance is therefore -3; choose nearby non-edges first.
        nearby = [i for i in negatives if float(features[i, -3]) <= 80.0]
        far = [i for i in negatives if i not in set(nearby)]
        need = min(len(negatives), NEGATIVE_RATIO * len(positives))
        chosen_neg = nearby[:need]
        if len(chosen_neg) < need:
            chosen_neg += far[: need - len(chosen_neg)]
        selected.extend((features[i], labels[i]) for i in positives + chosen_neg)
        total_pos += len(positives); total_neg += len(chosen_neg); hard_neg += sum(float(features[i, -3]) <= 80.0 for i in chosen_neg)
        by_category[item["row"].get("category", "unknown")] += len(positives) + len(chosen_neg)
    if not selected:
        raise RuntimeError("empty sampled training set")
    return torch.stack([x[0] for x in selected]), torch.stack([x[1] for x in selected]), {
        "positive_rows": total_pos, "negative_rows": total_neg, "hard_negative_rows": hard_neg,
        "negative_positive_ratio": total_neg / max(total_pos, 1), "rows_by_category": dict(by_category),
    }


def threshold_train(model: DirectPairResidual, items: list[dict]) -> tuple[float, dict]:
    model.eval()
    with torch.inference_mode():
        logits = torch.cat([item["base_logits"] + model(item["features"]) for item in items])
        labels = torch.cat([item["labels"] for item in items])
    choices = []
    for i in range(35, 100):
        t = i / 100
        selected = logits.sigmoid() >= t; positive = labels > .5
        tp = int((selected & positive).sum()); fp = int((selected & ~positive).sum()); fn = int((~selected & positive).sum())
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        choices.append((precision, recall, f1, t, tp, fp, fn))
    admissible = [x for x in choices if x[0] >= .99]
    chosen = max(admissible or choices, key=lambda x: (x[1], x[2], x[3]))
    precision, recall, f1, threshold, tp, fp, fn = chosen
    return threshold, {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "precision_constraint": ".99 when feasible"}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint SHA changed")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008)
    all_rows = rows(); train_rows = choose_train(all_rows); holdout_rows = sorted((r for r in all_rows if r.get("split") == "fresh_holdout"), key=lambda x: x["sample_key"])
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in holdout_rows}:
        raise RuntimeError("train/holdout molecule identity overlap")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.eval().requires_grad_(False)
    train_data = [_extract(r, model, config, need_labels=True) for r in train_rows]
    holdout_data = [_extract(r, model, config, need_labels=True) for r in holdout_rows]
    features, labels, sampling = sample_hard_negatives(train_data)
    scorer = DirectPairResidual(input_dim=features.size(1)); optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(labels) - labels.sum()) / labels.sum().clamp_min(1)).clamp(1, 32)
    base_logits = torch.cat([item["base_logits"] for item in train_data])
    # Train only on sampled rows; retain their corresponding base logits.
    sampled_base = []
    for item in train_data:
        for i in torch.where(item["labels"] > .5)[0].tolist(): sampled_base.append(item["base_logits"][i])
        negatives = torch.where(item["labels"] <= .5)[0].tolist(); nearby = [i for i in negatives if float(item["features"][i, -3]) <= 80.0]; far = [i for i in negatives if i not in set(nearby)]; need = min(len(negatives), NEGATIVE_RATIO * int((item["labels"] > .5).sum())); chosen = (nearby[:need] + far[:max(0, need-len(nearby))]); sampled_base.extend(item["base_logits"][i] for i in chosen)
    sampled_base = torch.stack(sampled_base)
    events = []
    for step in range(1, STEPS + 1):
        scorer.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(sampled_base + scorer(features), labels, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 200, 300): events.append({"step": step, "loss": float(loss)})
    scorer.eval().requires_grad_(False); threshold, calibration = threshold_train(scorer, train_data)
    baseline = _finish(_score_rows(holdout_data, None, BASE_PAIR_THRESHOLD)); scored = _finish(_score_rows(holdout_data, scorer, threshold))
    profiles = {}
    for profile in sorted({r["profile_id"] for r in holdout_rows}):
        subset = [x for x in holdout_data if x["row"]["profile_id"] == profile]
        profiles[profile] = {"baseline": _finish(_score_rows(subset, None, BASE_PAIR_THRESHOLD)), "scored": _finish(_score_rows(subset, scorer, threshold))}
    after = digest(INITIAL); report = {
        "schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage-0.1", "complete": True,
        "read_only": False, "training_performed": True, "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False, "dataset": str(DATA), "train_images": len(train_rows), "holdout_images": len(holdout_rows),
        "train_profiles": dict(Counter(r["profile_id"] for r in train_rows)), "train_categories": dict(Counter(r.get("category") for r in train_rows)),
        "sampling": sampling, "scorer": {"class": "DirectPairResidual", "parameters": sum(p.numel() for p in scorer.parameters()), "steps": STEPS, "events": events, "pos_weight": float(pos_weight), "calibration_train_only": calibration},
        "baseline_holdout": baseline, "scored_holdout": scored, "holdout_profiles": profiles,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False, "strict_holdout_edge_recall_improvement": scored["edge_recall"] > baseline["edge_recall"], "precision_ge_0.99": scored["edge_precision"] >= .99, "recall_gt_0.950": scored["edge_recall"] > .950, "dangling_le_0.01": scored["dangling_rate"] <= .01, "all_profile_recall_ge_0.90": all(v["scored"]["edge_recall"] >= .90 for v in profiles.values()), "production_go": False, "next": "diagnostic CPU result; no production merge"},
    }
    OUT.mkdir(parents=True); torch.save({"schema_version": report["schema_version"], "state_dict": scorer.state_dict(), "initial_checkpoint_sha256": SHA}, STATE); REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(REPORT), "baseline": baseline, "scored": scored, "calibration": calibration, "gate": report["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__": run()
