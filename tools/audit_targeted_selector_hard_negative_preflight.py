"""CPU-only candidate-level hard-negative/ranking preflight.

This is a new diagnostic, not a retry of the failed existence shadow.  It
fits a tiny linear candidate scorer on train-only pairwise hard negatives.  A
positive candidate is paired only with the top frozen-score confirmed
negatives from the same image/profile; holdout is evaluated once.  PNG crop
features are raw-image-only and labels are attached after extraction.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_group_image_features import extract


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-hard-negative-preflight-20261006-r1/report.json"
SEED = 20261006
HARD_NEG_K = 8
MARGIN = 0.5
STEPS = 200


def auc(pos: list[float], neg: list[float]) -> float:
    values = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        rank += ((i + 1 + j) / 2.0) * sum(label for _, label in values[i:j])
        i = j
    return (rank - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def load_candidates(filename: str, split: str) -> list[dict]:
    cache = json.loads((RELEASE / filename).read_text(encoding="utf-8"))["groups"]
    images: dict[str, Image.Image] = {}
    candidates = []
    for group in cache:
        if group["protected"]:
            continue
        image = images.get(group["sample_key"])
        if image is None:
            image = Image.open(RELEASE / "samples" / (group["sample_key"] + ".png")).convert("L")
            images[group["sample_key"]] = image
        for member in group["members"]:
            crop = extract(image, member["feature"][3], member["feature"][4])
            # Keep the frozen model features plus image-only descriptors.  No
            # graph/layer/text artifact is opened here.
            values = list(member["feature"]) + [crop["largest_component_fraction"], crop["ink_center_t192"]]
            positive = member["raw_target"] >= 0 and member["selected_target"] < 0
            negative = group["confirmed_negative"] and member["raw_target"] < 0 and member["selected_target"] < 0
            candidates.append({"sample_key": group["sample_key"], "profile_id": group["profile_id"],
                              "group_id": group["group_id"], "values": values,
                              "positive": positive, "negative": negative})
    return candidates


class LinearRanker(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def make_pairs(candidates: list[dict], mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positives = [c for c in candidates if c["positive"]]
    negatives_by_image: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        if c["negative"]:
            negatives_by_image[c["sample_key"]].append(c)
    pairs = []
    for positive in positives:
        pool = negatives_by_image.get(positive["sample_key"], [])
        # Frozen atom score is used only to select hard negatives, never as a
        # target or holdout-derived feature.  If the image has too few, use
        # the same profile pool; identity never crosses images.
        if len(pool) < HARD_NEG_K:
            pool = [c for c in candidates if c["negative"] and c["profile_id"] == positive["profile_id"]]
        pool = sorted(pool, key=lambda c: c["values"][0], reverse=True)[:HARD_NEG_K]
        for negative in pool:
            pairs.append((torch.tensor(positive["values"]), torch.tensor(negative["values"])))
    if not pairs:
        raise ValueError("No hard-negative pairs")
    pos = torch.stack([p[0] for p in pairs])
    neg = torch.stack([p[1] for p in pairs])
    return (pos - mean) / std, (neg - mean) / std


def evaluate(candidates: list[dict], ranker: LinearRanker, mean: torch.Tensor, std: torch.Tensor) -> dict:
    ranker.eval()
    with torch.no_grad():
        scores = ranker((torch.tensor([c["values"] for c in candidates]) - mean) / std).tolist()
    rows = [{**c, "score": float(score)} for c, score in zip(candidates, scores)]
    positives = [r["score"] for r in rows if r["positive"]]
    negatives = [r["score"] for r in rows if r["negative"]]
    ordered = sorted(rows, key=lambda r: r["score"], reverse=True)
    topk = {}
    for k in (32, 64, 128):
        chosen = ordered[:k]
        recovered = sum(r["positive"] for r in chosen)
        topk[str(k)] = {"selected": len(chosen), "recovered": recovered,
                        "precision": recovered / max(len(chosen), 1),
                        "recall": recovered / max(len(positives), 1)}
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["sample_key"]].append(row)
    rank_top8 = []
    for image_rows in by_image.values():
        image_order = sorted(image_rows, key=lambda r: r["score"], reverse=True)
        positives_here = [r for r in image_rows if r["positive"]]
        if positives_here:
            rank_top8.append(any(r in image_order[:8] for r in positives_here))
    pair_violations = 0
    pair_total = 0
    # Same-image positive-vs-negative ordering is the deployment-relevant pair
    # diagnostic; it does not create any new supervision.
    for image_rows in by_image.values():
        ps = [r["score"] for r in image_rows if r["positive"]]
        ns = [r["score"] for r in image_rows if r["negative"]]
        pair_total += len(ps) * len(ns)
        pair_violations += sum(p <= n - MARGIN for p in ps for n in ns)
    return {"candidates": len(rows), "positive": len(positives), "negative": len(negatives),
            "auc": auc(positives, negatives), "topk": topk,
            "positive_image_rank_top8": sum(rank_top8) / max(len(rank_top8), 1),
            "same_image_margin_violation_rate": pair_violations / max(pair_total, 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    torch.manual_seed(SEED)
    train = load_candidates("train-group-cache.json", "train")
    holdout = load_candidates("holdout-group-cache.json", "holdout")
    train_values = torch.tensor([c["values"] for c in train], dtype=torch.float32)
    mean, std = train_values.mean(0), train_values.std(0).clamp_min(1e-5)
    pos_pairs, neg_pairs = make_pairs(train, mean, std)
    ranker = LinearRanker(pos_pairs.shape[1])
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=2e-2, weight_decay=1e-3)
    events = []
    for step in range(1, STEPS + 1):
        pos_score, neg_score = ranker(pos_pairs), ranker(neg_pairs)
        loss = F.softplus(MARGIN - pos_score + neg_score).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(ranker.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    train_result = evaluate(train, ranker, mean, std)
    holdout_result = evaluate(holdout, ranker, mean, std)
    result = {
        "schema_version": "chemvector-targeted-selector-hard-negative-preflight-0.1",
        "complete": True, "read_only": False, "training_performed": True,
        "training_scope": "cpu_linear_ranker_train_only", "gpu_used": False,
        "optimizer_steps": STEPS, "sealed_test_read": False,
        "layers_or_graph_read": False, "checkpoint_written": False,
        "release_manifest_sha256": hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest(),
        "feature_dim": int(pos_pairs.shape[1]), "hard_negative_k": HARD_NEG_K, "margin": MARGIN,
        "pair_count": int(pos_pairs.shape[0]), "events": events,
        "train": train_result, "holdout": holdout_result,
        "gate": {"train_top128_precision_minimum_085": train_result["topk"]["128"]["precision"] >= 0.85,
                 "holdout_top128_precision_minimum_085": holdout_result["topk"]["128"]["precision"] >= 0.85,
                 "train_positive_image_rank_top8_minimum_085": train_result["positive_image_rank_top8"] >= 0.85,
                 "holdout_positive_image_rank_top8_minimum_085": holdout_result["positive_image_rank_top8"] >= 0.85,
                 "bounded_gpu_training_admissible": False,
                 "reason": "CPU preflight only; GPU authorization requires every precision/rank gate"},
    }
    result["gate"]["all_preflight_gates"] = all(v for k, v in result["gate"].items() if k.endswith("_085"))
    result["gate"]["bounded_gpu_training_admissible"] = result["gate"]["all_preflight_gates"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "pair_count": int(pos_pairs.shape[0]),
                      "train": train_result, "holdout": holdout_result, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
