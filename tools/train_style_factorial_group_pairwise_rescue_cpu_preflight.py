"""Fixed CPU group-pairwise ranking preflight for strict rescue candidates.

The frozen baseline's protected rows are retained unchanged and never enter
the rescue ranking.  Within each candidate image group, strict-positive
dropped rows are paired against hard negatives; near-unknown and
selected-unmatched rows are masked from the pairwise loss.  The scorer uses
only candidate scores and candidate-to-candidate geometry (no GT distance or
target IDs).  Evaluation is a fixed top-1/top-3 group policy, not a threshold
sweep.
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
CACHE_AUDIT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/cache-audit.json"
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-group-pairwise-rescue-cpu-preflight-20261007-r1/report.json"
SEED = 20261015
TRAIN_LIMIT = 120
DEV_LIMIT = 40
STEPS = 300
BATCH = 512
MARGIN = 0.5


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RankingHead(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_manifests() -> dict[str, dict]:
    result = {}
    for split, name in (("train_core", "train-core-manifest.json"), ("dev_core", "dev-core-manifest.json"), ("dev_stress", "dev-stress-manifest.json")):
        for row in json.loads((CORE / name).read_text(encoding="utf-8"))["rows"]:
            result[row["sample_key"]] = {**row, "replay_split": split}
    return result


def load_cache(wanted: set[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_key"] in wanted:
                groups[row["sample_key"]].append(row)
    return groups


def make_features(rows: list[dict]) -> np.ndarray:
    """Construct relative candidate features without consulting any labels."""
    points = np.asarray([[float(row["x_norm"]), float(row["y_norm"])] for row in rows], dtype=np.float32)
    raw = np.asarray([float(row["raw_score"]) for row in rows], dtype=np.float32)
    presence = np.asarray([float(row["presence_logit"]) for row in rows], dtype=np.float32)
    n = len(rows)
    if n > 1:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1) * 768.0
        distances += np.eye(n, dtype=np.float32) * 1e6
        nearest = distances.min(axis=1)
        density = (distances < 32.0).sum(axis=1).astype(np.float32)
    else:
        nearest, density = np.asarray([768.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    raw_rank = np.argsort(np.argsort(-raw, kind="stable"), kind="stable").astype(np.float32) / max(n, 1)
    presence_rank = np.argsort(np.argsort(-presence, kind="stable"), kind="stable").astype(np.float32) / max(n, 1)
    center = points.mean(axis=0)
    return np.column_stack((raw, presence, raw_rank, presence_rank, nearest / 768.0,
                            density / 128.0, points[:, 0] - center[0], points[:, 1] - center[1])).astype(np.float32)


def evaluate(groups: list[list[dict]], scores: list[np.ndarray]) -> dict:
    top1_strict = top1_hard = top1_masked = 0
    top3_strict = top3_hard = 0
    strict_groups = top1_group_hits = top3_group_hits = 0
    strict_candidates = top1_strict_candidates = top3_strict_candidates = 0
    selected_groups = 0
    for rows, logits in zip(groups, scores):
        # Protected rows are baseline output, not rescue candidates. Near and
        # selected-unmatched rows remain in ranking but are reported masked.
        eligible = np.asarray([row["label"] != "protected" for row in rows], dtype=bool)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            continue
        order = indices[np.argsort(-logits[indices], kind="stable")]
        first, top = order[0], order[:3]
        label_first = rows[int(first)]["label"]
        top_labels = [rows[int(index)]["label"] for index in top]
        if label_first == "strict_positive":
            top1_strict += 1
        elif label_first == "hard_negative":
            top1_hard += 1
        else:
            top1_masked += 1
        top1_strict_candidates += int(label_first == "strict_positive")
        top3_strict_candidates += sum(label == "strict_positive" for label in top_labels)
        top3_hard += sum(label == "hard_negative" for label in top_labels)
        top1_hard += 0
        has_strict = any(row["label"] == "strict_positive" for row in rows)
        if has_strict:
            strict_groups += 1
            top1_group_hits += int(label_first == "strict_positive")
            top3_group_hits += int("strict_positive" in top_labels)
            strict_candidates += sum(row["label"] == "strict_positive" for row in rows)
        selected_groups += 1
    return {
        "groups": len(groups), "groups_with_strict": strict_groups,
        "top1_strict_groups": top1_group_hits, "top3_strict_groups": top3_group_hits,
        "top1_group_recall": top1_group_hits / max(strict_groups, 1),
        "top3_group_recall": top3_group_hits / max(strict_groups, 1),
        "top1_selected_groups": selected_groups, "top1_strict": top1_strict,
        "top1_hard": top1_hard, "top1_masked": top1_masked,
        "top1_strict_precision_all_groups": top1_strict / max(selected_groups, 1),
        "top3_strict_candidates": top3_strict_candidates, "top3_hard_candidates": top3_hard,
        "strict_candidates": strict_candidates,
        "top1_strict_candidate_recall": top1_strict_candidates / max(strict_candidates, 1),
        "top3_strict_candidate_recall": top3_strict_candidates / max(strict_candidates, 1),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"]["jsonl_complete_and_valid"] or not audit["gate"]["matches_forward_aggregate"]:
        raise ValueError("candidate cache audit is not ready")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifests = load_manifests()
    by_variant: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in manifests.items():
        by_variant[row["replay_split"], row["variant"]].append(key)
    chosen: set[str] = set()
    chosen.update(sorted(by_variant["train_core", "coordgen"])[:TRAIN_LIMIT])
    chosen.update(sorted(by_variant["train_core", "original"])[:TRAIN_LIMIT])
    chosen.update(sorted(by_variant["train_core", "rdkit"])[:TRAIN_LIMIT])
    for variant in ("coordgen", "original", "rdkit"):
        chosen.update(sorted(by_variant["dev_core", variant])[:DEV_LIMIT])
    chosen.update(sorted(by_variant["dev_stress", "stress"])[:DEV_LIMIT])
    groups = load_cache(chosen)
    skipped = sorted(chosen - set(groups))
    chosen.intersection_update(groups)
    manifests = {key: value for key, value in manifests.items() if key in chosen}
    group_arrays = {key: make_features(groups[key]) for key in chosen}
    train_keys = [key for key in sorted(chosen) if manifests[key]["replay_split"] == "train_core"]
    # Only strict/hard rows form pairs. This makes near_unknown, protected,
    # and selected_unmatched unavailable to the ranking loss.
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for key in train_keys:
        rows = groups[key]; features = group_arrays[key]
        positives = [i for i, row in enumerate(rows) if row["label"] == "strict_positive"]
        negatives = [i for i, row in enumerate(rows) if row["label"] == "hard_negative"]
        for pos in positives:
            # Fixed deterministic four negatives per strict row.
            for neg in negatives[:4]:
                pairs.append((features[pos], features[neg]))
    if not pairs:
        raise ValueError("no strict/hard training pairs")
    xp = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
    xn = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
    all_train = np.concatenate((xp, xn), axis=0)
    mean, std = all_train.mean(0), np.maximum(all_train.std(0), 1e-5)
    model = RankingHead(xp.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    xp_t, xn_t = torch.tensor((xp - mean) / std), torch.tensor((xn - mean) / std)
    rng = np.random.default_rng(SEED); events = []
    for step in range(1, STEPS + 1):
        index = torch.tensor(rng.integers(0, len(xp), size=BATCH))
        pos_score, neg_score = model(xp_t[index]), model(xn_t[index])
        loss = F.relu(MARGIN - pos_score + neg_score).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True); optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    model.eval(); scores_by_key = {}
    with torch.no_grad():
        for key in sorted(chosen):
            rows = groups[key]; values = torch.tensor((group_arrays[key] - mean) / std)
            scores_by_key[key] = model(values).numpy()
    reports = {}
    for split, variants in (("train_core", ("coordgen", "original", "rdkit")), ("dev_core", ("coordgen", "original", "rdkit")), ("dev_stress", ("stress",))):
        for variant in variants:
            keys = [key for key in sorted(chosen) if manifests[key]["replay_split"] == split and manifests[key]["variant"] == variant]
            reports[f"{split}/{variant}"] = evaluate([groups[key] for key in keys], [scores_by_key[key] for key in keys])
    core = [reports[f"dev_core/{variant}"] for variant in ("coordgen", "original", "rdkit")]
    result = {
        "schema_version": "chemvector-style-factorial-group-pairwise-rescue-cpu-preflight-0.1",
        "complete": True, "training_performed": True, "training_scope": "within_group_pairwise_strict_rescue_cpu_only",
        "gpu_used": False, "optimizer_steps": STEPS, "checkpoint_written": False, "sealed_test_read": False,
        "cache_sha256": digest(CACHE), "cache_audit_sha256": digest(CACHE_AUDIT), "seed": SEED,
        "sample_limits": {"train_per_variant": TRAIN_LIMIT, "dev_per_variant": DEV_LIMIT}, "replay_skipped_samples": skipped,
        "margin": MARGIN, "batch_size": BATCH, "train_pairs": len(pairs),
        "features": ["raw_score", "presence_logit", "raw_rank", "presence_rank", "nearest_candidate", "local_density", "relative_x", "relative_y"],
        "masked_labels": ["protected", "near_unknown", "selected_unmatched"],
        "reports": reports, "events": events,
        "input_contract": {"gt_distance_used_as_input": False, "protected_baseline_preserved": True, "near_unknown_in_pairwise_loss": False, "selected_unmatched_in_pairwise_loss": False},
        "gate": {"each_core_top1_group_recall_ge_050": all(r["top1_group_recall"] >= .50 for r in core), "each_core_top3_group_recall_ge_050": all(r["top3_group_recall"] >= .50 for r in core), "stress_separate": True, "training_ready": False, "reason": "small CPU group ranking preflight; no GPU authorization"},
    }
    result["gate"]["training_ready"] = result["gate"]["each_core_top1_group_recall_ge_050"] and result["gate"]["each_core_top3_group_recall_ge_050"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "train_pairs": len(pairs), "reports": reports, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
