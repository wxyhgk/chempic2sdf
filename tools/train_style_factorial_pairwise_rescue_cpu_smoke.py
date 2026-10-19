"""Small CPU pairwise-rescue smoke on coordinate-defined candidate groups.

Groups are connected components of candidate coordinates within a fixed 12 px
radius; no GT distance or target atom id is read to form them.  Strict-drop
rows are group positives.  Near-unknown, selected-unmatched, and protected
rows are excluded from the loss.  Hard rows are only pairwise ranking
competitors (never BCE negatives).  The result is a diagnostic top-1 group
ranking, not selector authorization.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-pairwise-rescue-cpu-smoke-20261015-r1/report.json"
SEED = 20261016
RADIUS_PX = 12.0
MARGIN = 1.0
STEPS = 200
BATCH = 512
MAX_HARD_PER_POS = 8


class ScoreHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature(row: dict) -> list[float]:
    return [float(row["raw_score"]), float(row["presence_logit"]), float(row["x_norm"]), float(row["y_norm"])]


def groups(rows: list[dict]) -> list[list[dict]]:
    # Spatial bins avoid an O(n²) image-level scan while retaining exact
    # connected-component semantics for the fixed radius.
    cell = RADIUS_PX
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    parent = list(range(len(rows)))
    xy = [(float(r["x_px"]), float(r["y_px"])) for r in rows]

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb: parent[rb] = ra

    for i, (x, y) in enumerate(xy):
        bx, by = int(x // cell), int(y // cell)
        for nx in (bx - 1, bx, bx + 1):
            for ny in (by - 1, by, by + 1):
                for j in bins.get((nx, ny), []):
                    if (x - xy[j][0]) ** 2 + (y - xy[j][1]) ** 2 <= RADIUS_PX ** 2:
                        union(i, j)
        bins[(bx, by)].append(i)
    result: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows): result[root(i)].append(row)
    return list(result.values())


def load_rows(max_images: int) -> tuple[dict[str, list[dict]], dict]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["replay_split"] in {"train_core", "dev_core", "dev_stress"}:
                by_key[row["sample_key"]].append(row)
    train_keys = sorted(k for k, v in by_key.items() if v[0]["replay_split"] == "train_core")
    dev_keys = sorted(k for k, v in by_key.items() if v[0]["replay_split"] == "dev_core")
    stress_keys = sorted(k for k, v in by_key.items() if v[0]["replay_split"] == "dev_stress")
    chosen = train_keys[:max_images] + dev_keys[:max_images] + stress_keys[:max_images]
    return {k: by_key[k] for k in chosen}, {"train_images": min(max_images, len(train_keys)), "dev_images": min(max_images, len(dev_keys)), "stress_images": min(max_images, len(stress_keys))}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(min(torch.get_num_threads(), 8)); torch.manual_seed(SEED)
    by_key, image_counts = load_rows(args.max_images)
    train_groups, eval_groups = [], []
    for key, rows in by_key.items():
        gs = groups(rows)
        if rows[0]["replay_split"] == "train_core": train_groups.extend(gs)
        else: eval_groups.extend(gs)
    pair_rows = []
    for group in train_groups:
        pos = [r for r in group if r["label"] == "strict_positive"]
        hard = [r for r in group if r["label"] == "hard_negative"]
        for p in pos:
            choices = sorted(hard, key=lambda r: hashlib.sha256(f"{SEED}:{r['sample_key']}:{r['slot']}".encode()).hexdigest())[:MAX_HARD_PER_POS]
            pair_rows.extend((p, h) for h in choices)
    if not pair_rows: raise ValueError("smoke produced no strict-hard groups")
    train_x = np.asarray([feature(p) for p, _ in pair_rows] + [feature(h) for _, h in pair_rows], np.float32)
    mean, std = train_x.mean(0), np.maximum(train_x.std(0), 1e-5)
    head = ScoreHead(); optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    pos_x = torch.tensor((np.asarray([feature(p) for p, _ in pair_rows], np.float32) - mean) / std)
    hard_x = torch.tensor((np.asarray([feature(h) for _, h in pair_rows], np.float32) - mean) / std)
    gen = torch.Generator().manual_seed(SEED); events = []
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(pos_x), (min(BATCH, len(pos_x)),), generator=gen)
        margin_loss = F.relu(MARGIN - head(pos_x[idx]) + head(hard_x[idx])).mean()
        optimizer.zero_grad(set_to_none=True); margin_loss.backward(); norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True); optimizer.step()
        events.append({"step": step, "loss": float(margin_loss.detach()), "gradient_norm": float(norm)})
    head.eval(); metrics = defaultdict(lambda: {"groups": 0, "strict_top1": 0, "hard_top1": 0, "masked_top1": 0, "no_strict": 0})
    with torch.no_grad():
        for group in eval_groups:
            strict = [r for r in group if r["label"] == "strict_positive"]
            if not strict: continue
            x = torch.tensor((np.asarray([feature(r) for r in group], np.float32) - mean) / std)
            winner = group[int(torch.argmax(head(x)).item())]; split = winner["replay_split"]; style = winner["variant"]; key = f"{split}/{style}"; m = metrics[key]; m["groups"] += 1
            if winner["label"] == "strict_positive": m["strict_top1"] += 1
            elif winner["label"] == "hard_negative": m["hard_top1"] += 1
            else: m["masked_top1"] += 1
    for m in metrics.values(): m["top1_strict_rate"] = m["strict_top1"] / max(m["groups"], 1)
    result = {"schema_version": "chemvector-style-factorial-pairwise-rescue-cpu-smoke-0.1", "complete": True, "smoke": True, "training_performed": True, "gpu_used": False, "sealed_test_read": False, "optimizer_steps": STEPS, "batch_size": BATCH, "radius_px": RADIUS_PX, "margin": MARGIN, "max_hard_per_positive": MAX_HARD_PER_POS, "cache_sha256": digest(CACHE), "feature_names": ["raw_score", "presence_logit", "x_norm", "y_norm"], "forbidden_features": ["nearest_distance_px", "target_atom_id", "layers", "graph"], "image_counts": image_counts, "train_groups": len(train_groups), "pair_rows": len(pair_rows), "metrics": dict(metrics), "events": events, "gate": {"training_admissible": False, "next": "pairwise smoke only; do not treat top-1 as selector precision/recall or authorize GPU"}}
    output.parent.mkdir(parents=True, exist_ok=False); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(output), "image_counts": image_counts, "pair_rows": len(pair_rows), "metrics": dict(metrics)}, indent=2, ensure_ascii=False)); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--max-images", type=int, default=100); parser.add_argument("--output", default=str(OUTPUT)); run(parser.parse_args())
