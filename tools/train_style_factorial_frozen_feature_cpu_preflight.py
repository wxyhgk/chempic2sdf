"""Probe candidate labels with frozen initial high-resolution vision features.

This is a CPU-only linear/MLP diagnostic.  The initial vision backbone is
frozen; stride-4 and stride-8 feature maps are bilinearly sampled at candidate
coordinates and concatenated with raw/presence/x/y geometry.  No raster crop,
nearest-distance field, target atom id, layer, or graph supervision is used.
Use ``--max-samples 100`` for a cheap pipeline/cost smoke; a full run is
intentionally not started by this script.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-frozen-feature-cpu-preflight-20261010-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
SEED = 20261013
IMAGE_BATCH = 8
STEPS = 300
NEGATIVE_MULTIPLIER = 8
FIXED_GATE = 0.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Head(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def geo(row: dict) -> np.ndarray:
    # nearest_distance_px is deliberately not read; it is GT-derived.
    return np.asarray([row["raw_score"], row["presence_logit"], row["x_norm"], row["y_norm"]], np.float32)


def load_cache() -> tuple[dict[str, list[dict]], Counter]:
    rows: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            counts[row["label"]] += 1
            if row["label"] in {"strict_positive", "hard_negative", "near_unknown"}:
                rows[row["sample_key"]].append(row)
    return rows, counts


def choose_train(rows: dict[str, list[dict]]) -> set[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for values in rows.values():
        for row in values:
            if row["replay_split"] == "train_core" and row["label"] in {"strict_positive", "hard_negative"}:
                grouped[row["variant"]].append(row)
    chosen = []
    for style, values in grouped.items():
        pos = [r for r in values if r["label"] == "strict_positive"]
        neg = [r for r in values if r["label"] == "hard_negative"]
        n = min(len(neg), NEGATIVE_MULTIPLIER * len(pos))
        neg = sorted(neg, key=lambda r: hashlib.sha256(f"{SEED}:{r['sample_key']}:{r['slot']}".encode()).hexdigest())[:n]
        chosen.extend(pos + neg)
    return {f"{r['sample_key']}:{r['slot']}" for r in chosen}


def sample_features(feature4: torch.Tensor, feature8: torch.Tensor, rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.empty((0, feature4.shape[1] + feature8.shape[1] + 4), np.float32)
    coords = torch.tensor([[2.0 * float(r["x_norm"]) - 1.0, 2.0 * float(r["y_norm"]) - 1.0] for r in rows], dtype=feature4.dtype)
    # Feature maps are [1,C,H,W], candidate grid is [1,N,1,2].
    grid = coords.view(1, len(rows), 1, 2)
    p4 = F.grid_sample(feature4, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T
    p8 = F.grid_sample(feature8, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T
    vision = torch.cat((p4, p8), dim=1).detach().cpu().numpy().astype(np.float32)
    geometry = np.stack([geo(r) for r in rows], axis=0)
    return np.concatenate((vision, geometry), axis=1)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    rows_by_key, labels = load_cache()
    train_ids = choose_train(rows_by_key)
    wanted = {key for key, values in rows_by_key.items()
              if any((f"{r['sample_key']}:{r['slot']}" in train_ids) or r["replay_split"] in {"dev_core", "dev_stress"} for r in values)}
    model, config, _ = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial SHA mismatch")
    model.to("cpu").eval().requires_grad_(False)
    collator = AtomBondSceneCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds))
    shards = sorted(str(p) for p in DATA.joinpath("shards").glob("train-*.tar"))
    train_x, train_y, eval_x, eval_y, eval_style, eval_label = [], [], [], [], [], []
    pending, seen = [], set()
    processed = 0

    def flush(batch: list[dict]) -> None:
        nonlocal processed
        if not batch:
            return
        tensors = {k: v for k, v in collator(batch).items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            stroke = model.vision.forward_stroke_resolution(tensors["pixel_values"])
            high = model.vision.forward_high_resolution_from_stroke(stroke)
        for i, sample in enumerate(batch):
            values = rows_by_key[sample["key"]]
            train_rows = [r for r in values if f"{r['sample_key']}:{r['slot']}" in train_ids]
            eval_rows = [r for r in values if r["replay_split"] in {"dev_core", "dev_stress"}]
            f4 = stroke[i:i + 1]; f8 = high[i:i + 1]
            if train_rows:
                x = sample_features(f4, f8, train_rows)
                train_x.extend(x); train_y.extend(float(r["label"] == "strict_positive") for r in train_rows)
            if eval_rows:
                x = sample_features(f4, f8, eval_rows)
                eval_x.extend(x); eval_y.extend(float(r["label"] == "strict_positive") for r in eval_rows)
                eval_style.extend(r["variant"] for r in eval_rows); eval_label.extend(r["label"] for r in eval_rows)
            processed += 1

    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False, load_raster_layers=False):
        if sample["key"] not in wanted:
            continue
        seen.add(sample["key"])
        pending.append(sample)
        if len(pending) == IMAGE_BATCH:
            flush(pending); pending = []
            if args.max_samples and processed >= args.max_samples:
                break
    flush(pending)
    complete = seen == wanted
    result = {"schema_version": "chemvector-style-factorial-frozen-feature-cpu-preflight-0.1",
              "complete": complete, "smoke": bool(args.max_samples), "training_performed": False,
              "gpu_used": False, "sealed_test_read": False, "backward_called": False,
              "initial_sha256": INITIAL_SHA, "cache_sha256": digest(CACHE),
              "processed_samples": processed, "wanted_samples": len(wanted),
              "feature_maps": {"stride4": [96, 192, 192], "stride8": [192, 96, 96], "sampling": "bilinear"},
              "geometry_features": ["raw_score", "presence_logit", "x_norm", "y_norm"],
              "forbidden_features": ["nearest_distance_px", "target_atom_id", "layers", "graph"],
              "train_rows": len(train_y), "eval_rows": len(eval_y), "cache_labels": dict(labels),
              "gate": {"training_admissible": False, "feature_extraction_smoke_complete": complete,
                       "next": "if smoke timing and feature shapes are acceptable, run one bounded full CPU probe; never use smoke as accuracy evidence"}}
    if complete and train_y and not args.smoke_only:
        train_arr, y = np.asarray(train_x, np.float32), np.asarray(train_y, np.float32)
        eval_arr, ey = np.asarray(eval_x, np.float32), np.asarray(eval_y, np.float32)
        mean, std = train_arr.mean(0), np.maximum(train_arr.std(0), 1e-5)
        train_arr, eval_arr = (train_arr - mean) / std, (eval_arr - mean) / std
        head = Head(train_arr.shape[1]); optim = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
        xt, yt = torch.tensor(train_arr), torch.tensor(y); gen = torch.Generator().manual_seed(SEED)
        for _ in range(STEPS):
            idx = torch.randint(0, len(yt), (min(512, len(yt)),), generator=gen)
            loss = F.binary_cross_entropy_with_logits(head(xt[idx]), yt[idx]); optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad(): logits = head(torch.tensor(eval_arr)).numpy()
        metrics = {}
        for style in sorted(set(eval_style)):
            m = np.asarray(eval_style) == style; selected = logits[m] >= FIXED_GATE; truth = np.asarray(eval_y)[m].astype(bool)
            metrics[style] = {"candidates": int(m.sum()), "strict": int(truth.sum()), "selected": int(selected.sum()),
                              "strict_selected": int((selected & truth).sum()),
                              "near_unknown_selected": int((selected & (np.asarray(eval_label)[m] == "near_unknown")).sum()),
                              "strict_precision": float((selected & truth).sum() / max(selected.sum(), 1)),
                              "strict_recall": float((selected & truth).sum() / max(truth.sum(), 1))}
        result.update({"training_performed": True, "optimizer_steps": STEPS, "fixed_gate": FIXED_GATE, "metrics": metrics})
    output.parent.mkdir(parents=True, exist_ok=False); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False)); return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--max-samples", type=int, default=0); p.add_argument("--smoke-only", action="store_true"); p.add_argument("--output", default=str(OUTPUT)); return p.parse_args()


if __name__ == "__main__":
    args = parse_args(); args.smoke_only = args.smoke_only or bool(args.max_samples); run(args)
