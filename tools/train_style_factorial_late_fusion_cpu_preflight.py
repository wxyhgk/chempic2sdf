"""CPU-only geometry + local-pixel late-fusion preflight.

The candidate cache was produced by frozen-initial inference.  This probe fits
one tiny head using only candidate-observable values (raw atom score, set
presence logit, normalized x/y) and a candidate-centred raster crop.  It does
not use ``nearest_distance_px`` or any other GT-derived value from the cache.
Strict positives and confirmed far negatives are the only loss rows;
near-unknown and protected candidates are never negatives.  A single fixed
zero-logit gate is used for every style, with no threshold search.

The style shards are scanned once to decode PNGs.  The model is CPU-only and
no checkpoint is written; this is a readiness diagnostic, not authorization.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.data import ChemVectorIterableDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-late-fusion-cpu-preflight-20261010-r1/report.json"
SEED = 20261012
STEPS = 300
BATCH = 512
PATCH = 32
RESIZED = 16
NEGATIVE_MULTIPLIER = 8
FIXED_GATE = 0.0
CORE = ("coordgen", "original", "rdkit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LateFusionHead(nn.Module):
    def __init__(self, geometry_dim: int) -> None:
        super().__init__()
        self.pixel = nn.Sequential(nn.Linear(RESIZED * RESIZED, 64), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, 16), nn.GELU())
        self.head = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, pixel: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.pixel(pixel), self.geometry(geometry)), dim=-1)).squeeze(-1)


def patch_vector(gray: np.ndarray, x_norm: float, y_norm: float) -> np.ndarray:
    h, w = gray.shape
    cx, cy = round(float(x_norm) * w), round(float(y_norm) * h)
    half = PATCH // 2
    crop = np.full((PATCH, PATCH), 255, dtype=np.uint8)
    left, top = cx - half, cy - half
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left + PATCH), min(h, top + PATCH)
    crop[y0 - top:y1 - top, x0 - left:x1 - left] = gray[y0:y1, x0:x1]
    # Exact 2x2 area reduction avoids a PIL resize per candidate and remains
    # deterministic across workers/platforms.
    small = crop.reshape(RESIZED, 2, RESIZED, 2).mean(axis=(1, 3))
    return ((255.0 - small).reshape(-1) / 255.0).astype(np.float32)


def geo_vector(row: dict) -> np.ndarray:
    # Deliberately do not read nearest_distance_px: that field is GT-derived.
    return np.asarray([row["raw_score"], row["presence_logit"], row["x_norm"], row["y_norm"]], dtype=np.float32)


def selected_rows(cache: Path) -> tuple[dict[str, list[dict]], dict]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    with cache.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            label = row["label"]
            counts[label] += 1
            # protected is already retained by the frozen selector and is not
            # a negative; selected_unmatched remains unresolved/unknown.
            if label in {"strict_positive", "hard_negative", "near_unknown"}:
                by_key[row["sample_key"]].append(row)
    return by_key, {"cache_rows": sum(counts.values()), "cache_labels": dict(counts)}


def choose_train(rows_by_key: dict[str, list[dict]]) -> tuple[set[str], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rows in rows_by_key.values():
        for row in rows:
            if row["replay_split"] == "train_core" and row["label"] in {"strict_positive", "hard_negative"}:
                grouped[row["variant"]].append(row)
    chosen: list[dict] = []
    per_style = {}
    for style, rows in sorted(grouped.items()):
        positives = [r for r in rows if r["label"] == "strict_positive"]
        negatives = [r for r in rows if r["label"] == "hard_negative"]
        limit = min(len(negatives), NEGATIVE_MULTIPLIER * len(positives))
        negatives = sorted(negatives, key=lambda r: hashlib.sha256(
            f"{SEED}:{r['sample_key']}:{r['slot']}".encode()).hexdigest())[:limit]
        chosen.extend(positives + negatives)
        per_style[style] = {"strict_positive": len(positives), "hard_negative": len(negatives),
                            "candidate_rows": len(positives) + len(negatives)}
    return {f"{r['sample_key']}:{r['slot']}" for r in chosen}, per_style


def build_arrays(rows_by_key: dict[str, list[dict]], selected_train: set[str], shards: list[str]) -> tuple[dict, dict]:
    # Keep only rows required for this pass; image data are decoded one sample
    # at a time while iterating the committed shards.
    wanted: dict[str, list[dict]] = defaultdict(list)
    for key, rows in rows_by_key.items():
        for row in rows:
            split = row["replay_split"]
            is_train = f"{row['sample_key']}:{row['slot']}" in selected_train
            use = is_train or split in {"dev_core", "dev_stress"}
            if use:
                wanted[key].append(row)
    train_pixel, train_geo, train_y, train_style = [], [], [], []
    eval_data: dict[str, dict[str, list]] = defaultdict(lambda: {"pixel": [], "geo": [], "label": [], "style": []})
    seen: set[str] = set()
    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False,
                                            load_raster_layers=False):
        key = sample["key"]
        if key not in wanted:
            continue
        seen.add(key)
        gray = np.asarray(sample["image"].convert("L"), dtype=np.uint8)
        for row in wanted[key]:
            px = patch_vector(gray, row["x_norm"], row["y_norm"])
            geo = geo_vector(row)
            label = 1.0 if row["label"] == "strict_positive" else 0.0
            if f"{row['sample_key']}:{row['slot']}" in selected_train:
                train_pixel.append(px); train_geo.append(geo); train_y.append(label); train_style.append(row["variant"])
            if row["replay_split"] in {"dev_core", "dev_stress"}:
                key_split = row["replay_split"]
                eval_data[key_split]["pixel"].append(px); eval_data[key_split]["geo"].append(geo)
                eval_data[key_split]["label"].append(row["label"]); eval_data[key_split]["style"].append(row["variant"])
    if seen != set(wanted):
        raise ValueError(f"missing source images for {len(set(wanted) - seen)} samples")
    train = {"pixel": np.asarray(train_pixel, np.float32), "geo": np.asarray(train_geo, np.float32),
             "y": np.asarray(train_y, np.float32), "style": np.asarray(train_style)}
    ev = {split: {"pixel": np.asarray(d["pixel"], np.float32), "geo": np.asarray(d["geo"], np.float32),
                  "labels": np.asarray(d["label"], dtype=object), "style": np.asarray(d["style"])}
          for split, d in eval_data.items()}
    return train, ev


def evaluate(logits: np.ndarray, data: dict, style: str | None = None) -> dict:
    mask = np.ones(len(logits), dtype=bool) if style is None else (data["style"] == style)
    labels = data["labels"][mask]
    known = np.isin(labels, ["strict_positive", "hard_negative"])
    y = labels == "strict_positive"
    selected = logits[mask] >= FIXED_GATE
    strict_selected = int((selected & y).sum())
    # Near-unknown candidates remain in the conservative precision denominator
    # but never in the training/evaluation known-negative pool.
    return {"candidates": int(mask.sum()), "known_candidates": int(known.sum()), "strict_positive": int(y.sum()),
            "selected": int(selected.sum()), "strict_selected": strict_selected,
            "near_unknown_selected": int((selected & (labels == "near_unknown")).sum()),
            "hard_negative_selected": int((selected & (labels == "hard_negative")).sum()),
            "strict_precision_conservative": strict_selected / max(int(selected.sum()), 1),
            "strict_recall": strict_selected / max(int(y.sum()), 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.manual_seed(SEED)
    rows_by_key, cache_info = selected_rows(CACHE)
    selected_train, train_quota = choose_train(rows_by_key)
    shards = sorted(str(p) for p in DATA.joinpath("shards").glob("train-*.tar"))
    train, ev = build_arrays(rows_by_key, selected_train, shards)
    if not len(train["y"]):
        raise ValueError("empty train array")
    geo_mean, geo_std = train["geo"].mean(0), np.maximum(train["geo"].std(0), 1e-5)
    train_geo = (train["geo"] - geo_mean) / geo_std
    eval_norm = {k: {**v, "geo": (v["geo"] - geo_mean) / geo_std} for k, v in ev.items()}
    head = LateFusionHead(train_geo.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    x_pixel, x_geo, y = torch.tensor(train["pixel"]), torch.tensor(train_geo), torch.tensor(train["y"])
    generator = torch.Generator().manual_seed(SEED)
    events = []
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(y), (BATCH,), generator=generator)
        logits = head(x_pixel[idx], x_geo[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    head.eval()
    predictions = {}
    with torch.no_grad():
        for split, data in eval_norm.items():
            predictions[split] = head(torch.tensor(data["pixel"]), torch.tensor(data["geo"])).numpy()
    metrics = {}
    for split, data in eval_norm.items():
        metrics[split] = {style: evaluate(predictions[split], data, style) for style in sorted(set(data["style"]))}
        metrics[split]["all"] = evaluate(predictions[split], data)
    # near_unknown counts are bound from the cache, not reconstructed as labels
    # for the loss; they are reported independently for conservative review.
    near_counts = defaultdict(Counter)
    for rows in rows_by_key.values():
        for row in rows:
            if row["replay_split"] in {"dev_core", "dev_stress"}:
                near_counts[(row["replay_split"], row["variant"])][row["label"]] += 1
    result = {"schema_version": "chemvector-style-factorial-late-fusion-cpu-preflight-0.1",
              "complete": True, "training_performed": True, "training_scope": "cpu_late_fusion_only",
              "gpu_used": False, "optimizer_steps": STEPS, "batch_size": BATCH,
              "checkpoint_written": False, "sealed_test_read": False,
              "cache_sha256": digest(CACHE), "seed": SEED, "patch_px": PATCH, "resized_px": RESIZED,
              "geometry_features": ["raw_score", "presence_logit", "x_norm", "y_norm"],
              "forbidden_features": ["nearest_distance_px", "target_atom_id", "layers", "graph"],
              "fixed_logit_gate": FIXED_GATE, "negative_multiplier": NEGATIVE_MULTIPLIER,
              "cache": cache_info, "train_quota": train_quota,
              "train_rows": len(train["y"]), "eval_rows": {k: len(v["labels"]) for k, v in ev.items()},
              "metrics": metrics, "near_unknown_counts": {f"{s}/{v}": dict(c) for (s, v), c in sorted(near_counts.items())},
              "events": events,
              "gate": {"per_core_style_precision_ge_085": all(metrics["dev_core"][v]["strict_precision_conservative"] >= .85 for v in CORE),
                       "per_core_style_recall_ge_050": all(metrics["dev_core"][v]["strict_recall"] >= .50 for v in CORE),
                       "stress_separate": True, "training_ready": False,
                       "reason": "CPU preflight only; no GPU or production authorization"}}
    result["gate"]["strict_training_ready"] = result["gate"]["per_core_style_precision_ge_085"] and result["gate"]["per_core_style_recall_ge_050"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "train_rows": len(train["y"]), "metrics": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
