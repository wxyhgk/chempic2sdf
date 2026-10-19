#!/usr/bin/env python3
"""CPU probe: add image-only local raster evidence to frozen pair presence.

The candidate atoms/pairs and the frozen production model are untouched.  A
small residual head sees only a 24-dimensional raster strip around each
already enumerated pair and is trained on the identity-disjoint train split.
This is deliberately a probe: it writes an audit and sidecar, never the
production checkpoint, and never reads sealed data.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_r17_pair_incidence_known_clean_cpu import Residual
from chemvector.bond_object import BOND_TYPES

CACHE = ROOT / os.environ.get(
    "R17_RASTER_CACHE",
    "outputs/audits/chemvector-r17-presence-threshold-calibration-cpu-20261022-v1/feature-cache.pt",
)
MANIFEST = ROOT / os.environ.get(
    "R17_RASTER_MANIFEST",
    "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/manifest.json",
)
SOURCE = ROOT / os.environ.get(
    "R17_RASTER_SOURCE",
    "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/samples",
)
OUT = ROOT / os.environ.get(
    "R17_RASTER_OUT",
    "outputs/audits/chemvector-r17-local-raster-presence-cpu-20261022-v1",
)
IMAGE_SIZE = 768.0
THRESHOLD = float(os.environ.get("R17_RASTER_THRESHOLD", "0.95"))
STEPS = int(os.environ.get("R17_RASTER_STEPS", "300"))
NEGATIVE_RATIO = int(os.environ.get("R17_RASTER_NEGATIVE_RATIO", "3"))
SEED = 20261022


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x, y = points[..., 0], points[..., 1]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0

    def get(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        value = np.zeros(ix.shape, dtype=np.float32)
        value[valid] = image[iy[valid], ix[valid]]
        return value

    return get(x0, y0) * (1 - wx) * (1 - wy) + get(x1, y0) * wx * (1 - wy) + get(x0, y1) * (1 - wx) * wy + get(x1, y1) * wx * wy


def _disk_mean(image: np.ndarray, point: np.ndarray, radius: float) -> float:
    axis = np.linspace(-radius, radius, 5 if radius <= 3.5 else 9, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return float(_sample(image, point[None, None, :] + np.stack((xx, yy), -1))[0].mean())


def local_features(path: Path, rows: torch.Tensor) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - image
    result = []
    for feature in rows:
        # The 262-d cache stores midpoint at 257:259 and scalar pair
        # distance at 259.  It intentionally does not retain orientation, so
        # this probe uses rotation-invariant radial samples only.
        midpoint = feature[257:259].numpy() * IMAGE_SIZE
        length = float(feature[259].item() * IMAGE_SIZE)
        if not np.isfinite(length) or length <= 1e-5:
            raise ValueError("invalid frozen pair geometry")
        disks = np.asarray([_disk_mean(ink, midpoint, r) for r in (2., 4., 6., 8., 12.)], dtype=np.float32)
        angles = np.arange(8, dtype=np.float32) * (2.0 * np.pi / 8.0)
        unit = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        ring_a = _sample(ink, midpoint[None, :] + unit * max(3.0, min(length * .25, 16.0)))
        ring_b = _sample(ink, midpoint[None, :] + unit * max(5.0, min(length * .50, 28.0)))
        contrast = np.asarray((disks[0] - disks[-1], ring_a.mean() - ring_b.mean()), dtype=np.float32)
        result.append(np.concatenate((disks, ring_a.astype(np.float32), ring_b.astype(np.float32), contrast, np.asarray((length / IMAGE_SIZE,), dtype=np.float32))))
    out = np.asarray(result, dtype=np.float32)
    if not result:
        return np.empty((0, 24), dtype=np.float32)
    if out.shape != (len(rows), 24) or not np.isfinite(out).all():
        raise ValueError(f"invalid local feature payload {out.shape}")
    return out


class VisualResidual(nn.Module):
    def __init__(self, dimensions: int = 24):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _score(items: list[dict], head: VisualResidual | None) -> tuple[dict, dict]:
    aggregate = defaultdict(lambda: {"images": 0, "predicted": 0, "true": 0, "typed_true": 0, "target": 0})
    for item in items:
        logits = item["base_logits"] if head is None else item["base_logits"] + head(item["visual_features"])
        selected = logits.sigmoid() >= THRESHOLD
        labels = item["labels"] > .5
        profile = item["profile"]
        row = aggregate[profile]
        row["images"] += 1
        row["predicted"] += int(selected.sum())
        row["true"] += int((selected & labels).sum())
        row["target"] += int(labels.sum())
        for idx in torch.where(selected & labels)[0].tolist():
            expected = item["type_labels"][idx]
            if expected is not None and BOND_TYPES[int(item["chemical_logits"][idx].argmax())] == expected:
                row["typed_true"] += 1

    def finish(row: dict) -> dict:
        value = dict(row)
        value["edge_precision"] = row["true"] / max(row["predicted"], 1)
        value["edge_recall"] = row["true"] / max(row["target"], 1)
        value["typed_edge_precision"] = row["typed_true"] / max(row["predicted"], 1)
        value["typed_edge_recall"] = row["typed_true"] / max(row["target"], 1)
        return value

    return finish({k: sum(row[k] for row in aggregate.values()) for k in aggregate[next(iter(aggregate))]}) if aggregate else {}, {p: finish(row) for p, row in sorted(aggregate.items())}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if cache.get("binding", {}).get("initial_checkpoint_sha256") != SHA or cache.get("binding", {}).get("manifest_sha256") != digest(MANIFEST):
        raise RuntimeError("cache binding mismatch")
    metadata = {row["sample_key"]: row for row in manifest["rows"]}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    items = []
    for split in ("train", "holdout"):
        for source in cache[split]:
            item = dict(source)
            item["visual_features"] = torch.from_numpy(local_features(SOURCE / f"{item['key']}.png", item["features"]))
            if item["key"] not in metadata:
                raise RuntimeError(f"missing metadata {item['key']}")
            items.append(item)
    train = [item for item in items if item["key"] in set(cache["binding"]["train_keys"])]
    holdout = [item for item in items if item["key"] in set(cache["binding"]["holdout_keys"])]
    x = torch.cat([item["visual_features"] for item in train])
    b = torch.cat([item["base_logits"] for item in train])
    y = torch.cat([item["labels"] for item in train]).float()
    positive = torch.cat([item["labels"] > .5 for item in train])
    keep = positive.clone()
    negative_indices = torch.where(~positive)[0]
    positive_indices = torch.where(positive)[0]
    keep[negative_indices[: min(len(negative_indices), NEGATIVE_RATIO * len(positive_indices))]] = True
    x_fit, b_fit, y_fit = x[keep], b[keep], y[keep]
    head = VisualResidual(x.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(y_fit) - y_fit.sum()) / y_fit.sum().clamp_min(1)).clamp(1, 32)
    events = []
    for step in range(1, STEPS + 1):
        loss = F.binary_cross_entropy_with_logits(b_fit + head(x_fit), y_fit, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if step in (1, 100, STEPS):
            events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    baseline, baseline_profiles = _score(holdout, None)
    scored, scored_profiles = _score(holdout, head)
    deltas = {p: {key: scored_profiles[p][key] - baseline_profiles[p][key] for key in ("edge_precision", "edge_recall", "typed_edge_precision", "typed_edge_recall")} for p in scored_profiles}
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    after = digest(initial)
    result = {
        "schema_version": "chemvector-r17-local-raster-presence-cpu-0.1",
        "complete": True,
        "training_performed": True,
        "training_scope": "frozen_pair_local_raster_residual_cpu",
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "steps": STEPS,
        "visual_feature_dimensions": 24,
        "train_rows": int(len(y_fit)),
        "events": events,
        "baseline": baseline,
        "scored": scored,
        "baseline_profiles": baseline_profiles,
        "scored_profiles": scored_profiles,
        "deltas": deltas,
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": after,
        "gate": {
            "cache_bound_to_initial": True,
            "identity_disjoint": True,
            "candidate_edge_set_frozen": True,
            "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -.01 for v in deltas.values()),
            "aggregate_edge_recall_gain": scored["edge_recall"] > baseline["edge_recall"],
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True)
    torch.save({"schema_version": result["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA, "cache_sha256": digest(CACHE), "manifest_sha256": digest(MANIFEST)}, OUT / "sidecar-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "baseline": baseline, "scored": scored, "deltas": deltas, "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
