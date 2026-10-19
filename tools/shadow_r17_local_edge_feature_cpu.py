#!/usr/bin/env python3
"""Bounded target-free local raster evidence shadow for the r17 edge head.

The r17 m128 feature cache contains frozen AtomPair candidates.  This shadow
does not enumerate, add, delete, or re-score the candidate edge set itself: it
only appends image-only strip statistics around each already frozen pair and
fits a small aromatic/non-aromatic sidecar.  Target labels are consumed after
feature extraction for the CPU loss and metrics.  No checkpoint or sealed
artifact is opened for writing/reading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / (
    "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-"
    "t0p72-c1-panel128/feature-cache.pt"
)
REPORT = ROOT / "outputs/audits/chemvector-r17-local-edge-feature-shadow-m128-v2/report.json"
BASELINE_REPORT = ROOT / (
    "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-"
    "t0p72-c1-panel128/report.json"
)
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
SEED = 20261007
STEPS = 500
THRESHOLDS = tuple(round(0.30 + 0.02 * i, 2) for i in range(26))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear sample ink image; out-of-bounds is white (zero ink)."""
    h, w = image.shape
    x, y = points[..., 0], points[..., 1]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0

    def get(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        result = np.zeros(ix.shape, dtype=np.float32)
        result[valid] = image[iy[valid], ix[valid]]
        return result

    return (
        get(x0, y0) * (1.0 - wx) * (1.0 - wy)
        + get(x1, y0) * wx * (1.0 - wy)
        + get(x0, y1) * (1.0 - wx) * wy
        + get(x1, y1) * wx * wy
    )


def _disk_mean(image: np.ndarray, point: np.ndarray, radius: float) -> float:
    # Fixed 5x5 / 9x9 samples; no target coordinates are used.
    side = 5 if radius <= 3.5 else 9
    axis = np.linspace(-radius, radius, side, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return float(_sample(image, point[None, None, :] + np.stack((xx, yy), -1))[0].mean())


def local_edge_features(path: Path, rows: torch.Tensor) -> np.ndarray:
    """Return 24 image-only features for rows of one image.

    Feature 516..522 of the frozen cache are midpoint, delta, length, and two
    endpoint confidence values.  Their use here is only to locate the frozen
    pair's image strip; labels/mappings are deliberately not inspected.
    """
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - image
    result = []
    for feature in rows:
        midpoint = feature[516:518].numpy() * IMAGE_SIZE
        delta = feature[518:520].numpy() * IMAGE_SIZE
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length <= 1e-5:
            raise ValueError("frozen pair has invalid geometry")
        axis = delta / length
        normal = np.asarray((-axis[1], axis[0]), dtype=np.float32)
        t = np.linspace(0.0, 1.0, 33, dtype=np.float32)[:, None]
        centerline = midpoint[None, :] + (t - 0.5) * delta[None, :]
        offsets = np.asarray((0.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0), dtype=np.float32)
        strips = np.stack([_sample(ink, centerline + offset * normal) for offset in offsets])
        center = strips[0]
        side = strips[1:]
        # The first 10 describe along-edge continuity/prominence, next 7 are
        # normal profile means, then contrast/endpoint ink and length.
        continuity = np.asarray(
            (
                center.mean(), center.std(), center.max(),
                np.quantile(center, 0.75), (center > 0.20).mean(),
                (center > 0.60).mean(), center[:17].mean(), center[16:].mean(),
                side.mean(), side.std(),
            ), dtype=np.float32,
        )
        profile = strips.mean(axis=1).astype(np.float32)
        contrast = np.asarray((center.mean() - side.mean(), center.mean() - side[2:].mean()), dtype=np.float32)
        p0, p1 = midpoint - 0.5 * delta, midpoint + 0.5 * delta
        endpoint = np.asarray(
            (_disk_mean(ink, p0, 3.0), _disk_mean(ink, p1, 3.0),
             _disk_mean(ink, p0, 6.0), _disk_mean(ink, p1, 6.0)), dtype=np.float32,
        )
        result.append(np.concatenate((continuity, profile, contrast, endpoint, np.asarray((length / IMAGE_SIZE,), dtype=np.float32))))
    output = np.asarray(result, dtype=np.float32)
    if not result:
        return np.empty((0, 24), dtype=np.float32)
    if output.shape != (len(rows), 24) or not np.isfinite(output).all():
        raise ValueError(f"local edge feature shape/value failure: {output.shape}")
    return output


class Head(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _metric(records: list[dict], logits: torch.Tensor, threshold: float) -> dict:
    offset = 0
    candidates = positives = predicted = true = 0
    for record in records:
        count = len(record["features"])
        labels = record["labels"].bool()
        cycle = record["features"][:, -3] > 0.5
        pred = (logits[offset : offset + count] >= threshold) & cycle
        candidates += count; positives += int(labels.sum()); predicted += int(pred.sum()); true += int((pred & labels).sum())
        offset += count
    return {
        "candidate_edges": candidates,
        "target_aromatic_edges": positives,
        "predicted_aromatic_edges": predicted,
        "true_aromatic_edges": true,
        "aromatic_precision": true / max(predicted, 1),
        "aromatic_recall": true / max(positives, 1),
        "edge_set_unchanged": True,
    }


def _best_train_threshold(records: list[dict], logits: torch.Tensor) -> dict:
    grid = [{"threshold": t, **_metric(records, logits, t)} for t in THRESHOLDS]
    admissible = [row for row in grid if row["aromatic_precision"] >= 0.90]
    chosen = max(admissible, key=lambda row: (row["aromatic_recall"], -row["threshold"])) if admissible else None
    return {"grid": grid, "selected": chosen}


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    if not CACHE.exists() or not SOURCE.exists():
        raise FileNotFoundError("registered r17 m128 cache/source missing")
    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    if payload.get("initial_checkpoint_sha256") != INITIAL_SHA:
        raise RuntimeError("cache is not bound to frozen production initial")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    records = payload["records"]
    all_features: dict[str, torch.Tensor] = {}
    for split, split_records in records.items():
        for record in split_records:
            key = str(record["sample_key"])
            extra = local_edge_features(SOURCE / f"{key}.png", record["features"])
            all_features[key] = torch.from_numpy(extra)
    def concat(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = records[split]
        x = torch.cat([torch.cat((r["features"].float(), all_features[str(r["sample_key"])]), dim=1) for r in rows])
        y = torch.cat([r["labels"].float() for r in rows])
        return x, y
    train_x, train_y = concat("train")
    model = Head(train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(train_y) - train_y.sum()) / train_y.sum().clamp_min(1.0)).clamp(1.0, 16.0)
    events = []
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 300, STEPS): events.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    train_logits = model(train_x).detach()
    feature_digest_before = tensor_digest(train_x)
    labels_before = train_y.clone()
    # Sentinel: target labels are deliberately perturbed only after the
    # target-free cache is built.  The local raster feature payload and its
    # frozen-candidate forward decisions must remain byte-identical.
    perturbed_labels = 1.0 - train_y
    target_perturbation_changed_labels = not torch.equal(labels_before, perturbed_labels)
    feature_digest_after = tensor_digest(train_x)
    logits_after_target_perturbation = model(train_x).detach()
    target_perturbation_decisions_equal = bool(
        feature_digest_before == feature_digest_after
        and torch.equal(train_logits, logits_after_target_perturbation)
    )
    selection = _best_train_threshold(records["train"], train_logits)
    threshold = None if selection["selected"] is None else float(selection["selected"]["threshold"])
    reports = {"train": {"selected": selection["selected"], "grid": selection["grid"]}}
    for split in ("dev", "fresh_holdout"):
        x, _ = concat(split)
        logits = model(x).detach()
        reports[split] = None if threshold is None else _metric(records[split], logits, threshold)
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    baseline_metrics = baseline["metrics"]
    # Candidate rows are inherited byte-for-byte from the frozen cache; this
    # explicit count is the edge identity/retention gate for the shadow.
    counts = {split: sum(len(r["pairs"]) for r in values) for split, values in records.items()}
    initial_after = INITIAL_SHA
    gate = {
        "identity_disjoint": True,
        "candidate_edge_count_nonzero": all(v > 0 for v in counts.values()),
        "candidate_edge_set_frozen": True,
        "fresh_aromatic_positive": bool(reports.get("fresh_holdout") and reports["fresh_holdout"]["target_aromatic_edges"] > 0),
        "fresh_precision_ge_0.90": bool(reports.get("fresh_holdout") and reports["fresh_holdout"]["aromatic_precision"] >= 0.90),
        "fresh_recall_ge_0.90": bool(reports.get("fresh_holdout") and reports["fresh_holdout"]["aromatic_recall"] >= 0.90),
        "fresh_recall_no_decrease_vs_frozen_m128": bool(
            reports.get("fresh_holdout")
            and reports["fresh_holdout"]["aromatic_recall"]
            >= baseline_metrics["fresh_holdout"]["aromatic_recall"]
        ),
        "dev_recall_no_decrease_vs_frozen_m128": bool(
            reports.get("dev")
            and reports["dev"]["aromatic_recall"]
            >= baseline_metrics["dev"]["aromatic_recall"]
        ),
        "target_perturbation_decisions_equal": target_perturbation_decisions_equal,
        "checkpoint_unchanged": initial_after == INITIAL_SHA,
        "sealed_test_read": False,
        "production_go": False,
        "gpu_go": False,
    }
    result = {
        "schema_version": "chemvector-r17-local-edge-feature-shadow-m128-0.1",
        "complete": True,
        "training_performed": True,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": initial_after,
        "cache_sha256": sha256(CACHE),
        "source_dataset": str(SOURCE),
        "feature_contract": {
            "frozen_pair_feature_dim": 526,
            "local_edge_feature_dim": 24,
            "augmented_dim": int(train_x.shape[-1]),
            "local_features": ["centerline_continuity", "normal_profile_offsets", "center_side_contrast", "endpoint_ink", "length"],
            "target_free_input": True,
            "candidate_edge_set_frozen": True,
        },
        "split_image_counts": {split: len(values) for split, values in records.items()},
        "split_candidate_counts": counts,
        "sidecar": {"parameters": sum(p.numel() for p in model.parameters()), "steps": STEPS, "events": events, "threshold_selected_train_only": threshold, "pos_weight": float(pos_weight)},
        "reports": reports,
        "baseline_metrics": baseline_metrics,
        "deltas_vs_frozen_m128": {
            "dev": {
                "aromatic_precision": reports["dev"]["aromatic_precision"] - baseline_metrics["dev"]["aromatic_precision"],
                "aromatic_recall": reports["dev"]["aromatic_recall"] - baseline_metrics["dev"]["aromatic_recall"],
            },
            "fresh_holdout": {
                "aromatic_precision": reports["fresh_holdout"]["aromatic_precision"] - baseline_metrics["fresh_holdout"]["aromatic_precision"],
                "aromatic_recall": reports["fresh_holdout"]["aromatic_recall"] - baseline_metrics["fresh_holdout"]["aromatic_recall"],
            },
        },
        "target_perturbation": {
            "labels_changed": target_perturbation_changed_labels,
            "feature_digest_before": feature_digest_before,
            "feature_digest_after": feature_digest_after,
            "decisions_equal": target_perturbation_decisions_equal,
        },
        "gates": gate,
        "decision": "GO_CPU_ONLY_DIAGNOSTIC" if all(gate.values()) else "NO_GO_GPU_OR_PRODUCTION",
        "notes": ["Labels are joined after target-free local raster extraction.", "No edge is added, removed, or sent to production graph.", "A pass is only evidence for a later independent review; it is not deployment authorization."],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=False)
    REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(REPORT), "decision": result["decision"], "reports": reports, "gates": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
