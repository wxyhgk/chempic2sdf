#!/usr/bin/env python3
"""Read-only CPU sweep of the LSD proposal front-end.

This audit intentionally stops before ownership, sidecars, or graph repair.  It
compares detector knobs (LSD refine mode, minimum segment length and raster
scale) together with endpoint-cluster radius/mode.  Train metrics select one
configuration; the fixed holdout is then evaluated once for every selected
candidate.  Ground-truth atom centres and bonds are used only after proposal
construction for scoring.

The production detector is not modified by this script.  In particular, no
checkpoint, selector, OCR, bond head, or production graph is written and no
sealed split is read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
DEFAULT_OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-config-sweep-r5"


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def cluster_single_link(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Current endpoint clustering: deterministic single-link components."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [N,2]")
    dsu = _DSU(len(points))
    for i in range(len(points)):
        distance = np.linalg.norm(points[i + 1 :] - points[i], axis=-1)
        for offset in np.where(distance <= radius)[0].tolist():
            dsu.union(i, i + 1 + int(offset))
    return _labels_centres(dsu, points)


def cluster_greedy_centroid(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Bounded alternative that avoids long single-link chains.

    Endpoints are visited in detector order.  A point joins the nearest
    existing centre only when that centre is within ``radius``; the centre is
    updated online.  This is target-free and deterministic, but differs from
    single-link when a sequence of endpoints is individually close yet its
    total spread is large.
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [N,2]")
    centres: list[np.ndarray] = []
    counts: list[int] = []
    labels = np.full(len(points), -1, dtype=np.int64)
    for i, point in enumerate(points):
        if centres:
            distances = np.linalg.norm(np.asarray(centres) - point[None, :], axis=-1)
            label = int(np.argmin(distances))
            if float(distances[label]) <= radius:
                labels[i] = label
                count = counts[label]
                centres[label] = (centres[label] * count + point) / float(count + 1)
                counts[label] = count + 1
                continue
        labels[i] = len(centres)
        centres.append(point.astype(np.float64, copy=True))
        counts.append(1)
    return labels, np.asarray(centres, dtype=np.float64)


def _labels_centres(dsu: _DSU, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(points)):
        groups[dsu.find(i)].append(i)
    roots = sorted(groups, key=lambda root: min(groups[root]))
    labels = np.full(len(points), -1, dtype=np.int64)
    centres = []
    for label, root in enumerate(roots):
        members = groups[root]
        labels[members] = label
        centres.append(points[members].mean(axis=0))
    return labels, np.asarray(centres, dtype=np.float64)


def _canonical(segment: np.ndarray) -> np.ndarray:
    value = np.asarray(segment, dtype=np.float64).reshape(2, 2)
    if (float(value[0, 1]), float(value[0, 0])) > (float(value[1, 1]), float(value[1, 0])):
        return value[::-1].copy()
    return value.copy()


def detect_segments(image: Image.Image, *, minimum_length: float, refine: str,
                    scale: float, border_margin: float, maximum_candidates: int = 512) -> np.ndarray:
    """Detect LSD segments in a temporary raster coordinate frame."""
    if refine not in {"STD", "ADV"}:
        raise ValueError(refine)
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    original_h, original_w = grayscale.shape
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale != 1.0:
        resized = cv2.resize(grayscale, (max(1, round(original_w * scale)), max(1, round(original_h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    else:
        resized = grayscale
    mode = cv2.LSD_REFINE_ADV if refine == "ADV" else cv2.LSD_REFINE_STD
    detector = cv2.createLineSegmentDetector(mode)
    detected = detector.detect(255 - resized)[0]
    if detected is None:
        return np.empty((0, 2, 2), dtype=np.float64)
    h, w = resized.shape
    segments: list[np.ndarray] = []
    margin = float(border_margin) * scale
    for raw in detected:
        seg = _canonical(np.asarray(raw).reshape(-1)[:4])
        if float(np.linalg.norm(seg[1] - seg[0])) < float(minimum_length) * scale:
            continue
        x, y = seg[:, 0], seg[:, 1]
        if margin > 0 and (np.all(x <= margin) or np.all(x >= (w - 1) - margin) or np.all(y <= margin) or np.all(y >= (h - 1) - margin)):
            continue
        if scale != 1.0:
            seg = seg / float(scale)
        segments.append(seg)
    if not segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    # Keep strongest/longest candidates only when the detector emits a flood.
    segments.sort(key=lambda s: -float(np.linalg.norm(s[1] - s[0])))
    return np.stack(segments[: int(maximum_candidates)])


def nearest_mapping(centres: np.ndarray, targets: np.ndarray, radius: float) -> dict[int, int]:
    if len(centres) == 0 or len(targets) == 0:
        return {}
    distances = np.linalg.norm(centres[:, None] - targets[None, :], axis=-1)
    mapping: dict[int, int] = {}
    for row in range(len(centres)):
        col = int(distances[row].argmin())
        if float(distances[row, col]) <= radius:
            mapping[row] = col
    return mapping


def audit_sample(row: dict, *, minimum_length: float, refine: str, scale: float,
                 border_margin: float, cluster_radius: float, cluster_mode: str,
                 target_radius: float) -> dict:
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(base.with_suffix(".png")).convert("L")
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    segments = detect_segments(image, minimum_length=minimum_length, refine=refine,
                               scale=scale, border_margin=border_margin)
    points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2), dtype=np.float64)
    if cluster_mode == "single_link":
        labels, centres = cluster_single_link(points, cluster_radius)
    elif cluster_mode == "greedy_centroid":
        labels, centres = cluster_greedy_centroid(points, cluster_radius)
    else:
        raise ValueError(cluster_mode)
    edges = {tuple(sorted((int(labels[2 * i]), int(labels[2 * i + 1]))))
             for i in range(len(segments)) if int(labels[2 * i]) != int(labels[2 * i + 1])}
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float64)
    mapping = nearest_mapping(centres, targets, target_radius)
    mapped = {tuple(sorted((mapping[a], mapping[b]))) for a, b in edges
              if a in mapping and b in mapping and mapping[a] != mapping[b]}
    gt_edges = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    true = mapped & gt_edges
    return {
        "sample_key": row["sample_key"], "split": row["split"],
        "segments": int(len(segments)), "provisional_nodes": int(len(centres)),
        "provisional_edges": int(len(edges)), "mapped_nodes": int(len(mapping)),
        "gt_atoms": int(len(targets)), "gt_bonds": int(len(gt_edges)),
        "mapped_edges": int(len(mapped)), "true_edges": int(len(true)),
        "bond_recall": float(len(true) / len(gt_edges)) if gt_edges else 1.0,
        "edge_precision": float(len(true) / len(mapped)) if mapped else 1.0,
        "all_bonds_recovered": bool(true == gt_edges),
    }


def rows(split: str) -> list[dict]:
    values = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    out = [r for r in values if r.get("split") == split and r.get("render_status", "accepted") == "accepted"]
    out.sort(key=lambda r: r["sample_key"])
    return out


def summarize(values: list[dict], metadata: dict[str, dict]) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([float(v[name]) for v in values])) if values else 0.0
    grouped: dict[str, list[dict]] = defaultdict(list)
    for value in values:
        grouped[metadata[value["sample_key"]]["profile_id"]].append(value)
    return {
        "images": len(values), "mean_bond_recall": mean("bond_recall"),
        "mean_edge_precision": mean("edge_precision"),
        "images_all_bonds_recovered": int(sum(bool(v["all_bonds_recovered"]) for v in values)),
        "mean_provisional_nodes": mean("provisional_nodes"),
        "mean_provisional_edges": mean("provisional_edges"),
        "by_profile": {profile: {
            "images": len(group), "mean_bond_recall": float(np.mean([v["bond_recall"] for v in group])),
            "mean_edge_precision": float(np.mean([v["edge_precision"] for v in group])),
            "images_all_bonds_recovered": int(sum(bool(v["all_bonds_recovered"]) for v in group)),
        } for profile, group in sorted(grouped.items())},
    }


def _audit_kwargs(config: dict) -> dict:
    """Translate serialized config names to ``audit_sample`` parameters."""
    return {
        "minimum_length": config["minimum_length_px"], "refine": config["refine"],
        "scale": config["scale"], "border_margin": config["border_margin_px"],
        "cluster_radius": config["cluster_radius_px"], "cluster_mode": config["cluster_mode"],
        "target_radius": config["target_radius_px"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-lengths", default="2,3,5,8,12")
    parser.add_argument("--refines", default="STD,ADV")
    parser.add_argument("--scales", default="0.75,1.0,1.25")
    parser.add_argument("--cluster-radii", default="8,10,12,16")
    parser.add_argument("--cluster-modes", default="single_link,greedy_centroid")
    parser.add_argument("--target-radius", type=float, default=24.0)
    parser.add_argument("--border-margin", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    lengths = [float(x) for x in args.minimum_lengths.split(",") if x.strip()]
    refines = [x.strip().upper() for x in args.refines.split(",") if x.strip()]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    radii = [float(x) for x in args.cluster_radii.split(",") if x.strip()]
    modes = [x.strip() for x in args.cluster_modes.split(",") if x.strip()]
    train_rows, holdout_rows = rows("train"), rows("holdout")
    metadata = {r["sample_key"]: r for r in rows("train") + rows("holdout")}
    configs = [{"minimum_length_px": l, "refine": f, "scale": s, "cluster_radius_px": r, "cluster_mode": m,
                "border_margin_px": args.border_margin, "target_radius_px": args.target_radius}
               for l in lengths for f in refines for s in scales for r in radii for m in modes]
    train_sweeps = []
    for index, config in enumerate(configs, 1):
        values = [audit_sample(row, **_audit_kwargs(config)) for row in train_rows]
        train_sweeps.append({**config, **summarize(values, metadata)})
        print(f"train {index}/{len(configs)} {config} recall={train_sweeps[-1]['mean_bond_recall']:.4f} precision={train_sweeps[-1]['mean_edge_precision']:.4f}")
    eligible = [v for v in train_sweeps if v["mean_edge_precision"] >= 0.85]
    selected = max(eligible or train_sweeps, key=lambda v: (v["mean_bond_recall"], v["mean_edge_precision"], -v["minimum_length_px"]))
    # Report holdout for all configurations as a diagnostic, but selection is train-only.
    holdout_sweeps = []
    for index, config in enumerate(configs, 1):
        values = [audit_sample(row, **_audit_kwargs(config)) for row in holdout_rows]
        holdout_sweeps.append({**config, **summarize(values, metadata)})
        print(f"holdout {index}/{len(configs)} {config} recall={holdout_sweeps[-1]['mean_bond_recall']:.4f} precision={holdout_sweeps[-1]['mean_edge_precision']:.4f}")
    chosen_holdout = next(v for v in holdout_sweeps if all(v[k] == selected[k] for k in ("minimum_length_px", "refine", "scale", "cluster_radius_px", "cluster_mode", "border_margin_px", "target_radius_px")))
    baseline = next((v for v in holdout_sweeps if v["minimum_length_px"] == 5.0 and v["refine"] == "STD" and v["scale"] == 1.0 and v["cluster_radius_px"] == 10.0 and v["cluster_mode"] == "single_link"), None)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-lsd-config-sweep-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "temporary LSD detector -> target-free endpoint clustering -> provisional unordered edges; target graph post-hoc only",
        "split": {"train_images": len(train_rows), "holdout_images": len(holdout_rows)},
        "selection": {"source": "train-only", "selected": selected},
        "selected_holdout": chosen_holdout, "baseline_holdout": baseline,
        "train_sweeps": train_sweeps, "holdout_sweeps": holdout_sweeps,
        "gate": {"production_go": False, "gpu_shadow_admissible": False,
                 "reason": "proposal-only CPU comparison; no candidate was authorized for training"},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output / "report.json"), "selected": selected,
                      "selected_holdout": chosen_holdout, "baseline_holdout": baseline}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
