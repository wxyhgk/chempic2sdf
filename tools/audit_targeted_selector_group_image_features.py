"""Read-only audit of image-only local candidate features.

The released train/holdout group caches provide candidate coordinates produced
by the frozen initial model.  This script reads only the corresponding source
PNGs and those coordinates, extracts deterministic local ink features, and
reports post-hoc separability against the already-recorded cache labels.  It
does not read sealed data, layers/graph/text supervision, a checkpoint, or run
an optimizer.  Labels are consulted only *after* feature extraction for the
diagnostic AUC table.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-group-release-20261006-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-group-image-features-20261006-r1/report.json"
TRAIN_CACHE = RELEASE / "train-group-cache.json"
HOLDOUT_CACHE = RELEASE / "holdout-group-cache.json"

WINDOW = 40  # candidate-centred square in source 768x768 raster coordinates
THRESHOLDS = (64, 128, 192, 245)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _components(mask: np.ndarray) -> tuple[int, int, int]:
    """Return component count, largest area, and total foreground pixels."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=np.bool_)
    count = largest = total = 0
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        count += 1
        stack = [(int(y), int(x))]
        seen[y, x] = True
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            total += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        largest = max(largest, area)
    return count, largest, total


def extract(image: Image.Image, x_norm: float, y_norm: float) -> dict[str, float]:
    """Extract deterministic raster-only features around a candidate point."""
    gray = np.asarray(image, dtype=np.uint8)
    h, w = gray.shape
    cx, cy = float(x_norm) * w, float(y_norm) * h
    half = WINDOW // 2
    ix, iy = round(cx), round(cy)
    # Out-of-frame padding is white; no GT geometry is used to choose crop.
    crop = np.full((WINDOW, WINDOW), 255, dtype=np.uint8)
    left, top = ix - half, iy - half
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(w, left + WINDOW), min(h, top + WINDOW)
    crop[y0 - top:y1 - top, x0 - left:x1 - left] = gray[y0:y1, x0:x1]
    f: dict[str, float] = {}
    for threshold in THRESHOLDS:
        mask = crop < threshold
        f[f"ink_fraction_t{threshold}"] = float(mask.mean())
        f[f"ink_center_t{threshold}"] = float(mask[WINDOW // 4:3 * WINDOW // 4,
                                                    WINDOW // 4:3 * WINDOW // 4].mean())
    mask = crop < 245
    count, largest, total = _components(mask)
    f.update(component_count=float(count), largest_component_fraction=float(largest / (WINDOW * WINDOW)),
             total_ink_fraction=float(total / (WINDOW * WINDOW)))
    yy, xx = np.nonzero(mask)
    if len(xx):
        dx, dy = xx + 0.5 - half, yy + 0.5 - half
        mx, my = float(dx.mean() / WINDOW), float(dy.mean() / WINDOW)
        vx, vy = float(dx.var() / (WINDOW * WINDOW)), float(dy.var() / (WINDOW * WINDOW))
        cov = float(((dx - dx.mean()) * (dy - dy.mean())).mean() / (WINDOW * WINDOW))
        disc = math.sqrt(max((vx - vy) ** 2 + 4.0 * cov * cov, 0.0))
        major, minor = max((vx + vy + disc) / 2.0, 1e-12), max((vx + vy - disc) / 2.0, 1e-12)
        f.update(ink_centroid_x=mx, ink_centroid_y=my, ink_var_x=vx, ink_var_y=vy,
                 ink_covariance=cov, ink_anisotropy=float(major / minor))
    else:
        f.update(ink_centroid_x=0.0, ink_centroid_y=0.0, ink_var_x=0.0, ink_var_y=0.0,
                 ink_covariance=0.0, ink_anisotropy=0.0)
    # Four quadrant occupancies distinguish glyphs from a thin bond segment.
    for name, sl in {"nw": (slice(0, half), slice(0, half)),
                     "ne": (slice(0, half), slice(half, WINDOW)),
                     "sw": (slice(half, WINDOW), slice(0, half)),
                     "se": (slice(half, WINDOW), slice(half, WINDOW))}.items():
        f[f"quadrant_{name}"] = float(mask[sl].mean())
    # Central horizontal/vertical strips are image-only endpoint/junction cues.
    f["central_horizontal"] = float(mask[half - 1:half + 1, :].mean())
    f["central_vertical"] = float(mask[:, half - 1:half + 1].mean())
    f["ring_outer"] = float(mask[~_disk_mask(WINDOW, 0.30)].mean())
    f["ring_inner"] = float(mask[_disk_mask(WINDOW, 0.30)].mean())
    return f


def _disk_mask(size: int, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    return ((xx - center) ** 2 + (yy - center) ** 2) <= (radius * size) ** 2


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    # Rank implementation avoids materialising the 67 x 13k Cartesian product.
    values = [(v, 1) for v in positive] + [(v, 0) for v in negative]
    values.sort(key=lambda item: item[0])
    rank = 0
    pos_rank_sum = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        midrank = (i + 1 + j) / 2.0
        pos_rank_sum += midrank * sum(label for _, label in values[i:j])
        i = j
    return (pos_rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative))


def _cache_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["groups"]


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows_by_split = {"train": _cache_rows(TRAIN_CACHE), "holdout": _cache_rows(HOLDOUT_CACHE)}
    images: dict[str, Image.Image] = {}
    extracted = []
    feature_names: tuple[str, ...] | None = None
    source_png_digests: dict[str, str] = {}
    for split, groups in rows_by_split.items():
        for group in groups:
            key = group["sample_key"]
            if key not in images:
                png = RELEASE / "samples" / f"{key}.png"
                if not png.is_file():
                    raise ValueError(f"Missing source PNG: {png}")
                # Explicitly do not open .layers/.graph/.text records.
                source_png_digests[key] = digest(png)
                images[key] = Image.open(png).convert("L")
            for member in group["members"]:
                features = extract(images[key], member["feature"][3], member["feature"][4])
                feature_names = tuple(features) if feature_names is None else feature_names
                if tuple(features) != feature_names:
                    raise ValueError("Feature schema changed during extraction")
                # Labels are attached after extraction; no label enters extract().
                strict = member["raw_target"] >= 0 and member["selected_target"] < 0
                negative = (member["raw_target"] < 0 and member["selected_target"] < 0
                            and group["confirmed_negative"])
                if strict or negative:
                    extracted.append({"split": split, "features": features,
                                      "label": "positive" if strict else "negative"})
    if feature_names is None:
        raise ValueError("No candidate features")
    by_split: dict[str, dict] = {}
    for split in rows_by_split:
        examples = [e for e in extracted if e["split"] == split]
        auc = {}
        for name in feature_names:
            pos = [e["features"][name] for e in examples if e["label"] == "positive"]
            neg = [e["features"][name] for e in examples if e["label"] == "negative"]
            auc[name] = {"positive_count": len(pos), "negative_count": len(neg),
                         "auc_positive_higher": _auc(pos, neg)}
        by_split[split] = {"examples": len(examples),
                           "positive": sum(e["label"] == "positive" for e in examples),
                           "negative": sum(e["label"] == "negative" for e in examples),
                           "feature_auc": auc}
    result = {
        "schema_version": "chemvector-targeted-selector-group-image-features-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "checkpoint_read": False, "gpu_used": False,
        "sealed_test_read": False, "layers_read": False, "graph_or_text_supervision_read": False,
        "source": {"release": str(RELEASE.relative_to(ROOT)), "train_cache_sha256": digest(TRAIN_CACHE),
                   "holdout_cache_sha256": digest(HOLDOUT_CACHE), "png_count": len(images),
                   "png_sha256": source_png_digests},
        "crop": {"window_px": WINDOW, "center": "candidate normalized x/y from frozen cache",
                 "thresholds": list(THRESHOLDS), "feature_names": list(feature_names)},
        "splits": by_split,
        "interpretation": {
            "features_are_image_only": True,
            "labels_used_only_post_extraction": True,
            "not_a_training_result": True,
            "next_step": "If holdout AUC is weak or train/holdout ordering diverges, add a frozen visual encoder crop branch; do not retry the failed pointwise scorer.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "png_count": len(images), "splits": by_split}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
