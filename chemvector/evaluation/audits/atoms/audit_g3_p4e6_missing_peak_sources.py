#!/usr/bin/env python3
"""Localize every P4-E3 dev GT missed by the practical top1024 peak set."""

from __future__ import annotations

from collections import Counter
import copy
import html
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    build_atom_center_field_targets,
    decode_atom_center_field,
)
from chemvector.components.vision.regional_maxima import regional_maxima_mask
from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1d_boundary_band_set_ranker as p1d,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.atoms import (
    train_g3_p4e3_global_peak_rank_pilot20k as p4e3,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
    load_frozen_vision_checkpoint,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e6-missing-peak-sources"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P4E5_AUDIT = Path(
    "outputs/audits/chemvector-g3-p4e5-peak-budget-dev1024/audit.json"
)
P4E5_AUDIT_SHA256 = (
    "a2fcf57fd393a570f15218b39b3a7988226f6c56d1cdd49aeddd3f0f672929b5"
)
P4E3_CHECKPOINT = p4e3.OUTPUT / f"step-{p4e3.STEPS:06d}.pt"
P4E3_CHECKPOINT_SHA256 = (
    "1c358d0c936c687fa0e40ea5e1d9364dd98a663bb762d294db76b43d5f009593"
)
TOP_K = 1024
ALL_PEAK_LIMIT = 200_000
MATCH_RADIUS = 6.0
IMAGE_SIZE = 768
BATCH_SIZE = 8
EXPECTED_MATCHED = 32_342
EXPECTED_MISSING = 58
EXPECTED_FAILURE_IMAGES = 43
SEED = 20260846


def maximum_cardinality_assignments(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    radius_pixels: float = MATCH_RADIUS,
    image_size: float = IMAGE_SIZE,
) -> list[int | None]:
    """Return the same maximum-cardinality assignment used by G3 metrics."""

    if predicted.ndim != 2 or predicted.shape[-1] != 2:
        raise ValueError("P4-E6 predicted anchors must have shape [P,2]")
    if expected.ndim != 2 or expected.shape[-1] != 2:
        raise ValueError("P4-E6 expected anchors must have shape [G,2]")
    if min(radius_pixels, image_size) <= 0.0:
        raise ValueError("P4-E6 matching dimensions must be positive")
    if not len(expected):
        return []
    if not len(predicted):
        return [None] * len(expected)

    distances = torch.cdist(expected.float(), predicted.float()) * image_size
    eligible_columns = (distances <= radius_pixels).any(dim=0)
    eligible_indices = torch.where(eligible_columns)[0]
    eligible = distances[:, eligible_columns]
    target_count = len(expected)
    unmatched_cost = float(target_count + 1)
    real_cost = torch.where(
        eligible <= radius_pixels,
        eligible / radius_pixels,
        torch.full_like(eligible, 2.0 * unmatched_cost),
    )
    dummy_cost = torch.full(
        (target_count, target_count), unmatched_cost, dtype=distances.dtype
    )
    cost = torch.cat((real_cost, dummy_cost), dim=1)
    rows, columns = linear_sum_assignment(cost.numpy())
    assignments: list[int | None] = [None] * target_count
    for row, column in zip(rows, columns):
        if column < len(eligible_indices) and float(eligible[row, column]) <= radius_pixels:
            assignments[row] = int(eligible_indices[column])
    return assignments


def classify_missing_peak(
    *,
    top1024_within_radius: bool,
    all_peaks_within_radius: bool,
    gt_cell_is_regional_maximum: bool,
) -> str:
    if top1024_within_radius:
        return "matching_conflict"
    if all_peaks_within_radius:
        return "below_top1024"
    if not gt_cell_is_regional_maximum:
        return "gt_cell_not_regional_maximum"
    return "regional_peak_anchor_outside_radius"


def _load_model(device: torch.device):
    payload = torch.load(P4E3_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("model_type") != p4e3.MODEL_TYPE:
        raise ValueError("P4-E6 source model type differs")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.load_state_dict(payload["vision_last_stride4_block"], strict=True)
    block.requires_grad_(False).eval()
    field = ContinuousStride1AtomCenterField(atom_head).to(device)
    field.load_state_dict(payload["atom_center_field"], strict=True)
    field.requires_grad_(False).eval()
    return field, atom_head, vision, frozen_block, vision_sha


def _pixel_image(pixel_values: torch.Tensor) -> Image.Image:
    array = (
        (pixel_values.detach().cpu().permute(1, 2, 0) * 0.5 + 0.5)
        .clamp(0.0, 1.0)
        .mul(255)
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _draw_overlay(
    pixels: torch.Tensor,
    expected: torch.Tensor,
    assignments: list[int | None],
    failures: list[dict],
    path: Path,
) -> None:
    image = _pixel_image(pixels)
    draw = ImageDraw.Draw(image)
    missing = {int(item["target_index"]): item for item in failures}
    for index, anchor in enumerate(expected):
        x = float(anchor[0]) * IMAGE_SIZE
        y = float(anchor[1]) * IMAGE_SIZE
        color = "#00a65a" if assignments[index] is not None else "#e31a1c"
        radius = 4 if assignments[index] is not None else 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        if index in missing:
            item = missing[index]
            nearest = item.get("nearest_all_peak_xy")
            if nearest is not None:
                px, py = nearest
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline="#ff8c00", width=2)
                draw.line((x, y, px, py), fill="#ff8c00", width=1)
            draw.text((x + 10, y - 8), f"{index}:{item['category']}", fill="#e31a1c")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_index(rows: list[dict]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<article><img src='overlays/{file}'><h2>{key}</h2>"
            "<p>{summary}</p></article>".format(
                file=html.escape(row["overlay"]),
                key=html.escape(row["sample_key"]),
                summary=html.escape(
                    ", ".join(
                        f"GT#{item['target_index']} {item['category']}"
                        for item in row["failures"]
                    )
                ),
            )
        )
    document = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>P4-E6 漏峰审计</title><style>
body{font-family:system-ui;margin:24px;background:#f4f5f7;color:#17191c}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
article{background:white;border:1px solid #d8dce2;padding:12px;border-radius:6px}
img{width:100%;height:auto;background:white}h2{font-size:15px}p{font-size:13px}
</style></head><body><h1>P4-E6 漏峰审计</h1><main>""" + "".join(cards) + "</main></body></html>"
    (OUTPUT / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E6 output is not empty: {OUTPUT}")
    for path, expected in (
        (P4E5_AUDIT, P4E5_AUDIT_SHA256),
        (P4E3_CHECKPOINT, P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E6 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E6 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    dev, manifest_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    field, atom_head, vision, frozen_block, vision_sha = _load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=IMAGE_SIZE)
    started = time.perf_counter()
    failures = []
    image_rows = []
    matched_total = 0

    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(dev, collator, batch_size=BATCH_SIZE):
            objects, stroke = p1t.model_inputs(
                atom_head,
                vision,
                frozen_block,
                batch["pixel_values"],
                device=device,
                training=False,
            )
            with _autocast_context(device, "bf16"):
                output = field(objects, stroke)
            dense = decode_atom_center_field(output)
            all_peaks = extract_regional_maximum_objects(
                dense, maximum_objects=ALL_PEAK_LIMIT
            )
            maxima_masks = regional_maxima_mask(dense["scores"], connectivity=8).cpu()
            targets = build_atom_center_field_targets(
                batch["atom_anchors"].to(device),
                batch["atom_mask"].to(device),
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            )
            positive_masks = targets["positive"].cpu()
            logits = output["center_logits"].detach().float().cpu()[:, 0]

            for image_index, sample_key_value in enumerate(batch["keys"]):
                sample_key = str(sample_key_value)
                expected = batch["atom_anchors"][image_index][
                    batch["atom_mask"][image_index]
                ].float()
                peaks = all_peaks[image_index]
                top_anchors = peaks["anchors"][:TOP_K]
                assignments = maximum_cardinality_assignments(top_anchors, expected)
                matched_total += sum(value is not None for value in assignments)
                current_failures = []
                distances_top = torch.cdist(expected, top_anchors) * IMAGE_SIZE
                distances_all = torch.cdist(expected, peaks["anchors"]) * IMAGE_SIZE
                expected_pixels = expected * IMAGE_SIZE
                gt_distances = torch.cdist(expected_pixels, expected_pixels)
                gt_distances.fill_diagonal_(torch.inf)
                gt_cells = torch.floor(expected_pixels).clamp(0, IMAGE_SIZE - 1)
                gt_cell_distances = (
                    gt_cells[:, None] - gt_cells[None, :]
                ).abs().amax(dim=-1)
                gt_cell_distances.fill_diagonal_(torch.inf)

                for target_index, assigned in enumerate(assignments):
                    if assigned is not None:
                        continue
                    nearest_top_distance, nearest_top_index = distances_top[
                        target_index
                    ].min(dim=0)
                    nearest_all_distance, nearest_all_index = distances_all[
                        target_index
                    ].min(dim=0)
                    anchor = expected[target_index]
                    cell_x = min(int(float(anchor[0]) * IMAGE_SIZE), IMAGE_SIZE - 1)
                    cell_y = min(int(float(anchor[1]) * IMAGE_SIZE), IMAGE_SIZE - 1)
                    gt_cell_regional = bool(maxima_masks[image_index, cell_y, cell_x])
                    top_within = bool(nearest_top_distance <= MATCH_RADIUS)
                    all_within = bool(nearest_all_distance <= MATCH_RADIUS)
                    category = classify_missing_peak(
                        top1024_within_radius=top_within,
                        all_peaks_within_radius=all_within,
                        gt_cell_is_regional_maximum=gt_cell_regional,
                    )
                    top = max(cell_y - 1, 0)
                    bottom = min(cell_y + 2, IMAGE_SIZE)
                    left = max(cell_x - 1, 0)
                    right = min(cell_x + 2, IMAGE_SIZE)
                    local = logits[image_index, top:bottom, left:right]
                    flat_local = int(local.argmax())
                    local_width = right - left
                    competitor_y = top + flat_local // local_width
                    competitor_x = left + flat_local % local_width
                    competitor_positive = bool(
                        positive_masks[image_index, 0, competitor_y, competitor_x]
                    )
                    nearest_peak = peaks["anchors"][int(nearest_all_index)] * IMAGE_SIZE
                    record = {
                        "sample_key": sample_key,
                        "variant": sample_key.rsplit("_", 1)[-1],
                        "count_bucket": cycle.count_bucket(len(expected)),
                        "target_index": target_index,
                        "target_xy": [float(value) for value in expected_pixels[target_index]],
                        "category": category,
                        "nearest_top1024_distance_px": float(nearest_top_distance),
                        "nearest_top1024_rank": int(nearest_top_index) + 1,
                        "nearest_all_peak_distance_px": float(nearest_all_distance),
                        "nearest_all_peak_rank": int(nearest_all_index) + 1,
                        "nearest_all_peak_xy": [float(value) for value in nearest_peak],
                        "regional_maxima_count": int(peaks["regional_maxima_count"]),
                        "gt_cell_xy": [cell_x, cell_y],
                        "gt_cell_logit": float(logits[image_index, cell_y, cell_x]),
                        "gt_cell_is_regional_maximum": gt_cell_regional,
                        "adjacent_gt_cell": bool(
                            (gt_cell_distances[target_index] <= 1.0).any()
                        ),
                        "nearest_other_gt_distance_px": float(
                            gt_distances[target_index].min()
                        ),
                        "local_competitor_xy": [competitor_x, competitor_y],
                        "local_competitor_logit": float(
                            logits[image_index, competitor_y, competitor_x]
                        ),
                        "local_competitor_is_gt_cell": competitor_positive,
                    }
                    failures.append(record)
                    current_failures.append(record)

                if current_failures:
                    overlay_name = f"{sample_key}.png"
                    _draw_overlay(
                        batch["pixel_values"][image_index],
                        expected,
                        assignments,
                        current_failures,
                        OUTPUT / "overlays" / overlay_name,
                    )
                    image_rows.append(
                        {
                            "sample_key": sample_key,
                            "overlay": overlay_name,
                            "failures": current_failures,
                        }
                    )

    category_counts = Counter(item["category"] for item in failures)
    variant_counts = Counter(item["variant"] for item in failures)
    bucket_counts = Counter(item["count_bucket"] for item in failures)
    flag_counts = {
        "gt_cell_not_regional_maximum": sum(
            not item["gt_cell_is_regional_maximum"] for item in failures
        ),
        "adjacent_gt_cell": sum(item["adjacent_gt_cell"] for item in failures),
        "local_competitor_is_gt_cell": sum(
            item["local_competitor_is_gt_cell"] for item in failures
        ),
    }
    checks = {
        "top1024_matched_replays_p4e5": matched_total == EXPECTED_MATCHED,
        "missing_atom_count_58": len(failures) == EXPECTED_MISSING,
        "failure_image_count_43": len(image_rows) == EXPECTED_FAILURE_IMAGES,
        "all_peak_limit_not_reached": all(
            item["regional_maxima_count"] < ALL_PEAK_LIMIT for item in failures
        ),
        "confirm_v2_not_read": True,
        "external_and_sealed_not_read": True,
    }
    if not all(checks.values()):
        decision = "missing_peak_source_protocol_failed"
    elif category_counts["gt_cell_not_regional_maximum"] >= len(failures) / 2:
        decision = "local_peak_formation_primary"
    elif category_counts["below_top1024"] >= len(failures) / 2:
        decision = "deep_tail_ranking_primary"
    elif category_counts["matching_conflict"] >= len(failures) / 2:
        decision = "matching_conflict_primary"
    else:
        decision = "mixed_missing_peak_sources"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    _write_index(image_rows)
    audit = {
        "schema_version": "g3-p4e6-missing-peak-sources-audit-0.1",
        "decision": decision,
        "checks": checks,
        "summary": {
            "matched_atoms": matched_total,
            "missing_atoms": len(failures),
            "failure_images": len(image_rows),
            "category_counts": dict(sorted(category_counts.items())),
            "variant_counts": dict(sorted(variant_counts.items())),
            "count_bucket_counts": dict(sorted(bucket_counts.items())),
            "flag_counts": flag_counts,
            "nearest_all_peak_distance_px": {
                "mean": float(np.mean([item["nearest_all_peak_distance_px"] for item in failures])),
                "p50": float(np.percentile([item["nearest_all_peak_distance_px"] for item in failures], 50)),
                "p90": float(np.percentile([item["nearest_all_peak_distance_px"] for item in failures], 90)),
            },
        },
        "protocol": {
            "top_k": TOP_K,
            "all_peak_limit": ALL_PEAK_LIMIT,
            "match_radius_pixels": MATCH_RADIUS,
            "matching": "maximum-cardinality-then-minimum-distance-Hungarian",
            "pixel_cluster_radius": None,
            "training_performed": False,
        },
        "artifacts": {
            "failures": str((OUTPUT / "failures.jsonl").resolve()),
            "overlays": str((OUTPUT / "overlays").resolve()),
            "index": str((OUTPUT / "index.html").resolve()),
        },
        "manifest": manifest_audit,
        "lineage": {
            "checkpoint": str(P4E3_CHECKPOINT.resolve()),
            "checkpoint_sha256": P4E3_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "p4e5_audit_sha256": P4E5_AUDIT_SHA256,
        },
        "data_access": {
            "dev_png_read": True,
            "pilot_train_read": False,
            "confirm_v2_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
