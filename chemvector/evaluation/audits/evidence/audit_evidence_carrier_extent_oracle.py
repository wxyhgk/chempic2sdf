#!/usr/bin/env python3
"""Audit the geometry ceiling of bounded evidence-carrier extent targets.

The audit uses GT only after target-independent LSD carriers have been
generated.  It reconstructs each positive carrier from the continuous target
normal offset and left/right extent fractions while keeping the carrier
direction fixed.  The result is a representation/association ceiling, not a
learned-model metric.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2

from chemvector.audit_evidence_carrier_proposals import (
    COORDINATE_MAX,
    extract_targets,
    file_sha256,
)
from chemvector.evidence_carrier_extent_data import (
    IN_SCOPE_GROUP_KINDS,
    EvidenceCarrierExtentTargetConfig,
    EvidenceCarrierExtentWebDataset,
    build_evidence_carrier_extent_target,
)


SCHEMA_VERSION = "evidence-carrier-extent-oracle-0.1"
DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v099-evidence-carrier-extent-oracle-validation100/"
    "oracle_metrics.json"
)


def clip_infinite_line_to_image(
    center: np.ndarray,
    axis: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a line in pixel-center coordinates to ``[0,W-1] x [0,H-1]``."""

    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if center.shape != (2,) or axis.shape != (2,):
        raise ValueError("Line center and axis must contain x/y")
    if not np.isfinite(center).all() or not np.isfinite(axis).all():
        raise ValueError("Line center and axis must be finite")
    if width < 2 or height < 2:
        raise ValueError("Image dimensions must be at least 2")
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("Line axis must be nonzero")
    unit = axis / length
    lower = -math.inf
    upper = math.inf
    for dimension, maximum in ((0, float(width - 1)), (1, float(height - 1))):
        component = float(unit[dimension])
        coordinate = float(center[dimension])
        if abs(component) <= 1e-12:
            if not 0.0 <= coordinate <= maximum:
                return None
            continue
        values = sorted(((-coordinate) / component, (maximum - coordinate) / component))
        lower = max(lower, values[0])
        upper = min(upper, values[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return None
    return center + unit * lower, center + unit * upper


def reconstruct_continuous_target_segments(
    target: dict[str, torch.Tensor],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Reconstruct every padded carrier from continuous supervision fields."""

    required = (
        "carrier_segments",
        "carrier_mask",
        "geometry_loss_mask",
        "normal_offset_pixels",
        "extent_left_fraction",
        "extent_right_fraction",
    )
    missing = [name for name in required if name not in target]
    if missing:
        raise ValueError(f"Extent oracle target fields are missing: {missing}")
    carriers = target["carrier_segments"].detach().cpu().numpy().astype(np.float64)
    if carriers.ndim != 3 or carriers.shape[1:] != (2, 2):
        raise ValueError("carrier_segments must have shape [carriers,2,2]")
    scale = np.asarray((float(width - 1), float(height - 1)))
    carriers = carriers * scale
    valid = (
        target["carrier_mask"].bool() & target["geometry_loss_mask"].bool()
    ).detach().cpu().numpy()
    output = np.zeros_like(carriers)
    for carrier_index in np.flatnonzero(valid):
        carrier = carriers[carrier_index]
        delta = carrier[1] - carrier[0]
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise ValueError("Positive extent carrier is degenerate")
        axis = delta / length
        normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        center = carrier.mean(axis=0) + normal * float(
            target["normal_offset_pixels"][carrier_index]
        )
        clipped = clip_infinite_line_to_image(
            center,
            axis,
            width=width,
            height=height,
        )
        if clipped is None:
            raise ValueError("Positive extent carrier does not intersect its image")
        start, end = clipped
        interval = end - start
        left = float(target["extent_left_fraction"][carrier_index])
        right = float(target["extent_right_fraction"][carrier_index])
        if not 0.0 <= left < right <= 1.0:
            raise ValueError("Continuous extent fractions must satisfy 0<=left<right<=1")
        output[carrier_index] = np.stack(
            (start + interval * left, start + interval * right)
        )
    return output


def unordered_endpoint_error_q(
    predicted_pixels: np.ndarray,
    expected_q: tuple[tuple[float, float], tuple[float, float]],
    *,
    width: int,
    height: int,
) -> float:
    """Return maximum unordered endpoint error in the q2047 protocol."""

    predicted = np.asarray(predicted_pixels, dtype=np.float64)
    if predicted.shape != (2, 2):
        raise ValueError("Predicted line must have two x/y endpoints")
    expected = np.asarray(expected_q, dtype=np.float64)
    if expected.shape != (2, 2):
        raise ValueError("Expected q2047 line must have two endpoints")
    scale = np.asarray(
        (float(width - 1) / COORDINATE_MAX, float(height - 1) / COORDINATE_MAX)
    )
    expected_pixels = expected * scale
    direct = np.linalg.norm(predicted - expected_pixels, axis=-1).max()
    reverse = np.linalg.norm(predicted - expected_pixels[::-1], axis=-1).max()
    pixel_error = float(min(direct, reverse))
    # The current dataset is square.  Use the strictest axis conversion if a
    # future audit supplies a rectangular image.
    q_per_pixel = max(
        COORDINATE_MAX / float(width - 1),
        COORDINATE_MAX / float(height - 1),
    )
    return pixel_error * q_per_pixel


def audit_sample(
    sample: dict[str, Any],
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    tolerances_q: tuple[float, ...],
) -> dict[str, Any]:
    image = sample["image"]
    width, height = image.size
    target = build_evidence_carrier_extent_target(
        image,
        sample["primitives"],
        sample["layers"],
        target_config=target_config,
    )
    reconstructed = reconstruct_continuous_target_segments(
        target,
        width=width,
        height=height,
    )
    target_lines, target_groups, _ = extract_targets(
        sample["primitives"],
        sample["layers"],
        long_bond_pixels=target_config.long_bond_pixels,
    )
    ordinary = {
        line_id: line
        for line_id, line in enumerate(target_lines)
        if line.group_kind in IN_SCOPE_GROUP_KINDS
    }
    best_error = {line_id: math.inf for line_id in ordinary}
    positive_indices = torch.where(target["geometry_loss_mask"])[0].tolist()
    for carrier_index in positive_indices:
        line_id = int(target["target_line_id"][carrier_index])
        if line_id not in ordinary:
            raise ValueError("Positive carrier references an out-of-scope target")
        error = unordered_endpoint_error_q(
            reconstructed[carrier_index],
            ordinary[line_id].points_q,
            width=width,
            height=height,
        )
        best_error[line_id] = min(best_error[line_id], error)

    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for line_id, line in ordinary.items():
        groups[(line.group_index, line.group_kind)].append(line_id)
    by_tolerance = {}
    for tolerance in tolerances_q:
        line_matches = sum(error <= tolerance for error in best_error.values())
        group_matches = Counter()
        group_expected = Counter()
        long_expected = 0
        long_matched = 0
        for (_, kind), line_ids in groups.items():
            matched = all(best_error[line_id] <= tolerance for line_id in line_ids)
            group_expected[kind] += 1
            group_matches[kind] += int(matched)
            is_long = any(ordinary[line_id].long_group for line_id in line_ids)
            if is_long:
                long_expected += 1
                long_matched += int(matched)
        by_tolerance[tolerance_key(tolerance)] = {
            "line_expected": len(best_error),
            "line_matched": line_matches,
            "sample_exact": bool(best_error) and line_matches == len(best_error),
            "group_expected": dict(group_expected),
            "group_matched": dict(group_matches),
            "long_group_expected": long_expected,
            "long_group_matched": long_matched,
        }
    return {
        "sample_key": str(sample["key"]),
        "carrier_count": int(target["carrier_mask"].sum()),
        "positive_carrier_count": int(target["geometry_loss_mask"].sum()),
        "ordinary_target_line_count": len(ordinary),
        "tolerances": by_tolerance,
    }


def summarize_records(
    records: list[dict[str, Any]], tolerances_q: tuple[float, ...]
) -> dict[str, Any]:
    summary = {}
    for tolerance in tolerances_q:
        name = tolerance_key(tolerance)
        line_expected = sum(record["tolerances"][name]["line_expected"] for record in records)
        line_matched = sum(record["tolerances"][name]["line_matched"] for record in records)
        scored_samples = sum(
            record["tolerances"][name]["line_expected"] > 0 for record in records
        )
        sample_exact = sum(record["tolerances"][name]["sample_exact"] for record in records)
        group_expected = Counter()
        group_matched = Counter()
        long_expected = 0
        long_matched = 0
        for record in records:
            values = record["tolerances"][name]
            group_expected.update(values["group_expected"])
            group_matched.update(values["group_matched"])
            long_expected += values["long_group_expected"]
            long_matched += values["long_group_matched"]
        summary[name] = {
            "line_expected": line_expected,
            "line_matched": line_matched,
            "line_recall": line_matched / max(line_expected, 1),
            "sample_exact": sample_exact,
            "sample_scored": scored_samples,
            "sample_exact_rate": sample_exact / max(scored_samples, 1),
            "groups": {
                kind: {
                    "expected": group_expected[kind],
                    "matched": group_matched[kind],
                    "recall": group_matched[kind] / max(group_expected[kind], 1),
                }
                for kind in sorted(group_expected)
            },
            "long_groups": {
                "expected": long_expected,
                "matched": long_matched,
                "recall": long_matched / max(long_expected, 1),
            },
        }
    return summary


def tolerance_key(value: float) -> str:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("Tolerance must be positive and finite")
    return f"{float(value):g}q"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--tolerances-q", nargs="+", type=float, default=(8.0, 16.0, 32.0))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    tolerances = tuple(float(value) for value in args.tolerances_q)
    if not tolerances or any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("tolerances-q must contain positive finite values")
    tolerance_names = [tolerance_key(value) for value in tolerances]
    if len(set(tolerance_names)) != len(tolerance_names):
        raise ValueError("tolerances-q contains duplicate serialized keys")
    target_config = EvidenceCarrierExtentTargetConfig()
    records = [
        audit_sample(sample, target_config=target_config, tolerances_q=tolerances)
        for sample in EvidenceCarrierExtentWebDataset(args.shard, limit=args.limit)
    ]
    if not records:
        raise RuntimeError("Extent oracle dataset produced no samples")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard": str(Path(args.shard).resolve()),
        "shard_sha256": file_sha256(args.shard),
        "samples": len(records),
        "scope": "ordinary single_line/parallel_2/parallel_3 only",
        "target_geometry_used_for_proposal_generation": False,
        "gt_used_after_proposal_generation_for_association": True,
        "proposal_generator": {
            "name": "OpenCV LineSegmentDetector",
            "opencv_version": cv2.__version__,
            "refine": "LSD_REFINE_STD",
            "minimum_length_pixels": target_config.minimum_length_pixels,
        },
        "carrier_direction_is_fixed": True,
        "learned_model_used": False,
        "uses_internal_test": False,
        "target_config": asdict(target_config),
        "summary": summarize_records(records, tolerances),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
