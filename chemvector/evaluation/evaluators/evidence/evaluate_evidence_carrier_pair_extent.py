#!/usr/bin/env python3
"""Evaluate a frozen pair scorer followed by the frozen P0 extent model.

Each validation raster is decoded into LSD anchors and associated with ordinary
targets exactly once.  Pair matching acts only on those positive ordinary
anchors.  Selected pairs use their target-free overlap centerline; unmatched
anchors remain unchanged.  No threshold fitting or validation-time model
selection is available in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalOrdinaryPairWebDataset,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
    conditional_pair_sample_identity,
)
from chemvector.evidence_carrier_extent import (
    EvidenceCarrierExtentPredictor,
    decode_evidence_carrier_extent,
)
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentTargetConfig,
    reject_internal_test_sample,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_scorer import EvidenceCarrierPairScorer
from chemvector.train_evidence_carrier_extent import (
    DEFAULT_TOLERANCES_Q,
    autocast_context,
    evaluate_sample_geometry,
    expand_shards,
    file_sha256,
    load_checkpoint,
    shard_manifest,
    summarize_geometry_records,
    write_json,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    load_pair_scorer_checkpoint,
)


SCHEMA_VERSION = "evidence-carrier-pair-p0-validation-0.1"
P0_REFERENCE_Q8_ORDINARY_RECALL = 0.983907
MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS = 0.2
FORMAL_VALIDATION_SAMPLE_COUNT = 500
DEFAULT_VALIDATION_SHARDS = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-*.tar",
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0910-pair-p0-validation500/evaluation.json"
)


def _numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    return result


def pack_selected_pair_carriers(
    target: Mapping[str, Any],
    pair_evaluation: Mapping[str, Any],
) -> dict[str, torch.Tensor | list[list[int]]]:
    """Replace selected pairs with centerlines and retain every singleton."""

    anchors = _numpy(target["anchor_segments_pixels"], dtype=np.float64)
    target_ids = _numpy(target["anchor_target_line_ids"], dtype=np.int64)
    pair_indices = _numpy(target["pair_indices"], dtype=np.int64)
    centerlines = _numpy(
        target["pair_overlap_centerline_pixels"], dtype=np.float64
    )
    if anchors.size == 0:
        anchors = np.empty((0, 2, 2), dtype=np.float64)
    if pair_indices.size == 0:
        pair_indices = np.empty((0, 2), dtype=np.int64)
        centerlines = np.empty((0, 2, 2), dtype=np.float64)
    if anchors.ndim != 3 or anchors.shape[1:] != (2, 2):
        raise ValueError("Pair/P0 anchor segments must have shape [anchors,2,2]")
    if target_ids.shape != (len(anchors),):
        raise ValueError("Pair/P0 anchor target IDs differ from anchor geometry")
    if pair_indices.ndim != 2 or pair_indices.shape[1:] != (2,):
        raise ValueError("Pair/P0 candidate indices must have shape [pairs,2]")
    if centerlines.shape != (len(pair_indices), 2, 2):
        raise ValueError("Pair/P0 centerlines differ from pair candidates")
    if not np.isfinite(anchors).all() or not np.isfinite(centerlines).all():
        raise ValueError("Pair/P0 carrier geometry contains non-finite coordinates")

    selected_pairs = [
        tuple(map(int, value))
        for value in pair_evaluation.get("selected_anchor_pairs", ())
    ]
    selected_candidate_indices = [
        int(value)
        for value in pair_evaluation.get("selected_candidate_indices", ())
    ]
    if len(selected_pairs) != len(selected_candidate_indices):
        raise ValueError("Selected pair keys and candidate indices differ")
    selected_by_first: dict[int, tuple[tuple[int, int], int]] = {}
    used: set[int] = set()
    for pair, candidate_index in zip(
        selected_pairs, selected_candidate_indices, strict=True
    ):
        if (
            len(pair) != 2
            or min(pair) < 0
            or max(pair) >= len(anchors)
            or pair[0] == pair[1]
            or candidate_index < 0
            or candidate_index >= len(pair_indices)
        ):
            raise ValueError("Selected pair references invalid active geometry")
        canonical = tuple(sorted(pair))
        if tuple(map(int, pair_indices[candidate_index])) != canonical:
            raise ValueError("Selected pair differs from its candidate centerline")
        if used.intersection(canonical):
            raise ValueError("Selected pair matching is not anchor-disjoint")
        used.update(canonical)
        selected_by_first[canonical[0]] = (canonical, candidate_index)

    packed_segments = []
    packed_target_ids = []
    packed_geometry_mask = []
    source_anchor_groups = []
    for anchor_index in range(len(anchors)):
        selected = selected_by_first.get(anchor_index)
        if selected is not None:
            pair, candidate_index = selected
            first_target = int(target_ids[pair[0]])
            second_target = int(target_ids[pair[1]])
            same_target = first_target == second_target
            packed_segments.append(centerlines[candidate_index])
            packed_target_ids.append(first_target if same_target else -1)
            packed_geometry_mask.append(same_target)
            source_anchor_groups.append(list(pair))
        elif anchor_index not in used:
            packed_segments.append(anchors[anchor_index])
            packed_target_ids.append(int(target_ids[anchor_index]))
            packed_geometry_mask.append(True)
            source_anchor_groups.append([anchor_index])

    segment_array = (
        np.asarray(packed_segments, dtype=np.float32)
        if packed_segments
        else np.empty((0, 2, 2), dtype=np.float32)
    )
    return {
        "carrier_segments_pixels": torch.tensor(segment_array),
        "target_line_id": torch.tensor(packed_target_ids, dtype=torch.long),
        "geometry_mask": torch.tensor(packed_geometry_mask, dtype=torch.bool),
        "source_anchor_groups": source_anchor_groups,
    }


def _native_image_tensor(
    image: Image.Image,
    *,
    expected_image_size: int | None,
) -> torch.Tensor:
    if expected_image_size is not None and image.size != (
        expected_image_size,
        expected_image_size,
    ):
        raise ValueError(
            "Pair/P0 validation requires the native full raster without resize: "
            f"got {image.size}, expected {(expected_image_size, expected_image_size)}"
        )
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def _normalize_segments(
    segments_pixels: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    if segments_pixels.ndim != 3 or segments_pixels.shape[1:] != (2, 2):
        raise ValueError("P0 input carriers must have shape [carriers,2,2]")
    if width < 2 or height < 2:
        raise ValueError("P0 validation canvas must be at least 2x2")
    scale = segments_pixels.new_tensor((float(width - 1), float(height - 1)))
    normalized = segments_pixels / scale
    if len(normalized) and (
        not bool(torch.isfinite(normalized).all())
        or bool(((normalized < -1e-5) | (normalized > 1.0 + 1e-5)).any())
    ):
        raise ValueError("P0 packed carrier lies outside the source canvas")
    return normalized.clamp(0.0, 1.0)


@torch.inference_mode()
def _decode_baseline_and_packed(
    model: EvidenceCarrierExtentPredictor,
    image_tensor: torch.Tensor,
    baseline_segments_pixels: torch.Tensor,
    packed_segments_pixels: torch.Tensor,
    *,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use one frozen P0 forward for baseline and pair-packed carriers."""

    baseline = _normalize_segments(
        baseline_segments_pixels.float(), width=width, height=height
    )
    packed = _normalize_segments(
        packed_segments_pixels.float(), width=width, height=height
    )
    counts = (len(baseline), len(packed))
    capacity = max(max(counts), 1)
    carriers = torch.zeros((2, capacity, 2, 2), dtype=torch.float32)
    mask = torch.zeros((2, capacity), dtype=torch.bool)
    carriers[0, : counts[0]] = baseline
    carriers[1, : counts[1]] = packed
    mask[0, : counts[0]] = True
    mask[1, : counts[1]] = True
    pixels = torch.stack((image_tensor, image_tensor)).to(device)
    carriers = carriers.to(device)
    mask = mask.to(device)
    model.eval()
    with autocast_context(device):
        output = model(
            pixels,
            carriers,
            mask,
            coordinate_mode="normalized",
        )
    decoded = decode_evidence_carrier_extent(output, keep_threshold=0.0)
    refined = decoded["refined_segments_pixels"].detach().float().cpu()
    return refined[0, : counts[0]], refined[1, : counts[1]]


@torch.inference_mode()
def evaluate_pair_extent_sample(
    pair_model: EvidenceCarrierPairScorer,
    p0_model: EvidenceCarrierExtentPredictor,
    sample: Mapping[str, Any],
    *,
    pair_threshold: float,
    pairing_config: Any,
    target_config: EvidenceCarrierExtentTargetConfig,
    device: torch.device,
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
    expected_image_size: int | None = 768,
) -> dict[str, Any]:
    """Evaluate one sample after exactly one LSD/association target build."""

    reject_internal_test_sample(dict(sample))
    identity = conditional_pair_sample_identity(
        dict(sample), expected_split="validation"
    )
    image = sample.get("image")
    primitives = sample.get("primitives")
    layers = sample.get("layers")
    if not isinstance(image, Image.Image) or not isinstance(primitives, dict) or not isinstance(
        layers, dict
    ):
        raise ValueError("Pair/P0 validation sample lacks image or sidecars")
    width = int(layers.get("width", 0))
    height = int(layers.get("height", 0))
    if image.size != (width, height):
        raise ValueError("Pair/P0 raster and layer canvas differ")

    target = build_conditional_ordinary_pair_target(
        image,
        primitives,
        layers,
        target_config=target_config,
        pairing_config=pairing_config,
    )
    pair_features = target["pair_features"].unsqueeze(0).to(device)
    pair_mask = target["pair_mask"].unsqueeze(0).to(device)
    pair_model.eval()
    pair_scores = pair_model(pair_features, pair_mask)["pair_gains"][0]
    pair_scores = pair_scores[: len(target["pair_indices"])].detach().float().cpu()
    pair_evaluation = evaluate_evidence_carrier_pair_sample(
        target,
        pair_scores,
        threshold=pair_threshold,
        verify_permutation=True,
    )
    packed = pack_selected_pair_carriers(target, pair_evaluation)
    image_tensor = _native_image_tensor(
        image, expected_image_size=expected_image_size
    )
    baseline_predicted, packed_predicted = _decode_baseline_and_packed(
        p0_model,
        image_tensor,
        target["anchor_segments_pixels"],
        packed["carrier_segments_pixels"],
        width=width,
        height=height,
        device=device,
    )
    baseline_geometry = evaluate_sample_geometry(
        baseline_predicted,
        target["anchor_target_line_ids"],
        target["anchor_mask"],
        primitives,
        layers,
        target_config=target_config,
        tolerances_q=tolerances_q,
    )
    packed_geometry = evaluate_sample_geometry(
        packed_predicted,
        packed["target_line_id"],
        packed["geometry_mask"],
        primitives,
        layers,
        target_config=target_config,
        tolerances_q=tolerances_q,
    )
    return {
        "sample_key": identity.sample_key,
        "identity": identity.as_dict(),
        "raw_positive_anchor_count": len(target["anchor_segments_pixels"]),
        "packed_carrier_count": len(packed["carrier_segments_pixels"]),
        "source_anchor_groups": packed["source_anchor_groups"],
        "pair_matching": pair_evaluation,
        "baseline_geometry": baseline_geometry,
        "packed_geometry": packed_geometry,
        "lsd_association_passes": 1,
    }


def _geometry_aliases(summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in DEFAULT_TOLERANCES_Q:
        name = f"q{int(tolerance)}"
        values = summary[name]
        result[f"ordinary_{name}_member_recall"] = values["ordinary_members"][
            "recall"
        ]
        result[f"single_{name}_group_coverage"] = values["groups"]["single"][
            "coverage"
        ]
        result[f"parallel_2_{name}_group_coverage"] = values["groups"]["p2"][
            "coverage"
        ]
        result[f"parallel_3_{name}_group_coverage"] = values["groups"]["p3"][
            "coverage"
        ]
        result[f"long_{name}_group_coverage"] = values["long_groups"][
            "coverage"
        ]
        result[f"structural_{name}_sample_exact_rate"] = values[
            "structural_samples"
        ]["exact_rate"]
    return result


def summarize_pair_extent_validation(
    sample_records: Sequence[Mapping[str, Any]],
    *,
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
) -> dict[str, Any]:
    records = list(sample_records)
    if not records:
        raise ValueError("Pair/P0 validation summary requires at least one sample")
    if tolerances_q != DEFAULT_TOLERANCES_Q:
        raise ValueError("Pair/P0 formal summary requires fixed q8/q16/q32 tolerances")
    pair_metrics = aggregate_evidence_carrier_pair_evaluations(
        [record["pair_matching"] for record in records]
    )
    baseline_geometry = summarize_geometry_records(
        [record["baseline_geometry"] for record in records], tolerances_q
    )
    packed_geometry = summarize_geometry_records(
        [record["packed_geometry"] for record in records], tolerances_q
    )
    baseline_aliases = _geometry_aliases(baseline_geometry)
    packed_aliases = _geometry_aliases(packed_geometry)
    packed_q8 = float(packed_aliases["ordinary_q8_member_recall"])
    delta_pp = 100.0 * (packed_q8 - P0_REFERENCE_Q8_ORDINARY_RECALL)
    regression_pp = max(0.0, -delta_pp)
    q8_gate = regression_pp <= MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS + 1e-12
    return {
        "schema_version": SCHEMA_VERSION,
        "samples": len(records),
        "pair_matching": pair_metrics,
        "baseline_geometry": baseline_geometry,
        "packed_geometry": packed_geometry,
        "baseline_metrics": baseline_aliases,
        "packed_metrics": packed_aliases,
        "q8_reference_comparison": {
            "reference_recall": P0_REFERENCE_Q8_ORDINARY_RECALL,
            "reference_percent": 100.0 * P0_REFERENCE_Q8_ORDINARY_RECALL,
            "packed_recall": packed_q8,
            "packed_percent": 100.0 * packed_q8,
            "delta_percentage_points": delta_pp,
            "regression_percentage_points": regression_pp,
            "maximum_allowed_regression_percentage_points": (
                MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS
            ),
        },
        "gate": {
            "status": "passed" if q8_gate else "failed",
            "checks": {
                "q8_recall_regression_le_0_2pp": q8_gate,
            },
            "pair_topology_metrics_are_reported_not_recalibrated": True,
        },
        "protocol_checks": {
            "one_lsd_association_pass_per_sample": all(
                int(record.get("lsd_association_passes", 0)) == 1
                for record in records
            ),
            "pair_threshold_recalibrated_on_validation": False,
            "production_integration": False,
            "uses_internal_test": False,
        },
    }


@torch.inference_mode()
def evaluate_pair_p0_validation(
    pair_checkpoint_path: str | Path,
    p0_checkpoint_path: str | Path,
    validation_patterns: Sequence[str],
    *,
    device: torch.device,
    expected_samples: int = FORMAL_VALIDATION_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Run the frozen validation protocol without exposing calibration controls."""

    if expected_samples != FORMAL_VALIDATION_SAMPLE_COUNT:
        raise ValueError("Formal pair/P0 validation requires exactly 500 samples")
    validation_shards = expand_shards(validation_patterns)
    pair_model, pair_checkpoint = load_pair_scorer_checkpoint(
        pair_checkpoint_path, device=device
    )
    p0_model, target_config, p0_checkpoint = load_checkpoint(
        p0_checkpoint_path, device=device
    )
    if asdict(target_config) != asdict(EvidenceCarrierExtentTargetConfig()):
        raise ValueError("Pair/P0 validation requires the frozen default target protocol")
    pairing_config = conditional_pair_pairing_config(
        PAIRING_PROFILE_TRAINING_WIDE
    )
    if pair_checkpoint.get("pairing_config") != asdict(pairing_config):
        raise ValueError("Pair scorer checkpoint uses a different candidate profile")
    threshold = float(pair_checkpoint["gain_threshold"])
    for model in (pair_model, p0_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    dataset = ConditionalOrdinaryPairWebDataset(
        validation_shards,
        split="validation",
        render_variant=None,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        validate_hashes=True,
        limit=None,
    )
    records = [
        evaluate_pair_extent_sample(
            pair_model,
            p0_model,
            sample,
            pair_threshold=threshold,
            pairing_config=pairing_config,
            target_config=target_config,
            device=device,
            expected_image_size=768,
        )
        for sample in dataset
    ]
    if len(records) != expected_samples:
        raise RuntimeError(
            f"Formal pair/P0 validation read {len(records)} samples, expected "
            f"{expected_samples}"
        )
    summary = summarize_pair_extent_validation(records)
    return {
        **summary,
        "pair_checkpoint": {
            "path": str(Path(pair_checkpoint_path).expanduser().resolve()),
            "sha256": file_sha256(pair_checkpoint_path),
            "gain_threshold": threshold,
            "threshold_source": "frozen_train_cal_checkpoint",
        },
        "p0_checkpoint": {
            "path": str(Path(p0_checkpoint_path).expanduser().resolve()),
            "sha256": file_sha256(p0_checkpoint_path),
            "model_type": p0_checkpoint["model_type"],
            "global_step": p0_checkpoint["global_step"],
        },
        "validation_shards": shard_manifest(validation_shards),
        "sample_records": records,
        "evaluation_protocol": {
            "candidate_profile": PAIRING_PROFILE_TRAINING_WIDE,
            "pairing_config": asdict(pairing_config),
            "pair_threshold_recalibrated": False,
            "pair_threshold_override_available": False,
            "lsd_and_association_passes_per_sample": 1,
            "pair_scope": "positive ordinary anchors only",
            "selected_pair_geometry": "target-free overlap centerline",
            "unmatched_anchor_geometry": "original LSD anchor",
            "cross_target_geometry_credit": "none",
            "p0_keep_threshold": 0.0,
            "p0_weights_frozen": True,
            "production_integration": False,
            "uses_internal_test": False,
        },
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--p0-checkpoint", required=True)
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=list(DEFAULT_VALIDATION_SHARDS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_pair_p0_validation(
        args.pair_checkpoint,
        args.p0_checkpoint,
        args.validation_shards,
        device=_device(args.device),
    )
    write_json(args.output, report)
    print(
        {
            "output": str(Path(args.output).resolve()),
            "samples": report["samples"],
            "q8_recall": report["packed_metrics"]["ordinary_q8_member_recall"],
            "q8_regression_pp": report["q8_reference_comparison"][
                "regression_percentage_points"
            ],
            "gate": report["gate"]["status"],
        }
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FORMAL_VALIDATION_SAMPLE_COUNT",
    "MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS",
    "P0_REFERENCE_Q8_ORDINARY_RECALL",
    "evaluate_pair_extent_sample",
    "evaluate_pair_p0_validation",
    "pack_selected_pair_carriers",
    "summarize_pair_extent_validation",
]
