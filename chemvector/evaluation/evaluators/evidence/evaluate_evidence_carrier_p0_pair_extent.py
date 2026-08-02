#!/usr/bin/env python3
"""Evaluate B pair matching followed by the same frozen P0 extent model.

The packed geometry is compared with an on-site baseline produced by the same
P0 checkpoint, on the same image and in the same batched forward call.  A run
passes the extent regression check only when packed q8 ordinary-member recall
falls by no more than 0.2 percentage points relative to that live baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
    conditional_pair_pairing_config,
    validate_molecule_disjoint_splits,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
)
from chemvector.evaluate_evidence_carrier_p0_pair_scorer import (
    EXPECTED_RENDER_VARIANTS,
    FORMAL_IMAGE_SIZE,
    FORMAL_VALIDATION_SAMPLE_COUNT,
    VALIDATION_INTERPRETATION,
    _contains_internal_test,
    _device,
    _expand_validation_shards,
    _plain_json,
    _shard_descriptors,
    _train_cache_identities,
    evaluate_frozen_p0_pair_sample,
    load_frozen_p0_pair_pipeline,
)
from chemvector.evaluate_evidence_carrier_pair_extent import (
    _decode_baseline_and_packed,
    _geometry_aliases,
    _native_image_tensor,
    pack_selected_pair_carriers,
)
from chemvector.evaluate_evidence_carrier_pair_scorer import (
    pair_scorer_validation_gate,
)
from chemvector.train_evidence_carrier_extent import (
    DEFAULT_TOLERANCES_Q,
    evaluate_sample_geometry,
    summarize_geometry_records,
)


SCHEMA_VERSION = "evidence-carrier-p0-pair-p0-validation-0.1"
MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS = 0.2


@torch.inference_mode()
def evaluate_p0_pair_extent_sample(
    pair_model: torch.nn.Module,
    p0_model: torch.nn.Module,
    sample: Mapping[str, Any],
    *,
    pair_threshold: float,
    pairing_config: Any,
    target_config: Any,
    device: torch.device,
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
    expected_image_size: int = FORMAL_IMAGE_SIZE,
) -> dict[str, Any]:
    """Build the target once, score B once, then compare baseline and packed P0."""

    result = evaluate_frozen_p0_pair_sample(
        pair_model,
        p0_model,
        sample,
        threshold=pair_threshold,
        pairing_config=pairing_config,
        target_config=target_config,
        device=device,
        expected_image_size=expected_image_size,
    )
    image = sample.get("image")
    primitives = sample.get("primitives")
    layers = sample.get("layers")
    if not isinstance(image, Image.Image) or not isinstance(primitives, dict) or not isinstance(
        layers, dict
    ):
        raise ValueError("Frozen-P0 pair/P0 sample lacks image or sidecars")
    width = int(layers.get("width", 0))
    height = int(layers.get("height", 0))
    if image.size != (width, height):
        raise ValueError("Frozen-P0 pair/P0 raster and layer canvas differ")

    target = result["target"]
    pair_evaluation = result["pair_evaluation"]
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
    identity = result["identity"]
    return {
        "sample_key": identity.sample_key,
        "identity": identity.as_dict(),
        "raw_positive_anchor_count": result["anchor_count"],
        "pair_candidate_count": result["pair_count"],
        "packed_carrier_count": len(packed["carrier_segments_pixels"]),
        "source_anchor_groups": packed["source_anchor_groups"],
        "pair_matching": pair_evaluation,
        "baseline_geometry": baseline_geometry,
        "packed_geometry": packed_geometry,
        "target_builds": result["target_builds"],
        "p0_pair_evidence_forward_passes": result["p0_forward_passes"],
        "p0_baseline_packed_forward_passes": 1,
        "baseline_and_packed_share_forward": True,
    }


def summarize_p0_pair_extent_validation(
    sample_records: Sequence[Mapping[str, Any]],
    *,
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
) -> dict[str, Any]:
    records = list(sample_records)
    if not records:
        raise ValueError("Frozen-P0 pair/P0 summary requires at least one sample")
    if tolerances_q != DEFAULT_TOLERANCES_Q:
        raise ValueError("Formal frozen-P0 pair/P0 summary requires q8/q16/q32")
    pair_metrics = aggregate_evidence_carrier_pair_evaluations(
        [record["pair_matching"] for record in records]
    )
    baseline_geometry = summarize_geometry_records(
        [record["baseline_geometry"] for record in records], tolerances_q
    )
    packed_geometry = summarize_geometry_records(
        [record["packed_geometry"] for record in records], tolerances_q
    )
    baseline_metrics = _geometry_aliases(baseline_geometry)
    packed_metrics = _geometry_aliases(packed_geometry)
    baseline_q8 = float(baseline_metrics["ordinary_q8_member_recall"])
    packed_q8 = float(packed_metrics["ordinary_q8_member_recall"])
    delta_pp = 100.0 * (packed_q8 - baseline_q8)
    regression_pp = max(0.0, -delta_pp)
    q8_gate = regression_pp <= MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS + 1e-12
    protocol_checks = {
        "one_target_build_per_sample": all(
            int(record.get("target_builds", 0)) == 1 for record in records
        ),
        "one_pair_evidence_p0_forward_per_sample": all(
            int(record.get("p0_pair_evidence_forward_passes", 0)) == 1
            for record in records
        ),
        "baseline_and_packed_share_one_p0_forward": all(
            int(record.get("p0_baseline_packed_forward_passes", 0)) == 1
            and record.get("baseline_and_packed_share_forward") is True
            for record in records
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "samples": len(records),
        "pair_matching": pair_metrics,
        "baseline_geometry": baseline_geometry,
        "packed_geometry": packed_geometry,
        "baseline_metrics": baseline_metrics,
        "packed_metrics": packed_metrics,
        "q8_live_baseline_comparison": {
            "baseline_recall": baseline_q8,
            "baseline_percent": 100.0 * baseline_q8,
            "packed_recall": packed_q8,
            "packed_percent": 100.0 * packed_q8,
            "delta_percentage_points": delta_pp,
            "regression_percentage_points": regression_pp,
            "maximum_allowed_regression_percentage_points": (
                MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS
            ),
            "comparison_source": "same checkpoint, same samples, same batched forward",
            "hardcoded_reference_used": False,
        },
        "extent_gate": {
            "status": "GO" if q8_gate and all(protocol_checks.values()) else "NO-GO",
            "checks": {
                "packed_q8_regression_le_0_2pp": q8_gate,
                **protocol_checks,
            },
        },
        "protocol_checks": {
            **protocol_checks,
            "pair_threshold_recalibrated_on_validation": False,
            "p0_weights_frozen": True,
            "uses_internal_test": False,
        },
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False)
    if _contains_internal_test(checkpoint) or _contains_internal_test(output):
        raise ValueError("Frozen-P0 pair/P0 paths cannot reference internal_test")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen-P0 pair checkpoint is missing: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"Frozen-P0 pair/P0 output exists: {output}")
    validation_shards = _expand_validation_shards(args.validation_shards)
    device = _device(args.device)
    pipeline = load_frozen_p0_pair_pipeline(checkpoint, device=device)
    pairing_config = conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
    if pipeline["pair_checkpoint"].get("pairing_config") != asdict(pairing_config):
        raise ValueError("Frozen-P0 pair/P0 candidate profile differs")

    dataset = ConditionalOrdinaryPairWebDataset(
        validation_shards,
        split="validation",
        render_variant=None,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        validate_hashes=True,
        limit=FORMAL_VALIDATION_SAMPLE_COUNT,
    )
    records = []
    identities: list[ConditionalPairSampleIdentity] = []
    seen: set[str] = set()
    for sample in dataset:
        identity = sample.get("identity")
        if not isinstance(identity, ConditionalPairSampleIdentity):
            raise ValueError("Frozen-P0 pair/P0 sample lacks validated identity")
        if identity.sample_key in seen:
            raise ValueError("Frozen-P0 pair/P0 validation contains a duplicate sample")
        seen.add(identity.sample_key)
        identities.append(identity)
        records.append(
            evaluate_p0_pair_extent_sample(
                pipeline["pair_model"],
                pipeline["p0_model"],
                sample,
                pair_threshold=pipeline["pair_threshold"],
                pairing_config=pairing_config,
                target_config=pipeline["target_config"],
                device=device,
            )
        )
    if len(records) != FORMAL_VALIDATION_SAMPLE_COUNT:
        raise ValueError(
            f"Formal frozen-P0 pair/P0 validation requires exactly "
            f"{FORMAL_VALIDATION_SAMPLE_COUNT} samples, got {len(records)}"
        )

    train_identities = _train_cache_identities(pipeline["base_cache"])
    identity_audit = validate_molecule_disjoint_splits(
        train_identities,
        identities,
        left_name="train_base_cache",
        right_name="validation",
    )
    summary = summarize_p0_pair_extent_validation(records)
    render_variants = {identity.render_variant for identity in identities}
    pair_gate = pair_scorer_validation_gate(
        summary["pair_matching"],
        sample_count=len(records),
        expected_sample_count=FORMAL_VALIDATION_SAMPLE_COUNT,
        render_variants=render_variants,
        identity_audit=identity_audit,
        checkpoint=pipeline["pair_checkpoint"],
    )
    overall_status = (
        "GO"
        if pair_gate["status"] == "GO" and summary["extent_gate"]["status"] == "GO"
        else "NO-GO"
    )
    payload = {
        **summary,
        "status": overall_status,
        "interpretation": {
            "classification": VALIDATION_INTERPRETATION,
            "independent_generalization": False,
            "reason": (
                "The fixed-step upstream P0 contains metrics from this same validation "
                "split, although the checkpoint was selected by explicit step."
            ),
            "upstream_p0_checkpoint_selected_by_validation": pipeline[
                "p0_provenance"
            ]["checkpoint_selected_by_validation"],
        },
        "pair_validation_gate": pair_gate,
        "overall_gate": {
            "status": overall_status,
            "checks": {
                "pair_topology_gate_is_GO": pair_gate["status"] == "GO",
                "live_p0_extent_regression_gate_is_GO": (
                    summary["extent_gate"]["status"] == "GO"
                ),
            },
        },
        "checkpoint": {
            "pair": {
                "path": str(pipeline["pair_checkpoint_path"]),
                "sha256": pipeline["pair_checkpoint_sha256"],
                "gain_threshold": pipeline["pair_threshold"],
                "threshold_source": "frozen_train_cal_checkpoint",
            },
            "p0": {
                "path": str(pipeline["p0_checkpoint_path"]),
                "sha256": pipeline["p0_checkpoint_sha256"],
                "model_type": pipeline["p0_checkpoint"].get("model_type"),
                "global_step": pipeline["p0_checkpoint"].get("global_step"),
            },
        },
        "cache_provenance": {
            "base_cache_path": str(pipeline["base_cache_path"]),
            "base_cache_sha256": pipeline["base_cache_sha256"],
            "augmentation_cache_path": str(pipeline["augmentation_cache_path"]),
            "augmentation_cache_sha256": pipeline["augmentation_cache_sha256"],
            "anchor_feature_dtype": pipeline["augmentation_cache"].get(
                "anchor_feature_dtype"
            ),
            "p0_extraction_provenance": pipeline["augmentation_cache"].get(
                "p0_extraction_provenance"
            ),
        },
        "validation_shards": _shard_descriptors(validation_shards),
        "identity": {
            "sample_count": len(records),
            "render_variants": sorted(render_variants),
            "expected_render_variants": sorted(EXPECTED_RENDER_VARIANTS),
            "train_validation_zero_overlap": identity_audit,
        },
        "evaluation_protocol": {
            "formal_sample_count": FORMAL_VALIDATION_SAMPLE_COUNT,
            "pair_threshold_recalibrated": False,
            "pair_threshold_override_available": False,
            "p0_checkpoint_override_available": False,
            "baseline_reference": "live_same_batch",
            "hardcoded_p0_reference": False,
            "internal_test_read": False,
            "production_integration": False,
        },
        "sample_records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "samples": payload["samples"],
                "baseline_q8": payload["q8_live_baseline_comparison"][
                    "baseline_recall"
                ],
                "packed_q8": payload["q8_live_baseline_comparison"]["packed_recall"],
                "status": payload["status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "MAXIMUM_Q8_REGRESSION_PERCENTAGE_POINTS",
    "SCHEMA_VERSION",
    "build_parser",
    "evaluate",
    "evaluate_p0_pair_extent_sample",
    "summarize_p0_pair_extent_validation",
]
