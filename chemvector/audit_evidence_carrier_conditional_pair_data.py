#!/usr/bin/env python3
"""Audit v0.9.10 conditional ordinary pair labels on train/validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_NAMES,
    PAIRING_PROFILES,
    SCHEMA_VERSION,
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
    ConditionalPairSummary,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
    iter_conditional_pair_identities,
    partition_fit_calibration_identities,
    validate_molecule_disjoint_splits,
)
from chemvector.evidence_carrier_extent_data import EvidenceCarrierExtentTargetConfig
from chemvector.evidence_carrier_stroke_pairing import StrokeBoundaryPairingConfig


AUDIT_SCHEMA_VERSION = "evidence-carrier-conditional-pair-data-audit-0.1"
DEFAULT_TRAIN_SHARD = "Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-000000.tar"
DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0910-conditional-ordinary-pair-data-strict/summary.json"
)


def _identity_payload(identities: list[ConditionalPairSampleIdentity]) -> dict[str, Any]:
    molecule_ids = {identity.molecule_id for identity in identities}
    source_ids = {identity.source_structure_id for identity in identities}
    variants = {identity.render_variant for identity in identities}
    return {
        "samples": len(identities),
        "molecules": len(molecule_ids),
        "source_structures": len(source_ids),
        "render_variants": sorted(variants),
        "sample_keys": [identity.sample_key for identity in identities],
    }


def _summarize_keys(
    shards: list[str],
    *,
    split: str,
    render_variant: str | None,
    sample_keys: set[str],
    target_config: EvidenceCarrierExtentTargetConfig,
    pairing_config: StrokeBoundaryPairingConfig,
    progress_name: str,
) -> dict[str, Any]:
    summary = ConditionalPairSummary()
    dataset = ConditionalOrdinaryPairWebDataset(
        shards,
        split=split,
        render_variant=render_variant,
        allowed_sample_keys=sample_keys,
        validate_hashes=True,
    )
    for sample_index, sample in enumerate(dataset, 1):
        target = build_conditional_ordinary_pair_target(
            sample["image"],
            sample["primitives"],
            sample["layers"],
            target_config=target_config,
            pairing_config=pairing_config,
        )
        summary.update(target)
        if sample_index % 25 == 0 or sample_index == len(sample_keys):
            print(
                f"{progress_name}: {sample_index}/{len(sample_keys)}",
                flush=True,
            )
    return summary.result()


def _pairing_config(args: argparse.Namespace) -> StrokeBoundaryPairingConfig:
    replacements = {
        "maximum_angle_degrees": args.maximum_angle_degrees,
        "minimum_normal_separation_pixels": (
            args.minimum_normal_separation_pixels
        ),
        "maximum_normal_separation_pixels": (
            args.maximum_normal_separation_pixels
        ),
        "minimum_overlap_pixels": args.minimum_overlap_pixels,
        "minimum_shorter_interval_overlap": (
            args.minimum_shorter_interval_overlap
        ),
        "minimum_interior_ink_occupancy": (
            args.minimum_interior_ink_occupancy
        ),
    }
    explicit = {name: value for name, value in replacements.items() if value is not None}
    if args.pairing_profile == "custom":
        if not explicit:
            raise ValueError("pairing-profile=custom requires at least one override")
        return replace(conditional_pair_pairing_config("strict"), **explicit)
    if explicit:
        raise ValueError(
            "Named pairing profiles cannot be combined with per-field overrides; "
            "use --pairing-profile custom"
        )
    return conditional_pair_pairing_config(args.pairing_profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=[DEFAULT_TRAIN_SHARD])
    parser.add_argument(
        "--validation-shards", nargs="+", default=[DEFAULT_VALIDATION_SHARD]
    )
    parser.add_argument("--render-variant", default="original")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=125)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--partition-seed", default="chemvector-v0.9.10")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pairing-profile",
        choices=(*PAIRING_PROFILES, "custom"),
        default="strict",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--maximum-angle-degrees", type=float)
    parser.add_argument("--minimum-normal-separation-pixels", type=float)
    parser.add_argument("--maximum-normal-separation-pixels", type=float)
    parser.add_argument("--minimum-overlap-pixels", type=float)
    parser.add_argument("--minimum-shorter-interval-overlap", type=float)
    parser.add_argument("--minimum-interior-ink-occupancy", type=float)
    args = parser.parse_args()
    if args.train_samples < 2:
        parser.error("train-samples must be >=2")
    if not args.skip_validation and args.validation_samples < 1:
        parser.error("validation-samples must be >=1 unless --skip-validation is used")
    return args


def main() -> None:
    args = parse_args()
    pairing_config = _pairing_config(args)
    target_config = EvidenceCarrierExtentTargetConfig()
    render_variant = None if args.render_variant == "all" else args.render_variant

    train_identities = list(
        iter_conditional_pair_identities(
            args.train_shards,
            split="train",
            render_variant=render_variant,
            limit=args.train_samples,
        )
    )
    validation_identities = []
    if not args.skip_validation:
        validation_identities = list(
            iter_conditional_pair_identities(
                args.validation_shards,
                split="validation",
                render_variant=render_variant,
                limit=args.validation_samples,
            )
        )
    if len(train_identities) != args.train_samples:
        raise ValueError(
            f"Requested {args.train_samples} train identities, got {len(train_identities)}"
        )
    if not args.skip_validation and len(validation_identities) != args.validation_samples:
        raise ValueError(
            "Requested "
            f"{args.validation_samples} validation identities, got "
            f"{len(validation_identities)}"
        )
    partition = partition_fit_calibration_identities(
        train_identities,
        calibration_fraction=args.calibration_fraction,
        seed=args.partition_seed,
    )
    train_validation_disjoint = None
    if not args.skip_validation:
        train_validation_disjoint = validate_molecule_disjoint_splits(
            train_identities,
            validation_identities,
        )

    fit_keys = set(partition.fit_sample_keys)
    calibration_keys = set(partition.calibration_sample_keys)
    fit_summary = _summarize_keys(
        args.train_shards,
        split="train",
        render_variant=render_variant,
        sample_keys=fit_keys,
        target_config=target_config,
        pairing_config=pairing_config,
        progress_name="fit",
    )
    calibration_summary = _summarize_keys(
        args.train_shards,
        split="train",
        render_variant=render_variant,
        sample_keys=calibration_keys,
        target_config=target_config,
        pairing_config=pairing_config,
        progress_name="calibration",
    )
    validation_summary = None
    if not args.skip_validation:
        validation_keys = {identity.sample_key for identity in validation_identities}
        # Validation is scored once, after the train-only partition is frozen.
        validation_summary = _summarize_keys(
            args.validation_shards,
            split="validation",
            render_variant=render_variant,
            sample_keys=validation_keys,
            target_config=target_config,
            pairing_config=pairing_config,
            progress_name="validation-final",
        )

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "data_schema_version": SCHEMA_VERSION,
        "status": "completed_train_only" if args.skip_validation else "completed",
        "scope": "GT-routed positive ordinary LSD anchors only",
        "protocol": {
            "pair_candidate_inputs": [
                "raw raster pixels",
                "positive ordinary anchor segments from router oracle",
                "target-free StrokeBoundaryPairingConfig",
            ],
            "target_line_id_visible_to_candidate_generation": False,
            "pair_label_rule": "same target_line_id=1; distinct target_line_id=0",
            "all_distinct_target_compatible_pairs_retained": True,
            "singleton_baseline_gain": 0.0,
            "validation_scored_after_train_partition": not args.skip_validation,
            "validation_score_passes": 0 if args.skip_validation else 1,
            "internal_test_read": False,
        },
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "pairing_profile": args.pairing_profile,
        "pairing_config": asdict(pairing_config),
        "target_config": asdict(target_config),
        "render_mode": "all_variants" if render_variant is None else "single_variant",
        "render_variant": render_variant,
        "identity": {
            "train": _identity_payload(train_identities),
            "fit": _identity_payload(list(partition.fit)),
            "calibration": _identity_payload(list(partition.calibration)),
            "validation": _identity_payload(validation_identities),
            "fit_calibration_component_count": partition.component_count,
            "calibration_component_count": partition.calibration_component_count,
            "partition_seed": partition.seed,
            "train_validation_disjoint": train_validation_disjoint,
            "fit_calibration_disjoint": validate_molecule_disjoint_splits(
                partition.fit,
                partition.calibration,
                left_name="fit",
                right_name="calibration",
            ),
        },
        "fit": fit_summary,
        "calibration": calibration_summary,
        "validation_final": validation_summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
