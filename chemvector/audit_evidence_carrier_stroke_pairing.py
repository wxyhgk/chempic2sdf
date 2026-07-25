#!/usr/bin/env python3
"""Audit target-free LSD-boundary pairing on sealed validation slices.

The first validation slice is explicitly exploratory calibration.  Every later
render sharing its molecule or source-structure identity is excluded before
the molecule-disjoint validation slice is scored.  Raster pixels and raw LSD
geometry are the only algorithm inputs.  GT sidecars are consulted after pairs
and output centerlines have been frozen, solely for metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chemvector.audit_evidence_carrier_proposals import extract_targets, file_sha256
from chemvector.evidence_carrier_extent_data import (
    IN_SCOPE_GROUP_KINDS,
    POSITIVE_LABEL,
    EvidenceCarrierExtentTargetConfig,
    EvidenceCarrierExtentWebDataset,
    associate_evidence_carriers,
    detect_evidence_carriers,
)
from chemvector.evidence_carrier_stroke_pairing import (
    StrokeBoundaryPairingConfig,
    pair_stroke_boundaries,
)


SCHEMA_VERSION = "evidence-carrier-stroke-pairing-audit-0.4"
DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v099-evidence-carrier-stroke-pairing-molecule-disjoint-validation192/"
    "audit.json"
)


@dataclass(frozen=True, slots=True)
class ValidationSampleIdentity:
    sample_key: str
    molecule_id: str
    source_structure_id: str


@dataclass(frozen=True, slots=True)
class MoleculeDisjointPartition:
    calibration_samples: tuple[dict[str, Any], ...]
    calibration_identities: tuple[ValidationSampleIdentity, ...]
    molecule_disjoint_samples: tuple[dict[str, Any], ...]
    molecule_disjoint_identities: tuple[ValidationSampleIdentity, ...]
    excluded_cross_render_samples: tuple[dict[str, Any], ...]
    excluded_cross_render_identities: tuple[ValidationSampleIdentity, ...]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sample_key_hash(keys: list[str]) -> str:
    payload = "".join(f"{key}\n" for key in keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_set_hash(values: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _selected_pair_geometry_keys(
    raw_segments: np.ndarray,
    selected_pairs: tuple[Any, ...],
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    result = []
    for pair in selected_pairs:
        source_keys = sorted(
            tuple(float(value) for value in raw_segments[index].reshape(-1))
            for index in (pair.first_index, pair.second_index)
        )
        result.append((source_keys[0], source_keys[1]))
    return tuple(sorted(result))


def validation_sample_identity(sample: dict[str, Any]) -> ValidationSampleIdentity:
    """Recover molecule and source-structure identity from sample metadata."""

    meta = sample.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Validation sample lacks metadata")
    sample_key = str(sample.get("key", ""))
    if not sample_key or str(meta.get("sample_key", sample_key)) != sample_key:
        raise ValueError("Validation sample key and metadata sample_key disagree")
    molecule_id = str(meta.get("molecule_id", ""))
    source_sample_key = str(meta.get("source_sample_key", ""))
    source_variant = str(meta.get("source_variant", meta.get("variant", "")))
    source_pilot = str(meta.get("source_pilot", ""))
    source_split = str(meta.get("source_split", ""))
    required = {
        "molecule_id": molecule_id,
        "source_sample_key": source_sample_key,
        "source_variant": source_variant,
        "source_pilot": source_pilot,
        "source_split": source_split,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Validation metadata identity fields are missing: {missing}")
    suffix = f"_{source_variant}"
    if not source_sample_key.endswith(suffix):
        raise ValueError(
            "source_sample_key does not end with its declared source_variant"
        )
    source_record = source_sample_key[: -len(suffix)]
    if not source_record:
        raise ValueError("Derived source structure record is empty")
    return ValidationSampleIdentity(
        sample_key=sample_key,
        molecule_id=molecule_id,
        source_structure_id=f"{source_pilot}|{source_split}|{source_record}",
    )


def partition_molecule_disjoint_validation(
    samples: list[dict[str, Any]],
    *,
    calibration_count: int,
) -> MoleculeDisjointPartition:
    """Exclude every post-calibration sample sharing molecule or source ID."""

    if calibration_count < 1 or calibration_count >= len(samples):
        raise ValueError("calibration_count must split a nonempty sample sequence")
    identities = [validation_sample_identity(sample) for sample in samples]
    sample_keys = [identity.sample_key for identity in identities]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Validation sample keys must be unique")
    calibration_samples = tuple(samples[:calibration_count])
    calibration_identities = tuple(identities[:calibration_count])
    calibration_molecules = {
        identity.molecule_id for identity in calibration_identities
    }
    calibration_structures = {
        identity.source_structure_id for identity in calibration_identities
    }
    disjoint_samples = []
    disjoint_identities = []
    excluded_samples = []
    excluded_identities = []
    for sample, identity in zip(
        samples[calibration_count:], identities[calibration_count:]
    ):
        overlaps = (
            identity.molecule_id in calibration_molecules
            or identity.source_structure_id in calibration_structures
        )
        if overlaps:
            excluded_samples.append(sample)
            excluded_identities.append(identity)
        else:
            disjoint_samples.append(sample)
            disjoint_identities.append(identity)
    return MoleculeDisjointPartition(
        calibration_samples=calibration_samples,
        calibration_identities=calibration_identities,
        molecule_disjoint_samples=tuple(disjoint_samples),
        molecule_disjoint_identities=tuple(disjoint_identities),
        excluded_cross_render_samples=tuple(excluded_samples),
        excluded_cross_render_identities=tuple(excluded_identities),
    )


def _positive_indices_by_line(target: dict[str, Any]) -> dict[int, list[int]]:
    labels = target["keep_labels"].detach().cpu().numpy()
    line_ids = target["target_line_id"].detach().cpu().numpy()
    result: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(labels == POSITIVE_LABEL):
        result[int(line_ids[index])].append(int(index))
    return dict(result)


def _label_histogram(target: dict[str, Any]) -> dict[str, int]:
    labels = target["keep_labels"].detach().cpu().numpy()
    return {
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "ignored": int(np.sum(labels == -1)),
        "total": int(len(labels)),
    }


def audit_sample(
    sample: dict[str, Any],
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    pairing_config: StrokeBoundaryPairingConfig,
) -> dict[str, Any]:
    """Freeze raw pairing first, then expose target sidecars to metric code."""

    image = sample["image"]
    raw_segments = detect_evidence_carriers(
        image,
        target_config=target_config,
    )
    pairing = pair_stroke_boundaries(
        raw_segments,
        image,
        config=pairing_config,
    )
    reversed_segments = raw_segments[::-1].copy()
    reversed_pairing = pair_stroke_boundaries(
        reversed_segments,
        image,
        config=pairing_config,
    )
    carrier_set_equal = (
        pairing.segments_pixels().shape == reversed_pairing.segments_pixels().shape
        and bool(
            np.allclose(
                pairing.segments_pixels(),
                reversed_pairing.segments_pixels(),
                atol=1e-9,
                rtol=0.0,
            )
        )
    )
    pair_partition_equal = _selected_pair_geometry_keys(
        raw_segments,
        pairing.selected_pairs,
    ) == _selected_pair_geometry_keys(
        reversed_segments,
        reversed_pairing.selected_pairs,
    )
    packed_segments = pairing.segments_pixels()

    # GT is intentionally read only below this boundary.  Neither pair
    # selection nor centerline coordinates can change after these calls.
    raw_target = associate_evidence_carriers(
        raw_segments,
        sample["primitives"],
        sample["layers"],
        target_config=target_config,
    )
    packed_target = associate_evidence_carriers(
        packed_segments,
        sample["primitives"],
        sample["layers"],
        target_config=target_config,
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
    expected_ids = set(ordinary)
    raw_by_line = _positive_indices_by_line(raw_target)
    packed_by_line = _positive_indices_by_line(packed_target)
    raw_supported = expected_ids.intersection(raw_by_line)
    packed_supported = expected_ids.intersection(packed_by_line)

    raw_labels = raw_target["keep_labels"].detach().cpu().numpy()
    raw_line_ids = raw_target["target_line_id"].detach().cpu().numpy()
    pair_classes = Counter()
    selected_source_pairs: set[frozenset[int]] = set()
    cross_target_records = []
    for pair in pairing.selected_pairs:
        indices = (pair.first_index, pair.second_index)
        selected_source_pairs.add(frozenset(indices))
        positive = [raw_labels[index] == POSITIVE_LABEL for index in indices]
        if all(positive):
            first_id, second_id = (int(raw_line_ids[index]) for index in indices)
            if first_id == second_id:
                pair_classes["same_target_positive"] += 1
            else:
                pair_classes["cross_target_positive"] += 1
                cross_target_records.append(
                    {
                        "source_indices": list(indices),
                        "target_line_ids": [first_id, second_id],
                        "group_indices": [
                            ordinary[first_id].group_index,
                            ordinary[second_id].group_index,
                        ],
                        "group_kinds": [
                            ordinary[first_id].group_kind,
                            ordinary[second_id].group_kind,
                        ],
                        "angle_degrees": pair.geometry.angle_degrees,
                        "normal_separation_pixels": (
                            pair.geometry.normal_separation_pixels
                        ),
                        "shorter_interval_overlap": (
                            pair.geometry.shorter_interval_overlap
                        ),
                        "interior_ink_occupancy": (
                            pair.geometry.interior_ink_occupancy
                        ),
                    }
                )
        elif any(positive):
            pair_classes["one_positive_member"] += 1
        else:
            pair_classes["no_positive_member"] += 1

    multiplicity_two_expected = 0
    multiplicity_two_paired = 0
    for indices in raw_by_line.values():
        if len(indices) != 2:
            continue
        multiplicity_two_expected += 1
        multiplicity_two_paired += int(
            frozenset(indices) in selected_source_pairs
        )

    raw_positive_indices = {
        index
        for indices in raw_by_line.values()
        for index in indices
    }
    paired_source_indices = {
        index
        for pair in pairing.selected_pairs
        for index in (pair.first_index, pair.second_index)
    }

    p2_total = p2_raw_supported = p2_preserved = p2_direct_merge = 0
    for group in target_groups:
        if group.kind != "parallel_2" or len(group.line_indices) != 2:
            continue
        p2_total += 1
        first, second = map(int, group.line_indices)
        if first not in raw_supported or second not in raw_supported:
            continue
        p2_raw_supported += 1
        direct_merge = any(
            set(record["target_line_ids"]) == {first, second}
            for record in cross_target_records
        )
        p2_direct_merge += int(direct_merge)
        output_supported = first in packed_supported and second in packed_supported
        p2_preserved += int(output_supported and not direct_merge)

    raw_positive_count = sum(len(indices) for indices in raw_by_line.values())
    packed_positive_count = sum(
        len(indices) for indices in packed_by_line.values()
    )
    packed_multiplicity = Counter(
        len(packed_by_line.get(line_id, ())) for line_id in expected_ids
    )
    support_exact = bool(expected_ids) and packed_supported == expected_ids
    one_carrier_exact = (
        support_exact
        and all(len(packed_by_line.get(line_id, ())) == 1 for line_id in expected_ids)
        and pair_classes["cross_target_positive"] == 0
    )
    return {
        "sample_key": str(sample["key"]),
        "raw_labels": _label_histogram(raw_target),
        "packed_labels": _label_histogram(packed_target),
        "compatible_pairs": pairing.compatible_pair_count,
        "selected_pairs": len(pairing.selected_pairs),
        "unpaired_raw_anchors": len(raw_segments) - 2 * len(pairing.selected_pairs),
        "unpaired_positive_raw_anchors": len(
            raw_positive_indices.difference(paired_source_indices)
        ),
        "permutation_determinism": {
            "reverse_input_carrier_set_equal": carrier_set_equal,
            "reverse_input_pair_partition_equal": pair_partition_equal,
            "exact": carrier_set_equal and pair_partition_equal,
        },
        "pair_classes": dict(pair_classes),
        "cross_target_records": cross_target_records,
        "multiplicity_two": {
            "expected": multiplicity_two_expected,
            "paired": multiplicity_two_paired,
        },
        "extent_eligible_support": {
            "expected_ordinary_members": len(expected_ids),
            "raw_supported_members": len(raw_supported),
            "packed_supported_members": len(packed_supported),
            "raw_positive_carriers": raw_positive_count,
            "packed_positive_carriers": packed_positive_count,
            "raw_gt_positive_dedup_ratio": _ratio(
                len(raw_supported), raw_positive_count
            ),
            "packed_gt_positive_dedup_ratio": _ratio(
                len(packed_supported), packed_positive_count
            ),
            "lost_raw_supported_members": len(
                raw_supported.difference(packed_supported)
            ),
        },
        "packed_positive_multiplicity": {
            str(key): value for key, value in sorted(packed_multiplicity.items())
        },
        "parallel_2": {
            "total": p2_total,
            "raw_both_supported": p2_raw_supported,
            "strict_preserved": p2_preserved,
            "direct_cross_target_merges": p2_direct_merge,
        },
        "sample_exact": {
            "extent_eligible_support_all_members": support_exact,
            "one_positive_carrier_per_ordinary_member": one_carrier_exact,
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_classes = Counter()
    raw_labels = Counter()
    packed_labels = Counter()
    multiplicity = Counter()
    totals = Counter()
    cross_samples = support_exact = one_carrier_exact = permutation_exact = 0
    for record in records:
        pair_classes.update(record["pair_classes"])
        raw_labels.update(record["raw_labels"])
        packed_labels.update(record["packed_labels"])
        multiplicity.update(
            {int(key): value for key, value in record["packed_positive_multiplicity"].items()}
        )
        totals["compatible_pairs"] += record["compatible_pairs"]
        totals["selected_pairs"] += record["selected_pairs"]
        totals["unpaired_raw_anchors"] += record["unpaired_raw_anchors"]
        totals["unpaired_positive_raw_anchors"] += record[
            "unpaired_positive_raw_anchors"
        ]
        for key, value in record["multiplicity_two"].items():
            totals[f"multiplicity_two_{key}"] += value
        for key, value in record["extent_eligible_support"].items():
            if isinstance(value, int):
                totals[f"support_{key}"] += value
        for key, value in record["parallel_2"].items():
            totals[f"p2_{key}"] += value
        cross_samples += int(
            record["pair_classes"].get("cross_target_positive", 0) > 0
        )
        support_exact += int(
            record["sample_exact"]["extent_eligible_support_all_members"]
        )
        one_carrier_exact += int(
            record["sample_exact"]["one_positive_carrier_per_ordinary_member"]
        )
        permutation_exact += int(record["permutation_determinism"]["exact"])

    same_pairs = pair_classes["same_target_positive"]
    selected_pairs = totals["selected_pairs"]
    positive_positive_pairs = same_pairs + pair_classes["cross_target_positive"]
    expected_lines = totals["support_expected_ordinary_members"]
    raw_supported = totals["support_raw_supported_members"]
    packed_supported = totals["support_packed_supported_members"]
    raw_positive = totals["support_raw_positive_carriers"]
    packed_positive = totals["support_packed_positive_carriers"]
    samples = len(records)
    return {
        "samples": samples,
        "anchors": {
            "raw": raw_labels["total"],
            "packed": packed_labels["total"],
            "selected_pairs": selected_pairs,
            "compatible_pairs": totals["compatible_pairs"],
            "all_anchor_reduction": raw_labels["total"] - packed_labels["total"],
            "all_anchor_reduction_rate": _ratio(
                raw_labels["total"] - packed_labels["total"], raw_labels["total"]
            ),
            "unpaired_raw": totals["unpaired_raw_anchors"],
            "unpaired_positive_raw": totals["unpaired_positive_raw_anchors"],
            "raw_label_histogram": dict(raw_labels),
            "packed_label_histogram": dict(packed_labels),
        },
        "pairing": {
            "classes": dict(pair_classes),
            "same_target_fraction_all_selected": _ratio(
                same_pairs, selected_pairs
            ),
            "same_target_precision_among_two_gt_positive_members": _ratio(
                same_pairs, positive_positive_pairs
            ),
            "cross_target_merges": pair_classes["cross_target_positive"],
            "cross_target_merge_samples": cross_samples,
            "multiplicity_two_expected": totals["multiplicity_two_expected"],
            "multiplicity_two_paired": totals["multiplicity_two_paired"],
            "multiplicity_two_pair_recall": _ratio(
                totals["multiplicity_two_paired"],
                totals["multiplicity_two_expected"],
            ),
            "permutation_determinism": {
                "samples_checked": samples,
                "exact_samples": permutation_exact,
                "exact_rate": _ratio(permutation_exact, samples),
                "permutation": "reverse raw LSD input order",
            },
        },
        "extent_eligible_carrier_support": {
            "expected_ordinary_members": expected_lines,
            "raw_supported_members": raw_supported,
            "packed_supported_members": packed_supported,
            "raw_support_coverage": _ratio(raw_supported, expected_lines),
            "packed_support_coverage": _ratio(packed_supported, expected_lines),
            "support_coverage_loss_percentage_points": (
                100.0 * (raw_supported - packed_supported) / expected_lines
                if expected_lines
                else None
            ),
            "raw_positive_carriers": raw_positive,
            "packed_positive_carriers": packed_positive,
            "raw_gt_positive_dedup_ratio": _ratio(raw_supported, raw_positive),
            "packed_gt_positive_dedup_ratio": _ratio(
                packed_supported, packed_positive
            ),
            "positive_carrier_reduction": raw_positive - packed_positive,
            "positive_carrier_reduction_rate": _ratio(
                raw_positive - packed_positive, raw_positive
            ),
            "packed_positive_multiplicity": {
                str(key): value for key, value in sorted(multiplicity.items())
            },
        },
        "parallel_2_distinct_preservation": {
            "total": totals["p2_total"],
            "raw_both_supported": totals["p2_raw_both_supported"],
            "strict_preserved": totals["p2_strict_preserved"],
            "rate": _ratio(
                totals["p2_strict_preserved"], totals["p2_raw_both_supported"]
            ),
            "direct_cross_target_merges": totals[
                "p2_direct_cross_target_merges"
            ],
        },
        "sample_exact": {
            "extent_eligible_support_all_members": support_exact,
            "extent_eligible_support_all_members_rate": _ratio(
                support_exact, samples
            ),
            "one_positive_carrier_per_ordinary_member": one_carrier_exact,
            "one_positive_carrier_per_ordinary_member_rate": _ratio(
                one_carrier_exact, samples
            ),
        },
    }


def hard_preprocessor_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "cross_target_merges_equal_0": (
            summary["pairing"]["cross_target_merges"] == 0
        ),
        "one_positive_member_pairs_equal_0": (
            summary["pairing"]["classes"].get("one_positive_member", 0) == 0
        ),
        "permutation_determinism_equal_1": (
            summary["pairing"]["permutation_determinism"]["exact_rate"] == 1.0
        ),
        "extent_eligible_support_coverage_at_least_0_995": (
            (
                summary["extent_eligible_carrier_support"][
                    "packed_support_coverage"
                ]
                or 0.0
            )
            >= 0.995
        ),
        "support_coverage_loss_at_most_0_1pp": (
            (
                summary["extent_eligible_carrier_support"][
                    "support_coverage_loss_percentage_points"
                ]
                or 0.0
            )
            <= 0.1
        ),
        "parallel_2_preservation_equal_1": (
            summary["parallel_2_distinct_preservation"]["rate"] == 1.0
        ),
        "multiplicity_two_pair_recall_at_least_0_98": (
            (summary["pairing"]["multiplicity_two_pair_recall"] or 0.0) >= 0.98
        ),
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "scope": "deterministic hard preprocessor only",
        "checks": checks,
        "thresholds": {
            "cross_target_merges": 0,
            "one_positive_member_pairs": 0,
            "permutation_determinism": 1.0,
            "extent_eligible_support_coverage": 0.995,
            "maximum_support_coverage_loss_percentage_points": 0.1,
            "parallel_2_distinct_preservation": 1.0,
            "multiplicity_two_pair_recall": 0.98,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.calibration_count < 1 or args.validation_count <= args.calibration_count:
        raise ValueError("validation-count must exceed a positive calibration-count")
    resolved_shard = Path(args.shard).resolve()
    if "internal_test" in str(resolved_shard).lower().replace("-", "_"):
        raise RuntimeError(f"Refusing sealed internal_test shard: {resolved_shard}")
    target_config = EvidenceCarrierExtentTargetConfig()
    pairing_config = StrokeBoundaryPairingConfig()
    samples = list(
        EvidenceCarrierExtentWebDataset(
            str(resolved_shard),
            limit=args.validation_count,
        )
    )
    if len(samples) != args.validation_count:
        raise RuntimeError(
            f"Requested {args.validation_count} validation samples but read {len(samples)}"
        )
    partition = partition_molecule_disjoint_validation(
        samples,
        calibration_count=args.calibration_count,
    )
    calibration_records = [
        audit_sample(
            sample,
            target_config=target_config,
            pairing_config=pairing_config,
        )
        for sample in partition.calibration_samples
    ]
    disjoint_records = [
        audit_sample(
            sample,
            target_config=target_config,
            pairing_config=pairing_config,
        )
        for sample in partition.molecule_disjoint_samples
    ]

    calibration_summary = summarize_records(calibration_records)
    disjoint_summary = summarize_records(disjoint_records)
    calibration_keys = [
        identity.sample_key for identity in partition.calibration_identities
    ]
    disjoint_keys = [
        identity.sample_key for identity in partition.molecule_disjoint_identities
    ]
    excluded_keys = [
        identity.sample_key for identity in partition.excluded_cross_render_identities
    ]
    calibration_molecules = {
        identity.molecule_id for identity in partition.calibration_identities
    }
    disjoint_molecules = {
        identity.molecule_id
        for identity in partition.molecule_disjoint_identities
    }
    excluded_molecules = {
        identity.molecule_id
        for identity in partition.excluded_cross_render_identities
    }
    calibration_structures = {
        identity.source_structure_id
        for identity in partition.calibration_identities
    }
    disjoint_structures = {
        identity.source_structure_id
        for identity in partition.molecule_disjoint_identities
    }
    excluded_structures = {
        identity.source_structure_id
        for identity in partition.excluded_cross_render_identities
    }
    molecule_overlap = calibration_molecules.intersection(disjoint_molecules)
    structure_overlap = calibration_structures.intersection(disjoint_structures)
    if molecule_overlap or structure_overlap:
        raise RuntimeError("Molecule-disjoint validation partition leaked identity")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard": str(resolved_shard),
        "shard_sha256": file_sha256(resolved_shard),
        "sealed_internal_test_read": False,
        "opencv_version": cv2.__version__,
        "scope": (
            "ordinary single_line/parallel_2/parallel_3 members; "
            "carrier support only, without extent/q8 endpoint evaluation"
        ),
        "protocol": {
            "calibration_is_exploratory": True,
            "fixed_pairing_config_sha256": _canonical_json_hash(
                asdict(pairing_config)
            ),
            "fixed_config_declaration": (
                "configuration is declared fixed for this audit; the artifact "
                "does not prove chronology"
            ),
            "molecule_disjoint_filter_uses_metadata_only": True,
            "molecule_identity_fields": [
                "molecule_id",
                "source_pilot + source_split + source_sample_key_without_variant",
            ],
            "target_visible_to_pairing": False,
            "gt_used_only_after_pairs_and_centerlines_frozen": True,
            "unpaired_lsd_anchors_retained": True,
            "pair_centerline_uses_common_overlap_extent": True,
            "gt_positive_dedup_ratio_definition": (
                "ordinary members with extent-eligible support / carriers "
                "labeled GT-positive after packing; excludes ignored and negative "
                "outputs and is neither proposal precision nor line precision"
            ),
            "selection_tie_break": (
                "canonical segment geometry; independent of input array index"
            ),
            "pairing_config": asdict(pairing_config),
            "target_config": asdict(target_config),
        },
        "sample_partitions": {
            "validation_samples_scanned": len(samples),
            "calibration": {
                "count": len(calibration_keys),
                "sample_key_sha256": _sample_key_hash(calibration_keys),
                "molecule_count": len(calibration_molecules),
                "molecule_id_sha256": _identity_set_hash(calibration_molecules),
                "source_structure_count": len(calibration_structures),
                "source_structure_id_sha256": _identity_set_hash(
                    calibration_structures
                ),
                "first_key": calibration_keys[0],
                "last_key": calibration_keys[-1],
            },
            "excluded_same_molecule_cross_render": {
                "count": len(excluded_keys),
                "sample_key_sha256": _sample_key_hash(excluded_keys),
                "molecule_count": len(excluded_molecules),
                "molecule_id_sha256": _identity_set_hash(excluded_molecules),
                "source_structure_count": len(excluded_structures),
                "source_structure_id_sha256": _identity_set_hash(
                    excluded_structures
                ),
                "former_holdout400_role": "same-molecule cross-render diagnostic only",
                "former_artifact": (
                    "outputs/evaluations/"
                    "chemvector-v099-evidence-carrier-stroke-pairing-"
                    "validation-holdout400/audit.json"
                ),
            },
            "molecule_disjoint_validation": {
                "count": len(disjoint_keys),
                "sample_key_sha256": _sample_key_hash(disjoint_keys),
                "molecule_count": len(disjoint_molecules),
                "molecule_id_sha256": _identity_set_hash(disjoint_molecules),
                "source_structure_count": len(disjoint_structures),
                "source_structure_id_sha256": _identity_set_hash(
                    disjoint_structures
                ),
                "first_key": disjoint_keys[0],
                "last_key": disjoint_keys[-1],
            },
            "sample_key_overlap_count": len(
                set(calibration_keys).intersection(disjoint_keys)
            ),
            "molecule_id_overlap_count": len(molecule_overlap),
            "source_structure_id_overlap_count": len(structure_overlap),
        },
        "calibration_summary": calibration_summary,
        "molecule_disjoint_summary": disjoint_summary,
        "molecule_disjoint_gate": hard_preprocessor_gate(disjoint_summary),
        "calibration_records": calibration_records,
        "molecule_disjoint_records": disjoint_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample_partitions": payload["sample_partitions"],
                "calibration_summary": calibration_summary,
                "molecule_disjoint_summary": disjoint_summary,
                "molecule_disjoint_gate": payload["molecule_disjoint_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
