#!/usr/bin/env python3
"""Post-hoc audit of the exposed v0.9.16 tau calibration result.

This audit is deliberately read-only with respect to models and datasets.  It
consumes only the formal calibration JSON and the committed interval manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


DEFAULT_CALIBRATION = Path(
    "outputs/evaluations/chemvector-v0916-sample-tau-predictor/calibration.json"
)
DEFAULT_MANIFEST = Path(
    "outputs/cache/chemvector-v0916-sample-tau-interval-manifest.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/evaluations/chemvector-v0916-sample-tau-predictor-posthoc-audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_number(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    result = float(value)
    if math.isinf(result):
        return "+inf" if result > 0 else "-inf"
    if math.isnan(result):
        return "nan"
    return result


def signed_interval_adjustment(tau: float, lower: float, upper: float) -> float:
    if tau < lower:
        return lower - tau
    if tau > upper:
        return upper - tau
    return 0.0


def is_target_consistent(evaluation: Mapping[str, Any]) -> bool:
    selected = int(evaluation["selected_pair_count"])
    precision = evaluation.get("selected_pair_precision")
    precision_ok = selected == 0 or precision == 1.0
    pairable_ok = (
        not bool(evaluation["pairable_sample_eligible"])
        or evaluation.get("pairable_sample_exact") is True
    )
    return bool(
        int(evaluation["selected_cross_target_pairs"]) == 0
        and int(evaluation["parallel_2_violated_group_count"]) == 0
        and int(evaluation["multiplicity_2_matched_count"])
        == int(evaluation["multiplicity_2_target_count"])
        and int(evaluation["ordinary_support_lost_target_count"]) == 0
        and precision_ok
        and pairable_ok
    )


def identity_label(identity: Mapping[str, Any]) -> str:
    return f"{identity['sample_key']}:{identity['molecule_id']}:{identity['render_variant']}"


def candidate_detail(
    record: Mapping[str, Any],
    source_candidate_index: int,
    descriptor_names: Sequence[str],
) -> dict[str, Any]:
    target_free = record["target_free"]
    source_to_canonical = torch.as_tensor(
        target_free["source_to_canonical_candidate_indices"], dtype=torch.long
    )
    canonical_index = int(source_to_canonical[source_candidate_index])
    pair_indices = torch.as_tensor(target_free["pair_indices"], dtype=torch.long)
    pair_tokens = torch.as_tensor(target_free["pair_tokens"], dtype=torch.float32)
    token = pair_tokens[canonical_index]
    descriptor = {
        name: float(token[index]) for index, name in enumerate(descriptor_names)
    }
    return {
        "source_candidate_index": int(source_candidate_index),
        "canonical_candidate_index": canonical_index,
        "anchor_pair": [int(value) for value in pair_indices[canonical_index].tolist()],
        "raw_gain": float(token[-1]),
        "normalized_descriptor": descriptor,
    }


def compact_interval(interval: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss_lower_threshold": json_number(interval["loss_lower_threshold"]),
        "loss_upper_threshold": json_number(interval["loss_upper_threshold"]),
        "production_lower_threshold": json_number(interval["lower_threshold"]),
        "production_upper_threshold": json_number(interval["upper_threshold"]),
        "state_cardinality": int(interval["state_cardinality"]),
        "selected_candidate_indices": [
            int(value) for value in interval["selected_candidate_indices"]
        ],
    }


def choose_reference_interval(
    tau: float,
    predicted_indices: set[int],
    intervals: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], bool, float]:
    state_matches = [
        interval
        for interval in intervals
        if predicted_indices == set(map(int, interval["selected_candidate_indices"]))
    ]
    pool = state_matches or list(intervals)
    ranked = []
    for interval in pool:
        adjustment = signed_interval_adjustment(
            tau,
            float(interval["loss_lower_threshold"]),
            float(interval["loss_upper_threshold"]),
        )
        ranked.append((abs(adjustment), adjustment, interval))
    _, adjustment, selected = min(ranked, key=lambda item: (item[0], item[1]))
    return selected, bool(state_matches), float(adjustment)


def raw_count_features(record: Mapping[str, Any], names: Sequence[str]) -> dict[str, int]:
    counts = torch.as_tensor(record["target_free"]["raw_global_counts"], dtype=torch.long)
    return {name: int(counts[index]) for index, name in enumerate(names)}


def candidate_components(
    record: Mapping[str, Any],
    selected_source_indices: set[int],
    descriptor_names: Sequence[str],
) -> list[dict[str, Any]]:
    target_free = record["target_free"]
    pairs = [tuple(map(int, row)) for row in target_free["pair_indices"].tolist()]
    canonical_to_source = [
        int(value)
        for value in target_free["canonical_to_source_candidate_indices"].tolist()
    ]
    incident: dict[int, list[int]] = defaultdict(list)
    for candidate_index, (first, second) in enumerate(pairs):
        incident[first].append(candidate_index)
        incident[second].append(candidate_index)
    seen: set[int] = set()
    components = []
    for start in sorted(incident):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        nodes: set[int] = set()
        edges: set[int] = set()
        while stack:
            node = stack.pop()
            nodes.add(node)
            for canonical_index in incident[node]:
                edges.add(canonical_index)
                first, second = pairs[canonical_index]
                other = second if first == node else first
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        if len(edges) <= 1:
            continue
        candidates = []
        for canonical_index in sorted(edges):
            source_index = canonical_to_source[canonical_index]
            detail = candidate_detail(record, source_index, descriptor_names)
            detail["selected"] = source_index in selected_source_indices
            candidates.append(detail)
        components.append(
            {
                "anchor_ids": sorted(nodes),
                "candidate_count": len(edges),
                "candidates": candidates,
            }
        )
    return components


def format_adjustment(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value > 0:
        return f"+{value:.6f}"
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)

    forbidden_flags = (
        "fresh_read",
        "sealed_read",
        "validation_read",
        "internal_test_read",
        "mr_tadf_read",
    )
    for flag in forbidden_flags:
        if calibration.get(flag) is not False or manifest.get(flag) is not False:
            raise RuntimeError(f"Forbidden-read flag is not false: {flag}")
    if calibration.get("status") != "CALIBRATION-TAU-PREDICTOR-NO-GO":
        raise RuntimeError("Expected the exposed v0.9.16 NO-GO calibration")
    if calibration.get("threshold_search") is not False:
        raise RuntimeError("Calibration artifact unexpectedly reports threshold search")
    if calibration.get("checkpoint_sha256_unchanged") is not True:
        raise RuntimeError("Calibration changed the frozen checkpoint")

    records = manifest["partitions"]["calibration"]["records"]
    evaluations = calibration["sample_evaluations"]
    if len(records) != 900 or len(evaluations) != 900:
        raise RuntimeError("Expected exactly 900 formal calibration samples")
    for record, evaluation in zip(records, evaluations):
        if record["identity"] != evaluation["identity"]:
            raise RuntimeError("Calibration and manifest sample identities differ")

    descriptor_names = manifest["feature_contract"][
        "normalized_descriptor_feature_names"
    ]
    raw_count_names = manifest["feature_contract"]["raw_global_count_feature_names"]

    event_sets: dict[str, set[int]] = {
        "cross": set(),
        "p2_violation": set(),
        "support_loss": set(),
        "interval_miss": set(),
        "pairable_not_exact": set(),
    }
    interval_rows = []
    per_sample: dict[int, dict[str, Any]] = {}

    for index, (record, item) in enumerate(zip(records, evaluations)):
        structural = item["structural_evaluation"]
        if int(structural["selected_cross_target_pairs"]):
            event_sets["cross"].add(index)
        if int(structural["parallel_2_violated_group_count"]):
            event_sets["p2_violation"].add(index)
        if int(structural["ordinary_support_lost_target_count"]):
            event_sets["support_loss"].add(index)
        if item["robust_interval_hit"] is False:
            event_sets["interval_miss"].add(index)
        if (
            structural["pairable_sample_eligible"]
            and structural["pairable_sample_exact"] is False
        ):
            event_sets["pairable_not_exact"].add(index)

        target_aware = record["target_aware"]
        sample = {
            "sample_index": index,
            "identity": dict(record["identity"]),
            "predicted_tau": float(item["predicted_tau"]),
            "predicted_delta": float(item["predicted_delta"]),
            "loss_eligible": bool(item["loss_eligible"]),
            "loss_exclusion_reason": item["loss_exclusion_reason"],
            "raw_global_counts": raw_count_features(record, raw_count_names),
            "candidate_unreachable_anchor_pairs": target_aware[
                "candidate_unreachable_anchor_pairs"
            ],
            "selected_metrics": {
                "candidate_pair_count": int(structural["candidate_pair_count"]),
                "selected_pair_count": int(structural["selected_pair_count"]),
                "selected_same_target_pairs": int(
                    structural["selected_same_target_pairs"]
                ),
                "selected_cross_target_pairs": int(
                    structural["selected_cross_target_pairs"]
                ),
                "multiplicity_2_matched_count": int(
                    structural["multiplicity_2_matched_count"]
                ),
                "multiplicity_2_target_count": int(
                    structural["multiplicity_2_target_count"]
                ),
                "parallel_2_violated_group_count": int(
                    structural["parallel_2_violated_group_count"]
                ),
                "ordinary_support_lost_target_count": int(
                    structural["ordinary_support_lost_target_count"]
                ),
                "pairable_sample_exact": structural["pairable_sample_exact"],
                "target_consistent": is_target_consistent(structural),
            },
        }
        per_sample[index] = sample

        if item["robust_interval_hit"] is False:
            predicted_indices = set(
                map(int, structural["selected_candidate_indices"])
            )
            interval, state_equivalent, adjustment = choose_reference_interval(
                float(item["predicted_tau"]),
                predicted_indices,
                target_aware["oracle_equivalent_robust_intervals"],
            )
            oracle_indices = set(map(int, interval["selected_candidate_indices"]))
            predicted_only = sorted(predicted_indices - oracle_indices)
            oracle_only = sorted(oracle_indices - predicted_indices)
            target_consistent = is_target_consistent(structural)
            if state_equivalent:
                category = "robust_inset_only_same_oracle_state"
            elif int(structural["selected_cross_target_pairs"]):
                category = "scalar_too_low_cross_target_selection"
            elif (
                int(structural["multiplicity_2_matched_count"])
                < int(structural["multiplicity_2_target_count"])
            ):
                category = "scalar_too_high_multiplicity_2_miss"
            else:
                category = "other_scalar_displacement"
            interval_row = {
                **sample,
                "category": category,
                "state_equivalent_to_oracle": state_equivalent,
                "structurally_target_consistent": target_consistent,
                "signed_adjustment_to_robust_interval": adjustment,
                "reference_oracle_interval": compact_interval(interval),
                "predicted_only_candidates": [
                    candidate_detail(record, source_index, descriptor_names)
                    for source_index in predicted_only
                ],
                "oracle_only_candidates": [
                    candidate_detail(record, source_index, descriptor_names)
                    for source_index in oracle_only
                ],
            }
            interval_rows.append(interval_row)

    cross_rows = []
    for index in sorted(event_sets["cross"]):
        sample = per_sample[index]
        record = records[index]
        item = evaluations[index]
        reason = sample["loss_exclusion_reason"]
        if reason == "candidate_unreachable":
            root_cause = "candidate_absence_present"
        elif reason == "scalar_unreachable":
            root_cause = "pair_specific_local_ranking"
        elif index in event_sets["interval_miss"]:
            root_cause = "per_image_scalar_displacement"
        else:
            root_cause = "unclassified"
        matching_interval_row = next(
            (row for row in interval_rows if row["sample_index"] == index), None
        )
        selected_sources = set(
            map(
                int,
                item["structural_evaluation"]["selected_candidate_indices"],
            )
        )
        candidate_absence_evidence = None
        if reason == "candidate_unreachable":
            missing_anchors = {
                anchor
                for pair in record["target_aware"][
                    "candidate_unreachable_anchor_pairs"
                ]
                for anchor in pair
            }
            incident_selected = []
            for source_index in sorted(selected_sources):
                detail = candidate_detail(record, source_index, descriptor_names)
                if missing_anchors.intersection(detail["anchor_pair"]):
                    incident_selected.append(detail)
            candidate_absence_evidence = {
                "missing_expected_anchor_pairs": record["target_aware"][
                    "candidate_unreachable_anchor_pairs"
                ],
                "selected_candidates_incident_to_missing_target_anchors": (
                    incident_selected
                ),
                "attribution": (
                    "cross_localized_to_missing-target incident candidate"
                    if incident_selected
                    else "candidate absence and cross coexist; exact cross attribution "
                    "is unavailable in the formal manifest"
                ),
            }
        cross_rows.append(
            {
                **sample,
                "root_cause": root_cause,
                "overlaps": [
                    name
                    for name, values in event_sets.items()
                    if index in values and name != "cross"
                ],
                "signed_adjustment_to_robust_interval": (
                    matching_interval_row[
                        "signed_adjustment_to_robust_interval"
                    ]
                    if matching_interval_row
                    else None
                ),
                "predicted_only_candidates_relative_to_oracle": (
                    matching_interval_row["predicted_only_candidates"]
                    if matching_interval_row
                    else []
                ),
                "candidate_level_cross_identity": (
                    "exact"
                    if matching_interval_row
                    and len(
                        matching_interval_row[
                            "predicted_only_candidates"
                        ]
                    )
                    == 1
                    else "not_encoded_by_formal_manifest"
                ),
                "candidate_absence_evidence": candidate_absence_evidence,
            }
        )

    scalar_unreachable_rows = []
    for index, record in enumerate(records):
        if record["target_aware"]["exclusion_reason"] != "scalar_unreachable":
            continue
        item = evaluations[index]
        selected_sources = set(
            map(
                int,
                item["structural_evaluation"]["selected_candidate_indices"],
            )
        )
        components = candidate_components(record, selected_sources, descriptor_names)
        row = {
            **per_sample[index],
            "parametric_hull_state_count": int(
                record["target_aware"]["parametric_hull_state_count"]
            ),
            "target_consistent_state_count": int(
                record["target_aware"]["target_consistent_state_count"]
            ),
            "multi_candidate_components": components,
            "pair_specific_correction_required": True,
        }
        if index == 695:
            correct = candidate_detail(record, 0, descriptor_names)
            cross = candidate_detail(record, 30, descriptor_names)
            row["localized_rank_inversion"] = {
                "available_correct_candidate": correct,
                "selected_cross_candidate": cross,
                "cross_minus_correct_raw_gain": (
                    cross["raw_gain"] - correct["raw_gain"]
                ),
                "derivation": (
                    "There are 40 candidates, 39 selections and 39 M2 targets; "
                    "source 0 is the only unselected edge and the sample misses "
                    "exactly one M2 target, while source 30 shares anchor 57 and "
                    "is the sole selected cross edge."
                ),
            }
        scalar_unreachable_rows.append(row)

    def indices_to_labels(indices: set[int]) -> list[str]:
        return [identity_label(records[index]["identity"]) for index in sorted(indices)]

    intersections = {}
    event_names = list(event_sets)
    for first_index, first in enumerate(event_names):
        for second in event_names[first_index + 1 :]:
            overlap = event_sets[first] & event_sets[second]
            if overlap:
                intersections[f"{first}&{second}"] = {
                    "sample_count": len(overlap),
                    "samples": indices_to_labels(overlap),
                }

    structural_interval_failures = [
        row for row in interval_rows if not row["structurally_target_consistent"]
    ]
    robust_only_misses = [
        row for row in interval_rows if row["state_equivalent_to_oracle"]
    ]
    higher = [
        row
        for row in structural_interval_failures
        if row["signed_adjustment_to_robust_interval"] > 0
    ]
    lower = [
        row
        for row in structural_interval_failures
        if row["signed_adjustment_to_robust_interval"] < 0
    ]

    report = {
        "schema_version": "evidence-carrier-v0916-tau-posthoc-failure-audit-0.1",
        "status": "POSTHOC-AUDIT-COMPLETE",
        "scope": {
            "analysis_only": True,
            "calibration_exposed_before_audit": True,
            "tau_predictor_tuned": False,
            "tau_predictor_retrained": False,
            "models_edited": False,
            "inputs": "formal_calibration_json_and_committed_interval_manifest_only",
            "forbidden_read_flags": {
                flag: False for flag in forbidden_flags
            },
        },
        "source_artifacts": {
            "calibration_path": str(args.calibration.resolve()),
            "calibration_sha256": sha256_file(args.calibration),
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "checkpoint_sha256": calibration["checkpoint_sha256_before_calibration"],
        },
        "decision": {
            "calibration_status": calibration["status"],
            "no_go_is_real": True,
            "single_global_tau_bias_is_supported": False,
            "reason": (
                "Among loss-eligible structural failures, four samples require a "
                "higher tau to remove cross-target pairs while eight require a "
                "lower tau to recover M2 pairs. Two cross samples are scalar-"
                "unreachable and therefore require pair-specific correction."
            ),
        },
        "event_counts": {
            name: {
                "sample_count": len(values),
                "event_count": (
                    sum(
                        int(
                            evaluations[index]["structural_evaluation"][
                                {
                                    "cross": "selected_cross_target_pairs",
                                    "p2_violation": "parallel_2_violated_group_count",
                                    "support_loss": "ordinary_support_lost_target_count",
                                }[name]
                            ]
                        )
                        for index in values
                    )
                    if name in {"cross", "p2_violation", "support_loss"}
                    else len(values)
                ),
                "samples": indices_to_labels(values),
            }
            for name, values in event_sets.items()
        },
        "event_intersections": intersections,
        "root_cause_summary": {
            "cross_events": {
                "per_image_scalar_displacement": sum(
                    row["root_cause"] == "per_image_scalar_displacement"
                    for row in cross_rows
                ),
                "pair_specific_local_ranking": sum(
                    row["root_cause"] == "pair_specific_local_ranking"
                    for row in cross_rows
                ),
                "candidate_absence_present": sum(
                    row["root_cause"] == "candidate_absence_present"
                    for row in cross_rows
                ),
            },
            "interval_misses": {
                "total": len(interval_rows),
                "structural_failure": len(structural_interval_failures),
                "robust_inset_only_same_oracle_state": len(robust_only_misses),
                "structural_higher_tau_required": len(higher),
                "structural_lower_tau_required": len(lower),
            },
            "scalar_unreachable_samples": len(scalar_unreachable_rows),
            "candidate_unreachable_samples": int(
                manifest["partitions"]["calibration"][
                    "candidate_unreachable_sample_count"
                ]
            ),
            "candidate_unreachable_targets": int(
                manifest["partitions"]["calibration"][
                    "candidate_unreachable_target_count"
                ]
            ),
        },
        "cross_failures": cross_rows,
        "interval_misses": interval_rows,
        "scalar_unreachable_local_ranking": scalar_unreachable_rows,
        "limitations": [
            (
                "The formal target-free manifest intentionally omits per-anchor "
                "target_line_id values. When an oracle transition removes multiple "
                "candidates, the exact cross candidate cannot be named without "
                "reading a prohibited source cache."
            ),
            (
                "Candidate-unreachable samples have no oracle interval by protocol. "
                "The audit can prove candidate absence but cannot counterfactually "
                "attribute every simultaneous cross edge to tau versus local rank."
            ),
        ],
        "next_experiment_constraint": (
            "Do not tune on calibration900. Freeze this result; use a new fit-only "
            "grouped diagnostic split to test a pair-level residual or conflict-"
            "component correction on top of the global sample tau."
        ),
    }

    if report["event_counts"]["cross"]["event_count"] != 8:
        raise RuntimeError("Expected exactly eight selected cross-target pairs")
    if report["event_counts"]["p2_violation"]["event_count"] != 1:
        raise RuntimeError("Expected exactly one P2 violation")
    if report["event_counts"]["support_loss"]["event_count"] != 3:
        raise RuntimeError("Expected exactly three support losses")
    if report["event_counts"]["interval_miss"]["event_count"] != 16:
        raise RuntimeError("Expected exactly sixteen interval misses")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "diagnostic.json"
    md_path = args.output_dir / "README.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# v0.9.16 tau predictor post-hoc failure audit",
        "",
        "This is a read-only audit of the already exposed calibration900 result. "
        "No checkpoint, threshold, or model was changed.",
        "",
        "## Result",
        "",
        "The `CALIBRATION-TAU-PREDICTOR-NO-GO` decision is valid, but a single "
        "global bias correction is not supported. Of the 12 interval misses that "
        "actually change structure, four need a higher tau to remove a cross-target "
        "pair and eight need a lower tau to recover an M2 pair. Four additional "
        "interval misses already select the exact oracle candidate set and miss only "
        "the 5% robust training inset.",
        "",
        "The eight cross pairs split into four per-image scalar displacement cases, "
        "two candidate-unreachable cases, and two scalar-unreachable local ranking "
        "cases. The scalar-unreachable cases cannot be repaired by any one tau for "
        "their image and therefore require pair-specific correction.",
        "",
        "## Event overlap",
        "",
        "| Event | Samples/events | Important overlap |",
        "| --- | ---: | --- |",
        "| Cross target | 8 / 8 | 4 interval misses; 3 support losses; 1 P2 |",
        "| P2 violation | 1 / 1 | sample `000005989`, also cross + support + interval miss |",
        "| Support loss | 3 / 3 | samples `000002535`, `000005989`, `000007724` |",
        "| Robust interval miss | 16 / 16 | 12 structural; 4 same-oracle-state inset misses |",
        "",
        "## Cross root causes",
        "",
        "| Sample | Molecule / variant | Root cause | Tau | Robust adjustment | Overlap |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in cross_rows:
        identity = row["identity"]
        overlap = ", ".join(row["overlaps"]) or "none"
        md_lines.append(
            "| `{sample}` | `{molecule}` / `{variant}` | {cause} | {tau:.6f} | "
            "{adjustment} | {overlap} |".format(
                sample=identity["sample_key"],
                molecule=identity["molecule_id"],
                variant=identity["render_variant"],
                cause=row["root_cause"],
                tau=row["predicted_tau"],
                adjustment=format_adjustment(
                    row["signed_adjustment_to_robust_interval"]
                ),
                overlap=overlap,
            )
        )
    md_lines.extend(
        [
            "",
            "## Interval misses",
            "",
            "| Sample | Category | Structurally wrong | Signed robust adjustment |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in interval_rows:
        md_lines.append(
            "| `{sample}` | {category} | {wrong} | {adjustment} |".format(
                sample=row["identity"]["sample_key"],
                category=row["category"],
                wrong="yes" if not row["structurally_target_consistent"] else "no",
                adjustment=format_adjustment(
                    row["signed_adjustment_to_robust_interval"]
                ),
            )
        )
    localized = next(
        row["localized_rank_inversion"]
        for row in scalar_unreachable_rows
        if "localized_rank_inversion" in row
    )
    md_lines.extend(
        [
            "",
            "## Pair-specific evidence",
            "",
            "Sample `000007724` is a direct local rank inversion. Candidate source "
            f"`30` (`54,57`) is selected as the cross edge with raw gain "
            f"`{localized['selected_cross_candidate']['raw_gain']:.6f}`. The available "
            f"correct candidate source `0` (`0,57`) has raw gain "
            f"`{localized['available_correct_candidate']['raw_gain']:.6f}`. The cross "
            f"edge outranks it by `{localized['cross_minus_correct_raw_gain']:.6f}`. "
            "Because both edges share anchor 57, changing one image-level tau cannot "
            "reverse their order.",
            "",
            "Sample `000003944` is also scalar-unreachable. Its candidate graph has "
            "only disjoint edges, yet all 52 parametric states contain either a cross "
            "edge or another structural failure. This proves a score-ordering conflict: "
            "removing the cross with a scalar threshold necessarily removes a required "
            "lower-ranked pair as well.",
            "",
            "The full 20-D normalized descriptors and raw gains for every oracle-only "
            "or predicted-only candidate are recorded in `diagnostic.json`.",
            "",
            "## P2 and support attribution",
            "",
            "The sole P2 violation is sample `000005989`. Its only candidate beyond "
            "the oracle set is source `91`, anchor pair `(69,154)`, with raw gain "
            "`-0.561288`. The predicted tau is `-5.262639`; moving it into the robust "
            "oracle interval `[-0.519075, 0.240763]` removes that pair and restores "
            "both P2 preservation and ordinary support.",
            "",
            "The three support losses have three different causes: `000002535` has "
            "the expected candidate `(4,184)` absent; `000005989` is the scalar-too-"
            "low P2/cross case above; and `000007724` is the local rank inversion. "
            "For candidate-unreachable cross sample `000006383`, expected pair "
            "`(0,56)` is absent and selected source `0`, pair `(0,2)`, is incident to "
            "the missing target anchor, localizing the cross. In `000002535`, neither "
            "missing-target anchor participates in a selected pair, so candidate "
            "absence explains its M2/support failure but the formal manifest cannot "
            "uniquely attribute the simultaneous cross edge.",
            "",
            "## Interpretation boundary",
            "",
            "The committed target-free manifest deliberately omits per-anchor target "
            "IDs. Therefore an exact cross candidate can be named only when the oracle "
            "set difference or conflict component makes it unique. No source cache was "
            "opened to fill that gap.",
            "",
            "## Next constraint",
            "",
            "Do not reuse calibration900 for tuning. Freeze v0.9.16 and test a "
            "pair-level residual or conflict-component correction on a new fit-only, "
            "source-structure-grouped diagnostic split.",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    sums_path = args.output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{sha256_file(json_path)}  {json_path.name}\n"
        f"{sha256_file(md_path)}  {md_path.name}\n",
        encoding="utf-8",
    )
    print(json_path)
    print(md_path)
    print(sums_path)


if __name__ == "__main__":
    main()
