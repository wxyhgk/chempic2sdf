"""Evaluation and train-cal threshold selection for carrier pair matching.

This module does not train or score candidates.  It converts frozen pair
scores into relative gains over a zero-gain singleton baseline, calls the
deterministic matching core, and measures topology-preserving behavior against
the conditional ordinary-anchor labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np
import torch

from chemvector.bond_group_set import PARALLEL_2
from chemvector.evidence_carrier_pair_matching import (
    EvidenceCarrierPairCandidate,
    canonical_pair_geometry_key,
    solve_evidence_carrier_pair_matching,
)


SCHEMA_VERSION = "evidence-carrier-pair-evaluation-0.1"
SINGLETON_GAIN = 0.0
_PAIR_SCORE_KEYS = ("pair_gains", "pair_logits", "pair_scores")


class EvidenceCarrierPairEvaluationError(ValueError):
    """Raised when an evaluation target violates the single-sample protocol."""


def _numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise EvidenceCarrierPairEvaluationError(
            f"{name} must be array-like"
        ) from error
    return result


def _finite_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvidenceCarrierPairEvaluationError("threshold must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceCarrierPairEvaluationError("threshold must be finite")
    return result


def _required(target: Mapping[str, Any], name: str) -> Any:
    if name not in target:
        raise EvidenceCarrierPairEvaluationError(
            f"Pair evaluation target is missing {name!r}"
        )
    return target[name]


def _pair_scores(target: Mapping[str, Any], pair_scores: Any | None) -> np.ndarray:
    if pair_scores is None:
        present = [name for name in _PAIR_SCORE_KEYS if name in target]
        if len(present) != 1:
            raise EvidenceCarrierPairEvaluationError(
                "Provide pair_scores explicitly or exactly one of "
                f"{_PAIR_SCORE_KEYS}; found {present}"
            )
        pair_scores = target[present[0]]
    scores = _numpy(pair_scores, name="pair_scores").astype(np.float64, copy=False)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise EvidenceCarrierPairEvaluationError(
            "pair_scores must be a finite one-dimensional array"
        )
    return scores


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    anchor_segments_pixels: np.ndarray
    anchor_target_line_ids: np.ndarray
    anchor_source_indices: np.ndarray
    pair_indices: np.ndarray
    pair_scores: np.ndarray
    anchor_group_ids: np.ndarray
    anchor_group_types: np.ndarray
    group_protocol_source: str


def _integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = _numpy(value, name=name)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise EvidenceCarrierPairEvaluationError(
            f"{name} must be a one-dimensional integer array"
        )
    return array.astype(np.int64, copy=False)


def _prepare_sample(
    target: Mapping[str, Any],
    pair_scores: Any | None,
) -> _PreparedSample:
    if not isinstance(target, Mapping):
        raise EvidenceCarrierPairEvaluationError("target must be a mapping")
    segments = _numpy(
        _required(target, "anchor_segments_pixels"),
        name="anchor_segments_pixels",
    ).astype(np.float64, copy=False)
    if segments.size == 0:
        segments = np.empty((0, 2, 2), dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise EvidenceCarrierPairEvaluationError(
            "anchor_segments_pixels must have shape [anchors,2,2]"
        )
    if not np.isfinite(segments).all():
        raise EvidenceCarrierPairEvaluationError(
            "anchor_segments_pixels must contain finite coordinates"
        )

    target_ids = _integer_vector(
        _required(target, "anchor_target_line_ids"),
        name="anchor_target_line_ids",
    )
    source_indices = _integer_vector(
        _required(target, "anchor_source_indices"),
        name="anchor_source_indices",
    )
    if len(target_ids) != len(segments) or len(source_indices) != len(segments):
        raise EvidenceCarrierPairEvaluationError(
            "Anchor geometry, target IDs, and source indices must have equal length"
        )
    if (target_ids < 0).any() or (source_indices < 0).any():
        raise EvidenceCarrierPairEvaluationError(
            "Active anchor target IDs and source indices must be nonnegative"
        )
    if len(set(source_indices.tolist())) != len(source_indices):
        raise EvidenceCarrierPairEvaluationError(
            "anchor_source_indices must be unique"
        )

    pairs = _numpy(_required(target, "pair_indices"), name="pair_indices")
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(
        pairs.dtype, np.integer
    ):
        raise EvidenceCarrierPairEvaluationError(
            "pair_indices must have shape [pairs,2] and integer dtype"
        )
    pairs = pairs.astype(np.int64, copy=False)
    if len(pairs) and (
        (pairs < 0).any()
        or (pairs >= len(segments)).any()
        or (pairs[:, 0] >= pairs[:, 1]).any()
    ):
        raise EvidenceCarrierPairEvaluationError(
            "pair_indices must contain ascending active anchor indices"
        )
    pair_keys = [tuple(map(int, pair)) for pair in pairs.tolist()]
    if len(set(pair_keys)) != len(pair_keys):
        raise EvidenceCarrierPairEvaluationError("pair_indices contains duplicates")
    scores = _pair_scores(target, pair_scores)
    if len(scores) != len(pairs):
        raise EvidenceCarrierPairEvaluationError(
            "pair_scores length must equal pair_indices length"
        )
    if "pair_labels" in target:
        declared_labels = _integer_vector(
            target["pair_labels"], name="pair_labels"
        )
        if len(declared_labels) != len(pairs) or (
            (declared_labels < 0).any() or (declared_labels > 1).any()
        ):
            raise EvidenceCarrierPairEvaluationError(
                "pair_labels must contain one binary label per pair"
            )
        derived_labels = (
            target_ids[pairs[:, 0]] == target_ids[pairs[:, 1]]
        ).astype(np.int64, copy=False)
        if not np.array_equal(declared_labels, derived_labels):
            raise EvidenceCarrierPairEvaluationError(
                "pair_labels disagree with anchor target identities"
            )

    raw_names = {
        "raw_keep_labels",
        "raw_target_line_ids",
        "raw_group_ids",
        "raw_group_types",
    }
    present_raw = raw_names.intersection(target)
    if present_raw and present_raw != raw_names:
        raise EvidenceCarrierPairEvaluationError(
            "Pair evaluation requires either all raw keep/target/group fields "
            "or none of them"
        )
    if present_raw:
        raw_keep = _integer_vector(
            target["raw_keep_labels"], name="raw_keep_labels"
        )
        raw_target = _integer_vector(
            target["raw_target_line_ids"], name="raw_target_line_ids"
        )
        raw_group = _integer_vector(target["raw_group_ids"], name="raw_group_ids")
        raw_group_type = _integer_vector(
            target["raw_group_types"], name="raw_group_types"
        )
        raw_count = len(raw_keep)
        if not all(
            len(value) == raw_count
            for value in (raw_target, raw_group, raw_group_type)
        ):
            raise EvidenceCarrierPairEvaluationError(
                "Raw keep/target/group fields must have equal length"
            )
        if len(source_indices) and int(source_indices.max()) >= raw_count:
            raise EvidenceCarrierPairEvaluationError(
                "anchor_source_indices exceed raw carrier fields"
            )
        if len(source_indices) and not np.array_equal(
            raw_target[source_indices], target_ids
        ):
            raise EvidenceCarrierPairEvaluationError(
                "Anchor target IDs disagree with raw source target IDs"
            )
        if len(source_indices) and (
            (raw_group[source_indices] < 0).any()
            or (raw_keep[source_indices] != 1).any()
        ):
            raise EvidenceCarrierPairEvaluationError(
                "Active anchors must map to positive raw ordinary carriers"
            )
        anchor_group_ids = raw_group[source_indices]
        anchor_group_types = raw_group_type[source_indices]
        if "anchor_group_ids" in target or "anchor_group_types" in target:
            if not {"anchor_group_ids", "anchor_group_types"}.issubset(target):
                raise EvidenceCarrierPairEvaluationError(
                    "anchor_group_ids and anchor_group_types must appear together"
                )
            declared_group_ids = _integer_vector(
                target["anchor_group_ids"], name="anchor_group_ids"
            )
            declared_group_types = _integer_vector(
                target["anchor_group_types"], name="anchor_group_types"
            )
            if not np.array_equal(declared_group_ids, anchor_group_ids) or not np.array_equal(
                declared_group_types, anchor_group_types
            ):
                raise EvidenceCarrierPairEvaluationError(
                    "Anchor group metadata disagrees with full raw protocol"
                )
        protocol_source = "raw"
    else:
        anchor_group_ids = _integer_vector(
            _required(target, "anchor_group_ids"), name="anchor_group_ids"
        )
        anchor_group_types = _integer_vector(
            _required(target, "anchor_group_types"), name="anchor_group_types"
        )
        if len(anchor_group_ids) != len(segments) or len(anchor_group_types) != len(
            segments
        ):
            raise EvidenceCarrierPairEvaluationError(
                "Anchor group IDs/types must have one value per active anchor"
            )
        if (anchor_group_ids < 0).any():
            raise EvidenceCarrierPairEvaluationError(
                "Active anchor group IDs must be nonnegative"
            )
        protocol_source = "anchor_cache"
    return _PreparedSample(
        anchor_segments_pixels=segments,
        anchor_target_line_ids=target_ids,
        anchor_source_indices=source_indices,
        pair_indices=pairs,
        pair_scores=scores,
        anchor_group_ids=anchor_group_ids.astype(np.int64, copy=False),
        anchor_group_types=anchor_group_types.astype(np.int64, copy=False),
        group_protocol_source=protocol_source,
    )


def _group_protocol(
    sample: _PreparedSample,
) -> tuple[dict[int, tuple[int, int]], dict[int, frozenset[int]]]:
    target_protocol: dict[int, tuple[int, int]] = {}
    group_targets: dict[int, set[int]] = defaultdict(set)
    group_types: dict[int, int] = {}
    for target_id, group_id, group_type in zip(
        sample.anchor_target_line_ids.tolist(),
        sample.anchor_group_ids.tolist(),
        sample.anchor_group_types.tolist(),
    ):
        if group_id < 0:
            raise EvidenceCarrierPairEvaluationError(
                "Positive anchor target IDs require nonnegative group IDs"
            )
        protocol = (int(group_id), int(group_type))
        previous = target_protocol.setdefault(int(target_id), protocol)
        if previous != protocol:
            raise EvidenceCarrierPairEvaluationError(
                f"Target line {target_id} has inconsistent raw group metadata"
            )
        previous_type = group_types.setdefault(int(group_id), int(group_type))
        if previous_type != int(group_type):
            raise EvidenceCarrierPairEvaluationError(
                f"Raw group {group_id} has inconsistent group types"
            )
        group_targets[int(group_id)].add(int(target_id))

    for target_id in set(sample.anchor_target_line_ids.tolist()):
        if int(target_id) not in target_protocol:
            raise EvidenceCarrierPairEvaluationError(
                f"Active target line {target_id} lacks raw group metadata"
            )
    parallel_two = {}
    for group_id, targets in group_targets.items():
        if group_types[group_id] != PARALLEL_2:
            continue
        if len(targets) > 2:
            raise EvidenceCarrierPairEvaluationError(
                f"parallel_2 raw group {group_id} exposes more than two target lines"
            )
        if len(targets) == 2:
            parallel_two[group_id] = frozenset(targets)
    return target_protocol, parallel_two


def _matching_candidates(
    sample: _PreparedSample,
    threshold: float,
) -> tuple[EvidenceCarrierPairCandidate, ...]:
    candidates = []
    for pair, score in zip(sample.pair_indices.tolist(), sample.pair_scores.tolist()):
        first, second = map(int, pair)
        candidates.append(
            EvidenceCarrierPairCandidate(
                anchor_ids=(first, second),
                relative_gain=float(score) - threshold,
                canonical_geometry_key=canonical_pair_geometry_key(
                    sample.anchor_segments_pixels[first],
                    sample.anchor_segments_pixels[second],
                ),
            )
        )
    return tuple(candidates)


def evaluate_evidence_carrier_pair_sample(
    target: Mapping[str, Any],
    pair_scores: Any | None = None,
    *,
    threshold: float = 0.0,
    verify_permutation: bool = True,
) -> dict[str, Any]:
    """Match and evaluate one unpadded conditional ordinary pair sample."""

    threshold = _finite_threshold(threshold)
    if not isinstance(verify_permutation, bool):
        raise EvidenceCarrierPairEvaluationError(
            "verify_permutation must be a boolean"
        )
    sample = _prepare_sample(target, pair_scores)
    target_protocol, parallel_two_groups = _group_protocol(sample)
    candidates = _matching_candidates(sample, threshold)
    active = tuple(range(len(sample.anchor_segments_pixels)))
    matching = solve_evidence_carrier_pair_matching(active, candidates)
    selected_keys = tuple(
        candidate.anchor_ids for candidate in matching.selected_pairs
    )
    candidate_index_by_key = {
        tuple(map(int, pair)): index
        for index, pair in enumerate(sample.pair_indices.tolist())
    }
    selected_candidate_indices = tuple(
        candidate_index_by_key[key] for key in selected_keys
    )

    permutation_deterministic: bool | None = None
    if verify_permutation:
        permuted = solve_evidence_carrier_pair_matching(
            tuple(reversed(active)),
            tuple(reversed(candidates)),
        )
        permutation_deterministic = (
            tuple(candidate.anchor_ids for candidate in permuted.selected_pairs)
            == selected_keys
            and permuted.selected_singletons == matching.selected_singletons
            and math.isclose(
                permuted.total_gain,
                matching.total_gain,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    anchor_targets = sample.anchor_target_line_ids
    selected_same = []
    selected_cross = []
    violated_parallel_two_groups: set[int] = set()
    represented_targets = {
        int(anchor_targets[anchor_id])
        for anchor_id in matching.selected_singletons
    }
    for key in selected_keys:
        first_target = int(anchor_targets[key[0]])
        second_target = int(anchor_targets[key[1]])
        if first_target == second_target:
            selected_same.append(key)
            represented_targets.add(first_target)
            continue
        selected_cross.append(key)
        first_protocol = target_protocol[first_target]
        second_protocol = target_protocol[second_target]
        if (
            first_protocol == second_protocol
            and first_protocol[1] == PARALLEL_2
            and first_protocol[0] in parallel_two_groups
        ):
            violated_parallel_two_groups.add(first_protocol[0])

    selected_pair_count = len(selected_keys)
    selected_same_count = len(selected_same)
    selected_cross_count = len(selected_cross)
    precision = (
        selected_same_count / selected_pair_count if selected_pair_count else None
    )

    anchors_by_target: dict[int, list[int]] = defaultdict(list)
    for anchor_id, target_id in enumerate(anchor_targets.tolist()):
        anchors_by_target[int(target_id)].append(anchor_id)
    multiplicity_two_pairs = {
        tuple(indices): target_id
        for target_id, indices in anchors_by_target.items()
        if len(indices) == 2
    }
    pair_key_set = set(candidate_index_by_key)
    selected_key_set = set(selected_keys)
    multiplicity_two_matched = sum(
        expected in selected_key_set for expected in multiplicity_two_pairs
    )
    multiplicity_two_count = len(multiplicity_two_pairs)
    multiplicity_two_recall = (
        multiplicity_two_matched / multiplicity_two_count
        if multiplicity_two_count
        else None
    )
    pairable_multiplicity_two = {
        expected for expected in multiplicity_two_pairs if expected in pair_key_set
    }
    multi_anchor_counts = [
        len(indices) for indices in anchors_by_target.values() if len(indices) >= 2
    ]
    pairable_sample_eligible = bool(multiplicity_two_pairs) and (
        all(count == 2 for count in multi_anchor_counts)
        and len(pairable_multiplicity_two) == multiplicity_two_count
    )
    pairable_sample_exact = (
        selected_key_set == set(multiplicity_two_pairs)
        if pairable_sample_eligible
        else None
    )

    ordinary_targets = set(map(int, anchor_targets.tolist()))
    lost_targets = ordinary_targets - represented_targets
    ordinary_support_loss_pp = (
        100.0 * len(lost_targets) / len(ordinary_targets)
        if ordinary_targets
        else None
    )
    p2_count = len(parallel_two_groups)
    p2_violated = len(violated_parallel_two_groups)
    p2_preservation = (
        (p2_count - p2_violated) / p2_count if p2_count else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "threshold": threshold,
        "singleton_gain": SINGLETON_GAIN,
        "group_protocol_source": sample.group_protocol_source,
        "anchor_count": len(active),
        "candidate_pair_count": len(candidates),
        "selected_pair_count": selected_pair_count,
        "selected_singleton_count": len(matching.selected_singletons),
        "selected_same_target_pairs": selected_same_count,
        "selected_cross_target_pairs": selected_cross_count,
        "selected_pair_precision": precision,
        "selected_anchor_pairs": [list(key) for key in selected_keys],
        "selected_candidate_indices": list(selected_candidate_indices),
        "multiplicity_2_target_count": multiplicity_two_count,
        "multiplicity_2_matched_count": multiplicity_two_matched,
        "multiplicity_2_recall": multiplicity_two_recall,
        "pairable_multiplicity_2_target_count": len(pairable_multiplicity_two),
        "parallel_2_group_count": p2_count,
        "parallel_2_violated_group_count": p2_violated,
        "parallel_2_distinct_preservation": p2_preservation,
        "ordinary_supported_target_count_before": len(ordinary_targets),
        "ordinary_supported_target_count_after": len(represented_targets),
        "ordinary_support_lost_target_count": len(lost_targets),
        "ordinary_support_loss_pp": ordinary_support_loss_pp,
        "pairable_sample_eligible": pairable_sample_eligible,
        "pairable_sample_exact": pairable_sample_exact,
        "matching_total_gain": matching.total_gain,
        "matching_latency_ms": matching.latency_ms,
        "matching_feasible": matching.feasible,
        "matching_optimal": matching.optimal,
        "matching_solver_status": matching.solver_status,
        "permutation_checked": verify_permutation,
        "permutation_deterministic": permutation_deterministic,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_evidence_carrier_pair_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate count-weighted pair metrics and matching latency."""

    values = list(evaluations)
    selected = sum(int(item["selected_pair_count"]) for item in values)
    same = sum(int(item["selected_same_target_pairs"]) for item in values)
    cross = sum(int(item["selected_cross_target_pairs"]) for item in values)
    m2_targets = sum(int(item["multiplicity_2_target_count"]) for item in values)
    m2_matched = sum(int(item["multiplicity_2_matched_count"]) for item in values)
    p2_groups = sum(int(item["parallel_2_group_count"]) for item in values)
    p2_violated = sum(
        int(item["parallel_2_violated_group_count"]) for item in values
    )
    support_before = sum(
        int(item["ordinary_supported_target_count_before"]) for item in values
    )
    support_lost = sum(
        int(item["ordinary_support_lost_target_count"]) for item in values
    )
    pairable = [item for item in values if bool(item["pairable_sample_eligible"])]
    pairable_exact = sum(bool(item["pairable_sample_exact"]) for item in pairable)
    latencies = np.asarray(
        [float(item["matching_latency_ms"]) for item in values], dtype=np.float64
    )
    checked = [
        item for item in values if bool(item.get("permutation_checked", False))
    ]
    deterministic = (
        all(bool(item["permutation_deterministic"]) for item in checked)
        if checked
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(values),
        "anchor_count": sum(int(item["anchor_count"]) for item in values),
        "candidate_pair_count": sum(
            int(item["candidate_pair_count"]) for item in values
        ),
        "selected_pair_count": selected,
        "selected_same_target_pairs": same,
        "selected_cross_target_pairs": cross,
        "selected_pair_precision": _ratio(same, selected),
        "multiplicity_2_target_count": m2_targets,
        "multiplicity_2_matched_count": m2_matched,
        "multiplicity_2_recall": _ratio(m2_matched, m2_targets),
        "parallel_2_group_count": p2_groups,
        "parallel_2_violated_group_count": p2_violated,
        "parallel_2_distinct_preservation": (
            _ratio(p2_groups - p2_violated, p2_groups)
        ),
        "ordinary_supported_target_count_before": support_before,
        "ordinary_support_lost_target_count": support_lost,
        "ordinary_support_loss_pp": (
            100.0 * support_lost / support_before if support_before else None
        ),
        "pairable_sample_count": len(pairable),
        "pairable_sample_exact_count": pairable_exact,
        "pairable_sample_exact": _ratio(pairable_exact, len(pairable)),
        "matching_optimal_sample_count": sum(
            bool(item["matching_optimal"]) for item in values
        ),
        "matching_latency_ms": {
            "mean": float(latencies.mean()) if len(latencies) else None,
            "p50": float(np.percentile(latencies, 50)) if len(latencies) else None,
            "p95": float(np.percentile(latencies, 95)) if len(latencies) else None,
            "max": float(latencies.max()) if len(latencies) else None,
        },
        "permutation_checked_sample_count": len(checked),
        "permutation_deterministic": deterministic,
    }


def score_evidence_carrier_pair_record(
    record: Mapping[str, Any],
    pair_logits: Any,
    gain_threshold: float,
    verify_permutation: bool = False,
) -> dict[str, Any]:
    """Stable trainer-facing alias for one record and one score vector."""

    return evaluate_evidence_carrier_pair_sample(
        record,
        pair_logits,
        threshold=gain_threshold,
        verify_permutation=verify_permutation,
    )


def summarize_evidence_carrier_pair_records(
    records_or_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate scored results, or score embedded-logit records at threshold 0."""

    results = []
    for item in records_or_results:
        if "selected_pair_count" in item and item.get("schema_version") == SCHEMA_VERSION:
            results.append(dict(item))
        else:
            results.append(
                evaluate_evidence_carrier_pair_sample(
                    item,
                    threshold=0.0,
                    verify_permutation=False,
                )
            )
    return aggregate_evidence_carrier_pair_evaluations(results)


def _calibration_sample(
    item: Mapping[str, Any] | tuple[Mapping[str, Any], Any],
) -> tuple[Mapping[str, Any], Any | None]:
    if isinstance(item, tuple):
        if len(item) != 2 or not isinstance(item[0], Mapping):
            raise EvidenceCarrierPairEvaluationError(
                "Calibration tuples must be (target, pair_scores)"
            )
        return item[0], item[1]
    if not isinstance(item, Mapping):
        raise EvidenceCarrierPairEvaluationError(
            "Calibration samples must be targets or (target, pair_scores) tuples"
        )
    return item, None


def _evaluate_calibration_threshold(
    samples: Sequence[tuple[Mapping[str, Any], Any | None]],
    threshold: float,
    *,
    verify_permutation: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluations = [
        evaluate_evidence_carrier_pair_sample(
            target,
            scores,
            threshold=threshold,
            verify_permutation=verify_permutation,
        )
        for target, scores in samples
    ]
    return evaluations, aggregate_evidence_carrier_pair_evaluations(evaluations)


def _evenly_spaced_scores(values: np.ndarray, count: int) -> list[float]:
    if count <= 0 or not len(values):
        return []
    values = np.unique(values.astype(np.float64, copy=False))
    if len(values) <= count:
        return [float(value) for value in values.tolist()]
    indices = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return [float(values[index]) for index in np.unique(indices).tolist()]


def _threshold_candidates_from_source_scores(
    source_scores: np.ndarray,
) -> list[dict[str, Any]]:
    scores = np.unique(source_scores.astype(np.float64, copy=False))
    if not len(scores):
        raise EvidenceCarrierPairEvaluationError(
            "Threshold candidate construction requires at least one source score"
        )
    global_min = float(scores[0])
    lower_bound = float(np.nextafter(global_min, -math.inf))
    if not math.isfinite(lower_bound) or not lower_bound < global_min:
        raise EvidenceCarrierPairEvaluationError(
            "Cannot construct a finite threshold below the global minimum "
            f"calibration score {global_min}"
        )

    candidates_by_threshold: dict[float, dict[str, Any]] = {
        0.0: {
            "source_score": 0.0,
            "threshold": 0.0,
            "rule": "fixed_zero",
        },
        lower_bound: {
            "source_score": global_min,
            "threshold": lower_bound,
            "rule": "nextafter(global_min,-inf)",
        },
    }
    for source_score in scores.tolist():
        source_score = float(source_score)
        threshold = float(np.nextafter(source_score, math.inf))
        if not math.isfinite(threshold) or not threshold > source_score:
            raise EvidenceCarrierPairEvaluationError(
                "Cannot construct a finite threshold above calibration score "
                f"{source_score}"
            )
        candidates_by_threshold.setdefault(
            threshold,
            {
                "source_score": source_score,
                "threshold": threshold,
                "rule": "nextafter(unique_calibration_score,+inf)",
            },
        )

    for lower_score, upper_score in zip(scores[:-1], scores[1:], strict=True):
        lower_score = float(lower_score)
        upper_score = float(upper_score)
        midpoint = float(lower_score / 2.0 + upper_score / 2.0)
        if not math.isfinite(midpoint):
            raise EvidenceCarrierPairEvaluationError(
                "Cannot construct a finite midpoint between calibration scores "
                f"{lower_score} and {upper_score}"
            )
        if not lower_score < midpoint < upper_score:
            continue
        candidates_by_threshold.setdefault(
            midpoint,
            {
                "source_score": midpoint,
                "threshold": midpoint,
                "rule": "midpoint(adjacent_selected_scores)",
                "interval_lower_score": lower_score,
                "interval_upper_score": upper_score,
            },
        )
    return [
        candidates_by_threshold[threshold]
        for threshold in sorted(candidates_by_threshold)
    ]


def _threshold_source_score_protocol(
    prepared_samples: Sequence[_PreparedSample],
    *,
    max_threshold_candidates: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    all_scores = np.concatenate(
        [sample.pair_scores for sample in prepared_samples if len(sample.pair_scores)]
    ).astype(np.float64, copy=False)
    unique_scores = np.unique(all_scores)
    negative_parts = []
    for sample in prepared_samples:
        if not len(sample.pair_indices):
            continue
        labels = (
            sample.anchor_target_line_ids[sample.pair_indices[:, 0]]
            == sample.anchor_target_line_ids[sample.pair_indices[:, 1]]
        )
        if bool((~labels).any()):
            negative_parts.append(sample.pair_scores[~labels])
    negative_scores = (
        np.unique(np.concatenate(negative_parts).astype(np.float64, copy=False))
        if negative_parts
        else np.empty(0, dtype=np.float64)
    )

    mandatory: set[float] = {
        float(unique_scores[0]),
        float(unique_scores[-1]),
    }
    below_zero = unique_scores[unique_scores <= 0.0]
    above_zero = unique_scores[unique_scores >= 0.0]
    zero_neighborhood = []
    if len(below_zero):
        value = float(below_zero[-1])
        mandatory.add(value)
        zero_neighborhood.append(value)
    if len(above_zero):
        value = float(above_zero[0])
        mandatory.add(value)
        zero_neighborhood.append(value)
    max_negative_score = None
    if len(negative_scores):
        max_negative_score = float(negative_scores[-1])
        mandatory.add(max_negative_score)

    negative_quantiles: list[float] = []
    global_quantiles: list[float] = []
    if max_threshold_candidates is None:
        selected = set(map(float, unique_scores.tolist()))
    else:
        selected = set(mandatory)
        mandatory_candidates = _threshold_candidates_from_source_scores(
            np.asarray(sorted(selected), dtype=np.float64)
        )
        if len(mandatory_candidates) > max_threshold_candidates:
            raise EvidenceCarrierPairEvaluationError(
                "max_threshold_candidates is smaller than the mandatory safe "
                f"threshold candidate count {len(mandatory_candidates)}, including "
                "the global lower-bound, fixed zero and interval midpoints"
            )
        optional_source_capacity = (
            max_threshold_candidates - len(mandatory_candidates)
        ) // 2
        negative_available = np.asarray(
            [value for value in negative_scores.tolist() if float(value) not in selected],
            dtype=np.float64,
        )
        negative_budget = min(
            len(negative_available),
            (optional_source_capacity + 1) // 2,
        )
        negative_quantiles = _evenly_spaced_scores(
            negative_available,
            negative_budget,
        )
        selected.update(negative_quantiles)
        remaining_sources = optional_source_capacity - len(negative_quantiles)
        global_available = np.asarray(
            [value for value in unique_scores.tolist() if float(value) not in selected],
            dtype=np.float64,
        )
        global_quantiles = _evenly_spaced_scores(
            global_available,
            remaining_sources,
        )
        selected.update(global_quantiles)
        selected_candidates = _threshold_candidates_from_source_scores(
            np.asarray(sorted(selected), dtype=np.float64)
        )
        if len(selected_candidates) < max_threshold_candidates:
            for value in unique_scores.tolist():
                if float(value) in selected:
                    continue
                proposed = set(selected)
                proposed.add(float(value))
                proposed_candidates = _threshold_candidates_from_source_scores(
                    np.asarray(sorted(proposed), dtype=np.float64)
                )
                if len(proposed_candidates) > max_threshold_candidates:
                    continue
                selected.add(float(value))
                selected_candidates = proposed_candidates
                if len(selected_candidates) >= max_threshold_candidates:
                    break
    selected_scores = np.asarray(sorted(selected), dtype=np.float64)
    threshold_candidates = _threshold_candidates_from_source_scores(selected_scores)
    if (
        max_threshold_candidates is not None
        and len(threshold_candidates) > max_threshold_candidates
    ):
        raise RuntimeError("Threshold candidate selection exceeded its configured cap")
    lower_bound_threshold = next(
        float(candidate["threshold"])
        for candidate in threshold_candidates
        if candidate["rule"] == "nextafter(global_min,-inf)"
    )
    protocol = {
        "global_min_score": float(unique_scores[0]),
        "global_max_score": float(unique_scores[-1]),
        "lower_bound_source_score": float(unique_scores[0]),
        "lower_bound_threshold": lower_bound_threshold,
        "lower_bound_rule": "nextafter(global_min,-inf)",
        "lower_bound_threshold_mandatory": True,
        "fixed_zero_threshold": 0.0,
        "fixed_zero_threshold_mandatory": True,
        "interval_midpoint_rule": "midpoint(adjacent_selected_scores)",
        "bounded_candidate_search": True,
        "mathematically_complete_parametric_matching_search": False,
        "zero_neighborhood_scores": sorted(set(zero_neighborhood)),
        "max_negative_score": max_negative_score,
        "mandatory_source_scores": sorted(mandatory),
        "negative_quantile_source_scores": sorted(set(negative_quantiles)),
        "global_quantile_source_scores": sorted(set(global_quantiles)),
        "all_mandatory_scores_retained": mandatory.issubset(
            set(map(float, selected_scores.tolist()))
        ),
        "selected_source_score_count": len(selected_scores),
        "selected_interval_midpoint_count": sum(
            candidate["rule"] == "midpoint(adjacent_selected_scores)"
            for candidate in threshold_candidates
        ),
    }
    return selected_scores, threshold_candidates, protocol


def calibrate_evidence_carrier_pair_threshold(
    records: Sequence[Mapping[str, Any] | tuple[Mapping[str, Any], Any]],
    logits_by_sample: Sequence[Any] | None = None,
    max_threshold_candidates: int | None = None,
) -> dict[str, Any]:
    """Enumerate calibration-logit thresholds under hard topology constraints.

    This is a bounded candidate search, not a mathematically complete
    parametric-matching solve.  Candidates include score boundaries, adjacent
    selected-score interval midpoints, fixed zero and a mandatory
    nextafter(global_min, -inf) boundary.  Singleton gain remains exactly zero.
    """

    if logits_by_sample is not None:
        if len(records) != len(logits_by_sample):
            raise EvidenceCarrierPairEvaluationError(
                "records and logits_by_sample must have equal length"
            )
        samples = [
            _calibration_sample((record, logits))
            for record, logits in zip(records, logits_by_sample, strict=True)
        ]
    else:
        samples = [_calibration_sample(item) for item in records]
    if not samples:
        raise EvidenceCarrierPairEvaluationError(
            "Threshold calibration requires at least one calibration sample"
        )
    prepared_samples = [
        _prepare_sample(target, scores) for target, scores in samples
    ]
    prepared_scores = [sample.pair_scores for sample in prepared_samples]
    nonempty = [scores for scores in prepared_scores if len(scores)]
    if not nonempty:
        raise EvidenceCarrierPairEvaluationError(
            "Threshold calibration requires at least one calibration pair score"
        )
    total_unique_score_count = len(
        np.unique(np.concatenate(nonempty).astype(np.float64, copy=False))
    )
    if max_threshold_candidates is not None:
        if (
            isinstance(max_threshold_candidates, bool)
            or not isinstance(max_threshold_candidates, int)
            or max_threshold_candidates < 1
        ):
            raise EvidenceCarrierPairEvaluationError(
                "max_threshold_candidates must be a positive integer or None"
            )
    (
        unique_scores,
        threshold_candidates,
        threshold_candidate_protocol,
    ) = _threshold_source_score_protocol(
        prepared_samples,
        max_threshold_candidates=max_threshold_candidates,
    )
    threshold_rows = []
    feasible_rows = []
    for candidate in threshold_candidates:
        source_score = float(candidate["source_score"])
        threshold = float(candidate["threshold"])
        _, aggregate = _evaluate_calibration_threshold(
            samples,
            threshold,
            verify_permutation=False,
        )
        row = {
            "source_score": float(source_score),
            "threshold": threshold,
            "rule": str(candidate["rule"]),
            "selected_cross_target_pairs": aggregate[
                "selected_cross_target_pairs"
            ],
            "parallel_2_violated_group_count": aggregate[
                "parallel_2_violated_group_count"
            ],
            "multiplicity_2_matched_count": aggregate[
                "multiplicity_2_matched_count"
            ],
            "pairable_sample_exact_count": aggregate[
                "pairable_sample_exact_count"
            ],
            "ordinary_support_lost_target_count": aggregate[
                "ordinary_support_lost_target_count"
            ],
        }
        for name in ("interval_lower_score", "interval_upper_score"):
            if name in candidate:
                row[name] = float(candidate[name])
        threshold_rows.append(row)
        if (
            row["selected_cross_target_pairs"] == 0
            and row["parallel_2_violated_group_count"] == 0
        ):
            feasible_rows.append(row)
    if not feasible_rows:
        raise RuntimeError(
            "No calibration threshold satisfies cross=0 and parallel_2 preservation=1"
        )
    selected = max(
        feasible_rows,
        key=lambda row: (
            int(row["multiplicity_2_matched_count"]),
            int(row["pairable_sample_exact_count"]),
            -int(row["ordinary_support_lost_target_count"]),
            float(row["threshold"]),
        ),
    )
    final_evaluations, final_aggregate = _evaluate_calibration_threshold(
        samples,
        float(selected["threshold"]),
        verify_permutation=True,
    )
    if (
        final_aggregate["selected_cross_target_pairs"] != 0
        or final_aggregate["parallel_2_violated_group_count"] != 0
    ):
        raise RuntimeError("Selected calibration threshold violates hard constraints")
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration_sample_count": len(samples),
        "calibration_pair_score_count": sum(map(len, prepared_scores)),
        "candidate_threshold_count": len(threshold_rows),
        "total_unique_calibration_score_count": total_unique_score_count,
        "max_threshold_candidates": max_threshold_candidates,
        "threshold_candidate_protocol": threshold_candidate_protocol,
        "feasible_threshold_count": len(feasible_rows),
        "threshold": float(selected["threshold"]),
        "threshold_source_score": float(selected["source_score"]),
        "threshold_rule": str(selected["rule"]),
        "threshold_strictly_above_source_score": bool(
            float(selected["threshold"]) > float(selected["source_score"])
        ),
        "threshold_strictly_below_source_score": bool(
            float(selected["threshold"]) < float(selected["source_score"])
        ),
        "singleton_gain": SINGLETON_GAIN,
        "selection_priority": [
            "selected_cross_target_pairs == 0",
            "parallel_2_violated_group_count == 0",
            "maximize multiplicity_2_matched_count",
            "maximize pairable_sample_exact_count",
            "minimize ordinary_support_lost_target_count",
            "prefer higher threshold",
        ],
        "selected_metrics": final_aggregate,
        "sample_evaluations": final_evaluations,
        "threshold_trace": threshold_rows,
    }


__all__ = [
    "SCHEMA_VERSION",
    "SINGLETON_GAIN",
    "EvidenceCarrierPairEvaluationError",
    "aggregate_evidence_carrier_pair_evaluations",
    "calibrate_evidence_carrier_pair_threshold",
    "evaluate_evidence_carrier_pair_sample",
    "score_evidence_carrier_pair_record",
    "summarize_evidence_carrier_pair_records",
]
