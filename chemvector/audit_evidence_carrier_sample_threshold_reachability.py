#!/usr/bin/env python3
"""Train-only exact scalar-threshold reachability audit for frozen pair scores.

The audit never trains a model and never reads validation, fresh, sealed,
internal-test, or MR-TADF data.  For every frozen train fit/calibration sample
the formal path discovers the parametric matching upper hull and replays the
existing production solver at three strict witnesses inside each representable
open interval.  Samples already at the candidate-relative optimum may use a
conservative low-threshold early certificate.  Exact candidate-score and
fixed-cardinality boundaries are capped second-pass diagnostics only.

This is an oracle ceiling, not a deployable calibrator: ground truth is used
only to choose the best production-certified open-interval state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, eye as sparse_eye, vstack
import torch

import chemvector.evidence_carrier_p0_pair_scorer as p0_pair_scorer_module
import chemvector.evidence_carrier_pair_evaluation as pair_evaluation_module
import chemvector.evidence_carrier_pair_matching as pair_matching_module
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_matching import (
    canonical_pair_geometry_key,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    DEFAULT_AUGMENTATION_CACHE,
    DEFAULT_BASE_CACHE,
    joined_training_records,
    load_p0_pair_scorer_checkpoint,
    load_train_only_p0_pair_caches,
    score_p0_pair_records,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    pair_scorer_train_cal_gate,
)


SCHEMA_VERSION = "evidence-carrier-sample-threshold-reachability-audit-0.1"
DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal/best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v0915-sample-threshold-reachability/calibration.json"
)
BASE_CACHE_SHA256 = (
    "c0d82ff0c524431635d1609c33ea3e7340c2dd1b23ee279db43e86f27aec7697"
)
AUGMENTATION_CACHE_SHA256 = (
    "b7b50afcb664a7aeb0485f6a61f4bf8358e4b0544d4eaa8b02e6cf5d64a55eb1"
)
CHECKPOINT_SHA256 = (
    "67ba638de2841089612b428a8cf43b45206cd21f4da9aa7e182fb269160c7234"
)
EXPECTED_PARTITION_SAMPLE_COUNTS = {"fit": 8_100, "calibration": 900}
PARTITIONS = ("fit", "calibration")
DETAIL_POLICIES = ("failures", "all", "none")
DEFAULT_MAX_DETAILED_FRONTIERS = 128
_WORKER_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_FORBIDDEN_PATH_MARKERS = (
    "fresh",
    "validation",
    "sealed",
    "internal_test",
    "mr_tadf",
    "mrtadf",
)


class SampleThresholdReachabilityError(RuntimeError):
    """Raised when a frozen input or exact frontier contract is violated."""


@dataclass(frozen=True, slots=True)
class FixedCardinalityMatching:
    cardinality: int
    raw_score_sum: float
    selected_raw_score_sum: float
    selected_candidate_indices: tuple[int, ...]
    selected_anchor_pairs: tuple[tuple[int, int], ...]
    solver_status: str
    tie_break_solve_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cardinality": self.cardinality,
            "raw_score_sum": self.raw_score_sum,
            "selected_raw_score_sum": self.selected_raw_score_sum,
            "selected_candidate_indices": list(self.selected_candidate_indices),
            "selected_anchor_pairs": [list(pair) for pair in self.selected_anchor_pairs],
            "solver_status": self.solver_status,
            "tie_break_solve_count": self.tie_break_solve_count,
        }


@dataclass(frozen=True, slots=True)
class ParametricMatchingState:
    cardinality: int
    raw_score_sum: float
    selected_candidate_indices: tuple[int, ...]
    selected_anchor_pairs: tuple[tuple[int, int], ...]
    solved_at_threshold: float

    def line_value(self, threshold: float) -> float:
        return self.raw_score_sum - self.cardinality * float(threshold)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cardinality": self.cardinality,
            "raw_score_sum": self.raw_score_sum,
            "selected_candidate_indices": list(self.selected_candidate_indices),
            "selected_anchor_pairs": [list(pair) for pair in self.selected_anchor_pairs],
            "solved_at_threshold": self.solved_at_threshold,
        }


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    paths = {
        "audit": Path(__file__).resolve(),
        "matching": Path(pair_matching_module.__file__).resolve(),
        "evaluation": Path(pair_evaluation_module.__file__).resolve(),
        "p0_pair_scorer": Path(p0_pair_scorer_module.__file__).resolve(),
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _normalised_path_text(value: str | Path) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


def _reject_forbidden_path(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    resolved = raw.resolve(strict=False)
    for candidate in (raw, resolved):
        normalised = _normalised_path_text(candidate)
        if any(marker in normalised for marker in _FORBIDDEN_PATH_MARKERS):
            raise ValueError(
                f"{label} cannot reference fresh/validation/sealed/internal/MR-TADF"
            )
    return resolved


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _numpy(value: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if not np.isfinite(result).all():
        raise SampleThresholdReachabilityError(f"{name} contains non-finite values")
    return result


def _sample_arrays(
    record: Mapping[str, Any],
    pair_scores: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = _numpy(
        record["anchor_segments_pixels"],
        dtype=np.float64,
        name="anchor_segments_pixels",
    )
    pairs = _numpy(record["pair_indices"], dtype=np.int64, name="pair_indices")
    scores = _numpy(pair_scores, dtype=np.float64, name="pair_scores")
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise SampleThresholdReachabilityError(
            "anchor_segments_pixels must have shape [anchors,2,2]"
        )
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise SampleThresholdReachabilityError(
            "pair_indices must have shape [pairs,2]"
        )
    if scores.shape != (len(pairs),):
        raise SampleThresholdReachabilityError(
            "pair_scores must contain one scalar per candidate"
        )
    if len(pairs) and (
        (pairs < 0).any()
        or (pairs >= len(segments)).any()
        or (pairs[:, 0] >= pairs[:, 1]).any()
    ):
        raise SampleThresholdReachabilityError("pair_indices reference invalid anchors")
    if len({tuple(map(int, pair)) for pair in pairs.tolist()}) != len(pairs):
        raise SampleThresholdReachabilityError("pair_indices contains duplicates")
    return segments, pairs, scores


def _candidate_order(
    segments: np.ndarray,
    pairs: np.ndarray,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(pairs)),
            key=lambda index: (
                canonical_pair_geometry_key(
                    segments[int(pairs[index, 0])],
                    segments[int(pairs[index, 1])],
                ),
                tuple(map(int, pairs[index])),
            ),
        )
    )


def _incidence_matrix(anchor_count: int, ordered_pairs: np.ndarray):
    rows: list[int] = []
    columns: list[int] = []
    for column, pair in enumerate(ordered_pairs.tolist()):
        rows.extend((int(pair[0]), int(pair[1])))
        columns.extend((column, column))
    return coo_array(
        (
            np.ones(len(rows), dtype=np.float64),
            (rows, columns),
        ),
        shape=(anchor_count, len(ordered_pairs)),
        dtype=np.float64,
    ).tocsr()


def _score_tolerance(score: float) -> float:
    return 64.0 * float(np.spacing(max(1.0, abs(float(score)))))


def _solve_milp(
    *,
    c: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    constraints: LinearConstraint,
):
    def solve(options: dict[str, float | int]):
        return milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Unrecognized options detected: \{'threads'\}\. "
                r"These will be passed to HiGHS verbatim\."
            ),
            category=RuntimeWarning,
        )
        result = solve({"mip_rel_gap": 0.0, "threads": 1})
    if result.status == 4 and result.x is None:
        # HiGHS has a process-global scheduler. A prior MILP call may have
        # initialized it with another thread count, making a later override
        # return status 4. Reuse the initialized scheduler without changing
        # the objective, bounds, integrality, or constraints.
        result = solve({"mip_rel_gap": 0.0})
    return result


def solve_parametric_matching_primary(
    record: Mapping[str, Any],
    pair_scores: Any,
    threshold: float,
) -> ParametricMatchingState:
    """Solve the mathematical maximum of ``sum(score - threshold)`` once.

    This primary solve discovers candidate upper-hull lines.  It deliberately
    does not claim production tie semantics; every non-degenerate hull interval
    is certified later by the existing evaluator and matching implementation.
    """

    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    segments, pairs, scores = _sample_arrays(record, pair_scores)
    if not len(pairs):
        return ParametricMatchingState(0, 0.0, (), (), float(threshold))
    order = _candidate_order(segments, pairs)
    ordered_pairs = pairs[np.asarray(order, dtype=np.int64)]
    ordered_scores = scores[np.asarray(order, dtype=np.int64)]
    gains = ordered_scores - float(threshold)
    incidence = _incidence_matrix(len(segments), ordered_pairs)
    result = _solve_milp(
        c=-gains,
        integrality=np.ones(len(order), dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            incidence,
            np.full(len(segments), -np.inf, dtype=np.float64),
            np.ones(len(segments), dtype=np.float64),
        ),
    )
    if result.status != 0 or result.x is None:
        raise SampleThresholdReachabilityError(
            f"parametric primary solve failed: {result.status} {result.message}"
        )
    selected_ordered = [
        index
        for index, value in enumerate(np.rint(result.x).astype(np.int8))
        if value == 1 and gains[index] > 0.0
    ]
    selected_original = tuple(sorted(int(order[index]) for index in selected_ordered))
    selected_pairs = tuple(
        sorted(tuple(map(int, pairs[index])) for index in selected_original)
    )
    raw_score_sum = math.fsum(float(scores[index]) for index in selected_original)
    return ParametricMatchingState(
        cardinality=len(selected_original),
        raw_score_sum=raw_score_sum,
        selected_candidate_indices=selected_original,
        selected_anchor_pairs=selected_pairs,
        solved_at_threshold=float(threshold),
    )


def maximum_candidate_matching_cardinality(
    record: Mapping[str, Any],
    pair_scores: Any,
) -> int:
    segments, pairs, _ = _sample_arrays(record, pair_scores)
    if not len(pairs):
        return 0
    incidence = _incidence_matrix(len(segments), pairs)
    result = _solve_milp(
        c=-np.ones(len(pairs), dtype=np.float64),
        integrality=np.ones(len(pairs), dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            incidence,
            np.full(len(segments), -np.inf, dtype=np.float64),
            np.ones(len(segments), dtype=np.float64),
        ),
    )
    if result.status != 0 or result.x is None:
        raise SampleThresholdReachabilityError(
            f"maximum-cardinality solve failed: {result.status} {result.message}"
        )
    return int(np.rint(result.x).astype(np.int8).sum())


def _same_parametric_line(
    left: ParametricMatchingState,
    right: ParametricMatchingState,
) -> bool:
    return left.cardinality == right.cardinality and math.isclose(
        left.raw_score_sum,
        right.raw_score_sum,
        rel_tol=0.0,
        abs_tol=max(
            _score_tolerance(left.raw_score_sum),
            _score_tolerance(right.raw_score_sum),
        ),
    )


def discover_parametric_upper_hull(
    record: Mapping[str, Any],
    pair_scores: Any,
) -> tuple[ParametricMatchingState, ...]:
    """Discover robust scalar-threshold hull lines by intersection recursion."""

    _, _, scores = _sample_arrays(record, pair_scores)
    if not len(scores):
        return (ParametricMatchingState(0, 0.0, (), (), 0.0),)
    absolute_sum = math.fsum(abs(float(value)) for value in scores.tolist())
    # For any two matchings, the raw-score-sum difference is at most 2*A.
    # At tau=-(2*A+1), every additional selected edge therefore improves the
    # objective by strictly more than any possible raw-score disadvantage.
    low_threshold = -(2.0 * absolute_sum + 1.0)
    high_threshold = _outer_open_witness(float(scores.max()), direction=1)
    low = solve_parametric_matching_primary(record, scores, low_threshold)
    high = solve_parametric_matching_primary(record, scores, high_threshold)
    if high.cardinality != 0:
        raise SampleThresholdReachabilityError(
            "parametric high-threshold endpoint is not the empty matching"
        )
    if low.cardinality < high.cardinality:
        raise SampleThresholdReachabilityError(
            "parametric endpoint cardinalities are not monotone"
        )
    maximum_cardinality = maximum_candidate_matching_cardinality(record, scores)
    if low.cardinality != maximum_cardinality:
        raise SampleThresholdReachabilityError(
            "parametric low-threshold endpoint is not maximum-cardinality"
        )
    visited: set[tuple[int, int, float, float]] = set()

    def recurse(
        left: ParametricMatchingState,
        right: ParametricMatchingState,
        depth: int,
    ) -> list[ParametricMatchingState]:
        if depth > len(scores) + 2:
            raise SampleThresholdReachabilityError(
                "parametric hull recursion exceeded the candidate bound"
            )
        if _same_parametric_line(left, right):
            return [left]
        if left.cardinality <= right.cardinality:
            return [left, right]
        crossing = (
            (left.raw_score_sum - right.raw_score_sum)
            / (left.cardinality - right.cardinality)
        )
        if not math.isfinite(crossing):
            raise SampleThresholdReachabilityError(
                "parametric hull produced a non-finite line intersection"
            )
        signature = (
            left.cardinality,
            right.cardinality,
            float(left.raw_score_sum),
            float(right.raw_score_sum),
        )
        if signature in visited:
            raise SampleThresholdReachabilityError(
                "parametric hull recursion repeated one line pair"
            )
        visited.add(signature)
        middle = solve_parametric_matching_primary(record, scores, crossing)
        common = left.line_value(crossing)
        middle_value = middle.line_value(crossing)
        tolerance = max(
            _score_tolerance(common),
            _score_tolerance(middle_value),
        )
        is_strictly_above = middle_value > common + tolerance
        is_intermediate = (
            right.cardinality < middle.cardinality < left.cardinality
        )
        if is_strictly_above and is_intermediate:
            left_part = recurse(left, middle, depth + 1)
            right_part = recurse(middle, right, depth + 1)
            return [*left_part[:-1], *right_part]
        return [left, right]

    hull = recurse(low, high, 0)
    deduplicated = []
    for state in hull:
        if deduplicated and _same_parametric_line(deduplicated[-1], state):
            continue
        deduplicated.append(state)
    if any(
        left.cardinality <= right.cardinality
        for left, right in zip(deduplicated[:-1], deduplicated[1:], strict=True)
    ):
        raise SampleThresholdReachabilityError(
            "parametric hull cardinalities are not strictly decreasing"
        )
    return tuple(deduplicated)


def perfect_max_cardinality_open_certificate(
    record: Mapping[str, Any],
    pair_scores: Any,
) -> dict[str, Any] | None:
    """Return a strict low-interval oracle certificate when already perfect."""

    _, pairs, scores = _sample_arrays(record, pair_scores)
    if not len(pairs):
        return None
    absolute_sum = math.fsum(abs(float(value)) for value in scores.tolist())
    low_threshold = -(2.0 * absolute_sum + 1.0)
    mathematical = solve_parametric_matching_primary(
        record,
        scores,
        low_threshold,
    )
    maximum_cardinality = maximum_candidate_matching_cardinality(record, scores)
    if mathematical.cardinality != maximum_cardinality:
        raise SampleThresholdReachabilityError(
            "perfect-certificate low endpoint is not maximum-cardinality"
        )
    # The whole conservative interval lies strictly below ``low_threshold``,
    # where every extra edge dominates every possible raw-score disadvantage.
    # Its three probes are exact 25/50/75 percent interior witnesses.
    lower_threshold = 4.0 * low_threshold
    upper_threshold = low_threshold
    interval_width = upper_threshold - lower_threshold
    thresholds = [
        lower_threshold + interval_width * fraction
        for fraction in (0.25, 0.5, 0.75)
    ]
    replays = [
        evaluate_evidence_carrier_pair_sample(
            record,
            scores,
            threshold=threshold,
            verify_permutation=False,
        )
        for threshold in thresholds
    ]
    if any(not evaluation["matching_optimal"] for evaluation in replays):
        raise SampleThresholdReachabilityError(
            "perfect-certificate production replay was not optimal"
        )
    states = {_state_key(evaluation) for evaluation in replays}
    if len(states) != 1:
        return None
    evaluation = replays[-1]
    precision = evaluation.get("selected_pair_precision")
    pairable_exact = evaluation.get("pairable_sample_exact")
    perfect = bool(
        evaluation["matching_optimal"]
        and int(evaluation["selected_cross_target_pairs"]) == 0
        and int(evaluation["parallel_2_violated_group_count"]) == 0
        and int(evaluation["ordinary_support_lost_target_count"]) == 0
        and precision == 1.0
        and int(evaluation["multiplicity_2_matched_count"])
        == int(evaluation["pairable_multiplicity_2_target_count"])
        and (
            not bool(evaluation["pairable_sample_eligible"])
            or pairable_exact is True
        )
    )
    if not perfect:
        return None
    verification_threshold = thresholds[-1]
    verified = evaluate_evidence_carrier_pair_sample(
        record,
        scores,
        threshold=verification_threshold,
        verify_permutation=True,
    )
    if _state_key(verified) != next(iter(states)):
        raise SampleThresholdReachabilityError(
            "perfect-certificate permutation replay changed the selected state"
        )
    if verified.get("permutation_deterministic") is not True:
        raise SampleThresholdReachabilityError(
            "perfect-certificate permutation replay was not deterministic"
        )
    state_key = _state_key(verified)
    return {
        "mathematical_state": mathematical,
        "thresholds": thresholds,
        "oracle_threshold": verification_threshold,
        "selected_anchor_pairs": state_key,
        "evaluation": verified,
        "interval_certification": {
            "expected_hull_cardinality": mathematical.cardinality,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "interval_width": interval_width,
            "minimum_boundary_margin": min(
                thresholds[0] - lower_threshold,
                upper_threshold - thresholds[-1],
            ),
            "thresholds": thresholds,
            "representable_quartiles": True,
            "production_states": [
                {
                    "threshold": threshold,
                    "selected_anchor_pairs": [
                        list(pair) for pair in _state_key(current)
                    ],
                }
                for threshold, current in zip(thresholds, replays, strict=True)
            ],
            "production_state_consistent": True,
            "production_certified_open_interval": True,
            "conservative_low_threshold_subinterval": True,
            "upper_boundary_not_computed_due_early_certificate": True,
            "early_certificate_not_training_interval": True,
        },
    }


def parametric_open_interval_witnesses(
    hull: Sequence[ParametricMatchingState],
) -> tuple[dict[str, Any], ...]:
    if not hull:
        raise ValueError("parametric hull cannot be empty")
    if len(hull) == 1:
        return (
            {
                "expected_hull_cardinality": hull[0].cardinality,
                "lower_threshold": None,
                "upper_threshold": None,
                "interval_width": None,
                "minimum_boundary_margin": None,
                "thresholds": [-1.0, 0.0, 1.0],
                "representable_quartiles": True,
            },
        )
    boundaries = []
    for left, right in zip(hull[:-1], hull[1:], strict=True):
        boundary = (
            (left.raw_score_sum - right.raw_score_sum)
            / (left.cardinality - right.cardinality)
        )
        if not math.isfinite(boundary):
            raise SampleThresholdReachabilityError(
                "parametric hull boundary is not finite"
            )
        boundaries.append(float(boundary))
    witnesses = []
    for index, state in enumerate(hull):
        lower = boundaries[index - 1] if index else -math.inf
        upper = boundaries[index] if index < len(boundaries) else math.inf
        if lower >= upper:
            continue
        if math.isinf(lower) and math.isinf(upper):
            thresholds = [-1.0, 0.0, 1.0]
            width = None
            margin = None
        elif math.isinf(lower):
            scale = max(1.0, abs(float(upper)))
            thresholds = [
                float(upper - 3.0 * scale),
                float(upper - 2.0 * scale),
                float(upper - scale),
            ]
            width = None
            margin = float(scale)
        elif math.isinf(upper):
            scale = max(1.0, abs(float(lower)))
            thresholds = [
                float(lower + scale),
                float(lower + 2.0 * scale),
                float(lower + 3.0 * scale),
            ]
            width = None
            margin = float(scale)
        else:
            width = float(upper - lower)
            thresholds = [
                float(lower + width * fraction) for fraction in (0.25, 0.5, 0.75)
            ]
            if (
                len(set(thresholds)) != 3
                or any(not lower < value < upper for value in thresholds)
            ):
                witnesses.append(
                    {
                        "expected_hull_cardinality": state.cardinality,
                        "lower_threshold": float(lower),
                        "upper_threshold": float(upper),
                        "interval_width": width,
                        "minimum_boundary_margin": None,
                        "thresholds": [],
                        "representable_quartiles": False,
                    }
                )
                continue
            margin = min(
                thresholds[0] - lower,
                upper - thresholds[-1],
            )
        witnesses.append(
            {
                "expected_hull_cardinality": state.cardinality,
                "lower_threshold": _bound_json(lower),
                "upper_threshold": _bound_json(upper),
                "interval_width": width,
                "minimum_boundary_margin": margin,
                "thresholds": thresholds,
                "representable_quartiles": True,
            }
        )
    return tuple(witnesses)


def replay_parametric_open_intervals(
    record: Mapping[str, Any],
    pair_scores: Any,
    hull: Sequence[ParametricMatchingState],
) -> dict[str, Any]:
    _, pairs, scores = _sample_arrays(record, pair_scores)
    states: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    certifications = []
    candidate_by_pair = {
        tuple(map(int, pair)): index for index, pair in enumerate(pairs.tolist())
    }
    for interval in parametric_open_interval_witnesses(hull):
        expected_line = next(
            item
            for item in hull
            if item.cardinality == int(interval["expected_hull_cardinality"])
        )
        interval_replays = []
        for threshold in interval["thresholds"]:
            threshold = float(threshold)
            evaluation = evaluate_evidence_carrier_pair_sample(
                record,
                scores,
                threshold=threshold,
                verify_permutation=False,
            )
            if not evaluation["matching_optimal"]:
                raise SampleThresholdReachabilityError(
                    "production open-interval replay was not optimal"
                )
            state_key = _state_key(evaluation)
            raw_sum = math.fsum(
                float(scores[candidate_by_pair[pair]]) for pair in state_key
            )
            production_value = raw_sum - len(state_key) * threshold
            expected_value = expected_line.line_value(threshold)
            tolerance = max(
                _score_tolerance(production_value),
                _score_tolerance(expected_value),
            )
            if production_value < expected_value - tolerance:
                raise SampleThresholdReachabilityError(
                    "production open-interval replay lost the mathematical hull objective"
                )
            interval_replays.append(
                {
                    "threshold": threshold,
                    "state_key": state_key,
                    "raw_score_sum": raw_sum,
                    "evaluation": evaluation,
                }
            )
        distinct_states = {item["state_key"] for item in interval_replays}
        certified = bool(interval["representable_quartiles"]) and len(distinct_states) == 1
        certifications.append(
            {
                **interval,
                "production_states": [
                    {
                        "threshold": item["threshold"],
                        "selected_anchor_pairs": [
                            list(pair) for pair in item["state_key"]
                        ],
                    }
                    for item in interval_replays
                ],
                "production_state_consistent": len(distinct_states) == 1,
                "production_certified_open_interval": certified,
            }
        )
        if not certified:
            continue
        representative = interval_replays[1]
        state_key = representative["state_key"]
        evaluation = representative["evaluation"]
        raw_sum = representative["raw_score_sum"]
        current = states.get(state_key)
        if current is None:
            current = {
                "selected_anchor_pairs": [list(pair) for pair in state_key],
                "selected_candidate_indices": list(
                    evaluation["selected_candidate_indices"]
                ),
                "cardinality": len(state_key),
                "raw_score_sum": raw_sum,
                "production_certified_open_interval": True,
                "boundary_only": False,
                "witnesses": [],
                "evaluation": {
                    name: value
                    for name, value in evaluation.items()
                    if name not in {"matching_latency_ms", "threshold"}
                },
            }
            states[state_key] = current
        current["witnesses"].append(
            {
                **interval,
                "robust_open_interval": True,
                "production_cardinality": len(state_key),
            }
        )
    return {
        "states": tuple(
            sorted(
                states.values(),
                key=lambda item: (
                    item["cardinality"],
                    item["selected_anchor_pairs"],
                ),
            )
        ),
        "interval_certifications": tuple(certifications),
    }


def solve_fixed_cardinality_matching(
    record: Mapping[str, Any],
    pair_scores: Any,
    cardinality: int,
) -> FixedCardinalityMatching | None:
    """Return the canonical maximum raw-score matching with exactly ``k`` pairs."""

    segments, pairs, scores = _sample_arrays(record, pair_scores)
    if (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, int)
        or cardinality < 0
    ):
        raise ValueError("cardinality must be a non-negative integer")
    if cardinality == 0:
        return FixedCardinalityMatching(
            cardinality=0,
            raw_score_sum=0.0,
            selected_raw_score_sum=0.0,
            selected_candidate_indices=(),
            selected_anchor_pairs=(),
            solver_status="optimal",
            tie_break_solve_count=0,
        )
    if cardinality > len(segments) // 2 or cardinality > len(pairs):
        return None

    order = _candidate_order(segments, pairs)
    ordered_pairs = pairs[np.asarray(order, dtype=np.int64)]
    ordered_scores = scores[np.asarray(order, dtype=np.int64)]
    candidate_count = len(order)
    incidence = _incidence_matrix(len(segments), ordered_pairs)
    cardinality_row = coo_array(
        np.ones((1, candidate_count), dtype=np.float64)
    ).tocsr()
    base_matrix = vstack((incidence, cardinality_row), format="csr")
    lower = np.concatenate(
        (
            np.full(len(segments), -np.inf, dtype=np.float64),
            np.asarray([float(cardinality)], dtype=np.float64),
        )
    )
    upper = np.concatenate(
        (
            np.ones(len(segments), dtype=np.float64),
            np.asarray([float(cardinality)], dtype=np.float64),
        )
    )
    integrality = np.ones(candidate_count, dtype=np.uint8)
    bounds = Bounds(0.0, 1.0)
    primary = _solve_milp(
        c=-ordered_scores,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(base_matrix, lower, upper),
    )
    if primary.status == 2:
        return None
    if primary.status != 0 or primary.x is None:
        raise SampleThresholdReachabilityError(
            f"fixed-cardinality primary solve failed: {primary.status} {primary.message}"
        )
    primary_selection = np.rint(primary.x).astype(np.int8)
    optimum = float(np.dot(ordered_scores, primary_selection))
    tolerance = _score_tolerance(optimum)

    no_good = np.where(primary_selection == 1, -1.0, 1.0)
    no_good_lower = 1.0 - float(primary_selection.sum())
    alternative_matrix = vstack(
        (base_matrix, coo_array(no_good.reshape(1, -1))),
        format="csr",
    )
    alternative = _solve_milp(
        c=-ordered_scores,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            alternative_matrix,
            np.concatenate((lower, np.asarray([no_good_lower]))),
            np.concatenate((upper, np.asarray([np.inf]))),
        ),
    )
    unique = alternative.status == 2
    if alternative.status == 0 and alternative.x is not None:
        alternative_selection = np.rint(alternative.x).astype(np.int8)
        alternative_score = float(np.dot(ordered_scores, alternative_selection))
        unique = alternative_score < optimum - tolerance
    elif alternative.status not in (0, 2):
        raise SampleThresholdReachabilityError(
            "fixed-cardinality uniqueness solve failed: "
            f"{alternative.status} {alternative.message}"
        )

    tie_break_solve_count = 1
    if unique:
        fixed = primary_selection
        solver_status = "optimal_unique"
    else:
        score_row = coo_array(ordered_scores.reshape(1, -1)).tocsr()
        identity = sparse_eye(candidate_count, dtype=np.float64, format="csr")
        tie_matrix = vstack(
            (incidence, cardinality_row, score_row, identity),
            format="csr",
        )
        row_count = tie_matrix.shape[0]
        tie_lower = np.full(row_count, -np.inf, dtype=np.float64)
        tie_upper = np.full(row_count, np.inf, dtype=np.float64)
        tie_upper[: len(segments)] = 1.0
        cardinality_row_index = len(segments)
        tie_lower[cardinality_row_index] = float(cardinality)
        tie_upper[cardinality_row_index] = float(cardinality)
        score_row_index = cardinality_row_index + 1
        tie_lower[score_row_index] = optimum - tolerance
        tie_upper[score_row_index] = optimum + tolerance
        identity_offset = score_row_index + 1
        fixed = np.full(candidate_count, -1, dtype=np.int8)
        for candidate_index, pair in enumerate(ordered_pairs.tolist()):
            if fixed[candidate_index] >= 0:
                continue
            trial_lower = tie_lower.copy()
            trial_upper = tie_upper.copy()
            trial_lower[identity_offset + candidate_index] = 1.0
            trial_upper[identity_offset + candidate_index] = 1.0
            trial = _solve_milp(
                c=np.zeros(candidate_count, dtype=np.float64),
                integrality=integrality,
                bounds=bounds,
                constraints=LinearConstraint(
                    tie_matrix,
                    trial_lower,
                    trial_upper,
                ),
            )
            tie_break_solve_count += 1
            if trial.status == 0 and trial.x is not None:
                fixed[candidate_index] = 1
                tie_lower[identity_offset + candidate_index] = 1.0
                tie_upper[identity_offset + candidate_index] = 1.0
                selected_anchors = {int(pair[0]), int(pair[1])}
                for later in range(candidate_index + 1, candidate_count):
                    if fixed[later] >= 0:
                        continue
                    if selected_anchors.intersection(map(int, ordered_pairs[later])):
                        fixed[later] = 0
                        tie_lower[identity_offset + later] = 0.0
                        tie_upper[identity_offset + later] = 0.0
            elif trial.status == 2:
                fixed[candidate_index] = 0
                tie_lower[identity_offset + candidate_index] = 0.0
                tie_upper[identity_offset + candidate_index] = 0.0
            else:
                raise SampleThresholdReachabilityError(
                    "fixed-cardinality tie-break failed: "
                    f"{trial.status} {trial.message}"
                )
        solver_status = "optimal_canonical_tie_break"

    if int(fixed.sum()) != cardinality:
        raise SampleThresholdReachabilityError(
            "fixed-cardinality result has the wrong selected pair count"
        )
    selected_ordered = [index for index, value in enumerate(fixed) if value == 1]
    selected_original = tuple(sorted(int(order[index]) for index in selected_ordered))
    selected_pairs = tuple(
        sorted(tuple(map(int, pairs[index])) for index in selected_original)
    )
    raw_score_sum = math.fsum(float(scores[index]) for index in selected_original)
    if not math.isclose(raw_score_sum, optimum, rel_tol=0.0, abs_tol=tolerance):
        raise SampleThresholdReachabilityError(
            "canonical fixed-cardinality matching lost the primary score"
        )
    return FixedCardinalityMatching(
        cardinality=cardinality,
        raw_score_sum=optimum,
        selected_raw_score_sum=raw_score_sum,
        selected_candidate_indices=selected_original,
        selected_anchor_pairs=selected_pairs,
        solver_status=solver_status,
        tie_break_solve_count=tie_break_solve_count,
    )


def fixed_cardinality_frontier(
    record: Mapping[str, Any],
    pair_scores: Any,
) -> tuple[FixedCardinalityMatching, ...]:
    segments, pairs, _ = _sample_arrays(record, pair_scores)
    maximum = min(len(segments) // 2, len(pairs))
    result = [solve_fixed_cardinality_matching(record, pair_scores, 0)]
    for cardinality in range(1, maximum + 1):
        current = solve_fixed_cardinality_matching(
            record,
            pair_scores,
            cardinality,
        )
        if current is None:
            break
        result.append(current)
    return tuple(item for item in result if item is not None)


def _bound_json(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def cardinality_envelope(
    frontier: Sequence[FixedCardinalityMatching],
    pair_scores: Any,
) -> tuple[dict[str, Any], ...]:
    scores = _numpy(pair_scores, dtype=np.float64, name="pair_scores")
    result = []
    for current in frontier:
        lower = -math.inf
        upper = math.inf
        for other in frontier:
            if other.cardinality > current.cardinality:
                lower = max(
                    lower,
                    (other.raw_score_sum - current.raw_score_sum)
                    / (other.cardinality - current.cardinality),
                )
            elif other.cardinality < current.cardinality:
                upper = min(
                    upper,
                    (current.raw_score_sum - other.raw_score_sum)
                    / (current.cardinality - other.cardinality),
                )
        positivity_upper = math.inf
        if current.selected_candidate_indices:
            positivity_upper = min(
                float(scores[index]) for index in current.selected_candidate_indices
            )
            upper = min(upper, positivity_upper)
        tolerance = _score_tolerance(current.raw_score_sum)
        result.append(
            {
                **current.as_dict(),
                "lower_threshold": _bound_json(lower),
                "upper_threshold": _bound_json(upper),
                "positivity_upper_threshold": _bound_json(positivity_upper),
                "potentially_reachable": lower <= upper + tolerance,
                "upper_is_strict_for_selected_edge_positivity": bool(
                    current.selected_candidate_indices
                    and upper >= positivity_upper - tolerance
                ),
            }
        )
    return tuple(result)


def _strict_midpoint(lower: float, upper: float) -> float | None:
    midpoint = float(lower / 2.0 + upper / 2.0)
    if math.isfinite(midpoint) and lower < midpoint < upper:
        return midpoint
    candidate = float(np.nextafter(lower, upper))
    return candidate if math.isfinite(candidate) and candidate < upper else None


def _outer_open_witness(boundary: float, *, direction: int) -> float:
    scale = max(1.0, abs(float(boundary)))
    candidate = float(boundary) + float(direction) * scale
    if math.isfinite(candidate) and (
        (direction < 0 and candidate < boundary)
        or (direction > 0 and candidate > boundary)
    ):
        return candidate
    candidate = float(
        np.nextafter(
            boundary,
            -math.inf if direction < 0 else math.inf,
        )
    )
    if not math.isfinite(candidate) or candidate == boundary:
        raise SampleThresholdReachabilityError(
            "cannot construct a finite outer threshold witness"
        )
    return candidate


def threshold_probe_values(
    pair_scores: Any,
    envelope: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return exact boundaries, adjacent floats, and robust open witnesses.

    The mathematical cardinality envelope proposes critical values.  It does
    not by itself prove floating-point production behavior because the solver
    uses a 64-ulp objective tolerance.  Every returned value is therefore
    replayed through the production evaluator, and only probes explicitly
    marked ``open_interval`` may contribute to the scalar oracle ceiling.
    """

    scores = _numpy(pair_scores, dtype=np.float64, name="pair_scores")
    critical = {0.0, *(float(value) for value in scores.tolist())}
    for row in envelope:
        for name in (
            "lower_threshold",
            "upper_threshold",
            "positivity_upper_threshold",
        ):
            value = row.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                critical.add(float(value))
    ordered = sorted(critical)
    if not ordered:
        ordered = [0.0]
    probes: dict[float, set[str]] = {}

    def add(value: float, kind: str) -> None:
        if math.isfinite(value):
            probes.setdefault(float(value), set()).add(kind)

    for value in ordered:
        add(value, "boundary")
        add(float(np.nextafter(value, -math.inf)), "boundary_neighbor")
        add(float(np.nextafter(value, math.inf)), "boundary_neighbor")

    add(_outer_open_witness(ordered[0], direction=-1), "open_interval")
    add(_outer_open_witness(ordered[-1], direction=1), "open_interval")
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        midpoint = _strict_midpoint(lower, upper)
        if midpoint is not None:
            add(midpoint, "open_interval")
    return tuple(
        {
            "threshold": threshold,
            "kinds": sorted(kinds),
            "robust_open_interval": "open_interval" in kinds,
        }
        for threshold, kinds in sorted(probes.items())
    )


def _state_key(evaluation: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(tuple(map(int, pair)) for pair in evaluation["selected_anchor_pairs"])
    )


def replay_threshold_frontier(
    record: Mapping[str, Any],
    pair_scores: Any,
    frontier: Sequence[FixedCardinalityMatching],
    envelope: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the production matcher at the complete scalar critical set."""

    _, pairs, scores = _sample_arrays(record, pair_scores)
    by_cardinality = {item.cardinality: item for item in frontier}
    states: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    probe_rows = threshold_probe_values(scores, envelope)
    probe_replays = []
    for probe in probe_rows:
        threshold = float(probe["threshold"])
        evaluation = evaluate_evidence_carrier_pair_sample(
            record,
            scores,
            threshold=threshold,
            verify_permutation=False,
        )
        if not evaluation["matching_optimal"]:
            raise SampleThresholdReachabilityError(
                "production matching replay was not optimal"
            )
        state = _state_key(evaluation)
        cardinality = len(state)
        expected = by_cardinality.get(cardinality)
        if expected is None:
            raise SampleThresholdReachabilityError(
                "production replay selected an infeasible cardinality"
            )
        candidate_by_pair = {
            tuple(map(int, pair)): index for index, pair in enumerate(pairs.tolist())
        }
        raw_sum = math.fsum(float(scores[candidate_by_pair[pair]]) for pair in state)
        tolerance = _score_tolerance(expected.raw_score_sum)
        if not math.isclose(
            raw_sum,
            expected.raw_score_sum,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise SampleThresholdReachabilityError(
                "production replay is not same-cardinality raw-score optimal"
            )
        current = states.get(state)
        if current is None:
            compact_evaluation = {
                name: value
                for name, value in evaluation.items()
                if name not in {"matching_latency_ms", "threshold"}
            }
            current = {
                "selected_anchor_pairs": [list(pair) for pair in state],
                "selected_candidate_indices": list(
                    evaluation["selected_candidate_indices"]
                ),
                "cardinality": cardinality,
                "raw_score_sum": raw_sum,
                "witnesses": [],
                "evaluation": compact_evaluation,
            }
            states[state] = current
        witness = {
            "threshold": threshold,
            "kinds": list(probe["kinds"]),
            "robust_open_interval": bool(probe["robust_open_interval"]),
        }
        current["witnesses"].append(witness)
        probe_replays.append(
            {
                **witness,
                "selected_anchor_pairs": [list(pair) for pair in state],
            }
        )
    result = []
    for state in states.values():
        state["witnesses"] = sorted(
            state["witnesses"],
            key=lambda item: float(item["threshold"]),
        )
        state["production_certified_open_interval"] = any(
            witness["robust_open_interval"] for witness in state["witnesses"]
        )
        state["boundary_only"] = not state["production_certified_open_interval"]
        result.append(state)
    result = tuple(
        sorted(
            result,
            key=lambda item: (
                item["cardinality"],
                item["selected_anchor_pairs"],
            ),
        )
    )
    replay_by_threshold = {
        float(item["threshold"]): item for item in probe_replays
    }
    boundary_certifications = []
    for threshold in sorted(
        float(item["threshold"])
        for item in probe_rows
        if "boundary" in item["kinds"]
    ):
        exact = replay_by_threshold[threshold]
        left = replay_by_threshold.get(float(np.nextafter(threshold, -math.inf)))
        right = replay_by_threshold.get(float(np.nextafter(threshold, math.inf)))
        exact_state = exact["selected_anchor_pairs"]
        boundary_certifications.append(
            {
                "threshold": threshold,
                "left_nextafter_state": (
                    left["selected_anchor_pairs"] if left is not None else None
                ),
                "exact_state": exact_state,
                "right_nextafter_state": (
                    right["selected_anchor_pairs"] if right is not None else None
                ),
                "exact_matches_left_nextafter": bool(
                    left is not None and exact_state == left["selected_anchor_pairs"]
                ),
                "exact_matches_right_nextafter": bool(
                    right is not None and exact_state == right["selected_anchor_pairs"]
                ),
            }
        )
    return {
        "states": result,
        "probe_replays": tuple(probe_replays),
        "boundary_certifications": tuple(boundary_certifications),
    }


def _pairable_expected_matching(
    record: Mapping[str, Any],
) -> tuple[tuple[int, int], ...] | None:
    target_ids = _numpy(
        record["anchor_target_line_ids"],
        dtype=np.int64,
        name="anchor_target_line_ids",
    )
    pairs = _numpy(record["pair_indices"], dtype=np.int64, name="pair_indices")
    counts = Counter(map(int, target_ids.tolist()))
    multi = [count for count in counts.values() if count > 1]
    multiplicity_two = sorted(target for target, count in counts.items() if count == 2)
    if not multiplicity_two or any(count != 2 for count in multi):
        return None
    candidates = {tuple(map(int, pair)) for pair in pairs.tolist()}
    expected = []
    for target in multiplicity_two:
        anchors = tuple(
            index for index, value in enumerate(target_ids.tolist()) if int(value) == target
        )
        if len(anchors) != 2 or anchors not in candidates:
            return None
        expected.append(anchors)
    return tuple(sorted(expected))


def _oracle_state_key(state: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = state["evaluation"]
    precision = evaluation.get("selected_pair_precision")
    threshold = min(
        (
            threshold
            for witness in state["witnesses"]
            if witness["robust_open_interval"]
            for threshold in witness["thresholds"]
        ),
        key=lambda value: (abs(float(value)), float(value)),
    )
    return (
        -int(evaluation["selected_cross_target_pairs"]),
        -int(evaluation["parallel_2_violated_group_count"]),
        int(evaluation["multiplicity_2_matched_count"]),
        int(evaluation.get("pairable_sample_exact") is True),
        -int(evaluation["ordinary_support_lost_target_count"]),
        float(precision or 0.0),
        -abs(float(threshold)),
    )


def _sample_from_perfect_certificate(
    record: Mapping[str, Any],
    pair_scores: Any,
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    _, pairs, scores = _sample_arrays(record, pair_scores)
    evaluation = dict(certificate["evaluation"])
    state_key = tuple(
        tuple(map(int, pair)) for pair in certificate["selected_anchor_pairs"]
    )
    candidate_by_pair = {
        tuple(map(int, pair)): index for index, pair in enumerate(pairs.tolist())
    }
    raw_sum = math.fsum(float(scores[candidate_by_pair[pair]]) for pair in state_key)
    interval = dict(certificate["interval_certification"])
    compact_evaluation = {
        name: value
        for name, value in evaluation.items()
        if name not in {"matching_latency_ms", "threshold"}
    }
    reachable_state = {
        "selected_anchor_pairs": [list(pair) for pair in state_key],
        "selected_candidate_indices": list(evaluation["selected_candidate_indices"]),
        "cardinality": len(state_key),
        "raw_score_sum": raw_sum,
        "production_certified_open_interval": True,
        "boundary_only": False,
        "witnesses": [
            {
                **interval,
                "robust_open_interval": True,
                "production_cardinality": len(state_key),
            }
        ],
        "evaluation": compact_evaluation,
    }
    expected = _pairable_expected_matching(record)
    scalar_reachable_exact: bool | None = None
    rank_failure: bool | None = None
    if expected is not None:
        scalar_reachable_exact = evaluation.get("pairable_sample_exact") is True
        if not scalar_reachable_exact:
            raise SampleThresholdReachabilityError(
                "perfect certificate did not preserve the pairable expected matching"
            )
        rank_failure = False
    oracle_interval_names = (
        "expected_hull_cardinality",
        "lower_threshold",
        "upper_threshold",
        "interval_width",
        "minimum_boundary_margin",
        "thresholds",
        "conservative_low_threshold_subinterval",
        "upper_boundary_not_computed_due_early_certificate",
        "early_certificate_not_training_interval",
    )
    return {
        "identity": dict(record.get("identity", {})),
        "anchor_count": int(len(record["anchor_segments_pixels"])),
        "candidate_pair_count": int(len(record["pair_indices"])),
        "early_certificate": True,
        "early_certificate_not_training_interval": True,
        "critical_threshold_probe_count": len(certificate["thresholds"]),
        "reachable_state_count": 1,
        "robust_open_interval_state_count": 1,
        "boundary_only_state_count": 0,
        "parametric_hull_state_count": 1,
        "parametric_hull_fully_enumerated": False,
        "certified_interval_count": 1,
        "parametric_upper_hull": [certificate["mathematical_state"].as_dict()],
        "parametric_interval_certifications": [interval],
        "reachable_states": [reachable_state],
        "pairable_expected_matching": (
            [list(pair) for pair in expected] if expected is not None else None
        ),
        "same_cardinality_rank_failure": rank_failure,
        "scalar_reachable_pairable_exact": scalar_reachable_exact,
        "oracle_threshold": float(certificate["oracle_threshold"]),
        "oracle_interval": {
            name: interval[name] for name in oracle_interval_names if name in interval
        },
        "oracle_evaluation": evaluation,
        "detail_threshold_probe_count": 0,
    }


def audit_sample_threshold_reachability(
    record: Mapping[str, Any],
    pair_scores: Any,
    *,
    detail_policy: str = "all",
) -> dict[str, Any]:
    if detail_policy not in DETAIL_POLICIES:
        raise ValueError(f"detail_policy must be one of {DETAIL_POLICIES}")
    if detail_policy != "all":
        certificate = perfect_max_cardinality_open_certificate(record, pair_scores)
        if certificate is not None:
            early_sample = _sample_from_perfect_certificate(
                record,
                pair_scores,
                certificate,
            )
            if detail_policy == "none" or not _sample_failed_oracle_gate(early_sample):
                return early_sample
    hull = discover_parametric_upper_hull(record, pair_scores)
    parametric_replay = replay_parametric_open_intervals(record, pair_scores, hull)
    states = parametric_replay["states"]
    if not states:
        raise SampleThresholdReachabilityError(
            "parametric replay produced no production-certified open interval state"
        )
    selected = max(states, key=_oracle_state_key)
    witness = min(
        (
            threshold
            for item in selected["witnesses"]
            if item["robust_open_interval"]
            for threshold in item["thresholds"]
        ),
        key=lambda value: (abs(float(value)), float(value)),
    )
    oracle_interval = next(
        item
        for item in selected["witnesses"]
        if witness in item["thresholds"]
    )
    oracle_evaluation = evaluate_evidence_carrier_pair_sample(
        record,
        pair_scores,
        threshold=float(witness),
        verify_permutation=True,
    )
    if _state_key(oracle_evaluation) != tuple(
        tuple(map(int, pair)) for pair in selected["selected_anchor_pairs"]
    ):
        raise SampleThresholdReachabilityError(
            "oracle verification replay differs from its frontier state"
        )

    expected = _pairable_expected_matching(record)
    rank_failure: bool | None = None
    scalar_reachable_exact: bool | None = None
    if expected is not None:
        scalar_reachable_exact = any(
            state["evaluation"].get("pairable_sample_exact") is True
            for state in states
        )
        if scalar_reachable_exact:
            rank_failure = False
        else:
            same_cardinality = solve_fixed_cardinality_matching(
                record,
                pair_scores,
                len(expected),
            )
            if same_cardinality is None:
                raise SampleThresholdReachabilityError(
                    "pairable expected matching cardinality is infeasible"
                )
            rank_failure = set(same_cardinality.selected_anchor_pairs) != set(expected)
    identity = dict(record.get("identity", {}))
    sample = {
        "identity": identity,
        "anchor_count": int(len(record["anchor_segments_pixels"])),
        "candidate_pair_count": int(len(record["pair_indices"])),
        "early_certificate": False,
        "early_certificate_not_training_interval": False,
        "critical_threshold_probe_count": sum(
            len(interval["thresholds"])
            for interval in parametric_open_interval_witnesses(hull)
        ),
        "reachable_state_count": len(states),
        "robust_open_interval_state_count": len(states),
        "boundary_only_state_count": 0,
        "parametric_hull_state_count": len(hull),
        "parametric_hull_fully_enumerated": True,
        "certified_interval_count": sum(
            item["production_certified_open_interval"]
            for item in parametric_replay["interval_certifications"]
        ),
        "parametric_upper_hull": [item.as_dict() for item in hull],
        "parametric_interval_certifications": list(
            parametric_replay["interval_certifications"]
        ),
        "reachable_states": list(states),
        "pairable_expected_matching": (
            [list(pair) for pair in expected] if expected is not None else None
        ),
        "same_cardinality_rank_failure": rank_failure,
        "scalar_reachable_pairable_exact": scalar_reachable_exact,
        "oracle_threshold": float(witness),
        "oracle_interval": {
            name: oracle_interval[name]
            for name in (
                "expected_hull_cardinality",
                "lower_threshold",
                "upper_threshold",
                "interval_width",
                "minimum_boundary_margin",
                "thresholds",
            )
        },
        "oracle_evaluation": oracle_evaluation,
    }
    retain_detail = detail_policy == "all" or (
        detail_policy == "failures" and _sample_failed_oracle_gate(sample)
    )
    if retain_detail:
        frontier = fixed_cardinality_frontier(record, pair_scores)
        envelope = cardinality_envelope(frontier, pair_scores)
        replay = replay_threshold_frontier(
            record,
            pair_scores,
            frontier,
            envelope,
        )
        sample.update(
            {
                "detail_threshold_probe_count": len(
                    threshold_probe_values(pair_scores, envelope)
                ),
                "fixed_cardinality_frontier": [item.as_dict() for item in frontier],
                "cardinality_envelope": list(envelope),
                "boundary_reachable_states": list(replay["states"]),
                "boundary_certifications": list(replay["boundary_certifications"]),
            }
        )
    else:
        sample["detail_threshold_probe_count"] = 0
    return sample


def _partition_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluations = [sample["oracle_evaluation"] for sample in samples]
    aggregate = aggregate_evidence_carrier_pair_evaluations(evaluations)
    pairable = [
        sample
        for sample in samples
        if sample["pairable_expected_matching"] is not None
    ]
    return {
        "sample_count": len(samples),
        "aggregate": aggregate,
        "early_certificate_sample_count": sum(
            sample.get("early_certificate") is True for sample in samples
        ),
        "pairable_sample_count": len(pairable),
        "same_cardinality_rank_failure_count": sum(
            sample["same_cardinality_rank_failure"] is True for sample in pairable
        ),
        "scalar_unreachable_pairable_sample_count": sum(
            sample["scalar_reachable_pairable_exact"] is False for sample in pairable
        ),
        "scalar_reachable_pairable_exact_count": sum(
            sample["scalar_reachable_pairable_exact"] is True for sample in pairable
        ),
        "critical_threshold_probe_count": sum(
            int(sample["critical_threshold_probe_count"]) for sample in samples
        ),
        "reachable_state_count": sum(
            int(sample["reachable_state_count"]) for sample in samples
        ),
    }


def _scalar_oracle_capability_gate(
    pair_scorer_gate: Mapping[str, Any],
) -> dict[str, Any]:
    excluded_checks = {"matching_p95_below_20ms"}
    checks = dict(pair_scorer_gate["checks"])
    diagnostic_checks = {
        name: checks.pop(name) for name in excluded_checks if name in checks
    }
    thresholds = dict(pair_scorer_gate["thresholds"])
    diagnostic_thresholds = {}
    if "maximum_matching_p95_ms" in thresholds:
        diagnostic_thresholds["maximum_matching_p95_ms"] = thresholds.pop(
            "maximum_matching_p95_ms"
        )
    return {
        **dict(pair_scorer_gate),
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "thresholds": thresholds,
        "diagnostic_checks": diagnostic_checks,
        "diagnostic_thresholds": diagnostic_thresholds,
        "excluded_from_capability_status": sorted(excluded_checks),
        "base_pair_scorer_gate_status": pair_scorer_gate["status"],
    }


def _sample_failed_oracle_gate(sample: Mapping[str, Any]) -> bool:
    evaluation = sample["oracle_evaluation"]
    return bool(
        sample["same_cardinality_rank_failure"] is True
        or sample["scalar_reachable_pairable_exact"] is False
        or int(evaluation["selected_cross_target_pairs"]) > 0
        or int(evaluation["parallel_2_violated_group_count"]) > 0
        or int(evaluation["multiplicity_2_matched_count"])
        < int(evaluation["multiplicity_2_target_count"])
        or evaluation.get("pairable_sample_exact") is False
        or int(evaluation["ordinary_support_lost_target_count"]) > 0
        or evaluation.get("permutation_deterministic") is not True
    )


def _compact_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {
        "parametric_upper_hull",
        "parametric_interval_certifications",
        "fixed_cardinality_frontier",
        "cardinality_envelope",
        "reachable_states",
        "boundary_reachable_states",
        "boundary_certifications",
    }
    return {name: value for name, value in sample.items() if name not in omitted}


def _compact_failure_frontier(sample: Mapping[str, Any]) -> dict[str, Any]:
    states = []
    detailed_states = sample.get(
        "boundary_reachable_states",
        sample["reachable_states"],
    )
    for state in detailed_states:
        robust = [
            witness
            for witness in state["witnesses"]
            if witness["robust_open_interval"]
        ]
        boundary = [
            witness
            for witness in state["witnesses"]
            if "boundary" in witness["kinds"]
        ]
        states.append(
            {
                "selected_anchor_pairs": state["selected_anchor_pairs"],
                "selected_candidate_indices": state["selected_candidate_indices"],
                "cardinality": state["cardinality"],
                "raw_score_sum": state["raw_score_sum"],
                "production_certified_open_interval": state[
                    "production_certified_open_interval"
                ],
                "boundary_only": state["boundary_only"],
                "open_interval_witness_count": len(robust),
                "boundary_witness_count": len(boundary),
                "representative_open_threshold": (
                    min(
                        (
                            threshold
                            for item in robust
                            for threshold in item.get(
                                "thresholds",
                                [item.get("threshold")],
                            )
                            if threshold is not None
                        ),
                        key=lambda value: (abs(float(value)), float(value)),
                    )
                    if robust
                    else None
                ),
                "evaluation": state["evaluation"],
            }
        )
    transitions = []
    for item in sample.get("boundary_certifications", ()):
        observed = {
            json.dumps(value, sort_keys=True)
            for value in (
                item["left_nextafter_state"],
                item["exact_state"],
                item["right_nextafter_state"],
            )
            if value is not None
        }
        if len(observed) > 1:
            transitions.append(item)
    return {
        **_compact_sample(sample),
        "fixed_cardinality_frontier": sample.get("fixed_cardinality_frontier", []),
        "cardinality_envelope": sample.get("cardinality_envelope", []),
        "reachable_states": states,
        "boundary_transition_count": len(transitions),
        "boundary_transitions": transitions,
    }


def _audit_sample_task(
    task: tuple[Mapping[str, Any], torch.Tensor, str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record, score, detail_policy = task
    sample = audit_sample_threshold_reachability(
        record,
        score,
        detail_policy=detail_policy,
    )
    retain_detail = detail_policy == "all" or (
        detail_policy == "failures" and _sample_failed_oracle_gate(sample)
    )
    return (
        _compact_sample(sample),
        _compact_failure_frontier(sample) if retain_detail else None,
    )


def _initialise_audit_worker() -> None:
    for name in _WORKER_THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


@contextmanager
def _single_thread_worker_environment():
    previous = {name: os.environ.get(name) for name in _WORKER_THREAD_ENVIRONMENT}
    try:
        for name in _WORKER_THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _ordered_sample_audits(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[torch.Tensor],
    *,
    detail_policy: str,
    workers: int,
):
    tasks = zip(records, scores, (detail_policy for _ in records), strict=True)
    if workers == 1:
        for task in tasks:
            yield _audit_sample_task(task)
        return
    context = mp.get_context("spawn")
    with _single_thread_worker_environment():
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialise_audit_worker,
        ) as executor:
            indexed_tasks = iter(enumerate(tasks))
            pending = {}
            ready = {}
            next_output_index = 0
            window = max(1, workers * 4)

            def fill_window() -> None:
                while len(pending) + len(ready) < window:
                    try:
                        index, task = next(indexed_tasks)
                    except StopIteration:
                        return
                    future = executor.submit(_audit_sample_task, task)
                    pending[future] = index

            fill_window()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    ready[pending.pop(future)] = future
                while next_output_index in ready:
                    future = ready.pop(next_output_index)
                    yield future.result()
                    next_output_index += 1
                    fill_window()


def _detailed_frontier_indices(
    samples: Sequence[Mapping[str, Any]],
    *,
    detail_policy: str,
    maximum: int,
) -> tuple[tuple[int, ...], int]:
    if detail_policy not in DETAIL_POLICIES:
        raise ValueError(f"detail_policy must be one of {DETAIL_POLICIES}")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("maximum must be a non-negative integer")
    if detail_policy == "none":
        return (), 0
    if detail_policy == "all":
        eligible = tuple(range(len(samples)))
    else:
        eligible = tuple(
            index
            for index, sample in enumerate(samples)
            if _sample_failed_oracle_gate(sample)
        )
    return eligible[:maximum], max(0, len(eligible) - maximum)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _identity_is_train_only(record: Mapping[str, Any]) -> None:
    identity = record.get("identity")
    if not isinstance(identity, Mapping) or identity.get("source_split") != "train":
        raise SampleThresholdReachabilityError("audit records must be train-only")
    normalised = _normalised_path_text(json.dumps(dict(identity), sort_keys=True))
    if any(marker in normalised for marker in _FORBIDDEN_PATH_MARKERS):
        raise SampleThresholdReachabilityError(
            "audit record identity references forbidden data"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Threshold reachability output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _plain_json(payload),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    base_path = _reject_forbidden_path(args.base_cache, label="base cache")
    augmentation_path = _reject_forbidden_path(
        args.augmentation_cache,
        label="P0 sidecar",
    )
    checkpoint_path = _reject_forbidden_path(args.checkpoint, label="B checkpoint")
    output_path = _reject_forbidden_path(args.output, label="output")
    expected_hashes = {
        "base_cache": BASE_CACHE_SHA256,
        "augmentation_cache": AUGMENTATION_CACHE_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
    }
    actual_hashes = {
        "base_cache": _file_sha256(base_path),
        "augmentation_cache": _file_sha256(augmentation_path),
        "checkpoint": _file_sha256(checkpoint_path),
    }
    if actual_hashes != expected_hashes:
        raise SampleThresholdReachabilityError(
            f"frozen input SHA256 mismatch: {actual_hashes}"
        )

    requested = tuple(args.partitions)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("partitions must be unique and non-empty")
    if any(partition not in PARTITIONS for partition in requested):
        raise ValueError("partitions may contain only fit and calibration")
    limit = args.limit_per_partition
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit_per_partition must be a positive integer or None")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (
        isinstance(args.workers, bool)
        or not isinstance(args.workers, int)
        or args.workers < 1
    ):
        raise ValueError("workers must be a positive integer")
    if args.detail_policy not in DETAIL_POLICIES:
        raise ValueError(f"detail_policy must be one of {DETAIL_POLICIES}")
    if (
        isinstance(args.max_detailed_frontiers, bool)
        or not isinstance(args.max_detailed_frontiers, int)
        or args.max_detailed_frontiers < 0
    ):
        raise ValueError("max_detailed_frontiers must be a non-negative integer")

    implementation_sha256 = _implementation_sha256()

    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        base_path,
        augmentation_path,
    )
    model, checkpoint = load_p0_pair_scorer_checkpoint(
        checkpoint_path,
        device="cpu",
        verify_source_hashes=False,
    )
    if (
        checkpoint.get("base_cache_sha256") != BASE_CACHE_SHA256
        or checkpoint.get("augmentation_cache_sha256")
        != AUGMENTATION_CACHE_SHA256
    ):
        raise SampleThresholdReachabilityError(
            "B checkpoint is not bound to the frozen train-only caches"
        )
    device = _device(args.device)
    if device.type != "cpu" and args.workers != 1:
        raise ValueError("workers > 1 requires CPU scoring for parallel audits")
    model.to(device)
    anchor_feature_dim = int(augmentation_cache["anchor_feature_dim"])

    partition_payloads: dict[str, Any] = {}
    formal_counts = True
    for partition in requested:
        records = joined_training_records(base_cache, augmentation_cache, partition)
        if limit is not None:
            records = records[:limit]
        for record in records:
            _identity_is_train_only(record)
        expected_count = EXPECTED_PARTITION_SAMPLE_COUNTS[partition]
        formal_counts = formal_counts and len(records) == expected_count
        scores = score_p0_pair_records(
            model,
            records,
            anchor_feature_dim=anchor_feature_dim,
            batch_size=args.batch_size,
            device=device,
        )
        sample_payloads = []
        for compact, _ in _ordered_sample_audits(
            records,
            scores,
            detail_policy="none",
            workers=min(args.workers, len(records)),
        ):
            sample_payloads.append(compact)
        detail_indices, truncated_detail_count = _detailed_frontier_indices(
            sample_payloads,
            detail_policy=args.detail_policy,
            maximum=args.max_detailed_frontiers,
        )
        detailed_frontiers = []
        if detail_indices:
            detail_records = [records[index] for index in detail_indices]
            detail_scores = [scores[index] for index in detail_indices]
            detailed_results = _ordered_sample_audits(
                detail_records,
                detail_scores,
                detail_policy="all",
                workers=min(args.workers, len(detail_records)),
            )
            for sample_index, (_, detail) in zip(
                detail_indices,
                detailed_results,
                strict=True,
            ):
                if detail is None:
                    raise SampleThresholdReachabilityError(
                        "full detail replay did not return a frontier"
                    )
                detailed_frontiers.append(
                    {"partition_sample_index": sample_index, **detail}
                )
        partition_payloads[partition] = {
            **_partition_summary(sample_payloads),
            "expected_sample_count": expected_count,
            "complete_partition": len(records) == expected_count,
            "samples": sample_payloads,
            "detailed_frontier_eligible_count": (
                len(detailed_frontiers) + truncated_detail_count
            ),
            "detailed_frontier_count": len(detailed_frontiers),
            "detailed_frontier_truncated_count": truncated_detail_count,
            "detailed_frontiers": detailed_frontiers,
        }

    train_cal_formal_eligible = (
        requested == PARTITIONS
        and limit is None
        and formal_counts
        and set(partition_payloads) == set(PARTITIONS)
    )
    calibration_formal_eligible = (
        requested == ("calibration",)
        and limit is None
        and partition_payloads.get("calibration", {}).get("complete_partition") is True
    )
    formal_eligible = train_cal_formal_eligible or calibration_formal_eligible
    gate = None
    if "calibration" in partition_payloads:
        calibration_records = joined_training_records(
            base_cache,
            augmentation_cache,
            "calibration",
        )
        if limit is not None:
            calibration_records = calibration_records[:limit]
        gate = _scalar_oracle_capability_gate(
            pair_scorer_train_cal_gate(
                {"selected_metrics": partition_payloads["calibration"]["aggregate"]},
                calibration_records=calibration_records,
                calibration_component_count=int(
                    base_cache["identity_partition"][
                        "calibration_component_count"
                    ]
                ),
            )
        )
    if train_cal_formal_eligible:
        status = (
            "SCALAR-THRESHOLD-ORACLE-GO"
            if gate is not None and gate["status"] == "GO"
            else "SCALAR-THRESHOLD-ORACLE-NO-GO"
        )
        formal_scope = "fit_and_calibration"
    elif calibration_formal_eligible:
        status = (
            "CALIBRATION-SCALAR-ORACLE-GO"
            if gate is not None and gate["status"] == "GO"
            else "CALIBRATION-SCALAR-ORACLE-NO-GO"
        )
        formal_scope = "calibration_only"
    else:
        status = "PARTIAL-DIAGNOSTIC"
        formal_scope = "partial"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "formal_eligible": formal_eligible,
        "formal_scope": formal_scope,
        "train_cal_formal_eligible": train_cal_formal_eligible,
        "calibration_formal_eligible": calibration_formal_eligible,
        "oracle_ceiling_only": True,
        "deployable_model": False,
        "training_steps": 0,
        "parameter_updates": 0,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
        "protocol": {
            "same_cardinality_objective": "maximize_sum_raw_pair_score",
            "threshold_objective": "maximize_sum(score-threshold)",
            "singleton_gain": 0.0,
            "formal_frontier": (
                "parametric_upper_hull_intersection_recursion unless a perfect "
                "maximum-cardinality open certificate proves the structural ceiling"
            ),
            "formal_open_interval_certification": (
                "production state must agree at strict 25/50/75 percent witnesses"
            ),
            "formal_first_pass_detail_policy": "none",
            "perfect_state_early_certificate": (
                "maximum-cardinality production state agrees at three strict "
                "witnesses inside a conservative low-threshold subinterval"
            ),
            "early_certificate_not_training_interval": True,
            "training_interval_manifest_requirement": (
                "disable the early shortcut and compute adjacent parametric hull "
                "boundaries before emitting tau predictor labels"
            ),
            "failure_detail_second_pass": (
                "exact score and fixed-cardinality boundaries are replayed only "
                "for capped diagnostic samples after formal metrics are fixed"
            ),
            "failure_detail_critical_values": [
                "all_candidate_scores",
                "all_fixed-cardinality upper-envelope bounds",
            ],
            "replay": (
                "existing production solver at three strict witnesses in every "
                "non-degenerate generated interval"
            ),
            "boundary_policy": (
                "exact and nextafter boundary states are reported but only "
                "production-certified open-interval states count toward GO"
            ),
            "floating_point_completeness_claimed": False,
            "mathematical_envelope_is_candidate_generator_only": True,
            "candidate_generation_is_target_free": True,
            "oracle_label_usage": (
                "labels choose among production-certified states, or certify that "
                "one state reaches every candidate-relative structural upper bound"
            ),
            "temperature_scope": (
                "T is frozen to 1 at the original pair-score scale; positive "
                "temperature is redundant only in exact arithmetic"
            ),
            "production_temperature_scale_invariance_claimed": False,
            "production_temperature_numeric_effects": (
                "out of scope because the 64-ulp tie tolerance is not invariant "
                "under positive score scaling"
            ),
            "matching_latency_is_diagnostic_not_capability_gate": True,
            "sequential_gate_mode": (
                "default calibration-only invocation; fit+calibration requires an "
                "explicit second invocation after calibration GO"
            ),
        },
        "input": {
            "base_cache_path": str(base_path),
            "augmentation_cache_path": str(augmentation_path),
            "checkpoint_path": str(checkpoint_path),
            "sha256": actual_hashes,
            "partitions": list(requested),
            "limit_per_partition": limit,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "worker_start_method": "spawn",
            "worker_numeric_threads": 1,
            "highs_threads": 1,
            "worker_task_window_per_worker": 4,
            "detail_policy": args.detail_policy,
            "max_detailed_frontiers": int(args.max_detailed_frontiers),
        },
        "implementation_sha256": implementation_sha256,
        "partitions": partition_payloads,
        "formal_gate": gate,
    }
    if _implementation_sha256() != implementation_sha256:
        raise SampleThresholdReachabilityError(
            "audit implementation changed during execution"
        )
    _write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", default=DEFAULT_BASE_CACHE)
    parser.add_argument("--augmentation-cache", default=DEFAULT_AUGMENTATION_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=PARTITIONS,
        default=["calibration"],
    )
    parser.add_argument("--limit-per-partition", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-detailed-frontiers",
        type=int,
        default=DEFAULT_MAX_DETAILED_FRONTIERS,
    )
    parser.add_argument(
        "--detail-policy",
        choices=DETAIL_POLICIES,
        default="failures",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "status": report["status"],
                "formal_eligible": report["formal_eligible"],
                "training_steps": report["training_steps"],
                "parameter_updates": report["parameter_updates"],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AUGMENTATION_CACHE_SHA256",
    "BASE_CACHE_SHA256",
    "CHECKPOINT_SHA256",
    "SCHEMA_VERSION",
    "FixedCardinalityMatching",
    "SampleThresholdReachabilityError",
    "audit",
    "audit_sample_threshold_reachability",
    "build_parser",
    "cardinality_envelope",
    "fixed_cardinality_frontier",
    "main",
    "replay_threshold_frontier",
    "solve_fixed_cardinality_matching",
    "threshold_probe_values",
]
