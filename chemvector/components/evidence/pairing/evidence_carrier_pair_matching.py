"""Deterministic pair/singleton matching for evidence-carrier anchors.

Every active raw anchor is represented exactly once: either by one selected
two-anchor stroke candidate or by its zero-gain singleton fallback.  Pair
scores are relative gains over that all-singleton baseline, so non-positive
pairs can never be selected.

The primary solve is a maximum-weight matching MILP.  Equal-score solutions
are made unique by lexicographically fixing candidate decisions in canonical
geometry/ID order.  This uses repeated HiGHS feasibility solves rather than a
sum of fractional ranks, whose subset sums are not generally unique.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, eye as sparse_eye, vstack


class EvidenceCarrierPairMatchingValidationError(ValueError):
    """Raised when the pair/singleton matching contract is malformed."""


def _anchor_id(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EvidenceCarrierPairMatchingValidationError(
            f"{name} must be an integer"
        )
    result = int(value)
    if result < 0:
        raise EvidenceCarrierPairMatchingValidationError(
            f"{name} must be non-negative"
        )
    return result


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvidenceCarrierPairMatchingValidationError(
            f"{name} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceCarrierPairMatchingValidationError(
            f"{name} must be finite"
        )
    return result


def canonical_segment_geometry_key(segment_pixels: Any) -> tuple[float, ...]:
    """Return an endpoint-order-independent key for one finite line segment."""

    segment = np.asarray(segment_pixels, dtype=np.float64)
    if segment.shape != (2, 2):
        raise EvidenceCarrierPairMatchingValidationError(
            "segment_pixels must have shape [2,2]"
        )
    if not np.isfinite(segment).all():
        raise EvidenceCarrierPairMatchingValidationError(
            "segment_pixels must contain finite coordinates"
        )
    first = tuple(float(value) for value in segment[0])
    second = tuple(float(value) for value in segment[1])
    lower, upper = sorted((first, second))
    return (*lower, *upper)


def canonical_pair_geometry_key(
    first_segment_pixels: Any,
    second_segment_pixels: Any,
) -> tuple[float, ...]:
    """Return a segment- and endpoint-order-independent pair geometry key."""

    first = canonical_segment_geometry_key(first_segment_pixels)
    second = canonical_segment_geometry_key(second_segment_pixels)
    lower, upper = sorted((first, second))
    return (*lower, *upper)


@dataclass(frozen=True, slots=True)
class EvidenceCarrierPairCandidate:
    """One possible merge of two active raw evidence anchors.

    ``relative_gain`` is measured against retaining both anchors as separate
    zero-gain singletons.  ``canonical_geometry_key`` is optional, but a solve
    must either provide it for every candidate or for none.  When present it
    owns tie-breaking before the canonical anchor IDs.
    """

    anchor_ids: tuple[int, int]
    relative_gain: float
    canonical_geometry_key: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        try:
            raw_anchor_ids = tuple(self.anchor_ids)
        except TypeError as error:
            raise EvidenceCarrierPairMatchingValidationError(
                "anchor_ids must contain exactly two integers"
            ) from error
        if len(raw_anchor_ids) != 2:
            raise EvidenceCarrierPairMatchingValidationError(
                "anchor_ids must contain exactly two integers"
            )
        anchor_ids = tuple(
            sorted(
                _anchor_id(value, name="candidate anchor ID")
                for value in raw_anchor_ids
            )
        )
        if anchor_ids[0] == anchor_ids[1]:
            raise EvidenceCarrierPairMatchingValidationError(
                "A pair candidate requires two distinct anchor IDs"
            )

        relative_gain = _finite_real(
            self.relative_gain,
            name="relative_gain",
        )
        geometry = self.canonical_geometry_key
        if geometry is not None:
            if isinstance(geometry, (str, bytes)):
                raise EvidenceCarrierPairMatchingValidationError(
                    "canonical_geometry_key must be a numeric sequence or None"
                )
            try:
                raw_geometry = tuple(geometry)
            except TypeError as error:
                raise EvidenceCarrierPairMatchingValidationError(
                    "canonical_geometry_key must be a numeric sequence or None"
                ) from error
            if not raw_geometry:
                raise EvidenceCarrierPairMatchingValidationError(
                    "canonical_geometry_key cannot be empty"
                )
            geometry = tuple(
                _finite_real(value, name="canonical geometry value")
                for value in raw_geometry
            )

        object.__setattr__(self, "anchor_ids", anchor_ids)
        object.__setattr__(self, "relative_gain", relative_gain)
        object.__setattr__(self, "canonical_geometry_key", geometry)


@dataclass(frozen=True, slots=True)
class EvidenceCarrierPairMatchingResult:
    """Deterministic partition of active anchors into pairs and singletons."""

    selected_pairs: tuple[EvidenceCarrierPairCandidate, ...]
    selected_singletons: tuple[int, ...]
    total_gain: float
    feasible: bool
    optimal: bool
    solver_status: str
    solver_status_code: int
    solver_message: str
    latency_ms: float
    positive_candidate_count: int
    tie_break_solve_count: int

    @property
    def selected_singleton_anchor_ids(self) -> tuple[int, ...]:
        """Explicit alias for callers that name the singleton values as IDs."""

        return self.selected_singletons


def _candidate_key(
    candidate: EvidenceCarrierPairCandidate,
) -> tuple[Any, ...]:
    geometry = candidate.canonical_geometry_key
    return (
        () if geometry is None else geometry,
        candidate.anchor_ids,
    )


def _normalise_inputs(
    active_anchor_ids: Sequence[int],
    candidates: Sequence[EvidenceCarrierPairCandidate],
) -> tuple[tuple[int, ...], tuple[EvidenceCarrierPairCandidate, ...]]:
    if isinstance(active_anchor_ids, (str, bytes)) or not isinstance(
        active_anchor_ids,
        Sequence,
    ):
        raise EvidenceCarrierPairMatchingValidationError(
            "active_anchor_ids must be a sequence of integers"
        )
    active = tuple(
        _anchor_id(value, name="active anchor ID") for value in active_anchor_ids
    )
    if len(set(active)) != len(active):
        raise EvidenceCarrierPairMatchingValidationError(
            "active_anchor_ids contains duplicates"
        )
    active = tuple(sorted(active))

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise EvidenceCarrierPairMatchingValidationError(
            "candidates must be a sequence of EvidenceCarrierPairCandidate values"
        )
    active_set = set(active)
    pair_keys: set[tuple[int, int]] = set()
    normalised = []
    geometry_modes: set[bool] = set()
    for candidate in candidates:
        if not isinstance(candidate, EvidenceCarrierPairCandidate):
            raise EvidenceCarrierPairMatchingValidationError(
                "Every candidate must be an EvidenceCarrierPairCandidate"
            )
        if candidate.anchor_ids in pair_keys:
            raise EvidenceCarrierPairMatchingValidationError(
                f"Duplicate pair candidate for anchors {candidate.anchor_ids}"
            )
        pair_keys.add(candidate.anchor_ids)
        missing = [
            anchor_id
            for anchor_id in candidate.anchor_ids
            if anchor_id not in active_set
        ]
        if missing:
            raise EvidenceCarrierPairMatchingValidationError(
                f"Candidate references inactive anchor ID {missing[0]}"
            )
        geometry_modes.add(candidate.canonical_geometry_key is not None)
        normalised.append(candidate)
    if len(geometry_modes) > 1:
        raise EvidenceCarrierPairMatchingValidationError(
            "canonical_geometry_key must be provided for every candidate or none"
        )
    return active, tuple(sorted(normalised, key=_candidate_key))


def _status_name(status_code: int) -> str:
    return {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }.get(status_code, "unknown")


def _empty_result(
    active: tuple[int, ...],
    *,
    started: float,
    message: str,
) -> EvidenceCarrierPairMatchingResult:
    return EvidenceCarrierPairMatchingResult(
        selected_pairs=(),
        selected_singletons=active,
        total_gain=0.0,
        feasible=True,
        optimal=True,
        solver_status="optimal",
        solver_status_code=0,
        solver_message=message,
        latency_ms=(perf_counter() - started) * 1000.0,
        positive_candidate_count=0,
        tie_break_solve_count=0,
    )


def _failure_result(
    active: tuple[int, ...],
    *,
    started: float,
    positive_candidate_count: int,
    tie_break_solve_count: int,
    status_code: int,
    message: str,
    prefix: str,
) -> EvidenceCarrierPairMatchingResult:
    return EvidenceCarrierPairMatchingResult(
        selected_pairs=(),
        selected_singletons=active,
        total_gain=0.0,
        feasible=True,
        optimal=False,
        solver_status=f"{prefix}_{_status_name(status_code)}",
        solver_status_code=status_code,
        solver_message=message,
        latency_ms=(perf_counter() - started) * 1000.0,
        positive_candidate_count=positive_candidate_count,
        tie_break_solve_count=tie_break_solve_count,
    )


def _success_result(
    active: tuple[int, ...],
    selected_pairs: tuple[EvidenceCarrierPairCandidate, ...],
    *,
    started: float,
    positive_candidate_count: int,
    tie_break_solve_count: int,
    message: str,
) -> EvidenceCarrierPairMatchingResult:
    paired_anchors = {
        anchor_id
        for candidate in selected_pairs
        for anchor_id in candidate.anchor_ids
    }
    selected_singletons = tuple(
        anchor_id for anchor_id in active if anchor_id not in paired_anchors
    )
    return EvidenceCarrierPairMatchingResult(
        selected_pairs=selected_pairs,
        selected_singletons=selected_singletons,
        total_gain=math.fsum(
            candidate.relative_gain for candidate in selected_pairs
        ),
        feasible=True,
        optimal=True,
        solver_status="optimal",
        solver_status_code=0,
        solver_message=message,
        latency_ms=(perf_counter() - started) * 1000.0,
        positive_candidate_count=positive_candidate_count,
        tie_break_solve_count=tie_break_solve_count,
    )


def solve_evidence_carrier_pair_matching(
    active_anchor_ids: Sequence[int],
    candidates: Sequence[EvidenceCarrierPairCandidate],
    *,
    score_tolerance: float = 0.0,
) -> EvidenceCarrierPairMatchingResult:
    """Select maximum-gain disjoint pairs; retain every other anchor singly.

    The all-singleton partition is always feasible and has total gain zero.
    Candidates with gain less than or equal to zero are therefore excluded
    before optimization.  Among primary-optimal matchings, the solver
    lexicographically prefers inclusion in canonical geometry/ID order.  The
    resulting partition is invariant to active-anchor and candidate ordering.
    """

    started = perf_counter()
    active, normalised = _normalise_inputs(active_anchor_ids, candidates)
    score_tolerance = _finite_real(score_tolerance, name="score_tolerance")
    if score_tolerance < 0.0:
        raise EvidenceCarrierPairMatchingValidationError(
            "score_tolerance must be non-negative"
        )
    positive = tuple(
        candidate for candidate in normalised if candidate.relative_gain > 0.0
    )
    if not positive:
        return _empty_result(
            active,
            started=started,
            message="No positive-gain pair candidates; all anchors are singletons",
        )

    row_by_anchor = {anchor_id: row for row, anchor_id in enumerate(active)}
    rows = []
    columns = []
    for column, candidate in enumerate(positive):
        for anchor_id in candidate.anchor_ids:
            rows.append(row_by_anchor[anchor_id])
            columns.append(column)
    incidence = coo_array(
        (
            np.ones(len(rows), dtype=np.float64),
            (rows, columns),
        ),
        shape=(len(active), len(positive)),
        dtype=np.float64,
    ).tocsr()
    gains = np.asarray(
        [candidate.relative_gain for candidate in positive],
        dtype=np.float64,
    )
    integrality = np.ones(len(positive), dtype=np.uint8)
    bounds = Bounds(0.0, 1.0)
    matching_constraint = LinearConstraint(
        incidence,
        np.full(len(active), -np.inf, dtype=np.float64),
        np.ones(len(active), dtype=np.float64),
    )
    options = {"mip_rel_gap": 0.0}
    primary = milp(
        c=-gains,
        integrality=integrality,
        bounds=bounds,
        constraints=matching_constraint,
        options=options,
    )
    if primary.status != 0 or primary.x is None:
        return _failure_result(
            active,
            started=started,
            positive_candidate_count=len(positive),
            tie_break_solve_count=0,
            status_code=int(primary.status),
            message=str(primary.message),
            prefix="primary",
        )

    primary_selection = np.rint(primary.x).astype(np.int8)
    primary_gain = float(np.dot(gains, primary_selection))
    tolerance = max(
        score_tolerance,
        64.0 * float(np.spacing(max(1.0, abs(primary_gain)))),
    )

    # Most learned floating-point gains have a unique optimum.  Certify that
    # with one no-good solve before paying for exact lexicographic fixing.
    # The no-good row excludes precisely the primary binary vector.
    no_good = np.where(primary_selection == 1, -1.0, 1.0)
    no_good_lower = 1.0 - float(np.sum(primary_selection))
    alternative_matrix = vstack(
        (incidence, coo_array(no_good.reshape(1, -1))),
        format="csr",
    )
    alternative = milp(
        c=-gains,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            alternative_matrix,
            np.concatenate(
                (
                    np.full(len(active), -np.inf, dtype=np.float64),
                    np.asarray([no_good_lower], dtype=np.float64),
                )
            ),
            np.concatenate(
                (
                    np.ones(len(active), dtype=np.float64),
                    np.asarray([np.inf], dtype=np.float64),
                )
            ),
        ),
        options=options,
    )
    if alternative.status == 2:
        selected_pairs = tuple(
            candidate
            for candidate, selected in zip(
                positive,
                primary_selection,
                strict=True,
            )
            if selected == 1
        )
        return _success_result(
            active,
            selected_pairs,
            started=started,
            positive_candidate_count=len(positive),
            tie_break_solve_count=1,
            message="HiGHS primary optimum certified unique",
        )
    if alternative.status == 0 and alternative.x is not None:
        alternative_selection = np.rint(alternative.x).astype(np.int8)
        alternative_gain = float(np.dot(gains, alternative_selection))
        if alternative_gain < primary_gain - tolerance:
            selected_pairs = tuple(
                candidate
                for candidate, selected in zip(
                    positive,
                    primary_selection,
                    strict=True,
                )
                if selected == 1
            )
            return _success_result(
                active,
                selected_pairs,
                started=started,
                positive_candidate_count=len(positive),
                tie_break_solve_count=1,
                message=(
                    "HiGHS primary optimum certified unique within "
                    f"score tolerance {tolerance:.3g}"
                ),
            )

    identity = sparse_eye(len(positive), dtype=np.float64, format="csr")
    score_row = coo_array(gains.reshape(1, -1)).tocsr()
    tie_matrix = vstack((incidence, score_row, identity), format="csr")
    row_count = tie_matrix.shape[0]
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.full(row_count, np.inf, dtype=np.float64)
    upper[: len(active)] = 1.0
    score_row_index = len(active)
    lower[score_row_index] = primary_gain - tolerance
    upper[score_row_index] = primary_gain + tolerance
    identity_offset = score_row_index + 1

    fixed = np.full(len(positive), -1, dtype=np.int8)
    tie_break_solve_count = 0
    for candidate_index, candidate in enumerate(positive):
        if fixed[candidate_index] >= 0:
            continue
        trial_lower = lower.copy()
        trial_upper = upper.copy()
        trial_lower[identity_offset + candidate_index] = 1.0
        trial_upper[identity_offset + candidate_index] = 1.0
        trial = milp(
            c=np.zeros(len(positive), dtype=np.float64),
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(
                tie_matrix,
                trial_lower,
                trial_upper,
            ),
            options=options,
        )
        tie_break_solve_count += 1
        if trial.status == 0 and trial.x is not None:
            fixed[candidate_index] = 1
            lower[identity_offset + candidate_index] = 1.0
            upper[identity_offset + candidate_index] = 1.0
            selected_anchors = set(candidate.anchor_ids)
            for later_index in range(candidate_index + 1, len(positive)):
                if fixed[later_index] >= 0:
                    continue
                if selected_anchors.intersection(positive[later_index].anchor_ids):
                    fixed[later_index] = 0
                    lower[identity_offset + later_index] = 0.0
                    upper[identity_offset + later_index] = 0.0
        elif trial.status == 2:
            fixed[candidate_index] = 0
            lower[identity_offset + candidate_index] = 0.0
            upper[identity_offset + candidate_index] = 0.0
        else:
            return _failure_result(
                active,
                started=started,
                positive_candidate_count=len(positive),
                tie_break_solve_count=tie_break_solve_count,
                status_code=int(trial.status),
                message=str(trial.message),
                prefix="tie_break",
            )

    selected_pairs = tuple(
        candidate
        for candidate, selected in zip(positive, fixed, strict=True)
        if selected == 1
    )
    total_gain = math.fsum(
        candidate.relative_gain for candidate in selected_pairs
    )
    if total_gain + tolerance < primary_gain:
        return _failure_result(
            active,
            started=started,
            positive_candidate_count=len(positive),
            tie_break_solve_count=tie_break_solve_count,
            status_code=4,
            message=(
                "Canonical tie-break lost primary objective: "
                f"{total_gain} < {primary_gain}"
            ),
            prefix="validation",
        )

    return _success_result(
        active,
        selected_pairs,
        started=started,
        positive_candidate_count=len(positive),
        tie_break_solve_count=tie_break_solve_count + 1,
        message=(
            "HiGHS primary optimum and exact canonical lexicographic "
            f"tie-break completed ({tie_break_solve_count} feasibility solves)"
        ),
    )


PairCandidate = EvidenceCarrierPairCandidate
PairSingletonMatchingResult = EvidenceCarrierPairMatchingResult
PairMatchingValidationError = EvidenceCarrierPairMatchingValidationError
solve_pair_singleton_matching = solve_evidence_carrier_pair_matching


__all__ = [
    "EvidenceCarrierPairCandidate",
    "EvidenceCarrierPairMatchingResult",
    "EvidenceCarrierPairMatchingValidationError",
    "PairCandidate",
    "PairMatchingValidationError",
    "PairSingletonMatchingResult",
    "canonical_pair_geometry_key",
    "canonical_segment_geometry_key",
    "solve_evidence_carrier_pair_matching",
    "solve_pair_singleton_matching",
]
