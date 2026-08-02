"""Deterministic weighted exact cover for frozen bond proposals.

The solver in this module is deliberately independent of candidate generation
and learned models.  It accepts an already-scored set of typed bond-group
candidates and selects a partition of the active physical proposals.  Every
active proposal must belong to exactly one selected chemical group, or to an
optional internal DROP variable when proposal rejection is enabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, vstack

from chemvector.bond_group_set import (
    HASHED_WEDGE,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
    SOLID_QUAD,
    SOLID_TRIANGLE,
)
from chemvector.primitive_set import LINE, QUAD, TRIANGLE


GROUP_KIND_TO_TYPE = {
    "single_line": SINGLE_LINE,
    "parallel_2": PARALLEL_2,
    "parallel_3": PARALLEL_3,
    "hashed_wedge": HASHED_WEDGE,
    "solid_triangle": SOLID_TRIANGLE,
    "solid_quad": SOLID_QUAD,
}
GROUP_TYPE_TO_KIND = {
    group_type: kind for kind, group_type in GROUP_KIND_TO_TYPE.items()
}

PROPOSAL_KIND_TO_TYPE = {
    "line": LINE,
    "triangle": TRIANGLE,
    "quad": QUAD,
}
PROPOSAL_TYPE_TO_KIND = {
    proposal_type: kind for kind, proposal_type in PROPOSAL_KIND_TO_TYPE.items()
}

_FIXED_MEMBER_COUNTS = {
    SINGLE_LINE: 1,
    PARALLEL_2: 2,
    PARALLEL_3: 3,
    SOLID_TRIANGLE: 1,
    SOLID_QUAD: 1,
}
_EXPECTED_PROPOSAL_TYPES = {
    SINGLE_LINE: LINE,
    PARALLEL_2: LINE,
    PARALLEL_3: LINE,
    HASHED_WEDGE: LINE,
    SOLID_TRIANGLE: TRIANGLE,
    SOLID_QUAD: QUAD,
}


class BondGroupExactCoverValidationError(ValueError):
    """Raised when candidates do not satisfy the exact-cover protocol."""


def _strict_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise BondGroupExactCoverValidationError(f"{name} must be an integer")
    return int(value)


def _normalise_group_identity(
    kind: str | None,
    group_type: int | None,
) -> tuple[str, int]:
    if kind is None and group_type is None:
        raise BondGroupExactCoverValidationError(
            "A candidate requires kind or group_type"
        )

    normalised_kind: str | None = None
    normalised_type: int | None = None
    if kind is not None:
        if not isinstance(kind, str) or kind not in GROUP_KIND_TO_TYPE:
            raise BondGroupExactCoverValidationError(
                f"Unsupported bond-group kind: {kind!r}"
            )
        normalised_kind = kind
        normalised_type = GROUP_KIND_TO_TYPE[kind]

    if group_type is not None:
        candidate_type = _strict_integer(group_type, name="group_type")
        if candidate_type not in GROUP_TYPE_TO_KIND:
            raise BondGroupExactCoverValidationError(
                f"Unsupported bond-group type: {candidate_type}"
            )
        candidate_kind = GROUP_TYPE_TO_KIND[candidate_type]
        if normalised_kind is not None and normalised_kind != candidate_kind:
            raise BondGroupExactCoverValidationError(
                "Candidate kind/group_type disagree: "
                f"{normalised_kind!r} != {candidate_kind!r}"
            )
        normalised_kind = candidate_kind
        normalised_type = candidate_type

    assert normalised_kind is not None and normalised_type is not None
    return normalised_kind, normalised_type


def _validate_member_count(group_type: int, member_count: int) -> None:
    if group_type == HASHED_WEDGE:
        if member_count < 2:
            raise BondGroupExactCoverValidationError(
                "hashed_wedge requires at least two line proposals"
            )
        return

    expected = _FIXED_MEMBER_COUNTS[group_type]
    if member_count != expected:
        kind = GROUP_TYPE_TO_KIND[group_type]
        raise BondGroupExactCoverValidationError(
            f"{kind} requires exactly {expected} proposal(s), got {member_count}"
        )


@dataclass(frozen=True, slots=True)
class BondGroupCandidate:
    """A scored, typed candidate group over existing physical proposals.

    ``kind`` and ``group_type`` are two views of the same identity.  Callers
    may provide either one or both; both are populated after validation.
    Proposal indices are canonicalised to ascending order so solver behaviour
    cannot depend on member order.
    """

    proposal_indices: tuple[int, ...]
    score: float
    kind: str | None = None
    group_type: int | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        kind, group_type = _normalise_group_identity(self.kind, self.group_type)

        try:
            raw_indices = tuple(self.proposal_indices)
        except TypeError as error:
            raise BondGroupExactCoverValidationError(
                "proposal_indices must be an iterable of integers"
            ) from error
        indices = tuple(
            _strict_integer(index, name="proposal index") for index in raw_indices
        )
        if any(index < 0 for index in indices):
            raise BondGroupExactCoverValidationError(
                "proposal indices must be non-negative"
            )
        if len(set(indices)) != len(indices):
            raise BondGroupExactCoverValidationError(
                "A candidate cannot contain the same proposal more than once"
            )
        indices = tuple(sorted(indices))
        _validate_member_count(group_type, len(indices))

        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise BondGroupExactCoverValidationError("Candidate score must be numeric")
        score = float(self.score)
        if not math.isfinite(score):
            raise BondGroupExactCoverValidationError(
                "Candidate score must be finite"
            )

        metadata = self.metadata
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise BondGroupExactCoverValidationError(
                    "Candidate metadata must be a mapping or None"
                )
            metadata = dict(metadata)

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "group_type", group_type)
        object.__setattr__(self, "proposal_indices", indices)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class BondGroupExactCoverResult:
    """Result of one weighted exact-cover solve."""

    selected_candidates: tuple[BondGroupCandidate, ...]
    feasible: bool
    objective: float | None
    solver_status: str
    solver_status_code: int
    solver_message: str
    latency_ms: float
    dropped_proposal_indices: tuple[int, ...] = ()

    @property
    def selection(self) -> tuple[BondGroupCandidate, ...]:
        """Concise alias used by downstream decoders."""

        return self.selected_candidates


def _normalise_proposal_type(value: Any, *, proposal_index: int) -> int:
    if isinstance(value, str):
        if value not in PROPOSAL_KIND_TO_TYPE:
            raise BondGroupExactCoverValidationError(
                f"Proposal {proposal_index} has unsupported physical type {value!r}"
            )
        return PROPOSAL_KIND_TO_TYPE[value]

    proposal_type = _strict_integer(
        value,
        name=f"physical type for proposal {proposal_index}",
    )
    if proposal_type not in PROPOSAL_TYPE_TO_KIND:
        raise BondGroupExactCoverValidationError(
            f"Proposal {proposal_index} has unsupported physical type {proposal_type}"
        )
    return proposal_type


def _normalise_proposals(
    proposal_types: Mapping[int, str | int] | Sequence[str | int],
    active_proposal_indices: Sequence[int] | None,
) -> tuple[dict[int, int], tuple[int, ...]]:
    if isinstance(proposal_types, Mapping):
        type_by_index: dict[int, int] = {}
        for raw_index, raw_type in proposal_types.items():
            index = _strict_integer(raw_index, name="proposal_types key")
            if index < 0:
                raise BondGroupExactCoverValidationError(
                    "proposal_types keys must be non-negative"
                )
            type_by_index[index] = _normalise_proposal_type(
                raw_type,
                proposal_index=index,
            )
        default_active = tuple(sorted(type_by_index))
    else:
        if isinstance(proposal_types, (str, bytes)) or not isinstance(
            proposal_types, Sequence
        ):
            raise BondGroupExactCoverValidationError(
                "proposal_types must be a mapping or sequence"
            )
        type_by_index = {
            index: _normalise_proposal_type(value, proposal_index=index)
            for index, value in enumerate(proposal_types)
        }
        default_active = tuple(range(len(proposal_types)))

    if active_proposal_indices is None:
        active = default_active
    else:
        if isinstance(active_proposal_indices, (str, bytes)):
            raise BondGroupExactCoverValidationError(
                "active_proposal_indices must be a sequence of integers"
            )
        active = tuple(
            _strict_integer(index, name="active proposal index")
            for index in active_proposal_indices
        )
        if len(set(active)) != len(active):
            raise BondGroupExactCoverValidationError(
                "active_proposal_indices contains duplicates"
            )
        active = tuple(sorted(active))

    missing_types = [index for index in active if index not in type_by_index]
    if missing_types:
        raise BondGroupExactCoverValidationError(
            f"Active proposal index is out of bounds: {missing_types[0]}"
        )
    return type_by_index, active


def _candidate_key(candidate: BondGroupCandidate) -> tuple[Any, ...]:
    return (
        candidate.proposal_indices,
        int(candidate.group_type),
        candidate.kind,
        candidate.score,
    )


def _validate_candidates(
    candidates: Sequence[BondGroupCandidate],
    *,
    type_by_index: Mapping[int, int],
    active: tuple[int, ...],
    require_coverage: bool = True,
) -> tuple[BondGroupCandidate, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise BondGroupExactCoverValidationError(
            "candidates must be a sequence of BondGroupCandidate values"
        )

    active_set = set(active)
    normalised: list[BondGroupCandidate] = []
    semantic_keys: set[tuple[int, tuple[int, ...]]] = set()
    covered: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, BondGroupCandidate):
            raise BondGroupExactCoverValidationError(
                "Every candidate must be a BondGroupCandidate"
            )
        semantic_key = (int(candidate.group_type), candidate.proposal_indices)
        if semantic_key in semantic_keys:
            raise BondGroupExactCoverValidationError(
                "Duplicate typed candidate for proposals "
                f"{candidate.proposal_indices}"
            )
        semantic_keys.add(semantic_key)

        for proposal_index in candidate.proposal_indices:
            if proposal_index not in type_by_index:
                raise BondGroupExactCoverValidationError(
                    f"Candidate proposal index is out of bounds: {proposal_index}"
                )
            if proposal_index not in active_set:
                raise BondGroupExactCoverValidationError(
                    f"Candidate references inactive proposal {proposal_index}"
                )

        expected_type = _EXPECTED_PROPOSAL_TYPES[int(candidate.group_type)]
        for proposal_index in candidate.proposal_indices:
            actual_type = type_by_index[proposal_index]
            if actual_type != expected_type:
                raise BondGroupExactCoverValidationError(
                    f"{candidate.kind} requires {PROPOSAL_TYPE_TO_KIND[expected_type]} "
                    f"proposals, but proposal {proposal_index} is "
                    f"{PROPOSAL_TYPE_TO_KIND[actual_type]}"
                )

        covered.update(candidate.proposal_indices)
        normalised.append(candidate)

    uncovered = sorted(active_set - covered)
    if require_coverage and uncovered:
        raise BondGroupExactCoverValidationError(
            f"Active proposal {uncovered[0]} is not covered by any candidate"
        )
    return tuple(sorted(normalised, key=_candidate_key))


def _normalise_drop_scores(
    drop_scores: Mapping[int, Real] | Sequence[Real] | None,
    *,
    type_by_index: Mapping[int, int],
    active: tuple[int, ...],
) -> tuple[float, ...]:
    if drop_scores is None:
        return (0.0,) * len(active)

    def finite_score(value: Any, *, proposal_index: int) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise BondGroupExactCoverValidationError(
                f"Drop score for proposal {proposal_index} must be numeric"
            )
        score = float(value)
        if not math.isfinite(score):
            raise BondGroupExactCoverValidationError(
                f"Drop score for proposal {proposal_index} must be finite"
            )
        return score

    if isinstance(drop_scores, Mapping):
        score_by_index: dict[int, float] = {}
        for raw_index, raw_score in drop_scores.items():
            index = _strict_integer(raw_index, name="drop_scores key")
            if index not in type_by_index:
                raise BondGroupExactCoverValidationError(
                    f"Drop score proposal index is out of bounds: {index}"
                )
            score_by_index[index] = finite_score(
                raw_score,
                proposal_index=index,
            )
        return tuple(score_by_index.get(index, 0.0) for index in active)

    if isinstance(drop_scores, (str, bytes)) or not isinstance(
        drop_scores,
        Sequence,
    ):
        raise BondGroupExactCoverValidationError(
            "drop_scores must be a mapping or sequence of numeric values"
        )
    values = tuple(
        finite_score(value, proposal_index=index)
        for index, value in enumerate(drop_scores)
    )
    missing = [index for index in active if index >= len(values)]
    if missing:
        raise BondGroupExactCoverValidationError(
            f"Drop score is missing for active proposal {missing[0]}"
        )
    return tuple(values[index] for index in active)


def _status_name(status_code: int) -> str:
    return {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }.get(status_code, "unknown")


def _build_partition_constraints(
    candidates: tuple[BondGroupCandidate, ...],
    active: tuple[int, ...],
    known_group_count: int | None,
    *,
    include_drop: bool,
) -> tuple[coo_array, np.ndarray, np.ndarray]:
    row_by_proposal = {proposal: row for row, proposal in enumerate(active)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for column, candidate in enumerate(candidates):
        for proposal in candidate.proposal_indices:
            rows.append(row_by_proposal[proposal])
            columns.append(column)
            values.append(1.0)

    chemical_count = len(candidates)
    if include_drop:
        for offset, proposal in enumerate(active):
            rows.append(row_by_proposal[proposal])
            columns.append(chemical_count + offset)
            values.append(1.0)

    row_count = len(active)
    if known_group_count is not None:
        rows.extend([row_count] * chemical_count)
        columns.extend(range(chemical_count))
        values.extend([1.0] * chemical_count)
        row_count += 1

    variable_count = chemical_count + (len(active) if include_drop else 0)
    matrix = coo_array(
        (values, (rows, columns)),
        shape=(row_count, variable_count),
        dtype=np.float64,
    ).tocsr()
    lower = np.ones(row_count, dtype=np.float64)
    upper = np.ones(row_count, dtype=np.float64)
    if known_group_count is not None:
        lower[-1] = float(known_group_count)
        upper[-1] = float(known_group_count)
    return matrix, lower, upper


def _is_valid_selection(
    selected: tuple[BondGroupCandidate, ...],
    dropped: tuple[int, ...],
    active: tuple[int, ...],
    known_group_count: int | None,
) -> bool:
    if known_group_count is not None and len(selected) != known_group_count:
        return False
    ownership = {proposal: 0 for proposal in active}
    for candidate in selected:
        for proposal in candidate.proposal_indices:
            ownership[proposal] += 1
    if len(set(dropped)) != len(dropped):
        return False
    for proposal in dropped:
        if proposal not in ownership:
            return False
        ownership[proposal] += 1
    return all(count == 1 for count in ownership.values())


def solve_bond_group_exact_cover(
    candidates: Sequence[BondGroupCandidate],
    proposal_types: Mapping[int, str | int] | Sequence[str | int],
    *,
    active_proposal_indices: Sequence[int] | None = None,
    known_group_count: int | None = None,
    time_limit_seconds: float | None = None,
    score_tolerance: float = 1e-9,
    allow_drop: bool = False,
    drop_scores: Mapping[int, Real] | Sequence[Real] | None = None,
) -> BondGroupExactCoverResult:
    """Select the maximum-score exact partition of active proposals.

    The primary objective maximises the sum of chemical-candidate and optional
    per-proposal DROP scores.  A second HiGHS solve, constrained to the primary
    optimum within ``score_tolerance``, deterministically prefers fewer DROP
    variables, fewer chemical groups, and then canonical variable order.
    Candidate input order therefore cannot change the chosen partition.

    ``allow_drop=True`` gives every active proposal one private DROP variable.
    Passing ``drop_scores`` also enables DROP and defaults unspecified mapping
    entries to zero.  DROP variables are not returned as typed candidates and
    do not contribute to ``known_group_count``.

    Without DROP, missing candidate coverage is malformed input and raises
    :class:`BondGroupExactCoverValidationError`.  With DROP, uncovered active
    proposals are valid and must be rejected by the solver.  A fully covered
    candidate pool that cannot form a partition returns ``feasible=False``.
    """

    started = perf_counter()
    type_by_index, active = _normalise_proposals(
        proposal_types,
        active_proposal_indices,
    )
    if not isinstance(allow_drop, bool):
        raise BondGroupExactCoverValidationError("allow_drop must be a boolean")
    drop_enabled = allow_drop or drop_scores is not None
    drop_score_values = _normalise_drop_scores(
        drop_scores,
        type_by_index=type_by_index,
        active=active,
    )
    normalised = _validate_candidates(
        candidates,
        type_by_index=type_by_index,
        active=active,
        require_coverage=not drop_enabled,
    )

    if known_group_count is not None:
        known_group_count = _strict_integer(
            known_group_count,
            name="known_group_count",
        )
        if known_group_count < 0:
            raise BondGroupExactCoverValidationError(
                "known_group_count must be non-negative"
            )
    if isinstance(score_tolerance, bool) or not isinstance(score_tolerance, Real):
        raise BondGroupExactCoverValidationError("score_tolerance must be numeric")
    score_tolerance = float(score_tolerance)
    if not math.isfinite(score_tolerance) or score_tolerance < 0.0:
        raise BondGroupExactCoverValidationError(
            "score_tolerance must be finite and non-negative"
        )
    if time_limit_seconds is not None:
        if isinstance(time_limit_seconds, bool) or not isinstance(
            time_limit_seconds, Real
        ):
            raise BondGroupExactCoverValidationError(
                "time_limit_seconds must be numeric"
            )
        time_limit_seconds = float(time_limit_seconds)
        if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0:
            raise BondGroupExactCoverValidationError(
                "time_limit_seconds must be finite and positive"
            )

    if not active:
        feasible = known_group_count in (None, 0)
        latency_ms = (perf_counter() - started) * 1000.0
        return BondGroupExactCoverResult(
            selected_candidates=(),
            feasible=feasible,
            objective=0.0 if feasible else None,
            solver_status="optimal" if feasible else "infeasible",
            solver_status_code=0 if feasible else 2,
            solver_message=(
                "Empty exact cover" if feasible else "Known count is non-zero"
            ),
            latency_ms=latency_ms,
            dropped_proposal_indices=(),
        )

    matrix, lower, upper = _build_partition_constraints(
        normalised,
        active,
        known_group_count,
        include_drop=drop_enabled,
    )
    chemical_scores = np.asarray(
        [candidate.score for candidate in normalised],
        dtype=np.float64,
    )
    if drop_enabled:
        scores = np.concatenate(
            (chemical_scores, np.asarray(drop_score_values, dtype=np.float64))
        )
    else:
        scores = chemical_scores
    constraints = LinearConstraint(matrix, lower, upper)
    options: dict[str, float] = {}
    if time_limit_seconds is not None:
        options["time_limit"] = time_limit_seconds

    primary = milp(
        c=-scores,
        integrality=np.ones(len(scores), dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options=options or None,
    )
    final = primary

    if primary.status == 0 and primary.x is not None:
        primary_objective = float(np.dot(scores, np.rint(primary.x)))
        score_row = coo_array(scores.reshape(1, -1)).tocsr()
        tie_matrix = vstack((matrix, score_row), format="csr")
        tolerance = max(
            score_tolerance,
            64.0 * float(np.spacing(max(1.0, abs(primary_objective)))),
        )
        tie_lower = np.concatenate(
            (lower, np.asarray([primary_objective - tolerance]))
        )
        tie_upper = np.concatenate(
            (upper, np.asarray([primary_objective + tolerance]))
        )

        # DROP count dominates chemical group count.  Canonical fractional
        # ranks break all remaining ties independently of caller order.
        variable_count = len(scores)
        rank = np.arange(variable_count, dtype=np.float64)
        canonical_rank = (rank + 1.0) / (
            2.0 * (variable_count + 1.0) ** 2
        )
        secondary = 1.0 + canonical_rank
        if drop_enabled:
            secondary[len(normalised) :] += float(len(active))
        tie_options = dict(options)
        if time_limit_seconds is not None:
            remaining = time_limit_seconds - (perf_counter() - started)
            if remaining > 0.0:
                tie_options["time_limit"] = remaining
            else:
                tie_options = {}
        if time_limit_seconds is None or tie_options:
            tied = milp(
                c=secondary,
                integrality=np.ones(len(scores), dtype=np.uint8),
                bounds=Bounds(0.0, 1.0),
                constraints=LinearConstraint(tie_matrix, tie_lower, tie_upper),
                options=tie_options or None,
            )
            if tied.x is not None:
                final = tied

    selected: tuple[BondGroupCandidate, ...] = ()
    dropped: tuple[int, ...] = ()
    feasible = False
    objective: float | None = None
    if final.x is not None:
        selected = tuple(
            candidate
            for candidate, value in zip(
                normalised,
                final.x[: len(normalised)],
                strict=True,
            )
            if value > 0.5
        )
        if drop_enabled:
            dropped = tuple(
                proposal
                for proposal, value in zip(
                    active,
                    final.x[len(normalised) :],
                    strict=True,
                )
                if value > 0.5
            )
        feasible = _is_valid_selection(
            selected,
            dropped,
            active,
            known_group_count,
        )
        if feasible:
            drop_score_by_index = dict(zip(active, drop_score_values, strict=True))
            objective = math.fsum(
                [candidate.score for candidate in selected]
                + [drop_score_by_index[index] for index in dropped]
            )
        else:
            selected = ()
            dropped = ()

    status_code = int(final.status)
    status = _status_name(status_code)
    if feasible and status_code != 0:
        status = f"feasible_{status}"
    latency_ms = (perf_counter() - started) * 1000.0
    return BondGroupExactCoverResult(
        selected_candidates=selected,
        feasible=feasible,
        objective=objective,
        solver_status=status,
        solver_status_code=status_code,
        solver_message=str(final.message),
        latency_ms=latency_ms,
        dropped_proposal_indices=dropped,
    )


solve_exact_cover = solve_bond_group_exact_cover
ExactCoverCandidate = BondGroupCandidate
ExactCoverResult = BondGroupExactCoverResult


__all__ = [
    "BondGroupCandidate",
    "BondGroupExactCoverResult",
    "BondGroupExactCoverValidationError",
    "ExactCoverCandidate",
    "ExactCoverResult",
    "GROUP_KIND_TO_TYPE",
    "GROUP_TYPE_TO_KIND",
    "PROPOSAL_KIND_TO_TYPE",
    "PROPOSAL_TYPE_TO_KIND",
    "solve_bond_group_exact_cover",
    "solve_exact_cover",
]
