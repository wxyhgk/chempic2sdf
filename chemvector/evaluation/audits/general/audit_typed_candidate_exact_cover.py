#!/usr/bin/env python3
"""Audit geometry-only typed group candidates over frozen flat proposals.

The candidate generator receives only frozen proposal geometry, primitive type,
and the oracle-usable proposal mask.  GT group identities are used afterwards
to measure candidate recall and to certify whether the GT partition is an exact
cover.  This is deliberately a feasibility audit, not an end-to-end score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    PARALLEL_3,
    PHYSICAL_GROUP_NAMES,
    SINGLE_LINE,
    SOLID_QUAD,
    SOLID_TRIANGLE,
)
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    aggregate_group_slot_targets,
)
from chemvector.primitive_set import LINE, QUAD, TRIANGLE
from chemvector.train_bond_group_proposal_membership import (
    DEFAULT_CAPACITY_SHARD,
    SOURCE_PRIMITIVE_CHECKPOINT,
    SOURCE_PRIMITIVE_SHA256,
    autocast_context,
    expand_shards,
    load_frozen_primitive_checkpoint,
    reject_internal_test_metadata,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


AUDIT_SCHEMA_VERSION = "typed-candidate-exact-cover-audit-0.1"
DEFAULT_OUTPUT_DIR = (
    "outputs/evaluations/"
    "chemvector-v097-typed-candidate-affinity-oracle-capacity100-final"
)
PHYSICAL_TYPES = (
    SINGLE_LINE,
    PARALLEL_2,
    PARALLEL_3,
    HASHED_WEDGE,
    SOLID_TRIANGLE,
    SOLID_QUAD,
)


@dataclass(frozen=True, slots=True)
class CandidateGeometryThresholds:
    """Fixed q2047 gates used without per-image or GT adaptation."""

    parallel_max_angle_degrees: float = 30.0
    parallel_min_length_ratio: float = 0.10
    parallel_max_center_distance_q: float = 110.0
    parallel_max_center_distance_relative: float = 0.80
    parallel_max_longitudinal_offset_q: float = 90.0
    hashed_max_line_length_q: float = 80.0
    hashed_corridor_half_width_q: float = 40.0
    hashed_max_consecutive_gap_q: float = 64.0
    hashed_endpoint_margin_q: float = 2.0
    hashed_max_members: int = 143

    def __post_init__(self) -> None:
        positive = (
            self.parallel_max_angle_degrees,
            self.parallel_min_length_ratio,
            self.parallel_max_center_distance_q,
            self.parallel_max_center_distance_relative,
            self.parallel_max_longitudinal_offset_q,
            self.hashed_max_line_length_q,
            self.hashed_corridor_half_width_q,
            self.hashed_max_consecutive_gap_q,
            self.hashed_endpoint_margin_q,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Candidate geometry thresholds must be finite and positive")
        if self.parallel_max_angle_degrees > 90.0:
            raise ValueError("Parallel angle gate must not exceed 90 degrees")
        if self.parallel_min_length_ratio > 1.0:
            raise ValueError("Parallel length ratio must be in (0, 1]")
        if self.hashed_max_members < 2:
            raise ValueError("Hashed candidate capacity must be at least two")


@dataclass(frozen=True, slots=True, order=True)
class TypedCandidateGroup:
    """One typed, unordered proposal subset."""

    group_type: int
    members: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.group_type not in PHYSICAL_TYPES:
            raise ValueError(f"Unsupported candidate group type: {self.group_type}")
        if not self.members or tuple(sorted(set(self.members))) != self.members:
            raise ValueError("Candidate members must be unique, sorted and nonempty")


@dataclass(frozen=True, slots=True)
class _LineGeometry:
    proposal_index: int
    center: np.ndarray
    unit: np.ndarray
    length: float


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _line_geometries(
    proposal_geometry: np.ndarray,
    proposal_types: np.ndarray,
    usable_mask: np.ndarray,
) -> list[_LineGeometry]:
    result = []
    for index in np.flatnonzero(usable_mask & (proposal_types == LINE)):
        points = proposal_geometry[index, :4].astype(np.float64).reshape(2, 2)
        points *= 2047.0
        vector = points[1] - points[0]
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length <= 1e-6:
            continue
        result.append(
            _LineGeometry(
                proposal_index=int(index),
                center=points.mean(axis=0),
                unit=vector / length,
                length=length,
            )
        )
    return result


def _parallel_compatible(
    left: _LineGeometry,
    right: _LineGeometry,
    thresholds: CandidateGeometryThresholds,
) -> bool:
    cosine = float(np.clip(abs(np.dot(left.unit, right.unit)), 0.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    if angle > thresholds.parallel_max_angle_degrees:
        return False
    length_ratio = min(left.length, right.length) / max(left.length, right.length)
    if length_ratio < thresholds.parallel_min_length_ratio:
        return False
    delta = right.center - left.center
    center_distance = float(np.linalg.norm(delta))
    mean_length = 0.5 * (left.length + right.length)
    if center_distance > thresholds.parallel_max_center_distance_q:
        return False
    if center_distance / max(mean_length, 1e-6) > (
        thresholds.parallel_max_center_distance_relative
    ):
        return False
    longitudinal = max(
        abs(float(np.dot(delta, left.unit))),
        abs(float(np.dot(delta, right.unit))),
    )
    return longitudinal <= thresholds.parallel_max_longitudinal_offset_q


def _connected_components(
    proposal_indices: Iterable[int],
    affinity: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    remaining = set(int(value) for value in proposal_indices)
    result = []
    while remaining:
        root = min(remaining)
        stack = [root]
        remaining.remove(root)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = [
                other
                for other in tuple(remaining)
                if bool(affinity[current, other])
            ]
            for other in neighbors:
                remaining.remove(other)
                stack.append(other)
        result.append(tuple(sorted(component)))
    return tuple(result)


def _hashed_windows(
    lines: list[_LineGeometry],
    affinity: np.ndarray,
    thresholds: CandidateGeometryThresholds,
) -> set[tuple[int, ...]]:
    by_index = {line.proposal_index: line for line in lines}
    active = [
        line
        for line in lines
        if bool(affinity[line.proposal_index].any())
    ]
    result: set[tuple[int, ...]] = set()
    for left_index in range(len(active)):
        for right_index in range(left_index + 1, len(active)):
            left = active[left_index].proposal_index
            right = active[right_index].proposal_index
            if bool(affinity[left, right]):
                result.add(tuple(sorted((left, right))))

    components = _connected_components(
        (line.proposal_index for line in active),
        affinity,
    )
    for component in components:
        if len(component) >= 2:
            result.add(component)

    distinct_orders: set[tuple[int, ...]] = set()
    for component in components:
        component_lines = [by_index[index] for index in component]
        for left_index in range(len(component_lines)):
            for right_index in range(left_index + 1, len(component_lines)):
                left = component_lines[left_index]
                right = component_lines[right_index]
                axis_vector = right.center - left.center
                span = float(np.linalg.norm(axis_vector))
                if span <= 1e-6:
                    continue
                axis = axis_vector / span
                normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
                ordered = []
                for line in component_lines:
                    delta = line.center - left.center
                    projection = float(np.dot(delta, axis))
                    perpendicular = abs(float(np.dot(delta, normal)))
                    if not (
                        -thresholds.hashed_endpoint_margin_q
                        <= projection
                        <= span + thresholds.hashed_endpoint_margin_q
                    ):
                        continue
                    if perpendicular > thresholds.hashed_corridor_half_width_q:
                        continue
                    ordered.append((projection, line.proposal_index, line.center))
                ordered.sort(key=lambda value: (value[0], value[1]))
                if len(ordered) >= 2:
                    distinct_orders.add(tuple(value[1] for value in ordered))

    for order in distinct_orders:
        run_start = 0
        for boundary in range(1, len(order) + 1):
            split = boundary == len(order)
            if not split:
                distance = float(
                    np.linalg.norm(
                        by_index[order[boundary]].center
                        - by_index[order[boundary - 1]].center
                    )
                )
                split = distance > thresholds.hashed_max_consecutive_gap_q
            if not split:
                continue
            run = order[run_start:boundary]
            for start in range(len(run)):
                maximum = min(
                    len(run),
                    start + thresholds.hashed_max_members,
                )
                for end in range(start + 2, maximum + 1):
                    result.add(tuple(sorted(run[start:end])))
            run_start = boundary
    return result


def _validate_pair_affinities(
    pair_affinities: dict[int, torch.Tensor | np.ndarray],
    proposals: int,
) -> dict[int, np.ndarray]:
    required = (PARALLEL_2, PARALLEL_3, HASHED_WEDGE)
    missing = [group_type for group_type in required if group_type not in pair_affinities]
    if missing:
        raise ValueError(f"Pair affinities are missing group types: {missing}")
    result = {}
    for group_type in required:
        value = _as_numpy(pair_affinities[group_type]).astype(bool, copy=False)
        if value.shape != (proposals, proposals):
            raise ValueError("Each pair affinity must have shape [proposals, proposals]")
        if not np.array_equal(value, value.T):
            raise ValueError("Pair affinities must be symmetric")
        if bool(np.diag(value).any()):
            raise ValueError("Pair affinities must have a zero diagonal")
        result[group_type] = value
    return result


def geometry_pair_affinities(
    proposal_geometry: torch.Tensor | np.ndarray,
    proposal_types: torch.Tensor | np.ndarray,
    usable_mask: torch.Tensor | np.ndarray,
    *,
    thresholds: CandidateGeometryThresholds = CandidateGeometryThresholds(),
) -> dict[int, np.ndarray]:
    """Build deployable pair affinities from frozen geometry only."""

    geometry = _as_numpy(proposal_geometry)
    types = _as_numpy(proposal_types).astype(np.int64, copy=False)
    usable = _as_numpy(usable_mask).astype(bool, copy=False)
    if geometry.ndim != 2 or geometry.shape[1] != 8:
        raise ValueError("Proposal geometry must have shape [proposals, 8]")
    proposals = geometry.shape[0]
    if types.shape != (proposals,) or usable.shape != (proposals,):
        raise ValueError("Proposal types/mask must align with geometry")
    lines = _line_geometries(geometry, types, usable)
    parallel = np.zeros((proposals, proposals), dtype=bool)
    hashed = np.zeros_like(parallel)
    for left_index, left in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            right = lines[right_index]
            if _parallel_compatible(left, right, thresholds):
                parallel[left.proposal_index, right.proposal_index] = True
                parallel[right.proposal_index, left.proposal_index] = True
            center_distance = float(np.linalg.norm(right.center - left.center))
            if (
                left.length <= thresholds.hashed_max_line_length_q
                and right.length <= thresholds.hashed_max_line_length_q
                and center_distance <= thresholds.hashed_max_consecutive_gap_q
            ):
                hashed[left.proposal_index, right.proposal_index] = True
                hashed[right.proposal_index, left.proposal_index] = True
    return {
        PARALLEL_2: parallel.copy(),
        PARALLEL_3: parallel,
        HASHED_WEDGE: hashed,
    }


def oracle_pair_affinities_from_targets(
    targets: Iterable[TypedCandidateGroup],
    proposal_types: torch.Tensor | np.ndarray,
    usable_mask: torch.Tensor | np.ndarray,
) -> dict[int, np.ndarray]:
    """Create the explicit decoder oracle: perfect pair labels, no candidates."""

    types = _as_numpy(proposal_types).astype(np.int64, copy=False)
    usable = _as_numpy(usable_mask).astype(bool, copy=False)
    if types.ndim != 1 or usable.shape != types.shape:
        raise ValueError("Oracle pair labels require aligned proposal types/mask")
    result = {
        group_type: np.zeros((len(types), len(types)), dtype=bool)
        for group_type in (PARALLEL_2, PARALLEL_3, HASHED_WEDGE)
    }
    ownership = set()
    for target in targets:
        if any(member in ownership for member in target.members):
            raise ValueError("Oracle pair targets must form a proposal partition")
        ownership.update(target.members)
        if any(not usable[member] for member in target.members):
            raise ValueError("Oracle pair target references an unusable proposal")
        if target.group_type not in result:
            continue
        for left_offset, left in enumerate(target.members):
            for right in target.members[left_offset + 1 :]:
                result[target.group_type][left, right] = True
                result[target.group_type][right, left] = True
    return result


def generate_typed_candidate_groups_from_affinities(
    proposal_geometry: torch.Tensor | np.ndarray,
    proposal_types: torch.Tensor | np.ndarray,
    usable_mask: torch.Tensor | np.ndarray,
    pair_affinities: dict[int, torch.Tensor | np.ndarray],
    *,
    thresholds: CandidateGeometryThresholds = CandidateGeometryThresholds(),
) -> tuple[TypedCandidateGroup, ...]:
    """Run the shared affinity-to-candidate decoder for either audit layer."""

    geometry = _as_numpy(proposal_geometry)
    types = _as_numpy(proposal_types).astype(np.int64, copy=False)
    usable = _as_numpy(usable_mask).astype(bool, copy=False)
    if geometry.ndim != 2 or geometry.shape[1] != 8:
        raise ValueError("Proposal geometry must have shape [proposals, 8]")
    if types.shape != (geometry.shape[0],) or usable.shape != (geometry.shape[0],):
        raise ValueError("Proposal types/mask must align with geometry")
    if not np.isfinite(geometry[usable]).all():
        raise ValueError("Usable proposal geometry contains non-finite values")
    affinities = _validate_pair_affinities(pair_affinities, geometry.shape[0])

    candidates: set[TypedCandidateGroup] = set()
    lines = _line_geometries(geometry, types, usable)
    for line in lines:
        candidates.add(TypedCandidateGroup(SINGLE_LINE, (line.proposal_index,)))
    line_indices = [line.proposal_index for line in lines]
    for left_offset, left in enumerate(line_indices):
        for right in line_indices[left_offset + 1 :]:
            if affinities[PARALLEL_2][left, right]:
                candidates.add(TypedCandidateGroup(PARALLEL_2, (left, right)))
    for first_offset, first in enumerate(line_indices):
        for second_offset in range(first_offset + 1, len(line_indices)):
            second = line_indices[second_offset]
            if not affinities[PARALLEL_3][first, second]:
                continue
            for third in line_indices[second_offset + 1 :]:
                if (
                    affinities[PARALLEL_3][first, third]
                    and affinities[PARALLEL_3][second, third]
                ):
                    candidates.add(
                        TypedCandidateGroup(PARALLEL_3, (first, second, third))
                    )
    for members in _hashed_windows(lines, affinities[HASHED_WEDGE], thresholds):
        candidates.add(TypedCandidateGroup(HASHED_WEDGE, members))
    for proposal_type, group_type in (
        (TRIANGLE, SOLID_TRIANGLE),
        (QUAD, SOLID_QUAD),
    ):
        for index in np.flatnonzero(usable & (types == proposal_type)):
            candidates.add(TypedCandidateGroup(group_type, (int(index),)))
    return tuple(sorted(candidates))


def generate_typed_candidate_groups(
    proposal_geometry: torch.Tensor | np.ndarray,
    proposal_types: torch.Tensor | np.ndarray,
    usable_mask: torch.Tensor | np.ndarray,
    *,
    thresholds: CandidateGeometryThresholds = CandidateGeometryThresholds(),
) -> tuple[TypedCandidateGroup, ...]:
    """Enumerate candidates without receiving GT group IDs or memberships."""
    affinities = geometry_pair_affinities(
        proposal_geometry,
        proposal_types,
        usable_mask,
        thresholds=thresholds,
    )
    return generate_typed_candidate_groups_from_affinities(
        proposal_geometry,
        proposal_types,
        usable_mask,
        affinities,
        thresholds=thresholds,
    )


def target_groups_for_sample(
    targets: dict[str, torch.Tensor],
    batch_index: int,
) -> tuple[TypedCandidateGroup, ...]:
    """Convert oracle labels only after candidate generation."""

    result = []
    for slot in torch.where(targets["group_mask"][batch_index].bool())[0].tolist():
        members = tuple(
            int(index)
            for index in torch.where(
                targets["group_membership"][batch_index, slot].bool()
            )[0].tolist()
        )
        result.append(
            TypedCandidateGroup(
                int(targets["group_types"][batch_index, slot]),
                members,
            )
        )
    return tuple(result)


def candidate_recall_counts(
    candidates: Iterable[TypedCandidateGroup],
    targets: Iterable[TypedCandidateGroup],
) -> dict[str, Any]:
    candidate_set = set(candidates)
    expected = Counter()
    recalled = Counter()
    missing = []
    for target in targets:
        expected[target.group_type] += 1
        if target in candidate_set:
            recalled[target.group_type] += 1
        else:
            missing.append(target)
    return {
        "expected": expected,
        "recalled": recalled,
        "missing": tuple(missing),
    }


def solve_typed_exact_cover(
    candidates: Iterable[TypedCandidateGroup],
    usable_proposals: Iterable[int],
    required_type_counts: Counter[int],
    *,
    time_limit_seconds: float = 5.0,
) -> dict[str, Any]:
    """Solve a typed exact cover without GT membership identities."""

    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0:
        raise ValueError("Exact-cover time limit must be finite and positive")
    proposal_ids = tuple(sorted(set(int(value) for value in usable_proposals)))
    proposal_row = {value: index for index, value in enumerate(proposal_ids)}
    values = tuple(
        candidate
        for candidate in candidates
        if all(member in proposal_row for member in candidate.members)
    )
    if not proposal_ids:
        feasible = sum(required_type_counts.values()) == 0
        return {
            "feasible": feasible,
            "status": 0 if feasible else 2,
            "message": "empty exact cover" if feasible else "nonzero type count",
            "selected": (),
        }
    if not values:
        return {
            "feasible": False,
            "status": 2,
            "message": "no candidates",
            "selected": (),
        }

    active_types = tuple(
        group_type
        for group_type in PHYSICAL_TYPES
        if int(required_type_counts.get(group_type, 0)) > 0
    )
    type_row = {
        group_type: len(proposal_ids) + offset
        for offset, group_type in enumerate(active_types)
    }
    rows = []
    columns = []
    data = []
    for column, candidate in enumerate(values):
        for member in candidate.members:
            rows.append(proposal_row[member])
            columns.append(column)
            data.append(1.0)
        if candidate.group_type in type_row:
            rows.append(type_row[candidate.group_type])
            columns.append(column)
            data.append(1.0)
    matrix = coo_matrix(
        (data, (rows, columns)),
        shape=(len(proposal_ids) + len(active_types), len(values)),
        dtype=np.float64,
    ).tocsr()
    target = np.concatenate(
        (
            np.ones(len(proposal_ids), dtype=np.float64),
            np.asarray(
                [required_type_counts[group_type] for group_type in active_types],
                dtype=np.float64,
            ),
        )
    )
    result = milp(
        c=np.zeros(len(values), dtype=np.float64),
        integrality=np.ones(len(values), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, target, target),
        options={"time_limit": float(time_limit_seconds)},
    )
    selected: tuple[TypedCandidateGroup, ...] = ()
    feasible = False
    if result.x is not None:
        indices = np.flatnonzero(result.x >= 0.5)
        residual = np.asarray(matrix[:, indices].sum(axis=1)).reshape(-1) - target
        feasible = bool(np.max(np.abs(residual), initial=0.0) <= 1e-6)
        if feasible:
            selected = tuple(values[int(index)] for index in indices)
    return {
        "feasible": feasible,
        "status": int(result.status),
        "message": str(result.message),
        "selected": selected,
    }


def _counter_payload(counter: Counter[int]) -> dict[str, int]:
    return {
        PHYSICAL_GROUP_NAMES[group_type]: int(counter.get(group_type, 0))
        for group_type in PHYSICAL_TYPES
    }


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0.0, "p95": 0.0, "max": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
    }


def _new_layer_state() -> dict[str, Any]:
    return {
        "expected": Counter(),
        "recalled": Counter(),
        "candidate_types": Counter(),
        "candidate_counts": [],
        "gt_partition_feasible": 0,
        "exact_cover_feasible": 0,
        "solver_calls": 0,
        "solver_timeouts": 0,
    }


def _evaluate_candidate_layer(
    candidates: tuple[TypedCandidateGroup, ...],
    target_groups: tuple[TypedCandidateGroup, ...],
    usable: torch.Tensor,
    state: dict[str, Any],
    *,
    exact_cover_time_limit: float,
) -> dict[str, Any]:
    recall = candidate_recall_counts(candidates, target_groups)
    candidate_by_type = Counter(candidate.group_type for candidate in candidates)
    state["expected"].update(recall["expected"])
    state["recalled"].update(recall["recalled"])
    state["candidate_types"].update(candidate_by_type)
    state["candidate_counts"].append(len(candidates))
    gt_certificate = not recall["missing"]
    if gt_certificate:
        state["gt_partition_feasible"] += 1
        feasible = True
        solver_status = "gt_partition_certificate"
    else:
        state["solver_calls"] += 1
        solution = solve_typed_exact_cover(
            candidates,
            torch.where(usable)[0].tolist(),
            recall["expected"],
            time_limit_seconds=exact_cover_time_limit,
        )
        feasible = bool(solution["feasible"])
        solver_status = str(solution["message"])
        if int(solution["status"]) == 1 and not feasible:
            state["solver_timeouts"] += 1
    state["exact_cover_feasible"] += int(feasible)
    return {
        "candidate_count": len(candidates),
        "candidate_count_by_type": _counter_payload(candidate_by_type),
        "target_count_by_type": _counter_payload(recall["expected"]),
        "recalled_count_by_type": _counter_payload(recall["recalled"]),
        "missing_targets": [
            {
                "kind": PHYSICAL_GROUP_NAMES[target.group_type],
                "members": list(target.members),
            }
            for target in recall["missing"]
        ],
        "gt_partition_candidate_exact_cover": gt_certificate,
        "oracle_typed_count_exact_cover_feasible": feasible,
        "exact_cover_status": solver_status,
    }


def _finalize_layer_state(state: dict[str, Any], samples: int) -> dict[str, Any]:
    expected_all = sum(state["expected"].values())
    recalled_all = sum(state["recalled"].values())
    return {
        "candidate_groups": int(sum(state["candidate_counts"])),
        "candidate_count_per_image": _distribution(state["candidate_counts"]),
        "candidate_count_by_type": _counter_payload(state["candidate_types"]),
        "target_groups": int(expected_all),
        "recalled_target_groups": int(recalled_all),
        "candidate_recall": recalled_all / max(expected_all, 1),
        "candidate_recall_by_type": {
            PHYSICAL_GROUP_NAMES[group_type]: (
                state["recalled"][group_type] / state["expected"][group_type]
                if state["expected"][group_type]
                else None
            )
            for group_type in PHYSICAL_TYPES
        },
        "expected_by_type": _counter_payload(state["expected"]),
        "recalled_by_type": _counter_payload(state["recalled"]),
        "gt_partition_candidate_exact_cover_samples": state[
            "gt_partition_feasible"
        ],
        "gt_partition_candidate_exact_cover_rate": state[
            "gt_partition_feasible"
        ]
        / samples,
        "oracle_typed_count_exact_cover_feasible_samples": state[
            "exact_cover_feasible"
        ],
        "oracle_typed_count_exact_cover_feasible_rate": state[
            "exact_cover_feasible"
        ]
        / samples,
        "alternative_milp_calls": state["solver_calls"],
        "alternative_milp_timeouts": state["solver_timeouts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", nargs="+", default=[DEFAULT_CAPACITY_SHARD])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--primitive-checkpoint",
        default=str(SOURCE_PRIMITIVE_CHECKPOINT),
    )
    parser.add_argument(
        "--primitive-checkpoint-sha256",
        default=SOURCE_PRIMITIVE_SHA256,
    )
    parser.add_argument("--exact-cover-time-limit", type=float, default=5.0)
    return parser


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_results(
    output_dir: Path,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("Batch/image size must be positive and workers nonnegative")
    if args.exact_cover_time_limit <= 0.0:
        raise ValueError("--exact-cover-time-limit must be positive")
    shards = expand_shards(args.shard)
    device = _device(args.device)
    primitive_model, _, provenance = load_frozen_primitive_checkpoint(
        Path(args.primitive_checkpoint),
        expected_sha256=args.primitive_checkpoint_sha256,
        device=device,
    )
    collator = BondGroupProposalMembershipCollator(image_size=args.image_size)
    loader = DataLoader(
        VisualSVGWebDataset(shards, limit=args.limit),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    thresholds = CandidateGeometryThresholds()
    layer_states = {
        "geometry_only_deployable": _new_layer_state(),
        "perfect_pair_affinity_decoder_oracle": _new_layer_state(),
    }
    rows = []
    key_hash = hashlib.sha256()

    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                reject_internal_test_metadata(meta, str(key))
                key_hash.update(f"{key}\n".encode("utf-8"))
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_context(device):
                flat_output = primitive_model(pixels)
            proposal_batch = collator.attach_flat_output(batch, flat_output)
            targets = aggregate_group_slot_targets(proposal_batch, group_slots=192)
            proposal_mask = proposal_batch["proposal_mask"].bool()
            usable = proposal_mask & targets["proposal_keep"].bool()
            for batch_index, key in enumerate(batch["keys"]):
                geometry_candidates = generate_typed_candidate_groups(
                    proposal_batch["proposal_geometry"][batch_index],
                    proposal_batch["proposal_type"][batch_index],
                    usable[batch_index],
                    thresholds=thresholds,
                )
                target_groups = target_groups_for_sample(targets, batch_index)
                oracle_affinities = oracle_pair_affinities_from_targets(
                    target_groups,
                    proposal_batch["proposal_type"][batch_index],
                    usable[batch_index],
                )
                oracle_candidates = generate_typed_candidate_groups_from_affinities(
                    proposal_batch["proposal_geometry"][batch_index],
                    proposal_batch["proposal_type"][batch_index],
                    usable[batch_index],
                    oracle_affinities,
                    thresholds=thresholds,
                )
                rows.append(
                    {
                        "sample_key": str(key),
                        "oracle_usable_proposals": int(usable[batch_index].sum()),
                        "geometry_only_deployable": _evaluate_candidate_layer(
                            geometry_candidates,
                            target_groups,
                            usable[batch_index],
                            layer_states["geometry_only_deployable"],
                            exact_cover_time_limit=args.exact_cover_time_limit,
                        ),
                        "perfect_pair_affinity_decoder_oracle": (
                            _evaluate_candidate_layer(
                                oracle_candidates,
                                target_groups,
                                usable[batch_index],
                                layer_states[
                                    "perfect_pair_affinity_decoder_oracle"
                                ],
                                exact_cover_time_limit=args.exact_cover_time_limit,
                            )
                        ),
                    }
                )

    samples = len(rows)
    if not samples:
        raise RuntimeError("Typed candidate audit read no samples")
    metrics = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": "capacity100_frozen_flat_gt_associated_oracle_usable_pool",
        "shared_decoder": "typed_pair_affinity_to_candidate_generator",
        "geometry_only_deployable_input_contract": [
            "frozen_proposal_geometry_to_pair_affinity",
            "frozen_proposal_primitive_type",
            "oracle_usable_proposal_mask",
        ],
        "geometry_only_gt_not_used_for_candidate_generation": True,
        "perfect_pair_affinity_decoder_oracle_input_contract": [
            "perfect_gt_pair_affinity",
            "frozen_proposal_geometry_for_hashed_ordering",
            "frozen_proposal_primitive_type",
            "oracle_usable_proposal_mask",
        ],
        "perfect_pair_affinity_decoder_oracle_is_deployable": False,
        "gt_usage": [
            "both_layers_offline_candidate_recall_labels",
            "both_layers_offline_exact_cover_certificate",
            "both_layers_typed_count_constraints_only_for_alternative_milp",
            "decoder_oracle_perfect_pair_affinity_only",
        ],
        "hashed_member_count_scope": "oracle_usable_support_membership_only",
        "thresholds_q2047": asdict(thresholds),
        "proposal_pool": {
            "decode_mode": collator.decode_mode,
            "proposal_threshold": collator.proposal_threshold,
            "high_confidence_tolerance_q": collator.high_confidence_tolerance_q,
            "association_tolerance_q": collator.association_tolerance_q,
            "shell_weight": collator.shell_weight,
        },
        "source_primitive": provenance,
        "shards": [str(Path(shard).resolve()) for shard in shards],
        "sample_key_sha256": key_hash.hexdigest(),
        "samples": samples,
        "layers": {
            name: _finalize_layer_state(state, samples)
            for name, state in layer_states.items()
        },
        "uses_internal_test": False,
    }
    _write_results(Path(args.output_dir), metrics, rows)
    print(json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
