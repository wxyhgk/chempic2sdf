#!/usr/bin/env python3
"""Audit deterministic raster evidence as carriers for Visual SVG bond groups.

This is a read-only capacity audit.  OpenCV LSD supplies raster-anchored line
segments; no target geometry is used to create proposals.  Hungarian matching
is used only after proposal generation to measure the best available coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    import cv2
except ImportError as error:  # pragma: no cover - exercised by the CLI environment
    raise RuntimeError(
        "audit_evidence_carrier_proposals requires opencv-python-headless"
    ) from error


AUDIT_SCHEMA_VERSION = "evidence-carrier-proposal-audit-0.1"
COORDINATE_MAX = 2047.0
GROUP_KINDS = (
    "single_line",
    "parallel_2",
    "parallel_3",
    "hashed_wedge",
    "solid_triangle",
    "solid_quad",
)
DEFAULT_TOLERANCES_Q = (8.0, 16.0, 32.0)


@dataclass(frozen=True, slots=True)
class TargetLine:
    group_index: int
    group_kind: str
    points_q: tuple[tuple[float, float], tuple[float, float]]
    long_group: bool
    text_adjacent: bool


@dataclass(frozen=True, slots=True)
class TargetGroup:
    group_index: int
    kind: str
    line_indices: tuple[int, ...]
    long_group: bool
    text_adjacent: bool


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_kind(group: dict[str, Any], primitives: dict[int, dict[str, Any]]) -> str:
    members = [primitives[int(index)] for index in group.get("primitive_ids", [])]
    if len(members) == 1 and members[0].get("kind") == "triangle":
        return "solid_triangle"
    if len(members) == 1 and members[0].get("kind") == "quad":
        return "solid_quad"
    if str(group.get("direction", "")) == "BEGINDASH":
        return "hashed_wedge"
    if members and all(member.get("kind") == "line" for member in members):
        return {1: "single_line", 2: "parallel_2", 3: "parallel_3"}.get(
            len(members), "unsupported_line_group"
        )
    return "unsupported_group"


def extract_targets(
    primitives_payload: dict[str, Any],
    layers: dict[str, Any],
    *,
    long_bond_pixels: float,
) -> tuple[list[TargetLine], list[TargetGroup], Counter[str]]:
    """Build line targets and group slices without canonical reordering."""

    primitives = {
        int(item["primitive_id"]): item
        for item in primitives_payload.get("primitives", [])
    }
    layer_bonds = {
        int(item["bond_id"]): item for item in layers.get("bonds", [])
    }
    visible_atoms = {
        int(item["atom_id"])
        for item in layers.get("atoms", [])
        if bool(item.get("visible_text"))
    }
    target_lines: list[TargetLine] = []
    target_groups: list[TargetGroup] = []
    nonline = Counter()
    for group_index, group in enumerate(primitives_payload.get("bond_groups", [])):
        kind = _group_kind(group, primitives)
        bond = layer_bonds.get(int(group.get("bond_id", group_index)), {})
        begin = bond.get("begin")
        end = bond.get("end")
        logical_length = (
            math.dist(begin, end)
            if isinstance(begin, list)
            and isinstance(end, list)
            and len(begin) == len(end) == 2
            else 0.0
        )
        long_group = logical_length >= long_bond_pixels
        text_adjacent = (
            int(group.get("begin_atom", -1)) in visible_atoms
            or int(group.get("end_atom", -1)) in visible_atoms
        )
        line_indices = []
        for primitive_id in group.get("primitive_ids", []):
            primitive = primitives[int(primitive_id)]
            if primitive.get("kind") != "line":
                continue
            points = primitive.get("quantized_points")
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError("Line primitive lacks two quantized points")
            line_indices.append(len(target_lines))
            target_lines.append(
                TargetLine(
                    group_index=group_index,
                    group_kind=kind,
                    points_q=(
                        (float(points[0][0]), float(points[0][1])),
                        (float(points[1][0]), float(points[1][1])),
                    ),
                    long_group=long_group,
                    text_adjacent=text_adjacent,
                )
            )
        if not line_indices:
            nonline[kind] += 1
        target_groups.append(
            TargetGroup(
                group_index=group_index,
                kind=kind,
                line_indices=tuple(line_indices),
                long_group=long_group,
                text_adjacent=text_adjacent,
            )
        )
    return target_lines, target_groups, nonline


def _canonical_line(line: Iterable[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in line)
    if (y0, x0) > (y1, x1):
        return x1, y1, x0, y0
    return x0, y0, x1, y1


def detect_lsd_lines(
    image: Image.Image,
    *,
    oracle_bond_layer: bool,
    minimum_length_pixels: float,
) -> np.ndarray:
    """Return target-independent LSD segments in image pixel coordinates."""

    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = grayscale if oracle_bond_layer else 255 - grayscale
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(ink)[0]
    if detected is None:
        return np.empty((0, 2, 2), dtype=np.float64)
    result = []
    for raw in detected:
        line = _canonical_line(np.asarray(raw).reshape(-1)[:4])
        if math.dist(line[:2], line[2:]) < minimum_length_pixels:
            continue
        result.append(((line[0], line[1]), (line[2], line[3])))
    if not result:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def dense_local_expansion(proposals: np.ndarray) -> np.ndarray:
    """Create a bounded local carrier bank around each raster LSD segment.

    The fixed offsets are target independent.  They model a small reference
    refiner that can center a stroke and move either endpoint along its carrier.
    """

    if not len(proposals):
        return np.empty((0, 2, 2), dtype=np.float64)
    normal_offsets = (-1.0, 0.0, 1.0)
    endpoint_extensions = (
        (-2.0, -2.0),
        (0.0, 0.0),
        (2.0, 2.0),
        (4.0, 4.0),
        (8.0, 8.0),
        (0.0, 2.0),
        (2.0, 0.0),
        (0.0, 4.0),
        (4.0, 0.0),
        (0.0, 8.0),
        (8.0, 0.0),
        (4.0, 8.0),
        (8.0, 4.0),
    )
    expanded: dict[tuple[int, ...], np.ndarray] = {}
    for proposal in proposals:
        delta = proposal[1] - proposal[0]
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            continue
        unit = delta / length
        normal = np.asarray((-unit[1], unit[0]), dtype=np.float64)
        for normal_offset in normal_offsets:
            shift = normal * normal_offset
            for begin_extension, end_extension in endpoint_extensions:
                candidate = np.stack(
                    (
                        proposal[0] - unit * begin_extension + shift,
                        proposal[1] + unit * end_extension + shift,
                    )
                )
                canonical = np.asarray(_canonical_line(candidate.reshape(-1))).reshape(2, 2)
                key = tuple(np.rint(canonical.reshape(-1) * 4.0).astype(int).tolist())
                expanded.setdefault(key, canonical)
    return np.stack(tuple(expanded.values())) if expanded else np.empty((0, 2, 2))


def proposals_pixels_to_q(
    proposals: np.ndarray, width: int, height: int
) -> np.ndarray:
    if not len(proposals):
        return np.empty((0, 2, 2), dtype=np.float64)
    scale = np.asarray((COORDINATE_MAX / width, COORDINATE_MAX / height))
    return proposals * scale


def line_direction_support_matrix(
    proposals: np.ndarray,
    target_lines: list[TargetLine],
    *,
    width: int,
    height: int,
    maximum_angle_degrees: float,
    maximum_perpendicular_pixels: float,
    minimum_overlap_pixels: float,
) -> np.ndarray:
    """Test whether proposals carry a target line's direction and local ink.

    Endpoints may be incomplete.  The criterion asks only for a nearly
    collinear proposal with real longitudinal overlap, which is the evidence a
    dynamic carrier/reference refiner can use to infer the full extent.
    """

    if not len(proposals) or not target_lines:
        return np.zeros((len(proposals), len(target_lines)), dtype=bool)
    scale = np.asarray((width / COORDINATE_MAX, height / COORDINATE_MAX))
    targets = np.asarray([target.points_q for target in target_lines]) * scale
    target_delta = targets[:, 1] - targets[:, 0]
    target_length = np.linalg.norm(target_delta, axis=-1)
    target_unit = target_delta / np.maximum(target_length[:, None], 1e-9)
    target_normal = np.stack((-target_unit[:, 1], target_unit[:, 0]), axis=-1)

    proposal_delta = proposals[:, 1] - proposals[:, 0]
    proposal_length = np.linalg.norm(proposal_delta, axis=-1)
    proposal_unit = proposal_delta / np.maximum(proposal_length[:, None], 1e-9)
    cosine = np.abs(proposal_unit @ target_unit.T)
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

    proposal_midpoint = proposals.mean(axis=1)
    midpoint_delta = proposal_midpoint[:, None, :] - targets[None, :, 0, :]
    perpendicular = np.abs(
        np.einsum("ptd,td->pt", midpoint_delta, target_normal)
    )
    begin_projection = np.einsum(
        "ptd,td->pt",
        proposals[:, None, 0, :] - targets[None, :, 0, :],
        target_unit,
    )
    end_projection = np.einsum(
        "ptd,td->pt",
        proposals[:, None, 1, :] - targets[None, :, 0, :],
        target_unit,
    )
    lower = np.minimum(begin_projection, end_projection)
    upper = np.maximum(begin_projection, end_projection)
    overlap = np.maximum(
        0.0,
        np.minimum(upper, target_length[None, :]) - np.maximum(lower, 0.0),
    )
    required_overlap = np.minimum(
        minimum_overlap_pixels, target_length * 0.25
    )[None, :]
    return (
        (angle <= maximum_angle_degrees)
        & (perpendicular <= maximum_perpendicular_pixels)
        & (overlap >= required_overlap)
    )


def evaluate_direction_support(
    proposals: np.ndarray,
    target_lines: list[TargetLine],
    target_groups: list[TargetGroup],
    *,
    width: int,
    height: int,
    maximum_angle_degrees: float,
    maximum_perpendicular_pixels: float,
    minimum_overlap_pixels: float,
) -> dict[str, Any]:
    support = line_direction_support_matrix(
        proposals,
        target_lines,
        width=width,
        height=height,
        maximum_angle_degrees=maximum_angle_degrees,
        maximum_perpendicular_pixels=maximum_perpendicular_pixels,
        minimum_overlap_pixels=minimum_overlap_pixels,
    )

    def group_match_count(group: TargetGroup) -> int:
        if not group.line_indices:
            return 0
        group_cost = (~support[:, list(group.line_indices)]).astype(np.uint8)
        return len(one_to_one_matches(group_cost, 0.0))

    by_kind = {}
    for kind in ("single_line", "parallel_2", "parallel_3", "hashed_wedge"):
        indices = [
            index for index, target in enumerate(target_lines) if target.group_kind == kind
        ]
        supported = int(support[:, indices].any(axis=0).sum()) if indices else 0
        groups = [
            group
            for group in target_groups
            if group.kind == kind and group.line_indices
        ]
        match_counts = [group_match_count(group) for group in groups]
        by_kind[kind] = {
            "line_members": {"supported": supported, "expected": len(indices)},
            "groups": {
                "supported": sum(
                    count == len(group.line_indices)
                    for count, group in zip(match_counts, groups)
                ),
                "expected": len(groups),
            },
            "carrier_groups": {
                "supported": sum(
                    count >= (2 if kind == "hashed_wedge" else len(group.line_indices))
                    for count, group in zip(match_counts, groups)
                ),
                "expected": len(groups),
            },
        }

    slices = {}
    for name, predicate in (
        ("long_at_least_96px", lambda group: group.long_group),
        ("text_adjacent", lambda group: group.text_adjacent),
        ("not_text_adjacent", lambda group: not group.text_adjacent),
    ):
        groups = [
            group
            for group in target_groups
            if group.line_indices
            and group.kind != "hashed_wedge"
            and predicate(group)
        ]
        slices[name] = {
            "supported": sum(
                group_match_count(group) == len(group.line_indices)
                for group in groups
            ),
            "expected": len(groups),
        }
    return {"by_group_kind": by_kind, "ordinary_group_slices": slices}


def line_distance_matrix(
    proposals_q: np.ndarray, targets: list[TargetLine] | np.ndarray
) -> np.ndarray:
    if isinstance(targets, list):
        target_array = np.asarray([target.points_q for target in targets], dtype=np.float64)
    else:
        target_array = np.asarray(targets, dtype=np.float64)
    if not len(proposals_q) or not len(target_array):
        return np.empty((len(proposals_q), len(target_array)), dtype=np.float64)
    forward = np.linalg.norm(
        proposals_q[:, None, :, :] - target_array[None, :, :, :], axis=-1
    ).max(axis=-1)
    reverse = np.linalg.norm(
        proposals_q[:, None, :, :] - target_array[None, :, ::-1, :], axis=-1
    ).max(axis=-1)
    return np.minimum(forward, reverse)


def one_to_one_matches(cost: np.ndarray, tolerance_q: float) -> dict[int, int]:
    """Map targets to proposals while maximizing in-tolerance cardinality."""

    if not cost.size:
        return {}
    # Thresholding after a minimum-distance assignment can sacrifice a valid
    # edge to reduce the sum of several invalid distances.  Binary assignment
    # first maximizes the number of valid edges; geometry is not used to create
    # or score proposals anywhere before this audit-only oracle.
    binary_cost = (cost > tolerance_q).astype(np.uint8)
    proposal_indices, target_indices = linear_sum_assignment(binary_cost)
    return {
        int(target): int(proposal)
        for proposal, target in zip(proposal_indices, target_indices)
        if not bool(binary_cost[proposal, target])
    }


def _prf(matched: int, predicted: int, expected: int) -> dict[str, float | int]:
    precision = matched / max(predicted, 1)
    recall = matched / max(expected, 1)
    return {
        "matched": matched,
        "predicted": predicted,
        "expected": expected,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def evaluate_proposals(
    proposals_q: np.ndarray,
    target_lines: list[TargetLine],
    target_groups: list[TargetGroup],
    tolerances_q: tuple[float, ...],
) -> dict[str, Any]:
    cost = line_distance_matrix(proposals_q, target_lines)
    modes: dict[str, Any] = {}
    for tolerance in tolerances_q:
        matches = one_to_one_matches(cost, tolerance)
        matched_targets = set(matches)
        structural_indices = [
            index
            for index, target in enumerate(target_lines)
            if target.group_kind != "hashed_wedge"
        ]
        structural_cost = cost[:, structural_indices] if structural_indices else cost[:, :0]
        structural_matches = one_to_one_matches(structural_cost, tolerance)

        by_kind = {}
        for kind in ("single_line", "parallel_2", "parallel_3", "hashed_wedge"):
            indices = [
                index for index, target in enumerate(target_lines) if target.group_kind == kind
            ]
            kind_matches = one_to_one_matches(cost[:, indices], tolerance) if indices else {}
            by_kind[kind] = _prf(len(kind_matches), len(proposals_q), len(indices))

        group_rows = {}
        group_feasible: dict[int, tuple[bool, bool]] = {}
        for kind in ("single_line", "parallel_2", "parallel_3", "hashed_wedge"):
            groups = [
                group
                for group in target_groups
                if group.kind == kind and group.line_indices
            ]
            strict = 0
            carrier = 0
            for group in groups:
                group_cost = cost[:, list(group.line_indices)]
                count = len(one_to_one_matches(group_cost, tolerance))
                strict_ok = count == len(group.line_indices)
                required = 2 if kind == "hashed_wedge" else len(group.line_indices)
                carrier_ok = count >= required
                strict += int(strict_ok)
                carrier += int(carrier_ok)
                group_feasible[group.group_index] = (strict_ok, carrier_ok)
            group_rows[kind] = {
                "expected": len(groups),
                "strict_all_members": strict,
                "strict_all_members_recall": strict / max(len(groups), 1),
                "carrier_eligible": carrier,
                "carrier_eligible_recall": carrier / max(len(groups), 1),
            }

        def group_slice(predicate) -> dict[str, float | int]:
            groups = [
                group
                for group in target_groups
                if group.line_indices and group.kind != "hashed_wedge" and predicate(group)
            ]
            covered = sum(
                group_feasible.get(group.group_index, (False, False))[0]
                for group in groups
            )
            return {
                "expected": len(groups),
                "covered": covered,
                "recall": covered / max(len(groups), 1),
            }

        modes[f"{tolerance:g}q"] = {
            "all_line_members": _prf(
                len(matches), len(proposals_q), len(target_lines)
            ),
            "excluding_hashed_members": _prf(
                len(structural_matches), len(proposals_q), len(structural_indices)
            ),
            "member_by_group_kind": by_kind,
            "group_coverage": group_rows,
            "ordinary_group_slices": {
                "long_at_least_96px": group_slice(lambda group: group.long_group),
                "text_adjacent": group_slice(lambda group: group.text_adjacent),
                "not_text_adjacent": group_slice(lambda group: not group.text_adjacent),
            },
            "sample_exact": {
                "all_line_members": len(matched_targets) == len(target_lines),
                "excluding_hashed_members": len(structural_matches)
                == len(structural_indices),
            },
        }
    return modes


def translate_grayscale(
    image: Image.Image, *, dx: int, dy: int, oracle_bond_layer: bool
) -> Image.Image:
    if dx < 0 or dy < 0:
        raise ValueError("Translation audit currently requires nonnegative shifts")
    source = np.asarray(image.convert("L"), dtype=np.uint8)
    fill = 0 if oracle_bond_layer else 255
    translated = np.full_like(source, fill)
    if dx < source.shape[1] and dy < source.shape[0]:
        translated[dy:, dx:] = source[: source.shape[0] - dy, : source.shape[1] - dx]
    return Image.fromarray(translated, mode="L")


def translated_proposals(
    proposals: np.ndarray, *, dx: float, dy: float, width: int, height: int
) -> np.ndarray:
    shifted = proposals + np.asarray((dx, dy), dtype=np.float64)
    if not len(shifted):
        return shifted
    valid = (
        (shifted[..., 0] >= 0.0)
        & (shifted[..., 0] < width)
        & (shifted[..., 1] >= 0.0)
        & (shifted[..., 1] < height)
    ).all(axis=1)
    return shifted[valid]


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _empty_aggregate(tolerances_q: tuple[float, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": 0,
        "proposal_counts": [],
        "proposal_lengths_pixels": [],
        "tolerances": {},
    }
    for tolerance in tolerances_q:
        result["tolerances"][f"{tolerance:g}q"] = {
            "all": Counter(),
            "structural": Counter(),
            "by_kind": defaultdict(Counter),
            "groups": defaultdict(Counter),
            "slices": defaultdict(Counter),
            "sample_exact_all": 0,
            "sample_exact_structural": 0,
        }
    return result


def _accumulate(aggregate: dict[str, Any], proposals: np.ndarray, row: dict[str, Any]) -> None:
    aggregate["samples"] += 1
    aggregate["proposal_counts"].append(float(len(proposals)))
    aggregate["proposal_lengths_pixels"].extend(
        np.linalg.norm(proposals[:, 1] - proposals[:, 0], axis=-1).tolist()
        if len(proposals)
        else []
    )
    for tolerance, metrics in row.items():
        target = aggregate["tolerances"][tolerance]
        target["all"].update(
            {
                key: int(metrics["all_line_members"][key])
                for key in ("matched", "predicted", "expected")
            }
        )
        target["structural"].update(
            {
                key: int(metrics["excluding_hashed_members"][key])
                for key in ("matched", "predicted", "expected")
            }
        )
        for kind, values in metrics["member_by_group_kind"].items():
            target["by_kind"][kind].update(
                {
                    key: int(values[key])
                    for key in ("matched", "predicted", "expected")
                }
            )
        for kind, values in metrics["group_coverage"].items():
            target["groups"][kind].update(
                {
                    key: int(values[key])
                    for key in ("expected", "strict_all_members", "carrier_eligible")
                }
            )
        for name, values in metrics["ordinary_group_slices"].items():
            target["slices"][name].update(
                {key: int(values[key]) for key in ("expected", "covered")}
            )
        target["sample_exact_all"] += int(metrics["sample_exact"]["all_line_members"])
        target["sample_exact_structural"] += int(
            metrics["sample_exact"]["excluding_hashed_members"]
        )


def _summarize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    samples = max(int(aggregate["samples"]), 1)
    result = {
        "samples": int(aggregate["samples"]),
        "proposal_count": _distribution(aggregate["proposal_counts"]),
        "proposal_length_pixels": _distribution(aggregate["proposal_lengths_pixels"]),
        "tolerances": {},
    }
    for tolerance, values in aggregate["tolerances"].items():
        groups = {}
        for kind, counts in values["groups"].items():
            expected = int(counts["expected"])
            groups[kind] = {
                "expected": expected,
                "strict_all_members": int(counts["strict_all_members"]),
                "strict_all_members_recall": counts["strict_all_members"]
                / max(expected, 1),
                "carrier_eligible": int(counts["carrier_eligible"]),
                "carrier_eligible_recall": counts["carrier_eligible"]
                / max(expected, 1),
            }
        slices = {}
        for name, counts in values["slices"].items():
            expected = int(counts["expected"])
            slices[name] = {
                "expected": expected,
                "covered": int(counts["covered"]),
                "recall": counts["covered"] / max(expected, 1),
            }
        result["tolerances"][tolerance] = {
            "all_line_members": _prf(
                int(values["all"]["matched"]),
                int(values["all"]["predicted"]),
                int(values["all"]["expected"]),
            ),
            "excluding_hashed_members": _prf(
                int(values["structural"]["matched"]),
                int(values["structural"]["predicted"]),
                int(values["structural"]["expected"]),
            ),
            "member_by_group_kind": {
                kind: _prf(
                    int(counts["matched"]),
                    int(counts["predicted"]),
                    int(counts["expected"]),
                )
                for kind, counts in sorted(values["by_kind"].items())
            },
            "group_coverage": dict(sorted(groups.items())),
            "ordinary_group_slices": dict(sorted(slices.items())),
            "sample_exact": {
                "all_line_members": values["sample_exact_all"] / samples,
                "excluding_hashed_members": values["sample_exact_structural"] / samples,
            },
        }
    return result


def _not_internal(meta: dict[str, Any], shard: Path, sample_key: str) -> None:
    split = str(meta.get("split", meta.get("source_split", ""))).lower().replace(
        "-", "_"
    )
    if split == "internal_test" or "internal_test" in str(shard).lower().replace("-", "_"):
        raise RuntimeError(f"Refusing sealed internal_test sample: {sample_key}")


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"Missing tar member: {name}")
    return handle.read()


def _translation_accumulator(tolerances_q: tuple[float, ...]) -> dict[str, Counter]:
    return {f"{value:g}q": Counter() for value in tolerances_q}


def _direction_accumulator() -> dict[str, Any]:
    return {
        "by_group_kind": defaultdict(
            lambda: {
                "line_members": Counter(),
                "groups": Counter(),
                "carrier_groups": Counter(),
            }
        ),
        "ordinary_group_slices": defaultdict(Counter),
    }


def _accumulate_direction(accumulator: dict[str, Any], row: dict[str, Any]) -> None:
    for kind, values in row["by_group_kind"].items():
        for level in ("line_members", "groups", "carrier_groups"):
            accumulator["by_group_kind"][kind][level].update(
                {
                    key: int(values[level][key])
                    for key in ("supported", "expected")
                }
            )
    for name, values in row["ordinary_group_slices"].items():
        accumulator["ordinary_group_slices"][name].update(
            {
                key: int(values[key]) for key in ("supported", "expected")
            }
        )


def _support_summary(values: Counter) -> dict[str, float | int]:
    supported = int(values["supported"])
    expected = int(values["expected"])
    return {
        "supported": supported,
        "expected": expected,
        "recall": supported / max(expected, 1),
    }


def _summarize_direction(accumulator: dict[str, Any]) -> dict[str, Any]:
    return {
        "by_group_kind": {
            kind: {
                level: _support_summary(values[level])
                for level in ("line_members", "groups", "carrier_groups")
            }
            for kind, values in sorted(accumulator["by_group_kind"].items())
        },
        "ordinary_group_slices": {
            name: _support_summary(values)
            for name, values in sorted(
                accumulator["ordinary_group_slices"].items()
            )
        },
    }


def _accumulate_equivariance(
    accumulator: dict[str, Counter],
    expected_shifted_q: np.ndarray,
    detected_shifted_q: np.ndarray,
    tolerances_q: tuple[float, ...],
) -> None:
    cost = line_distance_matrix(detected_shifted_q, expected_shifted_q)
    for tolerance in tolerances_q:
        matched = len(one_to_one_matches(cost, tolerance))
        accumulator[f"{tolerance:g}q"].update(
            {
                "matched": matched,
                "predicted": len(detected_shifted_q),
                "expected": len(expected_shifted_q),
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--minimum-length-pixels", type=float, default=1.0)
    parser.add_argument("--long-bond-pixels", type=float, default=96.0)
    parser.add_argument("--tolerance-q", action="append", type=float)
    parser.add_argument("--translation-samples", type=int, default=25)
    parser.add_argument("--translation-dx", type=int, default=7)
    parser.add_argument("--translation-dy", type=int, default=11)
    parser.add_argument("--direction-angle-degrees", type=float, default=5.0)
    parser.add_argument("--direction-perpendicular-pixels", type=float, default=3.0)
    parser.add_argument("--direction-overlap-pixels", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    shard = Path(args.shard).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    tolerances_q = tuple(
        sorted(set(args.tolerance_q or DEFAULT_TOLERANCES_Q))
    )
    if not shard.is_file():
        raise FileNotFoundError(shard)
    if output.exists():
        raise RuntimeError(f"Audit output already exists: {output}")
    if min(args.limit, args.translation_samples) < 0:
        raise ValueError("Sample limits must be nonnegative")
    if args.minimum_length_pixels < 0.0 or args.long_bond_pixels <= 0.0:
        raise ValueError("Line length thresholds are invalid")
    if not tolerances_q or min(tolerances_q) < 0.0:
        raise ValueError("Tolerances must be nonnegative")
    if min(args.translation_dx, args.translation_dy) < 0:
        raise ValueError("Translation offsets must be nonnegative")
    if min(
        args.direction_angle_degrees,
        args.direction_perpendicular_pixels,
        args.direction_overlap_pixels,
    ) < 0.0:
        raise ValueError("Direction support thresholds must be nonnegative")
    if "internal_test" in str(shard).lower().replace("-", "_"):
        raise RuntimeError(f"Refusing sealed internal_test shard: {shard}")

    modes = {
        "combined_raster": False,
        "oracle_bond_layer": True,
    }
    aggregates = {
        mode: {
            "lsd_raw": _empty_aggregate(tolerances_q),
            "lsd_dense_local": _empty_aggregate(tolerances_q),
        }
        for mode in modes
    }
    equivariance = {
        mode: _translation_accumulator(tolerances_q) for mode in modes
    }
    direction_support = {mode: _direction_accumulator() for mode in modes}
    nonline_groups = Counter()
    samples_seen = 0
    translation_seen = 0

    with tarfile.open(shard, "r") as archive:
        keys = sorted(
            member.name.removesuffix(".primitives.json")
            for member in archive.getmembers()
            if member.name.endswith(".primitives.json")
        )
        for key in keys[: args.limit]:
            primitives = json.loads(_read_member(archive, key + ".primitives.json"))
            layers = json.loads(_read_member(archive, key + ".layers.json"))
            meta = json.loads(_read_member(archive, key + ".meta.json"))
            _not_internal(meta, shard, key)
            target_lines, target_groups, sample_nonline = extract_targets(
                primitives, layers, long_bond_pixels=args.long_bond_pixels
            )
            nonline_groups.update(sample_nonline)
            width, height = int(layers["width"]), int(layers["height"])
            for mode, oracle_bond_layer in modes.items():
                suffix = ".bond.png" if oracle_bond_layer else ".png"
                image = Image.open(io.BytesIO(_read_member(archive, key + suffix)))
                raw = detect_lsd_lines(
                    image,
                    oracle_bond_layer=oracle_bond_layer,
                    minimum_length_pixels=args.minimum_length_pixels,
                )
                dense = dense_local_expansion(raw)
                _accumulate_direction(
                    direction_support[mode],
                    evaluate_direction_support(
                        raw,
                        target_lines,
                        target_groups,
                        width=width,
                        height=height,
                        maximum_angle_degrees=args.direction_angle_degrees,
                        maximum_perpendicular_pixels=args.direction_perpendicular_pixels,
                        minimum_overlap_pixels=args.direction_overlap_pixels,
                    ),
                )
                for proposal_mode, proposals in (("lsd_raw", raw), ("lsd_dense_local", dense)):
                    proposals_q = proposals_pixels_to_q(proposals, width, height)
                    row = evaluate_proposals(
                        proposals_q, target_lines, target_groups, tolerances_q
                    )
                    _accumulate(aggregates[mode][proposal_mode], proposals, row)

                if translation_seen < args.translation_samples:
                    shifted_image = translate_grayscale(
                        image,
                        dx=args.translation_dx,
                        dy=args.translation_dy,
                        oracle_bond_layer=oracle_bond_layer,
                    )
                    shifted_detected = detect_lsd_lines(
                        shifted_image,
                        oracle_bond_layer=oracle_bond_layer,
                        minimum_length_pixels=args.minimum_length_pixels,
                    )
                    shifted_expected = translated_proposals(
                        raw,
                        dx=args.translation_dx,
                        dy=args.translation_dy,
                        width=width,
                        height=height,
                    )
                    _accumulate_equivariance(
                        equivariance[mode],
                        proposals_pixels_to_q(shifted_expected, width, height),
                        proposals_pixels_to_q(shifted_detected, width, height),
                        tolerances_q,
                    )
            samples_seen += 1
            if translation_seen < args.translation_samples:
                translation_seen += 1

    if not samples_seen:
        raise RuntimeError("Audit read no samples")
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "shard": str(shard),
        "shard_sha256": file_sha256(shard),
        "samples": samples_seen,
        "sealed_internal_test_read": False,
        "detector": {
            "name": "OpenCV LineSegmentDetector",
            "opencv_version": cv2.__version__,
            "refine": "LSD_REFINE_STD",
            "minimum_length_pixels": args.minimum_length_pixels,
            "target_geometry_used_for_proposal_generation": False,
            "dense_local_offsets_are_fixed": True,
        },
        "protocol": {
            "coordinate_max": int(COORDINATE_MAX),
            "tolerances_q": tolerances_q,
            "long_bond_pixels": args.long_bond_pixels,
            "hashed_strict_requires_all_members": True,
            "hashed_carrier_eligible_requires_members": 2,
            "oracle_bond_layer_contains_no_text": True,
            "direction_support": {
                "maximum_angle_degrees": args.direction_angle_degrees,
                "maximum_perpendicular_pixels": args.direction_perpendicular_pixels,
                "minimum_overlap_pixels": args.direction_overlap_pixels,
                "endpoint_completeness_required": False,
            },
        },
        "nonline_groups_not_covered_by_line_detector": dict(sorted(nonline_groups.items())),
        "modes": {
            mode: {
                proposal_mode: _summarize_aggregate(aggregate)
                for proposal_mode, aggregate in proposal_modes.items()
            }
            for mode, proposal_modes in aggregates.items()
        },
        "raw_lsd_direction_support": {
            mode: _summarize_direction(values)
            for mode, values in direction_support.items()
        },
        "translation_equivariance": {
            "samples": translation_seen,
            "dx": args.translation_dx,
            "dy": args.translation_dy,
            "proposal_mode": "lsd_raw",
            "modes": {
                mode: {
                    tolerance: _prf(
                        int(counts["matched"]),
                        int(counts["predicted"]),
                        int(counts["expected"]),
                    )
                    for tolerance, counts in values.items()
                }
                for mode, values in equivariance.items()
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
