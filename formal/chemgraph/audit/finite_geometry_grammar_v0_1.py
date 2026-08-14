"""Exact finite geometry grammar for the 3x3 ChemGraph micro-scenes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .finite_micro_scene_v0_1 import (
    CANVAS_SIZE,
    Bond,
    Node,
    Scene,
    validate_scene,
)


SIMPLE_LINE_ANGLE_VALID = "simple_line_angle_valid"
CROSSING_ALLOWED = "crossing_allowed"
ADVERSARIAL_INVALID = "adversarial_invalid"

GRAMMAR_LAYERS = (
    SIMPLE_LINE_ANGLE_VALID,
    CROSSING_ALLOWED,
    ADVERSARIAL_INVALID,
)

DOMAIN_LAYERS = {
    "simple_domain": frozenset({SIMPLE_LINE_ANGLE_VALID}),
    "crossing_domain": frozenset({SIMPLE_LINE_ANGLE_VALID, CROSSING_ALLOWED}),
    "all_domain": frozenset(GRAMMAR_LAYERS),
}

BOND_THROUGH_ACTIVE_NONINCIDENT_ATOM = "bond_through_active_nonincident_atom"
POSITIVE_LENGTH_COLLINEAR_OVERLAP = "positive_length_collinear_overlap"
SHARED_ENDPOINT_SAME_RAY_OVERLAP = "shared_endpoint_same_ray_overlap"
UNMARKED_180_DEGREE2 = "unmarked_180_degree2"
NONINCIDENT_PROPER_CROSSING = "nonincident_proper_crossing"
UNSUPPORTED_CROSSING_MULTIPLICITY = "unsupported_crossing_multiplicity"
DUPLICATE_NODE_COORDINATE = "duplicate_node_coordinate"
ZERO_LENGTH_PHYSICAL_MEMBER = "zero_length_physical_member"
OUT_OF_CANVAS_PHYSICAL_MEMBER = "out_of_canvas_physical_member"
PHYSICAL_MEMBER_SEMANTIC_MISMATCH = "physical_member_semantic_mismatch"
ORDER_DISPLAY_MEMBER_MISMATCH = "order_display_member_mismatch"
UNUSED_IMPLICIT_NODE = "unused_implicit_node"
NONSEMANTIC_T_TOUCH = "nonsemantic_t_touch"
UNSUPPORTED_TEXT_NOT_RENDERED = "unsupported_text_not_rendered"

ORDER_DISPLAY = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE"}


@dataclass(frozen=True, slots=True)
class GrammarClassification:
    layer: str
    reasons: tuple[str, ...]


def _cross(
    begin: tuple[int, int],
    end: tuple[int, int],
    point: tuple[int, int],
) -> int:
    return (end[0] - begin[0]) * (point[1] - begin[1]) - (
        end[1] - begin[1]
    ) * (point[0] - begin[0])


def _point_strictly_inside_segment(
    point: tuple[int, int],
    begin: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    if _cross(begin, end, point) != 0:
        return False
    return (
        (point[0] - begin[0]) * (point[0] - end[0])
        + (point[1] - begin[1]) * (point[1] - end[1])
        < 0
    )


def _positive_collinear_overlap(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    left_begin, left_end = left
    right_begin, right_end = right
    if _cross(left_begin, left_end, right_begin) != 0:
        return False
    if _cross(left_begin, left_end, right_end) != 0:
        return False
    use_x = abs(left_end[0] - left_begin[0]) >= abs(left_end[1] - left_begin[1])
    axis = 0 if use_x else 1
    left_interval = sorted((left_begin[axis], left_end[axis]))
    right_interval = sorted((right_begin[axis], right_end[axis]))
    return max(left_interval[0], right_interval[0]) < min(
        left_interval[1], right_interval[1]
    )


def _proper_crossing(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    left_begin, left_end = left
    right_begin, right_end = right
    left_side_0 = _cross(left_begin, left_end, right_begin)
    left_side_1 = _cross(left_begin, left_end, right_end)
    right_side_0 = _cross(right_begin, right_end, left_begin)
    right_side_1 = _cross(right_begin, right_end, left_end)
    return left_side_0 * left_side_1 < 0 and right_side_0 * right_side_1 < 0


def _crossing_point(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[Fraction, Fraction]:
    left_begin, left_end = left
    right_begin, right_end = right
    left_vector = (
        left_end[0] - left_begin[0],
        left_end[1] - left_begin[1],
    )
    right_vector = (
        right_end[0] - right_begin[0],
        right_end[1] - right_begin[1],
    )
    denominator = (
        left_vector[0] * right_vector[1]
        - left_vector[1] * right_vector[0]
    )
    if denominator == 0:
        raise ValueError("Parallel segments do not have one crossing point")
    displacement = (
        right_begin[0] - left_begin[0],
        right_begin[1] - left_begin[1],
    )
    numerator = (
        displacement[0] * right_vector[1]
        - displacement[1] * right_vector[0]
    )
    return (
        Fraction(left_begin[0] * denominator + numerator * left_vector[0], denominator),
        Fraction(left_begin[1] * denominator + numerator * left_vector[1], denominator),
    )


def _semantic_segment(
    bond: Bond,
    node_by_id: dict[int, Node],
) -> tuple[tuple[int, int], tuple[int, int]]:
    begin = node_by_id[bond.begin_atom]
    end = node_by_id[bond.end_atom]
    return (begin.x, begin.y), (end.x, end.y)


def classify_geometry(scene: Scene) -> GrammarClassification:
    validate_scene(scene)
    node_by_id = {node.atom_id: node for node in scene.nodes}
    segments = tuple(_semantic_segment(bond, node_by_id) for bond in scene.bonds)
    invalid_reasons: set[str] = set()

    coordinates = [(node.x, node.y) for node in scene.nodes]
    if len(coordinates) != len(set(coordinates)):
        invalid_reasons.add(DUPLICATE_NODE_COORDINATE)
    if any(node.color.startswith("TEXT:") for node in scene.nodes):
        invalid_reasons.add(UNSUPPORTED_TEXT_NOT_RENDERED)

    incident_atom_ids = {
        atom_id for bond in scene.bonds for atom_id in bond.endpoints
    }
    if any(
        node.atom_id not in incident_atom_ids and not node.color.startswith("TEXT:")
        for node in scene.nodes
    ):
        invalid_reasons.add(UNUSED_IMPLICIT_NODE)

    for bond, semantic_segment in zip(scene.bonds, segments, strict=True):
        if (
            ORDER_DISPLAY.get(bond.order) != bond.display_type
            or len(bond.members) != bond.order
        ):
            invalid_reasons.add(ORDER_DISPLAY_MEMBER_MISMATCH)
        semantic_begin, semantic_end = semantic_segment
        semantic_vector = (
            semantic_end[0] - semantic_begin[0],
            semantic_end[1] - semantic_begin[1],
        )
        canonical_semantic = {semantic_begin, semantic_end}
        canonical_members = set()
        lane_offsets: list[tuple[int, int]] = []
        for member in bond.members:
            if member.begin == member.end:
                invalid_reasons.add(ZERO_LENGTH_PHYSICAL_MEMBER)
            if any(
                coordinate < 0 or coordinate >= CANVAS_SIZE
                for point in (member.begin, member.end)
                for coordinate in point
            ):
                invalid_reasons.add(OUT_OF_CANVAS_PHYSICAL_MEMBER)
            member_vector = (
                member.end[0] - member.begin[0],
                member.end[1] - member.begin[1],
            )
            if bond.order == 1:
                if {member.begin, member.end} != canonical_semantic:
                    invalid_reasons.add(PHYSICAL_MEMBER_SEMANTIC_MISMATCH)
            else:
                if member_vector == semantic_vector:
                    aligned_begin, aligned_end = member.begin, member.end
                elif member_vector == (-semantic_vector[0], -semantic_vector[1]):
                    aligned_begin, aligned_end = member.end, member.begin
                else:
                    invalid_reasons.add(PHYSICAL_MEMBER_SEMANTIC_MISMATCH)
                    aligned_begin, aligned_end = member.begin, member.end
                begin_offset = (
                    aligned_begin[0] - semantic_begin[0],
                    aligned_begin[1] - semantic_begin[1],
                )
                end_offset = (
                    aligned_end[0] - semantic_end[0],
                    aligned_end[1] - semantic_end[1],
                )
                if (
                    begin_offset[0] * semantic_vector[0]
                    + begin_offset[1] * semantic_vector[1]
                    != 0
                    or end_offset[0] * semantic_vector[0]
                    + end_offset[1] * semantic_vector[1]
                    != 0
                ):
                    invalid_reasons.add(PHYSICAL_MEMBER_SEMANTIC_MISMATCH)
                lane_offsets.append(begin_offset)
            canonical_members.add(tuple(sorted((member.begin, member.end))))
        if len(canonical_members) != len(bond.members):
            invalid_reasons.add(PHYSICAL_MEMBER_SEMANTIC_MISMATCH)
        if bond.order > 1 and lane_offsets and (
            sum(offset[0] for offset in lane_offsets) != 0
            or sum(offset[1] for offset in lane_offsets) != 0
        ):
            invalid_reasons.add(PHYSICAL_MEMBER_SEMANTIC_MISMATCH)

    for bond, segment in zip(scene.bonds, segments, strict=True):
        incident = {bond.begin_atom, bond.end_atom}
        for node in scene.nodes:
            if node.atom_id in incident:
                continue
            if _point_strictly_inside_segment((node.x, node.y), *segment):
                invalid_reasons.add(BOND_THROUGH_ACTIVE_NONINCIDENT_ATOM)
                invalid_reasons.add(NONSEMANTIC_T_TOUCH)

    for left_index, right_index in _bond_pairs(scene):
        if not _positive_collinear_overlap(
            segments[left_index], segments[right_index]
        ):
            continue
        invalid_reasons.add(POSITIVE_LENGTH_COLLINEAR_OVERLAP)
        if set(scene.bonds[left_index].endpoints).intersection(
            scene.bonds[right_index].endpoints
        ):
            invalid_reasons.add(SHARED_ENDPOINT_SAME_RAY_OVERLAP)

    incident_bonds: dict[int, list[Bond]] = {node.atom_id: [] for node in scene.nodes}
    for bond in scene.bonds:
        incident_bonds[bond.begin_atom].append(bond)
        incident_bonds[bond.end_atom].append(bond)
    for node in scene.nodes:
        incident = incident_bonds[node.atom_id]
        if len(incident) != 2 or node.color.startswith("TEXT:"):
            continue
        vectors = []
        for bond in incident:
            other_id = (
                bond.end_atom if bond.begin_atom == node.atom_id else bond.begin_atom
            )
            other = node_by_id[other_id]
            vectors.append((other.x - node.x, other.y - node.y))
        cross = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
        dot = vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]
        if cross == 0 and dot < 0:
            invalid_reasons.add(UNMARKED_180_DEGREE2)

    crossing_points: dict[tuple[Fraction, Fraction], set[int]] = {}
    for left_index, right_index in _bond_pairs(scene):
        if set(scene.bonds[left_index].endpoints).intersection(
            scene.bonds[right_index].endpoints
        ):
            continue
        if _proper_crossing(segments[left_index], segments[right_index]):
            point = _crossing_point(segments[left_index], segments[right_index])
            crossing_points.setdefault(point, set()).update((left_index, right_index))
    if any(len(bonds) > 2 for bonds in crossing_points.values()):
        invalid_reasons.add(UNSUPPORTED_CROSSING_MULTIPLICITY)

    if invalid_reasons:
        return GrammarClassification(ADVERSARIAL_INVALID, tuple(sorted(invalid_reasons)))

    if crossing_points:
        return GrammarClassification(
            CROSSING_ALLOWED, (NONINCIDENT_PROPER_CROSSING,)
        )
    return GrammarClassification(SIMPLE_LINE_ANGLE_VALID, ())


def _bond_pairs(scene: Scene):
    for left_index in range(len(scene.bonds)):
        for right_index in range(left_index + 1, len(scene.bonds)):
            yield left_index, right_index


__all__ = [
    "ADVERSARIAL_INVALID",
    "BOND_THROUGH_ACTIVE_NONINCIDENT_ATOM",
    "CROSSING_ALLOWED",
    "DUPLICATE_NODE_COORDINATE",
    "DOMAIN_LAYERS",
    "GRAMMAR_LAYERS",
    "GrammarClassification",
    "NONINCIDENT_PROPER_CROSSING",
    "NONSEMANTIC_T_TOUCH",
    "ORDER_DISPLAY",
    "ORDER_DISPLAY_MEMBER_MISMATCH",
    "OUT_OF_CANVAS_PHYSICAL_MEMBER",
    "PHYSICAL_MEMBER_SEMANTIC_MISMATCH",
    "POSITIVE_LENGTH_COLLINEAR_OVERLAP",
    "SHARED_ENDPOINT_SAME_RAY_OVERLAP",
    "SIMPLE_LINE_ANGLE_VALID",
    "UNMARKED_180_DEGREE2",
    "UNSUPPORTED_CROSSING_MULTIPLICITY",
    "UNSUPPORTED_TEXT_NOT_RENDERED",
    "UNUSED_IMPLICIT_NODE",
    "ZERO_LENGTH_PHYSICAL_MEMBER",
    "classify_geometry",
]
