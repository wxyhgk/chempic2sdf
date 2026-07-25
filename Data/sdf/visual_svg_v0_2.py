#!/usr/bin/env python3
"""Canonical group-aware Visual SVG v0.2 JSON protocol.

The protocol deliberately contains only visible drawing information.  Source
bond/atom identifiers and chemical semantics are used only while converting a
v0.1 audit sidecar and never appear in the serialized model target.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import Any, TypeAlias

try:
    from .visual_primitives_v0_1 import validate_visual_primitive_references
    from .visual_svg_sequence_v0_1 import (
        COORDINATE_MAX,
        STROKE_WIDTH_MAX,
        STROKE_WIDTH_MIN,
        LineRecordV01,
        ParsedVisualSVGSequenceV01,
        QuadRecordV01,
        TextBoxRecordV01,
        TriangleRecordV01,
        canonicalize_visual_svg_v0_1,
    )
except ImportError:
    from visual_primitives_v0_1 import validate_visual_primitive_references
    from visual_svg_sequence_v0_1 import (
        COORDINATE_MAX,
        STROKE_WIDTH_MAX,
        STROKE_WIDTH_MIN,
        LineRecordV01,
        ParsedVisualSVGSequenceV01,
        QuadRecordV01,
        TextBoxRecordV01,
        TriangleRecordV01,
        canonicalize_visual_svg_v0_1,
    )


SCHEMA_VERSION = "visual-svg-0.2"
UNITS = "q2047"
GROUP_KINDS = (
    "single_line",
    "parallel_2",
    "parallel_3",
    "hashed_wedge",
    "solid_triangle",
    "solid_quad",
)
GROUP_KIND_RANK = {kind: index for index, kind in enumerate(GROUP_KINDS)}


class VisualSVGV02Error(ValueError):
    """Raised when a v0.2 target is malformed or non-canonical."""


@dataclass(frozen=True, slots=True)
class LineMemberV02:
    x0: int
    y0: int
    x1: int
    y1: int
    stroke_width: int


@dataclass(frozen=True, slots=True)
class TriangleMemberV02:
    x0: int
    y0: int
    x1: int
    y1: int
    x2: int
    y2: int
    stroke_width: int


@dataclass(frozen=True, slots=True)
class QuadMemberV02:
    x0: int
    y0: int
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    stroke_width: int


@dataclass(frozen=True, slots=True)
class TextBoxRecordV02:
    x0: int
    y0: int
    x1: int
    y1: int


PhysicalMemberV02: TypeAlias = LineMemberV02 | TriangleMemberV02 | QuadMemberV02


@dataclass(frozen=True, slots=True)
class BondGroupRecordV02:
    kind: str
    members: tuple[PhysicalMemberV02, ...]


@dataclass(frozen=True, slots=True)
class ParsedVisualSVGSequenceV02:
    bond_groups: tuple[BondGroupRecordV02, ...]
    text_boxes: tuple[TextBoxRecordV02, ...]

    @property
    def group_count(self) -> int:
        return len(self.bond_groups)

    @property
    def primitive_count(self) -> int:
        return sum(len(group.members) for group in self.bond_groups) + len(
            self.text_boxes
        )


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisualSVGV02Error(f"{field} must be an integer")
    return value


def _coordinate(value: Any, field: str) -> int:
    result = _integer(value, field)
    if not 0 <= result <= COORDINATE_MAX:
        raise VisualSVGV02Error(
            f"{field}={result} is outside 0..{COORDINATE_MAX}"
        )
    return result


def _stroke_width(value: Any) -> int:
    result = _integer(value, "stroke_width")
    if not STROKE_WIDTH_MIN <= result <= STROKE_WIDTH_MAX:
        raise VisualSVGV02Error(
            f"stroke_width={result} is outside "
            f"{STROKE_WIDTH_MIN}..{STROKE_WIDTH_MAX}"
        )
    return result


def _canonical_cycle(
    points: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    def rotations(values: tuple[tuple[int, int], ...]):
        for index in range(len(values)):
            yield values[index:] + values[:index]

    candidates = [*rotations(points), *rotations(tuple(reversed(points)))]
    return min(candidates, key=lambda values: tuple((y, x) for x, y in values))


def _orientation(
    left: tuple[int, int],
    middle: tuple[int, int],
    right: tuple[int, int],
) -> int:
    value = (
        (middle[1] - left[1]) * (right[0] - middle[0])
        - (middle[0] - left[0]) * (right[1] - middle[1])
    )
    return 0 if value == 0 else 1 if value > 0 else -1


def _on_segment(
    left: tuple[int, int],
    point: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return (
        min(left[0], right[0]) <= point[0] <= max(left[0], right[0])
        and min(left[1], right[1]) <= point[1] <= max(left[1], right[1])
    )


def _segments_intersect(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    orientations = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
        return True
    return (
        (orientations[0] == 0 and _on_segment(a, c, b))
        or (orientations[1] == 0 and _on_segment(a, d, b))
        or (orientations[2] == 0 and _on_segment(c, a, d))
        or (orientations[3] == 0 and _on_segment(c, b, d))
    )


def _canonical_line(member: LineMemberV02) -> LineMemberV02:
    for field, value in zip(
        ("x0", "y0", "x1", "y1"),
        (member.x0, member.y0, member.x1, member.y1),
    ):
        _coordinate(value, field)
    _stroke_width(member.stroke_width)
    if (member.x0, member.y0) == (member.x1, member.y1):
        raise VisualSVGV02Error("A line member must have two distinct endpoints")
    if (member.y1, member.x1) < (member.y0, member.x0):
        return LineMemberV02(
            member.x1,
            member.y1,
            member.x0,
            member.y0,
            member.stroke_width,
        )
    return member


def _canonical_triangle(member: TriangleMemberV02) -> TriangleMemberV02:
    values = (
        member.x0,
        member.y0,
        member.x1,
        member.y1,
        member.x2,
        member.y2,
    )
    for field, value in zip(("x0", "y0", "x1", "y1", "x2", "y2"), values):
        _coordinate(value, field)
    _stroke_width(member.stroke_width)
    points = tuple(zip(values[::2], values[1::2]))
    if len(set(points)) != 3:
        raise VisualSVGV02Error("A triangle member must have three distinct points")
    area2 = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, (*points[1:], points[0]))
    )
    if area2 == 0:
        raise VisualSVGV02Error("A triangle member must have non-zero area")
    points = _canonical_cycle(points)
    return TriangleMemberV02(
        points[0][0],
        points[0][1],
        points[1][0],
        points[1][1],
        points[2][0],
        points[2][1],
        member.stroke_width,
    )


def _canonical_quad(member: QuadMemberV02) -> QuadMemberV02:
    values = (
        member.x0,
        member.y0,
        member.x1,
        member.y1,
        member.x2,
        member.y2,
        member.x3,
        member.y3,
    )
    fields = ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3")
    for field, value in zip(fields, values):
        _coordinate(value, field)
    _stroke_width(member.stroke_width)
    points = tuple(zip(values[::2], values[1::2]))
    if len(set(points)) != 4:
        raise VisualSVGV02Error("A quad member must have four distinct points")
    if _segments_intersect(points[0], points[1], points[2], points[3]) or (
        _segments_intersect(points[1], points[2], points[3], points[0])
    ):
        raise VisualSVGV02Error(
            "A quad member must be a simple non-self-intersecting polygon"
        )
    area2 = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, (*points[1:], points[0]))
    )
    if area2 == 0:
        raise VisualSVGV02Error("A quad member must have non-zero area")
    points = _canonical_cycle(points)
    return QuadMemberV02(
        points[0][0],
        points[0][1],
        points[1][0],
        points[1][1],
        points[2][0],
        points[2][1],
        points[3][0],
        points[3][1],
        member.stroke_width,
    )


def _canonical_text_box(record: TextBoxRecordV02) -> TextBoxRecordV02:
    for field, value in zip(
        ("x0", "y0", "x1", "y1"),
        (record.x0, record.y0, record.x1, record.y1),
    ):
        _coordinate(value, field)
    if record.x0 >= record.x1 or record.y0 >= record.y1:
        raise VisualSVGV02Error("A text box must have positive width and height")
    return record


def _canonical_member(member: PhysicalMemberV02) -> PhysicalMemberV02:
    if isinstance(member, LineMemberV02):
        return _canonical_line(member)
    if isinstance(member, TriangleMemberV02):
        return _canonical_triangle(member)
    if isinstance(member, QuadMemberV02):
        return _canonical_quad(member)
    raise VisualSVGV02Error(f"Unsupported member type: {type(member).__name__}")


def _member_points(member: PhysicalMemberV02) -> tuple[tuple[int, int], ...]:
    if isinstance(member, LineMemberV02):
        return ((member.x0, member.y0), (member.x1, member.y1))
    if isinstance(member, TriangleMemberV02):
        return (
            (member.x0, member.y0),
            (member.x1, member.y1),
            (member.x2, member.y2),
        )
    return (
        (member.x0, member.y0),
        (member.x1, member.y1),
        (member.x2, member.y2),
        (member.x3, member.y3),
    )


def _member_sort_key(member: PhysicalMemberV02) -> tuple[Any, ...]:
    rank = {
        LineMemberV02: 0,
        TriangleMemberV02: 1,
        QuadMemberV02: 2,
    }[type(member)]
    points = _member_points(member)
    return (
        rank,
        Fraction(sum(y for _, y in points), len(points)),
        Fraction(sum(x for x, _ in points), len(points)),
        tuple(value for x, y in points for value in (y, x)),
        member.stroke_width,
    )


def _validate_group_shape(group: BondGroupRecordV02) -> None:
    if group.kind not in GROUP_KIND_RANK:
        raise VisualSVGV02Error(f"Unsupported bond-group kind: {group.kind!r}")
    expected: dict[str, tuple[type, int | None]] = {
        "single_line": (LineMemberV02, 1),
        "parallel_2": (LineMemberV02, 2),
        "parallel_3": (LineMemberV02, 3),
        "hashed_wedge": (LineMemberV02, None),
        "solid_triangle": (TriangleMemberV02, 1),
        "solid_quad": (QuadMemberV02, 1),
    }
    member_type, count = expected[group.kind]
    if count is not None and len(group.members) != count:
        raise VisualSVGV02Error(
            f"{group.kind} requires exactly {count} member(s), got {len(group.members)}"
        )
    if group.kind == "hashed_wedge" and len(group.members) < 2:
        raise VisualSVGV02Error("hashed_wedge requires at least two line members")
    if not all(isinstance(member, member_type) for member in group.members):
        raise VisualSVGV02Error(
            f"{group.kind} only accepts {member_type.__name__} members"
        )


def _canonical_group(group: BondGroupRecordV02) -> BondGroupRecordV02:
    members = tuple(
        sorted(
            (_canonical_member(member) for member in group.members),
            key=_member_sort_key,
        )
    )
    canonical = BondGroupRecordV02(str(group.kind), members)
    _validate_group_shape(canonical)
    return canonical


def _group_sort_key(group: BondGroupRecordV02) -> tuple[Any, ...]:
    points = tuple(
        point for member in group.members for point in _member_points(member)
    )
    return (
        GROUP_KIND_RANK[group.kind],
        Fraction(sum(y for _, y in points), len(points)),
        Fraction(sum(x for x, _ in points), len(points)),
        tuple(_member_sort_key(member) for member in group.members),
    )


def canonicalize_visual_svg_v0_2(
    parsed: ParsedVisualSVGSequenceV02,
) -> ParsedVisualSVGSequenceV02:
    groups = tuple(
        sorted(
            (_canonical_group(group) for group in parsed.bond_groups),
            key=_group_sort_key,
        )
    )
    if not groups:
        raise VisualSVGV02Error("Visual SVG v0.2 requires at least one bond group")
    text_boxes = tuple(
        sorted(
            (_canonical_text_box(record) for record in parsed.text_boxes),
            key=lambda record: (
                record.y0 + record.y1,
                record.x0 + record.x1,
                record.y0,
                record.x0,
                record.y1,
                record.x1,
            ),
        )
    )
    return ParsedVisualSVGSequenceV02(groups, text_boxes)


def _points_dict(points: tuple[tuple[int, int], ...]) -> list[list[int]]:
    return [[x, y] for x, y in points]


def _member_dict(member: PhysicalMemberV02) -> dict[str, Any]:
    if isinstance(member, LineMemberV02):
        kind = "line"
    elif isinstance(member, TriangleMemberV02):
        kind = "triangle"
    else:
        kind = "quad"
    return {
        "kind": kind,
        "points": _points_dict(_member_points(member)),
        "stroke_width": member.stroke_width,
    }


def visual_svg_v0_2_to_dict(parsed: ParsedVisualSVGSequenceV02) -> dict[str, Any]:
    parsed = canonicalize_visual_svg_v0_2(parsed)
    return {
        "schema_version": SCHEMA_VERSION,
        "units": UNITS,
        "bond_groups": [
            {
                "kind": group.kind,
                "members": [_member_dict(member) for member in group.members],
            }
            for group in parsed.bond_groups
        ],
        "text_boxes": [
            {
                "kind": "text_box",
                "points": [[record.x0, record.y0], [record.x1, record.y1]],
            }
            for record in parsed.text_boxes
        ],
    }


def serialize_parsed_visual_svg_v0_2(parsed: ParsedVisualSVGSequenceV02) -> str:
    """Serialize one canonical JSON target with stable field ordering."""

    return json.dumps(
        visual_svg_v0_2_to_dict(parsed),
        ensure_ascii=True,
        separators=(",", ":"),
    ) + "\n"


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisualSVGV02Error(f"{context} must be an object")
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise VisualSVGV02Error(
            f"{context} fields differ; missing={missing}, extra={extra}"
        )
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualSVGV02Error(f"Duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _parse_points(value: Any, count: int, context: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise VisualSVGV02Error(f"{context} requires exactly {count} points")
    points = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise VisualSVGV02Error(f"{context}.points[{index}] must be [x,y]")
        points.append(
            (
                _coordinate(point[0], f"{context}.points[{index}].x"),
                _coordinate(point[1], f"{context}.points[{index}].y"),
            )
        )
    return tuple(points)


def _parse_member(value: Any, context: str) -> PhysicalMemberV02:
    item = _exact_keys(value, {"kind", "points", "stroke_width"}, context)
    kind = item["kind"]
    width = _stroke_width(item["stroke_width"])
    if kind == "line":
        points = _parse_points(item["points"], 2, context)
        return LineMemberV02(*points[0], *points[1], width)
    if kind == "triangle":
        points = _parse_points(item["points"], 3, context)
        return TriangleMemberV02(
            *(coordinate for point in points for coordinate in point), width
        )
    if kind == "quad":
        points = _parse_points(item["points"], 4, context)
        return QuadMemberV02(
            *(coordinate for point in points for coordinate in point), width
        )
    raise VisualSVGV02Error(f"Unsupported physical member kind: {kind!r}")


def parse_visual_svg_v0_2(sequence: str) -> ParsedVisualSVGSequenceV02:
    """Parse JSON and reject non-canonical group/member ordering."""

    if not isinstance(sequence, str):
        raise VisualSVGV02Error("Visual SVG v0.2 input must be text")
    try:
        raw = json.loads(sequence, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise VisualSVGV02Error(f"Invalid Visual SVG v0.2 JSON: {error}") from error
    root = _exact_keys(
        raw,
        {"schema_version", "units", "bond_groups", "text_boxes"},
        "root",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise VisualSVGV02Error(
            f"Unsupported schema_version: {root['schema_version']!r}"
        )
    if root["units"] != UNITS:
        raise VisualSVGV02Error(f"Unsupported units: {root['units']!r}")
    if not isinstance(root["bond_groups"], list):
        raise VisualSVGV02Error("bond_groups must be an array")
    if not isinstance(root["text_boxes"], list):
        raise VisualSVGV02Error("text_boxes must be an array")

    groups = []
    for group_index, value in enumerate(root["bond_groups"]):
        context = f"bond_groups[{group_index}]"
        item = _exact_keys(value, {"kind", "members"}, context)
        if not isinstance(item["kind"], str):
            raise VisualSVGV02Error(f"{context}.kind must be a string")
        if not isinstance(item["members"], list):
            raise VisualSVGV02Error(f"{context}.members must be an array")
        groups.append(
            BondGroupRecordV02(
                item["kind"],
                tuple(
                    _parse_member(member, f"{context}.members[{member_index}]")
                    for member_index, member in enumerate(item["members"])
                ),
            )
        )

    text_boxes = []
    for index, value in enumerate(root["text_boxes"]):
        context = f"text_boxes[{index}]"
        item = _exact_keys(value, {"kind", "points"}, context)
        if item["kind"] != "text_box":
            raise VisualSVGV02Error(f"{context}.kind must be 'text_box'")
        points = _parse_points(item["points"], 2, context)
        text_boxes.append(TextBoxRecordV02(*points[0], *points[1]))

    parsed = ParsedVisualSVGSequenceV02(tuple(groups), tuple(text_boxes))
    canonical = canonicalize_visual_svg_v0_2(parsed)
    if parsed != canonical:
        raise VisualSVGV02Error(
            "Visual SVG v0.2 groups or members are not in canonical order"
        )
    return parsed


def _member_from_v0_1_primitive(primitive: dict[str, Any]) -> PhysicalMemberV02:
    kind = primitive.get("kind")
    points = primitive.get("quantized_points")
    if kind == "line":
        parsed = _parse_points(points, 2, "v0.1 line")
        return LineMemberV02(
            *parsed[0], *parsed[1], _stroke_width(primitive.get("stroke_width_q"))
        )
    if kind == "triangle":
        parsed = _parse_points(points, 3, "v0.1 triangle")
        return TriangleMemberV02(
            *(coordinate for point in parsed for coordinate in point),
            _stroke_width(primitive.get("stroke_width_q")),
        )
    if kind == "quad":
        parsed = _parse_points(points, 4, "v0.1 quad")
        return QuadMemberV02(
            *(coordinate for point in parsed for coordinate in point),
            _stroke_width(primitive.get("stroke_width_q")),
        )
    raise VisualSVGV02Error(f"Unsupported v0.1 bond primitive kind: {kind!r}")


def _group_kind_from_v0_1(
    source_group: dict[str, Any],
    members: tuple[PhysicalMemberV02, ...],
) -> str:
    if len(members) == 1 and isinstance(members[0], TriangleMemberV02):
        return "solid_triangle"
    if len(members) == 1 and isinstance(members[0], QuadMemberV02):
        return "solid_quad"
    if not members or not all(isinstance(member, LineMemberV02) for member in members):
        raise VisualSVGV02Error("A v0.1 bond group has mixed or empty physical members")

    # BEGINDASH is a source-side rendering annotation.  It chooses the visible
    # group symbol but is intentionally omitted from the v0.2 model target.
    if str(source_group.get("direction", "")) == "BEGINDASH":
        return "hashed_wedge"
    if len(members) == 1:
        return "single_line"
    if len(members) == 2:
        return "parallel_2"
    if len(members) == 3:
        return "parallel_3"
    raise VisualSVGV02Error(
        f"Cannot classify a {len(members)}-line v0.1 group without a hashed-wedge hint"
    )


def visual_primitives_v0_1_to_visual_svg_v0_2(
    payload: dict[str, Any],
) -> ParsedVisualSVGSequenceV02:
    """Convert one full v0.1 audit sidecar into the chemistry-free v0.2 target."""

    if payload.get("schema_version") != "visual-primitives-0.1":
        raise VisualSVGV02Error(
            f"Unsupported source schema: {payload.get('schema_version')!r}"
        )
    validate_visual_primitive_references(payload)
    primitives = {
        int(primitive["primitive_id"]): primitive
        for primitive in payload.get("primitives", [])
    }
    groups = []
    for source_group in payload.get("bond_groups", []):
        members = tuple(
            _member_from_v0_1_primitive(primitives[int(primitive_id)])
            for primitive_id in source_group.get("primitive_ids", [])
        )
        groups.append(
            BondGroupRecordV02(
                _group_kind_from_v0_1(source_group, members),
                members,
            )
        )

    text_boxes = []
    for primitive in payload.get("primitives", []):
        if primitive.get("kind") != "text_box":
            continue
        points = _parse_points(
            primitive.get("quantized_points"),
            2,
            "v0.1 text_box",
        )
        text_boxes.append(TextBoxRecordV02(*points[0], *points[1]))
    return canonicalize_visual_svg_v0_2(
        ParsedVisualSVGSequenceV02(tuple(groups), tuple(text_boxes))
    )


def serialize_visual_svg_v0_2(payload: dict[str, Any]) -> str:
    return serialize_parsed_visual_svg_v0_2(
        visual_primitives_v0_1_to_visual_svg_v0_2(payload)
    )


def flatten_visual_svg_v0_2_to_v0_1(
    parsed: ParsedVisualSVGSequenceV02,
) -> ParsedVisualSVGSequenceV01:
    """Flatten grouped members into the canonical v0.1 physical target.

    The full v0.1 audit sidecar cannot be recreated because v0.2 intentionally
    drops owner IDs, chemical fields, raw pixel coordinates, and source z-order.
    The q2047 model target is reconstructed exactly.
    """

    parsed = canonicalize_visual_svg_v0_2(parsed)
    lines = []
    triangles = []
    quads = []
    for group in parsed.bond_groups:
        for member in group.members:
            if isinstance(member, LineMemberV02):
                lines.append(
                    LineRecordV01(
                        member.x0,
                        member.y0,
                        member.x1,
                        member.y1,
                        member.stroke_width,
                    )
                )
            elif isinstance(member, TriangleMemberV02):
                triangles.append(
                    TriangleRecordV01(
                        member.x0,
                        member.y0,
                        member.x1,
                        member.y1,
                        member.x2,
                        member.y2,
                        member.stroke_width,
                    )
                )
            else:
                quads.append(
                    QuadRecordV01(
                        member.x0,
                        member.y0,
                        member.x1,
                        member.y1,
                        member.x2,
                        member.y2,
                        member.x3,
                        member.y3,
                        member.stroke_width,
                    )
                )
    text_boxes = tuple(
        TextBoxRecordV01(record.x0, record.y0, record.x1, record.y1)
        for record in parsed.text_boxes
    )
    return canonicalize_visual_svg_v0_1(
        ParsedVisualSVGSequenceV01(
            tuple(lines),
            tuple(triangles),
            tuple(quads),
            text_boxes,
        )
    )
