#!/usr/bin/env python3
"""Restricted raster-to-vector sequence for visible chemical primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable


SCHEMA_VERSION = "visual-svg-0.1"
START_TOKEN = "<visual_svg v=0.1 units=q2047>"
END_TOKEN = "<visual_svg_end>"
COORDINATE_MAX = 2047
STROKE_WIDTH_MIN = 1
STROKE_WIDTH_MAX = 63


class VisualSVGSequenceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LineRecordV01:
    x0: int
    y0: int
    x1: int
    y1: int
    stroke_width: int


@dataclass(frozen=True, slots=True)
class TriangleRecordV01:
    x0: int
    y0: int
    x1: int
    y1: int
    x2: int
    y2: int
    stroke_width: int


@dataclass(frozen=True, slots=True)
class QuadRecordV01:
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
class TextBoxRecordV01:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True, slots=True)
class ParsedVisualSVGSequenceV01:
    lines: tuple[LineRecordV01, ...]
    triangles: tuple[TriangleRecordV01, ...]
    quads: tuple[QuadRecordV01, ...]
    text_boxes: tuple[TextBoxRecordV01, ...]

    @property
    def primitive_count(self) -> int:
        return len(self.lines) + len(self.triangles) + len(self.quads) + len(self.text_boxes)


LINE_RE = re.compile(r"^<line (\d+) (\d+) (\d+) (\d+) (\d+)>$")
TRIANGLE_RE = re.compile(
    r"^<triangle (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)>$"
)
QUAD_RE = re.compile(
    r"^<quad (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)>$"
)
TEXT_BOX_RE = re.compile(r"^<text_box (\d+) (\d+) (\d+) (\d+)>$")


def _coordinate(value: int, field: str) -> int:
    if not 0 <= value <= COORDINATE_MAX:
        raise VisualSVGSequenceError(f"{field}={value} is outside 0..{COORDINATE_MAX}")
    return value


def _stroke_width(value: int) -> int:
    if not STROKE_WIDTH_MIN <= value <= STROKE_WIDTH_MAX:
        raise VisualSVGSequenceError(
            f"stroke_width={value} is outside {STROKE_WIDTH_MIN}..{STROKE_WIDTH_MAX}"
        )
    return value


def _validate_line(record: LineRecordV01) -> None:
    for field, value in zip(
        ("x0", "y0", "x1", "y1"),
        (record.x0, record.y0, record.x1, record.y1),
    ):
        _coordinate(value, field)
    _stroke_width(record.stroke_width)
    if (record.x0, record.y0) == (record.x1, record.y1):
        raise VisualSVGSequenceError("A line must have two distinct endpoints")


def _validate_triangle(record: TriangleRecordV01) -> None:
    values = (record.x0, record.y0, record.x1, record.y1, record.x2, record.y2)
    for field, value in zip(("x0", "y0", "x1", "y1", "x2", "y2"), values):
        _coordinate(value, field)
    _stroke_width(record.stroke_width)
    points = ((record.x0, record.y0), (record.x1, record.y1), (record.x2, record.y2))
    if len(set(points)) != 3:
        raise VisualSVGSequenceError("A triangle must have three distinct points")
    area2 = (
        record.x0 * (record.y1 - record.y2)
        + record.x1 * (record.y2 - record.y0)
        + record.x2 * (record.y0 - record.y1)
    )
    if area2 == 0:
        raise VisualSVGSequenceError("A triangle must have non-zero area")


def _validate_quad(record: QuadRecordV01) -> None:
    values = (
        record.x0,
        record.y0,
        record.x1,
        record.y1,
        record.x2,
        record.y2,
        record.x3,
        record.y3,
    )
    for field, value in zip(
        ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
        values,
    ):
        _coordinate(value, field)
    _stroke_width(record.stroke_width)
    points = tuple(zip(values[::2], values[1::2]))
    if len(set(points)) != 4:
        raise VisualSVGSequenceError("A quad must have four distinct points")

    def orientation(left, middle, right) -> int:
        value = (
            (middle[1] - left[1]) * (right[0] - middle[0])
            - (middle[0] - left[0]) * (right[1] - middle[1])
        )
        return 0 if value == 0 else 1 if value > 0 else -1

    def on_segment(left, point, right) -> bool:
        return (
            min(left[0], right[0]) <= point[0] <= max(left[0], right[0])
            and min(left[1], right[1]) <= point[1] <= max(left[1], right[1])
        )

    def segments_intersect(a, b, c, d) -> bool:
        orientations = (
            orientation(a, b, c),
            orientation(a, b, d),
            orientation(c, d, a),
            orientation(c, d, b),
        )
        if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
            return True
        return (
            (orientations[0] == 0 and on_segment(a, c, b))
            or (orientations[1] == 0 and on_segment(a, d, b))
            or (orientations[2] == 0 and on_segment(c, a, d))
            or (orientations[3] == 0 and on_segment(c, b, d))
        )

    if segments_intersect(points[0], points[1], points[2], points[3]) or segments_intersect(
        points[1], points[2], points[3], points[0]
    ):
        raise VisualSVGSequenceError("A quad must be a simple non-self-intersecting polygon")
    area2 = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, (*points[1:], points[0]))
    )
    if area2 == 0:
        raise VisualSVGSequenceError("A quad must have non-zero area")


def _validate_text_box(record: TextBoxRecordV01) -> None:
    for field, value in zip(
        ("x0", "y0", "x1", "y1"),
        (record.x0, record.y0, record.x1, record.y1),
    ):
        _coordinate(value, field)
    if record.x0 >= record.x1 or record.y0 >= record.y1:
        raise VisualSVGSequenceError("A text box must have positive width and height")


def _canonical_line(record: LineRecordV01) -> LineRecordV01:
    if (record.y1, record.x1) < (record.y0, record.x0):
        return LineRecordV01(
            record.x1,
            record.y1,
            record.x0,
            record.y0,
            record.stroke_width,
        )
    return record


def _canonical_triangle(record: TriangleRecordV01) -> TriangleRecordV01:
    points = _canonical_cycle(
        ((record.x0, record.y0), (record.x1, record.y1), (record.x2, record.y2))
    )
    return TriangleRecordV01(
        points[0][0],
        points[0][1],
        points[1][0],
        points[1][1],
        points[2][0],
        points[2][1],
        record.stroke_width,
    )


def _canonical_cycle(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    def rotations(values: tuple[tuple[int, int], ...]):
        for index in range(len(values)):
            yield values[index:] + values[:index]

    candidates = [*rotations(points), *rotations(tuple(reversed(points)))]
    return min(candidates, key=lambda values: tuple((y, x) for x, y in values))


def _canonical_quad(record: QuadRecordV01) -> QuadRecordV01:
    points = _canonical_cycle(
        (
            (record.x0, record.y0),
            (record.x1, record.y1),
            (record.x2, record.y2),
            (record.x3, record.y3),
        )
    )
    return QuadRecordV01(
        points[0][0],
        points[0][1],
        points[1][0],
        points[1][1],
        points[2][0],
        points[2][1],
        points[3][0],
        points[3][1],
        record.stroke_width,
    )


def canonicalize_visual_svg_v0_1(
    parsed: ParsedVisualSVGSequenceV01,
) -> ParsedVisualSVGSequenceV01:
    lines = tuple(
        sorted(
            (_canonical_line(record) for record in parsed.lines),
            key=lambda record: (
                record.y0 + record.y1,
                record.x0 + record.x1,
                record.y0,
                record.x0,
                record.y1,
                record.x1,
                record.stroke_width,
            ),
        )
    )
    triangles = tuple(
        sorted(
            (_canonical_triangle(record) for record in parsed.triangles),
            key=lambda record: (
                record.y0 + record.y1 + record.y2,
                record.x0 + record.x1 + record.x2,
                record.y0,
                record.x0,
                record.y1,
                record.x1,
                record.y2,
                record.x2,
                record.stroke_width,
            ),
        )
    )
    quads = tuple(
        sorted(
            (_canonical_quad(record) for record in parsed.quads),
            key=lambda record: (
                record.y0 + record.y1 + record.y2 + record.y3,
                record.x0 + record.x1 + record.x2 + record.x3,
                record.y0,
                record.x0,
                record.y1,
                record.x1,
                record.y2,
                record.x2,
                record.y3,
                record.x3,
                record.stroke_width,
            ),
        )
    )
    text_boxes = tuple(
        sorted(
            parsed.text_boxes,
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
    canonical = ParsedVisualSVGSequenceV01(lines, triangles, quads, text_boxes)
    validate_parsed_visual_svg_v0_1(canonical)
    return canonical


def validate_parsed_visual_svg_v0_1(parsed: ParsedVisualSVGSequenceV01) -> None:
    if not parsed.lines and not parsed.triangles and not parsed.quads:
        raise VisualSVGSequenceError("Visual SVG must contain at least one bond primitive")
    for record in parsed.lines:
        _validate_line(record)
    for record in parsed.triangles:
        _validate_triangle(record)
    for record in parsed.quads:
        _validate_quad(record)
    for record in parsed.text_boxes:
        _validate_text_box(record)


def parsed_visual_svg_from_primitives(payload: dict[str, Any]) -> ParsedVisualSVGSequenceV01:
    if payload.get("schema_version") != "visual-primitives-0.1":
        raise VisualSVGSequenceError(
            f"Unsupported primitive schema: {payload.get('schema_version')}"
        )
    lines = []
    triangles = []
    quads = []
    text_boxes = []
    for primitive in payload.get("primitives", []):
        kind = primitive.get("kind")
        points = primitive.get("quantized_points")
        if kind == "line":
            if not isinstance(points, list) or len(points) != 2:
                raise VisualSVGSequenceError("A line primitive requires two quantized points")
            lines.append(
                LineRecordV01(
                    int(points[0][0]),
                    int(points[0][1]),
                    int(points[1][0]),
                    int(points[1][1]),
                    int(primitive["stroke_width_q"]),
                )
            )
        elif kind == "triangle":
            if not isinstance(points, list) or len(points) != 3:
                raise VisualSVGSequenceError(
                    "A triangle primitive requires three quantized points"
                )
            triangles.append(
                TriangleRecordV01(
                    int(points[0][0]),
                    int(points[0][1]),
                    int(points[1][0]),
                    int(points[1][1]),
                    int(points[2][0]),
                    int(points[2][1]),
                    int(primitive["stroke_width_q"]),
                )
            )
        elif kind == "quad":
            if not isinstance(points, list) or len(points) != 4:
                raise VisualSVGSequenceError(
                    "A quad primitive requires four quantized points"
                )
            quads.append(
                QuadRecordV01(
                    int(points[0][0]),
                    int(points[0][1]),
                    int(points[1][0]),
                    int(points[1][1]),
                    int(points[2][0]),
                    int(points[2][1]),
                    int(points[3][0]),
                    int(points[3][1]),
                    int(primitive["stroke_width_q"]),
                )
            )
        elif kind == "text_box":
            if not isinstance(points, list) or len(points) != 2:
                raise VisualSVGSequenceError(
                    "A text-box primitive requires two quantized corner points"
                )
            text_boxes.append(
                TextBoxRecordV01(
                    int(points[0][0]),
                    int(points[0][1]),
                    int(points[1][0]),
                    int(points[1][1]),
                )
            )
        else:
            raise VisualSVGSequenceError(f"Unsupported primitive kind: {kind}")
    return canonicalize_visual_svg_v0_1(
        ParsedVisualSVGSequenceV01(
            tuple(lines), tuple(triangles), tuple(quads), tuple(text_boxes)
        )
    )


def serialize_parsed_visual_svg_v0_1(parsed: ParsedVisualSVGSequenceV01) -> str:
    parsed = canonicalize_visual_svg_v0_1(parsed)
    lines = [START_TOKEN]
    lines.extend(
        f"<line {record.x0} {record.y0} {record.x1} {record.y1} {record.stroke_width}>"
        for record in parsed.lines
    )
    lines.extend(
        f"<triangle {record.x0} {record.y0} {record.x1} {record.y1} "
        f"{record.x2} {record.y2} {record.stroke_width}>"
        for record in parsed.triangles
    )
    lines.extend(
        f"<quad {record.x0} {record.y0} {record.x1} {record.y1} "
        f"{record.x2} {record.y2} {record.x3} {record.y3} {record.stroke_width}>"
        for record in parsed.quads
    )
    lines.extend(
        f"<text_box {record.x0} {record.y0} {record.x1} {record.y1}>"
        for record in parsed.text_boxes
    )
    lines.append(END_TOKEN)
    return "\n".join(lines) + "\n"


def serialize_visual_svg_v0_1(payload: dict[str, Any]) -> str:
    return serialize_parsed_visual_svg_v0_1(parsed_visual_svg_from_primitives(payload))


def parse_visual_svg_v0_1(sequence: str) -> ParsedVisualSVGSequenceV01:
    if not isinstance(sequence, str):
        raise VisualSVGSequenceError("Visual SVG input must be text")
    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != START_TOKEN or lines[-1] != END_TOKEN:
        raise VisualSVGSequenceError("Visual SVG has an invalid document boundary")
    parsed_lines = []
    triangles = []
    quads = []
    text_boxes = []
    phase = 0
    for line in lines[1:-1]:
        match = LINE_RE.fullmatch(line)
        if match is not None:
            if phase > 0:
                raise VisualSVGSequenceError("Line records must precede other record types")
            parsed_lines.append(LineRecordV01(*(int(value) for value in match.groups())))
            continue
        match = TRIANGLE_RE.fullmatch(line)
        if match is not None:
            if phase > 1:
                raise VisualSVGSequenceError("Triangle records must precede text boxes")
            phase = 1
            triangles.append(TriangleRecordV01(*(int(value) for value in match.groups())))
            continue
        match = QUAD_RE.fullmatch(line)
        if match is not None:
            if phase > 2:
                raise VisualSVGSequenceError("Quad records must precede text boxes")
            phase = 2
            quads.append(QuadRecordV01(*(int(value) for value in match.groups())))
            continue
        match = TEXT_BOX_RE.fullmatch(line)
        if match is not None:
            phase = 3
            text_boxes.append(TextBoxRecordV01(*(int(value) for value in match.groups())))
            continue
        raise VisualSVGSequenceError(f"Unsupported Visual SVG record: {line}")
    parsed = ParsedVisualSVGSequenceV01(
        tuple(parsed_lines),
        tuple(triangles),
        tuple(quads),
        tuple(text_boxes),
    )
    validate_parsed_visual_svg_v0_1(parsed)
    if parsed != canonicalize_visual_svg_v0_1(parsed):
        raise VisualSVGSequenceError("Visual SVG records are not in canonical order")
    return parsed


def records_visual_svg_v0_1(
    parsed: ParsedVisualSVGSequenceV01,
) -> Iterable[LineRecordV01 | TriangleRecordV01 | QuadRecordV01 | TextBoxRecordV01]:
    yield from parsed.lines
    yield from parsed.triangles
    yield from parsed.quads
    yield from parsed.text_boxes
