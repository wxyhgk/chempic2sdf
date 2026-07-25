#!/usr/bin/env python3
"""Extract physical drawing primitives from RDKit's semantic SVG output."""

from __future__ import annotations

from collections import defaultdict
import copy
import math
import re
import xml.etree.ElementTree as ET
from typing import Any

try:
    from .visual_svg_sequence_v0_1 import (
        COORDINATE_MAX,
        STROKE_WIDTH_MAX,
        STROKE_WIDTH_MIN,
        parsed_visual_svg_from_primitives,
    )
except ImportError:
    from visual_svg_sequence_v0_1 import (
        COORDINATE_MAX,
        STROKE_WIDTH_MAX,
        STROKE_WIDTH_MIN,
        parsed_visual_svg_from_primitives,
    )


SCHEMA_VERSION = "visual-primitives-0.1"
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
PATH_TOKEN_RE = re.compile(
    r"[A-Za-z]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


class VisualPrimitiveExtractionError(ValueError):
    pass


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def element_classes(element: ET.Element) -> tuple[str, ...]:
    return tuple(element.attrib.get("class", "").split())


def class_index(classes: tuple[str, ...], prefix: str) -> int | None:
    for value in classes:
        if value.startswith(prefix):
            suffix = value[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


def quantize_coordinate(value: float, extent: int) -> int:
    clipped = min(float(extent), max(0.0, float(value)))
    return int(round(clipped / float(extent) * COORDINATE_MAX))


def dequantize_coordinate(value: int, extent: int) -> float:
    return float(value) / COORDINATE_MAX * float(extent)


def quantize_stroke_width(value: float) -> int:
    return min(
        STROKE_WIDTH_MAX,
        max(STROKE_WIDTH_MIN, int(round(float(value) * 10.0))),
    )


def _style(element: ET.Element) -> dict[str, str]:
    values = {}
    for item in element.attrib.get("style", "").split(";"):
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        values[key.strip()] = value.strip()
    for key in ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin"):
        if key in element.attrib:
            values[key] = element.attrib[key]
    return values


def _stroke_width(element: ET.Element) -> float:
    value = _style(element).get("stroke-width", "1.0")
    if value.endswith("px"):
        value = value[:-2]
    try:
        width = float(value)
    except ValueError as error:
        raise VisualPrimitiveExtractionError(f"Invalid SVG stroke width: {value}") from error
    if width <= 0:
        raise VisualPrimitiveExtractionError(f"SVG stroke width must be positive: {width}")
    return width


def parse_ml_subpaths(path_data: str) -> list[tuple[list[tuple[float, float]], bool]]:
    tokens = PATH_TOKEN_RE.findall(path_data)
    if not tokens:
        raise VisualPrimitiveExtractionError("SVG path has no commands")
    subpaths: list[tuple[list[tuple[float, float]], bool]] = []
    points: list[tuple[float, float]] = []
    command: str | None = None
    cursor = 0

    def finish(closed: bool) -> None:
        nonlocal points
        if points:
            subpaths.append((points, closed))
            points = []

    while cursor < len(tokens):
        token = tokens[cursor]
        if token.isalpha():
            command = token
            cursor += 1
            if command in {"Z", "z"}:
                finish(True)
                command = None
                continue
            if command not in {"M", "L"}:
                raise VisualPrimitiveExtractionError(
                    f"Unsupported SVG path command {command!r}; only absolute M/L/Z are allowed"
                )
        if command is None:
            raise VisualPrimitiveExtractionError("SVG path coordinate lacks a command")
        if cursor + 1 >= len(tokens) or tokens[cursor].isalpha() or tokens[cursor + 1].isalpha():
            raise VisualPrimitiveExtractionError(f"SVG command {command} lacks an x/y pair")
        point = (float(tokens[cursor]), float(tokens[cursor + 1]))
        cursor += 2
        if command == "M":
            finish(False)
            points = [point]
            command = "L"
        else:
            if not points:
                raise VisualPrimitiveExtractionError("SVG L command appears before M")
            points.append(point)
    finish(False)
    return subpaths


def _round_point(point: tuple[float, float]) -> list[float]:
    return [round(point[0], 4), round(point[1], 4)]


def _quantized_point(point: tuple[float, float], width: int, height: int) -> list[int]:
    return [
        quantize_coordinate(point[0], width),
        quantize_coordinate(point[1], height),
    ]


def _line_primitive(
    begin: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int,
    height: int,
    stroke_width: float,
    owner_id: int,
    source_z_index: int,
) -> dict[str, Any]:
    q_begin = _quantized_point(begin, width, height)
    q_end = _quantized_point(end, width, height)
    quantization_adjusted = False
    if q_begin == q_end:
        dx = (end[0] - begin[0]) / width
        dy = (end[1] - begin[1]) / height
        axis = 0 if abs(dx) >= abs(dy) else 1
        delta = 1 if (dx if axis == 0 else dy) >= 0 else -1
        candidate = q_end[axis] + delta
        if 0 <= candidate <= COORDINATE_MAX:
            q_end[axis] = candidate
        else:
            q_begin[axis] -= delta
        quantization_adjusted = True
    if (q_end[1], q_end[0]) < (q_begin[1], q_begin[0]):
        begin, end = end, begin
        q_begin, q_end = q_end, q_begin
    return {
        "kind": "line",
        "owner_kind": "bond",
        "owner_id": owner_id,
        "points": [_round_point(begin), _round_point(end)],
        "quantized_points": [q_begin, q_end],
        "stroke_width": round(stroke_width, 4),
        "stroke_width_q": quantize_stroke_width(stroke_width),
        "quantization_adjusted": quantization_adjusted,
        "source_z_index": source_z_index,
    }


def _polygon_primitive(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    stroke_width: float,
    owner_id: int,
    source_z_index: int,
) -> dict[str, Any]:
    indexed = [
        (_quantized_point(point, width, height), point)
        for point in points
    ]
    def rotations(values):
        for index in range(len(values)):
            yield values[index:] + values[:index]

    candidates = [*rotations(indexed), *rotations(list(reversed(indexed)))]
    indexed = min(
        candidates,
        key=lambda values: tuple((item[0][1], item[0][0]) for item in values),
    )
    quantized = [item[0] for item in indexed]
    raw = [item[1] for item in indexed]
    area2 = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(quantized, (*quantized[1:], quantized[0]))
    )
    if area2 == 0:
        raise VisualPrimitiveExtractionError("Polygon collapsed after quantization")
    return {
        "kind": "triangle" if len(points) == 3 else "quad",
        "owner_kind": "bond",
        "owner_id": owner_id,
        "points": [_round_point(point) for point in raw],
        "quantized_points": quantized,
        "stroke_width": round(stroke_width, 4),
        "stroke_width_q": quantize_stroke_width(stroke_width),
        "source_z_index": source_z_index,
    }


def _text_box_primitive(
    atom: dict[str, Any],
    *,
    width: int,
    height: int,
    source_z_index: int,
) -> dict[str, Any]:
    x0, y0, x1, y1 = (float(value) for value in atom["bbox"])
    q0 = _quantized_point((x0, y0), width, height)
    q1 = _quantized_point((x1, y1), width, height)
    if q0[0] >= q1[0] or q0[1] >= q1[1]:
        raise VisualPrimitiveExtractionError(
            f"Text box collapsed after quantization for atom {atom['atom_id']}"
        )
    return {
        "kind": "text_box",
        "owner_kind": "atom",
        "owner_id": int(atom["atom_id"]),
        "text": str(atom.get("text", "")),
        "points": [[round(x0, 4), round(y0, 4)], [round(x1, 4), round(y1, 4)]],
        "quantized_points": [q0, q1],
        "source_z_index": source_z_index,
    }


def _primitive_sort_key(primitive: dict[str, Any]) -> tuple:
    rank = {"line": 0, "triangle": 1, "quad": 2, "text_box": 3}[
        primitive["kind"]
    ]
    points = primitive["quantized_points"]
    center_y = sum(point[1] for point in points)
    center_x = sum(point[0] for point in points)
    flattened = tuple(value for point in points for value in (point[1], point[0]))
    return (rank, center_y, center_x, flattened, primitive.get("stroke_width_q", 0))


def extract_visual_primitives_v0_1(
    svg: bytes,
    layers: dict[str, Any],
) -> dict[str, Any]:
    root = ET.fromstring(svg)
    width = int(layers["width"])
    height = int(layers["height"])
    primitives: list[dict[str, Any]] = []
    atom_z_indices: dict[int, int] = {}
    for source_z_index, element in enumerate(list(root)):
        classes = element_classes(element)
        bond_id = class_index(classes, "bond-")
        atom_id = class_index(classes, "atom-")
        if bond_id is None and atom_id is not None:
            atom_z_indices.setdefault(atom_id, source_z_index)
        if bond_id is None:
            continue
        if local_name(element) != "path" or not element.attrib.get("d"):
            raise VisualPrimitiveExtractionError(
                f"Bond {bond_id} uses unsupported SVG element {local_name(element)}"
            )
        stroke_width = _stroke_width(element)
        for points, closed in parse_ml_subpaths(element.attrib["d"]):
            if closed and len(points) > 1 and points[0] == points[-1]:
                points = points[:-1]
            if closed:
                if len(points) not in {3, 4}:
                    raise VisualPrimitiveExtractionError(
                        f"Bond {bond_id} closed path has {len(points)} vertices, expected 3 or 4"
                    )
                primitives.append(
                    _polygon_primitive(
                        points,
                        width=width,
                        height=height,
                        stroke_width=stroke_width,
                        owner_id=bond_id,
                        source_z_index=source_z_index,
                    )
                )
                continue
            if len(points) < 2:
                raise VisualPrimitiveExtractionError(
                    f"Bond {bond_id} open path has fewer than two points"
                )
            for begin, end in zip(points, points[1:]):
                primitives.append(
                    _line_primitive(
                        begin,
                        end,
                        width=width,
                        height=height,
                        stroke_width=stroke_width,
                        owner_id=bond_id,
                        source_z_index=source_z_index,
                    )
                )

    for atom in layers["atoms"]:
        if not atom.get("visible_text"):
            continue
        if atom.get("bbox") is None:
            raise VisualPrimitiveExtractionError(
                f"Visible atom {atom['atom_id']} lacks a text bounding box"
            )
        primitives.append(
            _text_box_primitive(
                atom,
                width=width,
                height=height,
                source_z_index=atom_z_indices.get(int(atom["atom_id"]), len(root)),
            )
        )

    primitives.sort(key=_primitive_sort_key)
    by_owner: dict[tuple[str, int], list[int]] = defaultdict(list)
    for primitive_id, primitive in enumerate(primitives):
        primitive["primitive_id"] = primitive_id
        primitive["canonical_index"] = primitive_id
        by_owner[(primitive["owner_kind"], int(primitive["owner_id"]))].append(
            primitive_id
        )

    bond_groups = []
    for bond in sorted(layers["bonds"], key=lambda item: int(item["bond_id"])):
        bond_id = int(bond["bond_id"])
        primitive_ids = by_owner.get(("bond", bond_id), [])
        if not primitive_ids:
            raise VisualPrimitiveExtractionError(f"Bond {bond_id} has no physical primitive")
        bond_groups.append(
            {
                "bond_id": bond_id,
                "begin_atom": int(bond["begin_atom"]),
                "end_atom": int(bond["end_atom"]),
                "chemical_type": str(bond["chemical_type"]),
                "display_type": str(bond["display_type"]),
                "direction": str(bond["direction"]),
                "primitive_ids": primitive_ids,
            }
        )

    text_instances = []
    for atom in sorted(layers["atoms"], key=lambda item: int(item["atom_id"])):
        if not atom.get("visible_text"):
            continue
        atom_id = int(atom["atom_id"])
        primitive_ids = by_owner.get(("atom", atom_id), [])
        if len(primitive_ids) != 1:
            raise VisualPrimitiveExtractionError(
                f"Visible atom {atom_id} has {len(primitive_ids)} text-box primitives"
            )
        text_instances.append(
            {
                "atom_id": atom_id,
                "text": str(atom.get("text", "")),
                "bbox": copy.deepcopy(atom["bbox"]),
                "primitive_ids": primitive_ids,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_key": str(layers["sample_key"]),
        "molecule_id": str(layers["molecule_id"]),
        "width": width,
        "height": height,
        "coordinate_space": "image_pixels",
        "coordinate_quantization": COORDINATE_MAX,
        "stroke_width_scale": 10,
        "bond_groups": bond_groups,
        "text_instances": text_instances,
        "primitives": primitives,
    }
    validate_visual_primitive_references(payload)
    parsed_visual_svg_from_primitives(payload)
    return payload


def validate_visual_primitive_references(payload: dict[str, Any]) -> None:
    primitives = payload.get("primitives", [])
    primitive_ids = [int(item["primitive_id"]) for item in primitives]
    if primitive_ids != list(range(len(primitives))):
        raise VisualPrimitiveExtractionError("Primitive IDs must be contiguous canonical order")
    if [int(item["canonical_index"]) for item in primitives] != primitive_ids:
        raise VisualPrimitiveExtractionError("Canonical indices must match primitive IDs")
    referenced = []
    bond_ids = set()
    for group in payload.get("bond_groups", []):
        bond_id = int(group["bond_id"])
        if bond_id in bond_ids:
            raise VisualPrimitiveExtractionError(f"Duplicate bond group {bond_id}")
        bond_ids.add(bond_id)
        for primitive_id in group["primitive_ids"]:
            primitive = primitives[int(primitive_id)]
            if primitive["owner_kind"] != "bond" or int(primitive["owner_id"]) != bond_id:
                raise VisualPrimitiveExtractionError(
                    f"Bond group {bond_id} references foreign primitive {primitive_id}"
                )
            referenced.append(int(primitive_id))
    atom_ids = set()
    for instance in payload.get("text_instances", []):
        atom_id = int(instance["atom_id"])
        if atom_id in atom_ids:
            raise VisualPrimitiveExtractionError(f"Duplicate text instance {atom_id}")
        atom_ids.add(atom_id)
        for primitive_id in instance["primitive_ids"]:
            primitive = primitives[int(primitive_id)]
            if primitive["owner_kind"] != "atom" or int(primitive["owner_id"]) != atom_id:
                raise VisualPrimitiveExtractionError(
                    f"Text instance {atom_id} references foreign primitive {primitive_id}"
                )
            referenced.append(int(primitive_id))
    if sorted(referenced) != primitive_ids:
        raise VisualPrimitiveExtractionError(
            "Every primitive must be referenced exactly once by its owner group"
        )


def primitives_to_bond_svg(payload: dict[str, Any]) -> bytes:
    parsed_visual_svg_from_primitives(payload)
    width = int(payload["width"])
    height = int(payload["height"])
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": f"{width}px",
            "height": f"{height}px",
            "viewBox": f"0 0 {width} {height}",
        },
    )
    for primitive in payload["primitives"]:
        kind = primitive["kind"]
        if kind == "text_box":
            continue
        points = [
            (
                dequantize_coordinate(int(point[0]), width),
                dequantize_coordinate(int(point[1]), height),
            )
            for point in primitive["quantized_points"]
        ]
        stroke_width = int(primitive["stroke_width_q"]) / 10.0
        if kind == "line":
            path_data = (
                f"M {points[0][0]:.4f},{points[0][1]:.4f} "
                f"L {points[1][0]:.4f},{points[1][1]:.4f}"
            )
            style = (
                "fill:none;fill-rule:evenodd;stroke:#000000;"
                f"stroke-width:{stroke_width:.1f}px;stroke-linecap:butt;"
                "stroke-linejoin:miter;stroke-opacity:1"
            )
        elif kind in {"triangle", "quad"}:
            path_data = "M " + " L ".join(
                f"{point[0]:.4f},{point[1]:.4f}" for point in points
            ) + " Z"
            style = (
                "fill:#000000;fill-rule:evenodd;fill-opacity:1;stroke:#000000;"
                f"stroke-width:{stroke_width:.1f}px;stroke-linecap:butt;"
                "stroke-linejoin:miter;stroke-miterlimit:10;stroke-opacity:1"
            )
        else:
            raise VisualPrimitiveExtractionError(f"Unsupported primitive kind: {kind}")
        ET.SubElement(root, f"{{{SVG_NS}}}path", {"d": path_data, "style": style})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def primitive_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = defaultdict(int)
    for primitive in payload["primitives"]:
        counts[str(primitive["kind"])] += 1
    return dict(sorted(counts.items()))
