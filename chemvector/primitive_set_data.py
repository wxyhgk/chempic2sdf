"""Padded unordered primitive-set targets for Visual SVG training."""

from __future__ import annotations

from typing import Any

import torch

from chemvector.visual_svg_data import VisualSVGCollator


PRIMITIVE_SET_QUERY_COUNT = 288
COORDINATE_QUANTIZATION = 2047.0
STROKE_WIDTH_QUANTIZATION = 63.0

NO_OBJECT_TYPE_ID = 0
PRIMITIVE_TYPE_TO_ID = {
    "line": 1,
    "triangle": 2,
    "quad": 3,
    "text_box": 4,
}
PRIMITIVE_ID_TO_TYPE = {
    type_id: kind for kind, type_id in PRIMITIVE_TYPE_TO_ID.items()
}

_POINT_COUNTS = {
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "text_box": 2,
}


def _quantized_points(
    primitive: dict[str, Any],
    *,
    expected_count: int,
) -> list[tuple[float, float]]:
    points = primitive.get("quantized_points")
    if not isinstance(points, list) or len(points) != expected_count:
        raise ValueError(
            f"{primitive.get('kind', 'primitive')} requires {expected_count} "
            "quantized points"
        )
    normalized = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each quantized point must contain x and y")
        x, y = point
        if not (0 <= x <= COORDINATE_QUANTIZATION):
            raise ValueError(f"Quantized x coordinate is out of range: {x}")
        if not (0 <= y <= COORDINATE_QUANTIZATION):
            raise ValueError(f"Quantized y coordinate is out of range: {y}")
        normalized.append(
            (
                float(x) / COORDINATE_QUANTIZATION,
                float(y) / COORDINATE_QUANTIZATION,
            )
        )
    return normalized


def _primitive_geometry(primitive: dict[str, Any]) -> tuple[list[float], int]:
    kind = primitive.get("kind")
    if kind not in PRIMITIVE_TYPE_TO_ID:
        raise ValueError(f"Unsupported primitive kind: {kind!r}")
    points = _quantized_points(
        primitive,
        expected_count=_POINT_COUNTS[kind],
    )
    if kind == "text_box":
        (x0, y0), (x1, y1) = points
        if x1 <= x0 or y1 <= y0:
            raise ValueError("text_box quantized points must define a positive box")
        return [
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0,
            x1 - x0,
            y1 - y0,
        ], 4
    flattened = [coordinate for point in points for coordinate in point]
    return flattened, len(flattened)


def build_primitive_set_target(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Encode one Visual SVG primitive payload into the frozen 288-query target."""

    primitives = payload.get("primitives")
    if not isinstance(primitives, list):
        raise ValueError("Primitive payload must contain a primitives list")
    primitive_count = len(primitives)
    if primitive_count > PRIMITIVE_SET_QUERY_COUNT:
        raise ValueError(
            f"Primitive count {primitive_count} exceeds frozen query capacity "
            f"{PRIMITIVE_SET_QUERY_COUNT}"
        )

    types = torch.full(
        (PRIMITIVE_SET_QUERY_COUNT,),
        NO_OBJECT_TYPE_ID,
        dtype=torch.long,
    )
    geometry = torch.zeros(
        (PRIMITIVE_SET_QUERY_COUNT, 8),
        dtype=torch.float32,
    )
    geometry_mask = torch.zeros(
        (PRIMITIVE_SET_QUERY_COUNT, 8),
        dtype=torch.bool,
    )
    stroke = torch.zeros(PRIMITIVE_SET_QUERY_COUNT, dtype=torch.float32)
    has_stroke = torch.zeros(PRIMITIVE_SET_QUERY_COUNT, dtype=torch.bool)
    target_mask = torch.zeros(PRIMITIVE_SET_QUERY_COUNT, dtype=torch.bool)

    for index, primitive in enumerate(primitives):
        kind = primitive.get("kind")
        values, value_count = _primitive_geometry(primitive)
        types[index] = PRIMITIVE_TYPE_TO_ID[kind]
        geometry[index, :value_count] = torch.tensor(values, dtype=torch.float32)
        geometry_mask[index, :value_count] = True
        target_mask[index] = True
        if kind != "text_box":
            stroke_width = primitive.get("stroke_width_q")
            if not isinstance(stroke_width, (int, float)) or not (
                1 <= stroke_width <= STROKE_WIDTH_QUANTIZATION
            ):
                raise ValueError(
                    f"Invalid quantized stroke width for {kind}: {stroke_width!r}"
                )
            stroke[index] = float(stroke_width) / STROKE_WIDTH_QUANTIZATION
            has_stroke[index] = True

    return {
        "types": types,
        "geometry": geometry,
        "geometry_mask": geometry_mask,
        "stroke": stroke,
        "has_stroke": has_stroke,
        "target_mask": target_mask,
    }


class PrimitiveSetCollator(VisualSVGCollator):
    """Collate raster images and frozen-capacity primitive-set targets."""

    def __init__(self, *, image_size: int = 768) -> None:
        # Tokenization is intentionally absent, but inheriting the image path keeps
        # normalization identical to the established Visual SVG loader.
        super().__init__(tokenizer=None, image_size=image_size)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        targets = [
            build_primitive_set_target(sample["primitives"])
            for sample in samples
        ]
        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "types": torch.stack([target["types"] for target in targets]),
            "geometry": torch.stack([target["geometry"] for target in targets]),
            "geometry_mask": torch.stack(
                [target["geometry_mask"] for target in targets]
            ),
            "stroke": torch.stack([target["stroke"] for target in targets]),
            "has_stroke": torch.stack(
                [target["has_stroke"] for target in targets]
            ),
            "target_mask": torch.stack(
                [target["target_mask"] for target in targets]
            ),
            "keys": [sample["key"] for sample in samples],
            "meta": [sample["meta"] for sample in samples],
            "bond_layers": [sample["bond_layer"] for sample in samples],
        }
