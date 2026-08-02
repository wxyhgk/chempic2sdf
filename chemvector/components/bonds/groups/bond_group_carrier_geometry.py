"""Typed carrier geometry codec for Visual SVG v0.2 bond groups.

The codec keeps group geometry compact and exposes only tensor fields suitable
for batching.  Coordinates and geometric distances are normalized by q2047;
stroke width is normalized by 63.  Decoding uses only PyTorch operations so it
can sit directly in a training loss.

Line-group parameter layouts:

* single_line: C0, C1
* parallel_2: two complete canonical line members
* parallel_3: C0, C1, half spacing
* hashed_wedge: narrow center, wide center, narrow/wide half widths

Carrier endpoints are physically unoriented.  ``carrier_parameter_variants``
returns endpoint, line-order, cycle and reversal equivalents so matching does
not introduce an arbitrary ordering seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from Data.sdf.visual_svg_v0_2 import (
    BondGroupRecordV02,
    LineMemberV02,
    QuadMemberV02,
    TriangleMemberV02,
)


NO_OBJECT = 0
SINGLE_LINE = 1
PARALLEL_2 = 2
PARALLEL_3 = 3
HASHED_WEDGE = 4
SOLID_TRIANGLE = 5
SOLID_QUAD = 6

CARRIER_KIND_TO_ID = {
    "single_line": SINGLE_LINE,
    "parallel_2": PARALLEL_2,
    "parallel_3": PARALLEL_3,
    "hashed_wedge": HASHED_WEDGE,
    "solid_triangle": SOLID_TRIANGLE,
    "solid_quad": SOLID_QUAD,
}
CARRIER_ID_TO_KIND = {value: key for key, value in CARRIER_KIND_TO_ID.items()}

CARRIER_POINT_WIDTH = 8
CARRIER_POSITIVE_WIDTH = 2
CARRIER_SIGNED_WIDTH = 2
CARRIER_VARIANT_COUNT = 8
DEFAULT_COORDINATE_SCALE = 2047.0
DEFAULT_STROKE_SCALE = 63.0

_POINT_COUNTS = (0, 4, 8, 4, 4, 6, 8)
_POSITIVE_COUNTS = (0, 0, 0, 1, 2, 0, 0)
_SIGNED_COUNTS = (0, 0, 0, 0, 0, 0, 0)
_VARIANT_COUNTS = (1, 2, 8, 2, 2, 6, 8)


@dataclass(frozen=True, slots=True)
class BondGroupCarrierEncoding:
    """One fixed-width typed carrier target."""

    kind: str
    kind_id: torch.Tensor
    carrier_points: torch.Tensor
    carrier_point_mask: torch.Tensor
    positive_params: torch.Tensor
    positive_param_mask: torch.Tensor
    signed_params: torch.Tensor
    signed_param_mask: torch.Tensor
    group_stroke: torch.Tensor
    member_count: torch.Tensor

    def as_target_dict(self) -> dict[str, torch.Tensor]:
        return {
            "kind_id": self.kind_id,
            "carrier_points": self.carrier_points,
            "carrier_point_mask": self.carrier_point_mask,
            "positive_params": self.positive_params,
            "positive_param_mask": self.positive_param_mask,
            "signed_params": self.signed_params,
            "signed_param_mask": self.signed_param_mask,
            "group_stroke": self.group_stroke,
            "member_count": self.member_count,
        }


@dataclass(frozen=True, slots=True)
class BondGroupCarrierVariants:
    """Equivalent parameterizations used by matching and regression loss."""

    carrier_points: torch.Tensor
    positive_params: torch.Tensor
    signed_params: torch.Tensor
    variant_mask: torch.Tensor
    carrier_point_mask: torch.Tensor
    positive_param_mask: torch.Tensor
    signed_param_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecodedBondGroupCarrier:
    """Padded physical-member tensors expanded from carrier parameters."""

    member_geometry: torch.Tensor
    member_geometry_mask: torch.Tensor
    member_mask: torch.Tensor
    member_stroke: torch.Tensor


def _validate_scales(coordinate_scale: float, stroke_scale: float) -> None:
    if coordinate_scale <= 0.0 or stroke_scale <= 0.0:
        raise ValueError("Carrier coordinate and stroke scales must be positive")


def _validate_kind_ids(kind_ids: torch.Tensor) -> None:
    if kind_ids.dtype == torch.bool or kind_ids.is_floating_point():
        raise ValueError("Carrier kind ids must use an integer tensor dtype")
    if bool(((kind_ids < NO_OBJECT) | (kind_ids > SOLID_QUAD)).any()):
        raise ValueError("Carrier kind id is outside the supported range 0..6")


def carrier_parameter_masks(
    kind_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active point, positive and signed parameter masks."""

    _validate_kind_ids(kind_ids)
    device = kind_ids.device
    point_counts = torch.as_tensor(_POINT_COUNTS, device=device)[kind_ids.long()]
    positive_counts = torch.as_tensor(_POSITIVE_COUNTS, device=device)[kind_ids.long()]
    signed_counts = torch.as_tensor(_SIGNED_COUNTS, device=device)[kind_ids.long()]
    point_mask = (
        torch.arange(CARRIER_POINT_WIDTH, device=device)
        < point_counts.unsqueeze(-1)
    )
    positive_mask = (
        torch.arange(CARRIER_POSITIVE_WIDTH, device=device)
        < positive_counts.unsqueeze(-1)
    )
    signed_mask = (
        torch.arange(CARRIER_SIGNED_WIDTH, device=device)
        < signed_counts.unsqueeze(-1)
    )
    return point_mask, positive_mask, signed_mask


def _coordinate_permutation(point_order: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for point in point_order:
        result.extend((2 * point, 2 * point + 1))
    result.extend(range(2 * len(point_order), CARRIER_POINT_WIDTH))
    return tuple(result)


def _variant_lookup_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity_points = tuple(range(CARRIER_POINT_WIDTH))
    reverse_line = (2, 3, 0, 1, 4, 5, 6, 7)
    point_rows = [
        [identity_points for _ in range(CARRIER_VARIANT_COUNT)]
        for _ in range(SOLID_QUAD + 1)
    ]
    for kind_id in (SINGLE_LINE, PARALLEL_3, HASHED_WEDGE):
        point_rows[kind_id][1] = reverse_line

    parallel_2_orders = []
    for swap_lines in (False, True):
        for flip_first in (False, True):
            for flip_second in (False, True):
                first = (1, 0) if flip_first else (0, 1)
                second = (3, 2) if flip_second else (2, 3)
                lines = (second, first) if swap_lines else (first, second)
                parallel_2_orders.append((*lines[0], *lines[1]))
    for index, order in enumerate(parallel_2_orders):
        point_rows[PARALLEL_2][index] = _coordinate_permutation(order)

    triangle_orders = (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
        (1, 0, 2),
        (0, 2, 1),
    )
    for index, order in enumerate(triangle_orders):
        point_rows[SOLID_TRIANGLE][index] = _coordinate_permutation(order)

    quad_orders = (
        (0, 1, 2, 3),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
        (3, 2, 1, 0),
        (2, 1, 0, 3),
        (1, 0, 3, 2),
        (0, 3, 2, 1),
    )
    for index, order in enumerate(quad_orders):
        point_rows[SOLID_QUAD][index] = _coordinate_permutation(order)

    identity_aux = tuple(range(CARRIER_POSITIVE_WIDTH))
    positive_rows = [
        [identity_aux for _ in range(CARRIER_VARIANT_COUNT)]
        for _ in range(SOLID_QUAD + 1)
    ]
    positive_rows[HASHED_WEDGE][1] = (1, 0)

    signed_rows = [
        [identity_aux for _ in range(CARRIER_VARIANT_COUNT)]
        for _ in range(SOLID_QUAD + 1)
    ]
    return (
        torch.tensor(point_rows, dtype=torch.long),
        torch.tensor(positive_rows, dtype=torch.long),
        torch.tensor(signed_rows, dtype=torch.long),
    )


_POINT_VARIANT_LOOKUP, _POSITIVE_VARIANT_LOOKUP, _SIGNED_VARIANT_LOOKUP = (
    _variant_lookup_tables()
)


def carrier_parameter_variants(
    carrier_points: torch.Tensor,
    positive_params: torch.Tensor,
    signed_params: torch.Tensor,
    kind_ids: torch.Tensor,
) -> BondGroupCarrierVariants:
    """Build direct/reversed or polygon cycle/reverse target variants.

    Output tensors insert a fixed variant dimension immediately before the
    parameter dimension.  Single, triple and hashed groups activate two
    variants, parallel_2 activates eight, triangles six, and quads eight.
    """

    prefix = carrier_points.shape[:-1]
    if carrier_points.shape[-1:] != (CARRIER_POINT_WIDTH,):
        raise ValueError("Carrier points must end with width 8")
    if positive_params.shape != (*prefix, CARRIER_POSITIVE_WIDTH):
        raise ValueError("Carrier positive parameters must end with width 2")
    if signed_params.shape != (*prefix, CARRIER_SIGNED_WIDTH):
        raise ValueError("Carrier signed parameters must end with width 2")
    if kind_ids.shape != prefix:
        raise ValueError("Carrier kind ids must match the parameter prefix shape")
    _validate_kind_ids(kind_ids)

    point_lookup = _POINT_VARIANT_LOOKUP.to(kind_ids.device)[kind_ids.long()]
    positive_lookup = _POSITIVE_VARIANT_LOOKUP.to(kind_ids.device)[kind_ids.long()]
    signed_lookup = _SIGNED_VARIANT_LOOKUP.to(kind_ids.device)[kind_ids.long()]
    points = torch.gather(
        carrier_points.unsqueeze(-2).expand(*prefix, CARRIER_VARIANT_COUNT, -1),
        -1,
        point_lookup,
    )
    positive = torch.gather(
        positive_params.unsqueeze(-2).expand(*prefix, CARRIER_VARIANT_COUNT, -1),
        -1,
        positive_lookup,
    )
    signed = torch.gather(
        signed_params.unsqueeze(-2).expand(*prefix, CARRIER_VARIANT_COUNT, -1),
        -1,
        signed_lookup,
    )
    counts = torch.as_tensor(_VARIANT_COUNTS, device=kind_ids.device)[kind_ids.long()]
    variant_mask = (
        torch.arange(CARRIER_VARIANT_COUNT, device=kind_ids.device)
        < counts.unsqueeze(-1)
    )
    point_mask, positive_mask, signed_mask = carrier_parameter_masks(kind_ids)
    return BondGroupCarrierVariants(
        carrier_points=points,
        positive_params=positive,
        signed_params=signed,
        variant_mask=variant_mask,
        carrier_point_mask=point_mask,
        positive_param_mask=positive_mask,
        signed_param_mask=signed_mask,
    )


def _line_tensor(group: BondGroupRecordV02, expected_count: int | None) -> torch.Tensor:
    if expected_count is not None and len(group.members) != expected_count:
        raise ValueError(
            f"{group.kind} requires {expected_count} members, got {len(group.members)}"
        )
    if not group.members or not all(
        isinstance(member, LineMemberV02) for member in group.members
    ):
        raise ValueError(f"{group.kind} requires line members")
    return torch.tensor(
        [
            ((member.x0, member.y0), (member.x1, member.y1))
            for member in group.members
        ],
        dtype=torch.float64,
    )


def _line_axis(lines: torch.Tensor) -> torch.Tensor:
    vectors = lines[:, 1] - lines[:, 0]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    if bool((lengths <= 0.0).any()):
        raise ValueError("Carrier line members must have non-zero length")
    units = vectors / lengths[:, None]
    scatter = units.transpose(0, 1) @ units
    _, eigenvectors = torch.linalg.eigh(scatter)
    return eigenvectors[:, -1]


def _center_axis(centers: torch.Tensor) -> torch.Tensor:
    centered = centers - centers.mean(dim=0)
    scatter = centered.transpose(0, 1) @ centered
    values, vectors = torch.linalg.eigh(scatter)
    if float(values[-1]) <= 0.0:
        raise ValueError("Hashed-wedge member centers must span a non-zero axis")
    return vectors[:, -1]


def _orient_lines(lines: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    projection0 = lines[:, 0] @ axis
    projection1 = lines[:, 1] @ axis
    reverse = projection1 < projection0
    result = lines.clone()
    result[reverse] = result[reverse].flip(1)
    return result


def _lexicographic_point_key(point: torch.Tensor) -> tuple[float, float]:
    return float(point[1]), float(point[0])


def _encode_single(lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left, right = lines[0]
    if _lexicographic_point_key(right) < _lexicographic_point_key(left):
        left, right = right, left
    points = torch.cat((left, right))
    return points, torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)


def _encode_parallel_2(
    lines: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canonical = []
    for line in lines:
        start, end = line
        if _lexicographic_point_key(end) < _lexicographic_point_key(start):
            start, end = end, start
        canonical.append(torch.stack((start, end)))
    canonical.sort(
        key=lambda line: (
            float(line[:, 1].mean()),
            float(line[:, 0].mean()),
            *_lexicographic_point_key(line[0]),
            *_lexicographic_point_key(line[1]),
        )
    )
    return (
        torch.cat((canonical[0].flatten(), canonical[1].flatten())),
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
    )


def _encode_parallel_3(
    lines: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = _line_axis(lines)
    oriented = _orient_lines(lines, axis)
    raw_start = oriented[:, 0].mean(dim=0)
    raw_end = oriented[:, 1].mean(dim=0)
    if _lexicographic_point_key(raw_end) < _lexicographic_point_key(raw_start):
        axis = -axis
        oriented = _orient_lines(lines, axis)
        raw_start = oriented[:, 0].mean(dim=0)
        raw_end = oriented[:, 1].mean(dim=0)
    center = (raw_start + raw_end) / 2.0
    half_length = torch.dot(raw_end - raw_start, axis) / 2.0
    start = center - half_length * axis
    end = center + half_length * axis
    normal = torch.stack((-axis[1], axis[0]))
    offsets = (oriented.mean(dim=1) - center) @ normal
    half_spacing = (offsets.max() - offsets.min()) / 2.0
    if float(half_spacing) <= 0.0:
        raise ValueError("parallel_3 carrier has non-positive half spacing")
    return (
        torch.cat((start, end)),
        half_spacing.reshape(1),
        torch.empty(0, dtype=torch.float64),
    )


def _encode_hashed_wedge(
    lines: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(lines) < 2:
        raise ValueError("hashed_wedge requires at least two line members")
    centers = lines.mean(dim=1)
    half_widths = torch.linalg.vector_norm(lines[:, 1] - lines[:, 0], dim=-1) / 2.0
    axis = _center_axis(centers)
    order = torch.argsort(centers @ axis)
    ordered_centers = centers[order]
    ordered_widths = half_widths[order]
    count = len(lines)
    t = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    design = torch.stack((torch.ones_like(t), t), dim=-1)

    width_solution = torch.linalg.lstsq(design, ordered_widths[:, None]).solution[:, 0]
    if float(width_solution[1]) < 0.0:
        ordered_centers = ordered_centers.flip(0)
        ordered_widths = ordered_widths.flip(0)
        width_solution = torch.linalg.lstsq(
            design,
            ordered_widths[:, None],
        ).solution[:, 0]
    center_solution = torch.linalg.lstsq(design, ordered_centers).solution
    start = center_solution[0]
    end = center_solution[0] + center_solution[1]
    width_start = width_solution[0]
    width_end = width_solution[0] + width_solution[1]
    if float(width_start) <= 0.0 or float(width_end) <= float(width_start):
        raise ValueError("hashed_wedge widths must grow from the canonical narrow end")
    return (
        torch.cat((start, end)),
        torch.stack((width_start, width_end)),
        torch.empty(0, dtype=torch.float64),
    )


def _polygon_coordinates(group: BondGroupRecordV02) -> torch.Tensor:
    if len(group.members) != 1:
        raise ValueError(f"{group.kind} requires exactly one polygon member")
    member = group.members[0]
    if group.kind == "solid_triangle" and isinstance(member, TriangleMemberV02):
        values = (member.x0, member.y0, member.x1, member.y1, member.x2, member.y2)
    elif group.kind == "solid_quad" and isinstance(member, QuadMemberV02):
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
    else:
        raise ValueError(f"{group.kind} has an incompatible polygon member")
    return torch.tensor(values, dtype=torch.float64)


def encode_bond_group_carrier(
    group: BondGroupRecordV02,
    *,
    coordinate_scale: float = DEFAULT_COORDINATE_SCALE,
    stroke_scale: float = DEFAULT_STROKE_SCALE,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> BondGroupCarrierEncoding:
    """Encode one Visual SVG v0.2 group into fixed-width carrier tensors."""

    _validate_scales(coordinate_scale, stroke_scale)
    if group.kind not in CARRIER_KIND_TO_ID:
        raise ValueError(f"Unsupported carrier group kind: {group.kind!r}")
    if not group.members:
        raise ValueError("Carrier groups must contain at least one member")
    strokes = {int(member.stroke_width) for member in group.members}
    if len(strokes) != 1:
        raise ValueError("All members in one carrier group must share one stroke width")
    stroke = next(iter(strokes))
    if not 1 <= stroke <= int(stroke_scale):
        raise ValueError(f"Carrier group stroke width is outside 1..{int(stroke_scale)}")

    if group.kind == "single_line":
        points, positive, signed = _encode_single(_line_tensor(group, 1))
    elif group.kind == "parallel_2":
        points, positive, signed = _encode_parallel_2(_line_tensor(group, 2))
    elif group.kind == "parallel_3":
        points, positive, signed = _encode_parallel_3(_line_tensor(group, 3))
    elif group.kind == "hashed_wedge":
        points, positive, signed = _encode_hashed_wedge(_line_tensor(group, None))
    else:
        points = _polygon_coordinates(group)
        positive = torch.empty(0, dtype=torch.float64)
        signed = torch.empty(0, dtype=torch.float64)

    kind_id = torch.tensor(CARRIER_KIND_TO_ID[group.kind], dtype=torch.long, device=device)
    point_mask, positive_mask, signed_mask = carrier_parameter_masks(kind_id)
    carrier_points = torch.zeros(CARRIER_POINT_WIDTH, dtype=torch.float64)
    carrier_points[: len(points)] = points / coordinate_scale
    positive_params = torch.zeros(CARRIER_POSITIVE_WIDTH, dtype=torch.float64)
    positive_params[: len(positive)] = positive / coordinate_scale
    signed_params = torch.zeros(CARRIER_SIGNED_WIDTH, dtype=torch.float64)
    signed_params[: len(signed)] = signed / coordinate_scale
    return BondGroupCarrierEncoding(
        kind=group.kind,
        kind_id=kind_id,
        carrier_points=carrier_points.to(device=device, dtype=dtype),
        carrier_point_mask=point_mask,
        positive_params=positive_params.to(device=device, dtype=dtype),
        positive_param_mask=positive_mask,
        signed_params=signed_params.to(device=device, dtype=dtype),
        signed_param_mask=signed_mask,
        group_stroke=torch.tensor(stroke / stroke_scale, dtype=dtype, device=device),
        member_count=torch.tensor(len(group.members), dtype=torch.long, device=device),
    )


def _validate_decode_inputs(
    carrier_points: torch.Tensor,
    positive_params: torch.Tensor,
    signed_params: torch.Tensor,
    kind_ids: torch.Tensor,
    member_counts: torch.Tensor,
    group_strokes: torch.Tensor | None,
    max_members: int,
) -> tuple[int, ...]:
    if max_members < 1:
        raise ValueError("Carrier max_members must be positive")
    prefix = carrier_points.shape[:-1]
    if carrier_points.shape[-1:] != (CARRIER_POINT_WIDTH,):
        raise ValueError("Carrier points must end with width 8")
    if positive_params.shape != (*prefix, CARRIER_POSITIVE_WIDTH):
        raise ValueError("Carrier positive parameters must end with width 2")
    if signed_params.shape != (*prefix, CARRIER_SIGNED_WIDTH):
        raise ValueError("Carrier signed parameters must end with width 2")
    if kind_ids.shape != prefix or member_counts.shape != prefix:
        raise ValueError("Carrier kind ids and member counts must match the prefix shape")
    if group_strokes is not None and group_strokes.shape != prefix:
        raise ValueError("Carrier group strokes must match the prefix shape")
    _validate_kind_ids(kind_ids)
    if member_counts.dtype == torch.bool or member_counts.is_floating_point():
        raise ValueError("Carrier member counts must use an integer tensor dtype")
    expected = (
        ((kind_ids == NO_OBJECT) & (member_counts == 0))
        | ((kind_ids == SINGLE_LINE) & (member_counts == 1))
        | ((kind_ids == PARALLEL_2) & (member_counts == 2))
        | ((kind_ids == PARALLEL_3) & (member_counts == 3))
        | ((kind_ids == HASHED_WEDGE) & (member_counts >= 2))
        | ((kind_ids == SOLID_TRIANGLE) & (member_counts == 1))
        | ((kind_ids == SOLID_QUAD) & (member_counts == 1))
    )
    if bool((~expected).any()):
        raise ValueError("Carrier member count is incompatible with its group kind")
    if bool((member_counts > max_members).any()):
        raise ValueError("Carrier member count exceeds max_members")
    return prefix


def _frame(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = points[:, :2]
    end = points[:, 2:4]
    delta = end - start
    axis = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    normal = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
    return start, end, normal


def decode_bond_group_carriers(
    carrier_points: torch.Tensor,
    positive_params: torch.Tensor,
    signed_params: torch.Tensor,
    kind_ids: torch.Tensor,
    member_counts: torch.Tensor,
    group_strokes: torch.Tensor | None = None,
    *,
    max_members: int = 32,
) -> DecodedBondGroupCarrier:
    """Differentiably expand typed carrier tensors into padded members."""

    prefix = _validate_decode_inputs(
        carrier_points,
        positive_params,
        signed_params,
        kind_ids,
        member_counts,
        group_strokes,
        max_members,
    )
    flat_size = 1
    for value in prefix:
        flat_size *= value
    points = carrier_points.reshape(flat_size, CARRIER_POINT_WIDTH)
    positive = positive_params.reshape(flat_size, CARRIER_POSITIVE_WIDTH)
    signed = signed_params.reshape(flat_size, CARRIER_SIGNED_WIDTH)
    kinds = kind_ids.reshape(flat_size)
    counts = member_counts.reshape(flat_size)
    geometry = carrier_points.new_zeros((flat_size, max_members, 8))
    member_mask = torch.arange(max_members, device=counts.device)[None] < counts[:, None]
    geometry_mask = torch.zeros(
        (flat_size, max_members, 8),
        dtype=torch.bool,
        device=carrier_points.device,
    )

    index = torch.where(kinds == SINGLE_LINE)[0]
    if len(index):
        geometry[index, 0, :4] = points[index, :4]

    index = torch.where(kinds == PARALLEL_2)[0]
    if len(index):
        geometry[index, 0, :4] = points[index, :4]
        geometry[index, 1, :4] = points[index, 4:8]

    index = torch.where(kinds == PARALLEL_3)[0]
    if len(index):
        start, end, normal = _frame(points[index])
        spacing = positive[index, 0, None]
        offsets = torch.stack((-spacing, torch.zeros_like(spacing), spacing), dim=1)
        starts = start[:, None, :] + offsets * normal[:, None, :]
        ends = end[:, None, :] + offsets * normal[:, None, :]
        geometry[index, :3, :4] = torch.cat((starts, ends), dim=-1)

    index = torch.where(kinds == HASHED_WEDGE)[0]
    if len(index):
        start, end, normal = _frame(points[index])
        local_counts = counts[index]
        steps = torch.arange(max_members, device=points.device, dtype=points.dtype)[None]
        t = steps / (local_counts.to(points.dtype)[:, None] - 1.0).clamp_min(1.0)
        centers = start[:, None, :] + t[:, :, None] * (end - start)[:, None, :]
        widths = positive[index, 0, None] + t * (
            positive[index, 1, None] - positive[index, 0, None]
        )
        left = centers - widths[:, :, None] * normal[:, None, :]
        right = centers + widths[:, :, None] * normal[:, None, :]
        geometry[index, :, :4] = torch.cat((left, right), dim=-1)

    index = torch.where(kinds == SOLID_TRIANGLE)[0]
    if len(index):
        geometry[index, 0, :6] = points[index, :6]
    index = torch.where(kinds == SOLID_QUAD)[0]
    if len(index):
        geometry[index, 0, :8] = points[index, :8]

    line_rows = (
        (kinds == SINGLE_LINE)
        | (kinds == PARALLEL_2)
        | (kinds == PARALLEL_3)
        | (kinds == HASHED_WEDGE)
    )
    geometry_mask[line_rows, :, :4] = member_mask[line_rows, :, None]
    geometry_mask[kinds == SOLID_TRIANGLE, 0, :6] = True
    geometry_mask[kinds == SOLID_QUAD, 0, :8] = True
    strokes = (
        carrier_points.new_zeros(prefix)
        if group_strokes is None
        else group_strokes.to(carrier_points.dtype)
    )
    member_stroke = strokes.reshape(flat_size, 1).expand(-1, max_members) * member_mask.to(
        carrier_points.dtype
    )
    return DecodedBondGroupCarrier(
        member_geometry=geometry.reshape(*prefix, max_members, 8),
        member_geometry_mask=geometry_mask.reshape(*prefix, max_members, 8),
        member_mask=member_mask.reshape(*prefix, max_members),
        member_stroke=member_stroke.reshape(*prefix, max_members),
    )


__all__ = [
    "CARRIER_ID_TO_KIND",
    "CARRIER_KIND_TO_ID",
    "CARRIER_POINT_WIDTH",
    "CARRIER_POSITIVE_WIDTH",
    "CARRIER_SIGNED_WIDTH",
    "CARRIER_VARIANT_COUNT",
    "BondGroupCarrierEncoding",
    "BondGroupCarrierVariants",
    "DecodedBondGroupCarrier",
    "carrier_parameter_masks",
    "carrier_parameter_variants",
    "decode_bond_group_carriers",
    "encode_bond_group_carrier",
]
