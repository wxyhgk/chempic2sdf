"""Unordered Visual SVG v0.2 bond-group set prediction.

The outer set is permutation invariant.  Members inside one group follow the
canonical Visual SVG v0.2 member order so a variable-length hashed wedge can be
represented without flattening it into unrelated image-wide line objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    LINE,
    LINE_CENTER_COST_WEIGHT,
    LINE_CENTER_LOSS_WEIGHT,
    LINE_ENDPOINT_COST_WEIGHT,
    LINE_ENDPOINT_LOSS_WEIGHT,
    LINE_LENGTH_COST_WEIGHT,
    LINE_LENGTH_LOSS_WEIGHT,
    LINE_ORIENTATION_COST_WEIGHT,
    LINE_ORIENTATION_LOSS_WEIGHT,
    QUAD,
    TRIANGLE,
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
    geometry_variants,
    line_endpoint_aligned_losses,
    pairwise_line_geometry_cost,
)


NO_OBJECT = 0
SINGLE_LINE = 1
PARALLEL_2 = 2
PARALLEL_3 = 3
HASHED_WEDGE = 4
SOLID_TRIANGLE = 5
SOLID_QUAD = 6
TEXT_BOX = 7

OBJECT_TYPES = (
    "no_object",
    "single_line",
    "parallel_2",
    "parallel_3",
    "hashed_wedge",
    "solid_triangle",
    "solid_quad",
    "text_box",
)
GROUP_KIND_TO_ID = {kind: index for index, kind in enumerate(OBJECT_TYPES) if index}
GROUP_ID_TO_KIND = {index: kind for kind, index in GROUP_KIND_TO_ID.items()}
LINE_GROUP_TYPES = (SINGLE_LINE, PARALLEL_2, PARALLEL_3, HASHED_WEDGE)
BOND_GROUP_TYPES = (*LINE_GROUP_TYPES, SOLID_TRIANGLE, SOLID_QUAD)


@dataclass(frozen=True, slots=True)
class BondGroupSetConfig(PrimitiveSetConfig):
    """Architecture contract for Visual SVG v0.2 object prediction."""

    queries: int = 192
    max_members: int = 32

    def __post_init__(self) -> None:
        PrimitiveSetConfig.__post_init__(self)
        if self.max_members < 1:
            raise ValueError("Bond Group Set max_members must be positive")
        if self.line_center_vector:
            raise ValueError("Bond Group Set does not support line_center_vector")


def _member_primitive_type(group_type: int) -> int:
    if group_type in LINE_GROUP_TYPES:
        return LINE
    if group_type == SOLID_TRIANGLE:
        return TRIANGLE
    if group_type == SOLID_QUAD:
        return QUAD
    raise ValueError(f"Object type {group_type} has no physical members")


def _active_coordinate_count(group_type: int) -> int:
    return {LINE: 4, TRIANGLE: 6, QUAD: 8}[
        _member_primitive_type(group_type)
    ]


class BondGroupSetPredictor(PrimitiveSetPredictor):
    """Predict bond symbols and text boxes as one unordered object set."""

    model_type = "bond-group-set-v0.1"

    def __init__(
        self,
        vision_config: ChemVectorConfig,
        config: BondGroupSetConfig = BondGroupSetConfig(),
    ) -> None:
        super().__init__(vision_config, config)
        self.config = config

        # Replace the flat primitive heads while retaining the established
        # ChemVisionEncoder, raster ink path, queries, decoder and local refiner.
        del self.geometry_head
        del self.stroke_head
        self.type_head = nn.Linear(config.d_model, len(OBJECT_TYPES))
        self.member_count_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.max_members + 1),
        )
        self.member_embedding = nn.Embedding(config.max_members, config.d_model)
        self.member_geometry_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 8),
        )
        self.member_stroke_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.text_geometry_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 4),
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.queries + 1),
        )
        self._reset_group_heads()

    def _reset_group_heads(self) -> None:
        nn.init.normal_(self.member_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.member_geometry_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.member_geometry_head[-1].bias)
        nn.init.normal_(self.text_geometry_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.text_geometry_head[-1].bias)
        with torch.no_grad():
            self.type_head.bias.zero_()
            self.type_head.bias[NO_OBJECT] = torch.log(
                self.type_head.bias.new_tensor(14.0)
            )
            self.member_count_head[-1].bias.zero_()
            self.member_stroke_head[-1].bias.fill_(-1.2)
            self.count_head[-1].bias.zero_()

    def _decode_object_features(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.config.endpoint_refine_kernel:
            memory_map, stroke, ink = self._memory_map(
                pixel_values,
                return_refine_inputs=True,
            )
            refine_map = self._endpoint_refine_map(stroke, ink)
        else:
            memory_map = self._memory_map(pixel_values)
            refine_map = None
        memory = self._memory_from_map(memory_map)
        references = self.reference_points.to(device=memory.device, dtype=memory.dtype)
        queries = (
            self.query_embedding.weight.to(memory.dtype)
            + self.query_position_projection(references)
        ).unsqueeze(0).expand(pixel_values.size(0), -1, -1)
        if self.config.local_query_kernel:
            queries = queries + self._local_query_features(memory_map, references)
        return self.decoder(queries, memory), memory, refine_map

    def _refine_member_points(
        self,
        coarse_geometry: torch.Tensor,
        member_features: torch.Tensor,
        refine_map: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, members, _ = coarse_geometry.shape
        points = coarse_geometry.reshape(batch, queries, members, 4, 2)
        grid = points.mul(2.0).sub(1.0).reshape(
            batch,
            queries * members * 4,
            1,
            2,
        )
        sampled = F.grid_sample(
            refine_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
            batch,
            queries,
            members,
            4,
            self.endpoint_refine_channels,
        )
        point_indices = torch.arange(4, device=member_features.device)
        point_features = self.endpoint_point_embedding(point_indices).to(
            member_features.dtype
        )
        point_features = point_features.view(1, 1, 1, 4, -1).expand(
            batch,
            queries,
            members,
            -1,
            -1,
        )
        expanded_members = member_features.unsqueeze(3).expand(-1, -1, -1, 4, -1)
        residual = self.endpoint_refine_mlp(
            torch.cat((expanded_members, sampled, point_features), dim=-1)
        )
        return torch.sigmoid(
            self._inverse_sigmoid(points) + residual
        ).flatten(-2)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
            raise ValueError(
                "Bond Group Set images must have shape [batch, 3, height, width]"
            )
        hidden, memory, refine_map = self._decode_object_features(pixel_values)
        references = self.reference_points.to(device=hidden.device, dtype=hidden.dtype)
        member_features = hidden.unsqueeze(2) + self.member_embedding.weight.to(
            hidden.dtype
        ).view(1, 1, self.config.max_members, self.config.d_model)
        point_references = references.repeat(1, 4).view(
            1,
            self.config.queries,
            1,
            8,
        )
        coarse_member_geometry = torch.sigmoid(
            self._inverse_sigmoid(point_references)
            + self.member_geometry_head(member_features)
        )
        member_geometry = coarse_member_geometry
        if refine_map is not None:
            member_geometry = self._refine_member_points(
                coarse_member_geometry,
                member_features,
                refine_map,
            )

        text_raw = self.text_geometry_head(hidden)
        text_geometry = torch.empty_like(text_raw)
        text_geometry[..., :2] = torch.sigmoid(
            self._inverse_sigmoid(references)[None] + text_raw[..., :2]
        )
        text_geometry[..., 2:4] = torch.sigmoid(text_raw[..., 2:4] - 2.2)
        output = {
            "type_logits": self.type_head(hidden),
            "count_logits": self.count_head(hidden.mean(dim=1)),
            "member_count_logits": self.member_count_head(hidden),
            "member_geometry": member_geometry,
            "member_stroke": self.member_stroke_head(member_features).squeeze(-1).sigmoid(),
            "text_geometry": text_geometry,
            "object_features": hidden,
            "memory": memory,
        }
        if refine_map is not None:
            output["coarse_member_geometry"] = coarse_member_geometry
        return output

    def config_dict(self) -> dict:
        return {
            "vision_config": asdict(self.vision_config),
            "bond_group_set_config": asdict(self.config),
        }


def _pairwise_member_geometry_cost(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_types: torch.Tensor,
    target_member_mask: torch.Tensor,
) -> torch.Tensor:
    """Return canonical-member geometry cost for every query-target pair."""

    if predicted.ndim != 3 or predicted.size(-1) != 8:
        raise ValueError("Predicted member geometry must have shape [queries, members, 8]")
    if target.ndim != 3 or target.size(-1) != 8:
        raise ValueError("Target member geometry must have shape [targets, members, 8]")
    if predicted.size(1) != target.size(1):
        raise ValueError("Predicted and target member capacities differ")
    costs = predicted.new_zeros((predicted.size(0), target.size(0)))
    for target_index in range(target.size(0)):
        group_type = int(target_types[target_index])
        if group_type == TEXT_BOX:
            continue
        member_indices = torch.where(target_member_mask[target_index])[0]
        if not len(member_indices):
            raise ValueError("A bond-group target must contain at least one member")
        primitive_type = _member_primitive_type(group_type)
        if primitive_type == LINE:
            parts = []
            for member_index in member_indices.tolist():
                endpoint, center, length, orientation = pairwise_line_geometry_cost(
                    predicted[:, member_index],
                    target[target_index, member_index].unsqueeze(0),
                )
                parts.append(
                    LINE_ENDPOINT_COST_WEIGHT * endpoint[:, 0]
                    + LINE_CENTER_COST_WEIGHT * center[:, 0]
                    + LINE_LENGTH_COST_WEIGHT * length[:, 0]
                    + LINE_ORIENTATION_COST_WEIGHT * orientation[:, 0]
                )
            costs[:, target_index] = torch.stack(parts, dim=-1).mean(dim=-1)
            continue
        size = _active_coordinate_count(group_type)
        member_index = int(member_indices[0])
        variants = geometry_variants(
            target[target_index, member_index].unsqueeze(0),
            primitive_type,
        )[0, :, :size]
        difference = (
            predicted[:, member_index, None, :size] - variants[None]
        ).abs().mean(dim=-1)
        costs[:, target_index] = difference.amin(dim=-1)
    return costs


def bond_group_set_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    class_cost_weight: float = 1.0,
    geometry_cost_weight: float = 4.0,
    member_count_cost_weight: float = 1.0,
    stroke_cost_weight: float = 0.25,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Hungarian-match target objects to unique queries."""

    if min(
        class_cost_weight,
        geometry_cost_weight,
        member_count_cost_weight,
        stroke_cost_weight,
    ) < 0.0:
        raise ValueError("Bond Group Set matching weights must be nonnegative")
    probabilities = output["type_logits"].detach().float().softmax(-1)
    member_count_probabilities = output["member_count_logits"].detach().float().softmax(-1)
    member_geometry = output["member_geometry"].detach().float()
    member_stroke = output["member_stroke"].detach().float()
    text_geometry = output["text_geometry"].detach().float()
    target_mask = targets["target_mask"].bool()
    if probabilities.size(0) != target_mask.size(0):
        raise ValueError("Bond Group Set output and target batches differ")
    if member_geometry.size(2) != targets["member_geometry"].size(2):
        raise ValueError("Bond Group Set member capacities differ")

    assignments = []
    for batch_index in range(probabilities.size(0)):
        target_indices = torch.where(target_mask[batch_index])[0]
        if len(target_indices) > probabilities.size(1):
            raise ValueError("Bond Group Set targets exceed configured query capacity")
        if not len(target_indices):
            empty = torch.empty(0, device=probabilities.device, dtype=torch.long)
            assignments.append((empty, empty))
            continue
        target_types = targets["types"][batch_index, target_indices].long()
        target_counts = targets["member_counts"][batch_index, target_indices].long()
        if int(target_counts.max()) >= member_count_probabilities.size(-1):
            raise ValueError("Bond-group member count exceeds model capacity")

        geometry_cost = _pairwise_member_geometry_cost(
            member_geometry[batch_index],
            targets["member_geometry"][batch_index, target_indices].float(),
            target_types,
            targets["member_mask"][batch_index, target_indices].bool(),
        )
        text_columns = torch.where(target_types == TEXT_BOX)[0]
        if len(text_columns):
            expected_text = targets["text_geometry"][
                batch_index,
                target_indices[text_columns],
            ].float()
            geometry_cost[:, text_columns] = (
                text_geometry[batch_index, :, None, :]
                - expected_text[None, :, :]
            ).abs().mean(dim=-1)

        class_cost = -probabilities[batch_index][:, target_types]
        count_cost = -member_count_probabilities[batch_index][:, target_counts]
        stroke_cost = torch.zeros_like(geometry_cost)
        for column, target_index in enumerate(target_indices.tolist()):
            active = targets["member_mask"][batch_index, target_index].bool()
            if bool(active.any()):
                stroke_cost[:, column] = (
                    member_stroke[batch_index, :, active]
                    - targets["member_stroke"][batch_index, target_index, active]
                    .float()[None]
                ).abs().mean(dim=-1)
        cost = (
            class_cost_weight * class_cost
            + geometry_cost_weight * geometry_cost
            + member_count_cost_weight * count_cost
            + stroke_cost_weight * stroke_cost
        )
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(rows, device=probabilities.device, dtype=torch.long),
                target_indices[
                    torch.as_tensor(columns, device=probabilities.device, dtype=torch.long)
                ],
            )
        )
    return assignments


def _align_members(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    group_types: torch.Tensor,
    member_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    aligned = torch.zeros_like(expected)
    coordinate_mask = torch.zeros_like(expected, dtype=torch.bool)
    for group_type in BOND_GROUP_TYPES:
        object_indices = torch.where(group_types == group_type)[0]
        if not len(object_indices):
            continue
        primitive_type = _member_primitive_type(group_type)
        size = _active_coordinate_count(group_type)
        for object_index in object_indices.tolist():
            indices = torch.where(member_mask[object_index])[0]
            if not len(indices):
                raise ValueError("A matched bond group has no active members")
            current_predicted = predicted[object_index, indices]
            current_expected = expected[object_index, indices]
            variants = geometry_variants(current_expected, primitive_type)
            differences = (
                current_predicted[:, None, :size] - variants[:, :, :size]
            ).abs().mean(dim=-1)
            best = differences.argmin(dim=-1)
            selected = variants[
                torch.arange(len(indices), device=predicted.device),
                best,
            ]
            aligned[object_index, indices, :size] = selected[:, :size]
            coordinate_mask[object_index, indices, :size] = True
    return aligned, coordinate_mask


def bond_group_set_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.1,
    geometry_weight: float = 5.0,
    text_geometry_weight: float = 5.0,
    stroke_weight: float = 0.5,
    member_count_weight: float = 1.0,
    cardinality_weight: float = 0.25,
    count_classification_weight: float = 0.25,
    image_size: float = 768.0,
    class_cost_weight: float = 1.0,
    geometry_cost_weight: float = 4.0,
    member_count_cost_weight: float = 1.0,
    stroke_cost_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """DETR-style outer-set loss with exact canonical member supervision."""

    if not 0.0 < no_object_weight <= 1.0:
        raise ValueError("Bond Group Set no-object weight must be in (0, 1]")
    if min(
        geometry_weight,
        text_geometry_weight,
        stroke_weight,
        member_count_weight,
        cardinality_weight,
        count_classification_weight,
    ) < 0.0:
        raise ValueError("Bond Group Set loss weights must be nonnegative")
    assignments = bond_group_set_assignments(
        output,
        targets,
        class_cost_weight=class_cost_weight,
        geometry_cost_weight=geometry_cost_weight,
        member_count_cost_weight=member_count_cost_weight,
        stroke_cost_weight=stroke_cost_weight,
    )
    type_targets = torch.full(
        output["type_logits"].shape[:2],
        NO_OBJECT,
        device=output["type_logits"].device,
        dtype=torch.long,
    )
    matched_predicted_geometry = []
    matched_expected_geometry = []
    matched_member_masks = []
    matched_types = []
    matched_predicted_strokes = []
    matched_expected_strokes = []
    matched_predicted_member_counts = []
    matched_expected_member_counts = []
    matched_predicted_text = []
    matched_expected_text = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        current_types = targets["types"][batch_index, target_indices].long()
        type_targets[batch_index, query_indices] = current_types
        matched_predicted_geometry.append(
            output["member_geometry"][batch_index, query_indices]
        )
        matched_expected_geometry.append(
            targets["member_geometry"][batch_index, target_indices]
        )
        matched_member_masks.append(
            targets["member_mask"][batch_index, target_indices]
        )
        matched_types.append(current_types)
        matched_predicted_strokes.append(
            output["member_stroke"][batch_index, query_indices]
        )
        matched_expected_strokes.append(
            targets["member_stroke"][batch_index, target_indices]
        )
        matched_predicted_member_counts.append(
            output["member_count_logits"][batch_index, query_indices]
        )
        matched_expected_member_counts.append(
            targets["member_counts"][batch_index, target_indices]
        )
        text_mask = current_types == TEXT_BOX
        if bool(text_mask.any()):
            matched_predicted_text.append(
                output["text_geometry"][batch_index, query_indices[text_mask]]
            )
            matched_expected_text.append(
                targets["text_geometry"][batch_index, target_indices[text_mask]]
            )

    class_weights = output["type_logits"].new_tensor(
        (no_object_weight, 1.0, 2.0, 4.0, 3.0, 3.0, 4.0, 1.5)
    )
    classification_loss = F.cross_entropy(
        output["type_logits"].transpose(1, 2),
        type_targets,
        weight=class_weights,
    )
    zero = classification_loss * 0.0
    line_losses = {
        "line_endpoint_loss": zero,
        "line_center_loss": zero,
        "line_length_loss": zero,
        "line_orientation_loss": zero,
    }
    if matched_types:
        predicted_geometry = torch.cat(matched_predicted_geometry)
        expected_geometry = torch.cat(matched_expected_geometry).to(
            predicted_geometry.dtype
        )
        member_mask = torch.cat(matched_member_masks).bool()
        group_types = torch.cat(matched_types)
        aligned, coordinate_mask = _align_members(
            predicted_geometry,
            expected_geometry,
            group_types,
            member_mask,
        )
        if bool(coordinate_mask.any()):
            pixel_errors = (predicted_geometry.float() - aligned.float()) * image_size
            geometry_loss = F.smooth_l1_loss(
                pixel_errors[coordinate_mask],
                torch.zeros_like(pixel_errors[coordinate_mask]),
                beta=4.0,
            ) / image_size
        else:
            geometry_loss = zero

        line_object_mask = torch.zeros_like(group_types, dtype=torch.bool)
        for group_type in LINE_GROUP_TYPES:
            line_object_mask |= group_types == group_type
        if bool(line_object_mask.any()):
            line_member_mask = member_mask & line_object_mask[:, None]
            predicted_lines = predicted_geometry[line_member_mask]
            expected_lines = expected_geometry[line_member_mask]
            line_losses = line_endpoint_aligned_losses(
                predicted_lines,
                expected_lines,
                image_size=image_size,
            )

        predicted_strokes = torch.cat(matched_predicted_strokes)
        expected_strokes = torch.cat(matched_expected_strokes).to(
            predicted_strokes.dtype
        )
        stroke_loss = (
            F.smooth_l1_loss(
                predicted_strokes[member_mask],
                expected_strokes[member_mask],
            )
            if bool(member_mask.any())
            else zero
        )
        member_count_loss = F.cross_entropy(
            torch.cat(matched_predicted_member_counts).float(),
            torch.cat(matched_expected_member_counts).long(),
        )
    else:
        geometry_loss = stroke_loss = member_count_loss = zero

    if matched_predicted_text:
        predicted_text = torch.cat(matched_predicted_text).float()
        expected_text = torch.cat(matched_expected_text).to(predicted_text.dtype)
        text_pixel_errors = (predicted_text - expected_text) * image_size
        text_geometry_loss = F.smooth_l1_loss(
            text_pixel_errors,
            torch.zeros_like(text_pixel_errors),
            beta=4.0,
        ) / image_size
    else:
        text_geometry_loss = zero

    probabilities = output["type_logits"].float().softmax(-1)
    predicted_counts = (1.0 - probabilities[..., NO_OBJECT]).sum(dim=1)
    expected_counts = targets["target_mask"].sum(dim=1).float()
    cardinality_loss = F.smooth_l1_loss(predicted_counts, expected_counts) / max(
        output["type_logits"].size(1),
        1,
    )
    expected_count_shape = (
        output["type_logits"].size(0),
        output["type_logits"].size(1) + 1,
    )
    if output["count_logits"].shape != expected_count_shape:
        raise ValueError(
            "Bond Group Set count logits must have shape "
            f"{expected_count_shape}, received {tuple(output['count_logits'].shape)}"
        )
    count_classification_loss = F.cross_entropy(
        output["count_logits"].float(),
        expected_counts.long(),
    )
    total = (
        classification_loss
        + geometry_weight * geometry_loss
        + text_geometry_weight * text_geometry_loss
        + stroke_weight * stroke_loss
        + member_count_weight * member_count_loss
        + cardinality_weight * cardinality_loss
        + count_classification_weight * count_classification_loss
        + LINE_ENDPOINT_LOSS_WEIGHT * line_losses["line_endpoint_loss"]
        + LINE_CENTER_LOSS_WEIGHT * line_losses["line_center_loss"]
        + LINE_LENGTH_LOSS_WEIGHT * line_losses["line_length_loss"]
        + LINE_ORIENTATION_LOSS_WEIGHT * line_losses["line_orientation_loss"]
    )
    return {
        "loss": total,
        "classification_loss": classification_loss,
        "geometry_loss": geometry_loss,
        "text_geometry_loss": text_geometry_loss,
        "stroke_loss": stroke_loss,
        "member_count_loss": member_count_loss,
        "cardinality_loss": cardinality_loss,
        "count_classification_loss": count_classification_loss,
        **line_losses,
        "predicted_count_mean": predicted_counts.mean(),
        "target_count_mean": expected_counts.mean(),
        "assignments": assignments,
    }


@torch.no_grad()
def decode_bond_group_set(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.5,
    mode: str = "threshold",
) -> list[list[dict[str, torch.Tensor | float | int | str]]]:
    """Decode model tensors while preserving predicted group member counts."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Bond Group Set threshold must be in [0, 1]")
    if mode not in {"threshold", "count_head"}:
        raise ValueError("Bond Group Set decode mode must be threshold or count_head")
    probabilities = output["type_logits"].float().softmax(-1)
    object_scores, object_types = probabilities[..., 1:].max(dim=-1)
    object_types = object_types + 1
    no_object_scores = probabilities[..., NO_OBJECT]
    member_counts = output["member_count_logits"].float().argmax(dim=-1)
    if mode == "count_head":
        expected_shape = (probabilities.size(0), probabilities.size(1) + 1)
        if output.get("count_logits") is None or output["count_logits"].shape != expected_shape:
            raise ValueError("count_head decoding requires valid total count logits")
        total_counts = output["count_logits"].float().argmax(dim=-1)

    decoded = []
    for batch_index in range(probabilities.size(0)):
        if mode == "threshold":
            keep = (object_scores[batch_index] >= threshold) & (
                object_scores[batch_index] > no_object_scores[batch_index]
            )
            query_indices = torch.where(keep)[0].tolist()
        else:
            ranking = object_scores[batch_index] - no_object_scores[batch_index]
            query_indices = torch.argsort(
                ranking,
                descending=True,
                stable=True,
            )[: int(total_counts[batch_index])].tolist()
        rows = []
        for query_index in query_indices:
            object_type = int(object_types[batch_index, query_index])
            row: dict[str, torch.Tensor | float | int | str] = {
                "query_index": query_index,
                "score": float(object_scores[batch_index, query_index]),
                "type": object_type,
                "kind": GROUP_ID_TO_KIND[object_type],
            }
            if object_type == TEXT_BOX:
                row["member_count"] = 0
                row["text_geometry"] = output["text_geometry"][
                    batch_index,
                    query_index,
                ].detach()
            else:
                member_count = int(member_counts[batch_index, query_index])
                row["member_count"] = member_count
                row["member_geometry"] = output["member_geometry"][
                    batch_index,
                    query_index,
                    :member_count,
                ].detach()
                row["member_stroke"] = output["member_stroke"][
                    batch_index,
                    query_index,
                    :member_count,
                ].detach()
            rows.append(row)
        decoded.append(rows)
    return decoded
