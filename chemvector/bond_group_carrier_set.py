"""Typed carrier-set prediction for compact Visual SVG bond groups.

The outer object set follows the existing DETR-style BondGroupSet contract,
but a bond group is represented by at most four semantic carrier points and a
small type-specific parameter vector.  Physical member expansion belongs to
``bond_group_carrier_geometry`` and is deliberately not duplicated here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.bond_group_set import (
    BOND_GROUP_TYPES,
    GROUP_ID_TO_KIND,
    HASHED_WEDGE,
    NO_OBJECT,
    OBJECT_TYPES,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
    SOLID_QUAD,
    SOLID_TRIANGLE,
    TEXT_BOX,
    BondGroupSetConfig,
    BondGroupSetPredictor,
)
from chemvector.model import ChemVectorConfig


CARRIER_COORDINATES = 8
CARRIER_POSITIVE_PARAMETERS = 2
CARRIER_SIGNED_PARAMETERS = 2
LEGACY_MEMBER_COUNT_CLASSES = 33
FIXED_MEMBER_COUNTS = {
    SINGLE_LINE: 1,
    PARALLEL_2: 2,
    PARALLEL_3: 3,
    SOLID_TRIANGLE: 1,
    SOLID_QUAD: 1,
}


@dataclass(frozen=True, slots=True)
class BondGroupCarrierSetConfig(BondGroupSetConfig):
    """Architecture contract for typed compact bond-group carriers."""

    queries: int = 192
    max_members: int = 143


@dataclass(frozen=True, slots=True)
class _CarrierVariants:
    points: torch.Tensor
    positive_params: torch.Tensor
    signed_params: torch.Tensor
    point_mask: torch.Tensor
    positive_mask: torch.Tensor
    signed_mask: torch.Tensor


def _carrier_variants(
    points: torch.Tensor,
    positive_params: torch.Tensor,
    signed_params: torch.Tensor,
    kind_id: int,
) -> _CarrierVariants:
    """Adapt the geometry module's symmetry variants to loss tensors.

    This is the sole model/geometry boundary.  Endpoint reversal, parallel-2
    trim transformation, hashed-wedge half-width exchange and polygon cycles
    are owned by ``carrier_parameter_variants``.
    """

    from chemvector.bond_group_carrier_geometry import (
        carrier_parameter_masks,
        carrier_parameter_variants,
    )

    result = carrier_parameter_variants(
        points.unsqueeze(0),
        positive_params.unsqueeze(0),
        signed_params.unsqueeze(0),
        torch.as_tensor([kind_id], device=points.device, dtype=torch.long),
    )
    try:
        active = result.variant_mask[0].bool()
        variant_points = result.carrier_points[0, active]
        variant_positive = result.positive_params[0, active]
        variant_signed = result.signed_params[0, active]
    except AttributeError as error:
        raise TypeError(
            "carrier_parameter_variants returned an unsupported result"
        ) from error
    variant_count = variant_points.size(0)
    masks = carrier_parameter_masks(
        torch.as_tensor([kind_id], device=points.device, dtype=torch.long)
    )
    point_mask = masks[0][0].expand(variant_count, -1)
    positive_mask = masks[1][0].expand(variant_count, -1)
    signed_mask = masks[2][0].expand(variant_count, -1)

    def expand_mask(mask: torch.Tensor, width: int, name: str) -> torch.Tensor:
        mask = mask.to(device=variant_points.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(variant_count, -1)
        if mask.shape != (variant_count, width):
            raise ValueError(
                f"Carrier {name} mask must have shape {(variant_count, width)}, "
                f"received {tuple(mask.shape)}"
            )
        return mask

    if variant_points.shape != (variant_count, CARRIER_COORDINATES):
        raise ValueError("Carrier point variants must have shape [variants, 8]")
    if variant_positive.shape != (
        variant_count,
        CARRIER_POSITIVE_PARAMETERS,
    ):
        raise ValueError("Carrier positive variants must have shape [variants, 2]")
    if variant_signed.shape != (
        variant_count,
        CARRIER_SIGNED_PARAMETERS,
    ):
        raise ValueError("Carrier signed variants must have shape [variants, 2]")
    return _CarrierVariants(
        points=variant_points,
        positive_params=variant_positive,
        signed_params=variant_signed,
        point_mask=expand_mask(point_mask, CARRIER_COORDINATES, "point"),
        positive_mask=expand_mask(
            positive_mask,
            CARRIER_POSITIVE_PARAMETERS,
            "positive-parameter",
        ),
        signed_mask=expand_mask(
            signed_mask,
            CARRIER_SIGNED_PARAMETERS,
            "signed-parameter",
        ),
    )


class BondGroupCarrierSetPredictor(BondGroupSetPredictor):
    """Predict an unordered set of typed bond carriers and text boxes."""

    model_type = "bond-group-carrier-set-v0.1"

    def __init__(
        self,
        vision_config: ChemVectorConfig,
        config: BondGroupCarrierSetConfig = BondGroupCarrierSetConfig(),
    ) -> None:
        super().__init__(vision_config, config)
        self.config = config

        # The member-count classifier remains useful for hashed wedges, while
        # all max_members-sized geometry/stroke heads are removed.
        del self.member_embedding
        del self.member_geometry_head
        del self.member_stroke_head

        # carrier_geometry_head and group_stroke_head deliberately match the
        # old slot-0 head shapes.  A loader can therefore copy
        # member_embedding[0], member_geometry_head and member_stroke_head
        # exactly instead of relearning the initial carrier.
        self.carrier_embedding = nn.Embedding(1, config.d_model)
        self.carrier_geometry_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, CARRIER_COORDINATES),
        )
        self.positive_params_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, CARRIER_POSITIVE_PARAMETERS),
        )
        self.group_stroke_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self._reset_carrier_heads()

    def _reset_carrier_heads(self) -> None:
        nn.init.normal_(self.carrier_embedding.weight, mean=0.0, std=0.02)
        for head in (
            self.carrier_geometry_head,
            self.positive_params_head,
        ):
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            # Visible line spacing and wedge half-width are small fractions of
            # the canvas; sigmoid(0)=0.5 would be an unusable starting scale.
            self.positive_params_head[-1].bias.fill_(-4.0)
            self.group_stroke_head[-1].bias.fill_(-1.2)

    def _member_count_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Preserve the frozen 0..32 count path when capacity is expanded."""

        count_features = self.member_count_head[1](self.member_count_head[0](hidden))
        output_layer = self.member_count_head[2]
        if output_layer.out_features <= LEGACY_MEMBER_COUNT_CLASSES:
            return output_layer(count_features)
        legacy_logits = F.linear(
            count_features,
            output_layer.weight[:LEGACY_MEMBER_COUNT_CLASSES],
            output_layer.bias[:LEGACY_MEMBER_COUNT_CLASSES],
        )
        expanded_logits = F.linear(
            count_features,
            output_layer.weight[LEGACY_MEMBER_COUNT_CLASSES:],
            output_layer.bias[LEGACY_MEMBER_COUNT_CLASSES:],
        )
        return torch.cat((legacy_logits, expanded_logits), dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
            raise ValueError(
                "Bond Group Carrier Set images must have shape "
                "[batch, 3, height, width]"
            )
        hidden, memory, refine_map = self._decode_object_features(pixel_values)
        references = self.reference_points.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        carrier_features = hidden + self.carrier_embedding.weight[0].to(hidden.dtype)
        point_references = references.repeat(1, 4)
        coarse_carrier_points = torch.sigmoid(
            self._inverse_sigmoid(point_references)[None]
            + self.carrier_geometry_head(carrier_features)
        )
        carrier_points = coarse_carrier_points
        if refine_map is not None:
            # Four semantic carrier points are refined once per object.  There
            # is no max_members x 4 endpoint grid in this architecture.
            carrier_points = self._refine_endpoints(
                coarse_carrier_points,
                carrier_features,
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
            "member_count_logits": self._member_count_logits(hidden),
            "carrier_points": carrier_points,
            "positive_params": torch.sigmoid(
                self.positive_params_head(carrier_features)
            ),
            # Reserved codec field.  The p2 8-coordinate carrier has no signed
            # geometric degrees of freedom, so no dead trainable head is kept.
            "signed_params": carrier_features.new_zeros(
                (*carrier_features.shape[:-1], CARRIER_SIGNED_PARAMETERS)
            ),
            "group_stroke": self.group_stroke_head(carrier_features)
            .squeeze(-1)
            .sigmoid(),
            "text_geometry": text_geometry,
            "object_features": hidden,
            "memory": memory,
        }
        if refine_map is not None:
            output["coarse_carrier_points"] = coarse_carrier_points
        return output

    def config_dict(self) -> dict[str, Any]:
        return {
            "vision_config": asdict(self.vision_config),
            "bond_group_carrier_set_config": asdict(self.config),
        }


def carrier_head_warm_start_state(
    bond_group_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map old BondGroupSet slot-0 heads to their carrier equivalents.

    Shared encoder/decoder, count/type/text and endpoint-refiner keys retain
    their names and can be copied by the eventual checkpoint loader directly.
    """

    required = (
        "member_embedding.weight",
        "member_geometry_head.0.weight",
        "member_geometry_head.0.bias",
        "member_geometry_head.2.weight",
        "member_geometry_head.2.bias",
        "member_stroke_head.0.weight",
        "member_stroke_head.0.bias",
        "member_stroke_head.2.weight",
        "member_stroke_head.2.bias",
    )
    missing = [name for name in required if name not in bond_group_state]
    if missing:
        raise KeyError(f"Bond-group warm start is missing keys: {missing}")
    embedding = bond_group_state["member_embedding.weight"]
    if embedding.ndim != 2 or embedding.size(0) < 1:
        raise ValueError("member_embedding.weight must contain slot 0")
    mapped = {"carrier_embedding.weight": embedding[:1].detach().clone()}
    for old_prefix, new_prefix in (
        ("member_geometry_head.", "carrier_geometry_head."),
        ("member_stroke_head.", "group_stroke_head."),
    ):
        for name, value in bond_group_state.items():
            if name.startswith(old_prefix):
                mapped[new_prefix + name[len(old_prefix) :]] = value.detach().clone()
    return mapped


def _masked_variant_distance(
    predicted_points: torch.Tensor,
    predicted_positive: torch.Tensor,
    predicted_signed: torch.Tensor,
    variants: _CarrierVariants,
) -> torch.Tensor:
    """Return one type-aware distance per prediction and target variant."""

    if predicted_points.ndim != 2 or predicted_points.size(-1) != CARRIER_COORDINATES:
        raise ValueError("Predicted carrier points must have shape [queries, 8]")
    if predicted_positive.shape != (
        predicted_points.size(0),
        CARRIER_POSITIVE_PARAMETERS,
    ):
        raise ValueError("Predicted positive parameters must have shape [queries, 2]")
    if predicted_signed.shape != (
        predicted_points.size(0),
        CARRIER_SIGNED_PARAMETERS,
    ):
        raise ValueError("Predicted signed parameters must have shape [queries, 2]")

    def distance(
        predicted: torch.Tensor,
        expected: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        difference = (predicted[:, None] - expected[None]).abs()
        weights = mask.to(difference.dtype)[None]
        denominator = weights.sum(dim=-1).clamp_min(1.0)
        return (difference * weights).sum(dim=-1) / denominator

    return (
        distance(predicted_points, variants.points, variants.point_mask)
        + distance(
            predicted_positive,
            variants.positive_params,
            variants.positive_mask,
        )
        + distance(predicted_signed, variants.signed_params, variants.signed_mask)
    )


def _pairwise_carrier_geometry_cost(
    predicted_points: torch.Tensor,
    predicted_positive: torch.Tensor,
    predicted_signed: torch.Tensor,
    target_points: torch.Tensor,
    target_positive: torch.Tensor,
    target_signed: torch.Tensor,
    target_types: torch.Tensor,
) -> torch.Tensor:
    costs = predicted_points.new_zeros(
        (predicted_points.size(0), target_points.size(0))
    )
    for target_index, kind_tensor in enumerate(target_types):
        kind_id = int(kind_tensor)
        if kind_id == TEXT_BOX:
            continue
        if kind_id not in BOND_GROUP_TYPES:
            raise ValueError(f"Unsupported carrier target type {kind_id}")
        variants = _carrier_variants(
            target_points[target_index],
            target_positive[target_index],
            target_signed[target_index],
            kind_id,
        )
        costs[:, target_index] = _masked_variant_distance(
            predicted_points,
            predicted_positive,
            predicted_signed,
            variants,
        ).amin(dim=-1)
    return costs


def bond_group_carrier_set_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    class_cost_weight: float = 1.0,
    geometry_cost_weight: float = 4.0,
    member_count_cost_weight: float = 1.0,
    stroke_cost_weight: float = 0.25,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Hungarian-match target carrier objects to unique model queries."""

    if min(
        class_cost_weight,
        geometry_cost_weight,
        member_count_cost_weight,
        stroke_cost_weight,
    ) < 0.0:
        raise ValueError("Carrier Set matching weights must be nonnegative")
    probabilities = output["type_logits"].detach().float().softmax(-1)
    member_probabilities = (
        output["member_count_logits"].detach().float().softmax(-1)
    )
    points = output["carrier_points"].detach().float()
    positive = output["positive_params"].detach().float()
    signed = output["signed_params"].detach().float()
    strokes = output["group_stroke"].detach().float()
    text_geometry = output["text_geometry"].detach().float()
    target_mask = targets["target_mask"].bool()
    if probabilities.size(0) != target_mask.size(0):
        raise ValueError("Carrier Set output and target batches differ")

    assignments = []
    for batch_index in range(probabilities.size(0)):
        target_indices = torch.where(target_mask[batch_index])[0]
        if len(target_indices) > probabilities.size(1):
            raise ValueError("Carrier targets exceed configured query capacity")
        if not len(target_indices):
            empty = torch.empty(0, device=probabilities.device, dtype=torch.long)
            assignments.append((empty, empty))
            continue
        target_types = targets["types"][batch_index, target_indices].long()
        target_counts = targets["member_counts"][batch_index, target_indices].long()
        if int(target_counts.max()) >= member_probabilities.size(-1):
            raise ValueError("Carrier member count exceeds model capacity")

        geometry_cost = _pairwise_carrier_geometry_cost(
            points[batch_index],
            positive[batch_index],
            signed[batch_index],
            targets["carrier_points"][batch_index, target_indices].float(),
            targets["positive_params"][batch_index, target_indices].float(),
            targets["signed_params"][batch_index, target_indices].float(),
            target_types,
        )
        text_columns = torch.where(target_types == TEXT_BOX)[0]
        if len(text_columns):
            expected_text = targets["text_geometry"][
                batch_index,
                target_indices[text_columns],
            ].float()
            geometry_cost[:, text_columns] = (
                text_geometry[batch_index, :, None] - expected_text[None]
            ).abs().mean(dim=-1)

        class_cost = -probabilities[batch_index][:, target_types]
        count_cost = -member_probabilities[batch_index][:, target_counts]
        target_strokes = targets["group_stroke"][
            batch_index,
            target_indices,
        ].float()
        stroke_cost = (strokes[batch_index, :, None] - target_strokes[None]).abs()
        stroke_cost[:, text_columns] = 0.0
        cost = (
            class_cost_weight * class_cost
            + geometry_cost_weight * geometry_cost
            + member_count_cost_weight * count_cost
            + stroke_cost_weight * stroke_cost
        )
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        column_indices = torch.as_tensor(
            columns,
            device=probabilities.device,
            dtype=torch.long,
        )
        assignments.append(
            (
                torch.as_tensor(rows, device=probabilities.device, dtype=torch.long),
                target_indices[column_indices],
            )
        )
    return assignments


def _align_matched_carriers(
    predicted_points: torch.Tensor,
    predicted_positive: torch.Tensor,
    predicted_signed: torch.Tensor,
    expected_points: torch.Tensor,
    expected_positive: torch.Tensor,
    expected_signed: torch.Tensor,
    group_types: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    aligned_points = torch.zeros_like(expected_points)
    aligned_positive = torch.zeros_like(expected_positive)
    aligned_signed = torch.zeros_like(expected_signed)
    point_mask = torch.zeros_like(expected_points, dtype=torch.bool)
    positive_mask = torch.zeros_like(expected_positive, dtype=torch.bool)
    signed_mask = torch.zeros_like(expected_signed, dtype=torch.bool)
    for object_index, kind_tensor in enumerate(group_types):
        kind_id = int(kind_tensor)
        if kind_id == TEXT_BOX:
            continue
        variants = _carrier_variants(
            expected_points[object_index],
            expected_positive[object_index],
            expected_signed[object_index],
            kind_id,
        )
        distances = _masked_variant_distance(
            predicted_points[object_index : object_index + 1],
            predicted_positive[object_index : object_index + 1],
            predicted_signed[object_index : object_index + 1],
            variants,
        )[0]
        best = int(distances.detach().argmin())
        aligned_points[object_index] = variants.points[best]
        aligned_positive[object_index] = variants.positive_params[best]
        aligned_signed[object_index] = variants.signed_params[best]
        point_mask[object_index] = variants.point_mask[best]
        positive_mask[object_index] = variants.positive_mask[best]
        signed_mask[object_index] = variants.signed_mask[best]
    return (
        aligned_points,
        aligned_positive,
        aligned_signed,
        point_mask,
        positive_mask,
        signed_mask,
    )


def bond_group_carrier_set_loss(
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
) -> dict[str, Any]:
    """DETR-style outer-set loss with typed carrier symmetries."""

    if not 0.0 < no_object_weight <= 1.0:
        raise ValueError("Carrier Set no-object weight must be in (0, 1]")
    if min(
        geometry_weight,
        text_geometry_weight,
        stroke_weight,
        member_count_weight,
        cardinality_weight,
        count_classification_weight,
    ) < 0.0:
        raise ValueError("Carrier Set loss weights must be nonnegative")
    assignments = bond_group_carrier_set_assignments(
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
    matched: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "predicted_points",
            "expected_points",
            "predicted_positive",
            "expected_positive",
            "predicted_signed",
            "expected_signed",
            "types",
            "predicted_stroke",
            "expected_stroke",
            "predicted_member_count",
            "expected_member_count",
            "predicted_text",
            "expected_text",
        )
    }
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        current_types = targets["types"][batch_index, target_indices].long()
        type_targets[batch_index, query_indices] = current_types
        for key, output_key in (
            ("predicted_points", "carrier_points"),
            ("predicted_positive", "positive_params"),
            ("predicted_signed", "signed_params"),
            ("predicted_stroke", "group_stroke"),
            ("predicted_member_count", "member_count_logits"),
        ):
            matched[key].append(output[output_key][batch_index, query_indices])
        for key, target_key in (
            ("expected_points", "carrier_points"),
            ("expected_positive", "positive_params"),
            ("expected_signed", "signed_params"),
            ("expected_stroke", "group_stroke"),
            ("expected_member_count", "member_counts"),
        ):
            matched[key].append(targets[target_key][batch_index, target_indices])
        matched["types"].append(current_types)
        text_mask = current_types == TEXT_BOX
        if bool(text_mask.any()):
            matched["predicted_text"].append(
                output["text_geometry"][batch_index, query_indices[text_mask]]
            )
            matched["expected_text"].append(
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
    point_loss = positive_param_loss = signed_param_loss = zero
    stroke_loss = member_count_loss = zero
    if matched["types"]:
        predicted_points = torch.cat(matched["predicted_points"]).float()
        predicted_positive = torch.cat(matched["predicted_positive"]).float()
        predicted_signed = torch.cat(matched["predicted_signed"]).float()
        expected_points = torch.cat(matched["expected_points"]).to(
            predicted_points.dtype
        )
        expected_positive = torch.cat(matched["expected_positive"]).to(
            predicted_positive.dtype
        )
        expected_signed = torch.cat(matched["expected_signed"]).to(
            predicted_signed.dtype
        )
        aligned = _align_matched_carriers(
            predicted_points,
            predicted_positive,
            predicted_signed,
            expected_points,
            expected_positive,
            expected_signed,
            torch.cat(matched["types"]),
        )
        aligned_points, aligned_positive, aligned_signed = aligned[:3]
        point_mask, positive_mask, signed_mask = aligned[3:]

        def pixel_huber(
            predicted: torch.Tensor,
            expected: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            if not bool(mask.any()):
                return zero
            errors = (predicted - expected) * image_size
            return F.smooth_l1_loss(
                errors[mask],
                torch.zeros_like(errors[mask]),
                beta=4.0,
            ) / image_size

        point_loss = pixel_huber(predicted_points, aligned_points, point_mask)
        positive_param_loss = pixel_huber(
            predicted_positive,
            aligned_positive,
            positive_mask,
        )
        signed_param_loss = pixel_huber(
            predicted_signed,
            aligned_signed,
            signed_mask,
        )
        bond_mask = torch.cat(matched["types"]) != TEXT_BOX
        predicted_stroke = torch.cat(matched["predicted_stroke"]).float()
        expected_stroke = torch.cat(matched["expected_stroke"]).to(
            predicted_stroke.dtype
        )
        if bool(bond_mask.any()):
            stroke_loss = F.smooth_l1_loss(
                predicted_stroke[bond_mask],
                expected_stroke[bond_mask],
            )
        member_count_loss = F.cross_entropy(
            torch.cat(matched["predicted_member_count"]).float(),
            torch.cat(matched["expected_member_count"]).long(),
        )
    geometry_loss = point_loss + positive_param_loss + signed_param_loss

    if matched["predicted_text"]:
        predicted_text = torch.cat(matched["predicted_text"]).float()
        expected_text = torch.cat(matched["expected_text"]).to(predicted_text.dtype)
        errors = (predicted_text - expected_text) * image_size
        text_geometry_loss = F.smooth_l1_loss(
            errors,
            torch.zeros_like(errors),
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
            "Carrier Set count logits must have shape "
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
    )
    return {
        "loss": total,
        "classification_loss": classification_loss,
        "geometry_loss": geometry_loss,
        "carrier_point_loss": point_loss,
        "positive_param_loss": positive_param_loss,
        "signed_param_loss": signed_param_loss,
        "text_geometry_loss": text_geometry_loss,
        "stroke_loss": stroke_loss,
        "member_count_loss": member_count_loss,
        "cardinality_loss": cardinality_loss,
        "count_classification_loss": count_classification_loss,
        "predicted_count_mean": predicted_counts.mean(),
        "target_count_mean": expected_counts.mean(),
        "assignments": assignments,
    }


@torch.no_grad()
def decode_bond_group_carrier_set(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.5,
    mode: str = "threshold",
) -> list[list[dict[str, torch.Tensor | float | int | str]]]:
    """Decode structured carrier rows without expanding members or JSON."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Carrier Set threshold must be in [0, 1]")
    if mode not in {"threshold", "count_head"}:
        raise ValueError("Carrier Set decode mode must be threshold or count_head")
    probabilities = output["type_logits"].float().softmax(-1)
    object_scores, object_types = probabilities[..., 1:].max(dim=-1)
    object_types = object_types + 1
    no_object_scores = probabilities[..., NO_OBJECT]
    member_count_logits = output["member_count_logits"].float()
    if member_count_logits.size(-1) < 3:
        raise ValueError("Carrier Set member count logits must support counts >= 2")
    raw_member_counts = member_count_logits.argmax(dim=-1)
    hashed_member_counts = member_count_logits[..., 2:].argmax(dim=-1) + 2
    if mode == "count_head":
        expected_shape = (probabilities.size(0), probabilities.size(1) + 1)
        if (
            output.get("count_logits") is None
            or output["count_logits"].shape != expected_shape
        ):
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
                row["raw_member_count"] = int(
                    raw_member_counts[batch_index, query_index]
                )
                row["text_geometry"] = output["text_geometry"][
                    batch_index,
                    query_index,
                ].detach()
            else:
                raw_member_count = int(
                    raw_member_counts[batch_index, query_index]
                )
                member_count = FIXED_MEMBER_COUNTS.get(
                    object_type,
                    int(hashed_member_counts[batch_index, query_index]),
                )
                row.update(
                    {
                        "member_count": member_count,
                        "raw_member_count": raw_member_count,
                        "carrier_points": output["carrier_points"][
                            batch_index,
                            query_index,
                        ].detach(),
                        "positive_params": output["positive_params"][
                            batch_index,
                            query_index,
                        ].detach(),
                        "signed_params": output["signed_params"][
                            batch_index,
                            query_index,
                        ].detach(),
                        "group_stroke": output["group_stroke"][
                            batch_index,
                            query_index,
                        ].detach(),
                    }
                )
            rows.append(row)
        decoded.append(rows)
    return decoded
