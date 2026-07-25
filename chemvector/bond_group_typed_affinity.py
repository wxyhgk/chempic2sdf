"""Typed proposal affinity for packing physical bond primitives.

The model keeps frozen flat proposals as the only geometric objects.  It
predicts a physical group type for every proposal and a symmetric typed
affinity for every unordered proposal pair.  There are no group slots,
representatives, or geometry regression heads in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_group_proposal_membership import (
    FIXED_MEMBER_COUNTS,
    HASHED_WEDGE,
    NO_GROUP,
    PARALLEL_2,
    PARALLEL_3,
    PHYSICAL_GROUP_NAMES,
    BondGroupProposalMembershipConfig,
    ProposalDescriptorEncoder,
)


PAIR_NONE = 0
PAIR_PARALLEL_2 = 1
PAIR_PARALLEL_3 = 2
PAIR_HASHED = 3
PAIR_IGNORE_INDEX = -100
PAIR_CLASS_NAMES = (
    "none",
    "parallel_2",
    "parallel_3",
    "hashed",
)
PAIR_CLASS_FOR_GROUP_TYPE = {
    PARALLEL_2: PAIR_PARALLEL_2,
    PARALLEL_3: PAIR_PARALLEL_3,
    HASHED_WEDGE: PAIR_HASHED,
}


@dataclass(frozen=True, slots=True)
class BondGroupTypedAffinityConfig:
    proposal_feature_dim: int = 256
    proposal_descriptor_dim: int = 23
    flat_type_classes: int = 5
    d_model: int = 192
    proposal_encoder_layers: int = 2
    heads: int = 6
    ffn_dim: int = 768
    pair_dim: int = 96
    pair_hidden_dim: int = 192
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.proposal_feature_dim,
            self.proposal_descriptor_dim,
            self.flat_type_classes,
            self.d_model,
            self.proposal_encoder_layers,
            self.heads,
            self.ffn_dim,
            self.pair_dim,
            self.pair_hidden_dim,
        )
        if min(positive) < 1:
            raise ValueError("Typed Affinity dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("Typed Affinity heads must divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Typed Affinity dropout must be in [0, 1)")


def _membership_encoder_config(
    config: BondGroupTypedAffinityConfig,
) -> BondGroupProposalMembershipConfig:
    return BondGroupProposalMembershipConfig(
        proposal_feature_dim=config.proposal_feature_dim,
        proposal_descriptor_dim=config.proposal_descriptor_dim,
        flat_type_classes=config.flat_type_classes,
        d_model=config.d_model,
        proposal_encoder_layers=config.proposal_encoder_layers,
        group_decoder_layers=1,
        heads=config.heads,
        ffn_dim=config.ffn_dim,
        group_queries=1,
        max_members=3,
        dropout=config.dropout,
    )


class BondGroupTypedAffinityPredictor(nn.Module):
    """Predict proposal types and symmetric typed pair affinities."""

    model_type = "bond-group-typed-affinity-v0.1"

    def __init__(
        self,
        config: BondGroupTypedAffinityConfig = BondGroupTypedAffinityConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.proposal_encoder = ProposalDescriptorEncoder(
            _membership_encoder_config(config)
        )
        self.node_group_type_head = nn.Linear(
            config.d_model,
            len(PHYSICAL_GROUP_NAMES),
        )
        self.pair_node_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.pair_dim),
            nn.GELU(),
            nn.LayerNorm(config.pair_dim),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(3 * config.pair_dim, config.pair_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pair_hidden_dim),
            nn.Linear(config.pair_hidden_dim, len(PAIR_CLASS_NAMES)),
        )
        self._reset_heads()

    def _reset_heads(self) -> None:
        with torch.no_grad():
            self.node_group_type_head.bias.zero_()
            self.node_group_type_head.bias[NO_GROUP] = math.log(2.0)
            self.pair_head[-1].bias.zero_()
            self.pair_head[-1].bias[PAIR_NONE] = math.log(3.0)

    def forward(
        self,
        proposal_features: torch.Tensor,
        proposal_descriptors: torch.Tensor,
        flat_type_logits: torch.Tensor,
        proposal_scores: torch.Tensor,
        proposal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = proposal_mask.bool()
        hidden = self.proposal_encoder(
            proposal_features,
            proposal_descriptors,
            flat_type_logits,
            proposal_scores,
            mask,
        )
        node_logits = self.node_group_type_head(hidden)

        pair_nodes = self.pair_node_encoder(hidden)
        first = pair_nodes.unsqueeze(2)
        second = pair_nodes.unsqueeze(1)
        symmetric_features = torch.cat(
            (
                first + second,
                (first - second).abs(),
                first * second,
            ),
            dim=-1,
        )
        pair_logits = self.pair_head(symmetric_features)
        pair_logits = 0.5 * (pair_logits + pair_logits.transpose(1, 2))

        padded_node = torch.full_like(node_logits, -20.0)
        padded_node[..., NO_GROUP] = 20.0
        node_logits = torch.where(mask.unsqueeze(-1), node_logits, padded_node)

        proposals = mask.size(1)
        diagonal = torch.eye(
            proposals,
            device=mask.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        pair_mask = (
            mask.unsqueeze(-1)
            & mask.unsqueeze(-2)
            & ~diagonal
        )
        isolated_pair = torch.full_like(pair_logits, -20.0)
        isolated_pair[..., PAIR_NONE] = 20.0
        pair_logits = torch.where(
            pair_mask.unsqueeze(-1),
            pair_logits,
            isolated_pair,
        )

        return {
            "node_group_type_logits": node_logits,
            "pair_logits": pair_logits,
            "pair_mask": pair_mask,
            "proposal_features": hidden,
        }

    def config_dict(self) -> dict[str, Any]:
        return {"typed_affinity_config": asdict(self.config)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _validate_aggregate_targets(
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> tuple[int, int, int]:
    required = {
        "group_membership",
        "group_types",
        "group_mask",
        "proposal_keep",
    }
    missing = sorted(required - targets.keys())
    if missing:
        raise ValueError(f"Typed Affinity target fields are missing: {missing}")
    if proposal_mask.ndim != 2:
        raise ValueError("Typed Affinity proposal mask must be two-dimensional")
    batch, proposals = proposal_mask.shape
    group_types = targets["group_types"]
    if group_types.ndim != 2 or group_types.size(0) != batch:
        raise ValueError("Typed Affinity group types have an invalid shape")
    groups = group_types.size(1)
    if targets["group_mask"].shape != (batch, groups):
        raise ValueError("Typed Affinity group mask and types differ")
    if targets["group_membership"].shape != (batch, groups, proposals):
        raise ValueError("Typed Affinity group membership has an invalid shape")
    if targets["proposal_keep"].shape != (batch, proposals):
        raise ValueError("Typed Affinity proposal_keep has an invalid shape")

    mask = proposal_mask.bool()
    active = targets["group_mask"].bool()
    membership = targets["group_membership"].bool()
    keep = targets["proposal_keep"].bool()
    if bool((keep & ~mask).any()):
        raise ValueError("Typed Affinity proposal_keep includes padding")
    if bool(membership[~active].any()):
        raise ValueError("Inactive Typed Affinity groups contain members")
    if bool((membership & ~mask.unsqueeze(1)).any()):
        raise ValueError("Typed Affinity group membership includes padding")
    if bool((group_types[~active] != NO_GROUP).any()):
        raise ValueError("Inactive Typed Affinity group types must be no_object")
    active_types = group_types[active]
    if bool(
        ((active_types <= NO_GROUP) | (active_types >= len(PHYSICAL_GROUP_NAMES))).any()
    ):
        raise ValueError("Typed Affinity group type is outside the physical range")

    claimed_count = membership.sum(dim=1)
    if bool((claimed_count > 1).any()):
        raise ValueError("One proposal belongs to multiple Typed Affinity groups")
    membership_union = claimed_count > 0
    if not torch.equal(membership_union, keep):
        raise ValueError(
            "Typed Affinity proposal_keep must equal the active membership union"
        )

    for batch_index in range(batch):
        for group_index in torch.where(active[batch_index])[0].tolist():
            group_type = int(group_types[batch_index, group_index])
            member_count = int(membership[batch_index, group_index].sum())
            if group_type == HASHED_WEDGE:
                if member_count < 2:
                    raise ValueError(
                        "Hashed Typed Affinity groups need at least two proposals"
                    )
            else:
                fixed_count = FIXED_MEMBER_COUNTS.get(group_type)
                if fixed_count is None or member_count != fixed_count:
                    raise ValueError(
                        "Fixed Typed Affinity group has invalid membership"
                    )
    return batch, groups, proposals


def _unordered_pairs(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(indices)
    if count < 2:
        empty = indices.new_empty((0,))
        return empty, empty
    offsets = torch.triu_indices(
        count,
        count,
        offset=1,
        device=indices.device,
    )
    return indices[offsets[0]], indices[offsets[1]]


def build_typed_affinity_targets(
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert aggregate group targets to node labels and symmetric pair labels."""

    batch, _, proposals = _validate_aggregate_targets(targets, proposal_mask)
    device = proposal_mask.device
    mask = proposal_mask.bool()
    active = targets["group_mask"].bool()
    membership = targets["group_membership"].bool()

    node_labels = torch.full(
        (batch, proposals),
        NO_GROUP,
        device=device,
        dtype=torch.long,
    )
    diagonal = torch.eye(proposals, device=device, dtype=torch.bool).unsqueeze(0)
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2) & ~diagonal
    pair_labels = torch.full(
        (batch, proposals, proposals),
        PAIR_IGNORE_INDEX,
        device=device,
        dtype=torch.long,
    )
    pair_labels[pair_mask] = PAIR_NONE
    pair_group_indices = torch.full_like(pair_labels, -1)

    for batch_index in range(batch):
        for group_index in torch.where(active[batch_index])[0].tolist():
            group_type = int(targets["group_types"][batch_index, group_index])
            members = membership[batch_index, group_index]
            member_indices = torch.where(members)[0]
            node_labels[batch_index, member_indices] = group_type
            pair_class = PAIR_CLASS_FOR_GROUP_TYPE.get(group_type)
            if pair_class is None:
                continue
            first, second = _unordered_pairs(member_indices)
            pair_labels[batch_index, first, second] = pair_class
            pair_labels[batch_index, second, first] = pair_class
            pair_group_indices[batch_index, first, second] = group_index
            pair_group_indices[batch_index, second, first] = group_index

    return {
        "node_group_type_labels": node_labels,
        "node_mask": mask,
        "pair_labels": pair_labels,
        "pair_mask": pair_mask,
        "pair_group_indices": pair_group_indices,
        "positive_pair_mask": pair_labels > PAIR_NONE,
    }


def type_macro_group_mean(
    group_values: torch.Tensor,
    group_types: torch.Tensor,
    group_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average groups within each present type, then average present types."""

    if group_values.ndim != 1 or group_types.shape != group_values.shape:
        raise ValueError("Typed Affinity macro values and types must be vectors")
    if not len(group_values):
        raise ValueError("Typed Affinity macro mean requires at least one group")
    if bool(
        ((group_types <= NO_GROUP) | (group_types >= len(PHYSICAL_GROUP_NAMES))).any()
    ):
        raise ValueError("Typed Affinity macro type is outside the physical range")
    if group_weights is None:
        weights = torch.ones_like(group_values, dtype=torch.float32)
    else:
        if group_weights.shape != group_values.shape:
            raise ValueError("Typed Affinity macro weights and values differ")
        weights = group_weights.float()
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError("Typed Affinity macro weights must be finite and positive")

    values = group_values.float()
    type_means = []
    for group_type in range(1, len(PHYSICAL_GROUP_NAMES)):
        selected = group_types == group_type
        if not bool(selected.any()):
            continue
        current_weights = weights[selected]
        type_means.append(
            (values[selected] * current_weights).sum()
            / current_weights.sum().clamp_min(1e-8)
        )
    return torch.stack(type_means).mean()


def _group_weights(
    targets: dict[str, torch.Tensor],
    active: torch.Tensor,
) -> torch.Tensor | None:
    weights = targets.get("group_weights")
    if weights is None:
        return None
    if weights.shape != active.shape:
        raise ValueError("Typed Affinity group weights and masks differ")
    selected = weights[active].float()
    if not bool(torch.isfinite(selected).all()) or bool((selected <= 0.0).any()):
        raise ValueError("Active Typed Affinity group weights must be positive")
    return weights.float()


def _node_type_losses(
    node_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> dict[str, Any]:
    active = targets["group_mask"].bool()
    membership = targets["group_membership"].bool()
    weights = _group_weights(targets, active)
    group_losses = []
    group_types = []
    selected_weights = []
    for batch_index in range(active.size(0)):
        for group_index in torch.where(active[batch_index])[0].tolist():
            group_type = int(targets["group_types"][batch_index, group_index])
            member_indices = torch.where(
                membership[batch_index, group_index]
            )[0]
            group_losses.append(
                F.cross_entropy(
                    node_logits[batch_index, member_indices].float(),
                    torch.full(
                        (len(member_indices),),
                        group_type,
                        device=node_logits.device,
                        dtype=torch.long,
                    ),
                    reduction="mean",
                )
            )
            group_types.append(group_type)
            if weights is not None:
                selected_weights.append(weights[batch_index, group_index])

    zero = node_logits.sum() * 0.0
    if group_losses:
        group_loss_tensor = torch.stack(group_losses)
        group_type_tensor = torch.tensor(
            group_types,
            device=node_logits.device,
            dtype=torch.long,
        )
        weight_tensor = (
            torch.stack(selected_weights) if selected_weights else None
        )
        positive = type_macro_group_mean(
            group_loss_tensor,
            group_type_tensor,
            weight_tensor,
        )
    else:
        positive = zero

    background_parts = []
    background = proposal_mask.bool() & ~targets["proposal_keep"].bool()
    for batch_index in range(background.size(0)):
        if bool(background[batch_index].any()):
            count = int(background[batch_index].sum())
            background_parts.append(
                F.cross_entropy(
                    node_logits[batch_index, background[batch_index]].float(),
                    torch.zeros(
                        count,
                        device=node_logits.device,
                        dtype=torch.long,
                    ),
                )
            )
    background_loss = (
        torch.stack(background_parts).mean() if background_parts else zero
    )
    return {
        "positive": positive,
        "background": background_loss,
        "group_count": len(group_losses),
        "background_image_count": len(background_parts),
    }


def node_type_macro_loss(
    node_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    background_weight: float = 0.1,
) -> dict[str, Any]:
    """Return group/type-macro node loss with a separate background term."""

    if background_weight < 0.0:
        raise ValueError("Typed Affinity background weight must be nonnegative")
    batch, _, proposals = _validate_aggregate_targets(targets, proposal_mask)
    if node_logits.shape != (batch, proposals, len(PHYSICAL_GROUP_NAMES)):
        raise ValueError("Typed Affinity node logits have an invalid shape")
    parts = _node_type_losses(node_logits, targets, proposal_mask)
    parts["loss"] = parts["positive"] + background_weight * parts["background"]
    return parts


def _positive_pair_loss(
    pair_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    active = targets["group_mask"].bool()
    membership = targets["group_membership"].bool()
    weights = _group_weights(targets, active)
    group_losses = []
    group_types = []
    selected_weights = []
    positive_pair_count = 0
    for batch_index in range(active.size(0)):
        for group_index in torch.where(active[batch_index])[0].tolist():
            group_type = int(targets["group_types"][batch_index, group_index])
            pair_class = PAIR_CLASS_FOR_GROUP_TYPE.get(group_type)
            if pair_class is None:
                continue
            member_indices = torch.where(
                membership[batch_index, group_index]
            )[0]
            first, second = _unordered_pairs(member_indices)
            positive_pair_count += len(first)
            group_losses.append(
                F.cross_entropy(
                    pair_logits[batch_index, first, second].float(),
                    torch.full(
                        (len(first),),
                        pair_class,
                        device=pair_logits.device,
                        dtype=torch.long,
                    ),
                    reduction="mean",
                )
            )
            group_types.append(group_type)
            if weights is not None:
                selected_weights.append(weights[batch_index, group_index])

    zero = pair_logits.sum() * 0.0
    if not group_losses:
        return {
            "loss": zero,
            "group_count": 0,
            "pair_count": 0,
        }
    return {
        "loss": type_macro_group_mean(
            torch.stack(group_losses),
            torch.tensor(group_types, device=pair_logits.device, dtype=torch.long),
            torch.stack(selected_weights) if selected_weights else None,
        ),
        "group_count": len(group_losses),
        "pair_count": positive_pair_count,
    }


def positive_affinity_pair_loss(
    pair_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> dict[str, Any]:
    """Average pairs per group, then macro-average p2/p3/hashed groups."""

    batch, _, proposals = _validate_aggregate_targets(targets, proposal_mask)
    if pair_logits.shape != (
        batch,
        proposals,
        proposals,
        len(PAIR_CLASS_NAMES),
    ):
        raise ValueError("Typed Affinity pair logits have an invalid shape")
    return _positive_pair_loss(pair_logits, targets)


def hard_negative_affinity_pair_loss(
    pair_logits: torch.Tensor,
    affinity_targets: dict[str, torch.Tensor],
    *,
    top_k: int | None = 64,
) -> dict[str, Any]:
    """Mine unordered negatives per image and macro-average image buckets.

    Images containing a positive affinity and images containing only negative
    affinities form separate buckets.  Consequently, an arbitrarily large
    all-negative bucket cannot outweigh the positive-containing bucket.
    """

    if top_k is not None and top_k < 1:
        raise ValueError("Typed Affinity negative top_k must be positive")
    labels = affinity_targets.get("pair_labels")
    pair_mask = affinity_targets.get("pair_mask")
    if labels is None or pair_mask is None:
        raise ValueError("Typed Affinity pair labels or mask are missing")
    if labels.ndim != 3 or pair_mask.shape != labels.shape:
        raise ValueError("Typed Affinity pair labels and mask differ")
    batch, proposals, second_proposals = labels.shape
    if proposals != second_proposals or pair_logits.shape != (
        batch,
        proposals,
        proposals,
        len(PAIR_CLASS_NAMES),
    ):
        raise ValueError("Typed Affinity negative logits have an invalid shape")

    upper = torch.triu(
        torch.ones(
            (proposals, proposals),
            device=pair_logits.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    positive_image_parts = []
    all_negative_image_parts = []
    selected_pair_count = 0
    for batch_index in range(batch):
        valid_upper = pair_mask[batch_index].bool() & upper
        negative = valid_upper & (labels[batch_index] == PAIR_NONE)
        if not bool(negative.any()):
            continue
        values = F.cross_entropy(
            pair_logits[batch_index, negative].float(),
            torch.full(
                (int(negative.sum()),),
                PAIR_NONE,
                device=pair_logits.device,
                dtype=torch.long,
            ),
            reduction="none",
        )
        if top_k is not None and len(values) > top_k:
            values = values.topk(top_k, largest=True, sorted=False).values
        selected_pair_count += len(values)
        image_loss = values.mean()
        has_positive = bool(
            (valid_upper & (labels[batch_index] > PAIR_NONE)).any()
        )
        if has_positive:
            positive_image_parts.append(image_loss)
        else:
            all_negative_image_parts.append(image_loss)

    zero = pair_logits.sum() * 0.0
    bucket_losses = []
    positive_image_loss = zero
    all_negative_image_loss = zero
    if positive_image_parts:
        positive_image_loss = torch.stack(positive_image_parts).mean()
        bucket_losses.append(positive_image_loss)
    if all_negative_image_parts:
        all_negative_image_loss = torch.stack(all_negative_image_parts).mean()
        bucket_losses.append(all_negative_image_loss)
    loss = torch.stack(bucket_losses).mean() if bucket_losses else zero
    return {
        "loss": loss,
        "positive_image_loss": positive_image_loss,
        "all_negative_image_loss": all_negative_image_loss,
        "positive_image_count": len(positive_image_parts),
        "all_negative_image_count": len(all_negative_image_parts),
        "selected_pair_count": selected_pair_count,
    }


def _validate_output(
    output: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> tuple[int, int]:
    required = {"node_group_type_logits", "pair_logits"}
    missing = sorted(required - output.keys())
    if missing:
        raise ValueError(f"Typed Affinity output fields are missing: {missing}")
    if proposal_mask.ndim != 2:
        raise ValueError("Typed Affinity proposal mask must be two-dimensional")
    batch, proposals = proposal_mask.shape
    if output["node_group_type_logits"].shape != (
        batch,
        proposals,
        len(PHYSICAL_GROUP_NAMES),
    ):
        raise ValueError("Typed Affinity node logits have an invalid shape")
    if output["pair_logits"].shape != (
        batch,
        proposals,
        proposals,
        len(PAIR_CLASS_NAMES),
    ):
        raise ValueError("Typed Affinity pair logits have an invalid shape")
    if not torch.equal(
        output["pair_logits"],
        output["pair_logits"].transpose(1, 2),
    ):
        raise ValueError("Typed Affinity pair logits must be strictly symmetric")
    return batch, proposals


def bond_group_typed_affinity_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    node_type_weight: float = 1.0,
    node_background_weight: float = 0.1,
    pair_positive_weight: float = 1.0,
    pair_negative_weight: float = 0.25,
    negative_top_k: int | None = 64,
) -> dict[str, Any]:
    """Joint node-type and typed-affinity supervision without object slots."""

    weights = (
        node_type_weight,
        node_background_weight,
        pair_positive_weight,
        pair_negative_weight,
    )
    if min(weights) < 0.0:
        raise ValueError("Typed Affinity loss weights must be nonnegative")
    _validate_output(output, proposal_mask)
    _validate_aggregate_targets(targets, proposal_mask)
    affinity_targets = build_typed_affinity_targets(targets, proposal_mask)
    node_parts = _node_type_losses(
        output["node_group_type_logits"],
        targets,
        proposal_mask,
    )
    positive_pair = _positive_pair_loss(output["pair_logits"], targets)
    negative_pair = hard_negative_affinity_pair_loss(
        output["pair_logits"],
        affinity_targets,
        top_k=negative_top_k,
    )
    node_loss = (
        node_parts["positive"]
        + node_background_weight * node_parts["background"]
    )
    pair_loss = (
        pair_positive_weight * positive_pair["loss"]
        + pair_negative_weight * negative_pair["loss"]
    )
    loss = node_type_weight * node_loss + pair_loss
    return {
        "loss": loss,
        "node_type_loss": node_loss,
        "node_type_positive_loss": node_parts["positive"],
        "node_type_background_loss": node_parts["background"],
        "pair_loss": pair_loss,
        "pair_positive_loss": positive_pair["loss"],
        "pair_negative_loss": negative_pair["loss"],
        "positive_group_count": positive_pair["group_count"],
        "positive_pair_count": positive_pair["pair_count"],
        "negative_pair_count": negative_pair["selected_pair_count"],
        "negative_positive_image_count": negative_pair["positive_image_count"],
        "negative_all_negative_image_count": negative_pair[
            "all_negative_image_count"
        ],
        "affinity_targets": affinity_targets,
    }


__all__ = [
    "PAIR_NONE",
    "PAIR_PARALLEL_2",
    "PAIR_PARALLEL_3",
    "PAIR_HASHED",
    "PAIR_IGNORE_INDEX",
    "PAIR_CLASS_NAMES",
    "BondGroupTypedAffinityConfig",
    "BondGroupTypedAffinityPredictor",
    "build_typed_affinity_targets",
    "type_macro_group_mean",
    "node_type_macro_loss",
    "positive_affinity_pair_loss",
    "hard_negative_affinity_pair_loss",
    "bond_group_typed_affinity_loss",
]
