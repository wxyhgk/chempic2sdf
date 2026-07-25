"""Proposal-anchored bond grouping without free group queries or geometry heads.

Every physical proposal is a potential representative.  Members point to one
representative through a symmetric proposal-to-proposal score matrix.  The
representative is latent during training: any member of a GT group may serve as
its root, so no arbitrary canonical member is introduced.
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
    PHYSICAL_GROUP_NAMES,
    BondGroupProposalMembershipConfig,
    ProposalDescriptorEncoder,
)


@dataclass(frozen=True, slots=True)
class BondGroupProposalPointerConfig:
    proposal_feature_dim: int = 256
    proposal_descriptor_dim: int = 23
    flat_type_classes: int = 5
    d_model: int = 192
    proposal_encoder_layers: int = 2
    heads: int = 6
    ffn_dim: int = 768
    max_members: int = 143
    max_group_count: int = 192
    predict_group_count: bool = True
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
            self.max_group_count,
        )
        if min(positive) < 1:
            raise ValueError("Proposal Pointer dimensions must be positive")
        if self.max_members < 3:
            raise ValueError("Proposal Pointer max_members must be at least three")
        if self.d_model % self.heads:
            raise ValueError("Proposal Pointer heads must divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Proposal Pointer dropout must be in [0, 1)")


def _membership_encoder_config(
    config: BondGroupProposalPointerConfig,
) -> BondGroupProposalMembershipConfig:
    """Build the exact adapter required by the shared descriptor encoder."""

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
        max_members=config.max_members,
        dropout=config.dropout,
    )


class BondGroupProposalPointerPredictor(nn.Module):
    """Predict proposal roots, group attributes and symmetric ownership scores."""

    model_type = "bond-group-proposal-pointer-v0.1"

    def __init__(
        self,
        config: BondGroupProposalPointerConfig = BondGroupProposalPointerConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.proposal_encoder = ProposalDescriptorEncoder(
            _membership_encoder_config(config)
        )
        self.keep_head = nn.Linear(config.d_model, 1)
        self.group_type_head = nn.Linear(config.d_model, len(PHYSICAL_GROUP_NAMES))
        self.member_count_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.max_members + 1),
        )
        self.pointer_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.pointer_node_bias = nn.Linear(config.d_model, 1)
        self.group_count_head: nn.Module | None = None
        if config.predict_group_count:
            self.group_count_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.max_group_count + 1),
            )
        self._reset_heads()

    def _reset_heads(self) -> None:
        with torch.no_grad():
            self.keep_head.bias.fill_(-2.0)
            self.group_type_head.bias.zero_()
            self.group_type_head.bias[NO_GROUP] = math.log(2.0)
            self.member_count_head[-1].bias.zero_()
            self.pointer_node_bias.bias.zero_()
            if self.group_count_head is not None:
                self.group_count_head[-1].bias.zero_()

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
        keep_logits = self.keep_head(hidden).squeeze(-1)
        group_type_logits = self.group_type_head(hidden)
        member_count_logits = self.member_count_head(hidden)

        pointer_vectors = F.normalize(
            self.pointer_projection(hidden).float(),
            dim=-1,
        ).to(hidden.dtype)
        pointer_logits = torch.einsum(
            "bpd,bqd->bpq",
            pointer_vectors,
            pointer_vectors,
        ) * math.sqrt(self.config.d_model)
        node_bias = self.pointer_node_bias(hidden).squeeze(-1)
        pointer_logits = pointer_logits + 0.5 * (
            node_bias.unsqueeze(-1) + node_bias.unsqueeze(-2)
        )
        pointer_logits = 0.5 * (
            pointer_logits + pointer_logits.transpose(-1, -2)
        )

        padded_type = torch.full_like(group_type_logits, -20.0)
        padded_type[..., NO_GROUP] = 20.0
        padded_count = torch.full_like(member_count_logits, -20.0)
        padded_count[..., 0] = 20.0
        keep_logits = keep_logits.masked_fill(~mask, -20.0)
        group_type_logits = torch.where(
            mask.unsqueeze(-1),
            group_type_logits,
            padded_type,
        )
        member_count_logits = torch.where(
            mask.unsqueeze(-1),
            member_count_logits,
            padded_count,
        )
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pointer_logits = pointer_logits.masked_fill(~pair_mask, -20.0)

        result = {
            "keep_logits": keep_logits,
            "group_type_logits": group_type_logits,
            "member_count_logits": member_count_logits,
            "pointer_logits": pointer_logits,
            "proposal_features": hidden,
        }
        if self.group_count_head is not None:
            weights = mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
                1.0
            )
            result["group_count_logits"] = self.group_count_head(pooled)
        return result

    def config_dict(self) -> dict[str, Any]:
        return {"proposal_pointer_config": asdict(self.config)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def type_macro_group_mean(
    group_values: torch.Tensor,
    group_types: torch.Tensor,
    group_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average within each type, then give every present type equal weight."""

    if group_values.ndim != 1 or group_types.shape != group_values.shape:
        raise ValueError("Macro group values and types must be one-dimensional")
    if not len(group_values):
        raise ValueError("Type-macro aggregation requires at least one group")
    if bool(
        ((group_types < 1) | (group_types >= len(PHYSICAL_GROUP_NAMES))).any()
    ):
        raise ValueError("Macro group type is outside the physical range")
    if group_weights is None:
        weights = torch.ones_like(group_values, dtype=torch.float32)
    else:
        if group_weights.shape != group_values.shape:
            raise ValueError("Macro group weights and values differ")
        weights = group_weights.float()
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError("Macro group weights must be finite and positive")
    values = group_values.float()
    type_means = []
    for group_type in torch.unique(group_types, sorted=True).tolist():
        selected = group_types == int(group_type)
        current_weights = weights[selected]
        type_means.append(
            (values[selected] * current_weights).sum()
            / current_weights.sum().clamp_min(1e-8)
        )
    return torch.stack(type_means).mean()


def candidate_root_ownership_costs(
    pointer_logits: torch.Tensor,
    member_mask: torch.Tensor,
    proposal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every member-root candidate and its mean ownership CE."""

    if pointer_logits.ndim != 2 or pointer_logits.size(0) != pointer_logits.size(1):
        raise ValueError("Pointer logits must be a square proposal matrix")
    proposals = pointer_logits.size(0)
    if member_mask.shape != (proposals,) or proposal_mask.shape != (proposals,):
        raise ValueError("Pointer member/proposal masks have invalid shapes")
    valid = proposal_mask.bool()
    members = member_mask.bool()
    if bool((members & ~valid).any()):
        raise ValueError("Group members include padded proposals")
    roots = torch.where(members)[0]
    if not len(roots):
        raise ValueError("A pointer group must contain at least one member")
    masked_logits = pointer_logits.float().masked_fill(~valid.unsqueeze(0), -1e4)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    costs = -log_probabilities[members][:, roots].mean(dim=0)
    return roots, costs


def _validate_targets(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> None:
    required_output = {
        "keep_logits",
        "group_type_logits",
        "member_count_logits",
        "pointer_logits",
    }
    required_targets = {
        "group_types",
        "group_membership",
        "group_mask",
        "member_counts",
    }
    missing_output = sorted(required_output - output.keys())
    missing_targets = sorted(required_targets - targets.keys())
    if missing_output or missing_targets:
        raise ValueError(
            f"Proposal Pointer fields are missing: output={missing_output}, "
            f"targets={missing_targets}"
        )
    keep = output["keep_logits"]
    if keep.ndim != 2:
        raise ValueError("Pointer keep logits must have shape [batch, proposals]")
    batch, proposals = keep.shape
    if proposal_mask.shape != (batch, proposals):
        raise ValueError("Pointer proposal mask has an invalid shape")
    if output["group_type_logits"].shape != (
        batch,
        proposals,
        len(PHYSICAL_GROUP_NAMES),
    ):
        raise ValueError("Pointer group type logits have an invalid shape")
    if output["member_count_logits"].shape[:2] != (batch, proposals):
        raise ValueError("Pointer member count logits have an invalid shape")
    if output["pointer_logits"].shape != (batch, proposals, proposals):
        raise ValueError("Pointer ownership logits have an invalid shape")
    if targets["group_types"].ndim != 2:
        raise ValueError("Pointer group target types must be two-dimensional")
    group_shape = targets["group_types"].shape
    if group_shape[0] != batch:
        raise ValueError("Pointer output and target batches differ")
    if targets["group_mask"].shape != group_shape:
        raise ValueError("Pointer group mask and types differ")
    if targets["member_counts"].shape != group_shape:
        raise ValueError("Pointer member counts and types differ")
    if targets["group_membership"].shape != (*group_shape, proposals):
        raise ValueError("Pointer group membership targets have an invalid shape")


def bond_group_proposal_pointer_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    group_weights: torch.Tensor | None = None,
    root_mode: str = "softmin",
    softmin_temperature: float = 0.25,
    pointer_weight: float = 2.0,
    keep_weight: float = 0.5,
    type_weight: float = 1.0,
    member_count_weight: float = 0.5,
    group_count_weight: float = 0.5,
    background_type_weight: float = 0.1,
    background_count_weight: float = 0.1,
) -> dict[str, Any]:
    """Latent-root ownership with group-normalized, type-macro losses."""

    if root_mode not in {"hard", "softmin"}:
        raise ValueError("Pointer root_mode must be hard or softmin")
    if softmin_temperature <= 0.0:
        raise ValueError("Pointer softmin temperature must be positive")
    weights = (
        pointer_weight,
        keep_weight,
        type_weight,
        member_count_weight,
        group_count_weight,
        background_type_weight,
        background_count_weight,
    )
    if min(weights) < 0.0:
        raise ValueError("Pointer loss weights must be nonnegative")
    _validate_targets(output, targets, proposal_mask)
    batch, proposals = output["keep_logits"].shape
    active_group_mask = targets["group_mask"].bool()
    supervision_weights = group_weights
    if supervision_weights is None:
        supervision_weights = targets.get("group_weights")
    if supervision_weights is not None:
        if supervision_weights.shape != active_group_mask.shape:
            raise ValueError("Pointer group weights and targets differ")
        active_weights = supervision_weights[active_group_mask].float()
        if not bool(torch.isfinite(active_weights).all()) or bool(
            (active_weights <= 0.0).any()
        ):
            raise ValueError("Active pointer group weights must be positive")

    selected_roots = torch.full_like(targets["group_types"], -1)
    pointer_group_losses = []
    type_group_losses = []
    count_group_losses = []
    positive_keep_losses = []
    negative_keep_group_losses = []
    negative_keep_group_types = []
    negative_keep_group_weights = []
    group_types = []
    resolved_group_weights = []
    background_keep_parts = []
    background_type_parts = []
    background_count_parts = []

    for batch_index in range(batch):
        valid = proposal_mask[batch_index].bool()
        occupied = torch.zeros(proposals, device=valid.device, dtype=torch.bool)
        target_indices = torch.where(active_group_mask[batch_index])[0]
        for target_index in target_indices.tolist():
            target_type = int(targets["group_types"][batch_index, target_index])
            target_count = int(targets["member_counts"][batch_index, target_index])
            members = targets["group_membership"][
                batch_index,
                target_index,
            ].bool()
            if bool((members & occupied).any()):
                raise ValueError("One proposal belongs to multiple pointer groups")
            if bool((members & ~valid).any()) or not bool(members.any()):
                raise ValueError("Pointer group membership is empty or padded")
            support_count = int(members.sum())
            if target_type == HASHED_WEDGE:
                if (
                    target_count < 2
                    or support_count < 2
                    or support_count > target_count
                ):
                    raise ValueError("Hashed pointer target has invalid support/count")
            else:
                fixed = FIXED_MEMBER_COUNTS.get(target_type)
                if fixed is None or target_count != fixed or support_count != fixed:
                    raise ValueError("Fixed pointer target has invalid support/count")
            if target_count >= output["member_count_logits"].size(-1):
                raise ValueError("Pointer target member count exceeds capacity")
            occupied |= members

            roots, root_costs = candidate_root_ownership_costs(
                output["pointer_logits"][batch_index],
                members,
                valid,
            )
            best = int(root_costs.argmin())
            selected_root = int(roots[best])
            selected_roots[batch_index, target_index] = selected_root
            if root_mode == "hard":
                pointer_loss = root_costs[best]
            else:
                pointer_loss = (
                    -softmin_temperature
                    * torch.logsumexp(-root_costs / softmin_temperature, dim=0)
                    + softmin_temperature * math.log(len(root_costs))
                )
            member_indices = torch.where(members)[0]
            type_targets = torch.full(
                (len(member_indices),),
                target_type,
                device=valid.device,
                dtype=torch.long,
            )
            count_targets = torch.full(
                (len(member_indices),),
                target_count,
                device=valid.device,
                dtype=torch.long,
            )
            type_loss = F.cross_entropy(
                output["group_type_logits"][batch_index, member_indices].float(),
                type_targets,
                reduction="mean",
            )
            count_loss = F.cross_entropy(
                output["member_count_logits"][batch_index, member_indices].float(),
                count_targets,
                reduction="mean",
            )
            positive_keep = F.softplus(
                -output["keep_logits"][batch_index, selected_root]
            )
            non_root_members = members.clone()
            non_root_members[selected_root] = False
            if bool(non_root_members.any()):
                negative_keep_group_losses.append(
                    F.softplus(
                        output["keep_logits"][batch_index, non_root_members]
                    ).mean()
                )
                negative_keep_group_types.append(target_type)
                negative_keep_group_weights.append(
                    1.0
                    if supervision_weights is None
                    else float(supervision_weights[batch_index, target_index])
                )
            pointer_group_losses.append(pointer_loss)
            type_group_losses.append(type_loss)
            count_group_losses.append(count_loss)
            positive_keep_losses.append(positive_keep)
            group_types.append(target_type)
            resolved_group_weights.append(
                1.0
                if supervision_weights is None
                else float(supervision_weights[batch_index, target_index])
            )

        background = valid & ~occupied
        if bool(background.any()):
            background_keep_parts.append(
                F.softplus(output["keep_logits"][batch_index, background]).mean()
            )
            background_type_parts.append(
                F.cross_entropy(
                    output["group_type_logits"][batch_index, background].float(),
                    torch.zeros(
                        int(background.sum()),
                        device=valid.device,
                        dtype=torch.long,
                    ),
                )
            )
            background_count_parts.append(
                F.cross_entropy(
                    output["member_count_logits"][batch_index, background].float(),
                    torch.zeros(
                        int(background.sum()),
                        device=valid.device,
                        dtype=torch.long,
                    ),
                )
            )

    if not pointer_group_losses:
        raise ValueError("Pointer loss requires at least one active group")
    device = output["keep_logits"].device
    group_type_tensor = torch.tensor(group_types, device=device, dtype=torch.long)
    group_weight_tensor = torch.tensor(
        resolved_group_weights,
        device=device,
        dtype=torch.float32,
    )
    pointer_loss = type_macro_group_mean(
        torch.stack(pointer_group_losses),
        group_type_tensor,
        group_weight_tensor,
    )
    group_type_loss = type_macro_group_mean(
        torch.stack(type_group_losses),
        group_type_tensor,
        group_weight_tensor,
    )
    member_count_loss = type_macro_group_mean(
        torch.stack(count_group_losses),
        group_type_tensor,
        group_weight_tensor,
    )
    positive_keep_loss = type_macro_group_mean(
        torch.stack(positive_keep_losses),
        group_type_tensor,
        group_weight_tensor,
    )
    zero = pointer_loss * 0.0
    negative_parts = []
    if negative_keep_group_losses:
        negative_parts.append(
            type_macro_group_mean(
                torch.stack(negative_keep_group_losses),
                torch.tensor(
                    negative_keep_group_types,
                    device=device,
                    dtype=torch.long,
                ),
                torch.tensor(
                    negative_keep_group_weights,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )
    if background_keep_parts:
        negative_parts.append(torch.stack(background_keep_parts).mean())
    negative_keep_loss = (
        torch.stack(negative_parts).mean() if negative_parts else zero
    )
    keep_loss = 0.5 * (positive_keep_loss + negative_keep_loss)
    background_type_loss = (
        torch.stack(background_type_parts).mean()
        if background_type_parts
        else zero
    )
    background_count_loss = (
        torch.stack(background_count_parts).mean()
        if background_count_parts
        else zero
    )

    group_count_logits = output.get("group_count_logits")
    if group_count_logits is None:
        group_count_loss = zero
    else:
        expected_counts = active_group_mask.sum(dim=-1).long()
        if group_count_logits.ndim != 2 or group_count_logits.size(0) != batch:
            raise ValueError("Pointer group count logits have an invalid shape")
        if int(expected_counts.max()) >= group_count_logits.size(-1):
            raise ValueError("Pointer target group count exceeds count capacity")
        group_count_loss = F.cross_entropy(
            group_count_logits.float(),
            expected_counts,
        )
    total = (
        pointer_weight * pointer_loss
        + keep_weight * keep_loss
        + type_weight * (
            group_type_loss + background_type_weight * background_type_loss
        )
        + member_count_weight * (
            member_count_loss + background_count_weight * background_count_loss
        )
        + group_count_weight * group_count_loss
    )
    return {
        "loss": total,
        "pointer_loss": pointer_loss,
        "keep_loss": keep_loss,
        "positive_keep_loss": positive_keep_loss,
        "negative_keep_loss": negative_keep_loss,
        "group_type_loss": group_type_loss,
        "background_type_loss": background_type_loss,
        "member_count_loss": member_count_loss,
        "background_count_loss": background_count_loss,
        "group_count_loss": group_count_loss,
        "selected_root_indices": selected_roots,
        "target_group_count_mean": active_group_mask.sum(dim=-1).float().mean(),
    }


@torch.no_grad()
def decode_bond_group_proposal_pointer(
    output: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    known_counts: torch.Tensor,
) -> list[list[dict[str, Any]]]:
    """Choose proposal roots, then assign every real proposal exactly once."""

    keep_logits = output["keep_logits"]
    pointer_logits = output["pointer_logits"]
    if keep_logits.ndim != 2:
        raise ValueError("Pointer decode keep logits must be two-dimensional")
    batch, proposals = keep_logits.shape
    if proposal_mask.shape != (batch, proposals):
        raise ValueError("Pointer decode proposal mask has an invalid shape")
    if known_counts.shape != (batch,):
        raise ValueError("Pointer decode known counts have an invalid shape")
    if pointer_logits.shape != (batch, proposals, proposals):
        raise ValueError("Pointer decode ownership logits have an invalid shape")
    if output["group_type_logits"].shape != (
        batch,
        proposals,
        len(PHYSICAL_GROUP_NAMES),
    ):
        raise ValueError("Pointer decode group type logits have an invalid shape")
    if output["member_count_logits"].shape[:2] != (batch, proposals):
        raise ValueError("Pointer decode member count logits have an invalid shape")

    keep_scores = keep_logits.float().sigmoid()
    group_types = output["group_type_logits"].float()[..., 1:].argmax(dim=-1) + 1
    raw_counts = output["member_count_logits"].float().argmax(dim=-1)
    hashed_counts = output["member_count_logits"].float()[..., 2:].argmax(dim=-1) + 2
    decoded = []
    for batch_index in range(batch):
        valid_indices = torch.where(proposal_mask[batch_index].bool())[0]
        count = int(known_counts[batch_index])
        if count < 0 or count > len(valid_indices):
            raise ValueError("Known pointer root count exceeds real proposals")
        if len(valid_indices) and count == 0:
            raise ValueError("Real proposals require at least one pointer root")
        ranked = valid_indices[
            torch.argsort(
                keep_scores[batch_index, valid_indices],
                descending=True,
                stable=True,
            )
        ]
        roots = ranked[:count]
        ownership = pointer_logits[batch_index][valid_indices][:, roots]
        assigned = ownership.argmax(dim=-1)
        root_position = {int(root): index for index, root in enumerate(roots.tolist())}
        for row_index, proposal in enumerate(valid_indices.tolist()):
            if proposal in root_position:
                assigned[row_index] = root_position[proposal]
        rows = []
        for root_offset, root in enumerate(roots.tolist()):
            members = valid_indices[assigned == root_offset].tolist()
            group_type = int(group_types[batch_index, root])
            raw_count = int(raw_counts[batch_index, root])
            member_count = (
                int(hashed_counts[batch_index, root])
                if group_type == HASHED_WEDGE
                else FIXED_MEMBER_COUNTS[group_type]
            )
            support_count = len(members)
            valid = (
                2 <= support_count <= member_count
                if group_type == HASHED_WEDGE
                else support_count == member_count
            )
            rows.append(
                {
                    "root_index": root,
                    "query_index": root,
                    "score": float(keep_scores[batch_index, root]),
                    "type": group_type,
                    "kind": PHYSICAL_GROUP_NAMES[group_type],
                    "raw_member_count": raw_count,
                    "member_count": member_count,
                    "support_count": support_count,
                    "proposal_indices": members,
                    "valid_membership": valid,
                }
            )
        decoded.append(rows)
    return decoded


__all__ = [
    "BondGroupProposalPointerConfig",
    "BondGroupProposalPointerPredictor",
    "bond_group_proposal_pointer_loss",
    "candidate_root_ownership_costs",
    "decode_bond_group_proposal_pointer",
    "type_macro_group_mean",
]
