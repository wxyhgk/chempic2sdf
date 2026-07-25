"""Group frozen physical proposals without regressing new coordinates.

This module is the v0.9.5 structural core.  A frozen flat primitive model is
expected to provide padded proposal features, descriptors, type logits and
scores.  The predictor only decides which proposals should survive, which
physical primitive type they represent, and which visible bond group owns
them.  It deliberately has no coordinate or geometry regression head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_group_set import (
    HASHED_WEDGE,
    NO_OBJECT as NO_GROUP,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
    SOLID_QUAD,
    SOLID_TRIANGLE,
)
from chemvector.primitive_set import (
    LINE as PROPOSAL_LINE,
    NO_OBJECT as NO_PROPOSAL,
    QUAD as PROPOSAL_QUAD,
    TRIANGLE as PROPOSAL_TRIANGLE,
)


PHYSICAL_GROUP_TYPES = (
    SINGLE_LINE,
    PARALLEL_2,
    PARALLEL_3,
    HASHED_WEDGE,
    SOLID_TRIANGLE,
    SOLID_QUAD,
)
PHYSICAL_GROUP_NAMES = (
    "no_object",
    "single_line",
    "parallel_2",
    "parallel_3",
    "hashed_wedge",
    "solid_triangle",
    "solid_quad",
)
PHYSICAL_PROPOSAL_TYPES = (
    "no_object",
    "line",
    "triangle",
    "quad",
)
FIXED_MEMBER_COUNTS = {
    SINGLE_LINE: 1,
    PARALLEL_2: 2,
    PARALLEL_3: 3,
    SOLID_TRIANGLE: 1,
    SOLID_QUAD: 1,
}
LINE_GROUP_TYPES = (SINGLE_LINE, PARALLEL_2, PARALLEL_3, HASHED_WEDGE)


@dataclass(frozen=True, slots=True)
class BondGroupProposalMembershipConfig:
    """Architecture for proposal-conditioned physical group membership."""

    proposal_feature_dim: int = 256
    proposal_descriptor_dim: int = 23
    flat_type_classes: int = 5
    d_model: int = 192
    proposal_encoder_layers: int = 2
    group_decoder_layers: int = 2
    heads: int = 6
    ffn_dim: int = 768
    group_queries: int = 192
    max_members: int = 143
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.proposal_feature_dim,
            self.proposal_descriptor_dim,
            self.flat_type_classes,
            self.d_model,
            self.proposal_encoder_layers,
            self.group_decoder_layers,
            self.heads,
            self.ffn_dim,
            self.group_queries,
        )
        if min(positive) < 1:
            raise ValueError("Proposal Membership dimensions must be positive")
        if self.max_members < 3:
            raise ValueError("Proposal Membership max_members must be at least three")
        if self.d_model % self.heads:
            raise ValueError("Proposal Membership heads must divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Proposal Membership dropout must be in [0, 1)")


class ProposalDescriptorEncoder(nn.Module):
    """Fuse frozen query features with explicit physical descriptors."""

    def __init__(self, config: BondGroupProposalMembershipConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(config.proposal_feature_dim),
            nn.Linear(config.proposal_feature_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        prior_dim = config.proposal_descriptor_dim + config.flat_type_classes + 1
        self.descriptor_encoder = nn.Sequential(
            nn.LayerNorm(prior_dim),
            nn.Linear(prior_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.proposal_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        proposal_features: torch.Tensor,
        proposal_descriptors: torch.Tensor,
        flat_type_logits: torch.Tensor,
        proposal_scores: torch.Tensor,
        proposal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if proposal_features.ndim != 3:
            raise ValueError("Proposal features must have shape [batch, proposals, dim]")
        batch, proposals, feature_dim = proposal_features.shape
        expected = {
            "proposal_features": (
                batch,
                proposals,
                self.config.proposal_feature_dim,
            ),
            "proposal_descriptors": (
                batch,
                proposals,
                self.config.proposal_descriptor_dim,
            ),
            "flat_type_logits": (
                batch,
                proposals,
                self.config.flat_type_classes,
            ),
            "proposal_scores": (batch, proposals),
            "proposal_mask": (batch, proposals),
        }
        values = {
            "proposal_features": proposal_features,
            "proposal_descriptors": proposal_descriptors,
            "flat_type_logits": flat_type_logits,
            "proposal_scores": proposal_scores,
            "proposal_mask": proposal_mask,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(
                    f"{name} has shape {tuple(values[name].shape)}, expected {shape}"
                )
        mask = proposal_mask.bool()
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("Every image must contain at least one padded proposal")
        for name in (
            "proposal_features",
            "proposal_descriptors",
            "flat_type_logits",
            "proposal_scores",
        ):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"{name} contains non-finite values")

        feature = self.feature_encoder(proposal_features)
        type_probabilities = flat_type_logits.float().softmax(dim=-1).to(
            proposal_descriptors.dtype
        )
        priors = torch.cat(
            (
                proposal_descriptors,
                type_probabilities,
                proposal_scores.unsqueeze(-1).to(proposal_descriptors.dtype),
            ),
            dim=-1,
        )
        descriptor = self.descriptor_encoder(priors)
        hidden = self.fusion(torch.cat((feature, descriptor), dim=-1))
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden = self.set_encoder(hidden, src_key_padding_mask=~mask)
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class BondGroupProposalMembershipPredictor(nn.Module):
    """Select and partition frozen proposals into physical bond groups."""

    model_type = "bond-group-proposal-membership-v0.1"

    def __init__(
        self,
        config: BondGroupProposalMembershipConfig = (
            BondGroupProposalMembershipConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.proposal_encoder = ProposalDescriptorEncoder(config)
        self.group_query_embedding = nn.Embedding(config.group_queries, config.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.group_decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.group_decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.proposal_keep_head = nn.Linear(config.d_model, 1)
        self.proposal_type_head = nn.Linear(
            config.d_model,
            len(PHYSICAL_PROPOSAL_TYPES),
        )
        self.group_type_head = nn.Linear(config.d_model, len(PHYSICAL_GROUP_NAMES))
        self.member_count_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.max_members + 1),
        )
        self.group_count_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.group_queries + 1),
        )
        self.membership_group_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.membership_proposal_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.membership_group_bias = nn.Linear(config.d_model, 1)
        self._reset_heads()

    def _reset_heads(self) -> None:
        nn.init.normal_(self.group_query_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.group_type_head.bias.zero_()
            self.group_type_head.bias[NO_GROUP] = math.log(14.0)
            self.proposal_type_head.bias.zero_()
            self.proposal_type_head.bias[NO_PROPOSAL] = math.log(4.0)
            self.proposal_keep_head.bias.fill_(-1.0)
            self.member_count_head[-1].bias.zero_()
            self.group_count_head[-1].bias.zero_()
            # With 192 slots, a milder prior would make the initial sum of
            # membership probabilities exceed one for almost every proposal.
            self.membership_group_bias.bias.fill_(-5.0)

    def forward(
        self,
        proposal_features: torch.Tensor,
        proposal_descriptors: torch.Tensor,
        flat_type_logits: torch.Tensor,
        proposal_scores: torch.Tensor,
        proposal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = proposal_mask.bool()
        proposal_hidden = self.proposal_encoder(
            proposal_features,
            proposal_descriptors,
            flat_type_logits,
            proposal_scores,
            mask,
        )
        batch = proposal_hidden.size(0)
        group_queries = self.group_query_embedding.weight.to(
            proposal_hidden.dtype
        ).unsqueeze(0).expand(batch, -1, -1)
        group_hidden = self.group_decoder(
            group_queries,
            proposal_hidden,
            memory_key_padding_mask=~mask,
        )
        group_vectors = self.membership_group_projection(group_hidden)
        proposal_vectors = self.membership_proposal_projection(proposal_hidden)
        membership_logits = torch.einsum(
            "bgd,bpd->bgp",
            group_vectors,
            proposal_vectors,
        ) / math.sqrt(self.config.d_model)
        membership_logits = membership_logits + self.membership_group_bias(
            group_hidden
        )
        membership_logits = membership_logits.masked_fill(
            ~mask[:, None, :],
            -20.0,
        )
        weights = mask.to(proposal_hidden.dtype).unsqueeze(-1)
        pooled = (proposal_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
            1.0
        )
        return {
            "proposal_keep_logits": self.proposal_keep_head(proposal_hidden).squeeze(-1),
            "proposal_type_logits": self.proposal_type_head(proposal_hidden),
            "group_type_logits": self.group_type_head(group_hidden),
            "member_count_logits": self.member_count_head(group_hidden),
            "group_count_logits": self.group_count_head(pooled),
            "membership_logits": membership_logits,
            "proposal_features": proposal_hidden,
            "group_features": group_hidden,
        }

    def config_dict(self) -> dict[str, Any]:
        return {"proposal_membership_config": asdict(self.config)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _expanded_proposal_mask(
    proposal_mask: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    if proposal_mask.shape == (shape[1],):
        return proposal_mask.bool().unsqueeze(0).expand(shape[0], -1)
    if proposal_mask.shape == shape:
        return proposal_mask.bool()
    raise ValueError(
        "Proposal mask must have shape [proposals] or [groups, proposals]"
    )


def group_normalized_membership_loss(
    membership_logits: torch.Tensor,
    membership_targets: torch.Tensor,
    proposal_mask: torch.Tensor,
    *,
    group_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return positive, negative and Dice losses with equal group weight."""

    if membership_logits.ndim != 2:
        raise ValueError("Membership logits must have shape [groups, proposals]")
    if membership_targets.shape != membership_logits.shape:
        raise ValueError("Membership targets and logits differ")
    valid = _expanded_proposal_mask(proposal_mask, membership_logits.shape)
    targets = membership_targets.bool()
    if bool((targets & ~valid).any()):
        raise ValueError("Membership targets include padded proposals")
    zero = membership_logits.sum() * 0.0
    if not len(membership_logits):
        return {
            "positive_bce": zero,
            "negative_bce": zero,
            "dice": zero,
        }
    if group_weights is None:
        weights = membership_logits.new_ones(len(membership_logits))
    else:
        if group_weights.shape != (len(membership_logits),):
            raise ValueError("Group weights must contain one value per group")
        weights = group_weights.to(
            device=membership_logits.device,
            dtype=membership_logits.dtype,
        )
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
            raise ValueError("Group weights must be finite and nonnegative")
        if not bool((weights > 0.0).any()):
            raise ValueError("At least one group weight must be positive")

    def weighted_mean(values: torch.Tensor, current_weights: torch.Tensor) -> torch.Tensor:
        return (values * current_weights).sum() / current_weights.sum().clamp_min(eps)

    positive_counts = targets.sum(dim=-1)
    if bool((positive_counts == 0).any()):
        raise ValueError("Every supervised group must contain a proposal")
    negative = valid & ~targets
    negative_counts = negative.sum(dim=-1)
    positive_bce = (
        F.softplus(-membership_logits) * targets.to(membership_logits.dtype)
    ).sum(dim=-1) / positive_counts.to(membership_logits.dtype)
    negative_rows = negative_counts > 0
    if bool(negative_rows.any()):
        negative_values = (
            F.softplus(membership_logits)
            * negative.to(membership_logits.dtype)
        ).sum(dim=-1)[negative_rows] / negative_counts[negative_rows].to(
            membership_logits.dtype
        )
        negative_bce = weighted_mean(negative_values, weights[negative_rows])
    else:
        negative_bce = zero
    probabilities = membership_logits.sigmoid() * valid.to(membership_logits.dtype)
    intersection = (
        probabilities * targets.to(membership_logits.dtype)
    ).sum(dim=-1)
    dice = 1.0 - (2.0 * intersection + eps) / (
        probabilities.sum(dim=-1)
        + positive_counts.to(membership_logits.dtype)
        + eps
    )
    return {
        "positive_bce": weighted_mean(positive_bce, weights),
        "negative_bce": negative_bce,
        "dice": weighted_mean(dice, weights),
    }


def proposal_slot_exclusivity_loss(
    membership_logits: torch.Tensor,
    group_type_logits: torch.Tensor,
    proposal_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize a physical proposal being claimed by multiple active slots."""

    if membership_logits.ndim != 3:
        raise ValueError("Membership logits must have shape [batch, groups, proposals]")
    batch, groups, proposals = membership_logits.shape
    if group_type_logits.shape != (batch, groups, len(PHYSICAL_GROUP_NAMES)):
        raise ValueError("Group type logits and membership slots differ")
    if proposal_mask.shape != (batch, proposals):
        raise ValueError("Proposal mask and membership proposals differ")
    active = 1.0 - group_type_logits.float().softmax(dim=-1)[..., NO_GROUP]
    claims = membership_logits.float().sigmoid() * active.unsqueeze(-1)
    excess = F.relu(claims.sum(dim=1) - 1.0).square()
    valid = proposal_mask.bool()
    return (excess * valid).sum() / valid.sum().clamp_min(1)


def _pairwise_membership_cost(
    membership_logits: torch.Tensor,
    target_membership: torch.Tensor,
    proposal_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return balanced BCE and Dice costs for every slot-target pair."""

    if membership_logits.ndim != 2:
        raise ValueError("Pairwise membership logits must be two-dimensional")
    if target_membership.ndim != 2:
        raise ValueError("Pairwise target membership must be two-dimensional")
    if membership_logits.size(1) != target_membership.size(1):
        raise ValueError("Pairwise membership proposal capacities differ")
    if proposal_mask.shape != (membership_logits.size(1),):
        raise ValueError("Pairwise proposal mask has an invalid shape")
    valid = proposal_mask.bool()
    targets = target_membership.bool()
    if bool((targets & ~valid.unsqueeze(0)).any()):
        raise ValueError("Pairwise target membership includes padding")
    positive_counts = targets.sum(dim=-1)
    if bool((positive_counts == 0).any()):
        raise ValueError("Every target group must contain a proposal")
    negative = valid.unsqueeze(0) & ~targets
    negative_counts = negative.sum(dim=-1)

    positive_terms = F.softplus(-membership_logits)
    positive = positive_terms @ targets.to(positive_terms.dtype).transpose(0, 1)
    positive = positive / positive_counts.to(positive.dtype).unsqueeze(0)
    negative_terms = F.softplus(membership_logits)
    negative_cost = negative_terms @ negative.to(negative_terms.dtype).transpose(0, 1)
    negative_cost = negative_cost / negative_counts.clamp_min(1).to(
        negative_cost.dtype
    ).unsqueeze(0)
    balanced_bce = positive + negative_cost

    probabilities = membership_logits.sigmoid() * valid.to(membership_logits.dtype)
    intersection = probabilities @ targets.to(probabilities.dtype).transpose(0, 1)
    predicted_sum = probabilities.sum(dim=-1, keepdim=True)
    target_sum = positive_counts.to(probabilities.dtype).unsqueeze(0)
    dice = 1.0 - (2.0 * intersection + eps) / (
        predicted_sum + target_sum + eps
    )
    return balanced_bce, dice


def _validate_assignment_targets(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> None:
    required_output = {
        "group_type_logits",
        "member_count_logits",
        "membership_logits",
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
            f"Membership assignment fields are missing: output={missing_output}, "
            f"targets={missing_targets}"
        )
    type_logits = output["group_type_logits"]
    membership = output["membership_logits"]
    if type_logits.ndim != 3 or type_logits.size(-1) != len(PHYSICAL_GROUP_NAMES):
        raise ValueError("Group type logits have an invalid shape")
    batch, groups, _ = type_logits.shape
    proposals = membership.size(-1)
    if membership.shape != (batch, groups, proposals):
        raise ValueError("Membership logits have an invalid shape")
    if proposal_mask.shape != (batch, proposals):
        raise ValueError("Proposal mask has an invalid shape")
    if targets["group_types"].ndim != 2:
        raise ValueError("Group target types must be two-dimensional")
    target_shape = targets["group_types"].shape
    if targets["group_mask"].shape != target_shape:
        raise ValueError("Group mask and group types differ")
    if targets["member_counts"].shape != target_shape:
        raise ValueError("Member counts and group types differ")
    if targets["group_membership"].shape != (*target_shape, proposals):
        raise ValueError("Group membership targets have an invalid shape")
    if target_shape[0] != batch:
        raise ValueError("Output and target batches differ")


def bond_group_proposal_membership_assignments(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    class_cost_weight: float = 1.0,
    membership_bce_cost_weight: float = 2.0,
    membership_dice_cost_weight: float = 2.0,
    member_count_cost_weight: float = 0.5,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Hungarian-match group slots using type, membership and count only."""

    if min(
        class_cost_weight,
        membership_bce_cost_weight,
        membership_dice_cost_weight,
        member_count_cost_weight,
    ) < 0.0:
        raise ValueError("Membership assignment weights must be nonnegative")
    _validate_assignment_targets(output, targets, proposal_mask)
    type_probabilities = output["group_type_logits"].detach().float().softmax(-1)
    count_probabilities = output["member_count_logits"].detach().float().softmax(-1)
    membership_logits = output["membership_logits"].detach().float()
    group_mask = targets["group_mask"].bool()
    assignments = []
    for batch_index in range(type_probabilities.size(0)):
        target_indices = torch.where(group_mask[batch_index])[0]
        if len(target_indices) > type_probabilities.size(1):
            raise ValueError("Group targets exceed configured slot capacity")
        if not len(target_indices):
            empty = torch.empty(
                0,
                device=type_probabilities.device,
                dtype=torch.long,
            )
            assignments.append((empty, empty))
            continue
        target_types = targets["group_types"][batch_index, target_indices].long()
        target_counts = targets["member_counts"][batch_index, target_indices].long()
        if bool(
            ((target_types < SINGLE_LINE) | (target_types > SOLID_QUAD)).any()
        ):
            raise ValueError("Physical group target type is outside 1..6")
        if int(target_counts.max()) >= count_probabilities.size(-1):
            raise ValueError("Target member count exceeds configured capacity")
        target_support = targets["group_membership"][
            batch_index,
            target_indices,
        ].bool().sum(dim=-1)
        for target_type, target_count, support_count in zip(
            target_types.tolist(),
            target_counts.tolist(),
            target_support.tolist(),
        ):
            if target_type == HASHED_WEDGE:
                if target_count < 2 or support_count < 2:
                    raise ValueError(
                        "Hashed-wedge targets require at least two members and "
                        "two supporting proposals"
                    )
                if support_count > target_count:
                    raise ValueError(
                        "Hashed-wedge support cannot exceed its member count"
                    )
            else:
                expected_count = FIXED_MEMBER_COUNTS[target_type]
                if target_count != expected_count or support_count != expected_count:
                    raise ValueError(
                        f"{PHYSICAL_GROUP_NAMES[target_type]} requires exactly "
                        f"{expected_count} member proposal(s)"
                    )
        bce_cost, dice_cost = _pairwise_membership_cost(
            membership_logits[batch_index],
            targets["group_membership"][
                batch_index,
                target_indices,
            ].bool(),
            proposal_mask[batch_index].bool(),
        )
        class_cost = -type_probabilities[batch_index][:, target_types]
        count_cost = -count_probabilities[batch_index][:, target_counts]
        cost = (
            class_cost_weight * class_cost
            + membership_bce_cost_weight * bce_cost
            + membership_dice_cost_weight * dice_cost
            + member_count_cost_weight * count_cost
        )
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(
                    rows,
                    device=type_probabilities.device,
                    dtype=torch.long,
                ),
                target_indices[
                    torch.as_tensor(
                        columns,
                        device=type_probabilities.device,
                        dtype=torch.long,
                    )
                ],
            )
        )
    return assignments


def _balanced_proposal_keep_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = logits.sum() * 0.0
    positive_rows = []
    negative_rows = []
    for batch_index in range(logits.size(0)):
        valid = mask[batch_index].bool()
        positive = valid & targets[batch_index].bool()
        negative = valid & ~targets[batch_index].bool()
        if bool(positive.any()):
            positive_rows.append(F.softplus(-logits[batch_index, positive]).mean())
        if bool(negative.any()):
            negative_rows.append(F.softplus(logits[batch_index, negative]).mean())
    return (
        torch.stack(positive_rows).mean() if positive_rows else zero,
        torch.stack(negative_rows).mean() if negative_rows else zero,
    )


def bond_group_proposal_membership_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    group_weights: torch.Tensor | None = None,
    no_object_weight: float = 0.1,
    proposal_no_object_weight: float = 0.1,
    group_type_weight: float = 1.0,
    member_count_weight: float = 0.5,
    group_count_weight: float = 0.5,
    membership_positive_weight: float = 1.0,
    membership_negative_weight: float = 1.0,
    membership_dice_weight: float = 2.0,
    unmatched_membership_weight: float = 0.1,
    exclusivity_weight: float = 0.25,
    proposal_keep_weight: float = 1.0,
    proposal_type_weight: float = 0.5,
    class_cost_weight: float = 1.0,
    membership_bce_cost_weight: float = 2.0,
    membership_dice_cost_weight: float = 2.0,
    member_count_cost_weight: float = 0.5,
) -> dict[str, Any]:
    """Train group slots without any target or predicted coordinate tensor."""

    if not 0.0 < no_object_weight <= 1.0:
        raise ValueError("Group no-object weight must be in (0, 1]")
    if not 0.0 < proposal_no_object_weight <= 1.0:
        raise ValueError("Proposal no-object weight must be in (0, 1]")
    loss_weights = (
        group_type_weight,
        member_count_weight,
        group_count_weight,
        membership_positive_weight,
        membership_negative_weight,
        membership_dice_weight,
        unmatched_membership_weight,
        exclusivity_weight,
        proposal_keep_weight,
        proposal_type_weight,
    )
    if min(loss_weights) < 0.0:
        raise ValueError("Proposal Membership loss weights must be nonnegative")
    required = {"proposal_keep", "proposal_types"}
    missing = sorted(required - targets.keys())
    if missing:
        raise ValueError(f"Proposal supervision fields are missing: {missing}")
    _validate_assignment_targets(output, targets, proposal_mask)
    batch, groups, proposals = output["membership_logits"].shape
    if output["proposal_keep_logits"].shape != (batch, proposals):
        raise ValueError("Proposal keep logits have an invalid shape")
    if output["proposal_type_logits"].shape != (
        batch,
        proposals,
        len(PHYSICAL_PROPOSAL_TYPES),
    ):
        raise ValueError("Corrected proposal type logits have an invalid shape")
    if targets["proposal_keep"].shape != (batch, proposals):
        raise ValueError("Proposal keep targets have an invalid shape")
    if targets["proposal_types"].shape != (batch, proposals):
        raise ValueError("Proposal type targets have an invalid shape")
    supervision_group_weights = group_weights
    if supervision_group_weights is None:
        supervision_group_weights = targets.get("group_weights")
    if supervision_group_weights is not None:
        if supervision_group_weights.shape != targets["group_mask"].shape:
            raise ValueError("Group weights and group targets differ")
        supervision_group_weights = supervision_group_weights.to(
            device=output["group_type_logits"].device,
            dtype=output["group_type_logits"].dtype,
        )
        active_weights = supervision_group_weights[
            targets["group_mask"].bool()
        ]
        if not bool(torch.isfinite(active_weights).all()) or bool(
            (active_weights <= 0.0).any()
        ):
            raise ValueError("Active group weights must be finite and positive")
    if bool((targets["proposal_keep"].bool() & ~proposal_mask.bool()).any()):
        raise ValueError("Proposal keep targets include padded proposals")
    claimed = (
        targets["group_membership"].bool()
        & targets["group_mask"].bool().unsqueeze(-1)
    ).any(dim=1)
    if bool((claimed & ~targets["proposal_keep"].bool()).any()):
        raise ValueError("A grouped proposal must also have a positive keep target")

    assignments = bond_group_proposal_membership_assignments(
        output,
        targets,
        proposal_mask,
        class_cost_weight=class_cost_weight,
        membership_bce_cost_weight=membership_bce_cost_weight,
        membership_dice_cost_weight=membership_dice_cost_weight,
        member_count_cost_weight=member_count_cost_weight,
    )
    type_targets = torch.full(
        (batch, groups),
        NO_GROUP,
        device=output["group_type_logits"].device,
        dtype=torch.long,
    )
    matched_membership_logits = []
    matched_membership_targets = []
    matched_proposal_masks = []
    matched_count_logits = []
    matched_count_targets = []
    matched_group_weights = []
    unmatched_membership_parts = []
    slot_loss_weights = output["group_type_logits"].new_ones((batch, groups))
    for batch_index, (slot_indices, target_indices) in enumerate(assignments):
        matched_slot_mask = torch.zeros(
            groups,
            device=output["membership_logits"].device,
            dtype=torch.bool,
        )
        if len(slot_indices):
            matched_slot_mask[slot_indices] = True
            type_targets[batch_index, slot_indices] = targets["group_types"][
                batch_index,
                target_indices,
            ].long()
            matched_membership_logits.append(
                output["membership_logits"][batch_index, slot_indices]
            )
            matched_membership_targets.append(
                targets["group_membership"][batch_index, target_indices]
            )
            matched_proposal_masks.append(
                proposal_mask[batch_index].bool().unsqueeze(0).expand(
                    len(slot_indices),
                    -1,
                )
            )
            matched_count_logits.append(
                output["member_count_logits"][batch_index, slot_indices]
            )
            matched_count_targets.append(
                targets["member_counts"][batch_index, target_indices]
            )
            current_weights = (
                torch.ones_like(target_indices, dtype=slot_loss_weights.dtype)
                if supervision_group_weights is None
                else supervision_group_weights[batch_index, target_indices]
            )
            matched_group_weights.append(current_weights)
            slot_loss_weights[batch_index, slot_indices] = current_weights
        unmatched_slots = torch.where(~matched_slot_mask)[0]
        valid_proposals = proposal_mask[batch_index].bool()
        if len(unmatched_slots) and bool(valid_proposals.any()):
            unmatched_membership_parts.append(
                F.softplus(
                    output["membership_logits"][batch_index][
                        unmatched_slots
                    ][:, valid_proposals]
                ).mean()
            )

    group_class_weights = output["group_type_logits"].new_ones(
        len(PHYSICAL_GROUP_NAMES)
    )
    group_class_weights[NO_GROUP] = no_object_weight
    group_type_values = F.cross_entropy(
        output["group_type_logits"].transpose(1, 2),
        type_targets,
        weight=group_class_weights,
        reduction="none",
    )
    group_type_loss = (
        group_type_values * slot_loss_weights
    ).sum() / slot_loss_weights.sum().clamp_min(1.0)
    zero = group_type_loss * 0.0
    if matched_membership_logits:
        membership_losses = group_normalized_membership_loss(
            torch.cat(matched_membership_logits),
            torch.cat(matched_membership_targets),
            torch.cat(matched_proposal_masks),
            group_weights=torch.cat(matched_group_weights),
        )
        member_count_values = F.cross_entropy(
            torch.cat(matched_count_logits).float(),
            torch.cat(matched_count_targets).long(),
            reduction="none",
        )
        combined_group_weights = torch.cat(matched_group_weights).float()
        member_count_loss = (
            member_count_values * combined_group_weights
        ).sum() / combined_group_weights.sum().clamp_min(1e-6)
    else:
        membership_losses = {
            "positive_bce": zero,
            "negative_bce": zero,
            "dice": zero,
        }
        member_count_loss = zero
    unmatched_membership_loss = (
        torch.stack(unmatched_membership_parts).mean()
        if unmatched_membership_parts
        else zero
    )

    expected_group_counts = targets["group_mask"].sum(dim=-1).long()
    expected_count_shape = (batch, groups + 1)
    if output["group_count_logits"].shape != expected_count_shape:
        raise ValueError(
            f"Group count logits must have shape {expected_count_shape}"
        )
    group_count_loss = F.cross_entropy(
        output["group_count_logits"].float(),
        expected_group_counts,
    )
    proposal_keep_positive_loss, proposal_keep_negative_loss = (
        _balanced_proposal_keep_loss(
            output["proposal_keep_logits"],
            targets["proposal_keep"].bool(),
            proposal_mask.bool(),
        )
    )
    valid_proposals = proposal_mask.bool()
    proposal_type_targets = targets["proposal_types"].long()
    if bool(
        (
            (proposal_type_targets[valid_proposals] < NO_PROPOSAL)
            | (proposal_type_targets[valid_proposals] > PROPOSAL_QUAD)
        ).any()
    ):
        raise ValueError("Proposal target type is outside 0..3")
    if bool(
        (
            targets["proposal_keep"].bool()
            & valid_proposals
            & (proposal_type_targets == NO_PROPOSAL)
        ).any()
    ):
        raise ValueError("Kept proposals require a physical primitive type")
    proposal_weights = output["proposal_type_logits"].new_ones(
        len(PHYSICAL_PROPOSAL_TYPES)
    )
    proposal_weights[NO_PROPOSAL] = proposal_no_object_weight
    proposal_type_loss = F.cross_entropy(
        output["proposal_type_logits"][valid_proposals].float(),
        proposal_type_targets[valid_proposals],
        weight=proposal_weights,
    )
    exclusivity_loss = proposal_slot_exclusivity_loss(
        output["membership_logits"],
        output["group_type_logits"],
        proposal_mask,
    )
    proposal_keep_loss = 0.5 * (
        proposal_keep_positive_loss + proposal_keep_negative_loss
    )
    total = (
        group_type_weight * group_type_loss
        + member_count_weight * member_count_loss
        + group_count_weight * group_count_loss
        + membership_positive_weight * membership_losses["positive_bce"]
        + membership_negative_weight * membership_losses["negative_bce"]
        + membership_dice_weight * membership_losses["dice"]
        + unmatched_membership_weight * unmatched_membership_loss
        + exclusivity_weight * exclusivity_loss
        + proposal_keep_weight * proposal_keep_loss
        + proposal_type_weight * proposal_type_loss
    )
    return {
        "loss": total,
        "group_type_loss": group_type_loss,
        "member_count_loss": member_count_loss,
        "group_count_loss": group_count_loss,
        "membership_positive_loss": membership_losses["positive_bce"],
        "membership_negative_loss": membership_losses["negative_bce"],
        "membership_dice_loss": membership_losses["dice"],
        "unmatched_membership_loss": unmatched_membership_loss,
        "exclusivity_loss": exclusivity_loss,
        "proposal_keep_loss": proposal_keep_loss,
        "proposal_keep_positive_loss": proposal_keep_positive_loss,
        "proposal_keep_negative_loss": proposal_keep_negative_loss,
        "proposal_type_loss": proposal_type_loss,
        "predicted_group_count_mean": (
            1.0
            - output["group_type_logits"].float().softmax(dim=-1)[..., NO_GROUP]
        ).sum(dim=-1).mean(),
        "target_group_count_mean": expected_group_counts.float().mean(),
        "supervised_group_weight_mean": (
            torch.cat(matched_group_weights).float().mean()
            if matched_group_weights
            else zero
        ),
        "assignments": assignments,
    }


def _required_proposal_type(group_type: int) -> int:
    if group_type in LINE_GROUP_TYPES:
        return PROPOSAL_LINE
    if group_type == SOLID_TRIANGLE:
        return PROPOSAL_TRIANGLE
    if group_type == SOLID_QUAD:
        return PROPOSAL_QUAD
    raise ValueError(f"Unsupported physical group type: {group_type}")


@torch.no_grad()
def decode_bond_group_proposal_membership(
    output: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
    *,
    mode: str = "learned_count",
    known_counts: torch.Tensor | None = None,
    membership_threshold: float = 0.5,
    proposal_keep_threshold: float = 0.5,
) -> list[list[dict[str, Any]]]:
    """Decode exclusive proposal memberships without producing coordinates."""

    if mode not in {"known_count", "learned_count"}:
        raise ValueError("Decode mode must be known_count or learned_count")
    if not 0.0 <= membership_threshold <= 1.0:
        raise ValueError("Membership threshold must be in [0, 1]")
    if not 0.0 <= proposal_keep_threshold <= 1.0:
        raise ValueError("Proposal keep threshold must be in [0, 1]")
    group_type_logits = output["group_type_logits"]
    membership_logits = output["membership_logits"]
    batch, groups, proposals = membership_logits.shape
    if group_type_logits.shape != (batch, groups, len(PHYSICAL_GROUP_NAMES)):
        raise ValueError("Decode group type logits have an invalid shape")
    if output["member_count_logits"].shape[:2] != (batch, groups):
        raise ValueError("Decode member count logits have an invalid shape")
    if output["member_count_logits"].size(-1) < 4:
        raise ValueError("Decode member count logits must support count three")
    if output["proposal_keep_logits"].shape != (batch, proposals):
        raise ValueError("Decode proposal keep logits have an invalid shape")
    if output["proposal_type_logits"].shape != (
        batch,
        proposals,
        len(PHYSICAL_PROPOSAL_TYPES),
    ):
        raise ValueError("Decode proposal type logits have an invalid shape")
    if proposal_mask.shape != (batch, proposals):
        raise ValueError("Decode proposal mask has an invalid shape")
    if mode == "known_count":
        if known_counts is None or known_counts.shape != (batch,):
            raise ValueError("known_count decoding requires one count per image")
        group_counts = known_counts.long()
    else:
        expected_shape = (batch, groups + 1)
        if output["group_count_logits"].shape != expected_shape:
            raise ValueError(
                f"Learned count logits must have shape {expected_shape}"
            )
        group_counts = output["group_count_logits"].float().argmax(dim=-1)
    if bool(((group_counts < 0) | (group_counts > groups)).any()):
        raise ValueError("Decoded group count is outside slot capacity")

    group_probabilities = group_type_logits.float().softmax(dim=-1)
    group_scores, group_types = group_probabilities[..., 1:].max(dim=-1)
    group_types = group_types + 1
    no_group_scores = group_probabilities[..., NO_GROUP]
    proposal_keep = output["proposal_keep_logits"].float().sigmoid()
    proposal_types = output["proposal_type_logits"].float().argmax(dim=-1)
    membership = membership_logits.float().sigmoid()
    member_counts = output["member_count_logits"].float()
    decoded = []
    for batch_index in range(batch):
        count = int(group_counts[batch_index])
        ranking = group_scores[batch_index] - no_group_scores[batch_index]
        selected_slots = torch.argsort(
            ranking,
            descending=True,
            stable=True,
        )[:count].tolist()
        slot_capacity: dict[int, int] = {}
        slot_member_count: dict[int, int] = {}
        for slot in selected_slots:
            group_type = int(group_types[batch_index, slot])
            if group_type == HASHED_WEDGE:
                predicted_count = int(
                    member_counts[batch_index, slot, 2:].argmax()
                ) + 2
            else:
                predicted_count = FIXED_MEMBER_COUNTS[group_type]
            slot_capacity[slot] = predicted_count
            slot_member_count[slot] = predicted_count

        candidates = []
        valid_mask = proposal_mask[batch_index].bool()
        for slot in selected_slots:
            required_type = _required_proposal_type(
                int(group_types[batch_index, slot])
            )
            for proposal in torch.where(valid_mask)[0].tolist():
                if int(proposal_types[batch_index, proposal]) != required_type:
                    continue
                membership_score = float(membership[batch_index, slot, proposal])
                keep_score = float(proposal_keep[batch_index, proposal])
                if membership_score < membership_threshold:
                    continue
                if keep_score < proposal_keep_threshold:
                    continue
                candidates.append(
                    (
                        membership_score * keep_score,
                        membership_score,
                        keep_score,
                        slot,
                        proposal,
                    )
                )
        candidates.sort(key=lambda row: (-row[0], row[3], row[4]))
        used_proposals: set[int] = set()
        selected_members: dict[int, list[tuple[int, float]]] = {
            slot: [] for slot in selected_slots
        }
        for _, membership_score, _, slot, proposal in candidates:
            if proposal in used_proposals:
                continue
            if len(selected_members[slot]) >= slot_capacity[slot]:
                continue
            selected_members[slot].append((proposal, membership_score))
            used_proposals.add(proposal)

        rows = []
        for slot in selected_slots:
            group_type = int(group_types[batch_index, slot])
            selected = sorted(
                selected_members[slot],
                key=lambda value: (-value[1], value[0]),
            )
            proposal_indices = [proposal for proposal, _ in selected]
            support_count = len(proposal_indices)
            required = 2 if group_type == HASHED_WEDGE else FIXED_MEMBER_COUNTS[
                group_type
            ]
            valid = (
                support_count >= required
                if group_type == HASHED_WEDGE
                else support_count == required
            )
            rows.append(
                {
                    "query_index": slot,
                    "score": float(group_scores[batch_index, slot]),
                    "type": group_type,
                    "kind": PHYSICAL_GROUP_NAMES[group_type],
                    "member_count": slot_member_count[slot],
                    "support_count": support_count,
                    "proposal_indices": proposal_indices,
                    "membership_scores": [score for _, score in selected],
                    "valid_membership": valid,
                }
            )
        decoded.append(rows)
    return decoded


__all__ = [
    "FIXED_MEMBER_COUNTS",
    "LINE_GROUP_TYPES",
    "PHYSICAL_GROUP_NAMES",
    "PHYSICAL_GROUP_TYPES",
    "PHYSICAL_PROPOSAL_TYPES",
    "BondGroupProposalMembershipConfig",
    "BondGroupProposalMembershipPredictor",
    "ProposalDescriptorEncoder",
    "bond_group_proposal_membership_assignments",
    "bond_group_proposal_membership_loss",
    "decode_bond_group_proposal_membership",
    "group_normalized_membership_loss",
    "proposal_slot_exclusivity_loss",
]
