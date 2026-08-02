"""Permutation-invariant count calibration over frozen grounded graph evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class GroundedCountConfig:
    maximum_candidates: int = 128
    hidden_dim: int = 128
    dropout: float = 0.0
    role_count: int = 0

    def __post_init__(self) -> None:
        if min(self.maximum_candidates, self.hidden_dim) < 1:
            raise ValueError("Grounded count dimensions must be positive")
        if self.role_count < 0:
            raise ValueError("Grounded count role count must be nonnegative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Grounded count dropout must be in [0, 1)")


def grounded_count_node_features(
    node_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    edge_logits: torch.Tensor,
    pair_indices: torch.Tensor,
    candidate_roles: torch.Tensor | None = None,
    role_count: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_logits.shape != candidate_mask.shape or node_logits.ndim != 2:
        raise ValueError("Node logits and candidate mask must have shape [batch, nodes]")
    if edge_logits.ndim != 2 or edge_logits.size(0) != node_logits.size(0):
        raise ValueError("Edge logits must have shape [batch, pairs]")
    if pair_indices.shape != (edge_logits.size(1), 2):
        raise ValueError("Pair indices do not match edge logits")
    if pair_indices.numel() and int(pair_indices.max()) >= node_logits.size(1):
        raise ValueError("Pair index is outside the node dimension")
    if role_count < 0:
        raise ValueError("Grounded count role count must be nonnegative")
    if role_count:
        if candidate_roles is None or candidate_roles.shape != candidate_mask.shape:
            raise ValueError("Grounded count candidate roles have wrong shape")
        if candidate_roles.numel() and (
            int(candidate_roles.min()) < 0
            or int(candidate_roles.max()) >= role_count
        ):
            raise ValueError("Grounded count candidate role is outside configured range")

    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    pair_mask = candidate_mask[:, begin] & candidate_mask[:, end]
    probabilities = edge_logits.float().sigmoid() * pair_mask
    incident = torch.zeros(
        (node_logits.size(0), node_logits.size(1), node_logits.size(1)),
        device=node_logits.device,
        dtype=torch.float32,
    )
    incident[:, begin, end] = probabilities
    incident[:, end, begin] = probabilities
    strongest = incident.topk(min(2, node_logits.size(1)), dim=-1).values
    if strongest.size(-1) == 1:
        strongest = torch.cat((strongest, torch.zeros_like(strongest)), dim=-1)
    degree = incident.sum(dim=-1)
    node_probability = node_logits.float().sigmoid()
    node_features = torch.stack(
        (
            node_logits.float().clamp(-12.0, 12.0) / 12.0,
            node_probability,
            (degree / 8.0).clamp_max(4.0) / 4.0,
            strongest[..., 0],
            strongest[..., 1],
            candidate_mask.float(),
        ),
        dim=-1,
    )
    edge_sum = probabilities.sum(dim=1)
    global_features = torch.stack(
        (
            candidate_mask.sum(dim=1) / node_logits.size(1),
            (node_probability * candidate_mask).sum(dim=1) / node_logits.size(1),
            (degree * candidate_mask).sum(dim=1) / (2.0 * node_logits.size(1)),
            edge_sum / node_logits.size(1),
            (probabilities >= 0.5).sum(dim=1) / node_logits.size(1),
            (probabilities >= 0.8).sum(dim=1) / node_logits.size(1),
        ),
        dim=-1,
    )
    if role_count:
        role_features = F.one_hot(
            candidate_roles.long(), num_classes=role_count
        ).float() * candidate_mask.unsqueeze(-1)
        role_globals = role_features.sum(dim=1) / node_logits.size(1)
        node_features = torch.cat((node_features, role_features), dim=-1)
        global_features = torch.cat((global_features, role_globals), dim=-1)
    return node_features, global_features


class GroundedCountCalibrator(nn.Module):
    """Predict only cardinality; frozen node logits continue to determine ranking."""

    def __init__(self, config: GroundedCountConfig) -> None:
        super().__init__()
        self.config = config
        node_feature_dim = 6 + config.role_count
        input_dim = (
            config.maximum_candidates * node_feature_dim + 6 + config.role_count
        )
        self.count_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.maximum_candidates + 1),
        )

    def forward(
        self,
        node_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_roles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_logits.size(1) != self.config.maximum_candidates:
            raise ValueError("Node count differs from grounded count config")
        node_features, global_features = grounded_count_node_features(
            node_logits,
            candidate_mask,
            edge_logits,
            pair_indices,
            candidate_roles,
            self.config.role_count,
        )
        order = node_logits.float().masked_fill(~candidate_mask, -30.0).argsort(
            dim=1, descending=True
        )
        sorted_features = node_features.gather(
            1, order.unsqueeze(-1).expand_as(node_features)
        )
        return self.count_head(
            torch.cat((sorted_features.flatten(1), global_features), dim=-1)
        )


class GroundedRoleCountCalibrator(nn.Module):
    """Predict one cardinality for each candidate role."""

    def __init__(self, config: GroundedCountConfig) -> None:
        super().__init__()
        if config.role_count < 1:
            raise ValueError("Role count calibrator requires configured roles")
        self.config = config
        node_feature_dim = 6 + config.role_count
        input_dim = (
            config.maximum_candidates * node_feature_dim + 6 + config.role_count
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.count_head = nn.Linear(
            config.hidden_dim,
            config.role_count * (config.maximum_candidates + 1),
        )

    def forward(
        self,
        node_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_roles: torch.Tensor,
    ) -> torch.Tensor:
        node_features, global_features = grounded_count_node_features(
            node_logits,
            candidate_mask,
            edge_logits,
            pair_indices,
            candidate_roles,
            self.config.role_count,
        )
        order = node_logits.float().masked_fill(~candidate_mask, -30.0).argsort(
            dim=1, descending=True
        )
        sorted_features = node_features.gather(
            1, order.unsqueeze(-1).expand_as(node_features)
        )
        hidden = self.encoder(
            torch.cat((sorted_features.flatten(1), global_features), dim=-1)
        )
        return self.count_head(hidden).reshape(
            node_logits.size(0),
            self.config.role_count,
            self.config.maximum_candidates + 1,
        )


def grounded_count_loss(count_logits: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
    if count_logits.ndim != 2 or target_counts.ndim != 1:
        raise ValueError("Count logits and targets must be batched")
    if count_logits.size(0) != target_counts.size(0):
        raise ValueError("Count batch sizes differ")
    if len(target_counts) and int(target_counts.max()) >= count_logits.size(1):
        raise ValueError("Target count exceeds calibrator capacity")
    return F.cross_entropy(count_logits.float(), target_counts.long())


@torch.no_grad()
def predict_grounded_counts(count_logits: torch.Tensor) -> torch.Tensor:
    return count_logits.argmax(dim=-1)


def grounded_role_count_loss(
    count_logits: torch.Tensor, target_counts: torch.Tensor
) -> torch.Tensor:
    if count_logits.ndim != 3 or target_counts.ndim != 2:
        raise ValueError("Role count logits and targets must be batched")
    if count_logits.shape[:2] != target_counts.shape:
        raise ValueError("Role count batch or role dimensions differ")
    if target_counts.numel() and int(target_counts.max()) >= count_logits.size(-1):
        raise ValueError("Role target count exceeds calibrator capacity")
    return F.cross_entropy(
        count_logits.float().flatten(0, 1), target_counts.long().flatten()
    )
