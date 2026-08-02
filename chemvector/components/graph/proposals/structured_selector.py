"""Whole-set candidate selection over frozen dense, query, and edge evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class StructuredSelectorConfig:
    candidate_dim: int = 192
    auxiliary_dim: int = 30
    visual_dim: int = 0
    d_model: int = 192
    heads: int = 6
    ffn_dim: int = 768
    layers: int = 2
    maximum_candidates: int = 256
    incident_topk: int = 4
    dropout: float = 0.0
    residual_scale: float = 12.0
    role_count: int = 0

    def __post_init__(self) -> None:
        if min(
            self.candidate_dim,
            self.auxiliary_dim,
            self.d_model,
            self.heads,
            self.ffn_dim,
            self.layers,
            self.maximum_candidates,
            self.incident_topk,
        ) < 1:
            raise ValueError("Structured selector dimensions must be positive")
        if self.visual_dim < 0:
            raise ValueError("Structured selector visual dimension must be nonnegative")
        if self.role_count < 0:
            raise ValueError("Structured selector role count must be nonnegative")
        if self.d_model % self.heads:
            raise ValueError("Structured selector heads must divide d_model")
        if self.incident_topk >= self.maximum_candidates:
            raise ValueError("Incident top-k must be smaller than maximum candidates")
        if not 0.0 <= self.dropout < 1.0 or self.residual_scale <= 0:
            raise ValueError("Structured selector dropout or residual scale is invalid")


def structured_incident_features(
    edge_logits: torch.Tensor,
    pair_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    incident_topk: int,
) -> torch.Tensor:
    if edge_logits.ndim != 2 or candidate_mask.ndim != 2:
        raise ValueError("Structured edge logits and candidate mask must be batched")
    if pair_indices.shape != (edge_logits.size(1), 2):
        raise ValueError("Pair indices do not match structured edge logits")
    if pair_indices.numel() and int(pair_indices.max()) >= candidate_mask.size(1):
        raise ValueError("Pair index exceeds candidate count")
    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    pair_mask = candidate_mask[:, begin] & candidate_mask[:, end]
    probabilities = edge_logits.float().sigmoid() * pair_mask
    incident = torch.zeros(
        (candidate_mask.size(0), candidate_mask.size(1), candidate_mask.size(1)),
        device=edge_logits.device,
        dtype=torch.float32,
    )
    incident[:, begin, end] = probabilities
    incident[:, end, begin] = probabilities
    strongest = incident.topk(incident_topk, dim=-1).values
    degree = incident.sum(dim=-1, keepdim=True)
    confident_degree = (incident >= 0.8).sum(dim=-1, keepdim=True).float()
    return torch.cat(
        (
            strongest,
            (degree / 16.0).clamp_max(4.0) / 4.0,
            (confident_degree / 8.0).clamp_max(4.0) / 4.0,
        ),
        dim=-1,
    )


class GroundedStructuredSelector(nn.Module):
    """Globally rank a high-recall candidate set without changing coordinates."""

    def __init__(self, config: StructuredSelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, config.d_model),
        )
        scalar_dim = config.incident_topk + 4
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.position_projection = nn.Linear(2, config.d_model)
        self.auxiliary_projection = nn.Sequential(
            nn.LayerNorm(config.auxiliary_dim),
            nn.Linear(config.auxiliary_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        if config.visual_dim:
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(config.visual_dim),
                nn.Linear(config.visual_dim, config.d_model),
            )
            nn.init.zeros_(self.visual_projection[-1].weight)
            nn.init.zeros_(self.visual_projection[-1].bias)
        else:
            self.visual_projection = None
        self.role_embedding = (
            nn.Embedding(config.role_count, config.d_model)
            if config.role_count
            else None
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
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.residual_score = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.residual_score.weight)
        nn.init.zeros_(self.residual_score.bias)

    def encode_context(
        self,
        candidate_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        points: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_auxiliary: torch.Tensor,
        candidate_visual: torch.Tensor | None = None,
        candidate_roles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_hidden.shape[:2] != candidate_mask.shape:
            raise ValueError("Structured candidate hidden and mask shapes differ")
        if candidate_hidden.size(1) != self.config.maximum_candidates:
            raise ValueError("Structured candidate count differs from config")
        if base_logits.shape != candidate_mask.shape:
            raise ValueError("Structured base logits must match candidate mask")
        if points.shape != (*candidate_mask.shape, 2):
            raise ValueError("Structured points must have shape [batch, candidates, 2]")
        if candidate_auxiliary.shape != (
            *candidate_mask.shape,
            self.config.auxiliary_dim,
        ):
            raise ValueError("Structured candidate auxiliary features have wrong shape")
        if self.visual_projection is not None:
            if candidate_visual is None or candidate_visual.shape != (
                *candidate_mask.shape,
                self.config.visual_dim,
            ):
                raise ValueError("Structured candidate visual features have wrong shape")
        elif candidate_visual is not None and candidate_visual.shape[:2] != candidate_mask.shape:
            raise ValueError("Structured candidate visual features have wrong shape")
        if self.role_embedding is not None:
            if candidate_roles is None or candidate_roles.shape != candidate_mask.shape:
                raise ValueError("Structured candidate roles have wrong shape")
            if candidate_roles.numel() and (
                int(candidate_roles.min()) < 0
                or int(candidate_roles.max()) >= self.config.role_count
            ):
                raise ValueError("Structured candidate role is outside configured range")
        elif candidate_roles is not None and candidate_roles.shape != candidate_mask.shape:
            raise ValueError("Structured candidate roles have wrong shape")
        incident = structured_incident_features(
            edge_logits,
            pair_indices,
            candidate_mask,
            incident_topk=self.config.incident_topk,
        )
        node_probability = base_logits.float().sigmoid()
        scalar = torch.cat(
            (
                base_logits.float().clamp(-12.0, 12.0).unsqueeze(-1) / 12.0,
                node_probability.unsqueeze(-1),
                incident,
            ),
            dim=-1,
        )
        hidden = (
            self.candidate_projection(candidate_hidden.float())
            + self.scalar_projection(scalar)
            + self.position_projection(points.float())
            + self.auxiliary_projection(candidate_auxiliary.float())
        )
        if self.visual_projection is not None:
            hidden = hidden + self.visual_projection(candidate_visual.float())
        if self.role_embedding is not None:
            hidden = hidden + self.role_embedding(candidate_roles.long())
        return self.context(hidden, src_key_padding_mask=~candidate_mask)

    def score_context(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[:2] != candidate_mask.shape:
            raise ValueError("Structured context and mask shapes differ")
        if base_logits.shape != candidate_mask.shape:
            raise ValueError("Structured base logits and mask shapes differ")
        raw_residual = self.residual_score(hidden).squeeze(-1)
        residual = self.config.residual_scale * torch.tanh(
            raw_residual / self.config.residual_scale
        )
        return (base_logits.float() + residual).masked_fill(~candidate_mask, -30.0)

    def forward(
        self,
        candidate_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        points: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_logits: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_auxiliary: torch.Tensor,
        candidate_visual: torch.Tensor | None = None,
        candidate_roles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode_context(
            candidate_hidden,
            base_logits,
            points,
            candidate_mask,
            edge_logits,
            pair_indices,
            candidate_auxiliary,
            candidate_visual,
            candidate_roles,
        )
        return self.score_context(hidden, base_logits, candidate_mask)


def structured_selector_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    whole_set_weight: float = 10.0,
    whole_set_margin: float = 1.0,
    boundary_size: int = 16,
    extreme_set_weight: float = 0.0,
    extreme_set_margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    if logits.shape != targets.shape or logits.shape != candidate_mask.shape:
        raise ValueError("Structured selector tensors must have matching shapes")
    if boundary_size < 1:
        raise ValueError("Structured selector boundary size must be positive")
    if extreme_set_weight < 0.0:
        raise ValueError("Structured selector extreme-set weight must be nonnegative")
    valid_logits = logits[candidate_mask]
    valid_targets = targets[candidate_mask]
    positives = valid_targets.sum().clamp_min(1.0)
    negatives = (valid_targets.numel() - positives).clamp_min(1.0)
    selection_loss = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=(negatives / positives).clamp(1.0, 32.0),
    )
    whole_set_losses = []
    extreme_set_losses = []
    for scores, labels, valid in zip(logits, targets.bool(), candidate_mask):
        positive_scores = scores[valid & labels]
        negative_scores = scores[valid & ~labels]
        if not len(positive_scores) or not len(negative_scores):
            continue
        hard_negatives = negative_scores.topk(
            min(boundary_size, len(negative_scores))
        ).values
        hard_positives = -(-positive_scores).topk(
            min(boundary_size, len(positive_scores))
        ).values
        whole_set_losses.append(
            F.softplus(
                whole_set_margin
                + hard_negatives[:, None]
                - hard_positives[None, :]
            ).mean()
        )
        extreme_set_losses.append(
            F.softplus(
                extreme_set_margin
                + negative_scores.max()
                - positive_scores.min()
            )
        )
    whole_set_loss = (
        torch.stack(whole_set_losses).mean()
        if whole_set_losses
        else selection_loss * 0.0
    )
    extreme_set_loss = (
        torch.stack(extreme_set_losses).mean()
        if extreme_set_losses
        else selection_loss * 0.0
    )
    return {
        "loss": (
            selection_loss
            + whole_set_weight * whole_set_loss
            + extreme_set_weight * extreme_set_loss
        ),
        "selection_loss": selection_loss,
        "whole_set_loss": whole_set_loss,
        "extreme_set_loss": extreme_set_loss,
    }
