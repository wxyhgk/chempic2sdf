"""Endpoint-incidence residual scorer for Evidence Carrier v0.9.18."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
)


ENDPOINT_RESIDUAL_PARAMETER_COUNT = 6_829
ENDPOINT_HIDDEN_DIM = 12
ENDPOINT_CONTEXT_DIM = 25
ENDPOINT_SYMMETRIC_CONTEXT_DIM = 50
POSITIVE_MARGIN = 0.25
NEGATIVE_SAFETY_MARGIN = 0.25
RANKING_MARGIN = 0.50
NEGATIVE_LOSS_WEIGHT = 2.0
WORST_NEGATIVE_LOSS_WEIGHT = 8.0
NEGATIVE_UPWARD_LOSS_WEIGHT = 2.0
RANKING_LOSS_WEIGHT = 0.25
RESIDUAL_L2_WEIGHT = 0.01


def _validate_inputs(
    pair_tokens: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor,
    global_count_features: torch.Tensor,
) -> tuple[int, int]:
    if (
        pair_tokens.ndim != 3
        or pair_tokens.size(-1) != SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
        or not pair_tokens.is_floating_point()
    ):
        raise ValueError("pair_tokens must have shape [batch,pairs,21]")
    batch, pairs = pair_tokens.shape[:2]
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices must have long shape [batch,pairs,2]")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [batch,pairs]")
    if (
        global_count_features.shape
        != (batch, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM)
        or not global_count_features.is_floating_point()
    ):
        raise ValueError("global_count_features must have shape [batch,4]")
    if len(
        {
            pair_tokens.device,
            pair_indices.device,
            pair_mask.device,
            global_count_features.device,
        }
    ) != 1:
        raise ValueError("endpoint residual inputs must share one device")
    if bool(pair_mask.any()):
        active_tokens = pair_tokens[pair_mask]
        active_indices = pair_indices[pair_mask]
        if not bool(torch.isfinite(active_tokens).all()):
            raise ValueError("active pair tokens contain non-finite values")
        if bool((active_indices < 0).any()) or bool(
            (active_indices[:, 0] == active_indices[:, 1]).any()
        ):
            raise ValueError("active pairs require two distinct nonnegative endpoints")
    if bool((pair_indices[~pair_mask] != -1).any()):
        raise ValueError("inactive pair indices must be padded with -1")
    if not bool(torch.isfinite(global_count_features).all()) or bool(
        (global_count_features < 0.0).any()
    ):
        raise ValueError("global count features must be finite and nonnegative")
    return batch, pairs


def endpoint_incidence_context(
    encoded_pairs: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a symmetric 50D context for the two endpoints of every pair."""

    if encoded_pairs.ndim != 3 or encoded_pairs.size(-1) != ENDPOINT_HIDDEN_DIM:
        raise ValueError("encoded_pairs must have shape [batch,pairs,12]")
    batch, pairs, hidden = encoded_pairs.shape
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices shape or dtype differs")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask shape or dtype differs")
    if len({encoded_pairs.device, pair_indices.device, pair_mask.device}) != 1:
        raise ValueError("endpoint context inputs must share one device")
    if not pairs or not bool(pair_mask.any()):
        return encoded_pairs.new_zeros(
            (batch, pairs, ENDPOINT_SYMMETRIC_CONTEXT_DIM)
        )

    active_indices = pair_indices[pair_mask]
    if bool((active_indices < 0).any()) or bool(
        (active_indices[:, 0] == active_indices[:, 1]).any()
    ):
        raise ValueError("active endpoint indices are invalid")
    anchor_capacity = int(active_indices.max()) + 1
    offsets = (
        torch.arange(batch, device=pair_indices.device, dtype=torch.long)
        * anchor_capacity
    )
    endpoint_ids = pair_indices.clamp_min(0) + offsets[:, None, None]
    endpoint_active = pair_mask[..., None].expand(-1, -1, 2)
    repeated_encoded = encoded_pairs[:, :, None, :].expand(-1, -1, 2, -1)
    flat_ids = endpoint_ids[endpoint_active]
    flat_values = repeated_encoded[endpoint_active]

    slot_count = batch * anchor_capacity
    sums = encoded_pairs.new_zeros((slot_count, hidden))
    sums.index_add_(0, flat_ids, flat_values)
    counts = encoded_pairs.new_zeros((slot_count,))
    counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=encoded_pairs.dtype))
    maximum = torch.full_like(sums, -torch.inf)
    maximum.scatter_reduce_(
        0,
        flat_ids[:, None].expand(-1, hidden),
        flat_values,
        reduce="amax",
        include_self=True,
    )
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    gathered_sum = sums[endpoint_ids]
    gathered_count = counts[endpoint_ids]
    gathered_max = maximum[endpoint_ids]
    current = encoded_pairs[:, :, None, :]
    other_count = (gathered_count - 1.0).clamp_min(0.0)
    other_mean = (gathered_sum - current) / other_count.clamp_min(1.0)[..., None]
    other_mean = torch.where(
        (other_count > 0.0)[..., None], other_mean, torch.zeros_like(other_mean)
    )
    endpoint_features = torch.cat(
        (other_mean, gathered_max, torch.log1p(other_count)[..., None]), dim=-1
    )
    first = endpoint_features[:, :, 0]
    second = endpoint_features[:, :, 1]
    symmetric = torch.cat(((first + second) * 0.5, (first - second).abs()), dim=-1)
    return torch.where(pair_mask[..., None], symmetric, torch.zeros_like(symmetric))


class EvidenceCarrierEndpointIncidenceResidual(nn.Module):
    """Predict pair residuals with endpoint-local and image-global context."""

    model_type = "evidence-carrier-endpoint-incidence-residual-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        if (
            isinstance(tau0, bool)
            or not isinstance(tau0, (int, float))
            or not math.isfinite(float(tau0))
        ):
            raise ValueError("tau0 must be finite")
        self.pair_encoder = nn.Sequential(
            nn.Linear(SAMPLE_THRESHOLD_PAIR_TOKEN_DIM, 24),
            nn.GELU(),
            nn.Linear(24, ENDPOINT_HIDDEN_DIM),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(99, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("endpoint residual final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != ENDPOINT_RESIDUAL_PARAMETER_COUNT:
            raise RuntimeError("endpoint residual parameter contract changed")

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, pairs = _validate_inputs(
            pair_tokens, pair_indices, pair_mask, global_count_features
        )
        encoded = self.pair_encoder(pair_tokens)
        active = pair_mask[..., None]
        encoded_sum = torch.where(active, encoded, torch.zeros_like(encoded)).sum(dim=1)
        counts = pair_mask.sum(dim=1)
        pooled_mean = encoded_sum / counts.clamp_min(1).to(encoded.dtype).unsqueeze(-1)
        empty_pool = encoded.new_zeros((batch, encoded.size(-1)))
        if pairs:
            pooled_max = torch.where(
                active, encoded, torch.full_like(encoded, -torch.inf)
            ).amax(dim=1)
            pooled_max = torch.where(
                (counts > 0).unsqueeze(-1), pooled_max, empty_pool
            )
        else:
            pooled_max = empty_pool
        image_context = torch.cat(
            (pooled_mean, pooled_max, global_count_features), dim=-1
        )
        endpoint_context = endpoint_incidence_context(encoded, pair_indices, pair_mask)
        expanded_global = image_context[:, None, :].expand(-1, pairs, -1)
        residual_input = torch.cat(
            (pair_tokens, expanded_global, endpoint_context), dim=-1
        )
        residual = self.residual_head(residual_input).squeeze(-1)
        residual = torch.where(pair_mask, residual, torch.zeros_like(residual))
        raw_gain = pair_tokens[..., -1]
        corrected_gain = raw_gain + residual
        margin = corrected_gain - self.tau0.to(
            device=corrected_gain.device, dtype=corrected_gain.dtype
        )
        margin = torch.where(pair_mask, margin, torch.zeros_like(margin))
        return {
            "residual": residual,
            "corrected_gain": corrected_gain,
            "margin": margin,
            "encoded_pairs": encoded,
            "endpoint_context": endpoint_context,
            "image_context": image_context,
            "pair_mask": pair_mask,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "tau0": float(self.tau0),
            "pair_token_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
            "pair_token_dim": SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
            "global_count_feature_names": list(
                SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
            ),
            "global_count_feature_dim": SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
            "pair_encoder": [21, 24, 12],
            "endpoint_pooling": ["other_mean", "all_max", "log1p_other_count"],
            "endpoint_symmetry": ["mean", "absolute_difference"],
            "residual_head": [99, 48, 24, 1],
            "zero_initialized_output": True,
            "parameter_count": self.parameter_count(),
        }


def endpoint_safe_residual_loss(
    margins: torch.Tensor,
    residuals: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return the fixed per-image v0.9.18 safety-first loss."""

    if margins.shape != residuals.shape or not margins.is_floating_point():
        raise ValueError("margins and residuals must share one floating shape")
    batch, pairs = margins.shape
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices must have long shape [batch,pairs,2]")
    if pair_labels.shape != (batch, pairs) or pair_labels.dtype != torch.long:
        raise ValueError("pair_labels must have long shape [batch,pairs]")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [batch,pairs]")
    if len(
        {margins.device, residuals.device, pair_indices.device, pair_labels.device, pair_mask.device}
    ) != 1:
        raise ValueError("endpoint loss inputs must share one device")

    zero = margins.sum() * 0.0
    positive_parts = []
    negative_parts = []
    worst_parts = []
    upward_parts = []
    ranking_parts = []
    l2_parts = []
    positive_count = negative_count = ranking_pair_count = 0
    for image_index in range(batch):
        active = pair_mask[image_index]
        if not bool(active.any()):
            continue
        current_margin = margins[image_index, active].float()
        current_residual = residuals[image_index, active].float()
        current_labels = pair_labels[image_index, active]
        current_indices = pair_indices[image_index, active]
        if not bool(torch.isfinite(current_margin).all()) or not bool(
            torch.isfinite(current_residual).all()
        ):
            raise ValueError("active endpoint loss values are non-finite")
        if bool(
            ((current_labels != 0) & (current_labels != 1)).any()
        ):
            raise ValueError("active pair labels must be binary")
        positive = current_labels == 1
        negative = current_labels == 0
        if bool(positive.any()):
            positive_parts.append(
                F.relu(POSITIVE_MARGIN - current_margin[positive]).square().mean()
            )
            positive_count += int(positive.sum())
        if bool(negative.any()):
            negative_margin = current_margin[negative]
            negative_parts.append(
                F.relu(negative_margin + NEGATIVE_SAFETY_MARGIN).square().mean()
            )
            worst_parts.append(
                F.relu(negative_margin.max() + NEGATIVE_SAFETY_MARGIN).square()
            )
            upward_parts.append(F.relu(current_residual[negative]).square().mean())
            negative_count += int(negative.sum())
        l2_parts.append(current_residual.square().mean())

        if bool(positive.any()) and bool(negative.any()):
            positive_indices = current_indices[positive]
            negative_indices = current_indices[negative]
            shared = (
                positive_indices[:, None, :, None]
                == negative_indices[None, :, None, :]
            ).any(dim=-1).any(dim=-1)
            if bool(shared.any()):
                differences = (
                    RANKING_MARGIN
                    + current_margin[negative].unsqueeze(0)
                    - current_margin[positive].unsqueeze(1)
                )
                ranking_parts.append(F.relu(differences[shared]).square().mean())
                ranking_pair_count += int(shared.sum())

    def mean_or_zero(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(parts).mean() if parts else zero

    positive_loss = mean_or_zero(positive_parts)
    negative_loss = mean_or_zero(negative_parts)
    worst_negative_loss = mean_or_zero(worst_parts)
    negative_upward_loss = mean_or_zero(upward_parts)
    ranking_loss = mean_or_zero(ranking_parts)
    residual_l2_loss = mean_or_zero(l2_parts)
    loss = (
        positive_loss
        + NEGATIVE_LOSS_WEIGHT * negative_loss
        + WORST_NEGATIVE_LOSS_WEIGHT * worst_negative_loss
        + NEGATIVE_UPWARD_LOSS_WEIGHT * negative_upward_loss
        + RANKING_LOSS_WEIGHT * ranking_loss
        + RESIDUAL_L2_WEIGHT * residual_l2_loss
    )
    return {
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "worst_negative_loss": worst_negative_loss,
        "negative_upward_loss": negative_upward_loss,
        "ranking_loss": ranking_loss,
        "residual_l2_loss": residual_l2_loss,
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "ranking_pair_count": ranking_pair_count,
        "loss": loss,
    }


__all__ = [
    "ENDPOINT_RESIDUAL_PARAMETER_COUNT",
    "EvidenceCarrierEndpointIncidenceResidual",
    "endpoint_incidence_context",
    "endpoint_safe_residual_loss",
]
