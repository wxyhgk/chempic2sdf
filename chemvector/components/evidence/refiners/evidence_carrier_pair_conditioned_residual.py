"""Pair-conditioned residual scorer for Evidence Carrier v0.9.17.

The model consumes only frozen, target-free pair descriptors.  A small
DeepSets encoder summarizes the candidate set and conditions an independent
residual for every candidate.  The production threshold and matching solver
remain outside this module.
"""

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


PAIR_RESIDUAL_PARAMETER_COUNT = 2_841
PAIR_NEGATIVE_LABEL = 0
PAIR_POSITIVE_LABEL = 1
PAIR_RESIDUAL_HARD_NEGATIVE_TOP_K = 4
PAIR_RESIDUAL_HARD_NEGATIVE_WEIGHT = 0.5
PAIR_RESIDUAL_L2_WEIGHT = 0.01


def _validate_inputs(
    pair_tokens: torch.Tensor,
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
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask shape or dtype is invalid")
    if (
        global_count_features.shape
        != (batch, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM)
        or not global_count_features.is_floating_point()
    ):
        raise ValueError("global_count_features must have shape [batch,4]")
    if len(
        {pair_tokens.device, pair_mask.device, global_count_features.device}
    ) != 1:
        raise ValueError("pair residual inputs must share one device")
    if bool(pair_mask.any()) and not bool(
        torch.isfinite(pair_tokens[pair_mask]).all()
    ):
        raise ValueError("active pair tokens contain non-finite values")
    if not bool(torch.isfinite(global_count_features).all()):
        raise ValueError("global count features contain non-finite values")
    if bool((global_count_features < 0.0).any()):
        raise ValueError("global log-count features cannot be negative")
    return batch, pairs


class EvidenceCarrierPairConditionedResidual(nn.Module):
    """Predict one target-free residual for every frozen pair gain."""

    model_type = "evidence-carrier-pair-conditioned-residual-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        if (
            isinstance(tau0, bool)
            or not isinstance(tau0, (int, float))
            or not math.isfinite(float(tau0))
        ):
            raise ValueError("tau0 must be finite")
        self.context_encoder = nn.Sequential(
            nn.Linear(SAMPLE_THRESHOLD_PAIR_TOKEN_DIM, 24),
            nn.GELU(),
            nn.Linear(24, 12),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(
                SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
                + 2 * 12
                + SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
                30,
            ),
            nn.GELU(),
            nn.Linear(30, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("pair residual final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != PAIR_RESIDUAL_PARAMETER_COUNT:
            raise RuntimeError("pair residual parameter contract changed")

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, pairs = _validate_inputs(
            pair_tokens,
            pair_mask,
            global_count_features,
        )
        encoded = self.context_encoder(pair_tokens)
        active = pair_mask[..., None]
        encoded_sum = torch.where(
            active, encoded, torch.zeros_like(encoded)
        ).sum(dim=1)
        counts = pair_mask.sum(dim=1)
        pooled_mean = encoded_sum / counts.clamp_min(1).to(
            encoded.dtype
        ).unsqueeze(-1)

        empty_pool = encoded.new_zeros((batch, encoded.size(-1)))
        if pairs:
            pooled_max = torch.where(
                active,
                encoded,
                torch.full_like(encoded, -torch.inf),
            ).amax(dim=1)
            pooled_max = torch.where(
                (counts > 0).unsqueeze(-1), pooled_max, empty_pool
            )
        else:
            pooled_max = empty_pool
        image_context = torch.cat(
            (pooled_mean, pooled_max, global_count_features), dim=-1
        )
        expanded_context = image_context[:, None, :].expand(-1, pairs, -1)
        residual_input = torch.cat((pair_tokens, expanded_context), dim=-1)
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
            "image_context": image_context,
            "pooled_pair_mean": pooled_mean,
            "pooled_pair_max": pooled_max,
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
            "context_encoder": [21, 24, 12],
            "pooling": ["mean", "max"],
            "residual_head": [49, 30, 16, 1],
            "zero_initialized_output": True,
            "parameter_count": self.parameter_count(),
        }


def _hard_negative_loss(
    margins: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, int]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("hard-negative top_k must be a positive integer")
    zero = margins.sum() * 0.0
    parts = []
    for batch_index in range(len(margins)):
        negative = pair_mask[batch_index] & (
            pair_labels[batch_index] == PAIR_NEGATIVE_LABEL
        )
        if not bool(negative.any()):
            continue
        scores = margins[batch_index, negative].float()
        hardest = scores.topk(min(top_k, len(scores))).values
        parts.append(F.softplus(hardest).mean())
    return (torch.stack(parts).mean() if parts else zero), len(parts)


def _classification_and_ranking_loss(
    margins: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
) -> dict[str, Any]:
    if margins.ndim != 2 or not margins.is_floating_point():
        raise ValueError("margins must be a floating [batch,pairs] tensor")
    batch, pairs = margins.shape
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices must have long shape [batch,pairs,2]")
    if pair_labels.shape != (batch, pairs) or pair_labels.dtype != torch.long:
        raise ValueError("pair_labels must have long shape [batch,pairs]")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [batch,pairs]")
    devices = {
        margins.device,
        pair_indices.device,
        pair_labels.device,
        pair_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("pair loss inputs must share one device")
    if bool(pair_mask.any()):
        active_margins = margins[pair_mask]
        active_indices = pair_indices[pair_mask]
        active_labels = pair_labels[pair_mask]
        if not bool(torch.isfinite(active_margins).all()):
            raise ValueError("active margins contain non-finite values")
        if bool((active_indices < 0).any()) or bool(
            (active_indices[:, 0] >= active_indices[:, 1]).any()
        ):
            raise ValueError("active pair indices must be nonnegative ascending pairs")
        invalid_labels = (active_labels != PAIR_NEGATIVE_LABEL) & (
            active_labels != PAIR_POSITIVE_LABEL
        )
        if bool(invalid_labels.any()):
            raise ValueError("active pair labels must be binary")

    zero = margins.sum() * 0.0
    image_classification = []
    image_ranking = []
    positive_count = negative_count = ranking_pair_count = 0
    for batch_index in range(batch):
        active = pair_mask[batch_index]
        if not bool(active.any()):
            continue
        current_margins = margins[batch_index, active].float()
        current_labels = pair_labels[batch_index, active]
        current_indices = pair_indices[batch_index, active]
        positive = current_labels == PAIR_POSITIVE_LABEL
        negative = current_labels == PAIR_NEGATIVE_LABEL
        class_parts = []
        if bool(positive.any()):
            class_parts.append(F.softplus(-current_margins[positive]).mean())
            positive_count += int(positive.sum())
        if bool(negative.any()):
            class_parts.append(F.softplus(current_margins[negative]).mean())
            negative_count += int(negative.sum())
        image_classification.append(torch.stack(class_parts).mean())

        if bool(positive.any()) and bool(negative.any()):
            positive_indices = current_indices[positive]
            negative_indices = current_indices[negative]
            shared = (
                positive_indices[:, None, :, None]
                == negative_indices[None, :, None, :]
            ).any(dim=-1).any(dim=-1)
            if bool(shared.any()):
                differences = (
                    1.0
                    + current_margins[negative].unsqueeze(0)
                    - current_margins[positive].unsqueeze(1)
                )
                image_ranking.append(F.softplus(differences[shared]).mean())
                ranking_pair_count += int(shared.sum())
    balanced = (
        torch.stack(image_classification).mean() if image_classification else zero
    )
    ranking = torch.stack(image_ranking).mean() if image_ranking else zero
    return {
        "balanced_bce": balanced,
        "ranking_loss": ranking,
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "ranking_pair_count": ranking_pair_count,
        "loss": balanced + 0.5 * ranking,
    }


def pair_conditioned_residual_loss(
    margins: torch.Tensor,
    residuals: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    hard_negative_top_k: int = PAIR_RESIDUAL_HARD_NEGATIVE_TOP_K,
    hard_negative_weight: float = PAIR_RESIDUAL_HARD_NEGATIVE_WEIGHT,
    residual_l2_weight: float = PAIR_RESIDUAL_L2_WEIGHT,
) -> dict[str, Any]:
    """Return the fixed v0.9.17 pair classification and residual loss."""

    if margins.shape != residuals.shape or not residuals.is_floating_point():
        raise ValueError("margins and residuals must share one floating shape")
    if margins.device != residuals.device:
        raise ValueError("margins and residuals must share one device")
    for name, value in (
        ("hard_negative_weight", hard_negative_weight),
        ("residual_l2_weight", residual_l2_weight),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    base = _classification_and_ranking_loss(
        margins,
        pair_indices,
        pair_labels,
        pair_mask,
    )
    hard_negative, hard_negative_images = _hard_negative_loss(
        margins,
        pair_labels,
        pair_mask,
        top_k=hard_negative_top_k,
    )
    residual_l2_parts = [
        residuals[batch_index, pair_mask[batch_index]].float().square().mean()
        for batch_index in range(len(residuals))
        if bool(pair_mask[batch_index].any())
    ]
    residual_l2 = (
        torch.stack(residual_l2_parts).mean()
        if residual_l2_parts
        else residuals.sum() * 0.0
    )
    loss = (
        base["loss"]
        + float(hard_negative_weight) * hard_negative
        + float(residual_l2_weight) * residual_l2
    )
    return {
        **base,
        "classification_and_ranking_loss": base["loss"],
        "hard_negative_loss": hard_negative,
        "images_with_hard_negatives": hard_negative_images,
        "residual_l2_loss": residual_l2,
        "images_with_residuals": len(residual_l2_parts),
        "loss": loss,
    }


__all__ = [
    "PAIR_RESIDUAL_HARD_NEGATIVE_TOP_K",
    "PAIR_RESIDUAL_HARD_NEGATIVE_WEIGHT",
    "PAIR_RESIDUAL_L2_WEIGHT",
    "PAIR_RESIDUAL_PARAMETER_COUNT",
    "EvidenceCarrierPairConditionedResidual",
    "pair_conditioned_residual_loss",
]
