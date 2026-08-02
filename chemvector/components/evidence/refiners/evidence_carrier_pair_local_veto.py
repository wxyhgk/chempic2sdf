"""Conservative one-way pair veto for Evidence Carrier v0.9.21."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.evidence_carrier_pair_local_spatial_residual import (
    PAIR_LOCAL_SPATIAL_PARAMETER_COUNT,
    EvidenceCarrierPairLocalSpatialResidual,
)


PAIR_LOCAL_VETO_PARAMETER_COUNT = PAIR_LOCAL_SPATIAL_PARAMETER_COUNT
VETO_INITIAL_BIAS = -4.0
VETO_THRESHOLD = 0.0
POSITIVE_KEEP_MARGIN = 0.5
HARD_NEGATIVE_VETO_MARGIN = 1.0
POSITIVE_LOSS_WEIGHT = 2.0
WORST_POSITIVE_LOSS_WEIGHT = 8.0
NEGATIVE_LOSS_WEIGHT = 1.0
HARD_NEGATIVE_LOSS_WEIGHT = 4.0
LOGIT_L2_WEIGHT = 0.001


class EvidenceCarrierPairLocalVeto(EvidenceCarrierPairLocalSpatialResidual):
    """Delete suspicious active candidates without ever increasing a score."""

    model_type = "evidence-carrier-pair-local-veto-v0.1"

    def __init__(self) -> None:
        super().__init__()
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("pair-local veto final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, VETO_INITIAL_BIAS)
        if self.parameter_count() != PAIR_LOCAL_VETO_PARAMETER_COUNT:
            raise RuntimeError("pair-local veto parameter contract changed")

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_local_rasters: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = super().forward(
            pair_tokens,
            pair_local_rasters,
            pair_indices,
            pair_mask,
            global_count_features,
        )
        veto_logits = encoded["residual"]
        raw_gain = pair_tokens[..., -1]
        tau0 = self.tau0.to(device=raw_gain.device, dtype=raw_gain.dtype)
        veto_mask = (
            pair_mask
            & (raw_gain > tau0)
            & (veto_logits > VETO_THRESHOLD)
        )
        removed_gain = tau0 - 1.0
        corrected_gain = torch.where(veto_mask, removed_gain, raw_gain)
        return {
            "veto_logits": veto_logits,
            "veto_mask": veto_mask,
            "corrected_gain": corrected_gain,
            "raw_gain": raw_gain,
            "local_features": encoded["local_features"],
            "encoded_pairs": encoded["encoded_pairs"],
            "endpoint_context": encoded["endpoint_context"],
            "image_context": encoded["image_context"],
            "pair_mask": pair_mask,
        }

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        config.update(
            {
                "model_type": self.model_type,
                "output": "one_way_active_pair_veto",
                "veto_initial_bias": VETO_INITIAL_BIAS,
                "veto_threshold": VETO_THRESHOLD,
                "veto_scope": "raw_gain_strictly_greater_than_tau0",
                "veto_action": "set_score_to_tau0_minus_1",
                "score_increase_possible": False,
                "parameter_count": self.parameter_count(),
            }
        )
        return config


def pair_local_veto_loss(
    veto_logits: torch.Tensor,
    raw_gain: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    tau0: float,
) -> dict[str, Any]:
    """Safety-first loss with explicit keep-positive and veto-hard-negative terms."""

    if veto_logits.shape != raw_gain.shape or not veto_logits.is_floating_point():
        raise ValueError("veto logits and raw gain must share one floating shape")
    batch, pairs = veto_logits.shape
    if pair_labels.shape != (batch, pairs) or pair_labels.dtype != torch.long:
        raise ValueError("pair_labels must have long shape [batch,pairs]")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [batch,pairs]")
    if len({veto_logits.device, raw_gain.device, pair_labels.device, pair_mask.device}) != 1:
        raise ValueError("pair-local veto loss inputs must share one device")
    if isinstance(tau0, bool) or not isinstance(tau0, (int, float)) or not math.isfinite(
        float(tau0)
    ):
        raise ValueError("tau0 must be finite")

    zero = veto_logits.sum() * 0.0
    positive_parts = []
    worst_positive_parts = []
    negative_parts = []
    hard_negative_parts = []
    l2_parts = []
    positive_count = negative_count = hard_negative_count = 0
    for image_index in range(batch):
        active = pair_mask[image_index]
        if not bool(active.any()):
            continue
        logits = veto_logits[image_index, active].float()
        gains = raw_gain[image_index, active].float()
        labels = pair_labels[image_index, active]
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(gains).all()):
            raise ValueError("active veto loss values are non-finite")
        if bool(((labels != 0) & (labels != 1)).any()):
            raise ValueError("active pair labels must be binary")
        positive = labels == 1
        negative = labels == 0
        hard_negative = negative & (gains > float(tau0))
        if bool(positive.any()):
            positive_logits = logits[positive]
            positive_parts.append(F.softplus(positive_logits + POSITIVE_KEEP_MARGIN).mean())
            worst_positive_parts.append(
                F.relu(positive_logits.max() + POSITIVE_KEEP_MARGIN).square()
            )
            positive_count += int(positive.sum())
        if bool(negative.any()):
            negative_logits = logits[negative]
            negative_parts.append(F.softplus(-negative_logits).mean())
            negative_count += int(negative.sum())
        if bool(hard_negative.any()):
            hard_logits = logits[hard_negative]
            hard_negative_parts.append(
                F.relu(HARD_NEGATIVE_VETO_MARGIN - hard_logits.min()).square()
            )
            hard_negative_count += int(hard_negative.sum())
        l2_parts.append(logits.square().mean())

    def mean_or_zero(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(parts).mean() if parts else zero

    positive_loss = mean_or_zero(positive_parts)
    worst_positive_loss = mean_or_zero(worst_positive_parts)
    negative_loss = mean_or_zero(negative_parts)
    hard_negative_loss = mean_or_zero(hard_negative_parts)
    logit_l2_loss = mean_or_zero(l2_parts)
    loss = (
        POSITIVE_LOSS_WEIGHT * positive_loss
        + WORST_POSITIVE_LOSS_WEIGHT * worst_positive_loss
        + NEGATIVE_LOSS_WEIGHT * negative_loss
        + HARD_NEGATIVE_LOSS_WEIGHT * hard_negative_loss
        + LOGIT_L2_WEIGHT * logit_l2_loss
    )
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "worst_positive_loss": worst_positive_loss,
        "negative_loss": negative_loss,
        "hard_negative_loss": hard_negative_loss,
        "logit_l2_loss": logit_l2_loss,
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "hard_negative_pair_count": hard_negative_count,
    }


__all__ = [
    "PAIR_LOCAL_VETO_PARAMETER_COUNT",
    "VETO_THRESHOLD",
    "EvidenceCarrierPairLocalVeto",
    "pair_local_veto_loss",
]
