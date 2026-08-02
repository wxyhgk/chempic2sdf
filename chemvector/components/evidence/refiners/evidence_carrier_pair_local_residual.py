"""Pair-local raster residual scorer for Evidence Carrier v0.9.19."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from chemvector.evidence_carrier_pair_local_raster import PairLocalRasterConfig
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
)


PAIR_LOCAL_ENCODED_DIM = 16
PAIR_LOCAL_VISUAL_DIM = 24
PAIR_LOCAL_ENDPOINT_CONTEXT_DIM = 33
PAIR_LOCAL_SYMMETRIC_CONTEXT_DIM = 66
PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT = 15_353


def _validate_inputs(
    pair_tokens: torch.Tensor,
    pair_local_rasters: torch.Tensor,
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
    crop_config = PairLocalRasterConfig()
    if pair_local_rasters.shape != (
        batch,
        pairs,
        2,
        crop_config.height,
        crop_config.width,
    ) or not pair_local_rasters.is_floating_point():
        raise ValueError("pair_local_rasters must have shape [batch,pairs,2,24,48]")
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices must have long shape [batch,pairs,2]")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask must have boolean shape [batch,pairs]")
    if global_count_features.shape != (
        batch,
        SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    ) or not global_count_features.is_floating_point():
        raise ValueError("global_count_features must have shape [batch,4]")
    devices = {
        pair_tokens.device,
        pair_local_rasters.device,
        pair_indices.device,
        pair_mask.device,
        global_count_features.device,
    }
    if len(devices) != 1:
        raise ValueError("pair-local residual inputs must share one device")
    if bool(pair_mask.any()):
        if not bool(torch.isfinite(pair_tokens[pair_mask]).all()):
            raise ValueError("active pair tokens contain non-finite values")
        active_rasters = pair_local_rasters[pair_mask]
        if not bool(torch.isfinite(active_rasters).all()) or bool(
            (active_rasters < 0.0).any() | (active_rasters > 1.0).any()
        ):
            raise ValueError("active pair local rasters must be finite in [0,1]")
        active_indices = pair_indices[pair_mask]
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


def pair_local_endpoint_incidence_context(
    encoded_pairs: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Return symmetric endpoint incidence context for 16D fused pair tokens."""

    if encoded_pairs.ndim != 3 or encoded_pairs.size(-1) != PAIR_LOCAL_ENCODED_DIM:
        raise ValueError("encoded_pairs must have shape [batch,pairs,16]")
    batch, pairs, hidden = encoded_pairs.shape
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices shape or dtype differs")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask shape or dtype differs")
    if len({encoded_pairs.device, pair_indices.device, pair_mask.device}) != 1:
        raise ValueError("endpoint context inputs must share one device")
    if not pairs or not bool(pair_mask.any()):
        return encoded_pairs.new_zeros(
            (batch, pairs, PAIR_LOCAL_SYMMETRIC_CONTEXT_DIM)
        )

    active_indices = pair_indices[pair_mask]
    if bool((active_indices < 0).any()) or bool(
        (active_indices[:, 0] == active_indices[:, 1]).any()
    ):
        raise ValueError("active endpoint indices are invalid")
    anchor_capacity = int(active_indices.max()) + 1
    offsets = torch.arange(batch, device=pair_indices.device) * anchor_capacity
    endpoint_ids = pair_indices.clamp_min(0) + offsets[:, None, None]
    endpoint_active = pair_mask[..., None].expand(-1, -1, 2)
    repeated = encoded_pairs[:, :, None, :].expand(-1, -1, 2, -1)
    flat_ids = endpoint_ids[endpoint_active]
    flat_values = repeated[endpoint_active]

    slots = batch * anchor_capacity
    sums = encoded_pairs.new_zeros((slots, hidden))
    sums.index_add_(0, flat_ids, flat_values)
    counts = encoded_pairs.new_zeros((slots,))
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
    other_count = (gathered_count - 1.0).clamp_min(0.0)
    other_mean = (gathered_sum - encoded_pairs[:, :, None, :]) / other_count.clamp_min(
        1.0
    )[..., None]
    other_mean = torch.where(
        (other_count > 0.0)[..., None], other_mean, torch.zeros_like(other_mean)
    )
    endpoint = torch.cat(
        (other_mean, gathered_max, torch.log1p(other_count)[..., None]), dim=-1
    )
    first, second = endpoint[:, :, 0], endpoint[:, :, 1]
    symmetric = torch.cat(((first + second) * 0.5, (first - second).abs()), dim=-1)
    return torch.where(pair_mask[..., None], symmetric, torch.zeros_like(symmetric))


class EvidenceCarrierPairLocalResidual(nn.Module):
    """Fuse complete-PNG local crops with frozen pair and incidence features."""

    model_type = "evidence-carrier-pair-local-residual-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        if isinstance(tau0, bool) or not isinstance(tau0, (int, float)) or not math.isfinite(
            float(tau0)
        ):
            raise ValueError("tau0 must be finite")
        self.local_encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, PAIR_LOCAL_VISUAL_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(SAMPLE_THRESHOLD_PAIR_TOKEN_DIM + PAIR_LOCAL_VISUAL_DIM, 36),
            nn.GELU(),
            nn.Linear(36, PAIR_LOCAL_ENCODED_DIM),
            nn.GELU(),
        )
        residual_input_dim = (
            SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
            + PAIR_LOCAL_VISUAL_DIM
            + 2 * PAIR_LOCAL_ENCODED_DIM
            + SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM
            + PAIR_LOCAL_SYMMETRIC_CONTEXT_DIM
        )
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("pair-local residual final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT:
            raise RuntimeError(
                f"pair-local residual parameter contract changed: {self.parameter_count()}"
            )

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_local_rasters: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, pairs = _validate_inputs(
            pair_tokens,
            pair_local_rasters,
            pair_indices,
            pair_mask,
            global_count_features,
        )
        local = pair_tokens.new_zeros((batch, pairs, PAIR_LOCAL_VISUAL_DIM))
        if bool(pair_mask.any()):
            encoded_local = self.local_encoder(pair_local_rasters[pair_mask]).flatten(1)
            local[pair_mask] = encoded_local
        encoded = self.pair_encoder(torch.cat((pair_tokens, local), dim=-1))
        encoded = torch.where(pair_mask[..., None], encoded, torch.zeros_like(encoded))
        counts = pair_mask.sum(dim=1)
        pooled_mean = encoded.sum(dim=1) / counts.clamp_min(1).to(encoded.dtype)[:, None]
        if pairs:
            pooled_max = torch.where(
                pair_mask[..., None], encoded, torch.full_like(encoded, -torch.inf)
            ).amax(dim=1)
            pooled_max = torch.where(
                (counts > 0)[:, None], pooled_max, torch.zeros_like(pooled_max)
            )
        else:
            pooled_max = encoded.new_zeros((batch, PAIR_LOCAL_ENCODED_DIM))
        image_context = torch.cat(
            (pooled_mean, pooled_max, global_count_features), dim=-1
        )
        endpoint_context = pair_local_endpoint_incidence_context(
            encoded, pair_indices, pair_mask
        )
        residual_input = torch.cat(
            (
                pair_tokens,
                local,
                image_context[:, None, :].expand(-1, pairs, -1),
                endpoint_context,
            ),
            dim=-1,
        )
        residual = self.residual_head(residual_input).squeeze(-1)
        residual = torch.where(pair_mask, residual, torch.zeros_like(residual))
        corrected_gain = pair_tokens[..., -1] + residual
        margin = corrected_gain - self.tau0.to(
            device=corrected_gain.device, dtype=corrected_gain.dtype
        )
        margin = torch.where(pair_mask, margin, torch.zeros_like(margin))
        return {
            "residual": residual,
            "corrected_gain": corrected_gain,
            "margin": margin,
            "local_features": local,
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
            "pair_local_raster": PairLocalRasterConfig().as_dict(),
            "local_encoder": [2, 8, 16, 24],
            "pair_encoder": [45, 36, 16],
            "global_count_feature_names": list(
                SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
            ),
            "endpoint_pooling": ["other_mean", "all_max", "log1p_other_count"],
            "endpoint_symmetry": ["mean", "absolute_difference"],
            "residual_head": [147, 48, 24, 1],
            "zero_initialized_output": True,
            "parameter_count": self.parameter_count(),
        }


__all__ = [
    "PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT",
    "EvidenceCarrierPairLocalResidual",
    "pair_local_endpoint_incidence_context",
]
