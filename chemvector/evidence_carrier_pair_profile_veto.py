"""High-resolution normal/longitudinal profile veto for v0.9.23."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chemvector.evidence_carrier_pair_local_residual import (
    PAIR_LOCAL_ENCODED_DIM,
    PAIR_LOCAL_SYMMETRIC_CONTEXT_DIM,
    _validate_inputs,
    pair_local_endpoint_incidence_context,
)
from chemvector.evidence_carrier_pair_local_veto import (
    VETO_INITIAL_BIAS,
    VETO_THRESHOLD,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
)


PAIR_PROFILE_RAW_DIM = 480
PAIR_PROFILE_FEATURE_DIM = 32
PAIR_PROFILE_VETO_PARAMETER_COUNT = 60_505


def pair_local_profile_features(pair_local_rasters: torch.Tensor) -> torch.Tensor:
    """Preserve 24-row normal and 48-column longitudinal evidence exactly."""

    if (
        pair_local_rasters.ndim != 5
        or tuple(pair_local_rasters.shape[-3:]) != (2, 24, 48)
        or not pair_local_rasters.is_floating_point()
    ):
        raise ValueError("pair_local_rasters must have shape [batch,pairs,2,24,48]")
    if not bool(torch.isfinite(pair_local_rasters).all()) or bool(
        (pair_local_rasters < 0.0).any() | (pair_local_rasters > 1.0).any()
    ):
        raise ValueError("pair local rasters must be finite in [0,1]")
    ink = pair_local_rasters[..., 0, :, :]
    mask = pair_local_rasters[..., 1, :, :]
    maps = torch.stack((ink, mask, ink * mask, ink * (1.0 - mask)), dim=-3)
    normal_profiles = torch.stack(
        tuple(maps[..., start : start + 16].mean(dim=-1) for start in (0, 16, 32)),
        dim=-2,
    )
    longitudinal_profiles = maps.mean(dim=-2)
    features = torch.cat(
        (normal_profiles.flatten(-3), longitudinal_profiles.flatten(-2)), dim=-1
    )
    if features.size(-1) != PAIR_PROFILE_RAW_DIM:
        raise RuntimeError("pair profile feature width changed")
    return features


class EvidenceCarrierPairProfileVeto(nn.Module):
    """One-way veto driven by unpooled candidate-relative raster profiles."""

    model_type = "evidence-carrier-pair-profile-veto-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        self.profile_encoder = nn.Sequential(
            nn.Linear(PAIR_PROFILE_RAW_DIM, 96),
            nn.GELU(),
            nn.Linear(96, PAIR_PROFILE_FEATURE_DIM),
            nn.GELU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(SAMPLE_THRESHOLD_PAIR_TOKEN_DIM + PAIR_PROFILE_FEATURE_DIM, 36),
            nn.GELU(),
            nn.Linear(36, PAIR_LOCAL_ENCODED_DIM),
            nn.GELU(),
        )
        residual_input_dim = (
            SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
            + PAIR_PROFILE_FEATURE_DIM
            + 2 * PAIR_LOCAL_ENCODED_DIM
            + SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM
            + PAIR_LOCAL_SYMMETRIC_CONTEXT_DIM
        )
        self.veto_head = nn.Sequential(
            nn.Linear(residual_input_dim, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )
        final = self.veto_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("profile veto final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, VETO_INITIAL_BIAS)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != PAIR_PROFILE_VETO_PARAMETER_COUNT:
            raise RuntimeError(
                f"profile veto parameter contract changed: {self.parameter_count()}"
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
        raw_profiles = pair_local_profile_features(pair_local_rasters)
        profiles = pair_tokens.new_zeros((batch, pairs, PAIR_PROFILE_FEATURE_DIM))
        if bool(pair_mask.any()):
            profiles[pair_mask] = self.profile_encoder(raw_profiles[pair_mask])
        encoded = self.pair_encoder(torch.cat((pair_tokens, profiles), dim=-1))
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
        veto_input = torch.cat(
            (
                pair_tokens,
                profiles,
                image_context[:, None, :].expand(-1, pairs, -1),
                endpoint_context,
            ),
            dim=-1,
        )
        veto_logits = self.veto_head(veto_input).squeeze(-1)
        veto_logits = torch.where(pair_mask, veto_logits, torch.zeros_like(veto_logits))
        raw_gain = pair_tokens[..., -1]
        tau0 = self.tau0.to(device=raw_gain.device, dtype=raw_gain.dtype)
        veto_mask = pair_mask & (raw_gain > tau0) & (veto_logits > VETO_THRESHOLD)
        corrected_gain = torch.where(veto_mask, tau0 - 1.0, raw_gain)
        return {
            "veto_logits": veto_logits,
            "veto_mask": veto_mask,
            "corrected_gain": corrected_gain,
            "raw_gain": raw_gain,
            "profile_features": profiles,
            "raw_profile_features": raw_profiles,
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
            "profile_maps": ["ink", "mask", "ink_times_mask", "ink_outside_mask"],
            "normal_profiles": ["left", "middle", "right", 24],
            "longitudinal_profile_width": 48,
            "raw_profile_dim": PAIR_PROFILE_RAW_DIM,
            "profile_encoder": [480, 96, 32],
            "pair_encoder": [53, 36, 16],
            "global_count_feature_names": list(
                SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
            ),
            "endpoint_pooling": ["other_mean", "all_max", "log1p_other_count"],
            "veto_head": [155, 48, 24, 1],
            "veto_initial_bias": VETO_INITIAL_BIAS,
            "veto_threshold": VETO_THRESHOLD,
            "score_increase_possible": False,
            "parameter_count": self.parameter_count(),
        }


__all__ = [
    "PAIR_PROFILE_RAW_DIM",
    "PAIR_PROFILE_VETO_PARAMETER_COUNT",
    "EvidenceCarrierPairProfileVeto",
    "pair_local_profile_features",
]
