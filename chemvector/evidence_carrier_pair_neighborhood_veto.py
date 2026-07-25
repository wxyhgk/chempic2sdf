"""Ordered carrier-neighborhood profile veto for Evidence Carrier v0.9.24."""

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
from chemvector.evidence_carrier_pair_neighborhood import (
    PAIR_NEIGHBORHOOD_FEATURE_DIM,
    PAIR_NEIGHBORHOOD_FEATURE_NAMES,
    PairNeighborhoodConfig,
)
from chemvector.evidence_carrier_pair_profile_veto import (
    PAIR_PROFILE_FEATURE_DIM,
    PAIR_PROFILE_RAW_DIM,
    pair_local_profile_features,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
)


PAIR_NEIGHBOR_ENCODED_DIM = 16
PAIR_NEIGHBOR_MAXIMUM = PairNeighborhoodConfig().maximum_neighbors
PAIR_NEIGHBOR_ORDERED_DIM = PAIR_NEIGHBOR_MAXIMUM * PAIR_NEIGHBOR_ENCODED_DIM
PAIR_NEIGHBOR_CONTEXT_DIM = (
    PAIR_NEIGHBOR_ORDERED_DIM + 2 * PAIR_NEIGHBOR_ENCODED_DIM + 1
)
PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT = 108_033


def _validate_neighborhood_inputs(
    neighbor_features: torch.Tensor,
    neighbor_mask: torch.Tensor,
    pair_mask: torch.Tensor,
) -> None:
    batch, pairs = pair_mask.shape
    if neighbor_features.shape != (
        batch,
        pairs,
        PAIR_NEIGHBOR_MAXIMUM,
        PAIR_NEIGHBORHOOD_FEATURE_DIM,
    ) or not neighbor_features.is_floating_point():
        raise ValueError("neighbor_features must have shape [batch,pairs,12,8]")
    if neighbor_mask.shape != (
        batch,
        pairs,
        PAIR_NEIGHBOR_MAXIMUM,
    ) or neighbor_mask.dtype != torch.bool:
        raise ValueError("neighbor_mask must have boolean shape [batch,pairs,12]")
    if len({neighbor_features.device, neighbor_mask.device, pair_mask.device}) != 1:
        raise ValueError("neighborhood inputs must share one device")
    if bool((neighbor_mask & ~pair_mask[..., None]).any()):
        raise ValueError("inactive pairs cannot contain active neighbors")
    if bool(neighbor_mask.any()):
        active = neighbor_features[neighbor_mask]
        if not bool(torch.isfinite(active).all()) or bool((active.abs() > 1.0).any()):
            raise ValueError("active neighborhood features must be finite in [-1,1]")


class EvidenceCarrierPairNeighborhoodVeto(nn.Module):
    """Delete suspicious active pairs using ordered nearby carrier components."""

    model_type = "evidence-carrier-pair-neighborhood-veto-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(PAIR_NEIGHBORHOOD_FEATURE_DIM, PAIR_NEIGHBOR_ENCODED_DIM),
            nn.GELU(),
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(PAIR_PROFILE_RAW_DIM, 96),
            nn.GELU(),
            nn.Linear(96, PAIR_PROFILE_FEATURE_DIM),
            nn.GELU(),
        )
        pair_input_dim = (
            SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
            + PAIR_PROFILE_FEATURE_DIM
            + PAIR_NEIGHBOR_CONTEXT_DIM
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, PAIR_LOCAL_ENCODED_DIM),
            nn.GELU(),
        )
        veto_input_dim = (
            SAMPLE_THRESHOLD_PAIR_TOKEN_DIM
            + PAIR_PROFILE_FEATURE_DIM
            + PAIR_NEIGHBOR_CONTEXT_DIM
            + 2 * PAIR_LOCAL_ENCODED_DIM
            + SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM
            + PAIR_LOCAL_SYMMETRIC_CONTEXT_DIM
        )
        self.veto_head = nn.Sequential(
            nn.Linear(veto_input_dim, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        final = self.veto_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("neighborhood veto final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, VETO_INITIAL_BIAS)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT:
            raise RuntimeError(
                "neighborhood veto parameter contract changed: "
                f"{self.parameter_count()}"
            )

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_local_rasters: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, pairs = _validate_inputs(
            pair_tokens,
            pair_local_rasters,
            pair_indices,
            pair_mask,
            global_count_features,
        )
        _validate_neighborhood_inputs(neighbor_features, neighbor_mask, pair_mask)

        raw_profiles = pair_local_profile_features(pair_local_rasters)
        profiles = pair_tokens.new_zeros((batch, pairs, PAIR_PROFILE_FEATURE_DIM))
        if bool(pair_mask.any()):
            profiles[pair_mask] = self.profile_encoder(raw_profiles[pair_mask])

        encoded_neighbors = pair_tokens.new_zeros(
            (batch, pairs, PAIR_NEIGHBOR_MAXIMUM, PAIR_NEIGHBOR_ENCODED_DIM)
        )
        if bool(neighbor_mask.any()):
            encoded_neighbors[neighbor_mask] = self.neighbor_encoder(
                neighbor_features[neighbor_mask]
            )
        neighbor_counts = neighbor_mask.sum(dim=-1)
        neighbor_mean = encoded_neighbors.sum(dim=-2) / neighbor_counts.clamp_min(1).to(
            encoded_neighbors.dtype
        )[..., None]
        neighbor_mean = torch.where(
            (neighbor_counts > 0)[..., None],
            neighbor_mean,
            torch.zeros_like(neighbor_mean),
        )
        neighbor_max = torch.where(
            neighbor_mask[..., None],
            encoded_neighbors,
            torch.full_like(encoded_neighbors, -torch.inf),
        ).amax(dim=-2)
        neighbor_max = torch.where(
            (neighbor_counts > 0)[..., None],
            neighbor_max,
            torch.zeros_like(neighbor_max),
        )
        neighbor_count_feature = (
            neighbor_counts.to(pair_tokens.dtype) / float(PAIR_NEIGHBOR_MAXIMUM)
        )[..., None]
        neighborhood_context = torch.cat(
            (
                encoded_neighbors.flatten(-2),
                neighbor_mean,
                neighbor_max,
                neighbor_count_feature,
            ),
            dim=-1,
        )
        neighborhood_context = torch.where(
            pair_mask[..., None],
            neighborhood_context,
            torch.zeros_like(neighborhood_context),
        )

        encoded = self.pair_encoder(
            torch.cat((pair_tokens, profiles, neighborhood_context), dim=-1)
        )
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
                neighborhood_context,
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
            "encoded_neighbors": encoded_neighbors,
            "neighborhood_context": neighborhood_context,
            "neighbor_counts": neighbor_counts,
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
            "profile_encoder": [480, 96, 32],
            "neighbor_feature_names": list(PAIR_NEIGHBORHOOD_FEATURE_NAMES),
            "neighbor_encoder": [8, 16],
            "maximum_neighbors": PAIR_NEIGHBOR_MAXIMUM,
            "neighbor_order": "signed_normal_offset",
            "neighbor_pooling": ["ordered_flatten", "mean", "max", "count"],
            "neighborhood_context_dim": PAIR_NEIGHBOR_CONTEXT_DIM,
            "pair_encoder": [278, 64, 16],
            "global_count_feature_names": list(
                SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_NAMES
            ),
            "endpoint_pooling": ["other_mean", "all_max", "log1p_other_count"],
            "veto_head": [380, 96, 32, 1],
            "veto_initial_bias": VETO_INITIAL_BIAS,
            "veto_threshold": VETO_THRESHOLD,
            "score_increase_possible": False,
            "parameter_count": self.parameter_count(),
        }


__all__ = [
    "PAIR_NEIGHBOR_CONTEXT_DIM",
    "PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT",
    "EvidenceCarrierPairNeighborhoodVeto",
]
