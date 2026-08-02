"""Shared-anchor component energy model for Evidence Carrier v0.9.26."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chemvector.evidence_carrier_pair_local_residual import _validate_inputs
from chemvector.evidence_carrier_pair_profile_veto import (
    PAIR_PROFILE_FEATURE_DIM,
    PAIR_PROFILE_RAW_DIM,
    pair_local_profile_features,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
)


COMPONENT_HIDDEN_DIM = 64
COMPONENT_ENDPOINT_CONTEXT_DIM = 4 * COMPONENT_HIDDEN_DIM + 2
COMPONENT_STRUCTURED_PARAMETER_COUNT = 154_497


def shared_anchor_pair_context(
    hidden: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate other candidate edges at both endpoints, symmetrically."""

    if hidden.ndim != 3 or hidden.size(-1) != COMPONENT_HIDDEN_DIM:
        raise ValueError("hidden must have shape [batch,pairs,64]")
    batch, pairs, width = hidden.shape
    if pair_indices.shape != (batch, pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("pair_indices shape or dtype differs")
    if pair_mask.shape != (batch, pairs) or pair_mask.dtype != torch.bool:
        raise ValueError("pair_mask shape or dtype differs")
    if len({hidden.device, pair_indices.device, pair_mask.device}) != 1:
        raise ValueError("component context inputs must share one device")
    if not pairs or not bool(pair_mask.any()):
        return hidden.new_zeros((batch, pairs, COMPONENT_ENDPOINT_CONTEXT_DIM))
    active_indices = pair_indices[pair_mask]
    if bool((active_indices < 0).any()) or bool(
        (active_indices[:, 0] == active_indices[:, 1]).any()
    ):
        raise ValueError("active component endpoint indices are invalid")
    anchor_capacity = int(active_indices.max()) + 1
    offsets = torch.arange(batch, device=hidden.device) * anchor_capacity
    endpoint_ids = pair_indices.clamp_min(0) + offsets[:, None, None]
    endpoint_active = pair_mask[..., None].expand(-1, -1, 2)
    repeated = hidden[:, :, None, :].expand(-1, -1, 2, -1)
    flat_ids = endpoint_ids[endpoint_active]
    flat_hidden = repeated[endpoint_active]
    slot_count = batch * anchor_capacity
    sums = hidden.new_zeros((slot_count, width))
    sums.index_add_(0, flat_ids, flat_hidden)
    counts = hidden.new_zeros((slot_count,))
    counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=hidden.dtype))
    maximum = torch.full_like(sums, -torch.inf)
    maximum.scatter_reduce_(
        0,
        flat_ids[:, None].expand(-1, width),
        flat_hidden,
        reduce="amax",
        include_self=True,
    )
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    gathered_sum = sums[endpoint_ids]
    gathered_count = counts[endpoint_ids]
    gathered_max = maximum[endpoint_ids]
    other_count = (gathered_count - 1.0).clamp_min(0.0)
    other_mean = (gathered_sum - hidden[:, :, None, :]) / other_count.clamp_min(1.0)[
        ..., None
    ]
    other_mean = torch.where(
        (other_count > 0.0)[..., None], other_mean, torch.zeros_like(other_mean)
    )
    endpoint = torch.cat(
        (other_mean, gathered_max, torch.log1p(other_count)[..., None]), dim=-1
    )
    first, second = endpoint[:, :, 0], endpoint[:, :, 1]
    context = torch.cat(((first + second) * 0.5, (first - second).abs()), dim=-1)
    return torch.where(pair_mask[..., None], context, torch.zeros_like(context))


class _ComponentMessageBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(COMPONENT_HIDDEN_DIM + COMPONENT_ENDPOINT_CONTEXT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, COMPONENT_HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(COMPONENT_HIDDEN_DIM)

    def forward(
        self,
        hidden: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = shared_anchor_pair_context(hidden, pair_indices, pair_mask)
        updated = self.norm(hidden + self.update(torch.cat((hidden, context), dim=-1)))
        return torch.where(pair_mask[..., None], updated, torch.zeros_like(updated))


class EvidenceCarrierComponentStructured(nn.Module):
    """Predict component-aware edge energy residuals for structured matching."""

    model_type = "evidence-carrier-component-structured-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        self.profile_encoder = nn.Sequential(
            nn.Linear(PAIR_PROFILE_RAW_DIM, 96),
            nn.GELU(),
            nn.Linear(96, PAIR_PROFILE_FEATURE_DIM),
            nn.GELU(),
        )
        self.input_encoder = nn.Sequential(
            nn.Linear(SAMPLE_THRESHOLD_PAIR_TOKEN_DIM + PAIR_PROFILE_FEATURE_DIM + 3, 64),
            nn.GELU(),
        )
        self.message_blocks = nn.ModuleList(
            [_ComponentMessageBlock(), _ComponentMessageBlock()]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(COMPONENT_HIDDEN_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        final = self.energy_head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("component energy final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != COMPONENT_STRUCTURED_PARAMETER_COUNT:
            raise RuntimeError(
                "component structured parameter contract changed: "
                f"{self.parameter_count()}"
            )

    def forward(
        self,
        pair_tokens: torch.Tensor,
        pair_local_rasters: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
        global_count_features: torch.Tensor,
        baseline_scores: torch.Tensor,
        component_anchor_counts: torch.Tensor,
        component_edge_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, pairs = _validate_inputs(
            pair_tokens,
            pair_local_rasters,
            pair_indices,
            pair_mask,
            global_count_features,
        )
        for name, value in (
            ("baseline_scores", baseline_scores),
            ("component_anchor_counts", component_anchor_counts),
            ("component_edge_counts", component_edge_counts),
        ):
            if value.shape != (batch, pairs) or not value.is_floating_point():
                raise ValueError(f"{name} must have floating shape [batch,pairs]")
        devices = {
            pair_tokens.device,
            baseline_scores.device,
            component_anchor_counts.device,
            component_edge_counts.device,
        }
        if len(devices) != 1:
            raise ValueError("component structured inputs must share one device")
        if bool(pair_mask.any()):
            active_values = torch.stack(
                (
                    baseline_scores[pair_mask],
                    component_anchor_counts[pair_mask],
                    component_edge_counts[pair_mask],
                ),
                dim=-1,
            )
            if not bool(torch.isfinite(active_values).all()) or bool(
                (active_values[:, 1:] < 0.0).any()
            ):
                raise ValueError("active component scalar features differ")

        raw_profiles = pair_local_profile_features(pair_local_rasters)
        profiles = pair_tokens.new_zeros((batch, pairs, PAIR_PROFILE_FEATURE_DIM))
        if bool(pair_mask.any()):
            profiles[pair_mask] = self.profile_encoder(raw_profiles[pair_mask])
        tau0 = self.tau0.to(device=baseline_scores.device, dtype=baseline_scores.dtype)
        scalar_features = torch.stack(
            (
                baseline_scores - tau0,
                torch.log1p(component_anchor_counts) / 4.0,
                torch.log1p(component_edge_counts) / 4.0,
            ),
            dim=-1,
        )
        hidden = self.input_encoder(
            torch.cat((pair_tokens, profiles, scalar_features), dim=-1)
        )
        hidden = torch.where(pair_mask[..., None], hidden, torch.zeros_like(hidden))
        message_states = []
        for block in self.message_blocks:
            hidden = block(hidden, pair_indices, pair_mask)
            message_states.append(hidden)
        residual = self.energy_head(hidden).squeeze(-1)
        residual = torch.where(pair_mask, residual, torch.zeros_like(residual))
        return {
            "edge_residual": residual,
            "edge_energy": baseline_scores - tau0 + residual,
            "profile_features": profiles,
            "raw_profile_features": raw_profiles,
            "hidden": hidden,
            "message_states": message_states,
            "pair_mask": pair_mask,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "tau0": float(self.tau0),
            "pair_token_names": list(SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES),
            "profile_encoder": [480, 96, 32],
            "input_encoder": [56, 64],
            "message_passing": {
                "rounds": 2,
                "relation": "shared_anchor",
                "endpoint_context": ["other_mean", "all_max", "log1p_other_count"],
                "symmetric_merge": ["mean", "absolute_difference"],
                "hidden_dim": COMPONENT_HIDDEN_DIM,
            },
            "energy_head": [64, 32, 1],
            "initial_residual": 0.0,
            "score_increase_at_output": False,
            "parameter_count": self.parameter_count(),
        }


__all__ = [
    "COMPONENT_STRUCTURED_PARAMETER_COUNT",
    "EvidenceCarrierComponentStructured",
    "shared_anchor_pair_context",
]
