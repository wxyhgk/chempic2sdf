"""Score-blind shared-anchor component model for Evidence Carrier v0.9.27."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chemvector.evidence_carrier_component_structured import (
    COMPONENT_HIDDEN_DIM,
    _ComponentMessageBlock,
)
from chemvector.evidence_carrier_pair_local_residual import _validate_inputs
from chemvector.evidence_carrier_pair_profile_veto import (
    PAIR_PROFILE_FEATURE_DIM,
    PAIR_PROFILE_RAW_DIM,
    pair_local_profile_features,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES,
)


SCORE_BLIND_PAIR_TOKEN_DIM = 20
COMPONENT_SCORE_BLIND_PARAMETER_COUNT = 154_369


class EvidenceCarrierComponentScoreBlind(nn.Module):
    """Predict component edge residuals without any learned score input."""

    model_type = "evidence-carrier-component-score-blind-v0.1"

    def __init__(self, *, tau0: float = SAMPLE_THRESHOLD_BASELINE_TAU) -> None:
        super().__init__()
        self.profile_encoder = nn.Sequential(
            nn.Linear(PAIR_PROFILE_RAW_DIM, 96),
            nn.GELU(),
            nn.Linear(96, PAIR_PROFILE_FEATURE_DIM),
            nn.GELU(),
        )
        self.input_encoder = nn.Sequential(
            nn.Linear(SCORE_BLIND_PAIR_TOKEN_DIM + PAIR_PROFILE_FEATURE_DIM + 2, 64),
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
            raise RuntimeError("score-blind energy final layer contract changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("tau0", torch.tensor(float(tau0), dtype=torch.float64))
        if self.parameter_count() != COMPONENT_SCORE_BLIND_PARAMETER_COUNT:
            raise RuntimeError(
                "score-blind parameter contract changed: "
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
        if len(
            {
                pair_tokens.device,
                baseline_scores.device,
                component_anchor_counts.device,
                component_edge_counts.device,
            }
        ) != 1:
            raise ValueError("score-blind inputs must share one device")

        raw_profiles = pair_local_profile_features(pair_local_rasters)
        profiles = pair_tokens.new_zeros((batch, pairs, PAIR_PROFILE_FEATURE_DIM))
        if bool(pair_mask.any()):
            profiles[pair_mask] = self.profile_encoder(raw_profiles[pair_mask])
        count_features = torch.stack(
            (
                torch.log1p(component_anchor_counts) / 4.0,
                torch.log1p(component_edge_counts) / 4.0,
            ),
            dim=-1,
        )
        visual_pair_tokens = pair_tokens[..., :SCORE_BLIND_PAIR_TOKEN_DIM]
        hidden = self.input_encoder(
            torch.cat((visual_pair_tokens, profiles, count_features), dim=-1)
        )
        hidden = torch.where(pair_mask[..., None], hidden, torch.zeros_like(hidden))
        message_states = []
        for block in self.message_blocks:
            hidden = block(hidden, pair_indices, pair_mask)
            message_states.append(hidden)
        residual = self.energy_head(hidden).squeeze(-1)
        residual = torch.where(pair_mask, residual, torch.zeros_like(residual))
        tau0 = self.tau0.to(device=baseline_scores.device, dtype=baseline_scores.dtype)
        return {
            "edge_residual": residual,
            "edge_energy": baseline_scores - tau0 + residual,
            "profile_features": profiles,
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
            "pair_token_names": list(
                SAMPLE_THRESHOLD_PAIR_TOKEN_NAMES[:SCORE_BLIND_PAIR_TOKEN_DIM]
            ),
            "excluded_score_inputs": ["raw_pair_gain", "baseline_scores"],
            "baseline_use": "external_matching_energy_only",
            "profile_encoder": [480, 96, 32],
            "input_encoder": [54, 64],
            "message_passing": {
                "rounds": 2,
                "relation": "shared_anchor",
                "hidden_dim": COMPONENT_HIDDEN_DIM,
            },
            "energy_head": [64, 32, 1],
            "initial_residual": 0.0,
            "score_increase_at_output": False,
            "parameter_count": self.parameter_count(),
        }


__all__ = [
    "COMPONENT_SCORE_BLIND_PARAMETER_COUNT",
    "EvidenceCarrierComponentScoreBlind",
]
