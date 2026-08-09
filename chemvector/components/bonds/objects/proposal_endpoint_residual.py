"""Proposal-level Bond endpoint refinement without changing object identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ProposalEndpointResidualConfig:
    feature_dim: int = 128
    type_dim: int = 8
    hidden_dim: int = 128
    maximum_delta_pixels: float = 32.0
    image_size: float = 768.0

    def __post_init__(self) -> None:
        if min(
            self.feature_dim,
            self.type_dim,
            self.hidden_dim,
            self.maximum_delta_pixels,
            self.image_size,
        ) <= 0:
            raise ValueError("Proposal endpoint residual dimensions must be positive")


class ProposalEndpointResidualHead(nn.Module):
    """Refine decoded endpoints while preserving every other proposal field."""

    def __init__(self, config: ProposalEndpointResidualConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.feature_dim + config.type_dim + 5
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        endpoints: torch.Tensor,
        types: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.size(-1) != self.config.feature_dim:
            raise ValueError("Bond proposal features have an invalid shape")
        batch, proposals, _ = features.shape
        expected = {
            "endpoints": (batch, proposals, 2, 2),
            "types": (batch, proposals, self.config.type_dim),
            "scores": (batch, proposals),
            "mask": (batch, proposals),
        }
        for name, shape in expected.items():
            value = {"endpoints": endpoints, "types": types, "scores": scores, "mask": mask}[name]
            if value.shape != shape:
                raise ValueError(f"Bond proposal {name} has an invalid shape")
        inputs = torch.cat(
            (features, endpoints.flatten(2), types, scores[..., None]), dim=-1
        )
        normalized_delta = self.network(inputs).reshape(batch, proposals, 2, 2).tanh()
        scale = self.config.maximum_delta_pixels / self.config.image_size
        corrected = (endpoints + normalized_delta.to(endpoints.dtype) * scale).clamp(0.0, 1.0)
        corrected = torch.where(mask[..., None, None], corrected, endpoints)
        return {
            "endpoints": corrected,
            "endpoint_delta": corrected - endpoints,
        }


def proposal_endpoint_residual_loss(
    output: dict[str, torch.Tensor],
    base_endpoints: torch.Tensor,
    target_endpoints: torch.Tensor,
    supervision_mask: torch.Tensor,
    proposal_mask: torch.Tensor,
    *,
    image_size: float = 768.0,
    preservation_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    if preservation_weight < 0.0:
        raise ValueError("Endpoint preservation weight must be non-negative")
    corrected = output["endpoints"]
    positive = supervision_mask & proposal_mask
    negative = proposal_mask & ~supervision_mask
    if not bool(positive.any()):
        raise ValueError("Endpoint residual batch has no supervised Bond proposal")
    endpoint_loss = F.smooth_l1_loss(
        corrected[positive].float() * image_size,
        target_endpoints[positive].float() * image_size,
        beta=1.0,
    )
    if bool(negative.any()):
        preservation_loss = (
            (corrected[negative].float() - base_endpoints[negative].float())
            .mul(image_size)
            .square()
            .mean()
        )
    else:
        preservation_loss = endpoint_loss * 0.0
    loss = endpoint_loss + preservation_weight * preservation_loss
    endpoint_error = torch.linalg.vector_norm(
        (corrected[positive].float() - target_endpoints[positive].float()) * image_size,
        dim=-1,
    )
    return {
        "loss": loss,
        "endpoint_loss": endpoint_loss,
        "preservation_loss": preservation_loss,
        "endpoint_mae_px": endpoint_error.mean(),
        "endpoint_max_px": endpoint_error.max(),
    }


__all__ = [
    "ProposalEndpointResidualConfig",
    "ProposalEndpointResidualHead",
    "proposal_endpoint_residual_loss",
]
