"""Pairwise outcome router for safe frozen relation-route selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


ALTERNATIVE_ROUTES = ("USE_NEW", "USE_OLD")
ROUTE_OUTCOMES = ("BOTH_WRONG", "BOTH_EXACT", "RESCUE", "HARMFUL")


@dataclass(frozen=True, slots=True)
class OutputRouteRiskRouterConfig:
    input_dim: int
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim) < 1:
            raise ValueError("Output route risk router dimensions must be positive")


class OutputRouteRiskRouter(nn.Module):
    def __init__(self, config: OutputRouteRiskRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                len(ALTERNATIVE_ROUTES) * len(ROUTE_OUTCOMES),
            ),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.config.input_dim:
            raise ValueError("Output route risk router feature shape differs")
        logits = self.network(features)
        return logits.view(
            len(features), len(ALTERNATIVE_ROUTES), len(ROUTE_OUTCOMES)
        )
