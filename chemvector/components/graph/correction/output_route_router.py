"""Small graph-output router for choosing among frozen relation routes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


ROUTE_ACTIONS = ("KEEP_FUSION", "USE_NEW", "USE_OLD")


@dataclass(frozen=True, slots=True)
class OutputRouteRouterConfig:
    input_dim: int
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim) < 1:
            raise ValueError("Output route router dimensions must be positive")


class OutputRouteRouter(nn.Module):
    def __init__(self, config: OutputRouteRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(ROUTE_ACTIONS)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.config.input_dim:
            raise ValueError("Output route router feature shape differs")
        return self.network(features)
