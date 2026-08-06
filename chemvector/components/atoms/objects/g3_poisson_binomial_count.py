"""Additive Count model induced by candidate occupancy probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_query import poisson_binomial_log_probs


@dataclass(frozen=True)
class G3PoissonBinomialCountConfig:
    query_dim: int = 8
    candidate_dim: int = 41
    global_dim: int = 256
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(
            self.query_dim,
            self.candidate_dim,
            self.global_dim,
            self.hidden_dim,
        ) < 1:
            raise ValueError("Poisson-binomial Count dimensions must be positive")


class G3PoissonBinomialCount(nn.Module):
    """Predict candidate occupancy, then derive Count without a class head."""

    def __init__(
        self,
        config: G3PoissonBinomialCountConfig = G3PoissonBinomialCountConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(config.query_dim),
            nn.Linear(config.query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(config.global_dim),
            nn.Linear(config.global_dim, hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(5 * hidden),
            nn.Linear(5 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.occupancy = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            query_features.ndim != 3
            or query_features.size(-1) != self.config.query_dim
        ):
            raise ValueError("Poisson-binomial Count query features have invalid shape")
        if (
            candidate_features.ndim != 3
            or candidate_features.size(-1) != self.config.candidate_dim
        ):
            raise ValueError(
                "Poisson-binomial Count candidate features have invalid shape"
            )
        if global_features.shape != (len(query_features), self.config.global_dim):
            raise ValueError("Poisson-binomial Count global features have invalid shape")
        if candidate_features.size(0) != query_features.size(0):
            raise ValueError("Poisson-binomial Count batch dimensions differ")

        query = self.query_encoder(query_features)
        candidate = self.candidate_encoder(candidate_features)
        global_hidden = self.global_encoder(global_features)
        context = self.context(
            torch.cat(
                (
                    query.mean(dim=1),
                    query.max(dim=1).values,
                    candidate.mean(dim=1),
                    candidate.max(dim=1).values,
                    global_hidden,
                ),
                dim=-1,
            )
        )
        expanded = context.unsqueeze(1).expand(-1, candidate.size(1), -1)
        presence_logits = self.occupancy(
            torch.cat((candidate, expanded), dim=-1)
        ).squeeze(-1)
        count_log_probs = poisson_binomial_log_probs(presence_logits)
        return {
            "presence_logits": presence_logits,
            "count_log_probs": count_log_probs,
            "expected_count": presence_logits.float().sigmoid().sum(dim=1),
        }


def g3_poisson_binomial_count_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    expected_count_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    if expected_count_weight < 0:
        raise ValueError("Expected Count weight must be nonnegative")
    log_probs = output["count_log_probs"]
    if targets.ndim != 1 or targets.size(0) != log_probs.size(0):
        raise ValueError("Poisson-binomial Count targets have invalid shape")
    if bool((targets < 0).any()) or bool((targets >= log_probs.size(1)).any()):
        raise ValueError("Poisson-binomial Count target exceeds candidate capacity")
    distribution_loss = F.nll_loss(log_probs, targets.long())
    relative_count_loss = (
        (output["expected_count"] - targets.float()).abs()
        / targets.float().clamp_min(1.0)
    ).mean()
    return {
        "loss": distribution_loss + expected_count_weight * relative_count_loss,
        "distribution_loss": distribution_loss,
        "relative_count_loss": relative_count_loss,
    }


__all__ = [
    "G3PoissonBinomialCount",
    "G3PoissonBinomialCountConfig",
    "g3_poisson_binomial_count_loss",
]
