"""Deterministic low-capacity KEEP/MOVE decision over explicit evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


EXPLICIT_KEEP_MOVE_FEATURES = 7


@dataclass(frozen=True, slots=True)
class ExplicitKeepMoveLinearConfig:
    feature_dim: int = EXPLICIT_KEEP_MOVE_FEATURES
    l2: float = 0.01
    newton_steps: int = 64

    def __post_init__(self) -> None:
        if self.feature_dim != EXPLICIT_KEEP_MOVE_FEATURES:
            raise ValueError("explicit KEEP/MOVE feature dimension differs")
        if not math.isfinite(self.l2) or self.l2 <= 0.0:
            raise ValueError("explicit KEEP/MOVE L2 must be positive")
        if self.newton_steps < 1:
            raise ValueError("explicit KEEP/MOVE Newton steps must be positive")


class ExplicitKeepMoveLinear(nn.Module):
    model_type = "explicit-keep-move-linear-v0.1"

    def __init__(self, config: ExplicitKeepMoveLinearConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExplicitKeepMoveLinearConfig()
        self.register_buffer("feature_mean", torch.zeros(self.config.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.config.feature_dim))
        self.linear = nn.Linear(self.config.feature_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.config.feature_dim:
            raise ValueError("explicit KEEP/MOVE features have invalid shape")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("explicit KEEP/MOVE features contain non-finite values")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.linear(normalized).squeeze(-1)

    def config_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "feature_dim": self.config.feature_dim,
            "l2": self.config.l2,
            "newton_steps": self.config.newton_steps,
            "parameter_count": self.parameter_count,
        }


def _objective(
    design: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    l2: float,
) -> torch.Tensor:
    logits = design @ weights
    return F.softplus(logits).sub(labels * logits).sum() + 0.5 * l2 * weights[1:].square().sum()


@torch.no_grad()
def fit_explicit_keep_move_linear(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: ExplicitKeepMoveLinearConfig | None = None,
) -> tuple[ExplicitKeepMoveLinear, dict]:
    config = config or ExplicitKeepMoveLinearConfig()
    if features.ndim != 2 or features.size(1) != config.feature_dim:
        raise ValueError("explicit KEEP/MOVE fit features have invalid shape")
    if labels.shape != (len(features),) or labels.dtype != torch.bool:
        raise ValueError("explicit KEEP/MOVE fit labels must be a boolean vector")
    if min(int(labels.sum()), int((~labels).sum())) == 0:
        raise ValueError("explicit KEEP/MOVE fit requires both classes")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("explicit KEEP/MOVE fit features contain non-finite values")

    values = features.detach().cpu().double()
    targets = labels.detach().cpu().double()
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    scale = torch.where(scale >= 1e-8, scale, torch.ones_like(scale))
    normalized = (values - mean) / scale
    design = torch.cat((torch.ones(len(values), 1), normalized), dim=1)
    weights = torch.zeros(config.feature_dim + 1, dtype=torch.float64)
    penalty = torch.zeros_like(weights)
    penalty[1:] = config.l2
    identity = torch.eye(len(weights), dtype=torch.float64) * 1e-8
    completed = 0
    maximum_update = math.inf
    for completed in range(1, config.newton_steps + 1):
        logits = (design @ weights).clamp(-30.0, 30.0)
        probabilities = logits.sigmoid()
        variance = (probabilities * (1.0 - probabilities)).clamp_min(1e-8)
        gradient = design.T @ (probabilities - targets) + penalty * weights
        hessian = design.T @ (variance[:, None] * design)
        hessian = hessian + torch.diag(penalty) + identity
        direction = torch.linalg.solve(hessian, gradient)
        before = _objective(design, targets, weights, config.l2)
        step_size = 1.0
        candidate = weights - direction
        for _ in range(16):
            candidate = weights - step_size * direction
            if bool(_objective(design, targets, candidate, config.l2) <= before + 1e-12):
                break
            step_size *= 0.5
        weights = candidate
        maximum_update = float(direction.abs().max() * step_size)
        if maximum_update < 1e-10:
            break

    model = ExplicitKeepMoveLinear(config)
    model.feature_mean.copy_(mean.float())
    model.feature_scale.copy_(scale.float())
    model.linear.bias.copy_(weights[:1].float())
    model.linear.weight.copy_(weights[1:].float()[None])
    logits = model(features.detach().cpu().float())
    accuracy = float(((logits >= 0.0) == labels.cpu()).float().mean())
    return model, {
        "samples": len(features),
        "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
        "newton_steps_completed": completed,
        "maximum_final_update": maximum_update,
        "training_accuracy": accuracy,
        "objective": float(_objective(design, targets, weights, config.l2)),
    }


__all__ = [
    "EXPLICIT_KEEP_MOVE_FEATURES",
    "ExplicitKeepMoveLinear",
    "ExplicitKeepMoveLinearConfig",
    "fit_explicit_keep_move_linear",
]
