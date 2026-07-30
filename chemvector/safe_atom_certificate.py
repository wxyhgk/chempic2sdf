"""Independent unsafe-action certificate for directional Atom deletion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.safe_atom_action_model import ACTION_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class UnsafeActionCertificateConfig:
    input_dim: int = ACTION_FEATURE_DIM
    hidden_dim: int = 128
    bottleneck_dim: int = 64

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.bottleneck_dim) < 1:
            raise ValueError("Unsafe certificate dimensions must be positive")


class UnsafeActionCertificate(nn.Module):
    def __init__(self, config: UnsafeActionCertificateConfig | None = None) -> None:
        super().__init__()
        self.config = config or UnsafeActionCertificateConfig()
        self.network = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.bottleneck_dim),
            nn.GELU(),
            nn.Linear(self.config.bottleneck_dim, 1),
        )
        nn.init.constant_(self.network[-1].bias, 2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.config.input_dim:
            raise ValueError("Unsafe certificate features have an invalid shape")
        return self.network(features).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def unsafe_certificate_loss(
    logits: torch.Tensor,
    unsafe_targets: torch.Tensor,
    *,
    unsafe_weight: float = 8.0,
) -> torch.Tensor:
    if logits.shape != unsafe_targets.shape:
        raise ValueError("Unsafe certificate target shape differs")
    if unsafe_weight < 1.0:
        raise ValueError("Unsafe certificate weight must be at least one")
    weights = torch.where(
        unsafe_targets,
        logits.new_full(logits.shape, unsafe_weight),
        logits.new_ones(logits.shape),
    )
    return F.binary_cross_entropy_with_logits(
        logits.float(), unsafe_targets.float(), weight=weights
    )


def molecule_fold(molecule_id: str, *, folds: int = 5) -> int:
    if folds < 2:
        raise ValueError("At least two molecule folds are required")
    digest = hashlib.sha256(molecule_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def strict_unsafe_threshold(unsafe_probabilities: torch.Tensor) -> float:
    if unsafe_probabilities.ndim != 1 or not len(unsafe_probabilities):
        raise ValueError("Unsafe calibration probabilities must be non-empty")
    minimum = unsafe_probabilities.float().min()
    threshold = torch.nextafter(minimum, minimum.new_zeros(()))
    return float(threshold)


__all__ = [
    "UnsafeActionCertificate",
    "UnsafeActionCertificateConfig",
    "molecule_fold",
    "strict_unsafe_threshold",
    "unsafe_certificate_loss",
]
