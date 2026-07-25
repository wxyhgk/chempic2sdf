"""Spatially pooled pair-local residual scorer for Evidence Carrier v0.9.20."""

from __future__ import annotations

from typing import Any

from torch import nn

from chemvector.evidence_carrier_pair_local_residual import (
    EvidenceCarrierPairLocalResidual,
)


PAIR_LOCAL_SPATIAL_PARAMETER_COUNT = 37_313


class EvidenceCarrierPairLocalSpatialResidual(EvidenceCarrierPairLocalResidual):
    """Retain a coarse 3x6 spatial grid before producing the 24D local token."""

    model_type = "evidence-carrier-pair-local-spatial-residual-v0.1"

    def __init__(self) -> None:
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(24 * 3 * 6, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
        )
        if self.parameter_count() != PAIR_LOCAL_SPATIAL_PARAMETER_COUNT:
            raise RuntimeError(
                "pair-local spatial parameter contract changed: "
                f"{self.parameter_count()}"
            )

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        config.update(
            {
                "model_type": self.model_type,
                "local_encoder": [2, 8, 16, 24, "adaptive_3x6", 48, 24],
                "local_spatial_grid": [3, 6],
                "parameter_count": self.parameter_count(),
            }
        )
        return config


__all__ = [
    "PAIR_LOCAL_SPATIAL_PARAMETER_COUNT",
    "EvidenceCarrierPairLocalSpatialResidual",
]
