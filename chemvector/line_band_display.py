"""Visual primitive classifier for one confirmed bond line band."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chemvector.bond_object import BOND_TYPES
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    extract_line_bands,
)


@dataclass(frozen=True, slots=True)
class LineBandDisplayConfig:
    line_samples: int = 96
    line_band: int = 33
    line_band_radius_pixels: float = 16.0
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        if min(self.line_samples, self.line_band, self.hidden_dim) < 1:
            raise ValueError("Line-band display dimensions must be positive")
        if self.line_band_radius_pixels <= 0:
            raise ValueError("Line-band display radius must be positive")

    def sampling_config(self) -> NodeConditionedBondConfig:
        return NodeConditionedBondConfig(
            line_samples=self.line_samples,
            line_band=self.line_band,
            line_band_radius_pixels=self.line_band_radius_pixels,
            hidden_dim=self.hidden_dim,
        )


class LineBandDisplayRefiner(nn.Module):
    """Classify visible stroke multiplicity without graph or absolute-position cues."""

    def __init__(self, config: LineBandDisplayConfig | None = None) -> None:
        super().__init__()
        self.config = config or LineBandDisplayConfig()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((5, 12)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 12, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, len(BOND_TYPES)),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bands, _ = extract_line_bands(
            pixel_values,
            points,
            pair_indices,
            pair_mask,
            self.config.sampling_config(),
        )
        return {
            "display_logits": self.classifier(self.encoder(bands)),
            "line_bands": bands,
        }
