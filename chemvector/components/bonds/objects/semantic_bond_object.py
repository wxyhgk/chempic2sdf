"""BondObject detection from a frozen semantic bond-ink probability map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chemvector.components.bonds.objects.dense_bond_object import (
    DenseBondObjectConfig,
    MultiScaleDenseBondObjectDetector,
)


@dataclass(frozen=True, slots=True)
class SemanticBondObjectConfig:
    stride4_channels: int = 64
    stride8_channels: int = 96
    stride16_channels: int = 128
    hidden_channels: int = 128
    slots: int = 3
    max_endpoint_offset: float = 0.5

    def __post_init__(self) -> None:
        if min(
            self.stride4_channels,
            self.stride8_channels,
            self.stride16_channels,
            self.hidden_channels,
            self.slots,
        ) < 1:
            raise ValueError("Semantic BondObject dimensions must be positive")
        if not 0.0 < self.max_endpoint_offset <= 1.0:
            raise ValueError("Semantic BondObject endpoint offset is invalid")


class ResidualInkBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class SemanticBondObjectDetector(nn.Module):
    """Encode only bond ink, then emit one dense object for each chemical bond."""

    def __init__(
        self, config: SemanticBondObjectConfig = SemanticBondObjectConfig()
    ) -> None:
        super().__init__()
        self.config = config
        self.stride2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualInkBlock(32),
        )
        self.stride4 = nn.Sequential(
            nn.Conv2d(32, config.stride4_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, config.stride4_channels),
            nn.GELU(),
            ResidualInkBlock(config.stride4_channels, 1),
            ResidualInkBlock(config.stride4_channels, 2),
        )
        self.stride8 = nn.Sequential(
            nn.Conv2d(
                config.stride4_channels,
                config.stride8_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, config.stride8_channels),
            nn.GELU(),
            ResidualInkBlock(config.stride8_channels, 1),
            ResidualInkBlock(config.stride8_channels, 2),
        )
        self.stride16 = nn.Sequential(
            nn.Conv2d(
                config.stride8_channels,
                config.stride16_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, config.stride16_channels),
            nn.GELU(),
            ResidualInkBlock(config.stride16_channels, 1),
            ResidualInkBlock(config.stride16_channels, 2),
        )
        dense_config = DenseBondObjectConfig(
            input_channels=config.stride4_channels,
            hidden_channels=config.hidden_channels,
            slots=config.slots,
            max_endpoint_offset=config.max_endpoint_offset,
        )
        self.detector = MultiScaleDenseBondObjectDetector(
            dense_config,
            stride8_channels=config.stride8_channels,
            stride16_channels=config.stride16_channels,
        )

    def forward(self, bond_probability: torch.Tensor) -> dict[str, torch.Tensor]:
        if bond_probability.ndim != 4 or bond_probability.size(1) != 1:
            raise ValueError("Semantic BondObject input must be [batch, 1, H, W]")
        stride2 = self.stride2(bond_probability)
        stride4 = self.stride4(stride2)
        stride8 = self.stride8(stride4)
        stride16 = self.stride16(stride8)
        return self.detector(stride4, stride8, stride16)
