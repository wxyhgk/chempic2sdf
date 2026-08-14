"""Differentiable separation of chemical bond ink and text ink."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SemanticInkSeparatorConfig:
    base_channels: int = 32
    bond_positive_weight: float = 8.0
    text_positive_weight: float = 20.0
    dice_weight: float = 2.0
    reconstruction_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.base_channels < 8:
            raise ValueError("Semantic ink base channels must be at least 8")
        if min(self.bond_positive_weight, self.text_positive_weight) <= 0:
            raise ValueError("Semantic ink positive weights must be positive")
        if min(self.dice_weight, self.reconstruction_weight) < 0:
            raise ValueError("Semantic ink auxiliary weights must be nonnegative")


class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 1
        self.input = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        return F.gelu(hidden + self.block(hidden))


class SemanticInkSeparator(nn.Module):
    """Small U-Net whose two output channels may overlap at label boundaries."""

    def __init__(
        self, config: SemanticInkSeparatorConfig = SemanticInkSeparatorConfig()
    ) -> None:
        super().__init__()
        self.config = config
        channels = config.base_channels
        self.encoder1 = ResidualConvBlock(3, channels)
        self.encoder2 = ResidualConvBlock(channels, channels * 2)
        self.encoder3 = ResidualConvBlock(channels * 2, channels * 3)
        self.bottleneck = ResidualConvBlock(channels * 3, channels * 4)
        self.decoder3 = ResidualConvBlock(channels * 7, channels * 3)
        self.decoder2 = ResidualConvBlock(channels * 5, channels * 2)
        self.decoder1 = ResidualConvBlock(channels * 3, channels)
        self.full_resolution = ResidualConvBlock(channels + 3, channels)
        self.head = nn.Conv2d(channels, 2, kernel_size=1)
        nn.init.constant_(self.head.bias, -3.0)

    @staticmethod
    def _down(inputs: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(inputs, kernel_size=2, stride=2)

    @staticmethod
    def _up(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4 or pixel_values.size(1) != 3:
            raise ValueError("Semantic ink input must have shape [batch, 3, H, W]")
        half = self.encoder1(self._down(pixel_values))
        quarter = self.encoder2(self._down(half))
        eighth = self.encoder3(self._down(quarter))
        sixteenth = self.bottleneck(self._down(eighth))
        eighth_out = self.decoder3(
            torch.cat((self._up(sixteenth, eighth), eighth), dim=1)
        )
        quarter_out = self.decoder2(
            torch.cat((self._up(eighth_out, quarter), quarter), dim=1)
        )
        half_out = self.decoder1(
            torch.cat((self._up(quarter_out, half), half), dim=1)
        )
        full = self.full_resolution(
            torch.cat((self._up(half_out, pixel_values), pixel_values), dim=1)
        )
        logits = self.head(full)
        return {
            "logits": logits,
            "bond_logits": logits[:, 0],
            "text_logits": logits[:, 1],
            "features": full,
        }


def _soft_dice_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (probabilities * targets).sum(dim=(-1, -2)) + 1.0
    denominator = (
        probabilities.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2)) + 1.0
    )
    return (1.0 - numerator / denominator).mean()


def semantic_ink_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: SemanticInkSeparatorConfig,
) -> dict[str, torch.Tensor]:
    if targets.ndim != 4 or targets.size(1) != 2:
        raise ValueError("Semantic ink targets must have shape [batch, 2, H, W]")
    logits = output["logits"]
    if logits.shape != targets.shape:
        raise ValueError("Semantic ink output and target shapes differ")
    logits = logits.float()
    targets = targets.float().clamp(0.0, 1.0)
    positive_weights = logits.new_tensor(
        (config.bond_positive_weight, config.text_positive_weight)
    ).view(1, 2, 1, 1)
    weighted_bce = -(
        positive_weights * targets * F.logsigmoid(logits)
        + (1.0 - targets) * F.logsigmoid(-logits)
    ).mean()
    probabilities = logits.sigmoid()
    dice = _soft_dice_loss(probabilities, targets)
    predicted_union = 1.0 - (1.0 - probabilities[:, 0]) * (
        1.0 - probabilities[:, 1]
    )
    target_union = targets.amax(dim=1)
    predicted_union = predicted_union.clamp(1e-5, 1.0 - 1e-5)
    reconstruction = -(
        target_union * predicted_union.log()
        + (1.0 - target_union) * torch.log1p(-predicted_union)
    ).mean()
    total = (
        weighted_bce
        + config.dice_weight * dice
        + config.reconstruction_weight * reconstruction
    )
    return {
        "loss": total,
        "weighted_bce": weighted_bce,
        "dice_loss": dice,
        "reconstruction_loss": reconstruction,
    }
