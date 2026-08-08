"""Dual-resolution 96px structural context for source-centered Atom recovery."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_SIZE = 96
LOCAL_SIZE = 32
IMAGE_SIZE = 768
LOCAL_START = (CONTEXT_SIZE - LOCAL_SIZE) // 2
LOCAL_END = LOCAL_START + LOCAL_SIZE


def anchor_centered_context_crops(
    values: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """Sample one source-centered 96x96 crop per normalized anchor."""

    if (
        values.ndim != 4
        or values.size(1) != 1
        or not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("P6 context source must be finite [batch,1,H,W]")
    if (
        anchors.ndim != 3
        or anchors.size(0) != values.size(0)
        or anchors.size(-1) != 2
        or not anchors.is_floating_point()
        or not bool(torch.isfinite(anchors).all())
        or bool(((anchors < 0.0) | (anchors > 1.0)).any())
    ):
        raise ValueError("P6 context anchors must be normalized [batch,count,2]")
    height, width = values.shape[-2:]
    axis = torch.arange(
        CONTEXT_SIZE, device=anchors.device, dtype=torch.float32
    ) - (CONTEXT_SIZE - 1) * 0.5
    y, x = torch.meshgrid(axis / height, axis / width, indexing="ij")
    offsets = torch.stack((x, y), dim=-1)
    grid = (anchors.float()[:, :, None, None] + offsets).mul(2.0).sub(1.0)
    outputs = []
    for batch_index in range(values.size(0)):
        count = anchors.size(1)
        current = F.grid_sample(
            values[batch_index : batch_index + 1].expand(count, -1, -1, -1),
            grid[batch_index],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        outputs.append(current)
    return torch.stack(outputs)


class _Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = 4 if output_channels % 4 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class StructuralContextPeakHead(nn.Module):
    """Encode all 96px while decoding only the source-centered 32px target area."""

    def __init__(self) -> None:
        super().__init__()
        axis = torch.arange(CONTEXT_SIZE, dtype=torch.float32)
        axis = axis - (CONTEXT_SIZE - 1) * 0.5
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((x, y), dim=0) / ((CONTEXT_SIZE - 1) * 0.5)
        self.register_buffer("relative_coordinates", coordinates, persistent=False)

        self.stem = _Block(4, 12)
        self.down1 = _Block(12, 16, stride=2)
        self.down2 = _Block(16, 24, stride=2)
        self.down3 = _Block(24, 32, stride=2)
        self.down4 = _Block(32, 40, stride=2)
        self.down5 = _Block(40, 48, stride=2)
        self.bottleneck = _Block(48, 48)
        self.candidate = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, 48),
            nn.GELU(),
            nn.Linear(48, 96),
        )
        self.up4 = _Block(48 + 40, 40)
        self.up3 = _Block(40 + 32, 32)
        self.up2 = _Block(32 + 24, 24)
        self.up1 = _Block(24 + 16, 16)
        self.up0 = _Block(16 + 12, 12)

        self.local = nn.Sequential(
            _Block(4, 16),
            _Block(16, 16),
        )
        self.merge = nn.Sequential(
            _Block(12 + 16, 24),
            nn.Conv2d(24, 1, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @staticmethod
    def _upsample(values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(values, size=skip.shape[-2:], mode="nearest")

    def forward(
        self,
        image_crops: torch.Tensor,
        score_crops: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(image_crops)
        expected = (batch, 1, CONTEXT_SIZE, CONTEXT_SIZE)
        if image_crops.shape != expected or score_crops.shape != expected:
            raise ValueError("P6 context crops must have shape [batch,1,96,96]")
        if candidate_scores.shape != (batch,) or candidate_ranks.shape != (batch,):
            raise ValueError("P6 candidate scalars must be batch vectors")
        values = image_crops, score_crops, candidate_scores, candidate_ranks
        if any(not value.is_floating_point() for value in values) or any(
            not bool(torch.isfinite(value).all()) for value in values
        ):
            raise ValueError("P6 inputs must be finite floating point")

        coordinates = self.relative_coordinates[None].expand(batch, -1, -1, -1)
        inputs = torch.cat((image_crops.float(), score_crops.float(), coordinates), dim=1)
        level0 = self.stem(inputs)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        level4 = self.down4(level3)
        level5 = self.bottleneck(self.down5(level4))
        scalars = torch.stack(
            (candidate_scores.float(), candidate_ranks.float()), dim=1
        )
        scale, bias = self.candidate(scalars).chunk(2, dim=1)
        level5 = level5 * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        decoded4 = self.up4(
            torch.cat((self._upsample(level5, level4), level4), dim=1)
        )
        decoded3 = self.up3(
            torch.cat((self._upsample(decoded4, level3), level3), dim=1)
        )
        decoded2 = self.up2(
            torch.cat((self._upsample(decoded3, level2), level2), dim=1)
        )
        decoded1 = self.up1(
            torch.cat((self._upsample(decoded2, level1), level1), dim=1)
        )
        decoded0 = self.up0(
            torch.cat((self._upsample(decoded1, level0), level0), dim=1)
        )
        context_center = decoded0[:, :, LOCAL_START:LOCAL_END, LOCAL_START:LOCAL_END]
        local_inputs = inputs[:, :, LOCAL_START:LOCAL_END, LOCAL_START:LOCAL_END]
        local = self.local(local_inputs)
        return self.merge(torch.cat((context_center, local), dim=1))[:, 0]


__all__ = [
    "CONTEXT_SIZE",
    "IMAGE_SIZE",
    "LOCAL_END",
    "LOCAL_SIZE",
    "LOCAL_START",
    "StructuralContextPeakHead",
    "anchor_centered_context_crops",
]
