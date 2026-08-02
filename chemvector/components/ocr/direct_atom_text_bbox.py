"""Anchor-conditioned text bbox regression for selected AtomObjects."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DirectAtomTextBBoxConfig:
    radii_pixels: tuple[float, ...] = (16.0, 32.0, 48.0)
    patch_size: int = 32
    image_size: float = 768.0
    hidden_dim: int = 128
    center_radius_pixels: float = 64.0
    maximum_half_extent_pixels: float = 96.0
    initial_half_extent_pixels: float = 12.0

    def __post_init__(self) -> None:
        values = (
            *self.radii_pixels,
            self.patch_size,
            self.image_size,
            self.hidden_dim,
            self.center_radius_pixels,
            self.maximum_half_extent_pixels,
            self.initial_half_extent_pixels,
        )
        if not self.radii_pixels or min(values) <= 0:
            raise ValueError("Text bbox dimensions must be positive")


class DirectAtomTextBBox(nn.Module):
    def __init__(self, config: DirectAtomTextBBoxConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectAtomTextBBoxConfig()
        channels = 3 * len(self.config.radii_pixels)
        encoded_size = (self.config.patch_size + 7) // 8
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(96 * encoded_size * encoded_size, self.config.hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(self.config.hidden_dim, 4)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias[:2].zero_()
            self.output.bias[2:].fill_(math.log(self.config.initial_half_extent_pixels))

    def _sample_scale(
        self, pixels: torch.Tensor, anchors: torch.Tensor, radius: float
    ) -> torch.Tensor:
        offsets = torch.linspace(
            -radius / self.config.image_size,
            radius / self.config.image_size,
            self.config.patch_size,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        y, x = torch.meshgrid(offsets, offsets, indexing="ij")
        grid = anchors[:, None, None, :] + torch.stack((x, y), dim=-1)[None]
        grid = grid.mul(2.0).sub(1.0).reshape(
            1, len(anchors) * self.config.patch_size, self.config.patch_size, 2
        )
        sampled = F.grid_sample(
            pixels[None], grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(
            3, len(anchors), self.config.patch_size, self.config.patch_size
        ).permute(1, 0, 2, 3)

    def forward(
        self,
        pixels: torch.Tensor,
        anchors: torch.Tensor,
        atom_indices: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if atom_indices.shape != sample_indices.shape:
            raise ValueError("Text bbox indices have incompatible shapes")
        if not len(atom_indices):
            empty = pixels.new_empty((0, 2))
            return {"centers": empty, "half_extents": empty, "raw": pixels.new_empty((0, 4))}
        rows = []
        values = []
        selected_anchors = anchors[sample_indices, atom_indices]
        for sample_index in sample_indices.unique(sorted=True).tolist():
            selected = torch.where(sample_indices == sample_index)[0]
            centers = selected_anchors[selected]
            patches = torch.cat(
                [
                    self._sample_scale(pixels[sample_index], centers, radius)
                    for radius in self.config.radii_pixels
                ],
                dim=1,
            )
            rows.append(selected)
            values.append(self.output(self.encoder(patches)))
        order = torch.argsort(torch.cat(rows))
        raw = torch.cat(values)[order]
        offsets = raw[:, :2].tanh() * (
            self.config.center_radius_pixels / self.config.image_size
        )
        half_extents = raw[:, 2:].exp().clamp(
            1.0, self.config.maximum_half_extent_pixels
        ) / self.config.image_size
        return {
            "centers": (selected_anchors + offsets).clamp(0.0, 1.0),
            "half_extents": half_extents,
            "raw": raw,
        }


def direct_atom_text_bbox_loss(
    output: dict[str, torch.Tensor],
    anchors: torch.Tensor,
    target_bboxes: torch.Tensor,
    config: DirectAtomTextBBoxConfig,
) -> dict[str, torch.Tensor]:
    target_offsets = (
        (target_bboxes[:, :2] - anchors) * config.image_size / config.center_radius_pixels
    ).clamp(-1.0, 1.0)
    predicted_offsets = output["raw"][:, :2].tanh()
    target_log_extents = (
        target_bboxes[:, 2:].mul(config.image_size / 2.0).clamp_min(1.0).log()
    )
    center_loss = F.smooth_l1_loss(predicted_offsets, target_offsets, beta=0.05)
    extent_loss = F.smooth_l1_loss(output["raw"][:, 2:], target_log_extents, beta=0.1)
    return {
        "loss": center_loss + 0.5 * extent_loss,
        "center_loss": center_loss,
        "extent_loss": extent_loss,
    }


__all__ = [
    "DirectAtomTextBBox",
    "DirectAtomTextBBoxConfig",
    "direct_atom_text_bbox_loss",
]
