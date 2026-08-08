"""Spatial secondary-peak decoder for a forced P4 refine source."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SpatialSecondaryPeakConfig:
    crop_size: int = 32
    image_size: int = 768
    minimum_radius_pixels: float = 4.0
    maximum_radius_pixels: float = 13.0
    target_sigma_pixels: float = 1.5

    def __post_init__(self) -> None:
        if self.crop_size != 32:
            raise ValueError("P5-E1 requires 32x32 crops")
        if self.image_size != 768:
            raise ValueError("P5-E1 requires a 768px source canvas")
        if not (
            0.0 < self.minimum_radius_pixels < self.maximum_radius_pixels < 16.0
        ):
            raise ValueError("P5-E1 spatial support radii are invalid")
        if not math.isfinite(self.target_sigma_pixels) or self.target_sigma_pixels <= 0:
            raise ValueError("P5-E1 target sigma must be positive and finite")


def spatial_offset_grid(
    config: SpatialSecondaryPeakConfig = SpatialSecondaryPeakConfig(),
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    axis = torch.arange(config.crop_size, device=device, dtype=torch.float32)
    axis = axis - (config.crop_size - 1) * 0.5
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1)


def spatial_support_mask(
    config: SpatialSecondaryPeakConfig = SpatialSecondaryPeakConfig(),
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    radius = torch.linalg.vector_norm(spatial_offset_grid(config, device=device), dim=-1)
    return (radius >= config.minimum_radius_pixels) & (
        radius <= config.maximum_radius_pixels
    )


def secondary_peak_targets(
    offsets_normalized: torch.Tensor,
    config: SpatialSecondaryPeakConfig = SpatialSecondaryPeakConfig(),
) -> torch.Tensor:
    if offsets_normalized.ndim != 2 or offsets_normalized.size(-1) != 2:
        raise ValueError("P5-E1 offsets must have shape [batch,2]")
    if not offsets_normalized.is_floating_point() or not bool(
        torch.isfinite(offsets_normalized).all()
    ):
        raise ValueError("P5-E1 offsets must be finite floating point")
    offsets_pixels = offsets_normalized.float() * config.image_size
    radius = torch.linalg.vector_norm(offsets_pixels, dim=-1)
    if bool(
        (radius < 6.0 - 1e-3).any() | (radius > 12.0 + 1e-3).any()
    ):
        raise ValueError("P5-E1 refine targets must lie in the (6,12]px protocol")
    grid = spatial_offset_grid(config, device=offsets_normalized.device)
    squared = (grid[None] - offsets_pixels[:, None, None]).square().sum(dim=-1)
    logits = -0.5 * squared / (config.target_sigma_pixels**2)
    support = spatial_support_mask(config, device=offsets_normalized.device)
    logits = logits.masked_fill(~support[None], -torch.inf)
    return torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)


def secondary_peak_loss(
    logits: torch.Tensor,
    offsets_normalized: torch.Tensor,
    config: SpatialSecondaryPeakConfig = SpatialSecondaryPeakConfig(),
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (config.crop_size, config.crop_size):
        raise ValueError("P5-E1 logits must have shape [batch,32,32]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("P5-E1 logits must be finite floating point")
    targets = secondary_peak_targets(offsets_normalized, config)
    support = spatial_support_mask(config, device=logits.device)
    masked = logits.float().masked_fill(~support[None], -torch.inf)
    log_probabilities = F.log_softmax(masked.flatten(1), dim=1).reshape_as(masked)
    return -(targets * log_probabilities.masked_fill(~support[None], 0.0)).sum(dim=(1, 2)).mean()


@torch.no_grad()
def decode_secondary_peak_offsets(
    logits: torch.Tensor,
    config: SpatialSecondaryPeakConfig = SpatialSecondaryPeakConfig(),
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (config.crop_size, config.crop_size):
        raise ValueError("P5-E1 decode logits must have shape [batch,32,32]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("P5-E1 decode logits must be finite floating point")
    support = spatial_support_mask(config, device=logits.device)
    flat = logits.float().masked_fill(~support[None], -torch.inf).flatten(1)
    indices = flat.argmax(dim=1)
    grid = spatial_offset_grid(config, device=logits.device).reshape(-1, 2)
    return grid[indices] / config.image_size


def dihedral_transform(
    image_crops: torch.Tensor,
    score_crops: torch.Tensor,
    offsets_normalized: torch.Tensor,
    transform_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one of eight exact square symmetries and transform vector targets."""

    if image_crops.shape != score_crops.shape or image_crops.ndim != 4:
        raise ValueError("P5-E1 dihedral crops must have equal [batch,1,H,W] shapes")
    if offsets_normalized.shape != (len(image_crops), 2):
        raise ValueError("P5-E1 dihedral offsets differ from crops")
    if transform_ids.shape != (len(image_crops),) or transform_ids.dtype != torch.long:
        raise ValueError("P5-E1 transform IDs must be a long batch vector")
    if bool(((transform_ids < 0) | (transform_ids > 7)).any()):
        raise ValueError("P5-E1 transform IDs must lie in 0..7")
    images = []
    scores = []
    offsets = []
    for index, transform in enumerate(transform_ids.tolist()):
        rotation = transform % 4
        flip = transform >= 4
        image = torch.rot90(image_crops[index], rotation, dims=(-2, -1))
        score = torch.rot90(score_crops[index], rotation, dims=(-2, -1))
        x, y = offsets_normalized[index]
        for _ in range(rotation):
            x, y = y, -x
        if flip:
            image = torch.flip(image, dims=(-1,))
            score = torch.flip(score, dims=(-1,))
            x = -x
        images.append(image)
        scores.append(score)
        offsets.append(torch.stack((x, y)))
    return torch.stack(images), torch.stack(scores), torch.stack(offsets)


class SpatialSecondaryPeakHead(nn.Module):
    def __init__(
        self, config: SpatialSecondaryPeakConfig = SpatialSecondaryPeakConfig()
    ) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.candidate = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 64),
        )
        self.conditioned = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(
        self,
        image_crops: torch.Tensor,
        score_crops: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(image_crops)
        if image_crops.shape != (batch, 1, 32, 32) or score_crops.shape != image_crops.shape:
            raise ValueError("P5-E1 crops must have shape [batch,1,32,32]")
        if candidate_scores.shape != (batch,) or candidate_ranks.shape != (batch,):
            raise ValueError("P5-E1 candidate scalars must be batch vectors")
        values = (
            image_crops,
            score_crops,
            candidate_scores,
            candidate_ranks,
        )
        if any(not value.is_floating_point() for value in values) or any(
            not bool(torch.isfinite(value).all()) for value in values
        ):
            raise ValueError("P5-E1 inputs must be finite floating point")
        spatial = self.stem(torch.cat((image_crops, score_crops), dim=1).float())
        scalar = torch.cat(
            (
                candidate_scores.float()[:, None],
                candidate_ranks.float()[:, None],
            ),
            dim=1,
        )
        film = self.candidate(scalar)
        scale, bias = film.chunk(2, dim=1)
        spatial = spatial * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        return self.conditioned(spatial)[:, 0]


__all__ = [
    "SpatialSecondaryPeakConfig",
    "SpatialSecondaryPeakHead",
    "decode_secondary_peak_offsets",
    "dihedral_transform",
    "secondary_peak_loss",
    "secondary_peak_targets",
    "spatial_offset_grid",
    "spatial_support_mask",
]
