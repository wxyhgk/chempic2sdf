"""Direction-aware ink evidence immediately inside and outside candidate endpoints."""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.inference_mode()
def endpoint_ray_features(
    pixel_values: torch.Tensor,
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    distances_pixels: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0, 11.0),
    band_offsets_pixels: tuple[float, ...] = (-2.0, 0.0, 2.0),
) -> torch.Tensor:
    """Return begin-in/out and end-in/out ink for every unordered pair."""

    if pixel_values.ndim != 4 or pixel_values.size(1) not in (1, 3):
        raise ValueError("pixel_values must be BCHW grayscale or RGB")
    if points.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate points and mask differ")
    if pair_indices.ndim != 2 or pair_indices.size(-1) != 2:
        raise ValueError("Pair indices must have shape [pairs, 2]")
    if not distances_pixels or not band_offsets_pixels:
        raise ValueError("Endpoint ray sampling grids must not be empty")
    if min(distances_pixels) <= 0:
        raise ValueError("Endpoint ray distances must be positive")

    begin_index = pair_indices[:, 0]
    end_index = pair_indices[:, 1]
    begin = points[:, begin_index].float()
    end = points[:, end_index].float()
    height, width = pixel_values.shape[-2:]
    delta_pixels = torch.stack(
        ((end[..., 0] - begin[..., 0]) * width, (end[..., 1] - begin[..., 1]) * height),
        dim=-1,
    )
    direction_pixels = delta_pixels / delta_pixels.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = torch.stack(
        (direction_pixels[..., 0] / width, direction_pixels[..., 1] / height),
        dim=-1,
    )
    perpendicular_pixels = torch.stack(
        (-direction_pixels[..., 1], direction_pixels[..., 0]), dim=-1
    )
    perpendicular = torch.stack(
        (
            perpendicular_pixels[..., 0] / width,
            perpendicular_pixels[..., 1] / height,
        ),
        dim=-1,
    )

    origins = torch.stack((begin, begin, end, end), dim=2)
    ray_directions = torch.stack(
        (direction, -direction, -direction, direction), dim=2
    )
    distances = torch.tensor(
        distances_pixels, device=points.device, dtype=torch.float32
    )
    offsets = torch.tensor(
        band_offsets_pixels, device=points.device, dtype=torch.float32
    )
    grid = (
        origins[:, :, :, None, None, :]
        + ray_directions[:, :, :, None, None, :] * distances[None, None, None, None, :, None]
        + perpendicular[:, :, None, None, None, :] * offsets[None, None, None, :, None, None]
    )
    batch_size, pair_count = points.size(0), pair_indices.size(0)
    grid = grid.mul(2.0).sub(1.0).reshape(
        batch_size,
        pair_count * 4 * len(band_offsets_pixels),
        len(distances_pixels),
        2,
    )
    if pixel_values.size(1) == 3:
        ink = 1.0 - pixel_values.add(1.0).mul(0.5).mean(dim=1, keepdim=True)
    else:
        ink = 1.0 - pixel_values.add(1.0).mul(0.5)
    sampled = F.grid_sample(
        ink,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(
        batch_size,
        pair_count,
        4,
        len(band_offsets_pixels),
        len(distances_pixels),
    )
    features = sampled.amax(dim=3).mean(dim=-1)
    pair_mask = candidate_mask[:, begin_index] & candidate_mask[:, end_index]
    return features.masked_fill(~pair_mask.unsqueeze(-1), 0.0)
