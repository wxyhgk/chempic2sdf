"""Classify one semantic bond object for every unordered atom pair."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES, DIRECTIONS


@dataclass(frozen=True, slots=True)
class NodeConditionedBondConfig:
    line_samples: int = 64
    line_band: int = 25
    line_band_radius_pixels: float = 12.0
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if min(self.line_samples, self.line_band, self.hidden_dim) < 1:
            raise ValueError("Bond reader dimensions must be positive")
        if self.line_band_radius_pixels <= 0:
            raise ValueError("Line-band radius must be positive")


def extract_line_bands(
    pixel_values: torch.Tensor,
    points: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor,
    config: NodeConditionedBondConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample oriented two-dimensional ink bands and their pair geometry."""

    if pixel_values.ndim != 4 or pixel_values.size(1) not in (1, 3):
        raise ValueError("pixel_values must be BCHW grayscale or RGB")
    if points.ndim != 3 or points.size(-1) != 2:
        raise ValueError("points must have shape [batch, points, 2]")
    if pair_indices.shape[:2] != pair_mask.shape or pair_indices.size(-1) != 2:
        raise ValueError("pair indices and mask shapes differ")

    batch_size, pair_count = pair_mask.shape
    gather = pair_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, 2)
    expanded = points[:, None].expand(-1, pair_count, -1, -1)
    endpoints = torch.gather(expanded, 2, gather)
    begin = endpoints[:, :, 0]
    delta = endpoints[:, :, 1] - begin
    height, width = pixel_values.shape[-2:]
    delta_pixels = torch.stack(
        (delta[..., 0] * width, delta[..., 1] * height), dim=-1
    )
    lengths = delta_pixels.norm(dim=-1).clamp_min(1e-6)
    direction = delta_pixels / lengths.unsqueeze(-1)
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    perpendicular = torch.stack(
        (perpendicular[..., 0] / width, perpendicular[..., 1] / height), dim=-1
    )

    positions = torch.linspace(
        0.04,
        0.96,
        config.line_samples,
        device=points.device,
        dtype=points.dtype,
    )
    offsets = torch.linspace(
        -config.line_band_radius_pixels,
        config.line_band_radius_pixels,
        config.line_band,
        device=points.device,
        dtype=points.dtype,
    )
    base = begin[:, :, None, :] + delta[:, :, None, :] * positions[None, None, :, None]
    grid = base[:, :, None, :, :] + (
        perpendicular[:, :, None, None, :]
        * offsets[None, None, :, None, None]
    )
    grid = grid.mul(2.0).sub(1.0).reshape(
        batch_size,
        pair_count * config.line_band,
        config.line_samples,
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
        config.line_band,
        config.line_samples,
    )
    selected = sampled[pair_mask].unsqueeze(1)
    geometry = torch.cat(
        (
            direction,
            lengths.unsqueeze(-1) / max(height, width),
            begin,
        ),
        dim=-1,
    )
    return selected, geometry[pair_mask]


def enumerate_all_pairs(point_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every unordered valid point pair without geometric filtering."""

    if point_mask.ndim != 2:
        raise ValueError("point_mask must have shape [batch, points]")
    pairs: list[torch.Tensor] = []
    maximum = 0
    for mask in point_mask:
        indices = torch.where(mask)[0]
        current = torch.combinations(indices, r=2)
        pairs.append(current)
        maximum = max(maximum, len(current))
    pair_indices = torch.zeros(
        (point_mask.size(0), maximum, 2),
        device=point_mask.device,
        dtype=torch.long,
    )
    pair_mask = torch.zeros(
        (point_mask.size(0), maximum),
        device=point_mask.device,
        dtype=torch.bool,
    )
    for batch_index, current in enumerate(pairs):
        pair_indices[batch_index, : len(current)] = current
        pair_mask[batch_index, : len(current)] = True
    return pair_indices, pair_mask


class NodeConditionedBondObject(nn.Module):
    """Read a wide 2-D line band and emit one object for an endpoint pair."""

    def __init__(self, config: NodeConditionedBondConfig | None = None) -> None:
        super().__init__()
        self.config = config or NodeConditionedBondConfig()
        self.band_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        self.shared = nn.Sequential(
            nn.Linear(48 * 4 * 8 + 32, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        self.edge = nn.Linear(self.config.hidden_dim, 1)
        self.display = nn.Linear(self.config.hidden_dim, len(BOND_TYPES))
        self.chemical = nn.Linear(self.config.hidden_dim, len(BOND_TYPES))
        self.direction = nn.Linear(self.config.hidden_dim, len(DIRECTIONS))

    def extract_line_bands(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_line_bands(
            pixel_values, points, pair_indices, pair_mask, self.config
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bands, geometry = self.extract_line_bands(
            pixel_values, points, pair_indices, pair_mask
        )
        hidden = self.shared(
            torch.cat((self.band_encoder(bands), self.geometry_encoder(geometry)), dim=-1)
        )
        return {
            "edge_logits": self.edge(hidden).squeeze(-1),
            "display_logits": self.display(hidden),
            "chemical_logits": self.chemical(hidden),
            "direction_logits": self.direction(hidden),
            "hidden": hidden,
            "line_bands": bands,
        }


def node_conditioned_bond_loss(
    output: dict[str, torch.Tensor],
    edge_targets: torch.Tensor,
    display_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    direction_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if output["edge_logits"].shape != edge_targets.shape:
        raise ValueError("Edge logits and targets differ")
    positives = edge_targets.bool()
    positive_count = positives.sum().clamp_min(1)
    negative_count = (~positives).sum().clamp_min(1)
    edge_loss = F.binary_cross_entropy_with_logits(
        output["edge_logits"],
        edge_targets.float(),
        pos_weight=(negative_count / positive_count).clamp_max(32.0),
    )
    zero = edge_loss * 0.0
    display_loss = (
        F.cross_entropy(output["display_logits"][positives], display_targets[positives])
        if positives.any()
        else zero
    )
    chemical_loss = (
        F.cross_entropy(output["chemical_logits"][positives], chemical_targets[positives])
        if positives.any()
        else zero
    )
    direction_loss = (
        F.cross_entropy(output["direction_logits"][positives], direction_targets[positives])
        if positives.any()
        else zero
    )
    return {
        "loss": edge_loss + display_loss + chemical_loss + 0.5 * direction_loss,
        "edge_loss": edge_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
        "direction_loss": direction_loss,
    }
