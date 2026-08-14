"""Local endpoint-set detection from a frozen semantic bond-ink map."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.vision.vertex import focal_heatmap_loss
from chemvector.components.bonds.objects.semantic_bond_object import ResidualInkBlock


@dataclass(frozen=True, slots=True)
class SemanticBondEndpointConfig:
    stride4_channels: int = 64
    stride8_channels: int = 96
    stride16_channels: int = 128
    hidden_channels: int = 96
    slots: int = 3

    def __post_init__(self) -> None:
        if min(
            self.stride4_channels,
            self.stride8_channels,
            self.stride16_channels,
            self.hidden_channels,
            self.slots,
        ) < 1:
            raise ValueError("Semantic Bond endpoint dimensions must be positive")


class SemanticBondEndpointDetector(nn.Module):
    """Predict an unordered set of visible Bond ports using only local offsets."""

    def __init__(
        self, config: SemanticBondEndpointConfig = SemanticBondEndpointConfig()
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
        self.from_stride8 = nn.Conv2d(
            config.stride8_channels, config.stride4_channels, kernel_size=1
        )
        self.from_stride16 = nn.Conv2d(
            config.stride16_channels, config.stride4_channels, kernel_size=1
        )
        self.endpoint_features = nn.Sequential(
            nn.Conv2d(
                config.stride4_channels,
                config.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, config.hidden_channels),
            nn.GELU(),
            ResidualInkBlock(config.hidden_channels, 1),
            ResidualInkBlock(config.hidden_channels, 2),
            ResidualInkBlock(config.hidden_channels, 4),
        )
        self.heatmap = nn.Conv2d(config.hidden_channels, config.slots, 1)
        self.offset = nn.Conv2d(config.hidden_channels, config.slots * 2, 1)
        nn.init.constant_(self.heatmap.bias, -4.6)

    def forward(self, bond_probability: torch.Tensor) -> dict[str, torch.Tensor]:
        if bond_probability.ndim != 4 or bond_probability.size(1) != 1:
            raise ValueError("Semantic Bond endpoint input must be [batch, 1, H, W]")
        stride2 = self.stride2(bond_probability)
        stride4 = self.stride4(stride2)
        stride8 = self.stride8(stride4)
        stride16 = self.stride16(stride8)
        size = stride4.shape[-2:]
        fused = stride4 + F.interpolate(
            self.from_stride8(stride8), size=size, mode="bilinear", align_corners=False
        ) + F.interpolate(
            self.from_stride16(stride16),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        features = self.endpoint_features(fused)
        batch, _, height, width = features.shape
        offsets = self.offset(features).reshape(
            batch, self.config.slots, 2, height, width
        ).sigmoid()
        y, x = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=features.dtype),
            torch.arange(width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        points = torch.stack(
            (
                (x[None, None] + offsets[:, :, 0]) / width,
                (y[None, None] + offsets[:, :, 1]) / height,
            ),
            dim=2,
        )
        return {
            "features": features,
            "heatmap_logits": self.heatmap(features),
            "offsets": offsets,
            "points": points,
        }


def _unique_ports(endpoints: torch.Tensor) -> torch.Tensor:
    if endpoints.ndim != 3 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("Bond endpoints must have shape [bonds, 2, 2]")
    if not len(endpoints):
        return endpoints.new_empty((0, 2))
    quantized = (endpoints.reshape(-1, 2) * 1023.0).round().long()
    unique = torch.unique(quantized, dim=0, sorted=True)
    return unique.to(endpoints.dtype) / 1023.0


def build_semantic_endpoint_targets(
    endpoints: torch.Tensor,
    mask: torch.Tensor,
    *,
    height: int,
    width: int,
    slots: int,
) -> dict[str, torch.Tensor]:
    if endpoints.ndim != 4 or endpoints.shape[-2:] != (2, 2):
        raise ValueError("Endpoint targets must have shape [batch, bonds, 2, 2]")
    if mask.shape != endpoints.shape[:2]:
        raise ValueError("Endpoint target mask shape differs")
    if min(height, width, slots) < 1:
        raise ValueError("Endpoint target grid dimensions must be positive")
    heatmap = endpoints.new_zeros((endpoints.size(0), slots, height, width))
    offsets = endpoints.new_zeros((endpoints.size(0), slots, 2, height, width))
    points = endpoints.new_zeros((endpoints.size(0), slots, 2, height, width))
    positive = torch.zeros_like(heatmap, dtype=torch.bool)
    occupancy = torch.zeros(
        (endpoints.size(0), height, width),
        device=endpoints.device,
        dtype=torch.long,
    )
    unique_counts = torch.zeros(
        endpoints.size(0), device=endpoints.device, dtype=torch.long
    )
    for batch_index in range(endpoints.size(0)):
        unique = _unique_ports(endpoints[batch_index, mask[batch_index]])
        unique_counts[batch_index] = len(unique)
        cells: dict[tuple[int, int], list[torch.Tensor]] = {}
        for point in unique:
            cell_x = min(width - 1, max(0, int(float(point[0]) * width)))
            cell_y = min(height - 1, max(0, int(float(point[1]) * height)))
            cells.setdefault((cell_y, cell_x), []).append(point)
        for (cell_y, cell_x), cell_points in cells.items():
            cell_points.sort(key=lambda point: (float(point[0]), float(point[1])))
            if len(cell_points) > slots:
                raise ValueError(
                    f"Semantic endpoint slot capacity {slots} is smaller than "
                    f"{len(cell_points)} ports in cell ({cell_x}, {cell_y})"
                )
            occupancy[batch_index, cell_y, cell_x] = len(cell_points)
            for slot, point in enumerate(cell_points):
                heatmap[batch_index, slot, cell_y, cell_x] = 1.0
                positive[batch_index, slot, cell_y, cell_x] = True
                points[batch_index, slot, :, cell_y, cell_x] = point
                offsets[batch_index, slot, :, cell_y, cell_x] = torch.stack(
                    (point[0] * width - cell_x, point[1] * height - cell_y)
                )
    return {
        "heatmap": heatmap,
        "offsets": offsets,
        "points": points,
        "positive": positive,
        "occupancy": occupancy,
        "unique_counts": unique_counts,
    }


@lru_cache(maxsize=None)
def _slot_permutations(slots: int, targets: int) -> torch.Tensor:
    if min(slots, targets) < 1 or targets > slots:
        raise ValueError("Endpoint slot assignment dimensions are invalid")
    return torch.tensor(tuple(permutations(range(slots), targets)), dtype=torch.long)


def align_semantic_endpoint_targets(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Match same-cell port targets to unordered endpoint slots."""

    slots = int(output["heatmap_logits"].size(1))
    aligned = {
        "heatmap": torch.zeros_like(targets["heatmap"]),
        "offsets": torch.zeros_like(targets["offsets"]),
        "points": torch.zeros_like(targets["points"]),
        "positive": torch.zeros_like(targets["positive"]),
        "occupancy": targets["occupancy"],
        "unique_counts": targets["unique_counts"],
    }
    occupied = torch.where(targets["occupancy"] > 0)
    for batch_index, cell_y, cell_x in zip(*(item.tolist() for item in occupied)):
        count = int(targets["occupancy"][batch_index, cell_y, cell_x])
        expected = targets["offsets"][batch_index, :count, :, cell_y, cell_x]
        predicted = output["offsets"][batch_index, :, :, cell_y, cell_x]
        cost = (predicted[:, None].float() - expected[None].float()).abs().sum(-1)
        assignments = _slot_permutations(slots, count).to(cost.device)
        target_indices = torch.arange(count, device=cost.device)
        selected = assignments[cost[assignments, target_indices].sum(-1).argmin()]
        aligned["heatmap"][batch_index, selected, cell_y, cell_x] = 1.0
        aligned["positive"][batch_index, selected, cell_y, cell_x] = True
        for target_index, slot in enumerate(selected.tolist()):
            for name in ("offsets", "points"):
                aligned[name][batch_index, slot, :, cell_y, cell_x] = targets[name][
                    batch_index, target_index, :, cell_y, cell_x
                ]
    return aligned


def semantic_endpoint_loss(
    output: dict[str, torch.Tensor],
    endpoints: torch.Tensor,
    mask: torch.Tensor,
    *,
    heatmap_set_weight: float = 0.1,
    cell_count_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    height, width = output["heatmap_logits"].shape[-2:]
    targets = build_semantic_endpoint_targets(
        endpoints,
        mask,
        height=height,
        width=width,
        slots=output["heatmap_logits"].size(1),
    )
    targets = align_semantic_endpoint_targets(output, targets)
    positive = targets["positive"]
    heatmap_loss = focal_heatmap_loss(output["heatmap_logits"], targets["heatmap"])
    zero = heatmap_loss * 0.0
    expanded = F.max_pool2d(positive.float(), 3, stride=1, padding=1).bool()
    rank_losses = []
    for batch_index in range(positive.size(0)):
        positive_logits = output["heatmap_logits"][batch_index][positive[batch_index]]
        negative_logits = output["heatmap_logits"][batch_index][~expanded[batch_index]]
        if len(positive_logits) and len(negative_logits):
            rank_losses.append(
                F.relu(1.0 - positive_logits.min() + negative_logits.max())
            )
    heatmap_set_loss = torch.stack(rank_losses).mean() if rank_losses else zero
    occupied = targets["occupancy"] > 0
    cell_count_loss = (
        F.smooth_l1_loss(
            output["heatmap_logits"].float().sigmoid().sum(1)[occupied],
            targets["occupancy"][occupied].float(),
        )
        if bool(occupied.any())
        else zero
    )
    if bool(positive.any()):
        predicted_offsets = output["offsets"].permute(0, 1, 3, 4, 2)[positive]
        expected_offsets = targets["offsets"].permute(0, 1, 3, 4, 2)[positive]
        offset_loss = F.smooth_l1_loss(predicted_offsets.float(), expected_offsets.float())
    else:
        offset_loss = zero
    return {
        "loss": (
            heatmap_loss
            + heatmap_set_weight * heatmap_set_loss
            + cell_count_weight * cell_count_loss
            + 4.0 * offset_loss
        ),
        "heatmap_loss": heatmap_loss,
        "heatmap_set_loss": heatmap_set_loss,
        "cell_count_loss": cell_count_loss,
        "offset_loss": offset_loss,
    }


@torch.no_grad()
def decode_semantic_endpoints(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.0,
    maximum_points: int = 256,
) -> list[list[dict[str, torch.Tensor | float]]]:
    if maximum_points < 1:
        raise ValueError("Maximum endpoint count must be positive")
    scores = output["heatmap_logits"].float().sigmoid()
    peaks = scores == F.max_pool2d(scores, 3, stride=1, padding=1)
    scores = scores * peaks
    batch, slots, height, width = scores.shape
    spatial_size = height * width
    decoded = []
    flattened_points = output["points"].permute(0, 1, 3, 4, 2).reshape(
        batch, slots * spatial_size, 2
    )
    for batch_index in range(batch):
        values, indices = scores[batch_index].flatten().topk(
            min(maximum_points, slots * spatial_size)
        )
        valid = values >= threshold
        values = values[valid]
        indices = indices[valid]
        decoded.append(
            [
                {"score": float(score), "point": point.detach().float().cpu()}
                for score, point in zip(
                    values.detach().float().cpu().tolist(),
                    flattened_points[batch_index, indices],
                )
            ]
        )
    return decoded
