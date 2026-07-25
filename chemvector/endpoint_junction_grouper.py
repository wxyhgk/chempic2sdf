"""Predict whether two semantic Bond endpoints belong to the same Junction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class EndpointJunctionGrouperConfig:
    input_dim: int = 15
    hidden_dim: int = 64


class EndpointJunctionGrouper(nn.Module):
    def __init__(self, config: EndpointJunctionGrouperConfig | None = None) -> None:
        super().__init__()
        self.config = config or EndpointJunctionGrouperConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != self.config.input_dim:
            raise ValueError("Endpoint pair features have an invalid shape")
        return self.network(features).squeeze(-1)


@dataclass(frozen=True, slots=True)
class VisualEndpointJunctionGrouperConfig:
    geometry_dim: int = 15
    patch_embedding_dim: int = 32
    hidden_dim: int = 96
    base_channels: int = 16


class VisualEndpointJunctionGrouper(nn.Module):
    """Classify same-Junction pairs from geometry and aligned endpoint ink."""

    def __init__(
        self,
        config: VisualEndpointJunctionGrouperConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or VisualEndpointJunctionGrouperConfig()
        channels = self.config.base_channels
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, channels * 2),
            nn.GELU(),
            nn.Conv2d(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 4, self.config.patch_embedding_dim),
            nn.GELU(),
        )
        visual_dim = self.config.patch_embedding_dim * 3
        self.network = nn.Sequential(
            nn.Linear(self.config.geometry_dim + visual_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.size(1) != 1:
            raise ValueError("Endpoint patches have an invalid shape")
        return self.patch_encoder(patches.float())

    def score_embeddings(
        self,
        geometry: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        if geometry.ndim != 2 or geometry.size(-1) != self.config.geometry_dim:
            raise ValueError("Endpoint pair geometry has an invalid shape")
        expected = (len(geometry), self.config.patch_embedding_dim)
        if first.shape != expected or second.shape != expected:
            raise ValueError("Endpoint patch embeddings have an invalid shape")
        visual = torch.cat(
            (first + second, (first - second).abs(), first * second), dim=-1
        )
        return self.network(torch.cat((geometry.float(), visual), dim=-1)).squeeze(-1)

    def forward(
        self,
        geometry: torch.Tensor,
        first_patches: torch.Tensor,
        second_patches: torch.Tensor,
    ) -> torch.Tensor:
        if first_patches.shape != second_patches.shape or len(first_patches) != len(geometry):
            raise ValueError("Endpoint patches and pair geometry differ")
        return self.score_embeddings(
            geometry,
            self.encode_patches(first_patches),
            self.encode_patches(second_patches),
        )


def sample_aligned_endpoint_ink_patches(
    pixel_values: torch.Tensor,
    bond_endpoints: torch.Tensor,
    *,
    patch_size: int = 32,
) -> torch.Tensor:
    """Crop endpoint patches with each Bond ray aligned toward image-right.

    ``bond_endpoints`` contains exactly two normalized semantic endpoints for
    every Bond Object. The aligned view makes line termination and line
    continuation comparable across arbitrary Bond rotations.
    """

    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 4 or pixel_values.size(0) != 1:
        raise ValueError("Endpoint patch sampling expects one image")
    if pixel_values.size(1) not in (1, 3):
        raise ValueError("Pixel values must have one or three channels")
    if bond_endpoints.ndim != 3 or bond_endpoints.shape[1:] != (2, 2):
        raise ValueError("Bond endpoints must have shape [bonds, 2, 2]")
    if patch_size < 4 or patch_size % 2:
        raise ValueError("Endpoint patch size must be an even integer of at least four")
    if not len(bond_endpoints):
        return torch.empty(
            (0, 1, patch_size, patch_size),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )

    endpoints = bond_endpoints.to(device=pixel_values.device, dtype=torch.float32)
    flat = endpoints.reshape(-1, 2)
    outgoing = torch.stack(
        (endpoints[:, 1] - endpoints[:, 0], endpoints[:, 0] - endpoints[:, 1]),
        dim=1,
    ).reshape(-1, 2)
    units = outgoing / torch.linalg.vector_norm(
        outgoing, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    perpendicular = torch.stack((-units[:, 1], units[:, 0]), dim=-1)

    height, width = pixel_values.shape[-2:]
    coordinates = torch.arange(
        patch_size, device=pixel_values.device, dtype=torch.float32
    )
    offset_y, offset_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    center = (patch_size - 1) / 2.0
    offset_x = offset_x - center
    offset_y = offset_y - center
    pixel_offsets = (
        offset_x[None, :, :, None] * units[:, None, None, :]
        + offset_y[None, :, :, None] * perpendicular[:, None, None, :]
    )
    scale = torch.tensor(
        (float(width), float(height)),
        device=pixel_values.device,
        dtype=torch.float32,
    )
    grid = (flat[:, None, None, :] + pixel_offsets / scale).mul(2.0).sub(1.0)
    gray = pixel_values.float().mean(dim=1, keepdim=True)
    ink = ((1.0 - gray) * 0.5).clamp(0.0, 1.0)
    return F.grid_sample(
        ink.expand(len(flat), -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def endpoint_pair_features(
    bond_objects: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bond_objects:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0, 15))
    endpoints = torch.stack(
        [item["endpoints"].detach().float().cpu() for item in bond_objects]
    )
    scores = torch.tensor([float(item["score"]) for item in bond_objects])
    flat = endpoints.reshape(-1, 2)
    pairs = torch.combinations(torch.arange(len(flat)), r=2)
    pairs = pairs[pairs[:, 0].div(2, rounding_mode="floor") != pairs[:, 1].div(2, rounding_mode="floor")]
    if not len(pairs):
        return pairs, torch.empty((0, 15))
    object_ids = torch.arange(len(flat)).div(2, rounding_mode="floor")
    endpoint_sides = torch.arange(len(flat)).remainder(2)
    other_indices = object_ids * 2 + (1 - endpoint_sides)
    outgoing = flat[other_indices] - flat
    lengths = torch.linalg.vector_norm(outgoing, dim=-1).clamp_min(1e-8)
    units = outgoing / lengths[:, None]
    first, second = pairs[:, 0], pairs[:, 1]
    delta = flat[second] - flat[first]
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    dot = (units[first] * units[second]).sum(-1, keepdim=True)
    cross = (
        units[first, 0] * units[second, 1]
        - units[first, 1] * units[second, 0]
    )[:, None]
    features = torch.cat(
        (
            delta * 8.0,
            delta.abs() * 8.0,
            distance * 8.0,
            units[first],
            units[second],
            dot,
            cross,
            lengths[first, None] * 4.0,
            lengths[second, None] * 4.0,
            scores[object_ids[first], None],
            scores[object_ids[second], None],
        ),
        dim=-1,
    )
    return pairs, features


@torch.no_grad()
def decode_endpoint_groups(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    endpoint_count: int,
    *,
    threshold: float,
    linkage: str = "complete",
) -> torch.Tensor:
    if logits.shape != (len(pairs),) or pairs.shape != (len(logits), 2):
        raise ValueError("Endpoint grouping logits and pairs differ")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Endpoint grouping threshold must be in [0, 1]")
    if linkage not in {"complete", "single"}:
        raise ValueError(f"Unknown endpoint grouping linkage: {linkage}")
    probabilities = logits.sigmoid().cpu()
    pair_probability = torch.zeros((endpoint_count, endpoint_count))
    pair_probability[pairs[:, 0], pairs[:, 1]] = probabilities
    pair_probability[pairs[:, 1], pairs[:, 0]] = probabilities
    groups = [{index} for index in range(endpoint_count)]
    endpoint_group = list(range(endpoint_count))
    for pair_index in probabilities.argsort(descending=True).tolist():
        if float(probabilities[pair_index]) < threshold:
            break
        first, second = (int(value) for value in pairs[pair_index].tolist())
        first_group = endpoint_group[first]
        second_group = endpoint_group[second]
        if first_group == second_group:
            continue
        left = groups[first_group]
        right = groups[second_group]
        if {item // 2 for item in left} & {item // 2 for item in right}:
            continue
        if linkage == "complete":
            cross_probabilities = torch.tensor(
                [pair_probability[a, b] for a in left for b in right]
            )
            if len(cross_probabilities) and bool(
                (cross_probabilities < threshold).any()
            ):
                continue
        merged = left | right
        groups[first_group] = merged
        groups[second_group] = set()
        for endpoint in merged:
            endpoint_group[endpoint] = first_group
    remap = {}
    result = []
    for group in endpoint_group:
        if group not in remap:
            remap[group] = len(remap)
        result.append(remap[group])
    return torch.tensor(result, dtype=torch.long)
