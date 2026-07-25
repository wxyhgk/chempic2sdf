"""Semantic bond classification from one canonical crop per endpoint pair."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES, BOND_TYPE_TO_ID


@dataclass(frozen=True, slots=True)
class SemanticBondRelationConfig:
    crop_width: int = 192
    local_height: int = 64
    context_height: int = 128
    local_radius_pixels: float = 24.0
    context_radius_pixels: float = 72.0
    longitudinal_margin_pixels: float = 40.0
    hidden_dim: int = 192

    def __post_init__(self) -> None:
        values = (
            self.crop_width,
            self.local_height,
            self.context_height,
            self.hidden_dim,
        )
        if min(values) < 1:
            raise ValueError("Semantic Bond dimensions must be positive")
        if min(
            self.local_radius_pixels,
            self.context_radius_pixels,
            self.longitudinal_margin_pixels,
        ) <= 0:
            raise ValueError("Semantic Bond crop distances must be positive")


def _canonical_relation_view(
    pixel_values: torch.Tensor,
    points: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    width_samples: int,
    height_samples: int,
    radius_pixels: float,
    margin_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, pair_count = pair_mask.shape
    gather = pair_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, 2)
    expanded = points[:, None].expand(-1, pair_count, -1, -1)
    endpoints = torch.gather(expanded, 2, gather)
    begin = endpoints[:, :, 0]
    delta = endpoints[:, :, 1] - begin
    image_height, image_width = pixel_values.shape[-2:]
    delta_pixels = torch.stack(
        (delta[..., 0] * image_width, delta[..., 1] * image_height), dim=-1
    )
    lengths = delta_pixels.norm(dim=-1).clamp_min(1.0)
    direction_pixels = delta_pixels / lengths.unsqueeze(-1)
    direction = torch.stack(
        (
            direction_pixels[..., 0] / image_width,
            direction_pixels[..., 1] / image_height,
        ),
        dim=-1,
    )
    perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    start = begin - direction * margin_pixels
    span = delta + direction * (2.0 * margin_pixels)

    longitudinal = torch.linspace(
        0.0,
        1.0,
        width_samples,
        device=points.device,
        dtype=points.dtype,
    )
    transverse = torch.linspace(
        -radius_pixels,
        radius_pixels,
        height_samples,
        device=points.device,
        dtype=points.dtype,
    )
    base = start[:, :, None, :] + span[:, :, None, :] * longitudinal[None, None, :, None]
    grid = base[:, :, None, :, :] + (
        perpendicular[:, :, None, None, :]
        * transverse[None, None, :, None, None]
    )
    sample_grid = grid.mul(2.0).sub(1.0).reshape(
        batch_size, pair_count * height_samples, width_samples, 2
    )
    rgb = F.grid_sample(
        pixel_values,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(
        batch_size,
        pixel_values.size(1),
        pair_count,
        height_samples,
        width_samples,
    ).permute(0, 2, 1, 3, 4)
    selected_rgb = rgb[pair_mask]
    brightness = selected_rgb.add(1.0).mul(0.5).mean(dim=1, keepdim=True)
    ink = (1.0 - brightness).mul(2.0).sub(1.0)

    total_lengths = lengths + 2.0 * margin_pixels
    endpoint_x = torch.stack(
        (margin_pixels / total_lengths, (lengths + margin_pixels) / total_lengths),
        dim=-1,
    )[pair_mask]
    x = longitudinal[None, None, None, :]
    y = torch.linspace(
        0.0,
        1.0,
        height_samples,
        device=points.device,
        dtype=points.dtype,
    )[None, None, :, None]
    endpoint_maps = torch.exp(
        -(
            (x - endpoint_x[:, :, None, None]).square()
            + (y - 0.5).square()
        )
        / (2.0 * 0.035**2)
    ).mul(2.0).sub(1.0)
    view = torch.cat((selected_rgb, ink, endpoint_maps), dim=1)
    return view, (lengths / max(image_height, image_width))[pair_mask, None]


class RelationViewEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 8)),
            nn.Flatten(),
            nn.Linear(128 * 3 * 8, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)


class SemanticBondRelation(nn.Module):
    def __init__(self, config: SemanticBondRelationConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticBondRelationConfig()
        self.local_encoder = RelationViewEncoder(self.config.hidden_dim)
        self.context_encoder = RelationViewEncoder(self.config.hidden_dim)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU()
        )
        self.shared = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2 + 32, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        self.display = nn.Linear(self.config.hidden_dim, len(BOND_TYPES))
        self.chemical = nn.Linear(self.config.hidden_dim, len(BOND_TYPES))

    def _encode(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local, lengths = _canonical_relation_view(
            pixel_values,
            points,
            pair_indices,
            pair_mask,
            width_samples=self.config.crop_width,
            height_samples=self.config.local_height,
            radius_pixels=self.config.local_radius_pixels,
            margin_pixels=self.config.longitudinal_margin_pixels,
        )
        context, context_lengths = _canonical_relation_view(
            pixel_values,
            points,
            pair_indices,
            pair_mask,
            width_samples=self.config.crop_width,
            height_samples=self.config.context_height,
            radius_pixels=self.config.context_radius_pixels,
            margin_pixels=self.config.longitudinal_margin_pixels,
        )
        if not torch.equal(lengths, context_lengths):
            raise ValueError("Semantic Bond view geometry differs")
        visual_features = torch.cat(
            (self.local_encoder(local), self.context_encoder(context)), dim=-1
        )
        hidden = self.shared(
            torch.cat((visual_features, self.geometry_encoder(lengths)), dim=-1)
        )
        return hidden, local, visual_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        forward_hidden, local, forward_visual = self._encode(
            pixel_values, points, pair_indices, pair_mask
        )
        reverse_hidden, _, reverse_visual = self._encode(
            pixel_values, points, pair_indices.flip(-1), pair_mask
        )
        hidden = (forward_hidden + reverse_hidden) * 0.5
        visual_features = (forward_visual + reverse_visual) * 0.5
        return {
            "display_logits": self.display(hidden),
            "chemical_logits": self.chemical(hidden),
            "hidden": hidden,
            "visual_features": visual_features,
            "local_views": local,
        }


class SemanticBondExistence(nn.Module):
    def __init__(self, config: SemanticBondRelationConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticBondRelationConfig()
        self.relation = SemanticBondRelation(self.config)
        self.edge_visual = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        self.edge = nn.Linear(self.config.hidden_dim, 1)

    @torch.no_grad()
    def initialize_edge_visual_from_relation(self) -> None:
        """Warm-start the visual-only edge path without copying geometry inputs."""

        self.edge_visual[0].weight.copy_(
            self.relation.shared[0].weight[:, : self.config.hidden_dim * 2]
        )
        self.edge_visual[0].bias.copy_(self.relation.shared[0].bias)
        self.edge_visual[2].load_state_dict(self.relation.shared[2].state_dict())

    def forward(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.relation(pixel_values, points, pair_indices, pair_mask)
        edge_hidden = self.edge_visual(output["visual_features"])
        return {
            **output,
            "edge_hidden": edge_hidden,
            "edge_logits": self.edge(edge_hidden).squeeze(-1),
        }


def semantic_bond_relation_loss(
    output: dict[str, torch.Tensor],
    display_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    display_weights: torch.Tensor | None = None,
    chemical_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    display_loss = F.cross_entropy(
        output["display_logits"], display_targets, weight=display_weights
    )
    chemical_loss = F.cross_entropy(
        output["chemical_logits"], chemical_targets, weight=chemical_weights
    )
    return {
        "loss": display_loss + chemical_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
    }


def apply_aromatic_ring_semantics(
    bond_objects: list[dict] | tuple[dict, ...],
    chemical_logits: torch.Tensor,
    *,
    mean_threshold: float = 0.72,
    minimum_threshold: float = 0.25,
) -> tuple[list[dict], dict]:
    if chemical_logits.shape != (len(bond_objects), len(BOND_TYPES)):
        raise ValueError("Semantic Bond logits differ from Bond objects")
    if not 0.0 < minimum_threshold <= mean_threshold < 1.0:
        raise ValueError("Aromatic ring thresholds are invalid")
    graph = nx.Graph()
    pair_to_index = {}
    for index, item in enumerate(bond_objects):
        pair = tuple(sorted(int(value) for value in item["junction_pair"]))
        graph.add_edge(*pair)
        pair_to_index[pair] = index
    aromatic_probabilities = chemical_logits.float().softmax(-1)[
        :, BOND_TYPE_TO_ID["AROMATIC"]
    ].cpu()
    promoted = set()
    accepted_cycles = []
    candidate_cycles = 0
    for cycle in nx.cycle_basis(graph):
        if len(cycle) not in {5, 6}:
            continue
        candidate_cycles += 1
        pairs = [
            tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
            for index in range(len(cycle))
        ]
        indices = [pair_to_index[pair] for pair in pairs]
        values = aromatic_probabilities[indices]
        if float(values.mean()) < mean_threshold or float(values.min()) < minimum_threshold:
            continue
        promoted.update(indices)
        accepted_cycles.append(
            {
                "size": len(cycle),
                "mean_aromatic_probability": float(values.mean()),
                "minimum_aromatic_probability": float(values.min()),
            }
        )
    updated = []
    aromatic_id = BOND_TYPE_TO_ID["AROMATIC"]
    changes = 0
    for index, item in enumerate(bond_objects):
        value = dict(item)
        if index in promoted:
            changes += int(int(value["chemical"]) != aromatic_id)
            value["chemical"] = aromatic_id
        updated.append(value)
    return updated, {
        "semantic_ring_candidates": candidate_cycles,
        "semantic_ring_accepted": len(accepted_cycles),
        "semantic_ring_promoted_bonds": len(promoted),
        "semantic_chemical_type_changes": changes,
        "semantic_ring_details": accepted_cycles,
    }


__all__ = [
    "SemanticBondRelation",
    "SemanticBondRelationConfig",
    "SemanticBondExistence",
    "apply_aromatic_ring_semantics",
    "semantic_bond_relation_loss",
]
