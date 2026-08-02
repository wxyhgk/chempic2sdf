"""Set-based semantic bond object detection."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_query import position_encoding_2d


BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_TYPE_TO_ID = {name: index for index, name in enumerate(BOND_TYPES)}
DIRECTIONS = ("NONE", "BEGINWEDGE", "BEGINDASH", "ENDUPRIGHT", "ENDDOWNRIGHT")
DIRECTION_TO_ID = {name: index for index, name in enumerate(DIRECTIONS)}


@dataclass(frozen=True, slots=True)
class BondObjectConfig:
    input_channels: int = 96
    d_model: int = 192
    queries: int = 128
    layers: int = 2
    heads: int = 6
    ffn_dim: int = 768
    memory_stride: int = 4
    local_feature_dim: int = 32
    local_radius: float = 32.0 / 768.0


class BondObjectDetector(nn.Module):
    def __init__(self, config: BondObjectConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.d_model,
                kernel_size=3,
                stride=config.memory_stride,
                padding=1,
            ),
            nn.GroupNorm(1, config.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=config.layers, norm=nn.LayerNorm(config.d_model)
        )
        self.queries = nn.Embedding(config.queries, config.d_model)
        self.local_projection = nn.Conv2d(
            config.input_channels, config.local_feature_dim, kernel_size=1
        )
        self.endpoint_refinement = nn.Sequential(
            nn.Linear(
                config.d_model + 2 * 9 * config.local_feature_dim,
                config.ffn_dim,
            ),
            nn.GELU(),
            nn.Linear(config.ffn_dim, 4),
        )
        self.second_endpoint_refinement = nn.Sequential(
            nn.Linear(
                config.d_model + 2 * 9 * config.local_feature_dim,
                config.ffn_dim,
            ),
            nn.GELU(),
            nn.Linear(config.ffn_dim, 4),
        )
        self.presence = nn.Linear(config.d_model, 1)
        self.endpoints = nn.Linear(config.d_model, 4)
        self.bbox = nn.Linear(config.d_model, 4)
        self.display_type = nn.Linear(config.d_model, len(BOND_TYPES))
        self.chemical_type = nn.Linear(config.d_model, len(BOND_TYPES))
        self.direction = nn.Linear(config.d_model, len(DIRECTIONS))
        nn.init.constant_(self.presence.bias, -2.0)
        nn.init.zeros_(self.endpoint_refinement[-1].weight)
        nn.init.zeros_(self.endpoint_refinement[-1].bias)
        nn.init.zeros_(self.second_endpoint_refinement[-1].weight)
        nn.init.zeros_(self.second_endpoint_refinement[-1].bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory_map = self.input_projection(features)
        height, width = memory_map.shape[-2:]
        memory = memory_map.flatten(2).transpose(1, 2)
        memory = memory + position_encoding_2d(
            height, width, self.config.d_model, memory.device, memory.dtype
        ).unsqueeze(0)
        queries = self.queries.weight.unsqueeze(0).expand(features.size(0), -1, -1)
        hidden = self.decoder(queries, memory)
        coarse_endpoints = self.endpoints(hidden).sigmoid().reshape(
            features.size(0), self.config.queries, 2, 2
        )
        local_features = self.local_projection(features)
        offsets = torch.tensor(
            (
                (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
                (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
                (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
            ),
            device=features.device,
            dtype=features.dtype,
        ) * self.config.local_radius
        sample_points = coarse_endpoints[:, :, :, None, :] + offsets[None, None, None, :, :]
        sample_grid = sample_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            local_features,
            sample_grid.reshape(features.size(0), -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
            features.size(0), self.config.queries, 2 * 9 * self.config.local_feature_dim
        )
        refinement = self.endpoint_refinement(torch.cat((hidden, sampled), dim=-1))
        first_endpoints = (
            coarse_endpoints
            + refinement.tanh().reshape(features.size(0), self.config.queries, 2, 2)
            * self.config.local_radius
        ).clamp(0.0, 1.0)
        second_points = first_endpoints[:, :, :, None, :] + 2.0 * offsets[None, None, None, :, :]
        second_grid = second_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        second_sampled = F.grid_sample(
            local_features,
            second_grid.reshape(features.size(0), -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        second_sampled = second_sampled.squeeze(-1).transpose(1, 2).reshape(
            features.size(0), self.config.queries, 2 * 9 * self.config.local_feature_dim
        )
        second_refinement = self.second_endpoint_refinement(
            torch.cat((hidden, second_sampled), dim=-1)
        )
        endpoints = (
            first_endpoints
            + second_refinement.tanh().reshape(features.size(0), self.config.queries, 2, 2)
            * (2.0 * self.config.local_radius)
        ).clamp(0.0, 1.0)
        return {
            "presence_logits": self.presence(hidden).squeeze(-1),
            "endpoints": endpoints,
            "coarse_endpoints": coarse_endpoints,
            "first_endpoints": first_endpoints,
            "bbox": self.bbox(hidden).sigmoid(),
            "display_logits": self.display_type(hidden),
            "chemical_logits": self.chemical_type(hidden),
            "direction_logits": self.direction(hidden),
        }


def unordered_endpoint_cost(
    predicted: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    forward = (
        predicted[:, None, 0].sub(targets[None, :, 0]).abs().sum(-1)
        + predicted[:, None, 1].sub(targets[None, :, 1]).abs().sum(-1)
    )
    reverse = (
        predicted[:, None, 0].sub(targets[None, :, 1]).abs().sum(-1)
        + predicted[:, None, 1].sub(targets[None, :, 0]).abs().sum(-1)
    )
    return torch.minimum(forward, reverse)


def bond_object_assignments(output: dict, targets: dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
    assignments = []
    for batch_index in range(output["endpoints"].size(0)):
        valid = targets["mask"][batch_index]
        count = int(valid.sum())
        if not count:
            empty = torch.empty(0, dtype=torch.long, device=output["endpoints"].device)
            assignments.append((empty, empty))
            continue
        endpoints = targets["endpoints"][batch_index, valid]
        boxes = targets["bbox"][batch_index, valid]
        display = targets["display"][batch_index, valid]
        chemical = targets["chemical"][batch_index, valid]
        cost = 8.0 * unordered_endpoint_cost(output["endpoints"][batch_index], endpoints)
        cost = cost + 2.0 * torch.cdist(output["bbox"][batch_index].float(), boxes.float(), p=1)
        cost = cost - output["display_logits"][batch_index].softmax(-1)[:, display]
        cost = cost - output["chemical_logits"][batch_index].softmax(-1)[:, chemical]
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append((
            torch.tensor(rows, device=cost.device, dtype=torch.long),
            torch.tensor(columns, device=cost.device, dtype=torch.long),
        ))
    return assignments


def bond_object_loss(
    output: dict,
    targets: dict,
    *,
    endpoint_tail_weight: float = 0.0,
    endpoint_barrier_weight: float = 0.0,
    endpoint_barrier_margin_pixels: float = 4.0,
    image_size: float = 768.0,
) -> dict[str, torch.Tensor]:
    if endpoint_tail_weight < 0 or endpoint_barrier_weight < 0:
        raise ValueError("endpoint loss weights must be nonnegative")
    if endpoint_barrier_margin_pixels < 0 or image_size <= 0:
        raise ValueError("endpoint barrier geometry is invalid")
    assignments = bond_object_assignments(output, targets)
    presence_targets = torch.zeros_like(output["presence_logits"])
    endpoint_losses = []
    bbox_losses = []
    display_losses = []
    chemical_losses = []
    direction_losses = []
    endpoint_tail_losses = []
    endpoint_barrier_losses = []
    for batch_index, (queries, target_offsets) in enumerate(assignments):
        if not len(queries):
            continue
        presence_targets[batch_index, queries] = 1.0
        valid_indices = torch.where(targets["mask"][batch_index])[0]
        target_indices = valid_indices[target_offsets]
        predicted = output["endpoints"][batch_index, queries]
        expected = targets["endpoints"][batch_index, target_indices]
        forward = (predicted - expected).abs().sum(dim=(-1, -2))
        reverse = (predicted - expected.flip(1)).abs().sum(dim=(-1, -2))
        object_errors = torch.minimum(forward, reverse)
        endpoint_losses.append(object_errors.mean())
        endpoint_tail_losses.append(object_errors.max())
        use_reverse = (reverse < forward)[:, None, None]
        aligned_expected = torch.where(use_reverse, expected.flip(1), expected)
        endpoint_distances = torch.linalg.vector_norm(
            (predicted - aligned_expected) * image_size,
            dim=-1,
        )
        endpoint_barrier_losses.append(
            F.relu(endpoint_distances - endpoint_barrier_margin_pixels).square().mean()
        )
        bbox_losses.append(F.smooth_l1_loss(
            output["bbox"][batch_index, queries], targets["bbox"][batch_index, target_indices]
        ))
        display_losses.append(F.cross_entropy(
            output["display_logits"][batch_index, queries], targets["display"][batch_index, target_indices]
        ))
        chemical_losses.append(F.cross_entropy(
            output["chemical_logits"][batch_index, queries], targets["chemical"][batch_index, target_indices]
        ))
        direction_losses.append(F.cross_entropy(
            output["direction_logits"][batch_index, queries], targets["direction"][batch_index, target_indices]
        ))
    positives = presence_targets.sum().clamp_min(1.0)
    negatives = (presence_targets.numel() - positives).clamp_min(1.0)
    presence_loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"], presence_targets, pos_weight=(negatives / positives).clamp_max(16.0)
    )
    zero = presence_loss * 0.0
    endpoint_loss = torch.stack(endpoint_losses).mean() if endpoint_losses else zero
    endpoint_tail_loss = (
        torch.stack(endpoint_tail_losses).mean() if endpoint_tail_losses else zero
    )
    endpoint_barrier_loss = (
        torch.stack(endpoint_barrier_losses).mean()
        if endpoint_barrier_losses
        else zero
    )
    bbox_loss = torch.stack(bbox_losses).mean() if bbox_losses else zero
    display_loss = torch.stack(display_losses).mean() if display_losses else zero
    chemical_loss = torch.stack(chemical_losses).mean() if chemical_losses else zero
    direction_loss = torch.stack(direction_losses).mean() if direction_losses else zero
    return {
        "loss": (
            presence_loss
            + 8.0 * endpoint_loss
            + endpoint_tail_weight * endpoint_tail_loss
            + endpoint_barrier_weight * endpoint_barrier_loss
            + 2.0 * bbox_loss
            + display_loss
            + chemical_loss
            + 0.5 * direction_loss
        ),
        "presence_loss": presence_loss,
        "endpoint_loss": endpoint_loss,
        "endpoint_tail_loss": endpoint_tail_loss,
        "endpoint_barrier_loss": endpoint_barrier_loss,
        "bbox_loss": bbox_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
        "direction_loss": direction_loss,
    }
