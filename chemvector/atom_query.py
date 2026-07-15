"""Set-based atom detection with learned queries and Hungarian matching."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class AtomQueryConfig:
    input_channels: int = 192
    d_model: int = 192
    queries: int = 128
    layers: int = 3
    heads: int = 6
    ffn_dim: int = 768
    dropout: float = 0.0
    memory_stride: int = 2
    local_feature_dim: int = 48
    local_radius: float = 32.0 / 768.0

    def __post_init__(self) -> None:
        if min(
            self.input_channels,
            self.d_model,
            self.queries,
            self.layers,
            self.heads,
            self.ffn_dim,
            self.local_feature_dim,
        ) < 1:
            raise ValueError("Atom Query dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.d_model % 4:
            raise ValueError("d_model must be divisible by 4 for 2D positions")
        if self.memory_stride not in {1, 2, 4}:
            raise ValueError("memory_stride must be 1, 2, or 4")
        if not 0.0 < self.local_radius < 0.5:
            raise ValueError("local_radius must be in (0, 0.5)")


def position_encoding_2d(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quarter = channels // 4
    frequencies = torch.exp(
        torch.arange(quarter, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    y_angles = y * frequencies[None, :]
    x_angles = x * frequencies[None, :]
    y_encoding = torch.cat((y_angles.sin(), y_angles.cos()), dim=-1)
    x_encoding = torch.cat((x_angles.sin(), x_angles.cos()), dim=-1)
    encoding = torch.cat(
        (
            y_encoding[:, None, :].expand(height, width, -1),
            x_encoding[None, :, :].expand(height, width, -1),
        ),
        dim=-1,
    )
    return encoding.reshape(height * width, channels).to(dtype=dtype)


def query_reference_points(queries: int) -> torch.Tensor:
    columns = math.ceil(math.sqrt(queries))
    rows = math.ceil(queries / columns)
    y, x = torch.meshgrid(
        (torch.arange(rows, dtype=torch.float32) + 0.5) / rows,
        (torch.arange(columns, dtype=torch.float32) + 0.5) / columns,
        indexing="ij",
    )
    return torch.stack((x.flatten(), y.flatten()), dim=-1)[:queries]


class AtomQueryDetector(nn.Module):
    def __init__(self, config: AtomQueryConfig) -> None:
        super().__init__()
        self.config = config
        kernel_size = 1 if config.memory_stride == 1 else 3
        padding = 0 if config.memory_stride == 1 else 1
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.d_model,
                kernel_size=kernel_size,
                stride=config.memory_stride,
                padding=padding,
            ),
            nn.GroupNorm(1, config.d_model),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.queries = nn.Embedding(config.queries, config.d_model)
        self.reference_projection = nn.Linear(2, config.d_model)
        self.register_buffer(
            "reference_points",
            query_reference_points(config.queries),
            persistent=True,
        )
        self.presence = nn.Linear(config.d_model, 1)
        self.center = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        self.aromatic = nn.Linear(config.d_model, 1)
        self.local_projection = nn.Conv2d(
            config.input_channels,
            config.local_feature_dim,
            kernel_size=1,
        )
        self.local_refinement = nn.Sequential(
            nn.Linear(config.d_model + config.local_feature_dim * 9, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )
        nn.init.constant_(self.presence.bias, -2.0)
        nn.init.zeros_(self.center[-1].weight)
        nn.init.zeros_(self.center[-1].bias)
        nn.init.zeros_(self.local_refinement[-1].weight)
        nn.init.zeros_(self.local_refinement[-1].bias)

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        if feature_map.ndim != 4 or feature_map.size(1) != self.config.input_channels:
            raise ValueError("Atom Query feature map has incompatible shape")
        memory = self.input_projection(feature_map)
        height, width = memory.shape[-2:]
        memory = memory.flatten(2).transpose(1, 2)
        memory = memory + position_encoding_2d(
            height,
            width,
            memory.size(-1),
            memory.device,
            memory.dtype,
        ).unsqueeze(0)
        query_content = self.queries.weight + self.reference_projection(
            self.reference_points
        )
        queries = query_content.unsqueeze(0).expand(feature_map.size(0), -1, -1)
        hidden = self.decoder(queries, memory)
        reference_logits = torch.logit(
            self.reference_points.clamp(1e-4, 1.0 - 1e-4)
        )
        coarse_centers = (
            reference_logits.unsqueeze(0) + self.center(hidden)
        ).sigmoid()
        local_features = self.local_projection(feature_map)
        offsets = torch.tensor(
            (
                (-1.0, -1.0),
                (0.0, -1.0),
                (1.0, -1.0),
                (-1.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (-1.0, 1.0),
                (0.0, 1.0),
                (1.0, 1.0),
            ),
            device=feature_map.device,
            dtype=feature_map.dtype,
        ) * self.config.local_radius
        sample_points = coarse_centers[:, :, None, :] + offsets[None, None, :, :]
        sample_grid = sample_points.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            local_features,
            sample_grid.reshape(feature_map.size(0), -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
            feature_map.size(0),
            self.config.queries,
            self.config.local_feature_dim * 9,
        )
        refinement = self.local_refinement(torch.cat((hidden, sampled), dim=-1))
        centers = (
            coarse_centers + refinement.tanh() * self.config.local_radius
        ).clamp(0.0, 1.0)
        return {
            "presence_logits": self.presence(hidden).squeeze(-1),
            "centers": centers,
            "coarse_centers": coarse_centers,
            "aromatic_logits": self.aromatic(hidden).squeeze(-1),
        }


def hungarian_assignments(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    assignments = []
    probabilities = output["presence_logits"].detach().sigmoid()
    centers = output["centers"].detach()
    for batch_index in range(centers.size(0)):
        targets = anchor_points[batch_index][anchor_mask[batch_index]]
        if not len(targets):
            empty = torch.empty(0, dtype=torch.long, device=centers.device)
            assignments.append((empty, empty))
            continue
        cost = torch.cdist(centers[batch_index].float(), targets.float(), p=1)
        cost = cost - 0.2 * probabilities[batch_index, :, None]
        query_indices, target_indices = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                torch.as_tensor(query_indices, dtype=torch.long, device=centers.device),
                torch.as_tensor(target_indices, dtype=torch.long, device=centers.device),
            )
        )
    return assignments


def atom_query_loss(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    aromatic_targets: torch.Tensor,
    *,
    no_object_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    assignments = hungarian_assignments(output, anchor_points, anchor_mask)
    presence_targets = torch.zeros_like(output["presence_logits"])
    presence_weights = torch.full_like(output["presence_logits"], no_object_weight)
    center_losses = []
    coarse_center_losses = []
    aromatic_losses = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        presence_targets[batch_index, query_indices] = 1.0
        presence_weights[batch_index, query_indices] = 1.0
        targets = anchor_points[batch_index][anchor_mask[batch_index]][target_indices]
        target_aromatic = aromatic_targets[batch_index][anchor_mask[batch_index]][
            target_indices
        ]
        center_losses.append(
            F.l1_loss(output["centers"][batch_index, query_indices], targets)
        )
        coarse_center_losses.append(
            F.l1_loss(
                output["coarse_centers"][batch_index, query_indices],
                targets,
            )
        )
        aromatic_losses.append(
            F.binary_cross_entropy_with_logits(
                output["aromatic_logits"][batch_index, query_indices],
                target_aromatic,
            )
        )
    presence_loss = (
        F.binary_cross_entropy_with_logits(
            output["presence_logits"],
            presence_targets,
            reduction="none",
        )
        * presence_weights
    ).sum() / presence_weights.sum().clamp_min(1.0)
    center_loss = (
        torch.stack(center_losses).mean()
        if center_losses
        else output["centers"].sum() * 0.0
    )
    aromatic_loss = (
        torch.stack(aromatic_losses).mean()
        if aromatic_losses
        else output["aromatic_logits"].sum() * 0.0
    )
    coarse_center_loss = (
        torch.stack(coarse_center_losses).mean()
        if coarse_center_losses
        else output["coarse_centers"].sum() * 0.0
    )
    expected_counts = anchor_mask.sum(dim=1).to(output["presence_logits"].dtype)
    predicted_counts = output["presence_logits"].sigmoid().sum(dim=1)
    cardinality_loss = (
        (predicted_counts - expected_counts).abs() / expected_counts.clamp_min(1.0)
    ).mean()
    loss = (
        presence_loss
        + 10.0 * center_loss
        + 2.0 * coarse_center_loss
        + 0.25 * aromatic_loss
        + 0.01 * cardinality_loss
    )
    return {
        "loss": loss,
        "presence_loss": presence_loss,
        "center_loss": center_loss,
        "coarse_center_loss": coarse_center_loss,
        "aromatic_loss": aromatic_loss,
        "cardinality_loss": cardinality_loss,
    }


@torch.no_grad()
def decode_atom_queries(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.5,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    probabilities = output["presence_logits"].sigmoid()
    aromatic = output["aromatic_logits"].sigmoid()
    decoded = []
    for batch_index in range(probabilities.size(0)):
        scores, indices = probabilities[batch_index].topk(
            min(maximum_vertices, probabilities.size(1))
        )
        vertices = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < threshold:
                break
            vertices.append(
                {
                    "x": float(output["centers"][batch_index, index, 0]),
                    "y": float(output["centers"][batch_index, index, 1]),
                    "score": score,
                    "aromatic": bool(aromatic[batch_index, index] >= 0.5),
                }
            )
        decoded.append(vertices)
    return decoded


def snap_queries_to_dense_candidates(
    queries: list[dict[str, float | bool]],
    candidates: list[dict[str, float | bool]],
    *,
    image_size: int = 768,
    maximum_distance_pixels: float = 64.0,
    score_weight_pixels: float = 2.0,
) -> list[dict[str, float | bool]]:
    if not queries or not candidates:
        return [dict(item) for item in queries]
    query_points = torch.tensor(
        [(float(item["x"]), float(item["y"])) for item in queries],
        dtype=torch.float32,
    )
    candidate_points = torch.tensor(
        [(float(item["x"]), float(item["y"])) for item in candidates],
        dtype=torch.float32,
    )
    distances = torch.cdist(query_points, candidate_points).mul(image_size)
    scores = torch.tensor(
        [float(item["score"]) for item in candidates],
        dtype=torch.float32,
    )
    cost = distances - score_weight_pixels * scores[None, :]
    query_indices, candidate_indices = linear_sum_assignment(cost.numpy())
    snapped = [dict(item) for item in queries]
    for query_index, candidate_index in zip(query_indices, candidate_indices):
        if float(distances[query_index, candidate_index]) > maximum_distance_pixels:
            continue
        candidate = candidates[int(candidate_index)]
        snapped[int(query_index)]["x"] = float(candidate["x"])
        snapped[int(query_index)]["y"] = float(candidate["y"])
        snapped[int(query_index)]["dense_score"] = float(candidate["score"])
    return snapped
