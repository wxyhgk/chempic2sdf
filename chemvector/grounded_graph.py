"""Joint candidate-node and visible-line reasoning after grounded selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.geometry import stable_pairwise_distances
from chemvector.grounded_selector import grounded_selector_loss


@dataclass(frozen=True, slots=True)
class GroundedGraphConfig:
    candidate_dim: int = 192
    node_dim: int = 128
    edge_dim: int = 64
    maximum_candidates: int = 128
    line_samples: int = 24
    line_band: int = 3
    line_band_radius_pixels: float = 3.0
    blocker_radius_pixels: float = 8.0
    incident_topk: int = 6
    matching_radius_pixels: float = 16.0
    node_residual_scale: float | None = None

    def __post_init__(self) -> None:
        if min(
            self.candidate_dim,
            self.node_dim,
            self.edge_dim,
            self.maximum_candidates,
            self.line_samples,
            self.line_band,
            self.incident_topk,
        ) < 1:
            raise ValueError("Grounded graph dimensions must be positive")
        if min(
            self.line_band_radius_pixels,
            self.blocker_radius_pixels,
            self.matching_radius_pixels,
        ) <= 0:
            raise ValueError("Grounded graph radii must be positive")
        if self.node_residual_scale is not None and self.node_residual_scale <= 0:
            raise ValueError("Node residual scale must be positive when configured")
        if self.incident_topk >= self.maximum_candidates:
            raise ValueError("Incident top-k must be smaller than maximum candidates")


def radius_aware_linear_assignment(
    distances: torch.Tensor,
    maximum_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximize in-radius match cardinality before minimizing total distance."""

    if distances.ndim != 2:
        raise ValueError("Assignment distances must be a matrix")
    if maximum_distance <= 0:
        raise ValueError("Assignment radius must be positive")
    if not distances.numel():
        empty = torch.empty(0, device=distances.device, dtype=torch.long)
        return empty, empty
    largest = float(distances.detach().max())
    penalty = (largest + maximum_distance + 1.0) * (
        min(distances.shape) + 1
    )
    costs = distances + (distances > maximum_distance).to(distances.dtype) * penalty
    row_indices, column_indices = linear_sum_assignment(
        costs.detach().cpu().numpy()
    )
    return (
        torch.as_tensor(row_indices, device=distances.device, dtype=torch.long),
        torch.as_tensor(column_indices, device=distances.device, dtype=torch.long),
    )


def grounded_graph_targets(
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match candidates once, then transfer atom adjacency to candidate pairs."""

    assignments = torch.full_like(candidate_mask, -1, dtype=torch.long)
    node_targets = torch.zeros_like(candidate_mask, dtype=torch.float32)
    for batch_index in range(points.size(0)):
        candidate_ids = torch.where(candidate_mask[batch_index])[0]
        target_ids = torch.where(anchor_mask[batch_index])[0]
        if not len(candidate_ids) or not len(target_ids):
            continue
        distances = stable_pairwise_distances(
            points[batch_index, candidate_ids],
            anchor_points[batch_index, target_ids],
        ).mul(image_size)
        candidate_offsets, target_offsets = radius_aware_linear_assignment(
            distances,
            matching_radius_pixels,
        )
        accepted = distances[candidate_offsets, target_offsets] <= matching_radius_pixels
        matched_candidates = candidate_ids[candidate_offsets[accepted]]
        matched_targets = target_ids[target_offsets[accepted]]
        assignments[batch_index, matched_candidates] = matched_targets
        node_targets[batch_index, matched_candidates] = 1.0

    begin = pair_indices[:, 0]
    end = pair_indices[:, 1]
    begin_targets = assignments[:, begin]
    end_targets = assignments[:, end]
    mapped = (begin_targets >= 0) & (end_targets >= 0)
    edge_targets = torch.zeros(
        (points.size(0), pair_indices.size(0)),
        device=points.device,
        dtype=torch.float32,
    )
    batch = torch.arange(points.size(0), device=points.device)[:, None]
    values = adjacency[
        batch.expand_as(mapped),
        begin_targets.clamp_min(0),
        end_targets.clamp_min(0),
    ]
    edge_targets[mapped] = values[mapped].float()
    return node_targets, edge_targets, assignments


class GroundedCandidateGraphRefiner(nn.Module):
    """Read every candidate pair and feed line evidence back into node selection."""

    def __init__(self, config: GroundedGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.node_projection = nn.Sequential(
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, config.node_dim),
            nn.GELU(),
        )
        self.line_encoder = nn.Sequential(
            nn.Linear(config.line_samples * config.line_band + 3, config.edge_dim),
            nn.GELU(),
            nn.Linear(config.edge_dim, config.edge_dim),
            nn.GELU(),
        )
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(config.node_dim * 2, config.edge_dim),
            nn.GELU(),
        )
        self.blocker_encoder = nn.Sequential(
            nn.Linear(3, config.edge_dim),
            nn.GELU(),
        )
        self.edge_classifier = nn.Linear(config.edge_dim, 1)
        self.edge_message = nn.Linear(config.edge_dim, config.node_dim)
        self.node_residual = nn.Sequential(
            nn.Linear(
                config.node_dim * 2 + config.incident_topk + 1,
                config.node_dim,
            ),
            nn.GELU(),
            nn.Linear(config.node_dim, 1),
        )
        nn.init.zeros_(self.node_residual[-1].weight)
        nn.init.zeros_(self.node_residual[-1].bias)

    def _line_bands(
        self,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = points[:, pair_indices[:, 0]]
        end = points[:, pair_indices[:, 1]]
        delta = end - begin
        height, width = pixel_values.shape[-2:]
        delta_pixels = torch.stack(
            (delta[..., 0] * width, delta[..., 1] * height), dim=-1
        )
        lengths = delta_pixels.norm(dim=-1).clamp_min(1e-6)
        direction = delta_pixels / lengths.unsqueeze(-1)
        perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
        perpendicular = torch.stack(
            (perpendicular[..., 0] / width, perpendicular[..., 1] / height),
            dim=-1,
        )
        positions = torch.linspace(
            0.06, 0.94, self.config.line_samples, device=points.device, dtype=points.dtype
        )
        offsets = torch.linspace(
            -self.config.line_band_radius_pixels,
            self.config.line_band_radius_pixels,
            self.config.line_band,
            device=points.device,
            dtype=points.dtype,
        )
        base = begin[:, :, None, :] + delta[:, :, None, :] * positions[None, None, :, None]
        grid = base[:, :, None, :, :] + (
            perpendicular[:, :, None, None, :] * offsets[None, None, :, None, None]
        )
        grid = grid.mul(2.0).sub(1.0).reshape(
            pixel_values.size(0),
            pair_indices.size(0) * self.config.line_band,
            self.config.line_samples,
            2,
        )
        ink = (1.0 - pixel_values.mean(dim=1, keepdim=True)) * 0.5
        sampled = F.grid_sample(
            ink, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).reshape(
            pixel_values.size(0),
            pair_indices.size(0),
            self.config.line_band * self.config.line_samples,
        )
        geometry = torch.cat(
            (direction, (lengths / max(height, width)).unsqueeze(-1)), dim=-1
        )
        return sampled, geometry

    def _blocker_features(
        self,
        points: torch.Tensor,
        pair_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_logits: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """Describe high-confidence candidates lying inside each endpoint segment."""

        begin = points[:, pair_indices[:, 0]]
        end = points[:, pair_indices[:, 1]]
        delta = end - begin
        length_squared = delta.square().sum(dim=-1).clamp_min(1e-8)
        relative = points[:, None, :, :] - begin[:, :, None, :]
        projection = (relative * delta[:, :, None, :]).sum(dim=-1) / length_squared[:, :, None]
        closest = begin[:, :, None, :] + projection[..., None] * delta[:, :, None, :]
        distances = (points[:, None, :, :] - closest).norm(dim=-1) * image_size
        interior = torch.sigmoid((projection - 0.06) * 50.0) * torch.sigmoid(
            (0.94 - projection) * 50.0
        )
        candidate_probability = candidate_logits.detach().sigmoid()[:, None, :]
        support = candidate_probability * interior * torch.exp(
            -0.5 * (distances / self.config.blocker_radius_pixels).square()
        )
        support = support * candidate_mask[:, None, :]
        endpoint_mask = torch.zeros_like(support, dtype=torch.bool)
        pair_ids = torch.arange(pair_indices.size(0), device=points.device)
        endpoint_mask[:, pair_ids, pair_indices[:, 0]] = True
        endpoint_mask[:, pair_ids, pair_indices[:, 1]] = True
        support = support.masked_fill(endpoint_mask, 0.0)
        minimum_distance = distances.masked_fill(
            (support <= 1e-4) | endpoint_mask | ~candidate_mask[:, None, :],
            float(image_size),
        ).amin(dim=-1)
        return torch.stack(
            (
                support.amax(dim=-1),
                support.sum(dim=-1).clamp_max(4.0) / 4.0,
                (minimum_distance / self.config.blocker_radius_pixels).clamp_max(4.0) / 4.0,
            ),
            dim=-1,
        )

    def forward(
        self,
        candidate_hidden: torch.Tensor,
        pixel_values: torch.Tensor,
        points: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_node_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if candidate_hidden.shape[:2] != candidate_mask.shape:
            raise ValueError("Candidate hidden and mask shapes differ")
        if points.shape != (*candidate_mask.shape, 2):
            raise ValueError("Candidate points must have shape [batch, candidates, 2]")
        if base_node_logits.shape != candidate_mask.shape:
            raise ValueError("Base node logits must match the candidate mask")
        if candidate_hidden.size(1) != self.config.maximum_candidates:
            raise ValueError("Candidate count differs from grounded graph config")

        node_hidden = self.node_projection(candidate_hidden.float())
        node_hidden = node_hidden.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        pair_indices = torch.triu_indices(
            self.config.maximum_candidates,
            self.config.maximum_candidates,
            offset=1,
            device=points.device,
        ).transpose(0, 1)
        begin = pair_indices[:, 0]
        end = pair_indices[:, 1]
        pair_mask = candidate_mask[:, begin] & candidate_mask[:, end]
        line_bands, geometry = self._line_bands(pixel_values, points, pair_indices)
        blocker_features = self._blocker_features(
            points,
            pair_indices,
            candidate_mask,
            base_node_logits,
            pixel_values.size(-1),
        )
        endpoint_sum = node_hidden[:, begin] + node_hidden[:, end]
        endpoint_difference = (node_hidden[:, begin] - node_hidden[:, end]).abs()
        edge_hidden = (
            self.line_encoder(torch.cat((line_bands, geometry), dim=-1))
            + self.endpoint_encoder(torch.cat((endpoint_sum, endpoint_difference), dim=-1))
            + self.blocker_encoder(blocker_features)
        )
        edge_logits = self.edge_classifier(edge_hidden).squeeze(-1)
        edge_logits = edge_logits.masked_fill(~pair_mask, -30.0)
        edge_probability = edge_logits.sigmoid() * pair_mask

        messages = self.edge_message(edge_hidden) * edge_probability.unsqueeze(-1)
        aggregated = torch.zeros_like(node_hidden)
        degree = torch.zeros(
            (*candidate_mask.shape, 1), device=points.device, dtype=node_hidden.dtype
        )
        scatter_begin = begin[None, :, None].expand(
            points.size(0), -1, self.config.node_dim
        )
        scatter_end = end[None, :, None].expand_as(scatter_begin)
        aggregated.scatter_add_(1, scatter_begin, messages)
        aggregated.scatter_add_(1, scatter_end, messages)
        degree.scatter_add_(
            1,
            begin[None, :, None].expand(points.size(0), -1, 1),
            edge_probability.unsqueeze(-1),
        )
        degree.scatter_add_(
            1,
            end[None, :, None].expand(points.size(0), -1, 1),
            edge_probability.unsqueeze(-1),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        incident = torch.zeros(
            (points.size(0), self.config.maximum_candidates, self.config.maximum_candidates),
            device=points.device,
            dtype=edge_probability.dtype,
        )
        incident[:, begin, end] = edge_probability
        incident[:, end, begin] = edge_probability
        strongest = incident.topk(self.config.incident_topk, dim=-1).values
        residual = self.node_residual(
            torch.cat((node_hidden, aggregated, strongest, degree), dim=-1)
        ).squeeze(-1)
        if self.config.node_residual_scale is not None:
            residual = self.config.node_residual_scale * torch.tanh(residual)
        node_logits = (base_node_logits.float() + residual).masked_fill(
            ~candidate_mask, -30.0
        )
        return {
            "points": points,
            "candidate_mask": candidate_mask,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "blocker_features": blocker_features,
            "incident_edge_support": strongest,
            "node_residual_logits": residual,
        }


def grounded_graph_loss(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    matching_radius_pixels: float,
    image_size: int,
    edge_loss_weight: float = 1.0,
    negative_ratio: int = 8,
    ranking_weight: float = 1.0,
    set_count_weight: float = 1.0,
    cardinality_weight: float = 0.1,
    whole_set_weight: float = 0.0,
    whole_set_margin: float = 1.0,
) -> dict[str, torch.Tensor | float]:
    node_targets, edge_targets, assignments = grounded_graph_targets(
        output["points"].detach(),
        output["candidate_mask"],
        output["pair_indices"],
        anchor_points,
        anchor_mask,
        adjacency,
        matching_radius_pixels=matching_radius_pixels,
        image_size=image_size,
    )
    node_losses = grounded_selector_loss(
        output["node_logits"],
        node_targets,
        candidate_mask=output["candidate_mask"],
        ranking_weight=ranking_weight,
        set_count_weight=set_count_weight,
        cardinality_weight=cardinality_weight,
    )
    whole_set_losses = []
    for logits, targets, valid in zip(
        output["node_logits"], node_targets.bool(), output["candidate_mask"]
    ):
        positive_scores = logits[valid & targets]
        negative_scores = logits[valid & ~targets]
        if not len(positive_scores) or not len(negative_scores):
            continue
        whole_set_losses.append(
            F.softplus(
                whole_set_margin
                + negative_scores.max()
                - positive_scores.min()
            )
        )
    whole_set_loss = (
        torch.stack(whole_set_losses).mean()
        if whole_set_losses
        else output["node_logits"].sum() * 0.0
    )
    structured_node_loss = node_losses["loss"] + whole_set_weight * whole_set_loss
    valid_edges = output["pair_mask"]
    logits = output["edge_logits"][valid_edges].float()
    labels = edge_targets[valid_edges]
    positives = torch.where(labels > 0.5)[0]
    negatives = torch.where(labels <= 0.5)[0]
    negative_count = min(len(negatives), max(len(positives) * negative_ratio, 32))
    if negative_count:
        hard_offsets = logits[negatives].detach().topk(negative_count).indices
        selected = torch.cat((positives, negatives[hard_offsets]))
    else:
        selected = positives
    if len(selected):
        selected_labels = labels[selected]
        positive_count = int(selected_labels.sum())
        pos_weight = min(
            (len(selected_labels) - positive_count) / max(positive_count, 1), 16.0
        )
        edge_loss = F.binary_cross_entropy_with_logits(
            logits[selected],
            selected_labels,
            pos_weight=logits.new_tensor(pos_weight),
        )
    else:
        edge_loss = output["edge_logits"].sum() * 0.0
    return {
        **node_losses,
        "loss": structured_node_loss + edge_loss_weight * edge_loss,
        "node_loss": structured_node_loss,
        "whole_set_loss": whole_set_loss,
        "edge_loss": edge_loss,
        "node_targets": node_targets,
        "edge_targets": edge_targets,
        "assignments": assignments,
        "proposal_recall": float(node_targets.sum() / anchor_mask.sum().clamp_min(1)),
        "selected_edges": float(len(selected)),
    }


def load_grounded_graph_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[GroundedCandidateGraphRefiner, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = GroundedGraphConfig(**payload["grounded_graph_config"])
    model = GroundedCandidateGraphRefiner(config).to(device)
    model.load_state_dict(payload["grounded_graph"])
    return model.eval(), payload
