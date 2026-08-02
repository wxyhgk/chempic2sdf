"""One-to-one filtering for dense vertex proposals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.geometry import stable_pairwise_distances


@dataclass(frozen=True, slots=True)
class ProposalKeepConfig:
    input_channels: int
    hidden_dim: int = 32
    maximum_proposals: int = 256
    proposal_threshold: float = 0.05
    proposal_nms_kernel_size: int = 3
    matching_radius_pixels: float = 16.0

    def __post_init__(self) -> None:
        if min(self.input_channels, self.hidden_dim, self.maximum_proposals) < 1:
            raise ValueError("Proposal keep dimensions must be positive")
        if not 0.0 < self.proposal_threshold < 1.0:
            raise ValueError("Proposal threshold must be in (0, 1)")
        if (
            self.proposal_nms_kernel_size < 1
            or self.proposal_nms_kernel_size % 2 == 0
        ):
            raise ValueError("Proposal NMS kernel size must be a positive odd integer")
        if self.matching_radius_pixels <= 0:
            raise ValueError("Proposal matching radius must be positive")


class ProposalKeepHead(nn.Module):
    """Classify dense peaks with local and wider visual context."""

    def __init__(self, config: ProposalKeepConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels + 4, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.contexts = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
            for dilation in (1, 4, 8)
        )
        self.output = nn.Conv2d(channels * 3, 1, kernel_size=1)
        nn.init.constant_(self.output.bias, -2.0)

    def forward(
        self,
        feature_map: torch.Tensor,
        dense_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        dense_features = torch.cat(
            (
                dense_output["vertex_heatmap_logits"].sigmoid(),
                dense_output["vertex_offset_logits"].sigmoid(),
                dense_output["vertex_aromatic_logits"].sigmoid(),
            ),
            dim=1,
        )
        if feature_map.shape[-2:] != dense_features.shape[-2:]:
            raise ValueError("Proposal feature maps must share one spatial resolution")
        hidden = self.stem(torch.cat((feature_map, dense_features), dim=1))
        contexts = [F.gelu(layer(hidden)) for layer in self.contexts]
        return self.output(torch.cat(contexts, dim=1))


def load_proposal_keep_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[ProposalKeepHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ProposalKeepConfig(**payload["proposal_keep_config"])
    head = ProposalKeepHead(config).to(device)
    head.load_state_dict(payload["proposal_keep"])
    return head.eval(), payload


def scale_vertex_batch(
    pixel_values: torch.Tensor,
    anchor_points: torch.Tensor,
    *,
    minimum_scale: float,
    maximum_scale: float,
    probability: float = 1.0,
    scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale drawings around the canvas center and transform their anchors."""

    if not 0.0 < minimum_scale <= maximum_scale <= 1.0:
        raise ValueError("Vertex scale range must lie in (0, 1]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Vertex scale probability must be in [0, 1]")
    batch_size = pixel_values.size(0)
    if scales is None:
        scales = torch.empty(
            batch_size,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).uniform_(minimum_scale, maximum_scale)
        if probability < 1.0:
            selected = torch.rand(batch_size, device=pixel_values.device) < probability
            scales = torch.where(selected, scales, torch.ones_like(scales))
    else:
        scales = scales.to(device=pixel_values.device, dtype=pixel_values.dtype)
        if scales.shape != (batch_size,):
            raise ValueError("Vertex scales must contain one value per image")
        if bool(((scales < minimum_scale) | (scales > maximum_scale)).any()):
            raise ValueError("Vertex scales fall outside the configured range")
    theta = torch.zeros(
        (batch_size, 2, 3),
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )
    theta[:, 0, 0] = scales.reciprocal()
    theta[:, 1, 1] = scales.reciprocal()
    grid = F.affine_grid(theta, pixel_values.shape, align_corners=False)
    # Inputs are normalized to [-1, 1], so subtracting white makes zero padding white.
    scaled_pixels = F.grid_sample(
        pixel_values - 1.0,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ) + 1.0
    scaled_anchors = (
        anchor_points - 0.5
    ) * scales[:, None, None] + 0.5
    return scaled_pixels, scaled_anchors, scales


@torch.no_grad()
def proposal_indices(
    dense_output: dict[str, torch.Tensor],
    config: ProposalKeepConfig,
) -> list[torch.Tensor]:
    heatmap = dense_output["vertex_heatmap_logits"].float().sigmoid()
    kernel = config.proposal_nms_kernel_size
    peaks = heatmap == F.max_pool2d(
        heatmap,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    peak_scores = heatmap.masked_fill(~peaks, 0.0).flatten(1)
    count = min(config.maximum_proposals, peak_scores.size(1))
    scores, indices = peak_scores.topk(count, dim=1)
    return [
        batch_indices[batch_scores >= config.proposal_threshold]
        for batch_scores, batch_indices in zip(scores, indices)
    ]


@torch.no_grad()
def proposal_points(
    dense_output: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch_index: int,
) -> torch.Tensor:
    height, width = dense_output["vertex_heatmap_logits"].shape[-2:]
    y = torch.div(indices, width, rounding_mode="floor")
    x = indices % width
    offsets = dense_output["vertex_offset_logits"].sigmoid()
    return torch.stack(
        (
            (x + offsets[batch_index, 0, y, x]) / width,
            (y + offsets[batch_index, 1, y, x]) / height,
        ),
        dim=-1,
    )


@torch.no_grad()
def proposal_positive_masks(
    dense_output: dict[str, torch.Tensor],
    indices: list[torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    config: ProposalKeepConfig,
    image_size: int,
) -> tuple[list[torch.Tensor], int, int]:
    masks = []
    matched_targets = 0
    target_count = int(anchor_mask.sum())
    for batch_index, batch_indices in enumerate(indices):
        positive = torch.zeros(
            len(batch_indices),
            dtype=torch.bool,
            device=batch_indices.device,
        )
        targets = anchor_points[batch_index][anchor_mask[batch_index]]
        if len(batch_indices) and len(targets):
            points = proposal_points(dense_output, batch_indices, batch_index)
            distances = stable_pairwise_distances(points, targets).mul(image_size)
            proposal_ids, target_ids = linear_sum_assignment(distances.cpu().numpy())
            proposal_ids = torch.as_tensor(
                proposal_ids,
                device=batch_indices.device,
                dtype=torch.long,
            )
            target_ids = torch.as_tensor(
                target_ids,
                device=batch_indices.device,
                dtype=torch.long,
            )
            accepted = distances[proposal_ids, target_ids] <= config.matching_radius_pixels
            positive[proposal_ids[accepted]] = True
            matched_targets += int(accepted.sum())
        masks.append(positive)
    return masks, matched_targets, target_count


def proposal_keep_loss(
    keep_logits: torch.Tensor,
    dense_output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    config: ProposalKeepConfig,
    image_size: int,
    cardinality_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor | float]:
    if cardinality_loss_weight < 0:
        raise ValueError("Proposal cardinality loss weight must be nonnegative")
    indices = proposal_indices(dense_output, config)
    positive_masks, matched_targets, target_count = proposal_positive_masks(
        dense_output,
        indices,
        anchor_points,
        anchor_mask,
        config,
        image_size,
    )
    losses = []
    positive_count = 0
    proposal_count = 0
    predicted_counts = []
    width = keep_logits.size(-1)
    for batch_index, (batch_indices, positive) in enumerate(
        zip(indices, positive_masks)
    ):
        if not len(batch_indices):
            predicted_counts.append(keep_logits[batch_index].sum() * 0.0)
            continue
        y = torch.div(batch_indices, width, rounding_mode="floor")
        x = batch_indices % width
        logits = keep_logits[batch_index, 0, y, x].float()
        targets = positive.to(logits.dtype)
        positives = int(positive.sum())
        negatives = len(positive) - positives
        positive_weight = min(negatives / max(positives, 1), 10.0)
        weights = torch.where(
            positive,
            torch.full_like(logits, positive_weight),
            torch.ones_like(logits),
        )
        losses.append(
            F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        )
        predicted_counts.append(logits.sigmoid().sum())
        positive_count += positives
        proposal_count += len(positive)
    classification_loss = (
        torch.stack(losses).mean() if losses else keep_logits.sum() * 0.0
    )
    if predicted_counts:
        predicted_count_tensor = torch.stack(predicted_counts)
        target_counts = anchor_mask.sum(dim=1).to(predicted_count_tensor.dtype)
        cardinality_loss = (
            (predicted_count_tensor - target_counts).abs()
            / target_counts.clamp_min(1.0)
        ).mean()
    else:
        cardinality_loss = keep_logits.sum() * 0.0
    loss = classification_loss + cardinality_loss_weight * cardinality_loss
    return {
        "loss": loss,
        "classification_loss": classification_loss,
        "cardinality_loss": cardinality_loss,
        "proposal_count": float(proposal_count),
        "positive_count": float(positive_count),
        "proposal_recall": matched_targets / max(target_count, 1),
    }


@torch.no_grad()
def decode_vertices_with_proposal_keep(
    dense_output: dict[str, torch.Tensor],
    keep_logits: torch.Tensor,
    config: ProposalKeepConfig,
    *,
    keep_threshold: float = 0.5,
    minimum_dense_fraction: float = 0.0,
    dense_fallback_threshold: float = 0.3,
    dense_fallback_maximum: int = 128,
) -> list[list[dict[str, float | bool]]]:
    if not 0.0 < keep_threshold < 1.0:
        raise ValueError("Proposal keep threshold must be in (0, 1)")
    if not 0.0 <= minimum_dense_fraction <= 1.0:
        raise ValueError("Minimum dense fraction must be in [0, 1]")
    if not 0.0 < dense_fallback_threshold < 1.0:
        raise ValueError("Dense fallback threshold must be in (0, 1)")
    if dense_fallback_maximum < 1:
        raise ValueError("Dense fallback maximum must be positive")
    indices = proposal_indices(dense_output, config)
    heatmap = dense_output["vertex_heatmap_logits"].sigmoid()
    aromatic = dense_output["vertex_aromatic_logits"].sigmoid()
    keep_probabilities = keep_logits.sigmoid()
    height, width = heatmap.shape[-2:]
    decoded = []
    for batch_index, batch_indices in enumerate(indices):
        if not len(batch_indices):
            decoded.append([])
            continue
        y = torch.div(batch_indices, width, rounding_mode="floor")
        x = batch_indices % width
        keep_scores = keep_probabilities[batch_index, 0, y, x]
        selected = keep_scores >= keep_threshold
        if minimum_dense_fraction:
            dense_scores = heatmap[batch_index, 0, y, x]
            baseline_positions = torch.nonzero(
                dense_scores >= dense_fallback_threshold,
                as_tuple=False,
            ).flatten()[:dense_fallback_maximum]
            required = math.ceil(len(baseline_positions) * minimum_dense_fraction)
            missing = max(required - int(selected.sum()), 0)
            if missing:
                restorable = baseline_positions[~selected[baseline_positions]]
                restore_order = keep_scores[restorable].argsort(descending=True)
                selected[restorable[restore_order[:missing]]] = True
        order = keep_scores.masked_fill(~selected, -1.0).argsort(descending=True)
        points = proposal_points(dense_output, batch_indices, batch_index)
        vertices = []
        for position in order.tolist():
            keep_score = float(keep_scores[position])
            if not bool(selected[position]):
                break
            cell_y = int(y[position])
            cell_x = int(x[position])
            vertices.append(
                {
                    "x": float(points[position, 0]),
                    "y": float(points[position, 1]),
                    "score": keep_score,
                    "dense_score": float(heatmap[batch_index, 0, cell_y, cell_x]),
                    "aromatic": bool(
                        aromatic[batch_index, 0, cell_y, cell_x] >= 0.5
                    ),
                }
            )
        decoded.append(vertices)
    return decoded
