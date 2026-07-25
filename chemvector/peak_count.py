"""Per-image vertex count calibration from sorted dense heatmap peaks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class PeakCountConfig:
    maximum_vertices: int = 128
    hidden_dim: int = 256
    dropout: float = 0.0
    soft_count_temperature: float = 0.03
    objective: str = "classification"

    def __post_init__(self) -> None:
        if min(self.maximum_vertices, self.hidden_dim) < 1:
            raise ValueError("Peak count dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Peak count dropout must be in [0, 1)")
        if self.soft_count_temperature <= 0:
            raise ValueError("Soft count temperature must be positive")
        if self.objective not in {"classification", "regression"}:
            raise ValueError("Unsupported peak count objective")


def sorted_peak_probabilities(
    output: dict[str, torch.Tensor],
    maximum_vertices: int,
) -> torch.Tensor:
    heatmap = output["vertex_heatmap_logits"].float().sigmoid()
    peaks = heatmap == F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peak_scores = heatmap.masked_fill(~peaks, 0.0).flatten(1)
    return peak_scores.topk(
        min(maximum_vertices, peak_scores.size(1)),
        dim=1,
    ).values


def peak_count_features(
    output: dict[str, torch.Tensor],
    config: PeakCountConfig,
) -> torch.Tensor:
    scores = sorted_peak_probabilities(output, config.maximum_vertices)
    thresholds = torch.linspace(
        0.1,
        0.9,
        9,
        device=scores.device,
        dtype=scores.dtype,
    )
    soft_counts = torch.sigmoid(
        (scores[:, :, None] - thresholds[None, None, :])
        / config.soft_count_temperature
    ).sum(dim=1) / config.maximum_vertices
    return torch.cat((scores, soft_counts), dim=-1)


class PeakCountCalibrator(nn.Module):
    def __init__(self, config: PeakCountConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.maximum_vertices + 9
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_dim,
                config.maximum_vertices + 1
                if config.objective == "classification"
                else 1,
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected = self.config.maximum_vertices + 9
        if features.ndim != 2 or features.size(1) != expected:
            raise ValueError("Peak count features have incompatible shape")
        return self.network(features)


def peak_count_loss(
    count_logits: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    target_counts = anchor_mask.sum(dim=1).long()
    if count_logits.size(-1) == 1:
        return F.smooth_l1_loss(
            count_logits[:, 0],
            target_counts.to(count_logits.dtype),
            beta=1.0,
        )
    if int(target_counts.max()) >= count_logits.size(-1):
        raise ValueError("Target count exceeds peak calibrator capacity")
    return F.cross_entropy(count_logits, target_counts)


@torch.no_grad()
def predict_peak_counts(
    count_output: torch.Tensor,
    maximum_vertices: int,
) -> torch.Tensor:
    if count_output.size(-1) == 1:
        return count_output[:, 0].round().clamp(0, maximum_vertices).long()
    return count_output.argmax(dim=-1).clamp(0, maximum_vertices)


@torch.no_grad()
def decode_vertices_by_count(
    output: dict[str, torch.Tensor],
    predicted_counts: torch.Tensor,
    *,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    heatmap = output["vertex_heatmap_logits"].sigmoid()
    peaks = heatmap == F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * peaks
    offsets = output["vertex_offset_logits"].sigmoid()
    aromatic = output["vertex_aromatic_logits"].sigmoid()
    height, width = heatmap.shape[-2:]
    decoded = []
    for batch_index, predicted_count in enumerate(predicted_counts.tolist()):
        count = min(max(int(predicted_count), 0), maximum_vertices, height * width)
        scores, indices = heatmap[batch_index, 0].flatten().topk(count)
        vertices = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            y, x = divmod(index, width)
            vertices.append(
                {
                    "x": (x + float(offsets[batch_index, 0, y, x])) / width,
                    "y": (y + float(offsets[batch_index, 1, y, x])) / height,
                    "score": score,
                    "aromatic": bool(aromatic[batch_index, 0, y, x] >= 0.5),
                }
            )
        decoded.append(vertices)
    return decoded
