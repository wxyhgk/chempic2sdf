"""Independent text-object center and extent detection."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.vertex import build_vertex_heatmap, focal_heatmap_loss


class TextObjectCenterHead(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.heatmap = nn.Conv2d(hidden_channels, 1, 1)
        self.offset = nn.Conv2d(hidden_channels, 2, 1)
        self.extent = nn.Conv2d(hidden_channels, 2, 1)
        nn.init.constant_(self.heatmap.bias, -4.6)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "heatmap_logits": self.heatmap(hidden),
            "offset_logits": self.offset(hidden),
            "extent_logits": self.extent(hidden),
        }


class TextObjectBBoxHead(TextObjectCenterHead):
    """Preserve the atom anchor while predicting an asymmetric text-box center."""

    def __init__(self, input_channels: int, hidden_channels: int = 64) -> None:
        super().__init__(input_channels, hidden_channels)
        self.bbox_offset = nn.Conv2d(hidden_channels, 2, 1)
        nn.init.zeros_(self.bbox_offset.weight)
        nn.init.zeros_(self.bbox_offset.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "heatmap_logits": self.heatmap(hidden),
            "offset_logits": self.offset(hidden),
            "extent_logits": self.extent(hidden),
            "bbox_offset_logits": self.bbox_offset(hidden),
        }


def text_object_loss(
    output: dict[str, torch.Tensor],
    centers: torch.Tensor,
    mask: torch.Tensor,
    extents: torch.Tensor,
    bbox_centers: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    height, width = output["heatmap_logits"].shape[-2:]
    heatmap_targets = build_vertex_heatmap(centers, mask, height, width)
    heatmap_loss = focal_heatmap_loss(output["heatmap_logits"], heatmap_targets)
    cell_x = (centers[..., 0] * width).clamp(0, width - 1e-4)
    cell_y = (centers[..., 1] * height).clamp(0, height - 1e-4)
    index_x = cell_x.long()
    index_y = cell_y.long()
    batch = torch.arange(centers.size(0), device=centers.device)[:, None]
    predicted_offsets = output["offset_logits"].sigmoid().permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ]
    target_offsets = torch.stack((cell_x.frac(), cell_y.frac()), dim=-1)
    offset_loss = (
        F.smooth_l1_loss(predicted_offsets[mask], target_offsets[mask])
        if mask.any()
        else heatmap_loss * 0.0
    )
    predicted_extents = output["extent_logits"].sigmoid().permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ] * 0.25
    extent_loss = (
        F.smooth_l1_loss(predicted_extents[mask], extents[mask])
        if mask.any()
        else heatmap_loss * 0.0
    )
    bbox_offset_loss = heatmap_loss * 0.0
    if "bbox_offset_logits" in output and bbox_centers is not None:
        predicted_bbox_ratios = output["bbox_offset_logits"].tanh().permute(
            0, 2, 3, 1
        )[batch, index_y, index_x]
        target_bbox_ratios = (
            (bbox_centers - centers) / extents.clamp_min(1.0 / 768.0)
        ).clamp(-1.0, 1.0)
        bbox_offset_loss = (
            F.smooth_l1_loss(
                predicted_bbox_ratios[mask], target_bbox_ratios[mask]
            )
            if mask.any()
            else heatmap_loss * 0.0
        )
    return {
        "loss": (
            heatmap_loss
            + 4.0 * offset_loss
            + 4.0 * extent_loss
            + 4.0 * bbox_offset_loss
        ),
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "extent_loss": extent_loss,
        "bbox_offset_loss": bbox_offset_loss,
    }


@torch.no_grad()
def decode_text_objects(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.3,
    maximum_objects: int = 128,
) -> list[list[dict[str, float]]]:
    heatmap = output["heatmap_logits"].sigmoid()
    peaks = heatmap == F.max_pool2d(heatmap, 3, stride=1, padding=1)
    scores_map = heatmap * peaks
    offsets = output["offset_logits"].sigmoid()
    extents = output["extent_logits"].sigmoid() * 0.25
    bbox_offsets = (
        output["bbox_offset_logits"].tanh()
        if "bbox_offset_logits" in output
        else torch.zeros_like(offsets)
    )
    height, width = heatmap.shape[-2:]
    decoded = []
    for batch_index in range(heatmap.size(0)):
        scores, indices = scores_map[batch_index, 0].flatten().topk(
            min(maximum_objects, height * width)
        )
        objects = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < threshold:
                break
            y, x = divmod(index, width)
            anchor_x = (x + float(offsets[batch_index, 0, y, x])) / width
            anchor_y = (y + float(offsets[batch_index, 1, y, x])) / height
            half_width = float(extents[batch_index, 0, y, x])
            half_height = float(extents[batch_index, 1, y, x])
            objects.append({
                "x": anchor_x,
                "y": anchor_y,
                "bbox_x": anchor_x
                + float(bbox_offsets[batch_index, 0, y, x]) * half_width,
                "bbox_y": anchor_y
                + float(bbox_offsets[batch_index, 1, y, x]) * half_height,
                "half_width": half_width,
                "half_height": half_height,
                "score": float(score),
                "aromatic": False,
            })
        decoded.append(objects)
    return decoded
