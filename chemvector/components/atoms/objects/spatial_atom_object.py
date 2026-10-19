"""Spatially anchored complete AtomObject proposals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.dense_bond_object import DilatedResidualBlock
from chemvector.vertex import build_vertex_heatmap, focal_heatmap_loss


@dataclass(frozen=True, slots=True)
class SpatialAtomObjectConfig:
    input_channels: int = 96
    hidden_channels: int = 128
    maximum_atoms: int = 128
    text_center_radius: float = 64.0 / 768.0
    maximum_text_extent: float = 96.0 / 768.0

    def __post_init__(self) -> None:
        if min(self.input_channels, self.hidden_channels, self.maximum_atoms) < 1:
            raise ValueError("Spatial AtomObject dimensions must be positive")
        if not 0.0 < self.text_center_radius < 0.5:
            raise ValueError("text_center_radius must be in (0, 0.5)")
        if not 0.0 < self.maximum_text_extent < 0.5:
            raise ValueError("maximum_text_extent must be in (0, 0.5)")


class SpatialAtomObjectDetector(nn.Module):
    """Predict anchor, role and text geometry from one shared spatial object cell."""

    def __init__(self, config: SpatialAtomObjectConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, 3, padding=1),
            nn.GroupNorm(1, config.hidden_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            *(
                DilatedResidualBlock(config.hidden_channels, dilation)
                for dilation in (1, 2, 4, 8)
            )
        )
        self.heatmap = nn.Conv2d(config.hidden_channels, 1, 1)
        self.anchor_offset = nn.Conv2d(config.hidden_channels, 2, 1)
        self.role = nn.Conv2d(config.hidden_channels, 2, 1)
        self.text_center_delta = nn.Conv2d(config.hidden_channels, 2, 1)
        self.text_extent = nn.Conv2d(config.hidden_channels, 2, 1)
        self.count = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_channels, config.maximum_atoms + 1),
        )
        nn.init.constant_(self.heatmap.bias, -4.6)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.context(self.input_projection(features))
        batch, _, height, width = hidden.shape
        offsets = self.anchor_offset(hidden).sigmoid()
        y, x = torch.meshgrid(
            torch.arange(height, device=hidden.device, dtype=hidden.dtype),
            torch.arange(width, device=hidden.device, dtype=hidden.dtype),
            indexing="ij",
        )
        anchors = torch.stack(
            (
                (x[None] + offsets[:, 0]) / width,
                (y[None] + offsets[:, 1]) / height,
            ),
            dim=1,
        )
        text_centers = (
            anchors
            + self.text_center_delta(hidden).tanh()
            * self.config.text_center_radius
        ).clamp(0.0, 1.0)
        text_extents = (
            self.text_extent(hidden).sigmoid() * self.config.maximum_text_extent
        )
        return {
            "object_features": hidden,
            "heatmap_logits": self.heatmap(hidden),
            "anchor_offset_logits": self.anchor_offset(hidden),
            "anchors": anchors,
            "role_logits": self.role(hidden),
            "text_centers": text_centers,
            "text_extents": text_extents,
            "count_logits": self.count(hidden),
        }


def spatial_atom_object_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    heatmap_radius_cells: int = 1,
    text_center_consistency_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the spatial atom-object objective.

    ``text_center_consistency_weight`` is intentionally opt-in.  The chemical
    attachment anchor remains the sole heatmap/ownership target; when enabled,
    visible text contributes only a weak translation-invariant geometry term.
    This prevents a text glyph center from becoming an atom candidate while
    still teaching the head how the glyph is displaced from its anchor.
    """
    if text_center_consistency_weight < 0.0:
        raise ValueError("text_center_consistency_weight must be nonnegative")
    anchors = targets["atom_anchors"]
    mask = targets["atom_mask"]
    height, width = output["heatmap_logits"].shape[-2:]
    heatmap = build_vertex_heatmap(
        anchors,
        mask,
        height,
        width,
        radius_cells=heatmap_radius_cells,
    )
    heatmap_loss = focal_heatmap_loss(output["heatmap_logits"], heatmap)
    cell_x = (anchors[..., 0] * width).clamp(0, width - 1e-4)
    cell_y = (anchors[..., 1] * height).clamp(0, height - 1e-4)
    index_x = cell_x.long()
    index_y = cell_y.long()
    batch = torch.arange(anchors.size(0), device=anchors.device)[:, None]
    positive_cells = torch.stack((batch.expand_as(index_x), index_y, index_x), dim=-1)
    for batch_index in range(anchors.size(0)):
        cells = positive_cells[batch_index, mask[batch_index]]
        if len(cells) != len(torch.unique(cells, dim=0)):
            raise ValueError("Multiple AtomObjects occupy one spatial cell")
    predicted_offsets = output["anchor_offset_logits"].sigmoid().permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ]
    expected_offsets = torch.stack((cell_x - index_x, cell_y - index_y), dim=-1)
    anchor_offset_loss = F.smooth_l1_loss(
        predicted_offsets[mask], expected_offsets[mask]
    )
    role_logits = output["role_logits"].permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ]
    role_loss = F.cross_entropy(role_logits[mask], targets["atom_roles"][mask])
    text = mask & targets["atom_text_mask"]
    predicted_centers = output["text_centers"].permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ]
    predicted_anchors = output["anchors"].permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ]
    predicted_extents = output["text_extents"].permute(0, 2, 3, 1)[
        batch, index_y, index_x
    ]
    expected_centers = targets["atom_text_center"]
    expected_extents = targets["atom_text_bbox"][..., 2:] / 2.0
    zero = heatmap_loss * 0.0
    text_center_loss = (
        F.smooth_l1_loss(
            predicted_centers[text],
            expected_centers[text],
            beta=4.0 / 768.0,
        )
        if bool(text.any())
        else zero
    )
    text_extent_loss = (
        F.smooth_l1_loss(
            predicted_extents[text],
            expected_extents[text],
            beta=2.0 / 768.0,
        )
        if bool(text.any())
        else zero
    )
    text_center_consistency_loss = (
        F.smooth_l1_loss(
            predicted_centers[text] - predicted_anchors[text],
            expected_centers[text] - anchors[text],
            beta=4.0 / 768.0,
        )
        if bool(text.any())
        else zero
    )
    count_targets = mask.sum(dim=1).long()
    count_loss = F.cross_entropy(output["count_logits"], count_targets)
    loss = (
        heatmap_loss
        + 2.0 * anchor_offset_loss
        + role_loss
        + 8.0 * text_center_loss
        + 8.0 * text_extent_loss
        + text_center_consistency_weight * text_center_consistency_loss
        + 0.1 * count_loss
    )
    return {
        "loss": loss,
        "heatmap_loss": heatmap_loss,
        "anchor_offset_loss": anchor_offset_loss,
        "role_loss": role_loss,
        "text_center_loss": text_center_loss,
        "text_center_consistency_loss": text_center_consistency_loss,
        "text_extent_loss": text_extent_loss,
        "count_loss": count_loss,
    }


def atom_local_peak_margin_loss(
    heatmap_logits: torch.Tensor,
    positive_cells: torch.Tensor,
    *,
    margin: float = 0.75,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Make each target cell beat nearby non-target peaks by a fixed margin.

    ``positive_cells`` is a dense boolean mask with shape ``[B, 1, H, W]``.
    All target cells are excluded from the local negative maximum, so nearby
    atoms do not become each other's negatives.  This is intentionally a
    separate objective from the historical focal heatmap loss.
    """
    if heatmap_logits.ndim != 4 or heatmap_logits.size(1) != 1:
        raise ValueError("Heatmap logits must have shape [B, 1, H, W]")
    if positive_cells.shape != heatmap_logits.shape or positive_cells.dtype is not torch.bool:
        raise ValueError("Positive-cell mask must match logits and be boolean")
    if margin < 0.0 or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Margin must be nonnegative and kernel size must be odd >= 3")
    if not bool(positive_cells.any()):
        return heatmap_logits.sum() * 0.0
    radius = kernel_size // 2
    patches = F.unfold(heatmap_logits, kernel_size=kernel_size, padding=radius)
    masks = F.unfold(positive_cells.float(), kernel_size=kernel_size, padding=radius).bool()
    negative_max = patches.masked_fill(masks, float("-inf")).amax(dim=1)
    positives = heatmap_logits[:, 0][positive_cells[:, 0]]
    local_negative = negative_max.view_as(heatmap_logits[:, 0])[positive_cells[:, 0]]
    finite = torch.isfinite(local_negative)
    if not bool(finite.any()):
        return heatmap_logits.sum() * 0.0
    return F.softplus(margin + local_negative[finite] - positives[finite]).mean()


def decode_spatial_atom_objects(
    output: dict[str, torch.Tensor], counts: torch.Tensor
) -> list[list[dict]]:
    decoded = []
    scores = output["heatmap_logits"].sigmoid()[:, 0]
    batch, height, width = scores.shape
    for batch_index in range(batch):
        count = int(counts[batch_index])
        values, indices = scores[batch_index].flatten().topk(count)
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices % width
        items = []
        for score, cell_y, cell_x in zip(values, y, x):
            anchor = output["anchors"][batch_index, :, cell_y, cell_x]
            center = output["text_centers"][batch_index, :, cell_y, cell_x]
            extent = output["text_extents"][batch_index, :, cell_y, cell_x]
            items.append(
                {
                    "score": float(score.detach()),
                    "anchor": anchor.detach(),
                    "role": int(
                        output["role_logits"][batch_index, :, cell_y, cell_x].argmax()
                    ),
                    "text_center": center.detach(),
                    "text_extent": extent.detach(),
                    "cell": (int(cell_x), int(cell_y)),
                }
            )
        decoded.append(items)
    return decoded
