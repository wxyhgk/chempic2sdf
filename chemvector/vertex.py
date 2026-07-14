"""Dense visual vertex targets, losses, and decoding."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


class VertexCollator:
    def __init__(self, *, image_size: int = 768) -> None:
        self.image_size = image_size

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        maximum_atoms = max(len(sample["render"]["atoms"]) for sample in samples)
        anchors = torch.zeros((len(samples), maximum_atoms, 2), dtype=torch.float32)
        mask = torch.zeros((len(samples), maximum_atoms), dtype=torch.bool)
        aromatic = torch.zeros((len(samples), maximum_atoms), dtype=torch.float32)
        for sample_index, sample in enumerate(samples):
            render = sample["render"]
            for position in render["atoms"]:
                atom_id = position["atom_id"]
                anchors[sample_index, atom_id] = torch.tensor(
                    (
                        position["image_x"] / render["width"],
                        position["image_y"] / render["height"],
                    )
                )
                mask[sample_index, atom_id] = True
                aromatic[sample_index, atom_id] = float(
                    sample["chemmol"]["atoms"][atom_id]["aromatic"]
                )
        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "anchor_points": anchors,
            "anchor_mask": mask,
            "aromatic_targets": aromatic,
            "keys": [sample["key"] for sample in samples],
        }


def build_vertex_heatmap(
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    target = torch.zeros(
        (anchor_points.size(0), 1, height, width),
        device=anchor_points.device,
        dtype=anchor_points.dtype,
    )
    cell_x = (anchor_points[..., 0] * width).clamp(0, width - 1e-4).long()
    cell_y = (anchor_points[..., 1] * height).clamp(0, height - 1e-4).long()
    batch = torch.arange(anchor_points.size(0), device=anchor_points.device)[:, None]
    batch = batch.expand_as(anchor_mask)
    target[batch[anchor_mask], 0, cell_y[anchor_mask], cell_x[anchor_mask]] = 1.0
    return target


def vertex_detection_loss(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    aromatic_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    heatmap_logits = output["vertex_heatmap_logits"]
    height, width = heatmap_logits.shape[-2:]
    heatmap_targets = build_vertex_heatmap(
        anchor_points,
        anchor_mask,
        height,
        width,
    )
    probabilities = heatmap_logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = heatmap_targets == 1
    negative = heatmap_targets < 1
    positive_loss = -torch.log(probabilities) * (1.0 - probabilities) ** 2 * positive
    negative_loss = (
        -torch.log(1.0 - probabilities)
        * probabilities**2
        * (1.0 - heatmap_targets) ** 4
        * negative
    )
    positive_count = positive.sum().clamp_min(1)
    heatmap_loss = (positive_loss.sum() + negative_loss.sum()) / positive_count

    cell_x = (anchor_points[..., 0] * width).clamp(0, width - 1e-4)
    cell_y = (anchor_points[..., 1] * height).clamp(0, height - 1e-4)
    index_x = cell_x.long()
    index_y = cell_y.long()
    batch = torch.arange(anchor_points.size(0), device=anchor_points.device)[:, None]
    predicted_offsets = output["vertex_offset_logits"].sigmoid().permute(0, 2, 3, 1)[
        batch,
        index_y,
        index_x,
    ]
    target_offsets = torch.stack((cell_x - index_x, cell_y - index_y), dim=-1)
    offset_loss = F.smooth_l1_loss(
        predicted_offsets[anchor_mask],
        target_offsets[anchor_mask],
    )
    predicted_aromatic = output["vertex_aromatic_logits"][:, 0][batch, index_y, index_x]
    aromatic_loss = F.binary_cross_entropy_with_logits(
        predicted_aromatic[anchor_mask],
        aromatic_targets[anchor_mask],
    )
    return {
        "loss": heatmap_loss + offset_loss + 0.5 * aromatic_loss,
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "aromatic_loss": aromatic_loss,
    }


@torch.no_grad()
def decode_vertices(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.3,
    maximum_vertices: int = 128,
) -> list[list[dict[str, float | bool]]]:
    heatmap = output["vertex_heatmap_logits"].sigmoid()
    peaks = heatmap == F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * peaks
    offsets = output["vertex_offset_logits"].sigmoid()
    aromatic = output["vertex_aromatic_logits"].sigmoid()
    height, width = heatmap.shape[-2:]
    decoded = []
    for batch_index in range(heatmap.size(0)):
        scores, indices = heatmap[batch_index, 0].flatten().topk(
            min(maximum_vertices, height * width)
        )
        vertices = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < threshold:
                break
            y, x = divmod(index, width)
            offset_x = float(offsets[batch_index, 0, y, x])
            offset_y = float(offsets[batch_index, 1, y, x])
            vertices.append(
                {
                    "x": (x + offset_x) / width,
                    "y": (y + offset_y) / height,
                    "score": score,
                    "aromatic": bool(aromatic[batch_index, 0, y, x] >= 0.5),
                }
            )
        decoded.append(vertices)
    return decoded
