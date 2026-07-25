"""Dense visual vertex targets, losses, and decoding."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from chemvector.data import rendered_atom_labels


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
        roles = torch.zeros((len(samples), maximum_atoms), dtype=torch.long)
        for sample_index, sample in enumerate(samples):
            render = sample["render"]
            rendered_labels = rendered_atom_labels(sample)
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
                roles[sample_index, atom_id] = int(rendered_labels[atom_id][1])
        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "anchor_points": anchors,
            "anchor_mask": mask,
            "aromatic_targets": aromatic,
            "role_targets": roles,
            "keys": [sample["key"] for sample in samples],
        }


def build_vertex_heatmap(
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    height: int,
    width: int,
    radius_cells: int = 0,
) -> torch.Tensor:
    if radius_cells < 0:
        raise ValueError("Heatmap radius must be nonnegative")
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
    if radius_cells:
        sigma = max(radius_cells / 3.0, 1e-6)
        for batch_index, x, y in zip(
            batch[anchor_mask].tolist(),
            cell_x[anchor_mask].tolist(),
            cell_y[anchor_mask].tolist(),
        ):
            left = max(0, x - radius_cells)
            right = min(width, x + radius_cells + 1)
            top = max(0, y - radius_cells)
            bottom = min(height, y + radius_cells + 1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(top, bottom, device=target.device),
                torch.arange(left, right, device=target.device),
                indexing="ij",
            )
            gaussian = torch.exp(
                -((grid_x - x).square() + (grid_y - y).square())
                / (2.0 * sigma * sigma)
            ).to(target.dtype)
            target[batch_index, 0, top:bottom, left:right] = torch.maximum(
                target[batch_index, 0, top:bottom, left:right],
                gaussian,
            )
    return target


def build_vertex_role_heatmap(
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    role_targets: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    target = torch.zeros(
        (anchor_points.size(0), 2, height, width),
        device=anchor_points.device,
        dtype=anchor_points.dtype,
    )
    cell_x = (anchor_points[..., 0] * width).clamp(0, width - 1e-4).long()
    cell_y = (anchor_points[..., 1] * height).clamp(0, height - 1e-4).long()
    batch = torch.arange(anchor_points.size(0), device=anchor_points.device)[:, None]
    batch = batch.expand_as(anchor_mask)
    target[
        batch[anchor_mask],
        role_targets[anchor_mask].long(),
        cell_y[anchor_mask],
        cell_x[anchor_mask],
    ] = 1.0
    return target


def focal_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    probabilities = logits.float().sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = targets == 1
    negative = targets < 1
    positive_loss = -torch.log(probabilities) * (1.0 - probabilities) ** 2 * positive
    negative_loss = (
        -torch.log(1.0 - probabilities)
        * probabilities**2
        * (1.0 - targets) ** 4
        * negative
    )
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1)


def vertex_detection_loss(
    output: dict[str, torch.Tensor],
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
    aromatic_targets: torch.Tensor,
    role_targets: torch.Tensor | None = None,
    role_loss_weight: float = 0.0,
    heatmap_radius_cells: int = 0,
) -> dict[str, torch.Tensor]:
    if role_loss_weight < 0:
        raise ValueError("Role loss weight must be nonnegative")
    heatmap_logits = output["vertex_heatmap_logits"]
    height, width = heatmap_logits.shape[-2:]
    heatmap_targets = build_vertex_heatmap(
        anchor_points,
        anchor_mask,
        height,
        width,
        heatmap_radius_cells,
    )
    heatmap_loss = focal_heatmap_loss(heatmap_logits, heatmap_targets)
    role_heatmap_loss = heatmap_loss * 0.0
    if role_targets is not None and role_loss_weight:
        role_heatmap_targets = build_vertex_role_heatmap(
            anchor_points,
            anchor_mask,
            role_targets,
            height,
            width,
        )
        role_heatmap_loss = focal_heatmap_loss(
            output["vertex_role_heatmap_logits"],
            role_heatmap_targets,
        )

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
        "loss": (
            heatmap_loss
            + offset_loss
            + 0.5 * aromatic_loss
            + role_loss_weight * role_heatmap_loss
        ),
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "aromatic_loss": aromatic_loss,
        "role_heatmap_loss": role_heatmap_loss,
    }


def vertex_teacher_consistency_loss(
    output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Preserve a frozen detector's dense outputs on replay images."""
    required = (
        "vertex_heatmap_logits",
        "vertex_offset_logits",
        "vertex_aromatic_logits",
    )
    for key in required:
        if output[key].shape != teacher_output[key].shape:
            raise ValueError(f"Teacher vertex output shape differs for {key}")

    teacher_heatmap = teacher_output["vertex_heatmap_logits"].detach()
    heatmap_dense_loss = F.smooth_l1_loss(
        output["vertex_heatmap_logits"],
        teacher_heatmap,
    )
    peak_weights = teacher_heatmap.sigmoid().square()
    weight_sum = peak_weights.sum().clamp_min(1e-6)
    heatmap_peak_loss = (
        F.smooth_l1_loss(
            output["vertex_heatmap_logits"],
            teacher_heatmap,
            reduction="none",
        )
        * peak_weights
    ).sum() / weight_sum
    offset_loss = (
        F.smooth_l1_loss(
            output["vertex_offset_logits"].sigmoid(),
            teacher_output["vertex_offset_logits"].detach().sigmoid(),
            reduction="none",
        )
        * peak_weights
    ).sum() / (weight_sum * output["vertex_offset_logits"].size(1))
    aromatic_loss = (
        F.smooth_l1_loss(
            output["vertex_aromatic_logits"].sigmoid(),
            teacher_output["vertex_aromatic_logits"].detach().sigmoid(),
            reduction="none",
        )
        * peak_weights
    ).sum() / weight_sum
    loss = heatmap_dense_loss + heatmap_peak_loss + offset_loss + aromatic_loss
    return {
        "teacher_loss": loss,
        "teacher_heatmap_loss": heatmap_dense_loss + heatmap_peak_loss,
        "teacher_offset_loss": offset_loss,
        "teacher_aromatic_loss": aromatic_loss,
    }


@torch.no_grad()
def decode_vertices(
    output: dict[str, torch.Tensor],
    *,
    threshold: float = 0.3,
    maximum_vertices: int = 128,
    role_aware: bool = False,
    nms_kernel_size: int = 3,
) -> list[list[dict[str, float | bool]]]:
    if nms_kernel_size < 1 or nms_kernel_size % 2 == 0:
        raise ValueError("NMS kernel size must be a positive odd integer")
    if role_aware:
        role_probabilities = output["vertex_role_heatmap_logits"].sigmoid()
        heatmap, role_ids = role_probabilities.max(dim=1, keepdim=True)
    else:
        heatmap = output["vertex_heatmap_logits"].sigmoid()
        role_ids = None
    raw_heatmap = heatmap
    peaks = raw_heatmap == F.max_pool2d(
        raw_heatmap,
        kernel_size=nms_kernel_size,
        stride=1,
        padding=nms_kernel_size // 2,
    )
    heatmap = raw_heatmap * peaks
    peak3 = raw_heatmap == F.max_pool2d(
        raw_heatmap, kernel_size=3, stride=1, padding=1
    )
    peak5 = raw_heatmap == F.max_pool2d(
        raw_heatmap, kernel_size=5, stride=1, padding=2
    )
    offsets = output["vertex_offset_logits"].sigmoid()
    aromatic = output["vertex_aromatic_logits"].sigmoid()
    height, width = heatmap.shape[-2:]
    padded_heatmap = F.pad(raw_heatmap, (2, 2, 2, 2))
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
            local_patch = padded_heatmap[
                batch_index, 0, y : y + 5, x : x + 5
            ].flatten()
            vertices.append(
                {
                    "x": (x + offset_x) / width,
                    "y": (y + offset_y) / height,
                    "score": score,
                    "aromatic": bool(aromatic[batch_index, 0, y, x] >= 0.5),
                    "grid_x": int(x),
                    "grid_y": int(y),
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "nms3_peak": bool(peak3[batch_index, 0, y, x]),
                    "nms5_peak": bool(peak5[batch_index, 0, y, x]),
                    "local_heatmap_patch": tuple(
                        float(value) for value in local_patch
                    ),
                    "role": (
                        "visible_text"
                        if role_ids is not None and int(role_ids[batch_index, 0, y, x]) == 1
                        else "implicit_junction"
                    ),
                }
            )
        decoded.append(vertices)
    return decoded
