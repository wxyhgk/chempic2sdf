"""One fixed dense top-K/NMS decoder shared by vertex shadow and baseline."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def decode_dense(output: dict[str, torch.Tensor], *, threshold: float = .05,
                 maximum_vertices: int = 128, nms_kernel_size: int = 3,
                 with_scores: bool = False) -> list[list[tuple]]:
    heatmap = output["vertex_heatmap_logits"].sigmoid()
    offsets = output["vertex_offset_logits"].sigmoid()
    b, _, h, w = heatmap.shape
    peaks = heatmap == F.max_pool2d(heatmap, nms_kernel_size, stride=1, padding=nms_kernel_size // 2)
    masked = heatmap * peaks
    result = []
    for bi in range(b):
        scores, indices = masked[bi, 0].flatten().topk(min(maximum_vertices, h * w))
        points = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            if score < threshold:
                break
            y, x = divmod(index, w)
            ox = float(offsets[bi, 0, y, x]); oy = float(offsets[bi, 1, y, x])
            point = ((x + ox) / w, (y + oy) / h)
            points.append(point + (float(score),) if with_scores else point)
        result.append(points)
    return result


def raw_to_dense(raw: dict[str, torch.Tensor], *, height: int = 192, width: int = 192) -> dict[str, torch.Tensor]:
    """Encode old raw proposals into the same cell/offset representation."""
    device = raw["atom_scores"].device
    heat = torch.full((1, 1, height, width), -20.0, device=device)
    off = torch.zeros((1, 2, height, width), device=device)
    scores = raw["atom_scores"][0].float()
    coords = raw["atom_geometry"][0, :, :2].float()
    for i in torch.where(raw["atom_mask"][0])[0].tolist():
        x = float(coords[i, 0]); y = float(coords[i, 1]); gx = min(width - 1, max(0, int(x * width))); gy = min(height - 1, max(0, int(y * height)))
        score = scores[i].clamp(1e-5, 1 - 1e-5)
        if score > heat[0, 0, gy, gx].sigmoid():
            heat[0, 0, gy, gx] = torch.logit(score)
            off[0, 0, gy, gx] = min(1.0, max(0.0, x * width - gx))
            off[0, 1, gy, gx] = min(1.0, max(0.0, y * height - gy))
    return {"vertex_heatmap_logits": heat, "vertex_offset_logits": torch.logit(off.clamp(1e-5, 1 - 1e-5))}
