"""Deterministic regional maxima for FP32 spatial maps."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch


_NEIGHBOR_OFFSETS = {
    4: ((-1, 0), (0, -1), (0, 1), (1, 0)),
    8: (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ),
}


@torch.no_grad()
def regional_maxima_mask(
    values: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    connectivity: int = 8,
) -> torch.Tensor:
    """Mark complete flat zones whose valid boundary contains no higher value.

    The final two dimensions are interpreted as the spatial plane. Any leading
    dimensions are independent maps. Equal values are compared exactly in FP32;
    no tolerance, score threshold, NMS window, or pixel-radius clustering is
    applied.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("Regional maxima values must be a torch.Tensor")
    if values.dtype != torch.float32:
        raise ValueError("Regional maxima values must use FP32")
    if values.ndim < 2 or min(values.shape[-2:]) < 1:
        raise ValueError("Regional maxima values require non-empty spatial dimensions")
    if connectivity not in _NEIGHBOR_OFFSETS:
        raise ValueError("Regional maxima connectivity must be 4 or 8")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Regional maxima values must be finite")
    if valid_mask is None:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    elif not isinstance(valid_mask, torch.Tensor):
        raise TypeError("Regional maxima valid_mask must be a torch.Tensor")
    elif valid_mask.dtype != torch.bool:
        raise ValueError("Regional maxima valid_mask must be boolean")
    elif valid_mask.shape != values.shape:
        raise ValueError("Regional maxima values and valid_mask shapes differ")
    elif valid_mask.device != values.device:
        raise ValueError("Regional maxima values and valid_mask devices differ")

    height, width = values.shape[-2:]
    maps = values.detach().cpu().contiguous().reshape(-1, height, width).numpy()
    valid_maps = (
        valid_mask.detach().cpu().contiguous().reshape(-1, height, width).numpy()
    )
    maxima = np.zeros_like(valid_maps, dtype=np.bool_)
    offsets = _NEIGHBOR_OFFSETS[connectivity]
    structure = (
        ndimage.generate_binary_structure(2, 1)
        if connectivity == 4
        else np.ones((3, 3), dtype=np.bool_)
    )

    for map_index, (current, valid) in enumerate(zip(maps, valid_maps)):
        working = np.where(valid, current, -np.inf)
        local = valid & (
            current
            == ndimage.maximum_filter(
                working, footprint=structure, mode="constant", cval=-np.inf
            )
        )

        # A non-maximum flat zone can have an interior that passes the local
        # maximum test. Reject every local component touching an equal-valued
        # valid neighbor that failed that test. A true regional maximum keeps
        # its complete flat zone, so it has no such escape edge.
        incomplete = np.zeros_like(local)
        for delta_y, delta_x in offsets:
            current_y = slice(max(0, -delta_y), min(height, height - delta_y))
            current_x = slice(max(0, -delta_x), min(width, width - delta_x))
            neighbor_y = slice(max(0, delta_y), min(height, height + delta_y))
            neighbor_x = slice(max(0, delta_x), min(width, width + delta_x))
            incomplete[current_y, current_x] |= (
                local[current_y, current_x]
                & valid[neighbor_y, neighbor_x]
                & (current[current_y, current_x] == current[neighbor_y, neighbor_x])
                & ~local[neighbor_y, neighbor_x]
            )

        labels, component_count = ndimage.label(local, structure=structure)
        rejected = np.zeros(component_count + 1, dtype=np.bool_)
        rejected[np.unique(labels[incomplete])] = True
        rejected[0] = False
        maxima[map_index] = local & ~rejected[labels]

    result = torch.from_numpy(maxima.reshape(values.shape))
    return result.to(device=values.device)


__all__ = ["regional_maxima_mask"]
