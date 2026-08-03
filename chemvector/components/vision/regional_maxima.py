"""Deterministic regional maxima for FP32 spatial maps."""

from __future__ import annotations

from collections import deque

import numpy as np
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

    for map_index, (current, valid) in enumerate(zip(maps, valid_maps)):
        visited = np.zeros((height, width), dtype=np.bool_)
        for start_y in range(height):
            for start_x in range(width):
                if visited[start_y, start_x] or not valid[start_y, start_x]:
                    continue
                value = current[start_y, start_x]
                visited[start_y, start_x] = True
                pending = deque(((start_y, start_x),))
                plateau = []
                is_maximum = True
                while pending:
                    y, x = pending.popleft()
                    plateau.append((y, x))
                    for delta_y, delta_x in offsets:
                        neighbor_y = y + delta_y
                        neighbor_x = x + delta_x
                        if not (
                            0 <= neighbor_y < height
                            and 0 <= neighbor_x < width
                            and valid[neighbor_y, neighbor_x]
                        ):
                            continue
                        neighbor = current[neighbor_y, neighbor_x]
                        if neighbor > value:
                            is_maximum = False
                        elif (
                            neighbor == value
                            and not visited[neighbor_y, neighbor_x]
                        ):
                            visited[neighbor_y, neighbor_x] = True
                            pending.append((neighbor_y, neighbor_x))
                if is_maximum:
                    for y, x in plateau:
                        maxima[map_index, y, x] = True

    result = torch.from_numpy(maxima.reshape(values.shape))
    return result.to(device=values.device)


__all__ = ["regional_maxima_mask"]
