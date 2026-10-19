"""Geometry-only attachment of isolated residual atoms to frozen bond ports."""

from __future__ import annotations

import torch


def attach_residual_candidates(
    points: torch.Tensor,
    bond_endpoints: torch.Tensor,
    bond_mask: torch.Tensor,
    *,
    radius_px: float = 12.0,
    image_size: float = 768.0,
    occupied_endpoints: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Assign residual points to at most one unoccupied frozen bond endpoint.

    Assignment is target-independent and deterministic: all point/endpoint
    pairs inside the radius are sorted by distance, then greedily claimed.
    Existing endpoint ownership can be supplied as a boolean ``[Q, 2]`` mask;
    those endpoints are never stolen by a residual candidate.
    """

    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError("points must have shape [N, 2]")
    if bond_endpoints.ndim != 3 or tuple(bond_endpoints.shape[1:]) != (2, 2):
        raise ValueError("bond_endpoints must have shape [Q, 2, 2]")
    if bond_mask.ndim != 1 or bond_mask.size(0) != bond_endpoints.size(0):
        raise ValueError("bond_mask must have shape [Q]")
    if radius_px <= 0.0 or image_size <= 0.0:
        raise ValueError("radius_px and image_size must be positive")
    if occupied_endpoints is None:
        occupied_endpoints = torch.zeros(
            bond_endpoints.shape[:2], dtype=torch.bool, device=bond_endpoints.device
        )
    if occupied_endpoints.shape != bond_endpoints.shape[:2]:
        raise ValueError("occupied_endpoints must have shape [Q, 2]")

    count = points.size(0)
    bond_ids = torch.full((count,), -1, dtype=torch.long, device=points.device)
    endpoint_ids = torch.full((count,), -1, dtype=torch.long, device=points.device)
    distances = torch.full((count,), float("inf"), dtype=points.dtype, device=points.device)
    if count == 0 or not bool(bond_mask.any()):
        return {"bond_indices": bond_ids, "endpoint_indices": endpoint_ids, "distances_px": distances, "matched": bond_ids >= 0}

    valid_endpoints = bond_mask[:, None] & ~occupied_endpoints
    valid_flat = torch.where(valid_endpoints.flatten())[0]
    if valid_flat.numel() == 0:
        return {"bond_indices": bond_ids, "endpoint_indices": endpoint_ids, "distances_px": distances, "matched": bond_ids >= 0}
    endpoint_points = bond_endpoints.reshape(-1, 2)[valid_flat]
    cost = torch.cdist(points * image_size, endpoint_points * image_size)
    rows, cols = torch.where(cost <= radius_px)
    if rows.numel() == 0:
        return {"bond_indices": bond_ids, "endpoint_indices": endpoint_ids, "distances_px": distances, "matched": bond_ids >= 0}
    order = sorted(
        range(rows.numel()),
        key=lambda index: (float(cost[rows[index], cols[index]]), int(rows[index]), int(cols[index])),
    )
    used_points: set[int] = set()
    used_endpoints: set[int] = set()
    for index in order:
        point_index = int(rows[index])
        endpoint_offset = int(cols[index])
        endpoint_flat = int(valid_flat[endpoint_offset])
        if point_index in used_points or endpoint_flat in used_endpoints:
            continue
        bond_index, endpoint_index = divmod(endpoint_flat, 2)
        used_points.add(point_index)
        used_endpoints.add(endpoint_flat)
        bond_ids[point_index] = bond_index
        endpoint_ids[point_index] = endpoint_index
        distances[point_index] = cost[rows[index], cols[index]]
    return {"bond_indices": bond_ids, "endpoint_indices": endpoint_ids, "distances_px": distances, "matched": bond_ids >= 0}


__all__ = ["attach_residual_candidates"]
