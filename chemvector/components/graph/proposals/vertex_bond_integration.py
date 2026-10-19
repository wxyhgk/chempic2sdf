"""Deterministic vertex-to-frozen-bond compatibility primitives.

These functions deliberately contain no target lookup.  They are used by the
integration shadow to turn a candidate point field and frozen bond endpoints
into an attachment table.  Target identities are evaluated by the audit only
after this table has been produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VertexAttachment:
    """Endpoint assignment for one image.

    ``endpoint_candidate`` has shape ``[Q, 2]`` and contains ``-1`` for an
    unmatched endpoint.  Candidate indices refer to the *deduplicated* input
    list.  Reuse across different bonds is retained for junction diagnostics.
    """

    endpoint_candidate: torch.Tensor
    endpoint_distance_px: torch.Tensor
    candidate_kept: torch.Tensor
    candidate_suppressed: torch.Tensor


def _validate_points(points: torch.Tensor, name: str) -> None:
    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    if not bool(torch.isfinite(points).all()):
        raise ValueError(f"{name} contains non-finite values")


def deduplicate_vertex_candidates(
    points: torch.Tensor,
    scores: torch.Tensor | None = None,
    *,
    radius_px: float = 6.0,
    image_size: float = 768.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep score-ranked points separated by ``radius_px``.

    The operation is candidate-only and therefore safe to run before any
    target labels are loaded.  Ties are resolved by the original index.
    Returns kept original indices and a boolean suppression mask.
    """

    _validate_points(points, "points")
    if radius_px <= 0 or image_size <= 0:
        raise ValueError("radius_px and image_size must be positive")
    count = points.size(0)
    if scores is None:
        scores = torch.zeros(count, dtype=points.dtype, device=points.device)
    if scores.ndim != 1 or scores.size(0) != count or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite with shape [N]")
    order = sorted(range(count), key=lambda i: (-float(scores[i]), i))
    kept: list[int] = []
    suppressed = torch.zeros(count, dtype=torch.bool, device=points.device)
    for index in order:
        if kept:
            distance = torch.linalg.vector_norm(
                (points[index] - points[torch.tensor(kept, device=points.device)]) * image_size,
                dim=-1,
            )
            if bool((distance <= radius_px).any()):
                suppressed[index] = True
                continue
        kept.append(index)
    return torch.tensor(kept, dtype=torch.long, device=points.device), suppressed


def attach_vertex_candidates(
    points: torch.Tensor,
    bond_endpoints: torch.Tensor,
    bond_mask: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    radius_px: float = 12.0,
    duplicate_radius_px: float = 6.0,
    image_size: float = 768.0,
) -> VertexAttachment:
    """Attach candidates to frozen endpoints without target information.

    Every active endpoint gets at most one candidate.  A candidate may be
    reused by endpoints belonging to different bonds (the expected junction
    behaviour), but cannot occupy both endpoints of the same bond.  When the
    nearest candidate is already used on the opposite endpoint of that bond,
    the next nearest admissible candidate is considered.
    """

    _validate_points(points, "points")
    if bond_endpoints.ndim != 3 or tuple(bond_endpoints.shape[1:]) != (2, 2):
        raise ValueError("bond_endpoints must have shape [Q, 2, 2]")
    if bond_mask.ndim != 1 or bond_mask.size(0) != bond_endpoints.size(0):
        raise ValueError("bond_mask must have shape [Q]")
    _validate_points(bond_endpoints.reshape(-1, 2), "bond_endpoints")
    if radius_px <= 0 or duplicate_radius_px <= 0 or image_size <= 0:
        raise ValueError("attachment radii and image_size must be positive")
    kept, suppressed = deduplicate_vertex_candidates(
        points, scores, radius_px=duplicate_radius_px, image_size=image_size
    )
    candidates = points[kept]
    q = bond_endpoints.size(0)
    endpoint_candidate = torch.full((q, 2), -1, dtype=torch.long, device=points.device)
    endpoint_distance = torch.full((q, 2), float("inf"), dtype=points.dtype, device=points.device)
    if not len(kept) or not bool(bond_mask.any()):
        return VertexAttachment(endpoint_candidate, endpoint_distance, kept, suppressed)

    distances = torch.cdist(bond_endpoints * image_size, candidates * image_size)
    # Process closest pairs globally for deterministic one-to-one endpoint
    # ownership.  Cross-bond candidate reuse remains legal.
    pairs: list[tuple[float, int, int, int]] = []
    for bond in torch.where(bond_mask)[0].tolist():
        for endpoint in range(2):
            for candidate in range(len(kept)):
                value = float(distances[bond, endpoint, candidate])
                if value <= radius_px:
                    pairs.append((value, int(bond), endpoint, candidate))
    pairs.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    used_on_bond: set[tuple[int, int]] = set()
    for value, bond, endpoint, candidate in pairs:
        if endpoint_candidate[bond, endpoint] >= 0:
            continue
        if (bond, candidate) in used_on_bond:
            continue
        endpoint_candidate[bond, endpoint] = candidate
        endpoint_distance[bond, endpoint] = value
        used_on_bond.add((bond, candidate))
    return VertexAttachment(endpoint_candidate, endpoint_distance, kept, suppressed)


def attachment_geometry_metrics(
    result: VertexAttachment,
    bond_mask: torch.Tensor,
) -> dict[str, int | float]:
    """Summarize geometry-only attachment, before any target labels."""

    active = torch.where(bond_mask)[0]
    if not len(active):
        return {
            "active_bonds": 0, "active_endpoints": 0, "attached_endpoints": 0,
            "endpoint_attachment_rate": 0.0, "two_endpoint_bonds": 0,
            "two_endpoint_bond_rate": 0.0, "candidate_reuse_endpoints": 0,
            "candidate_reuse_conflicts": 0,
        }
    assigned = result.endpoint_candidate[active] >= 0
    endpoint_count = int(assigned.numel())
    attached = int(assigned.sum())
    complete = int(assigned.all(dim=1).sum())
    values = result.endpoint_candidate[active][assigned]
    counts = torch.bincount(values) if len(values) else torch.zeros(0, dtype=torch.long)
    reused = int((counts > 1).sum()) if len(counts) else 0
    # Reuse is not automatically a violation: it is expected at junctions.
    # The conflict count is always zero here because same-bond reuse is vetoed.
    return {
        "active_bonds": int(len(active)),
        "active_endpoints": endpoint_count,
        "attached_endpoints": attached,
        "endpoint_attachment_rate": attached / max(endpoint_count, 1),
        "two_endpoint_bonds": complete,
        "two_endpoint_bond_rate": complete / max(len(active), 1),
        "candidate_reuse_endpoints": reused,
        "candidate_reuse_conflicts": 0,
    }


__all__ = [
    "VertexAttachment",
    "deduplicate_vertex_candidates",
    "attach_vertex_candidates",
    "attachment_geometry_metrics",
]
