"""Exact latent-positive matching for equivalent Atom candidate assignments."""

from __future__ import annotations

from itertools import product
import math

from scipy.optimize import linear_sum_assignment
import torch


def g3_gt_candidate_compatibility(
    target_anchors: torch.Tensor,
    candidate_anchors: torch.Tensor,
    *,
    radius_pixels: float = 6.0,
    image_size: int = 768,
) -> torch.Tensor:
    """Return the canonical FP32 GT-by-candidate compatibility matrix."""

    if target_anchors.ndim != 2 or target_anchors.shape[1:] != (2,):
        raise ValueError("G3 latent target anchors must have shape [GT,2]")
    if candidate_anchors.ndim != 2 or candidate_anchors.shape[1:] != (2,):
        raise ValueError("G3 latent candidate anchors must have shape [candidate,2]")
    if radius_pixels <= 0.0 or image_size < 1:
        raise ValueError("G3 latent compatibility radius or image size is invalid")
    return torch.cdist(
        target_anchors.detach().float(), candidate_anchors.detach().float()
    ) <= float(radius_pixels) / float(image_size)


def g3_unrestricted_fixed_cardinality_assignment(
    edge_scores: torch.Tensor, count: int
) -> dict[str, torch.Tensor | float]:
    """Find the exact best unique query-candidate matching of fixed cardinality."""

    if edge_scores.ndim != 2 or not bool(torch.isfinite(edge_scores).all()):
        raise ValueError("G3 unrestricted edge scores must be one finite matrix")
    query_count, candidate_count = edge_scores.shape
    if not 0 <= count <= min(query_count, candidate_count):
        raise ValueError("G3 unrestricted matching count is outside capacity")
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=edge_scores.device)
        return {"query_indices": empty, "candidate_indices": empty, "score": 0.0}
    dummy_queries = candidate_count - count
    dummy_candidates = query_count - count
    size = query_count + dummy_queries
    cost = torch.zeros((size, size), dtype=torch.float64)
    cost[:query_count, :candidate_count] = -edge_scores.detach().double().cpu()
    if dummy_queries and dummy_candidates:
        cost[query_count:, candidate_count:] = 1_000_000_000.0
    rows, columns = linear_sum_assignment(cost.numpy())
    rows = torch.as_tensor(rows, dtype=torch.long, device=edge_scores.device)
    columns = torch.as_tensor(columns, dtype=torch.long, device=edge_scores.device)
    real = (rows < query_count) & (columns < candidate_count)
    queries = rows[real]
    candidates = columns[real]
    if len(queries) != count:
        raise RuntimeError("G3 unrestricted matching cardinality differs")
    return {
        "query_indices": queries,
        "candidate_indices": candidates,
        "score": float(edge_scores.detach()[queries, candidates].double().sum().cpu()),
    }


def g3_candidate_set_covers_gt(
    candidate_indices: torch.Tensor, compatibility: torch.Tensor
) -> bool:
    """Test whether selected candidates admit a perfect one-to-one GT matching."""

    if compatibility.ndim != 2 or compatibility.dtype != torch.bool:
        raise ValueError("G3 latent compatibility must be one boolean matrix")
    gt_count, candidate_count = compatibility.shape
    if candidate_indices.shape != (gt_count,):
        return False
    if len(candidate_indices.unique()) != gt_count:
        return False
    if bool((candidate_indices < 0).any()) or bool(
        (candidate_indices >= candidate_count).any()
    ):
        return False
    selected = compatibility[:, candidate_indices].detach().cpu()
    rows, columns = linear_sum_assignment((~selected).to(torch.int8).numpy())
    return len(rows) == gt_count and bool(selected[rows, columns].all())


def g3_latent_positive_assignment(
    edge_scores: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    max_ownerships: int = 256,
) -> dict[str, torch.Tensor | float | int]:
    """Find the exact highest-scoring matching over all GT-compatible equivalents."""

    if edge_scores.ndim != 2 or not bool(torch.isfinite(edge_scores).all()):
        raise ValueError("G3 latent edge scores must be one finite matrix")
    if compatibility.ndim != 2 or compatibility.dtype != torch.bool:
        raise ValueError("G3 latent compatibility must be one boolean matrix")
    gt_count, candidate_count = compatibility.shape
    query_count, score_candidates = edge_scores.shape
    if candidate_count != score_candidates or not 0 < gt_count <= query_count:
        raise ValueError("G3 latent matching dimensions differ")
    if bool((compatibility.sum(dim=1) < 1).any()):
        raise RuntimeError("G3 latent GT has no compatible candidate")
    if max_ownerships < 1:
        raise ValueError("G3 latent ownership cap must be positive")

    compatibility_cpu = compatibility.detach().cpu()
    column_degree = compatibility_cpu.sum(dim=0)
    ambiguous = torch.where(column_degree > 1)[0].tolist()
    ownership_options = [
        torch.where(compatibility_cpu[:, candidate])[0].tolist()
        for candidate in ambiguous
    ]
    ownership_count = math.prod(len(options) for options in ownership_options)
    if ownership_count > max_ownerships:
        raise RuntimeError(
            f"G3 latent ownership combinations exceed cap: {ownership_count}"
        )
    exclusive_groups = [
        torch.where(
            compatibility_cpu[gt] & (column_degree == 1)
        )[0].tolist()
        for gt in range(gt_count)
    ]
    best = None
    best_score = float("-inf")
    ownership_iterator = product(*ownership_options) if ambiguous else [()]
    scores = edge_scores.detach().double().cpu()
    for ownership in ownership_iterator:
        groups = [list(values) for values in exclusive_groups]
        for candidate, gt in zip(ambiguous, ownership):
            groups[gt].append(candidate)
        if any(not values for values in groups):
            continue
        values = torch.empty((query_count, gt_count), dtype=torch.float64)
        best_candidates = torch.empty(
            (query_count, gt_count), dtype=torch.long
        )
        for gt, candidates in enumerate(groups):
            current, offsets = scores[:, candidates].max(dim=1)
            values[:, gt] = current
            best_candidates[:, gt] = torch.as_tensor(candidates)[offsets]
        query_rows, gt_columns = linear_sum_assignment((-values).numpy())
        query_rows = torch.as_tensor(query_rows, dtype=torch.long)
        gt_columns = torch.as_tensor(gt_columns, dtype=torch.long)
        current_score = float(values[query_rows, gt_columns].sum())
        if current_score > best_score:
            candidates = best_candidates[query_rows, gt_columns]
            best_score = current_score
            best = (query_rows, candidates, gt_columns)
    if best is None:
        raise RuntimeError("G3 latent positive matching is infeasible")
    queries, candidates, gt_indices = best
    if len(queries.unique()) != gt_count or len(candidates.unique()) != gt_count:
        raise RuntimeError("G3 latent positive matching uniqueness differs")
    device = edge_scores.device
    return {
        "query_indices": queries.to(device),
        "candidate_indices": candidates.to(device),
        "gt_indices": gt_indices.to(device),
        "score": best_score,
        "ambiguous_candidates": len(ambiguous),
        "ownership_combinations": ownership_count,
    }


__all__ = [
    "g3_candidate_set_covers_gt",
    "g3_gt_candidate_compatibility",
    "g3_latent_positive_assignment",
    "g3_unrestricted_fixed_cardinality_assignment",
]
