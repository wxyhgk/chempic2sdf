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


def g3_group_constrained_fixed_cardinality_assignment(
    edge_scores: torch.Tensor,
    count: int,
    group_candidates: torch.Tensor,
    group_cap: int,
) -> dict[str, torch.Tensor | float] | None:
    """Maximize a fixed-K matching with an upper bound on selected group columns."""

    if edge_scores.ndim != 2 or not bool(torch.isfinite(edge_scores).all()):
        raise ValueError("G3 constrained edge scores must be one finite matrix")
    query_count, candidate_count = edge_scores.shape
    if not 0 < count <= min(query_count, candidate_count):
        raise ValueError("G3 constrained matching count is outside capacity")
    if group_candidates.ndim != 1 or group_candidates.dtype != torch.bool:
        raise ValueError("G3 constrained candidate group must be one boolean vector")
    if len(group_candidates) != candidate_count:
        raise ValueError("G3 constrained candidate group width differs")
    group_size = int(group_candidates.sum())
    if not 0 <= group_cap <= group_size:
        raise ValueError("G3 constrained candidate group cap is invalid")
    if group_cap >= min(group_size, count):
        return g3_unrestricted_fixed_cardinality_assignment(edge_scores, count)

    dummy_query_count = candidate_count - count
    required_group_dummies = group_size - group_cap
    if required_group_dummies > dummy_query_count:
        return None
    dummy_candidate_count = query_count - count
    size = query_count + dummy_query_count
    forbidden = float("inf")
    cost = torch.zeros((size, size), dtype=torch.float64)
    cost[:query_count, :candidate_count] = -edge_scores.detach().double().cpu()
    if dummy_query_count and dummy_candidate_count:
        cost[query_count:, candidate_count:] = forbidden
    if required_group_dummies:
        non_group = ~group_candidates.detach().cpu()
        cost[
            query_count : query_count + required_group_dummies,
            :candidate_count,
        ][:, non_group] = forbidden
    try:
        rows, columns = linear_sum_assignment(cost.numpy())
    except ValueError:
        return None
    if not bool(torch.isfinite(cost[rows, columns]).all()):
        return None
    rows = torch.as_tensor(rows, dtype=torch.long, device=edge_scores.device)
    columns = torch.as_tensor(columns, dtype=torch.long, device=edge_scores.device)
    real = (rows < query_count) & (columns < candidate_count)
    queries = rows[real]
    candidates = columns[real]
    if len(queries) != count:
        raise RuntimeError("G3 constrained matching cardinality differs")
    selected_in_group = int(group_candidates.to(candidates.device)[candidates].sum())
    if selected_in_group > group_cap:
        raise RuntimeError("G3 constrained matching violated candidate group cap")
    return {
        "query_indices": queries,
        "candidate_indices": candidates,
        "score": float(edge_scores.detach()[queries, candidates].double().sum().cpu()),
        "selected_group_candidates": selected_in_group,
    }


def g3_candidate_set_gt_coverage(
    candidate_indices: torch.Tensor, compatibility: torch.Tensor
) -> int:
    """Return the maximum number of GT atoms covered by selected candidates."""

    if compatibility.ndim != 2 or compatibility.dtype != torch.bool:
        raise ValueError("G3 latent compatibility must be one boolean matrix")
    if candidate_indices.ndim != 1:
        raise ValueError("G3 selected candidates must be one index vector")
    _, candidate_count = compatibility.shape
    if len(candidate_indices.unique()) != len(candidate_indices):
        raise ValueError("G3 selected candidates must be unique")
    if bool((candidate_indices < 0).any()) or bool(
        (candidate_indices >= candidate_count).any()
    ):
        raise ValueError("G3 selected candidate index is outside compatibility")
    if len(candidate_indices) == 0:
        return 0
    selected = compatibility[:, candidate_indices].detach().cpu()
    rows, columns = linear_sum_assignment((~selected).to(torch.int8).numpy())
    return int(selected[rows, columns].sum())


def g3_candidate_set_covers_gt(
    candidate_indices: torch.Tensor, compatibility: torch.Tensor
) -> bool:
    """Test whether selected candidates admit a perfect one-to-one GT matching."""

    gt_count = compatibility.size(0)
    if candidate_indices.shape != (gt_count,):
        return False
    try:
        return g3_candidate_set_gt_coverage(candidate_indices, compatibility) == gt_count
    except ValueError:
        return False


def g3_hall_witnesses(
    compatibility: torch.Tensor,
    *,
    max_component_gt: int = 8,
    max_witnesses: int = 256,
) -> list[dict[str, torch.Tensor | int]]:
    """Enumerate deduplicated Hall-deficiency witnesses by compatibility component."""

    if compatibility.ndim != 2 or compatibility.dtype != torch.bool:
        raise ValueError("G3 Hall compatibility must be one boolean matrix")
    gt_count, candidate_count = compatibility.shape
    if gt_count < 1 or candidate_count < 1:
        raise ValueError("G3 Hall compatibility cannot be empty")
    if min(max_component_gt, max_witnesses) < 1:
        raise ValueError("G3 Hall witness limits must be positive")
    compatibility_cpu = compatibility.detach().cpu()
    adjacency = (
        compatibility_cpu.to(torch.int16)
        @ compatibility_cpu.to(torch.int16).T
    ) > 0
    remaining = set(range(gt_count))
    components = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            neighbors = torch.where(adjacency[current])[0].tolist()
            stack.extend(value for value in neighbors if value not in component)
        remaining -= component
        components.append(sorted(component))

    deduplicated = {}
    for component in components:
        if len(component) > max_component_gt:
            raise RuntimeError(
                f"G3 Hall GT component exceeds cap: {len(component)}"
            )
        for mask in range(1, 1 << len(component)):
            gt_indices = [
                component[offset]
                for offset in range(len(component))
                if mask & (1 << offset)
            ]
            neighborhood = compatibility_cpu[gt_indices].any(dim=0)
            cap = len(gt_indices) - 1
            key = (tuple(torch.where(neighborhood)[0].tolist()), cap)
            deduplicated.setdefault(key, tuple(gt_indices))
    if len(deduplicated) > max_witnesses:
        raise RuntimeError(
            f"G3 Hall witness count exceeds cap: {len(deduplicated)}"
        )
    witnesses = []
    for (candidate_indices, cap), gt_indices in sorted(deduplicated.items()):
        group = torch.zeros(candidate_count, dtype=torch.bool)
        group[list(candidate_indices)] = True
        witnesses.append({
            "gt_indices": torch.tensor(gt_indices, dtype=torch.long),
            "group_candidates": group,
            "group_cap": cap,
        })
    return witnesses


def g3_best_invalid_assignment(
    edge_scores: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    max_component_gt: int = 8,
    max_witnesses: int = 256,
) -> dict[str, torch.Tensor | float | int] | None:
    """Find the exact highest-scoring fixed-K matching with invalid GT coverage."""

    if edge_scores.ndim != 2 or not bool(torch.isfinite(edge_scores).all()):
        raise ValueError("G3 best-invalid edge scores must be one finite matrix")
    if compatibility.ndim != 2 or compatibility.dtype != torch.bool:
        raise ValueError("G3 best-invalid compatibility must be one boolean matrix")
    if compatibility.size(1) != edge_scores.size(1):
        raise ValueError("G3 best-invalid candidate widths differ")
    gt_count = compatibility.size(0)
    if not 0 < gt_count <= min(edge_scores.shape):
        raise ValueError("G3 best-invalid GT count is outside matching capacity")

    output_device = edge_scores.device
    scores_cpu = edge_scores.detach().double().cpu()
    compatibility_cpu = compatibility.detach().cpu()
    unrestricted = g3_unrestricted_fixed_cardinality_assignment(
        scores_cpu, gt_count
    )
    if not g3_candidate_set_covers_gt(
        unrestricted["candidate_indices"], compatibility_cpu
    ):
        return {
            **unrestricted,
            "query_indices": unrestricted["query_indices"].to(output_device),
            "candidate_indices": unrestricted["candidate_indices"].to(output_device),
            "witness_index": -1,
            "witness_count": 0,
        }
    witnesses = g3_hall_witnesses(
        compatibility_cpu,
        max_component_gt=max_component_gt,
        max_witnesses=max_witnesses,
    )
    best = None
    best_score = float("-inf")
    best_index = -1
    for index, witness in enumerate(witnesses):
        current = g3_group_constrained_fixed_cardinality_assignment(
            scores_cpu,
            gt_count,
            witness["group_candidates"],
            int(witness["group_cap"]),
        )
        if current is not None and current["score"] > best_score:
            best = current
            best_score = float(current["score"])
            best_index = index
    if best is None:
        return None
    if g3_candidate_set_covers_gt(best["candidate_indices"], compatibility_cpu):
        raise RuntimeError("G3 best-invalid solver returned valid GT coverage")
    return {
        **best,
        "query_indices": best["query_indices"].to(output_device),
        "candidate_indices": best["candidate_indices"].to(output_device),
        "witness_index": best_index,
        "witness_count": len(witnesses),
    }


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
    "g3_best_invalid_assignment",
    "g3_candidate_set_covers_gt",
    "g3_candidate_set_gt_coverage",
    "g3_group_constrained_fixed_cardinality_assignment",
    "g3_gt_candidate_compatibility",
    "g3_hall_witnesses",
    "g3_latent_positive_assignment",
    "g3_unrestricted_fixed_cardinality_assignment",
]
