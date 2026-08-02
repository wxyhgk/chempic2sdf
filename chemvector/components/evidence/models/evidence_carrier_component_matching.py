"""Enumerated structured matching for small carrier-pair components."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def enumerate_component_matchings(pair_indices: Any) -> torch.Tensor:
    """Enumerate every endpoint-disjoint matching in canonical order."""

    pairs = torch.as_tensor(pair_indices, dtype=torch.long).detach().cpu()
    if pairs.numel() == 0:
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.size(1) != 2:
        raise ValueError("pair_indices must have shape [edges,2]")
    if bool((pairs < 0).any()) or bool((pairs[:, 0] == pairs[:, 1]).any()):
        raise ValueError("component edges require distinct nonnegative endpoints")
    keys = [tuple(sorted(map(int, pair))) for pair in pairs.tolist()]
    if len(set(keys)) != len(keys):
        raise ValueError("component edges must be unique")
    edge_count = len(pairs)
    if not edge_count:
        return torch.zeros((1, 0), dtype=torch.bool)

    incident: dict[int, list[tuple[int, int]]] = {}
    for edge_index, (first, second) in enumerate(keys):
        incident.setdefault(first, []).append((second, edge_index))
        incident.setdefault(second, []).append((first, edge_index))
    for values in incident.values():
        values.sort(key=lambda item: (item[0], item[1]))

    matchings: set[tuple[int, ...]] = set()

    def visit(available: frozenset[int], selected: tuple[int, ...]) -> None:
        if not available:
            matchings.add(tuple(sorted(selected)))
            return
        first = min(available)
        visit(available - {first}, selected)
        for second, edge_index in incident[first]:
            if second in available:
                visit(available - {first, second}, selected + (edge_index,))

    visit(frozenset(incident), ())
    ordered = sorted(matchings, key=lambda selected: (len(selected), selected))
    result = torch.zeros((len(ordered), edge_count), dtype=torch.bool)
    for row, selected in enumerate(ordered):
        if selected:
            result[row, torch.tensor(selected, dtype=torch.long)] = True
    return result


def optimal_component_matchings(
    matching_masks: torch.Tensor,
    pair_labels: Any,
) -> torch.Tensor:
    """Mark configurations that maximise positives, then minimise negatives."""

    masks = torch.as_tensor(matching_masks, dtype=torch.bool).detach().cpu()
    labels = torch.as_tensor(pair_labels, dtype=torch.long).detach().cpu()
    if masks.ndim != 2 or labels.shape != (masks.size(1),):
        raise ValueError("matching masks and pair labels have incompatible shapes")
    if not len(masks):
        raise ValueError("at least one matching is required")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("pair labels must be binary")
    positives = (masks & (labels == 1)[None, :]).sum(dim=1)
    negatives = (masks & (labels == 0)[None, :]).sum(dim=1)
    best_positive = positives.max()
    positive_best = positives == best_positive
    best_negative = negatives[positive_best].min()
    return positive_best & (negatives == best_negative)


def component_structured_loss(
    edge_energies: torch.Tensor,
    matching_masks: torch.Tensor,
    optimal_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability assigned to all target-optimal matchings."""

    if edge_energies.ndim != 1 or not edge_energies.is_floating_point():
        raise ValueError("edge_energies must be a floating vector")
    if matching_masks.ndim != 2 or matching_masks.dtype != torch.bool:
        raise ValueError("matching_masks must be a boolean matrix")
    if matching_masks.size(1) != len(edge_energies):
        raise ValueError("matching masks must align with edge energies")
    if optimal_mask.shape != (matching_masks.size(0),) or optimal_mask.dtype != torch.bool:
        raise ValueError("optimal_mask must select matching rows")
    if not bool(optimal_mask.any()):
        raise ValueError("optimal_mask cannot be empty")
    if len({edge_energies.device, matching_masks.device, optimal_mask.device}) != 1:
        raise ValueError("structured matching tensors must share one device")
    energies = matching_masks.to(edge_energies.dtype) @ edge_energies
    return torch.logsumexp(energies, dim=0) - torch.logsumexp(
        energies[optimal_mask], dim=0
    )


def select_component_matching(
    edge_energies: torch.Tensor,
    matching_masks: torch.Tensor,
) -> torch.Tensor:
    """Select the first canonical maximum-energy matching."""

    if edge_energies.ndim != 1 or matching_masks.shape[1:] != (len(edge_energies),):
        raise ValueError("edge energies and matching masks differ")
    if matching_masks.dtype != torch.bool or not edge_energies.is_floating_point():
        raise ValueError("matching selection tensor dtypes differ")
    if not len(matching_masks):
        raise ValueError("at least one matching is required")
    energies = matching_masks.to(edge_energies.dtype) @ edge_energies
    return matching_masks[int(energies.argmax())]


def apply_component_matching_selection(
    baseline_scores: torch.Tensor,
    edge_residuals: torch.Tensor,
    components: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> torch.Tensor:
    """Keep only the learned matching in each cached active multi-component."""

    if baseline_scores.ndim != 1 or edge_residuals.shape != baseline_scores.shape:
        raise ValueError("baseline scores and residuals must share one vector shape")
    if not baseline_scores.is_floating_point() or not edge_residuals.is_floating_point():
        raise ValueError("baseline scores and residuals must be floating point")
    if baseline_scores.device != edge_residuals.device:
        raise ValueError("baseline scores and residuals must share one device")
    result = baseline_scores.clone()
    for component in components:
        indices = torch.as_tensor(
            component["active_pair_indices"], dtype=torch.long, device=result.device
        )
        masks = torch.as_tensor(
            component["matching_masks"], dtype=torch.bool, device=result.device
        )
        if indices.ndim != 1 or masks.shape[1:] != (len(indices),):
            raise ValueError("component matching cache shapes differ")
        if not len(indices):
            continue
        baseline = baseline_scores[indices]
        if bool((baseline <= float(threshold)).any()):
            raise ValueError("component cache contains inactive baseline pairs")
        energies = baseline - float(threshold) + edge_residuals[indices]
        selected = select_component_matching(energies, masks)
        result[indices] = float(threshold) - 1.0
        result[indices[selected]] = baseline[selected]
    if bool((result > baseline_scores).any()):
        raise RuntimeError("component selection increased a baseline score")
    return result


__all__ = [
    "apply_component_matching_selection",
    "component_structured_loss",
    "enumerate_component_matchings",
    "optimal_component_matchings",
    "select_component_matching",
]
