"""Target-free connected components of the carrier-pair candidate graph."""

from __future__ import annotations

from typing import Any

import torch


def pair_component_metadata(pair_indices: Any) -> dict[str, torch.Tensor]:
    """Return canonical per-pair component IDs and component sizes."""

    pairs = torch.as_tensor(pair_indices, dtype=torch.long).detach().cpu()
    if pairs.numel() == 0:
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.size(1) != 2:
        raise ValueError("pair_indices must have shape [pairs,2]")
    if bool((pairs < 0).any()) or bool((pairs[:, 0] == pairs[:, 1]).any()):
        raise ValueError("pair_indices must contain distinct nonnegative anchors")
    if not len(pairs):
        empty = torch.empty(0, dtype=torch.long)
        return {
            "component_ids": empty,
            "component_anchor_counts": empty,
            "component_edge_counts": empty,
            "multi_anchor_component": torch.empty(0, dtype=torch.bool),
        }

    anchor_count = int(pairs.max()) + 1
    parent = list(range(anchor_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in pairs.tolist():
        union(int(first), int(second))

    root_nodes: dict[int, set[int]] = {}
    root_edges: dict[int, int] = {}
    pair_roots = []
    for first, second in pairs.tolist():
        root = find(int(first))
        pair_roots.append(root)
        root_nodes.setdefault(root, set()).update((int(first), int(second)))
        root_edges[root] = root_edges.get(root, 0) + 1
    ordered_roots = sorted(root_nodes, key=lambda root: min(root_nodes[root]))
    component_id = {root: index for index, root in enumerate(ordered_roots)}
    ids = torch.tensor([component_id[root] for root in pair_roots], dtype=torch.long)
    anchor_counts = torch.tensor(
        [len(root_nodes[root]) for root in pair_roots], dtype=torch.long
    )
    edge_counts = torch.tensor(
        [root_edges[root] for root in pair_roots], dtype=torch.long
    )
    return {
        "component_ids": ids,
        "component_anchor_counts": anchor_counts,
        "component_edge_counts": edge_counts,
        "multi_anchor_component": anchor_counts > 2,
    }


def apply_multi_component_label_oracle(
    pair_scores: Any,
    pair_indices: Any,
    pair_labels: Any,
    *,
    threshold: float,
) -> torch.Tensor:
    """Replace only multi-anchor component scores with binary label-oracle scores."""

    scores = torch.as_tensor(pair_scores).detach().cpu().to(torch.float32).clone()
    labels = torch.as_tensor(pair_labels, dtype=torch.long).detach().cpu()
    metadata = pair_component_metadata(pair_indices)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("pair scores and labels must share one vector shape")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("pair labels must be binary")
    multi = metadata["multi_anchor_component"]
    scores[multi] = torch.where(
        labels[multi] == 1,
        torch.full_like(scores[multi], float(threshold) + 1.0),
        torch.full_like(scores[multi], float(threshold) - 1.0),
    )
    return scores


__all__ = [
    "apply_multi_component_label_oracle",
    "pair_component_metadata",
]
