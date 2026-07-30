"""Counterfactual single-node suppression for predicted AtomObjects."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def lexicographic_node_match(
    anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    kept: torch.Tensor,
    *,
    radius_px: float,
    image_size: float = 768.0,
    predicted_degrees: torch.Tensor | None = None,
    target_degrees: torch.Tensor | None = None,
) -> dict[int, int]:
    """Maximize in-radius match cardinality before minimizing assignment cost."""

    selected = torch.where(kept)[0]
    if not len(selected) or not len(target_anchors):
        return {}
    distances = (
        torch.cdist(anchors[selected].double(), target_anchors.double()) * image_size
    ).cpu().numpy()
    proposal_count, target_count = distances.shape
    pair_cost = distances.copy()
    if predicted_degrees is not None or target_degrees is not None:
        if predicted_degrees is None or target_degrees is None:
            raise ValueError("Both proposal and target degrees are required")
        proposal_degree = predicted_degrees[selected].double().cpu().numpy()[:, None]
        target_degree = target_degrees.double().cpu().numpy()[None, :]
        pair_cost += np.abs(proposal_degree - target_degree) * (radius_px + 1.0)

    valid = distances <= radius_px
    size = proposal_count + target_count
    maximum_pair_cost = (radius_px + 1.0) * 8.0
    unmatched_cost = (size + 1.0) * maximum_pair_cost
    forbidden_cost = unmatched_cost * 4.0
    cost = np.full((size, size), forbidden_cost, dtype=np.float64)
    cost[:proposal_count, :target_count] = np.where(
        valid, pair_cost, forbidden_cost
    )
    cost[:proposal_count, target_count:] = unmatched_cost
    cost[proposal_count:, :target_count] = unmatched_cost
    cost[proposal_count:, target_count:] = 0.0
    rows, columns = linear_sum_assignment(cost)
    return {
        int(selected[row]): int(column)
        for row, column in zip(rows, columns)
        if row < proposal_count and column < target_count and valid[row, column]
    }


def predicted_adjacency(sample: Mapping, threshold: float) -> list[set[int]]:
    adjacency = [set() for _ in range(len(sample["anchors"]))]
    selected = sample["probabilities"] >= threshold
    for begin, end in sample["pairs"][selected].tolist():
        begin, end = int(begin), int(end)
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    return adjacency


def unique_delete_nodes(
    anchors: torch.Tensor, *, maximum_distance_px: float, image_size: float = 768.0
) -> list[int]:
    if len(anchors) < 2:
        return []
    distances = torch.cdist(anchors.double(), anchors.double()) * image_size
    close = (distances <= maximum_distance_px) & ~torch.eye(
        len(anchors), dtype=torch.bool
    )
    return torch.where(close.any(dim=1))[0].tolist()


def _target_bond_types(sample: Mapping) -> dict[tuple[int, int], int]:
    target = sample["target_bonds"]
    if isinstance(target, dict):
        return {
            tuple(sorted((int(pair[0]), int(pair[1])))): int(chemical)
            for pair, chemical in target.items()
        }
    return {tuple(sorted((int(pair[0]), int(pair[1])))): -1 for pair in target}


def evaluate_kept_graph(
    sample: Mapping,
    kept: torch.Tensor,
    *,
    presence_threshold: float,
    radius_px: float,
    topology_aware: bool = False,
) -> dict:
    target_bonds = _target_bond_types(sample)
    adjacency = predicted_adjacency(sample, presence_threshold)
    proposal_degrees = torch.tensor(
        [sum(bool(kept[index]) for index in row) for row in adjacency],
        dtype=torch.float64,
    )
    target_degrees = torch.zeros(len(sample["target_anchors"]), dtype=torch.float64)
    for begin, end in target_bonds:
        target_degrees[begin] += 1
        target_degrees[end] += 1
    mapping = lexicographic_node_match(
        sample["anchors"],
        sample["target_anchors"],
        kept,
        radius_px=radius_px,
        predicted_degrees=proposal_degrees if topology_aware else None,
        target_degrees=target_degrees if topology_aware else None,
    )

    predicted_bonds: set[tuple[int, int]] = set()
    typed_bonds: set[tuple[int, int]] = set()
    invalid_edges = duplicate_edges = 0
    selected_edges = sample["probabilities"] >= presence_threshold
    for row in torch.where(selected_edges)[0].tolist():
        begin, end = (int(value) for value in sample["pairs"][row].tolist())
        if not kept[begin] or not kept[end]:
            continue
        if begin not in mapping or end not in mapping or mapping[begin] == mapping[end]:
            invalid_edges += 1
            continue
        pair = tuple(sorted((mapping[begin], mapping[end])))
        if pair in predicted_bonds:
            duplicate_edges += 1
        predicted_bonds.add(pair)
        expected_type = target_bonds.get(pair)
        if expected_type is not None and (
            expected_type < 0 or int(sample["chemical"][row]) == expected_type
        ):
            typed_bonds.add(pair)

    matched_targets = set(mapping.values())
    expressible_bonds = {
        pair for pair in target_bonds if pair[0] in matched_targets and pair[1] in matched_targets
    }
    target_bond_ids = set(target_bonds)
    covered_bonds = predicted_bonds & target_bond_ids
    atom_exact = (
        len(mapping) == int(kept.sum()) == len(sample["target_anchors"])
    )
    skeleton_exact = (
        atom_exact
        and invalid_edges == 0
        and duplicate_edges == 0
        and predicted_bonds == target_bond_ids
    )
    return {
        "matched_atom_ids": matched_targets,
        "expressible_bond_ids": expressible_bonds,
        "covered_bond_ids": covered_bonds,
        "typed_bond_ids": typed_bonds,
        "selected_atoms": int(kept.sum()),
        "matched_atoms": len(matched_targets),
        "target_atoms": len(sample["target_anchors"]),
        "atom_fp": int(kept.sum()) - len(mapping),
        "atom_fn": len(sample["target_anchors"]) - len(matched_targets),
        "bond_fp": len(predicted_bonds - target_bond_ids) + invalid_edges + duplicate_edges,
        "bond_fn": len(target_bond_ids - predicted_bonds),
        "invalid_edges": invalid_edges,
        "duplicate_edges": duplicate_edges,
        "atom_exact": atom_exact,
        "skeleton_exact": skeleton_exact,
    }


def action_is_safe(baseline: Mapping, candidate: Mapping) -> bool:
    set_fields = (
        "matched_atom_ids",
        "expressible_bond_ids",
        "covered_bond_ids",
        "typed_bond_ids",
    )
    if any(not baseline[name].issubset(candidate[name]) for name in set_fields):
        return False
    if baseline["atom_exact"] and not candidate["atom_exact"]:
        return False
    if baseline["skeleton_exact"] and not candidate["skeleton_exact"]:
        return False
    return (
        candidate["atom_fn"] <= baseline["atom_fn"]
        and candidate["bond_fn"] <= baseline["bond_fn"]
        and candidate["invalid_edges"] <= baseline["invalid_edges"]
        and candidate["duplicate_edges"] <= baseline["duplicate_edges"]
    )


def action_utility(metrics: Mapping) -> tuple[int, int, int, int]:
    return (
        int(metrics["skeleton_exact"]),
        int(metrics["atom_exact"]),
        -(int(metrics["atom_fp"]) + int(metrics["atom_fn"])),
        -(int(metrics["bond_fp"]) + int(metrics["bond_fn"])),
    )


def strict_winner_removed_nodes(
    baseline_utility: tuple[int, ...], actions: list[Mapping]
) -> list[int]:
    best_utility = max(
        (
            tuple(action["utility"])
            for action in actions
            if action["safe"] and tuple(action["utility"]) > baseline_utility
        ),
        default=None,
    )
    return [
        int(action["removed"])
        for action in actions
        if best_utility is not None
        and action["safe"]
        and tuple(action["utility"]) == best_utility
    ]


def label_single_delete_actions(
    sample: Mapping,
    *,
    presence_threshold: float,
    maximum_distance_px: float = 16.0,
    primary_radius_px: float = 6.0,
    secondary_radius_px: float = 10.0,
) -> dict:
    all_kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
    baseline_primary = evaluate_kept_graph(
        sample,
        all_kept,
        presence_threshold=presence_threshold,
        radius_px=primary_radius_px,
    )
    baseline_secondary = evaluate_kept_graph(
        sample,
        all_kept,
        presence_threshold=presence_threshold,
        radius_px=secondary_radius_px,
        topology_aware=True,
    )
    baseline_utility = action_utility(baseline_primary)
    actions = []
    for removed in unique_delete_nodes(
        sample["anchors"], maximum_distance_px=maximum_distance_px
    ):
        kept = all_kept.clone()
        kept[removed] = False
        primary = evaluate_kept_graph(
            sample,
            kept,
            presence_threshold=presence_threshold,
            radius_px=primary_radius_px,
        )
        secondary = evaluate_kept_graph(
            sample,
            kept,
            presence_threshold=presence_threshold,
            radius_px=secondary_radius_px,
            topology_aware=True,
        )
        safe = action_is_safe(baseline_primary, primary) and action_is_safe(
            baseline_secondary, secondary
        )
        utility = action_utility(primary)
        actions.append(
            {
                "removed": int(removed),
                "safe": safe,
                "strict_gain": safe and utility > baseline_utility,
                "utility": utility,
                "primary": primary,
                "secondary": secondary,
            }
        )
    winners = strict_winner_removed_nodes(baseline_utility, actions)
    return {
        "baseline_primary": baseline_primary,
        "baseline_secondary": baseline_secondary,
        "actions": actions,
        "winner_removed_nodes": winners,
    }


def apply_oracle_single_delete(sample: Mapping, labels: Mapping) -> torch.Tensor:
    kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
    winners = labels["winner_removed_nodes"]
    if winners:
        kept[min(int(value) for value in winners)] = False
    return kept


__all__ = [
    "action_is_safe",
    "action_utility",
    "apply_oracle_single_delete",
    "evaluate_kept_graph",
    "label_single_delete_actions",
    "lexicographic_node_match",
    "predicted_adjacency",
    "strict_winner_removed_nodes",
    "unique_delete_nodes",
]
