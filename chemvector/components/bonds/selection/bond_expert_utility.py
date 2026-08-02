"""Inference-safe features for choosing between BondObject experts."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch


SCORE_THRESHOLDS = (0.10, 0.20, 0.25, 0.30, 0.50, 0.75)
BOND_CLASSES = 4


def _safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float32)
    return float(array.mean()), float(array.std()), float(array.min()), float(array.max())


def _graph_features(objects: list[dict], *, radius_pixels: float) -> dict[str, float]:
    if not objects:
        return {
            "junction_count": 0.0,
            "projected_bond_count": 0.0,
            "raw_endpoint_clusters": 0.0,
            "self_loops": 0.0,
            "duplicate_pairs": 0.0,
            "components": 0.0,
            "degree_mean": 0.0,
            "degree_max": 0.0,
            "degree_over4": 0.0,
            "cycle_rank": 0.0,
        }
    endpoints = torch.stack(
        [item["endpoints"].detach().float().cpu() for item in objects]
    ).reshape(-1, 2)
    adjacency_matrix = torch.cdist(endpoints * 768.0, endpoints * 768.0) <= radius_pixels
    adjacency_matrix.fill_diagonal_(True)
    object_indices = torch.arange(len(objects))
    adjacency_matrix[object_indices * 2, object_indices * 2 + 1] = False
    adjacency_matrix[object_indices * 2 + 1, object_indices * 2] = False
    raw_clusters, endpoint_groups = connected_components(
        csr_matrix(adjacency_matrix.numpy()), directed=False, return_labels=True
    )
    pair_objects = {}
    self_loops = duplicate_pairs = 0
    for object_index, item in sorted(
        enumerate(objects), key=lambda pair: float(pair[1]["score"]), reverse=True
    ):
        begin = int(endpoint_groups[object_index * 2])
        end = int(endpoint_groups[object_index * 2 + 1])
        if begin == end:
            self_loops += 1
            continue
        pair = tuple(sorted((begin, end)))
        if pair in pair_objects:
            duplicate_pairs += 1
            continue
        pair_objects[pair] = item
    used_groups = sorted({value for pair in pair_objects for value in pair})
    remap = {old: new for new, old in enumerate(used_groups)}
    node_count = len(used_groups)
    adjacency = [set() for _ in range(node_count)]
    for begin_group, end_group in pair_objects:
        begin, end = remap[begin_group], remap[end_group]
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    components = 0
    seen = set()
    for start in range(node_count):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    degrees = [len(neighbors) for neighbors in adjacency]
    return {
        "junction_count": float(node_count),
        "projected_bond_count": float(len(pair_objects)),
        "raw_endpoint_clusters": float(raw_clusters),
        "self_loops": float(self_loops),
        "duplicate_pairs": float(duplicate_pairs),
        "components": float(components),
        "degree_mean": sum(degrees) / max(len(degrees), 1),
        "degree_max": float(max(degrees, default=0)),
        "degree_over4": float(sum(value > 4 for value in degrees)),
        "cycle_rank": float(
            max(len(pair_objects) - node_count + components, 0)
        ),
    }


def expert_features(
    ranked_objects: list[dict],
    *,
    object_threshold: float = 0.25,
    radius_pixels: float = 12.0,
) -> dict[str, float]:
    selected = [item for item in ranked_objects if float(item["score"]) >= object_threshold]
    scores = [float(item["score"]) for item in ranked_objects]
    selected_scores = [float(item["score"]) for item in selected]
    lengths = [
        float(torch.linalg.vector_norm(item["endpoints"][1] - item["endpoints"][0]))
        for item in selected
    ]
    centers = [item["endpoints"].float().mean(0) for item in selected]
    boundary = sum(
        bool(((center < 0.04) | (center > 0.96)).any()) for center in centers
    )
    score_mean, score_std, score_min, score_max = _safe_stats(selected_scores)
    length_mean, length_std, length_min, length_max = _safe_stats(lengths)
    types = Counter(int(item["chemical"]) for item in selected)
    result = {
        "selected_count": float(len(selected)),
        "score_mean": score_mean,
        "score_std": score_std,
        "score_min": score_min,
        "score_max": score_max,
        "top1_score": scores[0] if scores else 0.0,
        "top5_score_mean": sum(scores[:5]) / max(min(len(scores), 5), 1),
        "threshold_margin": min(
            (abs(value - object_threshold) for value in scores), default=1.0
        ),
        "length_mean": length_mean,
        "length_std": length_std,
        "length_min": length_min,
        "length_max": length_max,
        "boundary_center_fraction": boundary / max(len(selected), 1),
    }
    for threshold in SCORE_THRESHOLDS:
        result[f"count_score_ge_{threshold:.2f}"] = float(
            sum(value >= threshold for value in scores)
        )
    for class_id in range(BOND_CLASSES):
        result[f"chemical_fraction_{class_id}"] = types[class_id] / max(
            len(selected), 1
        )
    result.update(_graph_features(selected, radius_pixels=radius_pixels))
    return result


def disagreement_features(
    base_ranked: list[dict],
    expert_ranked: list[dict],
    *,
    object_threshold: float = 0.25,
) -> dict[str, float]:
    base = [item for item in base_ranked if float(item["score"]) >= object_threshold]
    expert = [item for item in expert_ranked if float(item["score"]) >= object_threshold]
    if base and expert:
        base_endpoints = torch.stack(
            [item["endpoints"].detach().float().cpu() for item in base]
        )
        expert_endpoints = torch.stack(
            [item["endpoints"].detach().float().cpu() for item in expert]
        )
        forward = torch.linalg.vector_norm(
            (base_endpoints[:, None] - expert_endpoints[None]) * 768.0,
            dim=-1,
        ).mean(-1)
        reverse = torch.linalg.vector_norm(
            (base_endpoints[:, None] - expert_endpoints.flip(1)[None]) * 768.0,
            dim=-1,
        ).mean(-1)
        costs = torch.minimum(forward, reverse).numpy()
        rows, columns = linear_sum_assignment(costs)
        matched_costs = [
            float(costs[row, column]) for row, column in zip(rows, columns)
        ]
        type_agreement = [
            int(base[row]["chemical"]) == int(expert[column]["chemical"])
            for row, column in zip(rows, columns)
        ]
        display_agreement = [
            int(base[row]["display"]) == int(expert[column]["display"])
            for row, column in zip(rows, columns)
        ]
        score_differences = [
            abs(float(base[row]["score"]) - float(expert[column]["score"]))
            for row, column in zip(rows, columns)
        ]
    else:
        matched_costs = []
        type_agreement = []
        display_agreement = []
        score_differences = []
    cost_mean, cost_std, _, cost_max = _safe_stats(matched_costs)
    score_difference_mean, _, _, score_difference_max = _safe_stats(score_differences)
    matched = len(matched_costs)
    return {
        "selected_count_difference": float(len(expert) - len(base)),
        "selected_count_absolute_difference": float(abs(len(expert) - len(base))),
        "matched_objects": float(matched),
        "unmatched_objects": float(len(base) + len(expert) - 2 * matched),
        "endpoint_match_mean_pixels": cost_mean,
        "endpoint_match_std_pixels": cost_std,
        "endpoint_match_max_pixels": cost_max,
        "chemical_type_agreement": sum(type_agreement) / max(matched, 1),
        "display_type_agreement": sum(display_agreement) / max(matched, 1),
        "matched_score_difference_mean": score_difference_mean,
        "matched_score_difference_max": score_difference_max,
    }


def combined_utility_features(
    base_ranked: list[dict],
    expert_ranked: list[dict],
    *,
    style_probability: float = 0.0,
    object_threshold: float = 0.25,
    radius_pixels: float = 12.0,
) -> dict[str, float]:
    features = {"style_probability": float(style_probability)}
    for prefix, ranked in (("base", base_ranked), ("expert", expert_ranked)):
        for name, value in expert_features(
            ranked,
            object_threshold=object_threshold,
            radius_pixels=radius_pixels,
        ).items():
            features[f"{prefix}_{name}"] = value
    features.update(
        disagreement_features(
            base_ranked, expert_ranked, object_threshold=object_threshold
        )
    )
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Bond expert utility features must be finite")
    return features
