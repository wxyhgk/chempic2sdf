"""Small directional model for one safe AtomObject deletion per image."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.safe_atom_suppression import predicted_adjacency, unique_delete_nodes


ATOM_FEATURE_DIM = 128
ATOM_GEOMETRY_DIM = 8
ACTION_SCALAR_DIM = 20
ACTION_FEATURE_DIM = ATOM_FEATURE_DIM * 3 + ATOM_GEOMETRY_DIM * 2 + ACTION_SCALAR_DIM


def _bbox_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left_min = left[:2] - left[2:]
    left_max = left[:2] + left[2:]
    right_min = right[:2] - right[2:]
    right_max = right[:2] + right[2:]
    intersection = (
        (torch.minimum(left_max, right_max) - torch.maximum(left_min, right_min))
        .clamp_min(0.0)
        .prod()
    )
    area_left = (2.0 * left[2:]).prod()
    area_right = (2.0 * right[2:]).prod()
    return float(intersection / (area_left + area_right - intersection).clamp_min(1e-9))


def delete_action_features(
    sample: dict,
    *,
    presence_threshold: float,
    maximum_distance_px: float = 16.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target-free features for unique DELETE(node) actions."""

    removed_nodes = unique_delete_nodes(
        sample["anchors"], maximum_distance_px=maximum_distance_px
    )
    if not removed_nodes:
        return torch.empty((0, ACTION_FEATURE_DIM)), torch.empty(0, dtype=torch.long)
    anchors = sample["anchors"].float()
    atom_features = sample["atom_features"].float()
    geometry = sample["atom_geometry"].float()
    scores = sample["scores"].float()
    # Keep action enumeration and feature extraction on the same numeric path.
    # float32 cdist can move a 15.999px pair above the 16px candidate boundary.
    distance_anchors = sample["anchors"].double()
    distances = torch.cdist(distance_anchors, distance_anchors) * 768.0
    adjacency = predicted_adjacency(sample, presence_threshold)
    pair_probability = {}
    for row, pair in enumerate(sample["pairs"].tolist()):
        left, right = sorted(int(value) for value in pair)
        pair_probability[(left, right)] = float(sample["probabilities"][row])
    rows = []
    for removed in removed_nodes:
        witnesses = torch.where(
            (distances[removed] <= maximum_distance_px)
            & (torch.arange(len(anchors)) != removed)
        )[0]
        witness_distances = distances[removed, witnesses]
        weights = torch.softmax(-witness_distances / 4.0, dim=0).float()
        pooled_feature = (atom_features[witnesses] * weights[:, None]).sum(dim=0)
        pooled_geometry_delta = (
            (geometry[witnesses] - geometry[removed]).abs() * weights[:, None]
        ).sum(dim=0)
        removed_neighbors = adjacency[removed]
        witness_degrees = torch.tensor(
            [len(adjacency[int(index)]) for index in witnesses], dtype=torch.float32
        )
        jaccards = []
        equal_adjacency = []
        common_counts = []
        removed_exclusive = []
        witness_exclusive = []
        direct_between = []
        bbox_ious = []
        witness_edge_means = []
        for witness_tensor in witnesses:
            witness = int(witness_tensor)
            left_neighbors = removed_neighbors - {witness}
            right_neighbors = adjacency[witness] - {removed}
            union = left_neighbors | right_neighbors
            common = left_neighbors & right_neighbors
            jaccards.append(len(common) / max(len(union), 1))
            equal_adjacency.append(float(bool(common) and left_neighbors == right_neighbors))
            common_counts.append(len(common))
            removed_exclusive.append(len(left_neighbors - right_neighbors))
            witness_exclusive.append(len(right_neighbors - left_neighbors))
            direct_between.append(
                pair_probability.get(tuple(sorted((removed, witness))), 0.0)
            )
            bbox_ious.append(_bbox_iou(geometry[removed, 4:8], geometry[witness, 4:8]))
            witness_values = [
                pair_probability.get(tuple(sorted((witness, neighbor))), 0.0)
                for neighbor in adjacency[witness]
            ]
            witness_edge_means.append(
                sum(witness_values) / max(len(witness_values), 1)
            )
        removed_edge_values = [
            pair_probability.get(tuple(sorted((removed, neighbor))), 0.0)
            for neighbor in removed_neighbors
        ]
        scalar = torch.tensor(
            [
                float(witness_distances.min() / maximum_distance_px),
                float(witness_distances.mean() / maximum_distance_px),
                len(witnesses) / 8.0,
                float(scores[removed]),
                float(scores[witnesses].max()),
                float(scores[witnesses].mean()),
                float((scores[witnesses] - scores[removed]).abs().max()),
                len(removed_neighbors) / 8.0,
                float(witness_degrees.max() / 8.0),
                float(witness_degrees.mean() / 8.0),
                max(jaccards),
                max(equal_adjacency),
                max(common_counts) / 8.0,
                max(removed_exclusive) / 8.0,
                max(witness_exclusive) / 8.0,
                max(direct_between),
                sum(direct_between) / len(direct_between),
                max(bbox_ious),
                sum(removed_edge_values) / max(len(removed_edge_values), 1),
                sum(witness_edge_means) / len(witness_edge_means),
            ],
            dtype=torch.float32,
        )
        rows.append(
            torch.cat(
                (
                    atom_features[removed],
                    pooled_feature,
                    (atom_features[removed] - pooled_feature).abs(),
                    geometry[removed],
                    pooled_geometry_delta,
                    scalar,
                )
            )
        )
    return torch.stack(rows), torch.tensor(removed_nodes, dtype=torch.long)


@dataclass(frozen=True, slots=True)
class SafeAtomActionConfig:
    input_dim: int = ACTION_FEATURE_DIM
    hidden_dim: int = 128
    bottleneck_dim: int = 64
    utility_classes: int = 4

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.bottleneck_dim,
            self.utility_classes,
        ) < 1:
            raise ValueError("Safe action dimensions must be positive")


class SafeAtomActionModel(nn.Module):
    def __init__(self, config: SafeAtomActionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SafeAtomActionConfig()
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.bottleneck_dim),
            nn.GELU(),
        )
        self.safety = nn.Linear(self.config.bottleneck_dim, 1)
        self.utility = nn.Linear(
            self.config.bottleneck_dim, self.config.utility_classes
        )
        self.rank = nn.Linear(self.config.bottleneck_dim, 1)
        nn.init.constant_(self.safety.bias, -4.0)
        nn.init.constant_(self.rank.bias, -2.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.size(-1) != self.config.input_dim:
            raise ValueError("Safe action features have an invalid shape")
        encoded = self.encoder(features)
        return {
            "safety_logits": self.safety(encoded).squeeze(-1),
            "utility_logits": self.utility(encoded),
            "rank_logits": self.rank(encoded).squeeze(-1),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def safe_atom_action_loss(
    output: dict[str, torch.Tensor],
    safe_targets: torch.Tensor,
    utility_targets: torch.Tensor,
    winner_targets: torch.Tensor,
    sample_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    safety_logits = output["safety_logits"].float()
    if not (
        safety_logits.shape
        == safe_targets.shape
        == utility_targets.shape
        == winner_targets.shape
        == sample_indices.shape
    ):
        raise ValueError("Safe action target shapes differ")
    unsafe_weight = torch.where(
        safe_targets,
        torch.ones_like(safety_logits),
        torch.full_like(safety_logits, 32.0),
    )
    safety = F.binary_cross_entropy_with_logits(
        safety_logits, safe_targets.float(), weight=unsafe_weight
    )
    utility = (
        F.cross_entropy(output["utility_logits"][safe_targets].float(), utility_targets[safe_targets])
        if safe_targets.any()
        else safety_logits.sum() * 0.0
    )
    action_scores = output["rank_logits"].float() + F.logsigmoid(safety_logits)
    listwise_rows = []
    for sample_index in sample_indices.unique(sorted=True):
        rows = torch.where(sample_indices == sample_index)[0]
        scores = action_scores[rows]
        denominator = torch.logsumexp(
            torch.cat((scores, scores.new_zeros(1))), dim=0
        )
        winners = winner_targets[rows]
        numerator = (
            torch.logsumexp(scores[winners], dim=0)
            if winners.any()
            else scores.new_zeros(())
        )
        listwise_rows.append(denominator - numerator)
    listwise = (
        torch.stack(listwise_rows).mean()
        if listwise_rows
        else safety_logits.sum() * 0.0
    )
    total = safety + utility * 0.5 + listwise * 0.25
    return {
        "total": total,
        "safety": safety,
        "utility": utility,
        "listwise": listwise,
    }


__all__ = [
    "ACTION_FEATURE_DIM",
    "SafeAtomActionConfig",
    "SafeAtomActionModel",
    "delete_action_features",
    "safe_atom_action_loss",
]
