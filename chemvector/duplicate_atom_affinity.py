"""Graph-aware same-object affinity for close predicted AtomObjects."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs


@dataclass(frozen=True, slots=True)
class DuplicateAtomAffinityConfig:
    atom_feature_dim: int = 128
    geometry_dim: int = 8
    context_dim: int = 10
    hidden_dim: int = 96
    maximum_distance_px: float = 16.0

    def __post_init__(self) -> None:
        if min(
            self.atom_feature_dim,
            self.geometry_dim,
            self.context_dim,
            self.hidden_dim,
            self.maximum_distance_px,
        ) <= 0:
            raise ValueError("Duplicate affinity dimensions must be positive")


def enumerate_close_atom_pairs(
    atom_mask: torch.Tensor,
    anchors: torch.Tensor,
    *,
    maximum_distance_px: float,
    image_size: float = 768.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs, sample_indices = enumerate_unordered_atom_pairs(atom_mask)
    if not len(pairs):
        return pairs, sample_indices
    distances = torch.linalg.vector_norm(
        anchors[sample_indices, pairs[:, 0]] - anchors[sample_indices, pairs[:, 1]],
        dim=-1,
    ) * image_size
    selected = distances <= maximum_distance_px
    return pairs[selected], sample_indices[selected]


def _bbox_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_min = left[:, :2] - left[:, 2:]
    left_max = left[:, :2] + left[:, 2:]
    right_min = right[:, :2] - right[:, 2:]
    right_max = right[:, :2] + right[:, 2:]
    intersection = (
        (torch.minimum(left_max, right_max) - torch.maximum(left_min, right_min))
        .clamp_min(0.0)
        .prod(dim=-1)
    )
    area_left = (2.0 * left[:, 2:]).prod(dim=-1)
    area_right = (2.0 * right[:, 2:]).prod(dim=-1)
    return intersection / (area_left + area_right - intersection).clamp_min(1e-9)


def duplicate_pair_context(
    candidate_pairs: torch.Tensor,
    candidate_samples: torch.Tensor,
    atoms: dict[str, torch.Tensor],
    direct_pairs: torch.Tensor,
    direct_samples: torch.Tensor,
    presence_probabilities: torch.Tensor,
    *,
    presence_threshold: float,
    maximum_distance_px: float,
) -> torch.Tensor:
    if not len(candidate_pairs):
        return atoms["atom_scores"].new_empty((0, 10))
    rows = []
    for sample_index in candidate_samples.unique(sorted=True).tolist():
        direct_rows = torch.where(direct_samples == sample_index)[0]
        adjacency: dict[int, set[int]] = {}
        pair_probability = {}
        for row in direct_rows.tolist():
            left, right = (int(value) for value in direct_pairs[row])
            probability = float(presence_probabilities[row].detach())
            pair_probability[(min(left, right), max(left, right))] = probability
            if probability >= presence_threshold:
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
        candidate_rows = torch.where(candidate_samples == sample_index)[0]
        for row in candidate_rows.tolist():
            left, right = (int(value) for value in candidate_pairs[row])
            left_neighbors = adjacency.get(left, set()) - {right}
            right_neighbors = adjacency.get(right, set()) - {left}
            union = left_neighbors | right_neighbors
            intersection = left_neighbors & right_neighbors
            jaccard = len(intersection) / max(len(union), 1)
            left_score = float(atoms["atom_scores"][sample_index, left].detach())
            right_score = float(atoms["atom_scores"][sample_index, right].detach())
            distance = float(
                torch.linalg.vector_norm(
                    atoms["atom_geometry"][sample_index, left, :2]
                    - atoms["atom_geometry"][sample_index, right, :2]
                ).detach()
                * 768.0
            )
            left_bbox = atoms["atom_geometry"][sample_index, left, 4:8][None]
            right_bbox = atoms["atom_geometry"][sample_index, right, 4:8][None]
            rows.append(
                (
                    row,
                    atoms["atom_scores"].new_tensor(
                        [
                            min(distance / maximum_distance_px, 1.0),
                            min(left_score, right_score),
                            max(left_score, right_score),
                            abs(left_score - right_score),
                            pair_probability.get((min(left, right), max(left, right)), 0.0),
                            jaccard,
                            float(left_neighbors == right_neighbors and bool(left_neighbors)),
                            min(len(left_neighbors), len(right_neighbors)) / 8.0,
                            max(len(left_neighbors), len(right_neighbors)) / 8.0,
                            float(_bbox_iou(left_bbox, right_bbox)[0].detach()),
                        ]
                    ),
                )
            )
    rows.sort(key=lambda item: item[0])
    return torch.stack([value for _, value in rows])


class DuplicateAtomAffinity(nn.Module):
    def __init__(self, config: DuplicateAtomAffinityConfig | None = None) -> None:
        super().__init__()
        self.config = config or DuplicateAtomAffinityConfig()
        input_dim = (
            2 * self.config.atom_feature_dim
            + self.config.geometry_dim
            + self.config.context_dim
        )
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        nn.init.constant_(self.network[-1].bias, -4.0)

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_geometry: torch.Tensor,
        pairs: torch.Tensor,
        sample_indices: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.shape != (len(pairs), self.config.context_dim):
            raise ValueError("Duplicate affinity context shape differs")
        if not len(pairs):
            return atom_features.new_empty(0)
        left = atom_features[sample_indices, pairs[:, 0]]
        right = atom_features[sample_indices, pairs[:, 1]]
        geometry_left = atom_geometry[sample_indices, pairs[:, 0]]
        geometry_right = atom_geometry[sample_indices, pairs[:, 1]]
        values = torch.cat(
            (
                (left - right).abs(),
                left * right,
                (geometry_left - geometry_right).abs(),
                context.to(left.dtype),
            ),
            dim=-1,
        )
        return self.network(values).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def duplicate_atom_affinity_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("Duplicate affinity targets shape differs")
    positives = targets.sum().clamp_min(1)
    negatives = (~targets).sum().clamp_min(1)
    return F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        pos_weight=(negatives / positives).clamp(1.0, 32.0),
    )


__all__ = [
    "DuplicateAtomAffinity",
    "DuplicateAtomAffinityConfig",
    "duplicate_atom_affinity_loss",
    "duplicate_pair_context",
    "enumerate_close_atom_pairs",
]
