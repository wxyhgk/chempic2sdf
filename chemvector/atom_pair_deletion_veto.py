"""Edge-centric veto model for removing false-positive AtomPair bonds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_pair_bond_group import GROUP_FEATURE_DIM


RELATION_FEATURE_DIM = 18


@dataclass(frozen=True, slots=True)
class AtomPairDeletionVetoConfig:
    input_dim: int = GROUP_FEATURE_DIM
    relation_dim: int = RELATION_FEATURE_DIM
    hidden_dim: int = 128
    maximum_atoms: int = 128
    image_size: float = 768.0
    line_samples: int = 17
    corridor_half_width_pixels: float = 3.0
    intermediate_radius_pixels: float = 8.0
    overlap_radius_pixels: float = 3.0
    relation_only: bool = False

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.relation_dim,
            self.hidden_dim,
            self.maximum_atoms,
            self.line_samples,
        ) < 1:
            raise ValueError("Deletion-veto dimensions must be positive")
        if min(
            self.image_size,
            self.corridor_half_width_pixels,
            self.intermediate_radius_pixels,
            self.overlap_radius_pixels,
        ) <= 0.0:
            raise ValueError("Deletion-veto geometry scales must be positive")


def _point_segment_distance(
    points: torch.Tensor,
    begins: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Return distances for [..., points, segments, 2] broadcast inputs."""

    delta = ends - begins
    squared = delta.square().sum(-1).clamp_min(1e-12)
    projection = ((points - begins) * delta).sum(-1) / squared
    projection = projection.clamp(0.0, 1.0)
    closest = begins + projection[..., None] * delta
    return torch.linalg.vector_norm(points - closest, dim=-1)


def _sample_ink_corridor(
    ink: torch.Tensor,
    begins: torch.Tensor,
    ends: torch.Tensor,
    *,
    config: AtomPairDeletionVetoConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not len(begins):
        empty = begins.new_empty((0, config.line_samples))
        return empty, begins.new_empty((0, config.line_samples, 2))
    fractions = torch.linspace(
        0.08,
        0.92,
        config.line_samples,
        device=begins.device,
        dtype=begins.dtype,
    )
    points = begins[:, None] + fractions[None, :, None] * (
        ends - begins
    )[:, None]
    delta = ends - begins
    length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    normal = torch.stack((-delta[:, 1], delta[:, 0]), dim=-1) / length[:, None]
    offsets = begins.new_tensor(
        (-config.corridor_half_width_pixels, 0.0, config.corridor_half_width_pixels)
    ) / config.image_size
    corridor = points[:, :, None] + normal[:, None, None] * offsets[None, None, :, None]
    grid = corridor.mul(2.0).sub(1.0).reshape(1, len(begins) * config.line_samples * 3, 1, 2)
    sampled = F.grid_sample(
        ink[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled[0, 0, :, 0].reshape(len(begins), config.line_samples, 3)
    return sampled.amax(-1), points


def _sample_relation_features(
    pixel_values: torch.Tensor,
    atom_anchors: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_presence_logits: torch.Tensor,
    atom_pairs: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    config: AtomPairDeletionVetoConfig,
) -> torch.Tensor:
    if not len(atom_pairs):
        return pixel_values.new_empty((0, config.relation_dim))
    anchors = atom_anchors.float()
    pairs = atom_pairs.long()
    begins = anchors[pairs[:, 0]]
    ends = anchors[pairs[:, 1]]
    delta = ends - begins
    length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    unit = delta / length[:, None]
    doubled_direction = torch.stack(
        (unit[:, 0].square() - unit[:, 1].square(), 2.0 * unit[:, 0] * unit[:, 1]),
        dim=-1,
    )

    grayscale = pixel_values.float().add(1.0).mul(0.5).mean(0, keepdim=True)
    ink = (1.0 - grayscale).clamp(0.0, 1.0)
    line_ink, points = _sample_ink_corridor(ink, begins, ends, config=config)
    ink_mean = line_ink.mean(-1)
    ink_min = line_ink.amin(-1)
    ink_std = line_ink.std(-1, unbiased=False)
    ink_coverage = torch.sigmoid((line_ink - 0.12) * 20.0).mean(-1)

    point_matrix = points[:, :, None, :]
    other_begins = begins[None, None, :, :]
    other_ends = ends[None, None, :, :]
    distances = _point_segment_distance(point_matrix, other_begins, other_ends)
    eye = torch.eye(len(pairs), device=pairs.device, dtype=torch.bool)
    distances = distances.masked_fill(eye[:, None], torch.inf)
    other_strength = base_logits.float().sigmoid()[None, None, :]
    explained = (
        torch.exp(
            -distances.square()
            * (config.image_size / config.overlap_radius_pixels) ** 2
        )
        * other_strength
    ).amax(-1)
    explained_mean = explained.mean(-1)
    explained_max = explained.amax(-1)
    exclusive_ink = (line_ink * (1.0 - explained)).mean(-1)
    exclusive_ratio = exclusive_ink / ink_mean.clamp_min(1e-4)

    atom_strength = atom_presence_logits.float().sigmoid() * atom_mask.float()
    atom_points = anchors[None]
    segment_delta = delta[:, None]
    squared_length = segment_delta.square().sum(-1).clamp_min(1e-12)
    projection = ((atom_points - begins[:, None]) * segment_delta).sum(-1) / squared_length
    closest = begins[:, None] + projection[..., None] * segment_delta
    atom_distance = torch.linalg.vector_norm(atom_points - closest, dim=-1)
    interior = (projection > 0.08) & (projection < 0.92) & atom_mask[None]
    endpoint_atoms = F.one_hot(pairs[:, 0], config.maximum_atoms).bool() | F.one_hot(
        pairs[:, 1], config.maximum_atoms
    ).bool()
    interior &= ~endpoint_atoms
    intermediate = torch.exp(
        -atom_distance.square()
        * (config.image_size / config.intermediate_radius_pixels) ** 2
    ) * atom_strength[None]
    intermediate = intermediate.masked_fill(~interior, 0.0)
    intermediate_max = intermediate.amax(-1)
    intermediate_sum = intermediate.sum(-1).log1p()

    maximum_atoms = config.maximum_atoms
    adjacency = base_logits.new_zeros((maximum_atoms, maximum_atoms)).float()
    probabilities = base_logits.float().sigmoid()
    adjacency[pairs[:, 0], pairs[:, 1]] = probabilities
    adjacency[pairs[:, 1], pairs[:, 0]] = probabilities
    common = torch.minimum(adjacency[pairs[:, 0]], adjacency[pairs[:, 1]])
    common = common.masked_fill(endpoint_atoms, 0.0)
    common_max = common.amax(-1)
    common_sum = common.sum(-1).log1p()
    common_collinear = (common * intermediate).amax(-1)

    degrees = adjacency.sum(-1)
    first_degree = degrees[pairs[:, 0]]
    second_degree = degrees[pairs[:, 1]]
    degree_mean = (first_degree + second_degree) * 0.5
    degree_difference = (first_degree - second_degree).abs()

    features = torch.cat(
        (
            (length / 2.0**0.5)[:, None],
            doubled_direction,
            ink_mean[:, None],
            ink_min[:, None],
            ink_std[:, None],
            ink_coverage[:, None],
            explained_mean[:, None],
            explained_max[:, None],
            exclusive_ink[:, None],
            exclusive_ratio[:, None],
            intermediate_max[:, None],
            intermediate_sum[:, None],
            common_max[:, None],
            common_sum[:, None],
            common_collinear[:, None],
            (degree_mean / 4.0)[:, None],
            (degree_difference / 4.0)[:, None],
        ),
        dim=-1,
    )
    if features.size(-1) != config.relation_dim:
        raise RuntimeError(
            f"Deletion-veto relation dim {features.size(-1)} != {config.relation_dim}"
        )
    return features


def build_atom_pair_relation_features(
    pixel_values: torch.Tensor,
    atom_anchors: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_presence_logits: torch.Tensor,
    sample_indices: torch.Tensor,
    atom_pairs: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    batch_size: int,
    config: AtomPairDeletionVetoConfig,
) -> torch.Tensor:
    if atom_anchors.shape[:2] != atom_mask.shape:
        raise ValueError("Atom anchors and mask dimensions differ")
    if atom_presence_logits.shape != atom_mask.shape:
        raise ValueError("Atom presence and mask dimensions differ")
    if atom_anchors.size(1) != config.maximum_atoms:
        raise ValueError("Atom capacity differs from deletion-veto config")
    output = pixel_values.new_empty((len(atom_pairs), config.relation_dim))
    for sample_index in range(batch_size):
        selected = sample_indices == sample_index
        if not bool(selected.any()):
            continue
        output[selected] = _sample_relation_features(
            pixel_values[sample_index],
            atom_anchors[sample_index],
            atom_mask[sample_index],
            atom_presence_logits[sample_index],
            atom_pairs[selected],
            base_logits[selected],
            config=config,
        )
    return output


class AtomPairDeletionVetoHead(nn.Module):
    """Score whether a base-positive edge should be removed."""

    def __init__(self, config: AtomPairDeletionVetoConfig) -> None:
        super().__init__()
        self.config = config
        if config.relation_only:
            self.local_encoder = None
            self.veto = nn.Sequential(
                nn.Linear(config.relation_dim + 1, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, 1),
            )
            nn.init.zeros_(self.veto[-1].weight)
            nn.init.constant_(self.veto[-1].bias, -4.0)
            return
        self.local_encoder = nn.Sequential(
            nn.Linear(config.input_dim + config.relation_dim + 1, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        context_dim = config.hidden_dim * 5 + config.relation_dim + 1
        self.veto = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.veto[-1].weight)
        nn.init.constant_(self.veto[-1].bias, -4.0)

    def forward(
        self,
        group_features: torch.Tensor,
        relation_features: torch.Tensor,
        base_logits: torch.Tensor,
        sample_indices: torch.Tensor,
        atom_pairs: torch.Tensor,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if not (
            len(group_features)
            == len(relation_features)
            == len(base_logits)
            == len(sample_indices)
            == len(atom_pairs)
        ):
            raise ValueError("Deletion-veto group tensors differ in length")
        if not len(group_features):
            return {"deletion_logits": base_logits.new_empty((0,))}
        if self.config.relation_only:
            return {
                "deletion_logits": self.veto(
                    torch.cat((relation_features, base_logits[:, None]), dim=-1)
                ).squeeze(-1)
            }
        hidden = self.local_encoder(
            torch.cat((group_features, relation_features, base_logits[:, None]), dim=-1)
        )
        endpoint_mean = hidden.new_zeros((self.config.maximum_atoms, self.config.hidden_dim))
        endpoint_max = hidden.new_full(endpoint_mean.shape, -torch.inf)
        first = hidden.new_zeros(hidden.shape)
        second = hidden.new_zeros(hidden.shape)
        sample_mean = hidden.new_zeros((batch_size, self.config.hidden_dim))
        sample_max = hidden.new_zeros((batch_size, self.config.hidden_dim))
        for sample_index in range(batch_size):
            mask = sample_indices == sample_index
            if not bool(mask.any()):
                continue
            values = hidden[mask]
            pairs = atom_pairs[mask]
            incidence = (
                F.one_hot(pairs[:, 0], self.config.maximum_atoms)
                + F.one_hot(pairs[:, 1], self.config.maximum_atoms)
            ).to(values.dtype)
            sums = incidence.T @ values
            counts = incidence.sum(0).clamp_min(1.0)
            endpoint_mean = sums / counts[:, None]
            endpoint_max.zero_()
            for atom_index in pairs.unique().tolist():
                endpoint_max[atom_index] = values[
                    (pairs == atom_index).any(-1)
                ].amax(0)
            first[mask] = torch.cat(
                (endpoint_mean[pairs[:, 0]], endpoint_max[pairs[:, 0]]), dim=-1
            ).reshape(len(values), 2, self.config.hidden_dim).mean(1)
            second[mask] = torch.cat(
                (endpoint_mean[pairs[:, 1]], endpoint_max[pairs[:, 1]]), dim=-1
            ).reshape(len(values), 2, self.config.hidden_dim).mean(1)
            sample_mean[sample_index] = values.mean(0)
            sample_max[sample_index] = values.amax(0)
        symmetric_endpoints = torch.cat(
            ((first + second) * 0.5, (first - second).abs()), dim=-1
        )
        context = torch.cat(
            (
                hidden,
                symmetric_endpoints,
                sample_mean[sample_indices],
                sample_max[sample_indices],
                relation_features,
                base_logits[:, None],
            ),
            dim=-1,
        )
        return {"deletion_logits": self.veto(context).squeeze(-1)}


def atom_pair_deletion_veto_loss(
    deletion_logits: torch.Tensor,
    edge_targets: torch.Tensor,
    base_logits: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    ranking_margin: float = 1.0,
    deletion_positive_weight: float | None = None,
    ranking_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    base_positive = base_logits >= 0.0
    deletion_targets = ~edge_targets
    mask = base_positive
    if not bool(mask.any()):
        zero = deletion_logits.sum() * 0.0
        return {"loss": zero, "binary_loss": zero, "ranking_loss": zero}
    selected_targets = deletion_targets[mask].float()
    positives = selected_targets.sum().clamp_min(1.0)
    negatives = (1.0 - selected_targets).sum().clamp_min(1.0)
    if deletion_positive_weight is None:
        positive_weight = (negatives / positives).clamp(1.0, 16.0)
    else:
        positive_weight = deletion_logits.new_tensor(float(deletion_positive_weight))
    binary = F.binary_cross_entropy_with_logits(
        deletion_logits[mask].float(),
        selected_targets,
        pos_weight=positive_weight,
    )
    ranking = []
    for sample_index in sample_indices.unique().tolist():
        sample = (sample_indices == sample_index) & mask
        false_edges = deletion_logits[sample & deletion_targets]
        true_edges = deletion_logits[sample & ~deletion_targets]
        if len(false_edges) and len(true_edges):
            ranking.append(
                F.relu(float(ranking_margin) - false_edges.min() + true_edges.max())
            )
    ranking_loss = torch.stack(ranking).mean() if ranking else binary * 0.0
    return {
        "loss": binary + float(ranking_weight) * ranking_loss,
        "binary_loss": binary,
        "ranking_loss": ranking_loss,
    }


def apply_atom_pair_deletion_veto(
    base_logits: torch.Tensor,
    deletion_logits: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    deleted = (base_logits >= 0.0) & (deletion_logits >= float(threshold))
    refined = torch.where(deleted, base_logits.new_full((), -1e-4), base_logits)
    return refined, deleted


def load_atom_pair_deletion_veto_head(
    path: str | Path,
    device: torch.device,
) -> tuple[AtomPairDeletionVetoHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "atom-pair-deletion-veto-0.1":
        raise ValueError("Unsupported AtomPair deletion-veto checkpoint schema")
    model = AtomPairDeletionVetoHead(
        AtomPairDeletionVetoConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload
