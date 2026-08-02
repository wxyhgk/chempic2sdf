"""Learn endpoint-to-Junction assignment with an explicit NULL outcome."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class JunctionBondAssemblerConfig:
    hidden_dim: int = 96

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("Assembler hidden dimension must be positive")


class JunctionBondAssembler(nn.Module):
    """Score every BondObject endpoint against every Junction and NULL."""

    pair_feature_dim = 17
    null_feature_dim = 7

    def __init__(self, config: JunctionBondAssemblerConfig | None = None) -> None:
        super().__init__()
        self.config = config or JunctionBondAssemblerConfig()
        self.pair_scorer = nn.Sequential(
            nn.Linear(self.pair_feature_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.null_scorer = nn.Sequential(
            nn.Linear(self.null_feature_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        roles: torch.Tensor,
        bond_endpoints: torch.Tensor,
        bond_scores: torch.Tensor,
    ) -> torch.Tensor:
        if points.ndim != 2 or points.size(-1) != 2:
            raise ValueError("points must have shape [junctions, 2]")
        if roles.shape != (len(points),):
            raise ValueError("roles and Junction points differ")
        if bond_endpoints.ndim != 3 or bond_endpoints.shape[1:] != (2, 2):
            raise ValueError("bond_endpoints must have shape [bonds, 2, 2]")
        if bond_scores.shape != (len(bond_endpoints),):
            raise ValueError("bond scores and endpoints differ")
        bonds = len(bond_endpoints)
        junctions = len(points)
        if junctions == 0:
            raise ValueError("At least one Junction candidate is required")
        endpoints = bond_endpoints[:, :, None, :].expand(-1, -1, junctions, -1)
        candidates = points[None, None].expand(bonds, 2, -1, -1)
        delta = candidates - endpoints
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        center = bond_endpoints.mean(dim=1)
        vector = bond_endpoints[:, 1] - bond_endpoints[:, 0]
        length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)
        unit = vector / length
        roles_one_hot = F.one_hot(roles.long(), num_classes=2).to(points.dtype)
        shared = torch.cat(
            (
                center,
                unit,
                length,
                bond_scores[:, None],
            ),
            dim=-1,
        )[:, None, None].expand(-1, 2, junctions, -1)
        pair_features = torch.cat(
            (
                delta,
                delta.abs(),
                distance,
                endpoints,
                candidates,
                shared,
                roles_one_hot[None, None].expand(bonds, 2, -1, -1),
            ),
            dim=-1,
        )
        pair_logits = self.pair_scorer(pair_features).squeeze(-1)
        nearest = distance.squeeze(-1).topk(
            min(2, max(junctions, 1)), largest=False, dim=-1
        ).values
        if junctions == 1:
            nearest = nearest.expand(-1, -1, 2)
        endpoint_index = torch.tensor(
            (0.0, 1.0), device=points.device, dtype=points.dtype
        )[None, :, None].expand(bonds, -1, -1)
        null_features = torch.cat(
            (
                nearest,
                bond_scores[:, None, None].expand(-1, 2, -1),
                length[:, None].expand(-1, 2, -1),
                bond_endpoints,
                endpoint_index,
            ),
            dim=-1,
        )
        null_logits = self.null_scorer(null_features)
        return torch.cat((pair_logits, null_logits), dim=-1)


def junction_bond_assembler_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    same_junction_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    if logits.shape[:2] != targets.shape or logits.ndim != 3:
        raise ValueError("Assembler logits and targets differ")
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    probabilities = logits.softmax(-1)[..., :-1]
    same_junction = (probabilities[:, 0] * probabilities[:, 1]).sum(-1).mean()
    return {
        "loss": loss + same_junction_weight * same_junction,
        "classification_loss": loss,
        "same_junction_loss": same_junction,
    }


@torch.no_grad()
def decode_junction_assignments(
    logits: torch.Tensor, bond_scores: torch.Tensor
) -> tuple[list[tuple[int, int] | None], dict]:
    if logits.ndim != 3 or logits.size(1) != 2:
        raise ValueError("Assembler logits must have shape [bonds, 2, classes]")
    null_id = logits.size(-1) - 1
    predictions = logits.argmax(-1)
    assignments: list[tuple[int, int] | None] = [None] * len(predictions)
    used_pairs = set()
    null_count = same_count = duplicate_count = 0
    order = bond_scores.argsort(descending=True).tolist()
    for bond_index in order:
        begin, end = predictions[bond_index].tolist()
        if begin == null_id or end == null_id:
            null_count += 1
            continue
        if begin == end:
            same_count += 1
            continue
        pair = tuple(sorted((begin, end)))
        if pair in used_pairs:
            duplicate_count += 1
            continue
        used_pairs.add(pair)
        assignments[bond_index] = pair
    return assignments, {
        "assembler_null_objects": null_count,
        "assembler_same_junction_objects": same_count,
        "assembler_duplicate_objects": duplicate_count,
    }


@torch.no_grad()
def decode_junction_assignment_rescue(
    logits: torch.Tensor,
    bond_scores: torch.Tensor,
    bond_endpoints: torch.Tensor,
    points: torch.Tensor,
) -> tuple[list[tuple[int, int] | None], dict]:
    """Keep nearest assignments, using learned pairs only for conflicts."""

    learned, learned_diagnostics = decode_junction_assignments(logits, bond_scores)
    if bond_endpoints.shape != (len(bond_scores), 2, 2):
        raise ValueError("Bond endpoint shape differs from scores")
    if points.ndim != 2 or points.size(-1) != 2 or not len(points):
        raise ValueError("Rescue decoding requires Junction points")
    nearest = torch.cdist(bond_endpoints, points).argmin(-1)
    assignments: list[tuple[int, int] | None] = [None] * len(bond_scores)
    used_pairs: set[tuple[int, int]] = set()
    rescued = rejected = 0
    for bond_index in bond_scores.argsort(descending=True).tolist():
        begin, end = nearest[bond_index].tolist()
        nearest_pair = tuple(sorted((begin, end)))
        if begin != end and nearest_pair not in used_pairs:
            assignments[bond_index] = nearest_pair
            used_pairs.add(nearest_pair)
            continue
        learned_pair = learned[bond_index]
        if learned_pair is not None and learned_pair not in used_pairs:
            assignments[bond_index] = learned_pair
            used_pairs.add(learned_pair)
            rescued += 1
        else:
            rejected += 1
    return assignments, {
        **learned_diagnostics,
        "assembler_rescued_conflicts": rescued,
        "assembler_rejected_conflicts": rejected,
    }
