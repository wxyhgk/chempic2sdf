"""Paired rendering consistency for supervised AtomPair graph context."""

from __future__ import annotations

from collections import defaultdict
import copy

import torch
from torch import nn
from torch.nn import functional as F


def flatten_target_atom_pairs(
    target_atom_pairs: list[list[tuple[int, int]]],
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = [pair for sample in target_atom_pairs for pair in sample]
    if not rows:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    return torch.tensor(rows, device=device, dtype=torch.long)


class AtomPairStyleResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("Style residual hidden dimension must be positive")
        self.delta = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, residual_hidden: torch.Tensor) -> torch.Tensor:
        return self.delta(residual_hidden).squeeze(-1)


@torch.no_grad()
def fold_style_residual_adapter(
    context_head, adapter: AtomPairStyleResidualAdapter, *, scale: float = 1.0
):
    if scale < 0.0:
        raise ValueError("Style adapter scale must be non-negative")
    folded = copy.deepcopy(context_head)
    final = folded.residual[-1]
    if final.in_features != adapter.delta.in_features:
        raise ValueError("Style adapter and context hidden dimensions differ")
    final.weight.add_(adapter.delta.weight, alpha=float(scale))
    final.bias.add_(adapter.delta.bias, alpha=float(scale))
    return folded


def paired_style_consistency_loss(
    logits: torch.Tensor,
    sample_indices: torch.Tensor,
    target_atom_pairs: torch.Tensor,
    molecule_group_indices: torch.Tensor,
    *,
    batch_size: int,
    maximum_atoms: int,
    valid_samples: torch.Tensor | None = None,
    normalization_counts: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not (
        len(logits) == len(sample_indices) == len(target_atom_pairs)
        and len(molecule_group_indices) == batch_size
    ):
        raise ValueError("Paired style consistency tensors differ in length")
    if target_atom_pairs.ndim != 2 or target_atom_pairs.size(-1) != 2:
        raise ValueError("Target atom pairs must have shape [groups,2]")
    if valid_samples is None:
        valid_samples = torch.ones(
            (batch_size,), device=logits.device, dtype=torch.bool
        )
    if len(valid_samples) != batch_size:
        raise ValueError("Paired style valid-sample mask differs from batch size")
    if normalization_counts is not None and len(normalization_counts) != batch_size:
        raise ValueError("Paired style normalization counts differ from batch size")
    probabilities = logits.float().sigmoid()
    zero = probabilities.sum() * 0.0
    edge_losses = []
    count_losses = []
    degree_losses = []
    aligned_pairs = 0

    for molecule_group in molecule_group_indices.unique(sorted=True).tolist():
        rendering_indices = torch.where(
            molecule_group_indices == molecule_group
        )[0].tolist()
        if len(rendering_indices) < 2 or not bool(
            valid_samples[rendering_indices].all()
        ):
            continue

        counts = torch.stack(
            [probabilities[sample_indices == index].sum() for index in rendering_indices]
        )
        if normalization_counts is not None:
            counts = counts / normalization_counts[rendering_indices].float().clamp_min(
                1.0
            )
        count_losses.append(F.mse_loss(counts, counts.mean().expand_as(counts)))

        values_by_pair: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        degrees = []
        degree_available = []
        for sample_index in rendering_indices:
            sample = sample_indices == sample_index
            sample_pairs = target_atom_pairs[sample]
            sample_probabilities = probabilities[sample]
            valid = (sample_pairs >= 0).all(-1)
            sample_pairs = sample_pairs[valid]
            sample_probabilities = sample_probabilities[valid]
            degree = probabilities.new_zeros((maximum_atoms,))
            available = torch.zeros(
                (maximum_atoms,), device=probabilities.device, dtype=torch.bool
            )
            for pair, probability in zip(sample_pairs.tolist(), sample_probabilities):
                first, second = map(int, pair)
                if first >= maximum_atoms or second >= maximum_atoms:
                    raise ValueError("Target atom pair exceeds configured capacity")
                values_by_pair[(first, second)].append(probability)
                degree[first] = degree[first] + probability
                degree[second] = degree[second] + probability
                available[first] = True
                available[second] = True
            degrees.append(degree)
            degree_available.append(available)

        for values in values_by_pair.values():
            if len(values) < 2:
                continue
            stacked = torch.stack(values)
            edge_losses.append(F.mse_loss(stacked, stacked.mean().expand_as(stacked)))
            aligned_pairs += 1

        availability = torch.stack(degree_available)
        shared_atoms = availability.all(0)
        if bool(shared_atoms.any()):
            stacked_degrees = torch.stack(degrees)[:, shared_atoms]
            mean_degree = stacked_degrees.mean(0, keepdim=True)
            degree_losses.append(
                F.mse_loss(stacked_degrees, mean_degree.expand_as(stacked_degrees))
            )

    edge = torch.stack(edge_losses).mean() if edge_losses else zero
    count = torch.stack(count_losses).mean() if count_losses else zero
    degree = torch.stack(degree_losses).mean() if degree_losses else zero
    return {
        "edge_consistency_loss": edge,
        "count_consistency_loss": count,
        "degree_consistency_loss": degree,
        "aligned_target_pairs": logits.new_tensor(
            aligned_pairs, dtype=torch.float32
        ),
    }


__all__ = [
    "AtomPairStyleResidualAdapter",
    "flatten_target_atom_pairs",
    "fold_style_residual_adapter",
    "paired_style_consistency_loss",
]
