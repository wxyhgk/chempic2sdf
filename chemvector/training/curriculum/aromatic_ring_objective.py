"""Differentiable aromatic-ring safety objective for pair chemical logits.

This module is deliberately detached from inference and the production
decoder.  It uses only confirmed training pair labels and never changes pair
presence, atom assignments, or non-aromatic graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPE_TO_ID, BOND_TYPES


@dataclass(frozen=True, slots=True)
class AromaticRingObjectiveConfig:
    """Soft safety floors corresponding to the production ring gate.

    The production gate is mean >= .40 and minimum >= .05.  The objective
    uses small positive headroom so a fitted chemical head does not sit on a
    hard decoder boundary.
    """

    mean_floor: float = 0.45
    minimum_floor: float = 0.10
    mean_weight: float = 1.0
    minimum_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_floor <= self.mean_floor < 1.0:
            raise ValueError("Aromatic safety floors are invalid")
        if self.mean_weight < 0.0 or self.minimum_weight < 0.0:
            raise ValueError("Aromatic safety weights must be non-negative")
        if self.mean_weight == 0.0 and self.minimum_weight == 0.0:
            raise ValueError("At least one aromatic safety weight is required")


def _validate_inputs(
    chemical_logits: torch.Tensor,
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
) -> None:
    if chemical_logits.ndim != 2 or chemical_logits.shape[1] != len(BOND_TYPES):
        raise ValueError("Chemical logits must have shape [pairs, bond_types]")
    pairs = len(chemical_logits)
    if pair_indices.shape != (pairs, 2) or pair_indices.dtype != torch.long:
        raise ValueError("Pair indices must have shape [pairs, 2] and dtype long")
    if (sample_indices.shape != (pairs,) or sample_indices.dtype != torch.long
            or known_mask.shape != (pairs,) or known_mask.dtype != torch.bool
            or presence_targets.shape != (pairs,)
            or chemical_targets.shape != (pairs,)):
        raise ValueError("Aromatic objective inputs have incompatible shapes")
    if chemical_targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Chemical targets must be integer class ids")
    if presence_targets.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Presence targets must be boolean or integer labels")
    if bool((pair_indices[:, 0] == pair_indices[:, 1]).any()):
        raise ValueError("A pair cannot connect an atom to itself")
    if bool((pair_indices < 0).any()):
        raise ValueError("Pair indices must be non-negative")
    if not bool(torch.isfinite(chemical_logits).all()):
        raise ValueError("Chemical logits must be finite")
    if bool(((chemical_targets < 0) | (chemical_targets >= len(BOND_TYPES))).any()):
        raise ValueError("Chemical targets are out of range")
    if bool(((presence_targets != 0) & (presence_targets != 1)).any()):
        raise ValueError("Presence targets must be binary")
    if bool((~known_mask & presence_targets.bool()).any()):
        raise ValueError("Unknown pairs cannot be positive aromatic targets")


def _aromatic_cycles(
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
) -> list[list[int]]:
    """Return row ids for complete 5/6-member target aromatic cycle basis.

    A target molecule can contain many chordless walks once noisy or fused
    labels are projected onto predicted pairs.  The objective is a bounded
    regularizer, so it uses NetworkX's linear-time cycle basis instead of an
    unbounded all-cycle enumeration.  Production inference keeps its own
    promotion algorithm unchanged.
    """
    aromatic_id = BOND_TYPE_TO_ID["AROMATIC"]
    result: list[list[int]] = []
    for sample in torch.unique(sample_indices, sorted=True).tolist():
        rows = [
            row for row in range(len(pair_indices))
            if int(sample_indices[row]) == int(sample)
            and bool(known_mask[row])
            and bool(presence_targets[row])
            and int(chemical_targets[row]) == aromatic_id
        ]
        graph = nx.Graph()
        pair_to_row: dict[tuple[int, int], int] = {}
        for row in rows:
            pair = tuple(sorted(int(value) for value in pair_indices[row].tolist()))
            if pair in pair_to_row:
                raise ValueError("Duplicate target pair in one sample")
            pair_to_row[pair] = row
            graph.add_edge(*pair)
        for cycle in nx.cycle_basis(graph):
            if len(cycle) not in {5, 6}:
                continue
            cycle_pairs = [
                tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
                for index in range(len(cycle))
            ]
            result.append([pair_to_row[pair] for pair in cycle_pairs])
    return result


def aromatic_ring_safety_loss(
    chemical_logits: torch.Tensor,
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    config: AromaticRingObjectiveConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """Penalize confirmed aromatic rings that approach decoder boundaries.

    The returned loss is differentiable with respect to ``chemical_logits``.
    Graph discovery is performed on detached labels only.  If no complete
    target ring is reachable, a differentiable scalar zero and an explicit
    diagnostic are returned.
    """
    _validate_inputs(
        chemical_logits, pair_indices, sample_indices, known_mask,
        presence_targets, chemical_targets,
    )
    config = config or AromaticRingObjectiveConfig()
    probabilities = chemical_logits.float().softmax(-1)[:, BOND_TYPE_TO_ID["AROMATIC"]]
    cycles = _aromatic_cycles(
        pair_indices, sample_indices, known_mask, presence_targets, chemical_targets,
    )
    if not cycles:
        zero = chemical_logits.sum() * 0.0
        return zero, {
            "rings": 0,
            "mean_loss": 0.0,
            "minimum_loss": 0.0,
            "reachable_aromatic_edges": 0,
        }
    mean_terms = []
    minimum_terms = []
    for rows in cycles:
        values = probabilities[torch.tensor(rows, device=probabilities.device)]
        mean_terms.append(F.relu(values.new_tensor(config.mean_floor) - values.mean()).square())
        minimum_terms.append(F.relu(values.new_tensor(config.minimum_floor) - values.min()).square())
    mean_loss = torch.stack(mean_terms).mean()
    minimum_loss = torch.stack(minimum_terms).mean()
    loss = config.mean_weight * mean_loss + config.minimum_weight * minimum_loss
    return loss, {
        "rings": len(cycles),
        "mean_loss": float(mean_loss.detach()),
        "minimum_loss": float(minimum_loss.detach()),
        "reachable_aromatic_edges": len({row for cycle in cycles for row in cycle}),
    }


__all__ = ["AromaticRingObjectiveConfig", "aromatic_ring_safety_loss"]
