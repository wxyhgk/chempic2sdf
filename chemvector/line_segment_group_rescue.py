"""Graph-constrained rescue for line-supported AtomPair groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chemvector.line_segment_rescue import AtomPairLineCandidate


@dataclass(frozen=True, slots=True)
class LineSegmentGroupRescueConfig:
    minimum_support_segments: int = 2
    activation_margin: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.minimum_support_segments, bool) or self.minimum_support_segments < 1:
            raise ValueError("minimum_support_segments must be a positive integer")
        if self.activation_margin <= 0.0:
            raise ValueError("activation_margin must be positive")


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        self.parent[second_root] = first_root
        return True


def rescue_disconnected_atom_pair_groups(
    group_output: dict[str, torch.Tensor],
    groups: dict,
    *,
    atom_presence_logits: torch.Tensor,
    atom_mask: torch.Tensor,
    line_candidates: list[tuple[AtomPairLineCandidate, ...]],
    atom_threshold: float,
    group_threshold: float,
    config: LineSegmentGroupRescueConfig = LineSegmentGroupRescueConfig(),
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Activate only line-supported groups that join disconnected components.

    The routine never creates a new AtomPair and never changes bond type.  It
    only raises the presence logit of an existing learned group.  Kruskal-style
    component updates prevent multiple rescue edges from creating arbitrary
    cycles in an already connected prediction.
    """

    if len(line_candidates) != atom_mask.size(0):
        raise ValueError("line_candidates must contain one tuple per batch item")
    if len(groups.get("members", ())) != atom_mask.size(0):
        raise ValueError("AtomPair groups must contain one member list per batch item")
    if len(groups.get("atom_pairs", ())) != atom_mask.size(0):
        raise ValueError("AtomPair groups do not expose atom_pairs")
    rescued_output = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in group_output.items()
    }
    presence = rescued_output["presence_logits"]
    diagnostics = []
    group_offset = 0
    for batch_index, sample_pairs in enumerate(groups["atom_pairs"]):
        selected_atoms = (
            atom_mask[batch_index]
            & (atom_presence_logits[batch_index] >= atom_threshold)
        ).cpu()
        components = _DisjointSet(atom_mask.size(1))
        for local_index, pair in enumerate(sample_pairs):
            first, second = pair
            if first == second or not bool(selected_atoms[first] and selected_atoms[second]):
                continue
            if float(presence[group_offset + local_index]) >= group_threshold:
                components.union(first, second)

        support = {
            candidate.atom_pair: candidate
            for candidate in line_candidates[batch_index]
            if len(candidate.support_segment_indices)
            >= config.minimum_support_segments
        }
        candidates = []
        for local_index, pair in enumerate(sample_pairs):
            first, second = pair
            candidate = support.get(pair)
            if candidate is None:
                continue
            if first == second or not bool(selected_atoms[first] and selected_atoms[second]):
                continue
            index = group_offset + local_index
            if float(presence[index]) >= group_threshold:
                continue
            candidates.append((local_index, index, pair, candidate))
        candidates.sort(
            key=lambda item: (
                -len(item[3].support_segment_indices),
                item[3].endpoint_error_pixels,
                item[3].angle_error_degrees,
                -float(presence[item[1]]),
                item[2],
            )
        )
        rescued = []
        rejected_cycle = []
        for local_index, index, pair, candidate in candidates:
            if not components.union(*pair):
                rejected_cycle.append(list(pair))
                continue
            presence[index] = max(
                float(presence[index]), group_threshold + config.activation_margin
            )
            rescued.append(
                {
                    "group": local_index,
                    "atom_pair": list(pair),
                    "support_segments": len(candidate.support_segment_indices),
                    "endpoint_error_pixels": candidate.endpoint_error_pixels,
                    "line_length_pixels": candidate.maximum_support_length_pixels,
                }
            )
        diagnostics.append(
            {
                "supported_groups": len(candidates),
                "rescued_groups": rescued,
                "cycle_rejections": rejected_cycle,
            }
        )
        group_offset += len(sample_pairs)
    if group_offset != len(presence):
        raise ValueError("AtomPair group count differs from presence output")
    return rescued_output, diagnostics

