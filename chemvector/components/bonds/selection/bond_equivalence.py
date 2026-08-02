"""Semantic equivalence operations for BondObject proposals."""

from __future__ import annotations

import torch


def collapse_incidence_pair_bonds(
    bond_presence_logits: torch.Tensor,
    bond_mask: torch.Tensor,
    incidence_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Keep one Bond proposal per unordered top-1 AtomObject incidence pair."""

    output_logits = bond_presence_logits.clone()
    output_mask = bond_mask.clone()
    groups: dict[tuple[int, int], list[int]] = {}
    self_pair_candidates = 0
    for bond_index in torch.where(bond_mask)[0].tolist():
        refs = incidence_logits[bond_index].argmax(-1).tolist()
        key = tuple(sorted((int(refs[0]), int(refs[1]))))
        groups.setdefault(key, []).append(bond_index)
        self_pair_candidates += int(key[0] == key[1])
    suppressed = []
    for indices in groups.values():
        keep = max(
            indices,
            key=lambda index: (float(bond_presence_logits[index]), -index),
        )
        for index in indices:
            if index == keep:
                continue
            output_mask[index] = False
            output_logits[index] = -10.0
            suppressed.append(index)
    return output_logits, output_mask, {
        "mode": "incidence-argmax-unordered-pair",
        "input_candidates": int(bond_mask.sum()),
        "groups": len(groups),
        "suppressed_candidates": len(suppressed),
        "self_pair_candidates": self_pair_candidates,
        "suppressed_indices": suppressed,
    }


__all__ = ["collapse_incidence_pair_bonds"]
