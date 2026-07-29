"""Graph-level chemical semantics for Direct AtomPair Bond predictions."""

from __future__ import annotations

import networkx as nx
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID, BOND_TYPES


def promote_complete_aromatic_rings(
    pair_indices: torch.Tensor,
    presence_probabilities: torch.Tensor,
    chemical_logits: torch.Tensor,
    *,
    presence_threshold: float,
    mean_threshold: float,
    minimum_threshold: float,
) -> tuple[torch.Tensor, dict]:
    """Promote complete 5/6-member predicted rings as one chemical object."""

    if pair_indices.shape != (len(presence_probabilities), 2):
        raise ValueError("AtomPair geometry and presence count differ")
    if chemical_logits.shape != (len(pair_indices), len(BOND_TYPES)):
        raise ValueError("AtomPair chemical logits have an invalid shape")
    if not 0.0 < presence_threshold < 1.0:
        raise ValueError("Presence threshold must be in (0, 1)")
    if not 0.0 < minimum_threshold <= mean_threshold < 1.0:
        raise ValueError("Aromatic thresholds are invalid")

    selected = presence_probabilities >= presence_threshold
    graph = nx.Graph()
    pair_to_row = {}
    for row in torch.where(selected)[0].tolist():
        pair = tuple(sorted(int(value) for value in pair_indices[row].tolist()))
        graph.add_edge(*pair)
        pair_to_row[pair] = row
    aromatic_id = BOND_TYPE_TO_ID["AROMATIC"]
    aromatic = chemical_logits.float().softmax(-1)[:, aromatic_id]
    chemical = chemical_logits.argmax(-1).clone()
    promoted_rows = set()
    accepted = []
    candidates = 0
    for cycle in nx.chordless_cycles(graph, length_bound=6):
        if len(cycle) not in {5, 6}:
            continue
        candidates += 1
        pairs = [
            tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
            for index in range(len(cycle))
        ]
        rows = [pair_to_row[pair] for pair in pairs]
        values = aromatic[rows]
        if float(values.mean()) < mean_threshold:
            continue
        if float(values.min()) < minimum_threshold:
            continue
        promoted_rows.update(rows)
        accepted.append(
            {
                "size": len(cycle),
                "mean_aromatic_probability": float(values.mean()),
                "minimum_aromatic_probability": float(values.min()),
            }
        )
    changed = 0
    if promoted_rows:
        rows = torch.tensor(sorted(promoted_rows), device=chemical.device)
        changed = int((chemical[rows] != aromatic_id).sum())
        chemical[rows] = aromatic_id
    return chemical, {
        "candidate_cycles": candidates,
        "accepted_cycles": len(accepted),
        "promoted_bonds": len(promoted_rows),
        "changed_bond_types": changed,
        "accepted_cycle_details": accepted,
    }
