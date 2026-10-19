"""A narrow CPU ranking contract, not a complete loss or training admission.

Only nodes with ONE spatially eligible candidate and reviewed stable fragments
may enter. Multi-candidate groups are explicitly rejected, never assigned an
argmax pseudo-label. No threshold, duplicate or graph-recovery guarantee.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SingleNodeContrast:
    node_id: str
    valid_slots: tuple[int, ...]
    fragment_slots: tuple[int, ...]


def singleton_rank_contract(logits, contrasts):
    if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cpu'
            or logits.dtype not in (torch.float32,torch.float64) or logits.ndim != 1
            or not torch.isfinite(logits).all()):
        raise ValueError('Finite one-dimensional CPU float32/64 logits required')
    if not contrasts or len({c.node_id for c in contrasts}) != len(contrasts):
        raise ValueError('Nonempty distinct node contrasts required')
    occupied, losses = set(), []
    for contrast in contrasts:
        if (not isinstance(contrast.node_id,str) or not contrast.node_id
                or len(contrast.valid_slots) != 1 or not contrast.fragment_slots):
            raise ValueError('Exactly one eligible candidate and nonempty stable fragments required')
        slots = contrast.valid_slots + contrast.fragment_slots
        if (len(set(slots)) != len(slots) or occupied & set(slots)
                or any(type(s) is not int or not 0 <= s < logits.numel() for s in slots)):
            raise ValueError('Disjoint unique in-range slots required; no cross-node ownership')
        occupied.update(slots)
        positive = logits[contrast.valid_slots[0]]
        negative = logits[list(contrast.fragment_slots)]
        losses.append(F.softplus(negative-positive).mean())
    return dict(loss=torch.stack(losses).mean(), per_node_losses=torch.stack(losses),
        supervised_slots=sorted(occupied), training_authorized=False,
        threshold_coverage_guaranteed=False, shared_parameter_direction_checked=False,
        scope='Singleton valid candidate versus its stable fragments only, equal node means')
