"""Junction-negative emphasized veto objective for Evidence Carrier v0.9.22."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from chemvector.evidence_carrier_pair_local_veto import (
    PAIR_LOCAL_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairLocalVeto,
    pair_local_veto_loss,
)


PAIR_JUNCTION_VETO_PARAMETER_COUNT = PAIR_LOCAL_VETO_PARAMETER_COUNT
JUNCTION_NEGATIVE_LOSS_WEIGHT = 4.0
WORST_JUNCTION_NEGATIVE_LOSS_WEIGHT = 8.0
JUNCTION_VETO_MARGIN = 1.0


class EvidenceCarrierPairJunctionVeto(EvidenceCarrierPairLocalVeto):
    """Same deployment model as v0.9.21 with a distinct training contract."""

    model_type = "evidence-carrier-pair-junction-veto-v0.1"

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        config.update(
            {
                "model_type": self.model_type,
                "training_emphasis": "single-line junction negatives",
                "group_type_in_forward": False,
                "parameter_count": self.parameter_count(),
            }
        )
        return config


def junction_negative_mask(
    pair_group_types: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Select negative pairs involving at least one single-line group."""

    if pair_group_types.shape != (*pair_labels.shape, 2) or pair_group_types.dtype != torch.long:
        raise ValueError("pair_group_types must have long shape [batch,pairs,2]")
    if pair_labels.shape != pair_mask.shape or pair_labels.dtype != torch.long:
        raise ValueError("pair labels shape or dtype differs")
    if pair_mask.dtype != torch.bool:
        raise ValueError("pair mask must be boolean")
    active_types = pair_group_types[pair_mask]
    if len(active_types) and bool((active_types < 1).any()):
        raise ValueError("active pair group types must be positive")
    contains_single = (pair_group_types == 1).any(dim=-1)
    supported_other = ((pair_group_types >= 1) & (pair_group_types <= 3)).all(dim=-1)
    return pair_mask & (pair_labels == 0) & contains_single & supported_other


def pair_junction_veto_loss(
    veto_logits: torch.Tensor,
    raw_gain: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_group_types: torch.Tensor,
    *,
    tau0: float,
) -> dict[str, Any]:
    base = pair_local_veto_loss(
        veto_logits,
        raw_gain,
        pair_labels,
        pair_mask,
        tau0=tau0,
    )
    selected = junction_negative_mask(pair_group_types, pair_labels, pair_mask)
    zero = veto_logits.sum() * 0.0
    means = []
    worst = []
    image_count = pair_count = 0
    for image_index in range(veto_logits.size(0)):
        current = selected[image_index]
        if not bool(current.any()):
            continue
        logits = veto_logits[image_index, current].float()
        means.append(F.softplus(-logits).mean())
        worst.append(F.relu(JUNCTION_VETO_MARGIN - logits.min()).square())
        image_count += 1
        pair_count += int(current.sum())
    junction_loss = torch.stack(means).mean() if means else zero
    worst_junction_loss = torch.stack(worst).mean() if worst else zero
    total = (
        base["loss"]
        + JUNCTION_NEGATIVE_LOSS_WEIGHT * junction_loss
        + WORST_JUNCTION_NEGATIVE_LOSS_WEIGHT * worst_junction_loss
    )
    return {
        **base,
        "loss": total,
        "junction_negative_loss": junction_loss,
        "worst_junction_negative_loss": worst_junction_loss,
        "junction_negative_pair_count": pair_count,
        "junction_negative_image_count": image_count,
    }


__all__ = [
    "PAIR_JUNCTION_VETO_PARAMETER_COUNT",
    "EvidenceCarrierPairJunctionVeto",
    "junction_negative_mask",
    "pair_junction_veto_loss",
]
