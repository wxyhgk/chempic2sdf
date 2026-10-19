"""Conservative joint presence/chemical objective for a future bounded probe.

This module is intentionally not wired into a launcher.  It makes the label
contract explicit: confirmed non-edges supervise presence, confirmed positive
edges supervise chemical type, and unknown pairs are excluded from both
terms.  Optional initial-logit replay is a stability constraint, not a source
of labels.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .aromatic_ring_objective import AromaticRingObjectiveConfig, aromatic_ring_safety_loss
from chemvector.bond_object import BOND_TYPE_TO_ID


def _validate(
    output: dict[str, torch.Tensor],
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    presence = output.get("presence_logits")
    chemical = output.get("chemical_logits")
    if (not isinstance(presence, torch.Tensor) or presence.ndim != 1
            or not isinstance(chemical, torch.Tensor) or chemical.ndim != 2
            or chemical.shape[0] != len(presence)):
        raise ValueError("Pair output has invalid presence/chemical shapes")
    rows = len(presence)
    if any(t.device != presence.device for t in (known_mask, presence_targets, chemical_targets)):
        raise ValueError("Joint edge tensors must share a device")
    if (known_mask.shape != (rows,) or known_mask.dtype != torch.bool
            or presence_targets.shape != (rows,)
            or chemical_targets.shape != (rows,)
            or presence_targets.dtype not in (torch.bool, torch.int8, torch.int16,
                                               torch.int32, torch.int64)
            or chemical_targets.dtype not in (torch.int8, torch.int16,
                                               torch.int32, torch.int64)):
        raise ValueError("Joint edge labels must be aligned tensors")
    if not bool(torch.isfinite(presence).all()) or not bool(torch.isfinite(chemical).all()):
        raise ValueError("Pair logits must be finite")
    if not bool(((presence_targets == 0) | (presence_targets == 1)).all()):
        raise ValueError("Presence targets must be binary")
    if not bool(((chemical_targets >= 0) & (chemical_targets < chemical.shape[1])).all()):
        raise ValueError("Chemical targets are out of range")
    return presence, chemical, known_mask


def _bernoulli_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Forward KL(reference || current), with a detached teacher target."""
    reference = torch.sigmoid(reference_logits.detach().float()).clamp(1e-6, 1.0 - 1e-6)
    current = torch.sigmoid(current_logits.float()).clamp(1e-6, 1.0 - 1e-6)
    return (
        reference * (reference.log() - current.log())
        + (1.0 - reference) * ((1.0 - reference).log() - (1.0 - current).log())
    ).mean()


def joint_edge_loss(
    output: dict[str, torch.Tensor],
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    pair_indices: torch.Tensor | None = None,
    sample_indices: torch.Tensor | None = None,
    presence_weight: float = 1.0,
    chemical_weight: float = 1.0,
    replay_weight: float = 0.0,
    reference_presence_logits: torch.Tensor | None = None,
    reference_chemical_logits: torch.Tensor | None = None,
    aromatic_safety_weight: float = 0.0,
    aromatic_safety_config: AromaticRingObjectiveConfig | None = None,
    aromatic_anchor_weight: float = 0.0,
    aromatic_anchor_epsilon: float = 0.005,
) -> tuple[torch.Tensor | None, dict]:
    """Compute a masked, replay-constrained pair objective.

    The returned ``None`` means that no known pair exists.  Presence still
    receives gradients whenever confirmed non-edges are available, including
    batches with no positive edge; callers can use ``presence_only`` rather
    than silently skipping such a batch.
    """
    for name, value in (("presence_weight", presence_weight),
                        ("chemical_weight", chemical_weight),
                        ("replay_weight", replay_weight),
                        ("aromatic_safety_weight", aromatic_safety_weight),
                        ("aromatic_anchor_weight", aromatic_anchor_weight)):
        if not isinstance(value, (int, float)) or value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if not isinstance(aromatic_anchor_epsilon, (int, float)) or aromatic_anchor_epsilon < 0.0:
        raise ValueError("aromatic_anchor_epsilon must be non-negative")
    presence, chemical, known = _validate(output, known_mask, presence_targets, chemical_targets)
    known_rows = known.nonzero(as_tuple=False).flatten()
    if not len(known_rows):
        return None, {"known_rows": 0, "positive_rows": 0, "negative_rows": 0,
                      "presence_loss": None, "chemical_loss": None,
                      "replay_loss": None, "aromatic_safety_loss": None,
                      "aromatic_anchor_loss": None,
                      "presence_only": False}
    p = presence[known]
    pt = presence_targets[known].bool()
    positive = pt
    negative = ~pt
    positive_count = positive.sum().clamp_min(1)
    negative_count = negative.sum().clamp_min(1)
    presence_loss = F.binary_cross_entropy_with_logits(
        p.float(), pt.float(), pos_weight=(negative_count / positive_count).clamp(1.0, 16.0),
    )
    chemical_loss = None
    if bool(positive.any()):
        chemical_loss = F.cross_entropy(chemical[known][positive].float(), chemical_targets[known][positive].long())
    base = presence_loss * float(presence_weight)
    if chemical_loss is not None:
        base = base + chemical_loss * float(chemical_weight)

    replay = base.new_zeros(())
    if replay_weight > 0.0:
        if (reference_presence_logits is None or reference_chemical_logits is None
                or reference_presence_logits.shape != presence.shape
                or reference_chemical_logits.shape != chemical.shape
                or reference_presence_logits.device != presence.device
                or reference_chemical_logits.device != chemical.device):
            raise ValueError("Replay logits must match current pair outputs")
        if (not bool(torch.isfinite(reference_presence_logits).all())
                or not bool(torch.isfinite(reference_chemical_logits).all())):
            raise ValueError("Replay logits must be finite")
        # The reference is a teacher only; neither replay branch backpropagates
        # into it.  Both terms are zero when current and reference agree.
        replay = _bernoulli_kl(p, reference_presence_logits[known])
        # Chemical type is undefined for confirmed non-edges.  Replay its
        # distribution only on confirmed positives; otherwise the stability
        # term would silently turn an unlabeled field into a target.
        if bool(positive.any()):
            replay = replay + F.kl_div(
                F.log_softmax(chemical[known][positive].float(), dim=-1),
                F.softmax(reference_chemical_logits[known][positive].detach().float(), dim=-1),
                reduction="batchmean",
            )
        base = base + float(replay_weight) * replay

    safety = base.new_zeros(())
    safety_details = {"rings": 0, "reachable_aromatic_edges": 0}
    if aromatic_safety_weight > 0.0:
        if pair_indices is None or sample_indices is None:
            raise ValueError("Pair/sample indices are required for aromatic safety")
        safety, safety_details = aromatic_ring_safety_loss(
            chemical, pair_indices, sample_indices, known,
            presence_targets, chemical_targets, config=aromatic_safety_config,
        )
        base = base + float(aromatic_safety_weight) * safety
    anchor = base.new_zeros(())
    if aromatic_anchor_weight > 0.0:
        if (reference_chemical_logits is None
                or reference_chemical_logits.shape != chemical.shape
                or reference_chemical_logits.device != chemical.device):
            raise ValueError("Initial chemical logits are required for aromatic anchor")
        aromatic_id = BOND_TYPE_TO_ID["AROMATIC"]
        positive_rows = known & presence_targets.bool()
        current_aromatic = chemical[positive_rows].float().softmax(-1)[:, aromatic_id]
        reference_aromatic = torch.softmax(
            reference_chemical_logits[positive_rows].detach().float(), dim=-1,
        )[:, aromatic_id]
        aromatic_targets = chemical_targets[positive_rows] == aromatic_id
        if bool(aromatic_targets.any()):
            anchor = F.relu(
                reference_aromatic[aromatic_targets]
                - current_aromatic[aromatic_targets]
                - float(aromatic_anchor_epsilon)
            ).square().mean()
            base = base + float(aromatic_anchor_weight) * anchor
    if not bool(torch.isfinite(base)):
        raise ValueError("Joint edge loss is non-finite")
    return base, {
        "known_rows": int(known.sum()),
        "positive_rows": int(positive.sum()),
        "negative_rows": int((~positive).sum()),
        "presence_loss": float(presence_loss.detach()),
        "chemical_loss": None if chemical_loss is None else float(chemical_loss.detach()),
        "replay_loss": float(replay.detach()) if replay_weight > 0.0 else 0.0,
        "aromatic_safety_loss": float(safety.detach()) if aromatic_safety_weight > 0.0 else 0.0,
        "aromatic_anchor_loss": float(anchor.detach()) if aromatic_anchor_weight > 0.0 else 0.0,
        "presence_only": chemical_loss is None,
        **safety_details,
    }


__all__ = ["joint_edge_loss"]
