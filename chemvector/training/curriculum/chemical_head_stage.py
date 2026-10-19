"""Non-authorizing chemical-output-only stage for a bounded pair probe."""

import torch
from torch import nn
from torch.nn import functional as F

from .aromatic_ring_objective import (
    AromaticRingObjectiveConfig,
    aromatic_ring_safety_loss,
)


class ChemicalHeadOnlyStage(nn.Module):
    """Keep pair evidence and presence fixed while fitting bond semantics."""

    def __init__(self, pair: nn.Module):
        super().__init__()
        if any(parameter.grad is not None for parameter in pair.parameters()):
            raise ValueError("A fresh pair module without gradients is required")
        if not isinstance(getattr(pair, "chemical", None), nn.Linear):
            raise ValueError("Pair module must expose a linear chemical output head")
        self.pair = pair
        self.pair.requires_grad_(False)
        self.pair.chemical.requires_grad_(True)
        self.eval()
        self.trainable_parameters()

    def train(self, mode: bool = True):
        if mode:
            raise ValueError("Chemical-head-only stage must remain in eval mode")
        return super().train(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        expected = {id(parameter) for parameter in self.pair.chemical.parameters()}
        actual = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if actual != expected or any(module.training for module in self.pair.modules()):
            raise ValueError("Only the eval-mode chemical output head may be trainable")
        return list(self.pair.chemical.parameters())

    def forward(self, *args, **kwargs):
        self.trainable_parameters()
        return self.pair(*args, **kwargs)


def chemical_head_loss(
    output: dict[str, torch.Tensor],
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
) -> torch.Tensor | None:
    """Return positive-pair CE only; unknown rows are an error, never negatives."""
    presence = output.get("presence_logits")
    chemical = output.get("chemical_logits")
    if (not isinstance(presence, torch.Tensor) or presence.ndim != 1
            or not isinstance(chemical, torch.Tensor) or chemical.ndim != 2
            or chemical.shape[0] != len(presence)):
        raise ValueError("Pair output has invalid presence/chemical shapes")
    rows = len(presence)
    if (known_mask.shape != (rows,) or known_mask.dtype != torch.bool
            or presence_targets.shape != (rows,) or chemical_targets.shape != (rows,)
            or presence_targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)
            or chemical_targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError("Known mask and pair targets must have matching row shapes")
    if not bool(known_mask.all()):
        raise ValueError("Unknown pair rows must be filtered before chemical supervision")
    if (not bool(((presence_targets == 0) | (presence_targets == 1)).all())
            or not bool(((chemical_targets >= 0) & (chemical_targets < chemical.shape[1])).all())):
        raise ValueError("Pair targets are out of range")
    positive = presence_targets.bool()
    if not bool(positive.any()):
        return None
    return F.cross_entropy(chemical[positive].float(), chemical_targets[positive].long())


def chemical_head_loss_with_aromatic_safety(
    output: dict[str, torch.Tensor],
    known_mask: torch.Tensor,
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    safety_weight: float = 0.0,
    safety_config: AromaticRingObjectiveConfig | None = None,
    reference_chemical_logits: torch.Tensor | None = None,
    trust_region_weight: float = 0.0,
) -> tuple[torch.Tensor | None, dict]:
    """Optionally add a target-ring safety term without changing old callers.

    ``safety_weight=0`` preserves the historical chemical-head objective and
    is the default.  Pair/sample indices are required even when the weight is
    zero so a future authorized probe cannot silently omit graph context.
    """
    if not isinstance(safety_weight, (int, float)) or safety_weight < 0.0:
        raise ValueError("Aromatic safety weight must be non-negative")
    if not isinstance(trust_region_weight, (int, float)) or trust_region_weight < 0.0:
        raise ValueError("Chemical trust-region weight must be non-negative")
    if reference_chemical_logits is not None and (
        not isinstance(reference_chemical_logits, torch.Tensor)
        or reference_chemical_logits.shape != output["chemical_logits"].shape
        or not bool(torch.isfinite(reference_chemical_logits).all())
    ):
        raise ValueError("Reference chemical logits must match output and be finite")
    if trust_region_weight > 0.0 and reference_chemical_logits is None:
        raise ValueError("Reference chemical logits are required for a trust region")
    base = chemical_head_loss(
        output, known_mask, presence_targets, chemical_targets,
    )
    if base is None:
        return None, {"chemical_loss": None, "aromatic_safety_loss": None, "trust_region_loss": None, "rings": 0}
    trust = base.new_zeros(())
    if trust_region_weight > 0.0:
        # Distillation in probability space keeps the initial decision surface
        # local without freezing the head.  It applies to both positive and
        # sampled-negative rows, unlike the historical CE term.
        trust = F.kl_div(
            F.log_softmax(output["chemical_logits"].float(), dim=-1),
            F.softmax(reference_chemical_logits.detach().float(), dim=-1),
            reduction="batchmean",
        )
    if safety_weight == 0.0:
        total = base + float(trust_region_weight) * trust
        return total, {
            "chemical_loss": float(base.detach()),
            "aromatic_safety_loss": 0.0,
            "trust_region_loss": float(trust.detach()),
            "trust_region_weight": float(trust_region_weight),
            "rings": 0,
            "safety_weight": 0.0,
        }
    ring_loss, details = aromatic_ring_safety_loss(
        output["chemical_logits"], pair_indices, sample_indices, known_mask,
        presence_targets, chemical_targets, config=safety_config,
    )
    total = base + float(safety_weight) * ring_loss + float(trust_region_weight) * trust
    return total, {
        "chemical_loss": float(base.detach()),
        "aromatic_safety_loss": float(ring_loss.detach()),
        "trust_region_loss": float(trust.detach()),
        "trust_region_weight": float(trust_region_weight),
        "safety_weight": float(safety_weight),
        **details,
    }
