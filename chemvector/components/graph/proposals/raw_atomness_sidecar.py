"""Contract for an isolated raw-candidate atomness sidecar.

The sidecar is intentionally not a selector rescue path.  It may filter raw
proposals that are not in the frozen baseline selection, while the baseline
selection is an immutable prefix.  A low sidecar logit means *abstain*; it
does not authorize graph repair, endpoint assignment, or a new atom.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch


@dataclass(frozen=True, slots=True)
class RawAtomnessDecision:
    """Auditable masks for one raw proposal set."""

    keep_mask: torch.Tensor
    protected_mask: torch.Tensor
    sidecar_selected_mask: torch.Tensor
    abstained_mask: torch.Tensor

    def __post_init__(self) -> None:
        masks = (self.keep_mask, self.protected_mask,
                 self.sidecar_selected_mask, self.abstained_mask)
        if any(mask.ndim != 1 or mask.dtype != torch.bool for mask in masks):
            raise ValueError("all decision masks must be one-dimensional bool tensors")
        lengths = {mask.numel() for mask in masks}
        if len(lengths) != 1:
            raise ValueError("decision masks must have equal length")


def apply_raw_atomness_sidecar(
    sidecar_logits: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> RawAtomnessDecision:
    """Apply a fixed sidecar gate without changing frozen selections.

    ``protected_mask`` is the baseline-selected mask and is copied into the
    output unchanged.  Only non-protected raw candidates are scored by the
    sidecar.  Candidates below the fixed gate are abstentions, not confirmed
    negatives; callers must not feed them into a rescue loss or graph repair.
    """

    if sidecar_logits.ndim != 1 or not sidecar_logits.is_floating_point():
        raise ValueError("sidecar_logits must be a one-dimensional floating tensor")
    if protected_mask.ndim != 1 or protected_mask.dtype != torch.bool:
        raise ValueError("protected_mask must be a one-dimensional bool tensor")
    if sidecar_logits.numel() != protected_mask.numel():
        raise ValueError("sidecar_logits and protected_mask must have equal length")
    if not torch.isfinite(sidecar_logits).all():
        raise ValueError("sidecar_logits must be finite")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    sidecar_selected = (~protected_mask) & (sidecar_logits >= float(threshold))
    abstained = (~protected_mask) & ~sidecar_selected
    keep = protected_mask | sidecar_selected
    return RawAtomnessDecision(keep, protected_mask.clone(), sidecar_selected, abstained)


def audit_raw_atomness_decision(
    labels: Sequence[str], decision: RawAtomnessDecision,
) -> dict[str, float | int]:
    """Compute an offline audit; labels never participate in selection.

    This helper is intentionally named ``audit`` and accepts labels only for
    evaluation reports.  It must not be called from model inference.
    """

    if len(labels) != decision.keep_mask.numel():
        raise ValueError("labels and decision masks must have equal length")
    allowed = {"protected", "strict_positive", "hard_negative",
               "near_unknown", "selected_unmatched"}
    if any(label not in allowed for label in labels):
        raise ValueError("unknown audit label")
    selected = decision.keep_mask.cpu().numpy()
    positive = [label in {"protected", "strict_positive"} for label in labels]
    known = [label in {"protected", "strict_positive", "hard_negative"} for label in labels]
    hard = [label == "hard_negative" for label in labels]
    unknown = [not value for value in known]
    true_selected = sum(a and b for a, b in zip(selected, positive))
    return {
        "selected": int(selected.sum()),
        "true_selected": int(true_selected),
        "conservative_precision": true_selected / max(int(selected.sum()), 1),
        "raw_recall": true_selected / max(sum(positive), 1),
        "hard_negative_selected": int(sum(a and b for a, b in zip(selected, hard))),
        "unknown_selected": int(sum(a and b for a, b in zip(selected, unknown))),
        "protected_retained": int(sum(a and label == "protected" for a, label in zip(selected, labels))),
    }


__all__ = ["RawAtomnessDecision", "apply_raw_atomness_sidecar", "audit_raw_atomness_decision"]
