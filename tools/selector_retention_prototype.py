"""Isolated candidate loss contract, NOT connected to any training launcher.

One proposed additive guard: per-image positive-only mean softplus(tau - z).
The signed-threshold softplus already exists in graph-exact surrogate; this
prototype isolates positive retention instead of another whole-graph max.
Proposed coefficient is fixed at 1 (no search), original full loss unchanged.
Real-model efficacy and safe finite updates have NOT been established.
"""

import math

import torch
from torch.nn import functional as F


def positive_retention_loss(logits, positive, mask, *, threshold):
    """Keep correct candidates above the EXISTING threshold, without count dilution.

    A negative candidate has zero DIRECT derivative here. Shared parameters can
    still raise its score: original negative suppression and old-image replay
    remain necessary. This function cannot certify a safe update direction.
    Empty-positive images contribute zero; padded candidates never contribute.
    """
    if (logits.ndim != 2 or logits.shape != positive.shape or logits.shape != mask.shape
            or not logits.is_floating_point() or positive.dtype != torch.bool or mask.dtype != torch.bool
            or logits.device != positive.device or logits.device != mask.device
            or logits.shape[0] == 0 or not math.isfinite(threshold)
            or bool((positive & ~mask).any()) or not bool(torch.isfinite(logits).all())):
        raise ValueError("Invalid positive retention inputs")
    values = F.softplus(threshold - logits.float())
    counts = positive.sum(-1).clamp_min(1)
    return ((values * positive).sum(-1) / counts).mean()


def proposed_objective(original_full_loss, logits, positive, mask, *, threshold):
    """Single fixed proposal; no replacement, tuning argument or optimizer."""
    if original_full_loss.ndim != 0 or not bool(torch.isfinite(original_full_loss)):
        raise ValueError("Expected finite scalar original full loss")
    return original_full_loss + positive_retention_loss(logits, positive, mask, threshold=threshold)
