"""Boundary-focused ranking loss for the frozen G3 top-128 Atom set."""

from __future__ import annotations

import torch
from torch.nn import functional as F


G3_BOUNDARY_BAND_SIZE = 8


def boundary_band_ranking_loss(
    ranked: dict[str, torch.Tensor],
    hybrid: dict[str, torch.Tensor],
    boundary_size: int = G3_BOUNDARY_BAND_SIZE,
) -> dict[str, torch.Tensor]:
    """Compare the weakest positive band with the strongest negative band."""

    if (
        isinstance(boundary_size, bool)
        or not isinstance(boundary_size, int)
        or boundary_size < 1
    ):
        raise ValueError("G3 boundary_size must be a positive integer")
    logits = ranked.get("ranking_logits")
    positive = hybrid.get("positive_mask")
    if logits is None or positive is None or logits.shape != positive.shape:
        raise ValueError("G3 boundary logits and hybrid targets differ")
    if logits.ndim != 2 or positive.dtype != torch.bool:
        raise ValueError("G3 boundary inputs must have [batch,candidates] shape")

    losses = []
    positive_count = negative_count = 0
    weak_positive_count = hard_negative_count = 0
    boundary_exact_images = 0
    for batch_index in range(len(logits)):
        mask = positive[batch_index]
        scores = logits[batch_index].float()
        positive_scores = scores[mask]
        negative_scores = scores[~mask]
        if not len(positive_scores):
            raise ValueError("G3 boundary loss requires a representable positive")
        if not len(negative_scores):
            raise ValueError("G3 boundary loss requires a negative candidate")
        weak_count = min(boundary_size, len(positive_scores))
        hard_count = min(boundary_size, len(negative_scores))
        weakest = positive_scores.topk(weak_count, largest=False).values
        strongest = negative_scores.topk(hard_count).values
        losses.append(
            F.softplus(strongest[:, None] - weakest[None, :]).mean()
        )
        positive_count += len(positive_scores)
        negative_count += len(negative_scores)
        weak_positive_count += weak_count
        hard_negative_count += hard_count
        boundary_exact_images += int(
            bool(positive_scores.min() > negative_scores.max())
        )

    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "positive_count": loss.new_tensor(float(positive_count)),
        "negative_count": loss.new_tensor(float(negative_count)),
        "weak_positive_count": loss.new_tensor(float(weak_positive_count)),
        "hard_negative_count": loss.new_tensor(float(hard_negative_count)),
        "boundary_exact_images": loss.new_tensor(float(boundary_exact_images)),
        "images": loss.new_tensor(float(len(logits))),
    }
