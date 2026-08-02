"""Numerically stable geometry helpers shared by coordinate supervision paths."""

from __future__ import annotations

import torch


def stable_pairwise_distances(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Euclidean distances without GEMM/TF32 cancellation from torch.cdist."""

    if first.ndim != 2 or second.ndim != 2 or first.size(-1) != second.size(-1):
        raise ValueError("Pairwise distance inputs must be two compatible matrices")
    delta = first.float()[:, None, :] - second.float()[None, :, :]
    return delta.square().sum(dim=-1).sqrt()
