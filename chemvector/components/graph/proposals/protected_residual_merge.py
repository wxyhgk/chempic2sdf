"""Ownership-safe merge for a frozen proposal selector and a residual sidecar.

The baseline selector is deliberately opaque to the residual path.  Residual
rows are never appended to the tensor consumed by the baseline set head; they
are filtered independently and merged only as graph candidates afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """A target-independent candidate list with stable provenance."""

    points: torch.Tensor
    scores: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.size(-1) != 2:
            raise ValueError("candidate points must have shape [N, 2]")
        if self.scores.ndim != 1 or self.scores.size(0) != self.points.size(0):
            raise ValueError("candidate scores must have shape [N]")
        if self.source not in {"baseline", "residual"}:
            raise ValueError("candidate source must be baseline or residual")


def _validate_same_device(*sets: CandidateSet) -> None:
    devices = {candidate.points.device for candidate in sets}
    devices.update(candidate.scores.device for candidate in sets)
    if len(devices) > 1:
        raise ValueError("candidate tensors must share one device")


def select_residual_candidates(
    points: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold: float = 0.5,
    maximum: int | None = None,
    suppression_radius: float = 0.0,
) -> CandidateSet:
    """Select residual rows without consulting baseline logits or targets."""

    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError("points must have shape [N, 2]")
    if scores.ndim != 1 or scores.size(0) != points.size(0):
        raise ValueError("scores must have shape [N]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if maximum is not None and maximum < 0:
        raise ValueError("maximum must be nonnegative")
    if suppression_radius < 0.0:
        raise ValueError("suppression_radius must be nonnegative")

    order = torch.argsort(scores, descending=True, stable=True)
    chosen: list[int] = []
    for raw_index in order.tolist():
        if float(scores[raw_index]) < threshold:
            break
        if suppression_radius and chosen:
            previous = points[torch.as_tensor(chosen, device=points.device)]
            if bool(torch.linalg.vector_norm(previous - points[raw_index], dim=-1).min() < suppression_radius):
                continue
        chosen.append(int(raw_index))
        if maximum is not None and len(chosen) >= maximum:
            break
    indices = torch.as_tensor(chosen, dtype=torch.long, device=points.device)
    return CandidateSet(points[indices], scores[indices], "residual")


def merge_candidates(
    baseline: CandidateSet,
    residual: CandidateSet,
    *,
    duplicate_radius: float = 0.0,
) -> CandidateSet:
    """Merge graph candidates while preserving baseline rows byte-for-byte.

    A residual row is dropped if it lies within ``duplicate_radius`` of a
    baseline row.  Baseline order and values are never sorted, normalized, or
    recomputed.  The returned source is ``baseline`` only for an empty residual
    branch; callers needing provenance should retain the two input sets.
    """

    if baseline.source != "baseline" or residual.source != "residual":
        raise ValueError("merge requires baseline then residual candidates")
    if duplicate_radius < 0.0:
        raise ValueError("duplicate_radius must be nonnegative")
    _validate_same_device(baseline, residual)
    keep: list[int] = []
    for index in range(residual.points.size(0)):
        if duplicate_radius and baseline.points.size(0):
            distance = torch.linalg.vector_norm(
                baseline.points - residual.points[index], dim=-1
            )
            if bool(distance.min() <= duplicate_radius):
                continue
        keep.append(index)
    indices = torch.as_tensor(keep, dtype=torch.long, device=residual.points.device)
    residual_points = residual.points[indices]
    residual_scores = residual.scores[indices]
    if not keep:
        return CandidateSet(baseline.points.clone(), baseline.scores.clone(), "baseline")
    return CandidateSet(
        torch.cat((baseline.points, residual_points), dim=0),
        torch.cat((baseline.scores, residual_scores), dim=0),
        "baseline",
    )


def baseline_prefix_unchanged(
    baseline_before: CandidateSet,
    merged: CandidateSet,
) -> bool:
    """Check the immutable prefix contract after an isolated merge."""

    if baseline_before.source != "baseline":
        raise ValueError("baseline_before must be a baseline candidate set")
    if merged.points.size(0) < baseline_before.points.size(0):
        return False
    count = baseline_before.points.size(0)
    return bool(
        torch.equal(merged.points[:count], baseline_before.points)
        and torch.equal(merged.scores[:count], baseline_before.scores)
    )


__all__ = [
    "CandidateSet",
    "baseline_prefix_unchanged",
    "merge_candidates",
    "select_residual_candidates",
]
