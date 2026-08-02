"""Dynamic one-to-one supervision for equivalent G3 Atom candidates."""

from __future__ import annotations

import math

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.atoms.g3_top128_set_ranker import (
    G3_TOP128_CANDIDATES,
)


G3_EQUIVALENT_RADIUS_PIXELS = 6.0


def _stable_score_ranks(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Return zero-based ranks with candidate identity breaking exact score ties."""

    identity_order = candidate_ids.argsort(stable=True)
    identity_scores = scores[identity_order]
    score_order = identity_scores.argsort(descending=True, stable=True)
    order = identity_order[score_order]
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), dtype=order.dtype)
    return ranks


@torch.no_grad()
def build_score_matched_equivalent_targets(
    ranked: dict[str, torch.Tensor],
    hybrid: dict[str, torch.Tensor],
    target_anchors: torch.Tensor,
    *,
    radius_pixels: float = G3_EQUIVALENT_RADIUS_PIXELS,
    image_size: int = 768,
) -> dict[str, torch.Tensor]:
    """Choose one score-preferred 6px-equivalent candidate for every fixed GT.

    ``hybrid`` fixes which GT atoms are representable. Candidate identity inside that
    set is reassigned by a maximum-cardinality bipartite match. Score ranks only break
    ties among full-cardinality assignments, so a high score can never drop a GT.
    """

    logits = ranked.get("ranking_logits")
    candidate_ids = ranked.get("candidate_ids")
    candidate_anchors = ranked.get("candidate_anchors")
    fixed_positive = hybrid.get("positive_mask")
    fixed_target_indices = hybrid.get("matched_target_indices")
    if (
        logits is None
        or candidate_ids is None
        or candidate_anchors is None
        or fixed_positive is None
        or fixed_target_indices is None
        or logits.shape != candidate_ids.shape
        or logits.shape != fixed_positive.shape
        or logits.shape != fixed_target_indices.shape
        or logits.ndim != 2
        or logits.size(1) != G3_TOP128_CANDIDATES
        or candidate_anchors.shape != (*logits.shape, 2)
    ):
        raise ValueError("G3 equivalent matcher inputs have incompatible shapes")
    if candidate_ids.dtype != torch.long or fixed_positive.dtype != torch.bool:
        raise ValueError("G3 equivalent matcher identities or mask have invalid dtype")
    if (
        target_anchors.ndim != 3
        or target_anchors.shape[0] != logits.shape[0]
        or target_anchors.shape[-1] != 2
    ):
        raise ValueError("G3 equivalent matcher target anchors have invalid shape")
    devices = {
        logits.device,
        candidate_ids.device,
        candidate_anchors.device,
        fixed_positive.device,
        fixed_target_indices.device,
        target_anchors.device,
    }
    if len(devices) != 1:
        raise ValueError("G3 equivalent matcher tensors must share one device")
    if not math.isfinite(float(radius_pixels)) or radius_pixels <= 0:
        raise ValueError("G3 equivalent radius must be positive and finite")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("G3 equivalent image size must be a positive integer")

    output_device = logits.device
    scores_cpu = logits.detach().float().cpu()
    ids_cpu = candidate_ids.detach().cpu()
    anchors_cpu = candidate_anchors.detach().double().cpu()
    fixed_positive_cpu = fixed_positive.detach().cpu()
    fixed_indices_cpu = fixed_target_indices.detach().cpu()
    targets_cpu = target_anchors.detach().double().cpu()
    radius = float(radius_pixels) / float(image_size)
    dynamic_positive = torch.zeros_like(fixed_positive_cpu)
    dynamic_indices = torch.full_like(fixed_indices_cpu, -1)
    reference_counts = fixed_positive_cpu.sum(dim=1).long()
    identity_changes = torch.zeros(len(logits), dtype=torch.long)
    alternative_candidates = torch.zeros(len(logits), dtype=torch.long)

    for batch_index in range(len(logits)):
        fixed_mask = fixed_positive_cpu[batch_index]
        target_indices = fixed_indices_cpu[batch_index, fixed_mask]
        if len(target_indices) < 1 or bool((target_indices < 0).any()):
            raise ValueError("G3 equivalent matcher requires representable fixed positives")
        if len(target_indices.unique()) != len(target_indices):
            raise ValueError("G3 fixed positives do not map one-to-one to GT atoms")
        target_indices = target_indices.sort().values
        current_targets = targets_cpu[batch_index, target_indices]
        compatible = (
            torch.cdist(anchors_cpu[batch_index], current_targets) <= radius
        )
        if bool((compatible.sum(dim=0) < 1).any()):
            raise RuntimeError("G3 fixed representable target lost all equivalent candidates")
        alternative_candidates[batch_index] = int(
            (compatible.sum(dim=0) - 1).sum()
        )

        ranks = _stable_score_ranks(
            scores_cpu[batch_index], ids_cpu[batch_index]
        ).double()
        target_count = len(target_indices)
        cardinality_penalty = float(target_count + 1)
        real_cost = ranks[:, None].expand(-1, target_count).clone()
        real_cost /= float(G3_TOP128_CANDIDATES)
        real_cost[~compatible] = 2.0 * cardinality_penalty
        dummy_cost = torch.full(
            (target_count, target_count), cardinality_penalty, dtype=torch.float64
        )
        cost = torch.cat((real_cost.T, dummy_cost), dim=1)
        rows, columns = linear_sum_assignment(cost.numpy())
        real_assignments = [
            (row, column)
            for row, column in zip(rows.tolist(), columns.tolist())
            if column < G3_TOP128_CANDIDATES and bool(compatible[column, row])
        ]
        if len(real_assignments) != target_count:
            raise RuntimeError("G3 equivalent matcher reduced fixed target cardinality")
        for row, column in real_assignments:
            dynamic_positive[batch_index, column] = True
            dynamic_indices[batch_index, column] = target_indices[row]
        identity_changes[batch_index] = int(
            (dynamic_positive[batch_index] ^ fixed_mask).sum() // 2
        )

    return {
        "positive_mask": dynamic_positive.to(output_device),
        "matched_target_indices": dynamic_indices.to(output_device),
        "representable_counts": reference_counts.to(output_device),
        "fixed_positive_mask": fixed_positive.detach().clone(),
        "identity_changes": identity_changes.to(output_device),
        "alternative_candidates": alternative_candidates.to(output_device),
    }
