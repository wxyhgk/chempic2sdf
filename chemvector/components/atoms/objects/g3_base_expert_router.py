"""Conservative fixed-threshold routing between G3 Atom base and expert outputs."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_unrestricted_fixed_cardinality_assignment,
)


G3_BASE_EXPERT_THRESHOLD = -1.8862037658691406


def g3_should_route_expert(base_joint_score_mean: float) -> bool:
    """Apply the frozen P3-E7 zero-break route rule."""

    return float(base_joint_score_mean) <= G3_BASE_EXPERT_THRESHOLD


def _edge_scores(output: dict[str, torch.Tensor]) -> torch.Tensor:
    existence = output["existence_logits"].float()
    pointer = output["pointer_logits"].float()
    if pointer.ndim != 3 or pointer.shape[:2] != existence.shape:
        raise ValueError("G3 base/expert output dimensions differ")
    return F.logsigmoid(existence).unsqueeze(-1) + pointer.log_softmax(dim=-1)


@torch.no_grad()
def decode_g3_base_expert_joint(
    base_output: dict[str, torch.Tensor],
    expert_output: dict[str, torch.Tensor],
    candidate_anchors: torch.Tensor,
    *,
    counts: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor | float | bool | str]]:
    """Decode both models with one count and conservatively choose an Atom set."""

    base_scores = _edge_scores(base_output)
    expert_scores = _edge_scores(expert_output)
    if expert_scores.shape != base_scores.shape:
        raise ValueError("G3 base/expert score shapes differ")
    batch, queries, candidates = base_scores.shape
    if candidate_anchors.shape != (batch, candidates, 2):
        raise ValueError("G3 base/expert candidate anchors have invalid shape")
    if counts is None:
        count_logits = base_output.get("count_logits")
        if count_logits is None or count_logits.shape[0] != batch:
            raise ValueError("G3 base output lacks valid Count logits")
        counts = count_logits.argmax(dim=-1)
    if counts.shape != (batch,):
        raise ValueError("G3 base/expert counts must have shape [batch]")

    decoded = []
    for index in range(batch):
        count = int(counts[index])
        if not 0 <= count <= min(queries, candidates):
            raise ValueError("G3 base/expert count is outside matching capacity")
        if count == 0:
            empty = torch.empty(0, dtype=torch.long, device=base_scores.device)
            decoded.append({
                "anchors": candidate_anchors.new_zeros((0, 2)),
                "query_indices": empty,
                "candidate_indices": empty,
                "source": "base",
                "routed_to_expert": False,
                "base_joint_score_mean": 0.0,
                "count": 0,
            })
            continue
        base = g3_unrestricted_fixed_cardinality_assignment(base_scores[index], count)
        expert = g3_unrestricted_fixed_cardinality_assignment(
            expert_scores[index], count
        )
        base_mean = float(base["score"]) / count
        route = g3_should_route_expert(base_mean)
        selected = expert if route else base
        candidate_indices = selected["candidate_indices"]
        decoded.append({
            "anchors": candidate_anchors[index, candidate_indices],
            "query_indices": selected["query_indices"],
            "candidate_indices": candidate_indices,
            "source": "expert" if route else "base",
            "routed_to_expert": route,
            "base_joint_score_mean": base_mean,
            "count": count,
        })
    return decoded


__all__ = [
    "G3_BASE_EXPERT_THRESHOLD",
    "decode_g3_base_expert_joint",
    "g3_should_route_expert",
]
