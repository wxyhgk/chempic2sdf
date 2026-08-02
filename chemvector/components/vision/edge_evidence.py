"""Chunked full-adjacency evidence from the semantic bond reader."""

from __future__ import annotations

from pathlib import Path

import torch

from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
)


def load_edge_teacher(
    checkpoint: str | Path, device: torch.device
) -> NodeConditionedBondObject:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher = NodeConditionedBondObject(
        NodeConditionedBondConfig(**payload["config"])
    ).to(device)
    teacher.load_state_dict(payload["node_conditioned_bond"])
    return teacher.eval().requires_grad_(False)


@torch.inference_mode()
def score_full_adjacency_edges(
    teacher: NodeConditionedBondObject,
    pixel_values: torch.Tensor,
    points: torch.Tensor,
    candidate_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Score every valid pair without dropping distant endpoints."""

    if chunk_size < 1:
        raise ValueError("Edge teacher chunk size must be positive")
    if pair_indices.ndim != 2 or pair_indices.size(-1) != 2:
        raise ValueError("Pair indices must have shape [pairs, 2]")
    if points.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate points and mask differ")
    scores = torch.full(
        (points.size(0), pair_indices.size(0)),
        -30.0,
        device=points.device,
        dtype=torch.float32,
    )
    for start in range(0, pair_indices.size(0), chunk_size):
        stop = min(start + chunk_size, pair_indices.size(0))
        current_pairs = pair_indices[start:stop]
        begin = current_pairs[:, 0]
        end = current_pairs[:, 1]
        current_mask = candidate_mask[:, begin] & candidate_mask[:, end]
        if not bool(current_mask.any()):
            continue
        batched_pairs = current_pairs[None].expand(points.size(0), -1, -1)
        with torch.autocast(device_type=points.device.type, dtype=torch.bfloat16):
            output = teacher(
                pixel_values,
                points,
                batched_pairs,
                current_mask,
            )
        current_scores = scores[:, start:stop]
        current_scores[current_mask] = output["edge_logits"].float()
    return scores
