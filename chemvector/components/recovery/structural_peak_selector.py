"""Listwise structural selector over frozen P6 candidate peaks."""

from __future__ import annotations

import torch
from torch import nn


CONTEXT_SIZE = 96
IMAGE_SIZE = 768
CANDIDATES = 8
DIMENSION = 64


class _Block(nn.Module):
    def __init__(self, inputs: int, outputs: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = 8 if outputs % 8 == 0 else 4
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, outputs),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class StructuralPeakSelector(nn.Module):
    """Score peak candidates jointly from source-centered structural context."""

    def __init__(self, *, use_peak_logits: bool = True) -> None:
        super().__init__()
        self.use_peak_logits = bool(use_peak_logits)
        axis = torch.arange(CONTEXT_SIZE, dtype=torch.float32)
        axis = axis - (CONTEXT_SIZE - 1) * 0.5
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((x, y), dim=0) / ((CONTEXT_SIZE - 1) * 0.5)
        self.register_buffer("relative_coordinates", coordinates, persistent=False)
        self.encoder = nn.Sequential(
            _Block(4, 16),
            _Block(16, 24, stride=2),
            _Block(24, 40, stride=2),
            _Block(40, DIMENSION, stride=2),
            _Block(DIMENSION, DIMENSION),
        )
        self.candidate = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, DIMENSION),
            nn.GELU(),
            nn.Linear(DIMENSION, DIMENSION),
        )
        self.cross_attention = nn.MultiheadAttention(
            DIMENSION, 4, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(DIMENSION)
        self.set_attention = nn.MultiheadAttention(
            DIMENSION, 4, batch_first=True
        )
        self.set_norm = nn.LayerNorm(DIMENSION)
        self.output = nn.Sequential(
            nn.Linear(DIMENSION, DIMENSION),
            nn.GELU(),
            nn.Linear(DIMENSION, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(
        self,
        image_crops: torch.Tensor,
        score_crops: torch.Tensor,
        candidate_offsets: torch.Tensor,
        candidate_peak_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        source_scores: torch.Tensor,
        source_ranks: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(image_crops)
        expected_crop = (batch, 1, CONTEXT_SIZE, CONTEXT_SIZE)
        if image_crops.shape != expected_crop or score_crops.shape != expected_crop:
            raise ValueError("P6 selector crops must have shape [batch,1,96,96]")
        if candidate_offsets.shape != (batch, CANDIDATES, 2):
            raise ValueError("P6 selector offsets must have shape [batch,8,2]")
        if candidate_peak_logits.shape != (batch, CANDIDATES):
            raise ValueError("P6 selector peak logits must have shape [batch,8]")
        if candidate_mask.shape != (batch, CANDIDATES) or candidate_mask.dtype != torch.bool:
            raise ValueError("P6 selector mask must be boolean [batch,8]")
        if source_scores.shape != (batch,) or source_ranks.shape != (batch,):
            raise ValueError("P6 selector source scalars must be batch vectors")
        floating = (
            image_crops,
            score_crops,
            candidate_offsets,
            candidate_peak_logits,
            source_scores,
            source_ranks,
        )
        if any(not value.is_floating_point() for value in floating) or any(
            not bool(torch.isfinite(value).all()) for value in floating
        ):
            raise ValueError("P6 selector inputs must be finite floating point")
        if not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("P6 selector requires at least one candidate per sample")

        coordinates = self.relative_coordinates[None].expand(batch, -1, -1, -1)
        context = self.encoder(
            torch.cat(
                (image_crops.float(), score_crops.float(), coordinates), dim=1
            )
        )
        tokens = context.flatten(2).transpose(1, 2)
        valid_logits = candidate_peak_logits.float().masked_fill(~candidate_mask, 0.0)
        count = candidate_mask.sum(dim=1, keepdim=True).float()
        mean = valid_logits.sum(dim=1, keepdim=True) / count
        variance = (
            ((valid_logits - mean).square() * candidate_mask).sum(dim=1, keepdim=True)
            / count
        )
        normalized_logits = (valid_logits - mean) / variance.sqrt().clamp_min(1e-4)
        if not self.use_peak_logits:
            normalized_logits = torch.zeros_like(normalized_logits)
        offsets_pixels = candidate_offsets.float() * IMAGE_SIZE
        radius = torch.linalg.vector_norm(offsets_pixels, dim=-1, keepdim=True)
        scalars = torch.cat(
            (
                offsets_pixels / 16.0,
                radius / 16.0,
                normalized_logits[..., None],
                source_scores.float()[:, None, None].expand(-1, CANDIDATES, -1),
                source_ranks.float()[:, None, None].expand(-1, CANDIDATES, -1),
            ),
            dim=-1,
        )
        queries = self.candidate(scalars)
        attended, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        queries = self.cross_norm(queries + attended)
        related, _ = self.set_attention(
            queries,
            queries,
            queries,
            key_padding_mask=~candidate_mask,
            need_weights=False,
        )
        queries = self.set_norm(queries + related)
        return self.output(queries)[:, :, 0].masked_fill(~candidate_mask, -torch.inf)


def compatible_peak_selection_loss(
    scores: torch.Tensor,
    candidate_offsets: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_offsets: torch.Tensor,
    *,
    radius_pixels: float = 6.0,
) -> torch.Tensor:
    """Maximize total probability assigned to every evaluation-equivalent peak."""

    if scores.shape != candidate_mask.shape or scores.shape != candidate_offsets.shape[:2]:
        raise ValueError("P6 selector loss candidate shapes differ")
    if target_offsets.shape != (len(scores), 2):
        raise ValueError("P6 selector loss targets must have shape [batch,2]")
    distances = torch.linalg.vector_norm(
        (candidate_offsets.float() - target_offsets.float()[:, None]) * IMAGE_SIZE,
        dim=-1,
    )
    compatible = candidate_mask & (distances <= radius_pixels)
    if not bool(compatible.any(dim=1).all()):
        raise ValueError("P6 selector loss received an unreachable target")
    all_logsumexp = torch.logsumexp(scores.float(), dim=1)
    positive_logsumexp = torch.logsumexp(
        scores.float().masked_fill(~compatible, -torch.inf), dim=1
    )
    return (all_logsumexp - positive_logsumexp).mean()


__all__ = [
    "CANDIDATES",
    "StructuralPeakSelector",
    "compatible_peak_selection_loss",
]
