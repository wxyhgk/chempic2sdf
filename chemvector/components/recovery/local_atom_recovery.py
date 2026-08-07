"""Small local head that adds split/refine candidates without changing the base set."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


BASE_ACTION = 0
SPLIT_ACTION = 1
REFINE_ACTION = 2


@dataclass(frozen=True, slots=True)
class LocalAtomRecoveryConfig:
    """Fixed P4-E9 input, geometry, and candidate-budget protocol."""

    candidate_feature_dim: int = 32
    crop_size: int = 32
    maximum_base_candidates: int = 128
    maximum_split_candidates: int = 16
    maximum_refine_candidates: int = 16
    maximum_recovery_candidates: int = 32
    image_size: int = 768
    split_radius_pixels: float = 8.0
    refine_radius_pixels: float = 12.0
    action_bias: float = -6.0
    action_threshold: float = 0.0
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        fixed = {
            "candidate_feature_dim": (self.candidate_feature_dim, 32),
            "crop_size": (self.crop_size, 32),
            "maximum_base_candidates": (self.maximum_base_candidates, 128),
            "maximum_split_candidates": (self.maximum_split_candidates, 16),
            "maximum_refine_candidates": (self.maximum_refine_candidates, 16),
            "maximum_recovery_candidates": (self.maximum_recovery_candidates, 32),
            "image_size": (self.image_size, 768),
        }
        for name, (actual, expected) in fixed.items():
            if actual != expected:
                raise ValueError(f"P4-E9 {name} must be {expected}")
        if (
            self.maximum_split_candidates + self.maximum_refine_candidates
            != self.maximum_recovery_candidates
        ):
            raise ValueError("P4-E9 route capacities must sum to the recovery capacity")
        if not math.isclose(self.split_radius_pixels, 8.0):
            raise ValueError("P4-E9 split radius must be 8 pixels")
        if not math.isclose(self.refine_radius_pixels, 12.0):
            raise ValueError("P4-E9 refine radius must be 12 pixels")
        if not math.isclose(self.action_bias, -6.0):
            raise ValueError("P4-E9 action bias must be -6")
        if self.hidden_dim < 8:
            raise ValueError("P4-E9 hidden dimension must be at least 8")
        if not math.isfinite(self.action_threshold):
            raise ValueError("P4-E9 action threshold must be finite")

    @property
    def source_id_capacity(self) -> int:
        return self.image_size * self.image_size

    @property
    def output_capacity(self) -> int:
        return self.maximum_base_candidates + self.maximum_recovery_candidates


class LocalAtomRecoveryHead(nn.Module):
    """Score bounded split/refine actions from local observable evidence.

    Image and score crops are supplied separately so the data path can audit that
    neither a rendered label layer nor any ground-truth graph enters this head.
    Offsets use normalized canvas coordinates in the returned dictionary.
    """

    def __init__(
        self, config: LocalAtomRecoveryConfig = LocalAtomRecoveryConfig()
    ) -> None:
        super().__init__()
        self.config = config
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.candidate_norm = nn.LayerNorm(config.candidate_feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(32 + config.candidate_feature_dim + 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.split_action = nn.Linear(config.hidden_dim, 1)
        self.refine_action = nn.Linear(config.hidden_dim, 1)
        self.split_offset = nn.Linear(config.hidden_dim, 2)
        self.refine_offset = nn.Linear(config.hidden_dim, 2)
        self._initialize_output_heads()
        if self.parameter_count >= 100_000:
            raise ValueError("P4-E9 local recovery head must have fewer than 100k parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _initialize_output_heads(self) -> None:
        for head in (self.split_action, self.refine_action):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, self.config.action_bias)
        for head in (self.split_offset, self.refine_offset):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _validate_inputs(
        self,
        image_crops: torch.Tensor,
        score_crops: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_ranks: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[int, int]:
        batch = candidate_features.size(0) if candidate_features.ndim else -1
        candidates = self.config.maximum_base_candidates
        crop_shape = (
            batch,
            candidates,
            1,
            self.config.crop_size,
            self.config.crop_size,
        )
        if image_crops.shape != crop_shape or score_crops.shape != crop_shape:
            raise ValueError(
                "P4-E9 image and score crops must have shape [batch,128,1,32,32]"
            )
        if candidate_features.shape != (
            batch,
            candidates,
            self.config.candidate_feature_dim,
        ):
            raise ValueError("P4-E9 candidate features must have shape [batch,128,32]")
        if candidate_scores.shape != (batch, candidates):
            raise ValueError("P4-E9 candidate scores must have shape [batch,128]")
        if candidate_ranks.shape != (batch, candidates):
            raise ValueError("P4-E9 candidate ranks must have shape [batch,128]")
        if candidate_mask.shape != (batch, candidates) or candidate_mask.dtype != torch.bool:
            raise ValueError("P4-E9 candidate mask must be a boolean [batch,128] tensor")
        tensors = (
            image_crops,
            score_crops,
            candidate_features,
            candidate_scores,
            candidate_ranks,
        )
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("P4-E9 observable inputs must be floating-point tensors")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("P4-E9 observable inputs must be finite")
        if bool(((candidate_ranks < 0.0) | (candidate_ranks > 1.0)).any()):
            raise ValueError("P4-E9 candidate ranks must be in [0,1]")
        devices = {value.device for value in (*tensors, candidate_mask)}
        if len(devices) != 1:
            raise ValueError("P4-E9 inputs must share one device")
        return batch, candidates

    def forward(
        self,
        image_crops: torch.Tensor,
        score_crops: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_ranks: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, candidates = self._validate_inputs(
            image_crops,
            score_crops,
            candidate_features,
            candidate_scores,
            candidate_ranks,
            candidate_mask,
        )
        crops = torch.cat((image_crops, score_crops), dim=2).reshape(
            batch * candidates, 2, self.config.crop_size, self.config.crop_size
        )
        crop_hidden = self.crop_encoder(crops.float()).reshape(batch, candidates, -1)
        candidate_hidden = self.candidate_norm(candidate_features.float())
        scalar = torch.stack(
            (
                candidate_scores.float().clamp(-12.0, 12.0) / 12.0,
                candidate_ranks.float(),
            ),
            dim=-1,
        )
        hidden = self.fusion(
            torch.cat((crop_hidden, candidate_hidden, scalar), dim=-1)
        )
        split_logits = self.split_action(hidden).squeeze(-1)
        refine_logits = self.refine_action(hidden).squeeze(-1)
        split_logits = split_logits.masked_fill(~candidate_mask, -30.0)
        refine_logits = refine_logits.masked_fill(~candidate_mask, -30.0)
        split_offsets = self.split_offset(hidden)
        split_offsets = (
            split_offsets.tanh()
            * self.config.split_radius_pixels
            / self.config.image_size
        )
        refine_offsets = (
            self.refine_offset(hidden).tanh()
            * self.config.refine_radius_pixels
            / self.config.image_size
        )
        return {
            "split_logits": split_logits,
            "refine_logits": refine_logits,
            "split_offsets": split_offsets,
            "refine_offsets": refine_offsets,
            "candidate_mask": candidate_mask,
        }


def _validate_decode_inputs(
    output: dict[str, torch.Tensor],
    base_anchors: torch.Tensor,
    base_source_ids: torch.Tensor,
    base_mask: torch.Tensor,
    config: LocalAtomRecoveryConfig,
) -> tuple[int, torch.device]:
    batch = base_anchors.size(0) if base_anchors.ndim else -1
    candidates = config.maximum_base_candidates
    if base_anchors.shape != (batch, candidates, 2):
        raise ValueError("P4-E9 base anchors must have shape [batch,128,2]")
    if base_source_ids.shape != (batch, candidates) or base_source_ids.dtype != torch.long:
        raise ValueError("P4-E9 source IDs must be a long [batch,128] tensor")
    if base_mask.shape != (batch, candidates) or base_mask.dtype != torch.bool:
        raise ValueError("P4-E9 base mask must be a boolean [batch,128] tensor")
    expected = {
        "split_logits": (batch, candidates),
        "refine_logits": (batch, candidates),
        "split_offsets": (batch, candidates, 2),
        "refine_offsets": (batch, candidates, 2),
    }
    for name, shape in expected.items():
        value = output.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"P4-E9 {name} has invalid shape")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"P4-E9 {name} must be finite floating point")
    if not base_anchors.is_floating_point() or not bool(torch.isfinite(base_anchors).all()):
        raise ValueError("P4-E9 base anchors must be finite floating point")
    devices = {
        base_anchors.device,
        base_source_ids.device,
        base_mask.device,
        *(output[name].device for name in expected),
    }
    if len(devices) != 1:
        raise ValueError("P4-E9 decode tensors must share one device")
    valid_ids = base_source_ids[base_mask]
    if bool(((valid_ids < 0) | (valid_ids >= config.source_id_capacity)).any()):
        raise ValueError("P4-E9 source IDs are outside the stride-1 field")
    for batch_index in range(batch):
        current = base_source_ids[batch_index, base_mask[batch_index]]
        if len(torch.unique(current)) != len(current):
            raise ValueError("P4-E9 valid base source IDs must be unique")
    return batch, base_anchors.device


def _stable_action_indices(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    maximum: int,
) -> torch.Tensor:
    eligible = mask & (logits >= threshold)
    indices = torch.where(eligible)[0]
    if not len(indices):
        return indices
    order = torch.argsort(logits[indices], descending=True, stable=True)
    return indices[order[:maximum]]


@torch.no_grad()
def decode_local_atom_recovery(
    output: dict[str, torch.Tensor],
    base_anchors: torch.Tensor,
    base_source_ids: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    config: LocalAtomRecoveryConfig = LocalAtomRecoveryConfig(),
    action_threshold: float | None = None,
) -> dict[str, torch.Tensor]:
    """Append bounded recovery candidates while preserving all base slots exactly.

    New IDs reserve two complete stride-1 source-ID namespaces:
    ``N + source_id`` is the split secondary and ``2N + source_id`` is the
    refine secondary, where ``N = 768 * 768``.
    """

    batch, device = _validate_decode_inputs(
        output, base_anchors, base_source_ids, base_mask, config
    )
    threshold = config.action_threshold if action_threshold is None else action_threshold
    if not math.isfinite(float(threshold)):
        raise ValueError("P4-E9 decode threshold must be finite")
    capacity = config.output_capacity
    anchors = base_anchors.new_zeros((batch, capacity, 2))
    source_ids = torch.zeros((batch, capacity), dtype=torch.long, device=device)
    mask = torch.zeros((batch, capacity), dtype=torch.bool, device=device)
    actions = torch.full(
        (batch, capacity), -1, dtype=torch.int8, device=device
    )
    parent_indices = torch.full(
        (batch, capacity), -1, dtype=torch.long, device=device
    )
    action_logits = base_anchors.new_full((batch, capacity), -30.0)
    base = config.maximum_base_candidates
    anchors[:, :base] = base_anchors
    source_ids[:, :base] = base_source_ids
    mask[:, :base] = base_mask
    actions[:, :base] = torch.where(
        base_mask,
        torch.tensor(BASE_ACTION, dtype=torch.int8, device=device),
        torch.tensor(-1, dtype=torch.int8, device=device),
    )
    parent_indices[:, :base] = torch.where(
        base_mask,
        torch.arange(base, device=device)[None],
        torch.full((batch, base), -1, device=device, dtype=torch.long),
    )

    namespace = config.source_id_capacity
    split_counts = torch.zeros(batch, dtype=torch.long, device=device)
    refine_counts = torch.zeros(batch, dtype=torch.long, device=device)
    for batch_index in range(batch):
        cursor = base
        split_wins = output["split_logits"][batch_index] >= output["refine_logits"][batch_index]
        split_mask = base_mask[batch_index] & split_wins
        refine_mask = base_mask[batch_index] & ~split_wins
        split_indices = _stable_action_indices(
            output["split_logits"][batch_index],
            split_mask,
            threshold=float(threshold),
            maximum=config.maximum_split_candidates,
        )
        for parent in split_indices.tolist():
            parent_id = base_source_ids[batch_index, parent]
            anchors[batch_index, cursor] = (
                base_anchors[batch_index, parent]
                + output["split_offsets"][batch_index, parent]
            ).clamp(0.0, 1.0)
            source_ids[batch_index, cursor] = namespace + parent_id
            mask[batch_index, cursor] = True
            actions[batch_index, cursor] = SPLIT_ACTION
            parent_indices[batch_index, cursor] = parent
            action_logits[batch_index, cursor] = output["split_logits"][
                batch_index, parent
            ]
            cursor += 1
        split_counts[batch_index] = cursor - base

        refine_indices = _stable_action_indices(
            output["refine_logits"][batch_index],
            refine_mask,
            threshold=float(threshold),
            maximum=config.maximum_refine_candidates,
        )
        for parent in refine_indices.tolist():
            parent_id = base_source_ids[batch_index, parent]
            anchors[batch_index, cursor] = (
                base_anchors[batch_index, parent]
                + output["refine_offsets"][batch_index, parent]
            ).clamp(0.0, 1.0)
            source_ids[batch_index, cursor] = 2 * namespace + parent_id
            mask[batch_index, cursor] = True
            actions[batch_index, cursor] = REFINE_ACTION
            parent_indices[batch_index, cursor] = parent
            action_logits[batch_index, cursor] = output["refine_logits"][
                batch_index, parent
            ]
            cursor += 1
        refine_counts[batch_index] = cursor - base - split_counts[batch_index]
        if cursor > capacity:
            raise RuntimeError("P4-E9 recovery decoder exceeded its fixed capacity")

    return {
        "anchors": anchors,
        "source_ids": source_ids,
        "mask": mask,
        "actions": actions,
        "parent_indices": parent_indices,
        "action_logits": action_logits,
        "split_candidate_counts": split_counts,
        "refine_candidate_counts": refine_counts,
    }


__all__ = [
    "BASE_ACTION",
    "REFINE_ACTION",
    "SPLIT_ACTION",
    "LocalAtomRecoveryConfig",
    "LocalAtomRecoveryHead",
    "decode_local_atom_recovery",
]
