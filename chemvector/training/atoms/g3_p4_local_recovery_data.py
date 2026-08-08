"""Observable inputs, targets, sampling, and loss for P4 local recovery."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F


CANDIDATES = 128
CANDIDATE_FEATURE_DIM = 32
CROP_SIZE = 32
NEGATIVE_RATIO = 3
ROUTES = ("ADD_SPLIT", "ADD_REFINE")
ROUTE_RADIUS_PIXELS = {"ADD_SPLIT": 8.0, "ADD_REFINE": 12.0}
SAMPLER_SEED = 20260849


def _validate_observable_fields(
    pixel_values: torch.Tensor,
    field_output: Mapping[str, torch.Tensor],
    peaks: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[int, int, int, torch.device]:
    if (
        pixel_values.ndim != 4
        or pixel_values.size(1) not in (1, 3)
        or min(pixel_values.shape[-2:]) < 1
        or not pixel_values.is_floating_point()
        or not bool(torch.isfinite(pixel_values).all())
    ):
        raise ValueError("P4 recovery pixels must be finite [batch,1|3,height,width]")
    if bool(((pixel_values < -1.0001) | (pixel_values > 1.0001)).any()):
        raise ValueError("P4 recovery pixels must use the normalized [-1,1] protocol")
    center_logits = field_output.get("center_logits")
    stride1_features = field_output.get("stride1_features")
    batch, _, height, width = pixel_values.shape
    if (
        not isinstance(center_logits, torch.Tensor)
        or center_logits.shape != (batch, 1, height, width)
        or not center_logits.is_floating_point()
        or not bool(torch.isfinite(center_logits).all())
    ):
        raise ValueError("P4 recovery center logits must be finite [batch,1,H,W]")
    if (
        not isinstance(stride1_features, torch.Tensor)
        or stride1_features.shape
        != (batch, CANDIDATE_FEATURE_DIM, height, width)
        or not stride1_features.is_floating_point()
        or not bool(torch.isfinite(stride1_features).all())
    ):
        raise ValueError("P4 recovery stride1 features must be finite [batch,32,H,W]")
    if len(peaks) != batch:
        raise ValueError("P4 recovery peak batch differs from observable tensors")
    devices = {
        pixel_values.device,
        center_logits.device,
        stride1_features.device,
    }
    if len(devices) != 1:
        raise ValueError("P4 recovery observable tensors must share one device")
    return batch, height, width, pixel_values.device


def _grayscale_darkness(pixel_values: torch.Tensor) -> torch.Tensor:
    if pixel_values.size(1) == 1:
        grayscale = pixel_values.float()
    else:
        weights = pixel_values.new_tensor((0.299, 0.587, 0.114)).reshape(1, 3, 1, 1)
        grayscale = (pixel_values.float() * weights).sum(dim=1, keepdim=True)
    return ((1.0 - grayscale) * 0.5).clamp(0.0, 1.0)


def _anchor_grid(
    anchors: torch.Tensor,
    *,
    height: int,
    width: int,
    crop_size: int = CROP_SIZE,
) -> torch.Tensor:
    if anchors.ndim != 3 or anchors.size(-1) != 2:
        raise ValueError("P4 recovery anchors must have shape [batch,candidates,2]")
    if crop_size != CROP_SIZE:
        raise ValueError("P4 recovery crop size must be 32")
    if not anchors.is_floating_point() or not bool(torch.isfinite(anchors).all()):
        raise ValueError("P4 recovery anchors must be finite floating point")
    if bool(((anchors < 0.0) | (anchors > 1.0)).any()):
        raise ValueError("P4 recovery anchors must be normalized to [0,1]")
    half = (crop_size - 1) * 0.5
    x_offsets = (
        torch.arange(crop_size, device=anchors.device, dtype=torch.float32) - half
    ) / float(width)
    y_offsets = (
        torch.arange(crop_size, device=anchors.device, dtype=torch.float32) - half
    ) / float(height)
    y, x = torch.meshgrid(y_offsets, x_offsets, indexing="ij")
    offsets = torch.stack((x, y), dim=-1)
    return (anchors.float()[:, :, None, None, :] + offsets).mul(2.0).sub(1.0)


def anchor_centered_crops(
    values: torch.Tensor,
    anchors: torch.Tensor,
    *,
    crop_size: int = CROP_SIZE,
) -> torch.Tensor:
    """Sample one symmetric 32x32 crop per normalized anchor."""

    if (
        values.ndim != 4
        or values.size(1) != 1
        or not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("P4 recovery crop source must be finite [batch,1,H,W]")
    if values.size(0) != anchors.size(0):
        raise ValueError("P4 recovery crop source and anchors have different batches")
    grid = _anchor_grid(
        anchors,
        height=values.size(2),
        width=values.size(3),
        crop_size=crop_size,
    )
    outputs = []
    # Expanding one image over its candidates avoids materializing B*N full images.
    for batch_index in range(values.size(0)):
        candidates = anchors.size(1)
        current = F.grid_sample(
            values[batch_index : batch_index + 1].expand(candidates, -1, -1, -1),
            grid[batch_index],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        outputs.append(current)
    return torch.stack(outputs, dim=0)


def normalized_candidate_scores(scores: torch.Tensor) -> torch.Tensor:
    """Apply the established per-image logit z-normalization to peak scores."""

    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("P4 recovery scores must have shape [batch,candidates]")
    if not bool(torch.isfinite(scores).all()) or bool(
        ((scores < 0.0) | (scores > 1.0)).any()
    ):
        raise ValueError("P4 recovery peak scores must be finite probabilities")
    logits = torch.logit(scores.float().clamp(1e-6, 1.0 - 1e-6))
    mean = logits.mean(dim=1, keepdim=True)
    scale = logits.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
    return (logits - mean) / scale


def normalized_candidate_ranks(
    scores: torch.Tensor, source_ids: torch.Tensor
) -> torch.Tensor:
    """Return rank/(N-1), using source ID to resolve equal-score candidates."""

    if scores.ndim != 2 or source_ids.shape != scores.shape:
        raise ValueError("P4 recovery score and source-ID shapes differ")
    if source_ids.dtype != torch.long:
        raise ValueError("P4 recovery source IDs must be long tensors")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("P4 recovery rank scores must be finite floating point")
    ranks = torch.empty_like(scores, dtype=torch.float32)
    for batch_index in range(scores.size(0)):
        ids = source_ids[batch_index]
        if len(torch.unique(ids)) != len(ids):
            raise ValueError("P4 recovery source IDs must be unique within an image")
        identity_order = ids.argsort(stable=True)
        score_order = scores[batch_index, identity_order].argsort(
            descending=True, stable=True
        )
        order = identity_order[score_order]
        current = torch.empty_like(order, dtype=torch.float32)
        current[order] = torch.arange(
            len(order), device=order.device, dtype=torch.float32
        )
        ranks[batch_index] = current / max(len(order) - 1, 1)
    return ranks


@torch.no_grad()
def build_local_recovery_observations(
    pixel_values: torch.Tensor,
    field_output: Mapping[str, torch.Tensor],
    peaks: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Build target-free P4 recovery inputs from frozen image predictions only."""

    batch, height, width, device = _validate_observable_fields(
        pixel_values, field_output, peaks
    )
    anchors = torch.empty((batch, CANDIDATES, 2), device=device)
    scores = torch.empty((batch, CANDIDATES), device=device)
    source_ids = torch.empty((batch, CANDIDATES), device=device, dtype=torch.long)
    for batch_index, peak in enumerate(peaks):
        peak_anchors = peak.get("anchors")
        peak_scores = peak.get("scores")
        peak_sources = peak.get("source_ids")
        if (
            not isinstance(peak_anchors, torch.Tensor)
            or peak_anchors.shape != (CANDIDATES, 2)
            or not isinstance(peak_scores, torch.Tensor)
            or peak_scores.shape != (CANDIDATES,)
            or not isinstance(peak_sources, torch.Tensor)
            or peak_sources.shape != (CANDIDATES,)
            or peak_sources.dtype != torch.long
        ):
            raise ValueError("P4 recovery requires exactly 128 complete peak objects")
        anchors[batch_index] = peak_anchors.detach().to(device=device, dtype=torch.float32)
        scores[batch_index] = peak_scores.detach().to(device=device, dtype=torch.float32)
        source_ids[batch_index] = peak_sources.detach().to(device=device)
    if bool(((source_ids < 0) | (source_ids >= height * width)).any()):
        raise ValueError("P4 recovery peak source ID is outside the stride1 field")
    if not bool(torch.isfinite(anchors).all() & torch.isfinite(scores).all()):
        raise ValueError("P4 recovery peak tensors must be finite")

    feature_map = field_output["stride1_features"].detach().float()
    flat_features = feature_map.permute(0, 2, 3, 1).reshape(
        batch, height * width, CANDIDATE_FEATURE_DIM
    )
    candidate_features = torch.gather(
        flat_features,
        1,
        source_ids[..., None].expand(-1, -1, CANDIDATE_FEATURE_DIM),
    )
    darkness = _grayscale_darkness(pixel_values.detach())
    score_field = field_output["center_logits"].detach().float()
    return {
        "image_crops": anchor_centered_crops(darkness, anchors),
        "score_crops": anchor_centered_crops(score_field, anchors),
        "candidate_features": candidate_features,
        "candidate_scores": normalized_candidate_scores(scores),
        "candidate_ranks": normalized_candidate_ranks(scores, source_ids),
        "candidate_anchors": anchors,
        "candidate_source_ids": source_ids,
        "candidate_mask": torch.ones(
            (batch, CANDIDATES), device=device, dtype=torch.bool
        ),
        "raw_scores": scores,
    }


def build_local_recovery_targets(
    sample_keys: Sequence[str],
    candidate_anchors: torch.Tensor,
    candidate_source_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    action_rows: Iterable[Mapping[str, object]],
    *,
    image_size: int = 768,
) -> dict[str, torch.Tensor]:
    """Build GT-derived labels separately from target-free model observations."""

    if (
        candidate_anchors.ndim != 3
        or candidate_anchors.shape[1:] != (CANDIDATES, 2)
        or candidate_source_ids.shape != candidate_anchors.shape[:2]
        or candidate_source_ids.dtype != torch.long
        or candidate_mask.shape != candidate_anchors.shape[:2]
        or candidate_mask.dtype != torch.bool
        or len(sample_keys) != candidate_anchors.size(0)
    ):
        raise ValueError("P4 recovery target base tensors have invalid shapes")
    if image_size < 1:
        raise ValueError("P4 recovery image size must be positive")
    if len(set(map(str, sample_keys))) != len(sample_keys):
        raise ValueError("P4 recovery target sample keys must be unique")
    device = candidate_anchors.device
    if candidate_source_ids.device != device or candidate_mask.device != device:
        raise ValueError("P4 recovery target base tensors must share one device")
    targets = {
        "split_targets": torch.zeros_like(candidate_mask),
        "refine_targets": torch.zeros_like(candidate_mask),
        "split_offsets": candidate_anchors.new_zeros(candidate_anchors.shape),
        "refine_offsets": candidate_anchors.new_zeros(candidate_anchors.shape),
        "candidate_mask": candidate_mask,
    }
    sample_index = {str(key): index for index, key in enumerate(sample_keys)}
    assigned: set[tuple[int, int]] = set()
    for row in action_rows:
        key = str(row.get("sample_key", ""))
        route = str(row.get("route", ""))
        if key not in sample_index:
            raise ValueError(f"P4 recovery target action references unknown sample: {key}")
        if route not in ROUTES:
            raise ValueError(f"P4 recovery target action has unknown route: {route}")
        batch_index = sample_index[key]
        source_id = int(row["base_source_id"])
        matches = torch.where(
            candidate_mask[batch_index]
            & (candidate_source_ids[batch_index] == source_id)
        )[0]
        if len(matches) != 1:
            raise ValueError("P4 recovery action source must match one valid base candidate")
        candidate_index = int(matches[0])
        identity = (batch_index, candidate_index)
        if identity in assigned:
            raise ValueError("P4 recovery base candidate received multiple actions")
        assigned.add(identity)
        target_anchor = candidate_anchors.new_tensor(row["target_anchor"])
        if target_anchor.shape != (2,) or not bool(torch.isfinite(target_anchor).all()):
            raise ValueError("P4 recovery target anchor must be a finite coordinate pair")
        offset = target_anchor - candidate_anchors[batch_index, candidate_index]
        distance = float(torch.linalg.vector_norm(offset.float()) * image_size)
        if distance > ROUTE_RADIUS_PIXELS[route] + 1e-4:
            raise ValueError("P4 recovery target offset exceeds its route radius")
        prefix = "split" if route == "ADD_SPLIT" else "refine"
        targets[f"{prefix}_targets"][batch_index, candidate_index] = True
        targets[f"{prefix}_offsets"][batch_index, candidate_index] = offset
    return targets


def deterministic_recovery_sample_mask(
    route_targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_source_ids: torch.Tensor,
    sample_keys: Sequence[str],
    *,
    negative_ratio: int = NEGATIVE_RATIO,
    seed: int = SAMPLER_SEED,
    round_index: int = 0,
) -> torch.Tensor:
    """Select all positives and exactly three hash-ranked negatives per positive."""

    if (
        route_targets.shape != candidate_mask.shape
        or route_targets.shape != candidate_source_ids.shape
        or route_targets.ndim != 2
        or route_targets.dtype != torch.bool
        or candidate_mask.dtype != torch.bool
        or candidate_source_ids.dtype != torch.long
        or len(sample_keys) != route_targets.size(0)
    ):
        raise ValueError("P4 recovery sampler tensors have invalid shapes or dtypes")
    if isinstance(negative_ratio, bool) or negative_ratio != NEGATIVE_RATIO:
        raise ValueError("P4 recovery sampler requires a fixed 1:3 ratio")
    if isinstance(round_index, bool) or round_index < 0:
        raise ValueError("P4 recovery sampler round must be nonnegative")
    if bool((route_targets & ~candidate_mask).any()):
        raise ValueError("P4 recovery positive target is outside the candidate mask")
    selected = route_targets.clone()
    positive_count = int(route_targets.sum())
    if not positive_count:
        return selected
    negative_rows = torch.where(candidate_mask & ~route_targets)
    required = negative_ratio * positive_count
    if len(negative_rows[0]) < required:
        raise ValueError("P4 recovery sampler lacks negatives for the fixed 1:3 ratio")
    ranked = []
    for batch_index, candidate_index in zip(
        negative_rows[0].tolist(), negative_rows[1].tolist()
    ):
        identity = (
            f"{seed}:{round_index}:{sample_keys[batch_index]}:"
            f"{int(candidate_source_ids[batch_index, candidate_index])}"
        )
        ranked.append(
            (
                hashlib.sha256(identity.encode("utf-8")).digest(),
                str(sample_keys[batch_index]),
                int(candidate_source_ids[batch_index, candidate_index]),
                batch_index,
                candidate_index,
            )
        )
    for _, _, _, batch_index, candidate_index in sorted(ranked)[:required]:
        selected[batch_index, candidate_index] = True
    return selected


def local_recovery_loss(
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    sample_mask: torch.Tensor,
    *,
    route: str,
    image_size: int = 768,
) -> dict[str, torch.Tensor]:
    """Combine sampled action BCE with offset loss on positive actions only."""

    if route not in ROUTES:
        raise ValueError(f"P4 recovery loss has unknown route: {route}")
    prefix = "split" if route == "ADD_SPLIT" else "refine"
    logits = output.get(f"{prefix}_logits")
    offsets = output.get(f"{prefix}_offsets")
    action_targets = targets.get(f"{prefix}_targets")
    offset_targets = targets.get(f"{prefix}_offsets")
    candidate_mask = targets.get("candidate_mask")
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(offsets, torch.Tensor)
        or not isinstance(action_targets, torch.Tensor)
        or not isinstance(offset_targets, torch.Tensor)
        or not isinstance(candidate_mask, torch.Tensor)
        or logits.shape != action_targets.shape
        or logits.shape != candidate_mask.shape
        or logits.shape != sample_mask.shape
        or offsets.shape != (*logits.shape, 2)
        or offset_targets.shape != offsets.shape
        or action_targets.dtype != torch.bool
        or candidate_mask.dtype != torch.bool
        or sample_mask.dtype != torch.bool
    ):
        raise ValueError("P4 recovery loss tensors have invalid shapes or dtypes")
    if bool((sample_mask & ~candidate_mask).any()):
        raise ValueError("P4 recovery loss sample exceeds the candidate mask")
    if bool((action_targets & ~sample_mask).any()):
        raise ValueError("P4 recovery loss sample omitted a positive action")
    if not bool(sample_mask.any()):
        raise ValueError("P4 recovery loss requires a nonempty deterministic sample")
    action_loss = F.binary_cross_entropy_with_logits(
        logits[sample_mask].float(), action_targets[sample_mask].float()
    )
    positive = action_targets & candidate_mask
    offset_loss = (
        F.smooth_l1_loss(
            offsets[positive].float() * image_size,
            offset_targets[positive].float() * image_size,
            beta=1.0,
        )
        if bool(positive.any())
        else logits.sum() * 0.0
    )
    return {
        "loss": action_loss + offset_loss,
        "action_loss": action_loss,
        "offset_loss": offset_loss,
        "positive_count": logits.new_tensor(float(positive.sum())),
        "sampled_count": logits.new_tensor(float(sample_mask.sum())),
    }


__all__ = [
    "CANDIDATES",
    "CANDIDATE_FEATURE_DIM",
    "CROP_SIZE",
    "NEGATIVE_RATIO",
    "ROUTES",
    "anchor_centered_crops",
    "build_local_recovery_observations",
    "build_local_recovery_targets",
    "deterministic_recovery_sample_mask",
    "local_recovery_loss",
    "normalized_candidate_ranks",
    "normalized_candidate_scores",
]
