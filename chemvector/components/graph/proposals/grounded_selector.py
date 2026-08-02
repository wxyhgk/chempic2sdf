"""Post-snap query selection over frozen Atom Query and dense vertex evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from chemvector.geometry import stable_pairwise_distances

from chemvector.atom_query import (
    coordinate_hungarian_assignments,
    poisson_binomial_log_probs,
    snap_queries_to_dense_candidates,
)


DENSE_CANDIDATE_AUX_DIM = 30


def sample_candidate_features(
    feature_map: torch.Tensor,
    candidate_centers: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    radius_cells: int = 1,
) -> torch.Tensor:
    """Sample a candidate-aligned square patch from a normalized feature map."""

    if feature_map.ndim != 4:
        raise ValueError("Candidate feature map must have shape [batch, channels, h, w]")
    if candidate_centers.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate centers and mask shapes differ")
    if feature_map.size(0) != candidate_centers.size(0):
        raise ValueError("Candidate feature map batch size differs")
    if radius_cells < 0:
        raise ValueError("Candidate feature radius must be nonnegative")
    height, width = feature_map.shape[-2:]
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(
            -radius_cells,
            radius_cells + 1,
            device=feature_map.device,
            dtype=torch.float32,
        ),
        torch.arange(
            -radius_cells,
            radius_cells + 1,
            device=feature_map.device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )
    offsets = torch.stack((offsets_x / width, offsets_y / height), dim=-1).flatten(0, 1)
    grid = candidate_centers[:, :, None, :] + offsets[None, None, :, :]
    grid = grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.permute(0, 2, 3, 1).flatten(2)
    return sampled * candidate_mask.unsqueeze(-1).to(sampled.dtype)


def sample_candidate_ink_patches(
    pixel_values: torch.Tensor,
    candidate_centers: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    patch_size: int = 32,
) -> torch.Tensor:
    """Sample sub-pixel aligned grayscale ink patches around every candidate."""

    if pixel_values.ndim != 4 or pixel_values.size(1) not in (1, 3):
        raise ValueError("Pixel values must have shape [batch, 1|3, h, w]")
    if candidate_centers.shape != (*candidate_mask.shape, 2):
        raise ValueError("Candidate centers and mask shapes differ")
    if pixel_values.size(0) != candidate_centers.size(0):
        raise ValueError("Pixel and candidate batch sizes differ")
    if patch_size < 4 or patch_size % 2:
        raise ValueError("Candidate patch size must be an even integer of at least four")

    height, width = pixel_values.shape[-2:]
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(patch_size, device=pixel_values.device, dtype=torch.float32),
        torch.arange(patch_size, device=pixel_values.device, dtype=torch.float32),
        indexing="ij",
    )
    center = (patch_size - 1) / 2.0
    offsets = torch.stack(
        ((offsets_x - center) / width, (offsets_y - center) / height), dim=-1
    )
    grid = candidate_centers[:, :, None, None, :] + offsets[None, None]
    grid = grid.mul(2.0).sub(1.0).reshape(
        pixel_values.size(0), candidate_centers.size(1) * patch_size, patch_size, 2
    )
    gray = pixel_values.float().mean(dim=1, keepdim=True)
    ink = ((1.0 - gray) * 0.5).clamp(0.0, 1.0)
    sampled = F.grid_sample(
        ink,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.reshape(
        pixel_values.size(0), 1, candidate_centers.size(1), patch_size, patch_size
    ).permute(0, 2, 1, 3, 4)
    return sampled * candidate_mask[:, :, None, None, None].to(sampled.dtype)


@dataclass(frozen=True, slots=True)
class GroundedSelectorConfig:
    query_dim: int = 192
    scalar_dim: int = 10
    d_model: int = 192
    heads: int = 6
    ffn_dim: int = 768
    layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.query_dim,
            self.scalar_dim,
            self.d_model,
            self.heads,
            self.ffn_dim,
            self.layers,
        ) < 1:
            raise ValueError("Grounded selector dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("Grounded selector heads must divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Grounded selector dropout must be in [0, 1)")


class GroundedQuerySelector(nn.Module):
    """Re-score all queries after one-to-one snapping to dense peaks."""

    def __init__(self, config: GroundedSelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Sequential(
            nn.LayerNorm(config.query_dim),
            nn.Linear(config.query_dim, config.d_model),
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(config.scalar_dim),
            nn.Linear(config.scalar_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.position_projection = nn.Linear(2, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.residual_score = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.residual_score.weight)
        nn.init.zeros_(self.residual_score.bias)

    def encode(
        self,
        query_features: torch.Tensor,
        scalar_features: torch.Tensor,
        grounded_centers: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_features.ndim != 3:
            raise ValueError("query_features must have shape [batch, queries, channels]")
        if scalar_features.shape[:2] != query_features.shape[:2]:
            raise ValueError("scalar_features must match the query batch")
        if grounded_centers.shape != (*query_features.shape[:2], 2):
            raise ValueError("grounded_centers must have shape [batch, queries, 2]")
        hidden = (
            self.query_projection(query_features.float())
            + self.scalar_projection(scalar_features.float())
            + self.position_projection(grounded_centers.float())
        )
        if candidate_mask is not None and candidate_mask.shape != query_features.shape[:2]:
            raise ValueError("candidate_mask must have shape [batch, queries]")
        hidden = self.context(
            hidden,
            src_key_padding_mask=(~candidate_mask if candidate_mask is not None else None),
        )
        return hidden

    def forward(
        self,
        query_features: torch.Tensor,
        scalar_features: torch.Tensor,
        grounded_centers: torch.Tensor,
        base_logits: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_logits.shape != query_features.shape[:2]:
            raise ValueError("base_logits must have shape [batch, queries]")
        hidden = self.encode(
            query_features,
            scalar_features,
            grounded_centers,
            candidate_mask,
        )
        return base_logits.float() + self.residual_score(hidden).squeeze(-1)


def ground_all_queries(
    output: dict[str, torch.Tensor],
    dense_candidates: list[list[dict[str, float | bool]]],
    *,
    image_size: int = 768,
    maximum_distance_pixels: float = 64.0,
) -> dict[str, torch.Tensor]:
    """Snap every query before selection and return fixed-size model features."""

    centers = output["centers"]
    if len(dense_candidates) != centers.size(0):
        raise ValueError("Dense candidate batch size differs from Atom Query output")
    grounded_batches = []
    scalar_batches = []
    for batch_index, candidate_items in enumerate(dense_candidates):
        query_items = [
            {
                "query_index": query_index,
                "x": float(centers[batch_index, query_index, 0]),
                "y": float(centers[batch_index, query_index, 1]),
                "score": float(output["selection_logits"][batch_index, query_index].sigmoid()),
            }
            for query_index in range(centers.size(1))
        ]
        snapped = snap_queries_to_dense_candidates(
            query_items,
            candidate_items,
            image_size=image_size,
            maximum_distance_pixels=maximum_distance_pixels,
        )
        grounded = []
        scalars = []
        for query_index, item in enumerate(snapped):
            original = centers[batch_index, query_index].detach().float().cpu()
            grounded_center = torch.tensor(
                (float(item["x"]), float(item["y"])),
                dtype=torch.float32,
            )
            delta = grounded_center - original
            snap_distance = float(delta.square().sum().sqrt()) * image_size
            snapped_to_peak = "dense_score" in item
            grounded.append(grounded_center)
            scalars.append(
                torch.tensor(
                    (
                        float(output["presence_logits"][batch_index, query_index]),
                        float(output["ranking_presence_logits"][batch_index, query_index]),
                        float(output["local_presence_logits"][batch_index, query_index]),
                        float(output["selection_context_logits"][batch_index, query_index]),
                        float(output["selection_logits"][batch_index, query_index]),
                        float(item.get("dense_score", 0.0)),
                        float(snapped_to_peak),
                        snap_distance / maximum_distance_pixels,
                        float(delta[0]),
                        float(delta[1]),
                    ),
                    dtype=torch.float32,
                )
            )
        grounded_batches.append(torch.stack(grounded))
        scalar_batches.append(torch.stack(scalars))
    device = centers.device
    return {
        "query_features": output["selection_features"],
        "scalar_features": torch.stack(scalar_batches).to(device),
        "grounded_centers": torch.stack(grounded_batches).to(device),
        "base_logits": output["selection_logits"],
        "aromatic_logits": output["aromatic_logits"],
    }


def grounded_selector_targets(
    grounded_centers: torch.Tensor,
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    output = {"centers": grounded_centers}
    targets = torch.zeros(
        grounded_centers.shape[:2],
        device=grounded_centers.device,
        dtype=torch.float32,
    )
    for batch_index, (query_indices, _) in enumerate(
        coordinate_hungarian_assignments(output, anchor_points, anchor_mask)
    ):
        targets[batch_index, query_indices] = 1.0
    return targets


def grounded_selector_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    ranking_weight: float = 1.0,
    set_count_weight: float = 1.0,
    cardinality_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("Grounded selector logits and targets must have shape [batch, queries]")
    if min(ranking_weight, set_count_weight, cardinality_weight) < 0:
        raise ValueError("Grounded selector loss weights must be nonnegative")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(targets, dtype=torch.bool)
    if candidate_mask.shape != targets.shape:
        raise ValueError("candidate_mask must match grounded selector targets")
    valid_logits = logits[candidate_mask]
    valid_targets = targets[candidate_mask]
    positives = valid_targets.sum().clamp_min(1.0)
    negatives = (valid_targets.numel() - positives).clamp_min(1.0)
    selection_loss = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=(negatives / positives).clamp(1.0, 20.0),
    )
    ranking_losses = []
    for scores, labels, valid in zip(logits, targets.bool(), candidate_mask):
        scores = scores[valid]
        labels = labels[valid]
        positive_scores = scores[labels]
        negative_scores = scores[~labels]
        if not len(positive_scores) or not len(negative_scores):
            continue
        hard_negative_count = min(len(negative_scores), 2 * len(positive_scores))
        hard_negatives = negative_scores.topk(hard_negative_count).values
        ranking_losses.append(
            F.softplus(hard_negatives[:, None] - positive_scores[None, :]).mean()
        )
    ranking_loss = (
        torch.stack(ranking_losses).mean()
        if ranking_losses
        else selection_loss * 0.0
    )
    expected_counts = targets.sum(dim=1).long()
    count_logits = logits.masked_fill(~candidate_mask, -30.0)
    set_count_loss = F.nll_loss(
        poisson_binomial_log_probs(count_logits),
        expected_counts,
    )
    predicted_counts = count_logits.sigmoid().sum(dim=1)
    cardinality_loss = (
        (predicted_counts - expected_counts).abs()
        / expected_counts.clamp_min(1).to(predicted_counts.dtype)
    ).mean()
    loss = (
        selection_loss
        + ranking_weight * ranking_loss
        + set_count_weight * set_count_loss
        + cardinality_weight * cardinality_loss
    )
    return {
        "loss": loss,
        "selection_loss": selection_loss,
        "ranking_loss": ranking_loss,
        "set_count_loss": set_count_loss,
        "cardinality_loss": cardinality_loss,
    }


@torch.no_grad()
def decode_grounded_queries(
    logits: torch.Tensor,
    grounded_centers: torch.Tensor,
    aromatic_logits: torch.Tensor,
    *,
    counts: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> list[list[dict[str, float | bool]]]:
    if candidate_mask is None:
        candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    masked_logits = logits.masked_fill(~candidate_mask, -30.0)
    if counts is None:
        counts = poisson_binomial_log_probs(masked_logits).argmax(dim=-1)
    decoded = []
    for batch_index, count in enumerate(counts.tolist()):
        count = min(max(int(count), 0), int(candidate_mask[batch_index].sum()))
        scores, indices = masked_logits[batch_index].sigmoid().topk(count)
        decoded.append(
            [
                {
                    "x": float(grounded_centers[batch_index, index, 0]),
                    "y": float(grounded_centers[batch_index, index, 1]),
                    "score": float(score),
                    "aromatic": bool(aromatic_logits[batch_index, index] >= 0.0),
                }
                for score, index in zip(scores.tolist(), indices.tolist())
            ]
        )
    return decoded


def fuse_dense_candidates_with_queries(
    output: dict[str, torch.Tensor],
    dense_candidates: list[list[dict[str, float | bool]]],
    *,
    maximum_candidates: int = 128,
    image_size: int = 768,
    query_scale_pixels: float = 16.0,
) -> dict[str, torch.Tensor]:
    """Keep dense peak coordinates fixed and attach nearby Atom Query context."""

    if maximum_candidates < 1 or query_scale_pixels <= 0:
        raise ValueError("Dense candidate fusion limits must be positive")
    if len(dense_candidates) != output["centers"].size(0):
        raise ValueError("Dense candidate batch size differs from Atom Query output")
    device = output["centers"].device
    feature_dim = output["selection_features"].size(-1)
    batch_size = output["centers"].size(0)
    fused_features = torch.zeros(
        (batch_size, maximum_candidates, feature_dim),
        device=device,
        dtype=output["selection_features"].dtype,
    )
    scalar_features = torch.zeros(
        (batch_size, maximum_candidates, 10),
        device=device,
        dtype=torch.float32,
    )
    candidate_centers = torch.zeros(
        (batch_size, maximum_candidates, 2),
        device=device,
        dtype=torch.float32,
    )
    base_logits = torch.full(
        (batch_size, maximum_candidates),
        -30.0,
        device=device,
        dtype=torch.float32,
    )
    aromatic_logits = torch.zeros_like(base_logits)
    candidate_mask = torch.zeros_like(base_logits, dtype=torch.bool)
    candidate_auxiliary = torch.zeros(
        (batch_size, maximum_candidates, DENSE_CANDIDATE_AUX_DIM),
        device=device,
        dtype=torch.float32,
    )
    for batch_index, items in enumerate(dense_candidates):
        items = items[:maximum_candidates]
        if not items:
            continue
        count = len(items)
        points = torch.tensor(
            [(float(item["x"]), float(item["y"])) for item in items],
            device=device,
            dtype=torch.float32,
        )
        distances = stable_pairwise_distances(
            points,
            output["centers"][batch_index].detach(),
        ) * image_size
        top_distances, top_indices = distances.topk(
            min(4, distances.size(1)),
            largest=False,
            dim=1,
        )
        weights = (-top_distances / query_scale_pixels).softmax(dim=1)
        nearby_features = output["selection_features"][batch_index][top_indices]
        fused = (nearby_features * weights[..., None].to(nearby_features.dtype)).sum(
            dim=1
        )
        nearest = top_indices[:, 0]
        dense_scores = torch.tensor(
            [float(item["score"]) for item in items],
            device=device,
            dtype=torch.float32,
        ).clamp(1e-5, 1.0 - 1e-5)
        dense_aromatic = torch.tensor(
            [float(bool(item.get("aromatic", False))) for item in items],
            device=device,
            dtype=torch.float32,
        )
        support = (-distances / query_scale_pixels).exp().sum(dim=1).log1p()
        scalars = torch.stack(
            (
                dense_scores,
                dense_aromatic,
                top_distances[:, 0] / image_size,
                (top_distances * weights).sum(dim=1) / image_size,
                output["presence_logits"][batch_index, nearest].float(),
                output["ranking_presence_logits"][batch_index, nearest].float(),
                output["local_presence_logits"][batch_index, nearest].float(),
                output["selection_context_logits"][batch_index, nearest].float(),
                output["selection_logits"][batch_index, nearest].float(),
                support,
            ),
            dim=-1,
        )
        fused_features[batch_index, :count] = fused
        scalar_features[batch_index, :count] = scalars
        candidate_centers[batch_index, :count] = points
        base_logits[batch_index, :count] = torch.logit(dense_scores)
        aromatic_logits[batch_index, :count] = dense_aromatic * 20.0 - 10.0
        candidate_mask[batch_index, :count] = True
        for candidate_index, item in enumerate(items):
            patch = tuple(float(value) for value in item.get("local_heatmap_patch", ()))
            if len(patch) not in (0, 25):
                raise ValueError("Dense candidate local heatmap patch must contain 25 values")
            values = (patch if patch else (0.0,) * 25) + (
                float(bool(item.get("nms3_peak", False))),
                float(bool(item.get("nms5_peak", False))),
                float(item.get("offset_x", 0.5)),
                float(item.get("offset_y", 0.5)),
                float(bool(item.get("aromatic", False))),
            )
            candidate_auxiliary[batch_index, candidate_index] = torch.tensor(
                values, device=device, dtype=torch.float32
            )
    return {
        "query_features": fused_features,
        "scalar_features": scalar_features,
        "grounded_centers": candidate_centers,
        "base_logits": base_logits,
        "aromatic_logits": aromatic_logits,
        "candidate_mask": candidate_mask,
        "candidate_auxiliary": candidate_auxiliary,
    }


def dense_candidate_targets(
    candidate_centers: torch.Tensor,
    candidate_mask: torch.Tensor,
    anchor_points: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    if candidate_mask.shape != candidate_centers.shape[:2]:
        raise ValueError("candidate_mask must match candidate centers")
    targets = torch.zeros_like(candidate_mask, dtype=torch.float32)
    for batch_index in range(candidate_centers.size(0)):
        valid_candidates = torch.where(candidate_mask[batch_index])[0]
        valid_targets = torch.where(anchor_mask[batch_index])[0]
        if not len(valid_candidates) or not len(valid_targets):
            continue
        distances = stable_pairwise_distances(
            candidate_centers[batch_index, valid_candidates],
            anchor_points[batch_index, valid_targets],
        )
        candidate_indices, _ = linear_sum_assignment(distances.detach().cpu().numpy())
        targets[batch_index, valid_candidates[candidate_indices]] = 1.0
    return targets
