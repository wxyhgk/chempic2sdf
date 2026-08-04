"""Set ranking for the fixed P1-W multiscale Atom peak protocol."""

from __future__ import annotations

import math

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F


G3_MULTISCALE_CANDIDATES = 128
G3_MULTISCALE_FEATURE_DIM = 32
G3_MULTISCALE_MODEL_DIM = 64
G3_MULTISCALE_HEADS = 4
G3_MULTISCALE_FFN_DIM = 128
G3_MULTISCALE_LAYERS = 3
G3_MULTISCALE_MATCH_RADIUS_PIXELS = 6.0


def _validate_candidates(candidates: dict[str, torch.Tensor]) -> int:
    required = {
        "candidate_ids",
        "candidate_features",
        "candidate_anchors",
        "raw_scores",
        "normalized_scores",
        "scale_ids",
        "scale_ranks",
    }
    missing = required - candidates.keys()
    if missing:
        raise ValueError(f"G3 multiscale candidates are missing: {sorted(missing)}")
    ids = candidates["candidate_ids"]
    if ids.ndim != 2 or ids.size(1) != G3_MULTISCALE_CANDIDATES:
        raise ValueError("G3 multiscale candidate IDs must have shape [batch,128]")
    batch = ids.size(0)
    expected = {
        "candidate_features": (batch, G3_MULTISCALE_CANDIDATES, G3_MULTISCALE_FEATURE_DIM),
        "candidate_anchors": (batch, G3_MULTISCALE_CANDIDATES, 2),
        "raw_scores": (batch, G3_MULTISCALE_CANDIDATES),
        "normalized_scores": (batch, G3_MULTISCALE_CANDIDATES),
        "scale_ids": (batch, G3_MULTISCALE_CANDIDATES),
        "scale_ranks": (batch, G3_MULTISCALE_CANDIDATES),
    }
    if ids.dtype != torch.long:
        raise ValueError("G3 multiscale candidate IDs must be int64")
    for name, shape in expected.items():
        if candidates[name].shape != shape:
            raise ValueError(f"G3 multiscale {name} has invalid shape")
    if candidates["scale_ids"].dtype != torch.long:
        raise ValueError("G3 multiscale scale IDs must be int64")
    if not bool(((candidates["scale_ids"] == 0) | (candidates["scale_ids"] == 1)).all()):
        raise ValueError("G3 multiscale scale IDs must be zero or one")
    if len({value.device for value in candidates.values()}) != 1:
        raise ValueError("G3 multiscale candidate tensors must share one device")
    for name, value in candidates.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"G3 multiscale {name} must be finite")
    for sample in ids:
        if len(sample.unique()) != G3_MULTISCALE_CANDIDATES:
            raise ValueError("G3 multiscale candidate identities must be unique")
    return batch


class G3MultiscalePeakSetRanker(nn.Module):
    """Rank 96 stride-2 and 32 stride-1 peaks as one unordered set."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(
            G3_MULTISCALE_FEATURE_DIM, elementwise_affine=False
        )
        self.feature_projection = nn.Linear(
            G3_MULTISCALE_FEATURE_DIM, G3_MULTISCALE_MODEL_DIM
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, G3_MULTISCALE_MODEL_DIM),
        )
        self.token_norm = nn.LayerNorm(G3_MULTISCALE_MODEL_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=G3_MULTISCALE_MODEL_DIM,
            nhead=G3_MULTISCALE_HEADS,
            dim_feedforward=G3_MULTISCALE_FFN_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=G3_MULTISCALE_LAYERS
        )
        self.residual_head = nn.Linear(G3_MULTISCALE_MODEL_DIM, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, candidates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _validate_candidates(candidates)
        features = candidates["candidate_features"].detach().float()
        anchors = candidates["candidate_anchors"].detach().float()
        raw_scores = candidates["raw_scores"].detach().float()
        normalized_scores = candidates["normalized_scores"].detach().float()
        scale_ranks = candidates["scale_ranks"].detach().float()
        scale_one_hot = F.one_hot(
            candidates["scale_ids"].detach(), num_classes=2
        ).float()
        geometry = torch.cat(
            (
                anchors,
                raw_scores[..., None],
                normalized_scores[..., None],
                scale_ranks[..., None],
                scale_one_hot,
            ),
            dim=-1,
        )
        hidden = self.feature_projection(self.feature_norm(features))
        hidden = self.token_norm(F.gelu(hidden + self.geometry_projection(geometry)))
        hidden = self.set_encoder(hidden)
        residual = self.residual_head(hidden).squeeze(-1)
        return {
            **candidates,
            "ranking_residual": residual,
            "ranking_logits": normalized_scores + residual,
        }


def _stable_score_ranks(scores: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    identity_order = ids.argsort(stable=True)
    score_order = scores[identity_order].argsort(descending=True, stable=True)
    order = identity_order[score_order]
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), dtype=order.dtype)
    return ranks


@torch.no_grad()
def build_dynamic_multiscale_targets(
    ranked: dict[str, torch.Tensor],
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    radius_pixels: float = G3_MULTISCALE_MATCH_RADIUS_PIXELS,
    image_size: int = 768,
) -> dict[str, torch.Tensor]:
    """Assign exactly one score-preferred compatible peak to every GT Atom."""

    batch = _validate_candidates(ranked)
    logits = ranked.get("ranking_logits")
    if logits is None or logits.shape != (batch, G3_MULTISCALE_CANDIDATES):
        raise ValueError("G3 multiscale ranking logits have invalid shape")
    if (
        target_anchors.ndim != 3
        or target_anchors.shape[0] != batch
        or target_anchors.shape[-1] != 2
        or target_mask.shape != target_anchors.shape[:2]
        or target_mask.dtype != torch.bool
    ):
        raise ValueError("G3 multiscale target tensors have invalid shapes")
    if target_anchors.device != logits.device or target_mask.device != logits.device:
        raise ValueError("G3 multiscale candidates and targets must share one device")
    if not math.isfinite(float(radius_pixels)) or radius_pixels <= 0:
        raise ValueError("G3 multiscale match radius must be positive and finite")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("G3 multiscale image size must be a positive integer")

    device = logits.device
    scores = logits.detach().float().cpu()
    ids = ranked["candidate_ids"].detach().cpu()
    anchors = ranked["candidate_anchors"].detach().double().cpu()
    targets = target_anchors.detach().double().cpu()
    masks = target_mask.detach().cpu()
    positive = torch.zeros((batch, G3_MULTISCALE_CANDIDATES), dtype=torch.bool)
    matched_indices = torch.full(
        (batch, G3_MULTISCALE_CANDIDATES), -1, dtype=torch.long
    )
    target_counts = masks.sum(dim=1).long()
    alternative_candidates = torch.zeros(batch, dtype=torch.long)
    radius = float(radius_pixels) / float(image_size)

    for batch_index in range(batch):
        target_indices = torch.where(masks[batch_index])[0]
        if not len(target_indices):
            raise ValueError("G3 multiscale matching requires at least one GT Atom")
        expected = targets[batch_index, target_indices]
        compatible = torch.cdist(anchors[batch_index], expected) <= radius
        if bool((compatible.sum(dim=0) < 1).any()):
            raise RuntimeError("G3 multiscale fixed candidate set cannot cover every GT")
        alternative_candidates[batch_index] = int(
            (compatible.sum(dim=0) - 1).clamp_min(0).sum()
        )
        ranks = _stable_score_ranks(scores[batch_index], ids[batch_index]).double()
        count = len(target_indices)
        penalty = float(count + 1)
        real_cost = ranks[:, None].expand(-1, count).clone()
        real_cost /= float(G3_MULTISCALE_CANDIDATES)
        real_cost[~compatible] = 2.0 * penalty
        dummy = torch.full((count, count), penalty, dtype=torch.float64)
        rows, columns = linear_sum_assignment(
            torch.cat((real_cost.T, dummy), dim=1).numpy()
        )
        assignments = [
            (row, column)
            for row, column in zip(rows.tolist(), columns.tolist())
            if column < G3_MULTISCALE_CANDIDATES
            and bool(compatible[column, row])
        ]
        if len(assignments) != count:
            raise RuntimeError("G3 multiscale dynamic match reduced target cardinality")
        for row, column in assignments:
            positive[batch_index, column] = True
            matched_indices[batch_index, column] = target_indices[row]

    return {
        "positive_mask": positive.to(device),
        "matched_target_indices": matched_indices.to(device),
        "target_counts": target_counts.to(device),
        "alternative_candidates": alternative_candidates.to(device),
    }


@torch.no_grad()
def select_known_count_multiscale(
    ranked: dict[str, torch.Tensor], counts: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the count-sized score prefix without consulting target coordinates."""

    batch = _validate_candidates(ranked)
    logits = ranked.get("ranking_logits")
    if logits is None or logits.shape != (batch, G3_MULTISCALE_CANDIDATES):
        raise ValueError("G3 multiscale selection logits have invalid shape")
    if counts.shape != (batch,) or counts.dtype != torch.long:
        raise ValueError("G3 multiscale known counts must be int64 [batch]")
    if bool(((counts < 1) | (counts > G3_MULTISCALE_CANDIDATES)).any()):
        raise ValueError("G3 multiscale known counts are outside 1..128")
    output = torch.zeros(
        batch, G3_MULTISCALE_CANDIDATES, dtype=torch.bool, device=logits.device
    )
    for batch_index in range(batch):
        identity_order = ranked["candidate_ids"][batch_index].argsort(stable=True)
        score_order = logits[batch_index, identity_order].argsort(
            descending=True, stable=True
        )
        selected = identity_order[score_order[: int(counts[batch_index])]]
        output[batch_index, selected] = True
    return {"selected_mask": output, "atom_counts": counts.detach().clone()}
