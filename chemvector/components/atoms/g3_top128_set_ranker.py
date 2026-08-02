"""Permutation-equivariant ranking over a frozen top-128 G3 Atom set."""

from __future__ import annotations

import math

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F


G3_TOP128_CANDIDATES = 128
G3_TOP128_FEATURE_DIM = 384
G3_TOP128_MODEL_DIM = 64
G3_TOP128_HEADS = 4
G3_TOP128_FFN_DIM = 128
G3_TOP128_SET_RANKER_PARAMETERS = 60_545


def _validate_frozen_dense(dense: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    required = {"presence_logits", "slot_features", "anchors"}
    missing = required - dense.keys()
    if missing:
        raise ValueError(f"G3 top-128 dense output is missing: {sorted(missing)}")
    logits = dense["presence_logits"]
    features = dense["slot_features"]
    anchors = dense["anchors"]
    if logits.ndim != 4 or logits.size(1) != 3:
        raise ValueError("G3 top-128 logits must have shape [batch,3,h,w]")
    batch, slots, height, width = logits.shape
    if features.shape != (
        batch,
        slots,
        G3_TOP128_FEATURE_DIM,
        height,
        width,
    ):
        raise ValueError("G3 top-128 slot features must have 384 channels")
    if anchors.shape != (batch, slots, 2, height, width):
        raise ValueError("G3 top-128 anchors and logits differ")
    if slots * height * width < G3_TOP128_CANDIDATES:
        raise ValueError("G3 dense Atom grid has fewer than 128 candidates")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("G3 top-128 base logits must be finite")
    return batch, height, width


def _stable_top128_ids(scores: torch.Tensor) -> torch.Tensor:
    """Resolve the top-128 boundary by ascending flat identity on exact ties."""

    threshold = scores.topk(G3_TOP128_CANDIDATES, sorted=False).values.min()
    above = torch.where(scores > threshold)[0]
    tied = torch.where(scores == threshold)[0]
    needed = G3_TOP128_CANDIDATES - len(above)
    if needed < 0 or len(tied) < needed:
        raise RuntimeError("G3 top-128 boundary could not be resolved")
    selected = torch.cat((above, tied[:needed])).sort().values
    if len(selected) != G3_TOP128_CANDIDATES or len(selected.unique()) != len(
        selected
    ):
        raise RuntimeError("G3 top-128 identities are not unique")
    return selected


@torch.no_grad()
def extract_frozen_top128(dense: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract a deterministic, target-free candidate set from frozen base scores."""

    batch, height, width = _validate_frozen_dense(dense)
    logits = dense["presence_logits"].detach().flatten(1)
    features = (
        dense["slot_features"]
        .detach()
        .permute(0, 1, 3, 4, 2)
        .reshape(batch, -1, G3_TOP128_FEATURE_DIM)
    )
    anchors = (
        dense["anchors"].detach().permute(0, 1, 3, 4, 2).reshape(batch, -1, 2)
    )
    candidate_ids = torch.stack(
        [_stable_top128_ids(sample.float()) for sample in logits]
    )
    gather_features = candidate_ids[..., None].expand(-1, -1, features.size(-1))
    gather_anchors = candidate_ids[..., None].expand(-1, -1, 2)
    return {
        "candidate_ids": candidate_ids,
        "candidate_features": features.gather(1, gather_features),
        "candidate_anchors": anchors.gather(1, gather_anchors),
        "base_logits": logits.gather(1, candidate_ids),
        "grid_height": torch.full(
            (batch,), height, device=logits.device, dtype=torch.long
        ),
        "grid_width": torch.full(
            (batch,), width, device=logits.device, dtype=torch.long
        ),
    }


def _validate_candidates(candidates: dict[str, torch.Tensor]) -> tuple[int, int]:
    required = {
        "candidate_ids",
        "candidate_features",
        "candidate_anchors",
        "base_logits",
        "grid_height",
        "grid_width",
    }
    missing = required - candidates.keys()
    if missing:
        raise ValueError(f"G3 top-128 candidates are missing: {sorted(missing)}")
    ids = candidates["candidate_ids"]
    features = candidates["candidate_features"]
    anchors = candidates["candidate_anchors"]
    logits = candidates["base_logits"]
    if ids.ndim != 2 or ids.size(1) != G3_TOP128_CANDIDATES:
        raise ValueError("G3 candidate IDs must have shape [batch,128]")
    batch = ids.size(0)
    if ids.dtype != torch.long or features.shape != (
        batch,
        G3_TOP128_CANDIDATES,
        G3_TOP128_FEATURE_DIM,
    ):
        raise ValueError("G3 top-128 candidate features or IDs differ")
    if anchors.shape != (batch, G3_TOP128_CANDIDATES, 2) or logits.shape != (
        batch,
        G3_TOP128_CANDIDATES,
    ):
        raise ValueError("G3 top-128 candidate geometry or scores differ")
    if candidates["grid_height"].shape != (batch,) or candidates[
        "grid_width"
    ].shape != (batch,):
        raise ValueError("G3 top-128 grid dimensions differ")
    devices = {value.device for value in candidates.values()}
    if len(devices) != 1:
        raise ValueError("G3 top-128 candidate tensors must share one device")
    if not bool(torch.isfinite(features).all()) or not bool(
        torch.isfinite(anchors).all() & torch.isfinite(logits).all()
    ):
        raise ValueError("G3 top-128 candidates must be finite")
    for batch_index in range(batch):
        if len(ids[batch_index].unique()) != G3_TOP128_CANDIDATES:
            raise ValueError("G3 top-128 candidate identities must be unique")
        capacity = int(candidates["grid_height"][batch_index]) * int(
            candidates["grid_width"][batch_index]
        ) * 3
        if capacity < G3_TOP128_CANDIDATES or bool(
            ((ids[batch_index] < 0) | (ids[batch_index] >= capacity)).any()
        ):
            raise ValueError("G3 top-128 candidate identity is outside the grid")
    return batch, G3_TOP128_CANDIDATES


class G3Top128SetRanker(nn.Module):
    """Apply one target-free self-attention layer to frozen Atom candidates."""

    def __init__(self, config) -> None:
        super().__init__()
        if int(config.d_model) != G3_TOP128_FEATURE_DIM:
            raise ValueError("G3 top-128 ranker requires d_model=384")
        if int(config.maximum_atoms) != G3_TOP128_CANDIDATES:
            raise ValueError("G3 top-128 ranker requires maximum_atoms=128")
        if int(config.atom_slots_per_cell) != 3:
            raise ValueError("G3 top-128 ranker requires three Atom slots")
        self.config = config
        self.feature_norm = nn.LayerNorm(
            G3_TOP128_FEATURE_DIM, elementwise_affine=False
        )
        self.feature_projection = nn.Linear(
            G3_TOP128_FEATURE_DIM, G3_TOP128_MODEL_DIM
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, G3_TOP128_MODEL_DIM),
        )
        self.token_norm = nn.LayerNorm(G3_TOP128_MODEL_DIM)
        self.set_encoder = nn.TransformerEncoderLayer(
            d_model=G3_TOP128_MODEL_DIM,
            nhead=G3_TOP128_HEADS,
            dim_feedforward=G3_TOP128_FFN_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.residual_head = nn.Linear(G3_TOP128_MODEL_DIM, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.parameter_count() != G3_TOP128_SET_RANKER_PARAMETERS:
            raise RuntimeError("G3 top-128 set-ranker parameter count changed")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, candidates: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        _validate_candidates(candidates)
        features = candidates["candidate_features"].detach()
        anchors = candidates["candidate_anchors"].detach()
        base_logits = candidates["base_logits"].detach()
        hidden = self.feature_projection(self.feature_norm(features.float()))
        geometry_score = torch.cat((anchors.float(), base_logits.float()[..., None]), -1)
        hidden = self.token_norm(
            F.gelu(hidden + self.geometry_projection(geometry_score))
        )
        hidden = self.set_encoder(hidden)
        residual = self.residual_head(hidden).squeeze(-1)
        return {
            **candidates,
            "base_logits": base_logits,
            "ranking_residual": residual,
            "ranking_logits": base_logits.float() + residual,
        }


def _canonical_flat_ids(
    anchors: torch.Tensor, mask: torch.Tensor, *, height: int, width: int
) -> list[tuple[int, int]]:
    cells: dict[tuple[int, int], list[int]] = {}
    for target_index in torch.where(mask)[0].tolist():
        anchor = anchors[target_index]
        cell_x = min(width - 1, max(0, int(float(anchor[0]) * width)))
        cell_y = min(height - 1, max(0, int(float(anchor[1]) * height)))
        cells.setdefault((cell_y, cell_x), []).append(target_index)
    identities = []
    for (cell_y, cell_x), indices in cells.items():
        coordinates = [
            tuple(anchors[index].detach().float().cpu().tolist())
            for index in indices
        ]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("G3 hybrid target rejects identical Atom coordinates")
        indices.sort(
            key=lambda index: (*anchors[index].detach().float().cpu().tolist(), index)
        )
        if len(indices) > 3:
            raise ValueError("G3 hybrid target exceeds canonical slot capacity")
        for slot, target_index in enumerate(indices):
            identities.append((slot * height * width + cell_y * width + cell_x, target_index))
    return identities


@torch.no_grad()
def build_hybrid_top128_targets(
    candidates: dict[str, torch.Tensor],
    atom_targets: dict[str, torch.Tensor],
    radius_pixels: float = 6,
    image_size: int = 768,
) -> dict[str, torch.Tensor]:
    """Prefer canonical identities, then match missing GT to frozen 6px geometry."""

    batch, count = _validate_candidates(candidates)
    if not math.isfinite(float(radius_pixels)) or radius_pixels <= 0:
        raise ValueError("G3 hybrid radius_pixels must be positive and finite")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("G3 hybrid image_size must be a positive integer")
    target_anchors = atom_targets.get("atom_anchors")
    target_mask = atom_targets.get("atom_mask")
    if (
        target_anchors is None
        or target_mask is None
        or target_anchors.ndim != 3
        or target_anchors.shape[-1] != 2
        or target_mask.shape != target_anchors.shape[:2]
        or len(target_anchors) != batch
    ):
        raise ValueError("G3 hybrid Atom targets have invalid shapes")
    if target_anchors.device != candidates["candidate_ids"].device:
        raise ValueError("G3 hybrid candidates and targets must share one device")

    output_device = target_anchors.device
    candidate_ids = candidates["candidate_ids"].detach().cpu()
    candidate_anchors = candidates["candidate_anchors"].detach().double().cpu()
    grid_height = candidates["grid_height"].detach().cpu()
    grid_width = candidates["grid_width"].detach().cpu()
    target_anchors_cpu = target_anchors.detach().double().cpu()
    target_mask_cpu = target_mask.detach().bool().cpu()
    positive = torch.zeros((batch, count), dtype=torch.bool)
    canonical_positive = torch.zeros_like(positive)
    matched_target_indices = torch.full(
        (batch, count), -1, dtype=torch.long
    )
    representable_counts = torch.zeros(batch, dtype=torch.long)
    canonical_counts = torch.zeros_like(representable_counts)
    target_counts = target_mask_cpu.sum(dim=1).long()
    radius = float(radius_pixels) / float(image_size)

    for batch_index in range(batch):
        height = int(grid_height[batch_index])
        width = int(grid_width[batch_index])
        ids = candidate_ids[batch_index]
        id_to_position = {int(value): index for index, value in enumerate(ids.tolist())}
        identities = _canonical_flat_ids(
            target_anchors_cpu[batch_index],
            target_mask_cpu[batch_index],
            height=height,
            width=width,
        )
        claimed_targets = set()
        claimed_candidates = set()
        for flat_id, target_index in identities:
            position = id_to_position.get(flat_id)
            if position is None:
                continue
            distance = torch.linalg.vector_norm(
                (
                    candidate_anchors[batch_index, position]
                    - target_anchors_cpu[batch_index, target_index]
                )
            )
            if float(distance) > radius:
                continue
            positive[batch_index, position] = True
            canonical_positive[batch_index, position] = True
            matched_target_indices[batch_index, position] = target_index
            claimed_targets.add(target_index)
            claimed_candidates.add(position)

        remaining_targets = sorted(
            (
                index
                for index in torch.where(target_mask_cpu[batch_index])[0].tolist()
                if index not in claimed_targets
            ),
            key=lambda index: (
                *target_anchors_cpu[batch_index, index].tolist(),
                index,
            ),
        )
        remaining_candidates = [
            index for index in range(count) if index not in claimed_candidates
        ]
        if remaining_targets and remaining_candidates:
            remaining_candidate_anchors = candidate_anchors[
                batch_index, remaining_candidates
            ]
            expected_anchors = target_anchors_cpu[batch_index, remaining_targets]
            distances = torch.cdist(
                remaining_candidate_anchors, expected_anchors, p=2
            )
            compatible = distances <= radius
            large = float(max(len(remaining_candidates), len(remaining_targets)) + 1)
            cost = torch.where(compatible, distances, torch.full_like(distances, large))
            row_tie = torch.arange(len(remaining_candidates), dtype=torch.float64)[:, None]
            column_tie = torch.arange(len(remaining_targets), dtype=torch.float64)[None]
            cost = cost + 1e-12 * (row_tie * max(len(remaining_targets), 1) + column_tie)
            rows, columns = linear_sum_assignment(cost.numpy())
            for row, column in zip(rows.tolist(), columns.tolist()):
                if not bool(compatible[row, column]):
                    continue
                position = remaining_candidates[row]
                target_index = remaining_targets[column]
                positive[batch_index, position] = True
                matched_target_indices[batch_index, position] = target_index

        representable_counts[batch_index] = positive[batch_index].sum()
        canonical_counts[batch_index] = canonical_positive[batch_index].sum()

    return {
        "positive_mask": positive.to(output_device),
        "canonical_positive_mask": canonical_positive.to(output_device),
        "matched_target_indices": matched_target_indices.to(output_device),
        "representable_counts": representable_counts.to(output_device),
        "canonical_counts": canonical_counts.to(output_device),
        "target_counts": target_counts.to(output_device),
    }


def top128_set_ranking_loss(
    ranked: dict[str, torch.Tensor],
    hybrid: dict[str, torch.Tensor],
    negative_multiplier: int = 4,
) -> dict[str, torch.Tensor]:
    """Apply margin-zero ranking against the hardest ``4 * M`` negatives."""

    if (
        isinstance(negative_multiplier, bool)
        or not isinstance(negative_multiplier, int)
        or negative_multiplier < 1
    ):
        raise ValueError("G3 top-128 negative_multiplier must be a positive integer")
    logits = ranked.get("ranking_logits")
    positive = hybrid.get("positive_mask")
    if logits is None or positive is None or logits.shape != positive.shape:
        raise ValueError("G3 top-128 ranked logits and hybrid targets differ")
    losses = []
    hard_negative_count = 0
    for batch_index in range(len(logits)):
        current_positive = positive[batch_index].bool()
        positive_count = int(current_positive.sum())
        if positive_count < 1:
            raise ValueError("G3 top-128 ranking requires a representable positive")
        scores = logits[batch_index].float()
        positive_scores = scores[current_positive]
        negative_scores = scores[~current_positive]
        selected_count = min(negative_multiplier * positive_count, len(negative_scores))
        if selected_count < 1:
            raise ValueError("G3 top-128 ranking requires a negative candidate")
        hard_negatives = negative_scores.topk(selected_count).values
        losses.append(
            F.softplus(hard_negatives[:, None] - positive_scores[None, :]).mean()
        )
        hard_negative_count += selected_count
    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "positive_count": positive.sum().detach().float(),
        "hard_negative_count": loss.new_tensor(float(hard_negative_count)),
    }


@torch.no_grad()
def select_ranked_top128(
    ranked: dict[str, torch.Tensor], counts: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Sort all 128 candidates, then expose a count-sized prefix."""

    logits = ranked.get("ranking_logits")
    if logits is None or logits.ndim != 2 or logits.size(1) != G3_TOP128_CANDIDATES:
        raise ValueError("G3 ranked logits must have shape [batch,128]")
    if (
        counts.shape != (len(logits),)
        or counts.dtype == torch.bool
        or counts.dtype.is_floating_point
        or counts.dtype.is_complex
    ):
        raise ValueError("G3 top-128 counts must be an integer batch vector")
    counts = counts.to(logits.device).long()
    if bool(((counts < 0) | (counts > G3_TOP128_CANDIDATES)).any()):
        raise ValueError("G3 top-128 count is outside 0..128")
    ids = ranked["candidate_ids"]
    identity_order = ids.argsort(dim=1, stable=True)
    identity_scores = logits.gather(1, identity_order)
    score_order = identity_scores.argsort(dim=1, descending=True, stable=True)
    order = identity_order.gather(1, score_order)
    mask = torch.arange(G3_TOP128_CANDIDATES, device=logits.device)[None] < counts[:, None]

    def gather(values: torch.Tensor) -> torch.Tensor:
        index = order.reshape(*order.shape, *([1] * (values.ndim - 2))).expand(
            *order.shape, *values.shape[2:]
        )
        return values.gather(1, index)

    selected_logits = gather(logits)
    return {
        "atom_counts": counts,
        "atom_mask": mask,
        "atom_flat_ids": gather(ids),
        "atom_presence_logits": selected_logits,
        "atom_scores": selected_logits.sigmoid(),
        "atom_anchors": gather(ranked["candidate_anchors"]),
        "atom_features": gather(ranked["candidate_features"]),
        "candidate_order": order,
    }
