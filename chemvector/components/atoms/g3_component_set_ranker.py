"""Target-free 3px Atom components and a masked component set ranker."""

from __future__ import annotations

import math

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.atoms.g3_top128_set_ranker import (
    G3_TOP128_CANDIDATES,
    G3_TOP128_FEATURE_DIM,
)


G3_COMPONENT_RADIUS_PIXELS = 3.0
G3_COMPONENT_MATCH_RADIUS_PIXELS = 6.0
G3_COMPONENT_MODEL_DIM = 64
G3_COMPONENT_HEADS = 4
G3_COMPONENT_FFN_DIM = 128


def target_free_component_labels(
    anchors: torch.Tensor,
    *,
    radius_pixels: float = G3_COMPONENT_RADIUS_PIXELS,
    image_size: int = 768,
) -> torch.Tensor:
    """Cluster one candidate set by deterministic anchor connected components."""

    if anchors.ndim != 2 or anchors.shape != (G3_TOP128_CANDIDATES, 2):
        raise ValueError("G3 component anchors must have shape [128,2]")
    if not math.isfinite(float(radius_pixels)) or radius_pixels <= 0:
        raise ValueError("G3 component radius must be positive and finite")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("G3 component image size must be a positive integer")
    distances = torch.cdist(anchors.detach().float().cpu(), anchors.detach().float().cpu())
    distances *= float(image_size)
    parent = list(range(G3_TOP128_CANDIDATES))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    rows, columns = torch.where(
        torch.triu(distances <= float(radius_pixels), diagonal=1)
    )
    for left, right in zip(rows.tolist(), columns.tolist()):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root
    roots = [find(index) for index in range(G3_TOP128_CANDIDATES)]
    mapping = {root: label for label, root in enumerate(sorted(set(roots)))}
    return torch.tensor([mapping[root] for root in roots], dtype=torch.long)


@torch.no_grad()
def build_target_free_components(candidates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collapse frozen top-128 candidates into padded 3px base-softmax components."""

    required = {"candidate_ids", "candidate_features", "candidate_anchors", "base_logits"}
    if required - candidates.keys():
        raise ValueError("G3 component candidates are missing required tensors")
    ids = candidates["candidate_ids"]
    features = candidates["candidate_features"]
    anchors = candidates["candidate_anchors"]
    base_logits = candidates["base_logits"]
    batch = len(ids)
    if (
        ids.shape != (batch, 128)
        or features.shape != (batch, 128, G3_TOP128_FEATURE_DIM)
        or anchors.shape != (batch, 128, 2)
        or base_logits.shape != (batch, 128)
    ):
        raise ValueError("G3 component candidate shapes differ")
    device = ids.device
    output_ids = torch.arange(128, device=device)[None].expand(batch, -1).clone()
    output_features = torch.zeros(batch, 128, G3_TOP128_FEATURE_DIM, device=device)
    output_anchors = torch.zeros(batch, 128, 2, device=device)
    output_logits = torch.full((batch, 128), -10_000.0, device=device)
    output_mask = torch.zeros(batch, 128, dtype=torch.bool, device=device)
    output_sizes = torch.zeros(batch, 128, device=device)
    output_spreads = torch.zeros(batch, 128, device=device)
    output_member_ids = torch.full(
        (batch, 128), -1, dtype=torch.long, device=device
    )
    component_counts = torch.zeros(batch, dtype=torch.long, device=device)
    label_digest_values = []
    for batch_index in range(batch):
        labels = target_free_component_labels(anchors[batch_index]).to(device)
        label_digest_values.append(labels.cpu())
        count = int(labels.max()) + 1
        component_counts[batch_index] = count
        output_mask[batch_index, :count] = True
        for component in range(count):
            members = torch.where(labels == component)[0]
            member_logits = base_logits[batch_index, members].detach().float()
            weights = member_logits.softmax(dim=0)
            member_anchors = anchors[batch_index, members].detach().float()
            member_features = features[batch_index, members].detach().float()
            representative = (member_anchors * weights[:, None]).sum(dim=0)
            output_anchors[batch_index, component] = representative
            output_features[batch_index, component] = (
                member_features * weights[:, None]
            ).sum(dim=0)
            output_logits[batch_index, component] = member_logits.max()
            output_sizes[batch_index, component] = len(members)
            output_spreads[batch_index, component] = (
                torch.linalg.vector_norm(member_anchors - representative, dim=1).max()
                * 768.0
            )
            identity_order = ids[batch_index, members].argsort(stable=True)
            ordered_scores = member_logits[identity_order]
            local = identity_order[
                ordered_scores.argsort(descending=True, stable=True)[0]
            ]
            output_member_ids[batch_index, component] = ids[
                batch_index, members[local]
            ]
    return {
        "candidate_ids": output_ids,
        "candidate_features": output_features,
        "candidate_anchors": output_anchors,
        "base_logits": output_logits,
        "component_mask": output_mask,
        "component_counts": component_counts,
        "component_sizes": output_sizes,
        "component_spreads_pixels": output_spreads,
        "representative_member_ids": output_member_ids,
        "source_component_labels": torch.stack(label_digest_values).to(device),
    }


class G3ComponentSetRanker(nn.Module):
    """Rank padded target-free Atom components with masked self-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(
            G3_TOP128_FEATURE_DIM, elementwise_affine=False
        )
        self.feature_projection = nn.Linear(
            G3_TOP128_FEATURE_DIM, G3_COMPONENT_MODEL_DIM
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, G3_COMPONENT_MODEL_DIM),
        )
        self.token_norm = nn.LayerNorm(G3_COMPONENT_MODEL_DIM)
        self.set_encoder = nn.TransformerEncoderLayer(
            d_model=G3_COMPONENT_MODEL_DIM,
            nhead=G3_COMPONENT_HEADS,
            dim_feedforward=G3_COMPONENT_FFN_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.residual_head = nn.Linear(G3_COMPONENT_MODEL_DIM, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = components["component_mask"]
        features = components["candidate_features"].detach()
        anchors = components["candidate_anchors"].detach()
        base_logits = components["base_logits"].detach()
        sizes = components["component_sizes"].detach()
        spreads = components["component_spreads_pixels"].detach()
        if mask.shape != base_logits.shape or features.shape[:2] != mask.shape:
            raise ValueError("G3 component ranker input shapes differ")
        geometry = torch.cat(
            (
                anchors.float(),
                base_logits.float().sigmoid()[..., None],
                (sizes.float() / 8.0)[..., None],
                (spreads.float() / 3.0)[..., None],
            ),
            dim=-1,
        )
        hidden = self.feature_projection(self.feature_norm(features.float()))
        hidden = hidden + self.geometry_projection(geometry)
        hidden = self.token_norm(hidden)
        hidden = hidden.masked_fill(~mask[..., None], 0.0)
        hidden = self.set_encoder(hidden, src_key_padding_mask=~mask)
        residual = self.residual_head(hidden).squeeze(-1)
        residual = residual.masked_fill(~mask, 0.0)
        logits = (base_logits.float() + residual).masked_fill(~mask, -10_000.0)
        return {**components, "ranking_residual": residual, "ranking_logits": logits}


@torch.no_grad()
def build_dynamic_component_targets(
    ranked: dict[str, torch.Tensor],
    fixed_hybrid: dict[str, torch.Tensor],
    target_anchors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fix representable GT geometry, then score-match one component per GT."""

    logits = ranked["ranking_logits"]
    anchors = ranked["candidate_anchors"]
    valid = ranked["component_mask"]
    fixed_positive = fixed_hybrid["positive_mask"]
    fixed_indices = fixed_hybrid["matched_target_indices"]
    output_positive = torch.zeros_like(valid)
    output_indices = torch.full_like(fixed_indices, -1)
    output_counts = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    for batch_index in range(len(logits)):
        fixed_targets = fixed_indices[batch_index, fixed_positive[batch_index]]
        fixed_targets = fixed_targets.sort().values
        expected = target_anchors[batch_index, fixed_targets].detach().double().cpu()
        valid_positions = torch.where(valid[batch_index])[0].cpu()
        current_anchors = anchors[batch_index, valid[batch_index]].detach().double().cpu()
        compatible = torch.cdist(current_anchors, expected) <= 6.0 / 768.0
        target_count = len(expected)
        cardinality_penalty = float(target_count + 1)
        distances = torch.cdist(current_anchors, expected)
        stage1_real = distances.T.clone()
        stage1_real[~compatible.T] = 2.0 * cardinality_penalty
        dummy = torch.full(
            (target_count, target_count), cardinality_penalty, dtype=torch.float64
        )
        rows, columns = linear_sum_assignment(
            torch.cat((stage1_real, dummy), dim=1).numpy()
        )
        representable_rows = [
            row
            for row, column in zip(rows.tolist(), columns.tolist())
            if column < len(valid_positions) and bool(compatible[column, row])
        ]
        if not representable_rows:
            raise RuntimeError("G3 component target has no representable GT")
        compatible_representable = compatible[:, representable_rows]
        scores = logits[batch_index, valid[batch_index]].detach().float().cpu()
        component_ids = ranked["candidate_ids"][batch_index, valid[batch_index]].cpu()
        identity_order = component_ids.argsort(stable=True)
        score_order = scores[identity_order].argsort(descending=True, stable=True)
        order = identity_order[score_order]
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(order))
        selected_target_count = len(representable_rows)
        penalty = float(selected_target_count + 1)
        real_cost = ranks.double()[:, None].expand(-1, selected_target_count).clone()
        real_cost /= max(len(valid_positions), 1)
        real_cost[~compatible_representable] = 2.0 * penalty
        dummy = torch.full(
            (selected_target_count, selected_target_count), penalty, dtype=torch.float64
        )
        rows, columns = linear_sum_assignment(
            torch.cat((real_cost.T, dummy), dim=1).numpy()
        )
        assignments = [
            (row, column)
            for row, column in zip(rows.tolist(), columns.tolist())
            if column < len(valid_positions)
            and bool(compatible_representable[column, row])
        ]
        if len(assignments) != selected_target_count:
            raise RuntimeError("G3 component dynamic match reduced target cardinality")
        for row, column in assignments:
            position = int(valid_positions[column])
            output_positive[batch_index, position] = True
            output_indices[batch_index, position] = fixed_targets[
                representable_rows[row]
            ]
        output_counts[batch_index] = selected_target_count
    return {
        "positive_mask": output_positive,
        "matched_target_indices": output_indices,
        "representable_counts": output_counts,
        "component_mask": valid.detach().clone(),
    }


def component_boundary_band_loss(
    ranked: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    boundary_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Apply boundary ranking only to valid components and one-to-one positives."""

    logits = ranked["ranking_logits"]
    valid = ranked["component_mask"]
    positive = targets["positive_mask"]
    if logits.shape != valid.shape or logits.shape != positive.shape:
        raise ValueError("G3 component boundary inputs differ")
    if isinstance(boundary_size, bool) or not isinstance(boundary_size, int) or boundary_size < 1:
        raise ValueError("G3 component boundary size must be a positive integer")
    losses = []
    positive_count = negative_count = 0
    boundary_exact_images = 0
    for batch_index in range(len(logits)):
        current_positive = positive[batch_index]
        current_negative = valid[batch_index] & ~current_positive
        positive_scores = logits[batch_index, current_positive].float()
        negative_scores = logits[batch_index, current_negative].float()
        if not len(positive_scores) or not len(negative_scores):
            raise ValueError("G3 component boundary loss requires both classes")
        weak_count = min(boundary_size, len(positive_scores))
        hard_count = min(boundary_size, len(negative_scores))
        weakest = positive_scores.topk(weak_count, largest=False).values
        strongest = negative_scores.topk(hard_count).values
        losses.append(F.softplus(strongest[:, None] - weakest[None, :]).mean())
        positive_count += len(positive_scores)
        negative_count += len(negative_scores)
        boundary_exact_images += int(
            bool(positive_scores.min() > negative_scores.max())
        )
    loss = torch.stack(losses).mean()
    return {
        "loss": loss,
        "positive_count": loss.new_tensor(float(positive_count)),
        "negative_count": loss.new_tensor(float(negative_count)),
        "boundary_exact_images": loss.new_tensor(float(boundary_exact_images)),
        "images": loss.new_tensor(float(len(logits))),
    }


@torch.no_grad()
def select_ranked_components(
    ranked: dict[str, torch.Tensor], counts: torch.Tensor
) -> dict[str, torch.Tensor]:
    logits = ranked["ranking_logits"]
    valid = ranked["component_mask"]
    if counts.shape != (len(logits),):
        raise ValueError("G3 component counts must be a batch vector")
    if bool((counts > valid.sum(dim=1)).any()):
        raise ValueError("G3 component count exceeds valid components")
    ids = ranked["candidate_ids"]
    identity_order = ids.argsort(dim=1, stable=True)
    score_order = logits.gather(1, identity_order).argsort(
        dim=1, descending=True, stable=True
    )
    order = identity_order.gather(1, score_order)
    selected_mask = torch.arange(128, device=logits.device)[None] < counts[:, None]

    def gather(values):
        index = order.reshape(*order.shape, *([1] * (values.ndim - 2))).expand(
            *order.shape, *values.shape[2:]
        )
        return values.gather(1, index)

    return {
        "atom_counts": counts,
        "atom_mask": selected_mask,
        "atom_anchors": gather(ranked["candidate_anchors"]),
        "atom_component_ids": gather(ids),
        "atom_scores": gather(logits).sigmoid(),
        "component_order": order,
    }
