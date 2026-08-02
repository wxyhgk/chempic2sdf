"""G3 hierarchical image-anchored Atom -> Bond set prediction.

This module intentionally consumes an existing visual feature map.  It owns no
OCR vocabulary, chemical text decoder, SDF writer, or vision backbone.  The
v0.1 public output stops at Bond display/chemical type; direction and stereo
remain outside this structural prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_query import position_encoding_2d
from chemvector.bond_group_query import (
    BondGroupQueryConfig,
    BondGroupQueryDetector,
    bond_group_query_assignments,
    bond_group_query_loss,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS, unordered_endpoint_cost
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    DilatedResidualBlock,
    dense_bond_object_loss,
)
from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    direct_atom_pair_loss,
    enumerate_unordered_atom_pairs,
)
from chemvector.spatial_incidence import (
    SpatialIncidenceConfig,
    SpatialIncidenceHead,
    spatial_incidence_loss,
)
from chemvector.vertex import focal_heatmap_loss


G3_CONFIDENCE_AWARE_BOND_MATCHING_WEIGHT = 2.0


@dataclass(frozen=True, slots=True)
class G3HierarchicalGraphConfig:
    """Dimensions for the backbone-independent G3 graph head."""

    input_channels: int = 96
    hidden_channels: int = 128
    d_model: int = 192
    atom_slots_per_cell: int = 3
    maximum_atoms: int = 128
    bond_slots_per_cell: int = 3
    maximum_bonds: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 6
    decoder_ffn_dim: int = 768
    memory_stride: int = 4
    incidence_hidden_dim: int = 128
    incidence_pair_hidden_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        dimensions = (
            self.input_channels,
            self.hidden_channels,
            self.d_model,
            self.atom_slots_per_cell,
            self.maximum_atoms,
            self.bond_slots_per_cell,
            self.maximum_bonds,
            self.decoder_layers,
            self.decoder_heads,
            self.decoder_ffn_dim,
            self.memory_stride,
            self.incidence_hidden_dim,
            self.incidence_pair_hidden_dim,
        )
        if min(dimensions) < 1:
            raise ValueError("G3 dimensions must be positive")
        if self.d_model % self.decoder_heads:
            raise ValueError("G3 decoder_heads must divide d_model")
        if self.d_model % 4:
            raise ValueError("G3 d_model must be divisible by four")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("G3 dropout must be in [0, 1)")


class SpatialAtomSlotHead(nn.Module):
    """Predict multiple image-conditioned Atom proposals in every spatial cell."""

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(1, config.hidden_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            *(
                DilatedResidualBlock(config.hidden_channels, dilation)
                for dilation in (1, 2, 4, 8)
            )
        )
        slots = config.atom_slots_per_cell
        self.presence = nn.Conv2d(config.hidden_channels, slots, kernel_size=1)
        self.anchor_offset = nn.Conv2d(
            config.hidden_channels, 2 * slots, kernel_size=1
        )
        self.feature_projection = nn.Conv2d(
            config.hidden_channels, config.d_model, kernel_size=1
        )
        self.slot_embedding = nn.Embedding(slots, config.d_model)
        self.slot_norm = nn.LayerNorm(config.d_model)
        self.count = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_channels, config.maximum_atoms + 1),
        )
        nn.init.constant_(self.presence.bias, -4.6)
        nn.init.normal_(self.slot_embedding.weight, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.size(1) != self.config.input_channels:
            raise ValueError(
                "G3 Atom features must have shape [batch, input_channels, h, w]"
            )
        hidden = self.context(self.input_projection(features))
        batch, _, height, width = hidden.shape
        slots = self.config.atom_slots_per_cell
        offsets = self.anchor_offset(hidden).reshape(
            batch, slots, 2, height, width
        ).sigmoid()
        y, x = torch.meshgrid(
            torch.arange(height, device=hidden.device, dtype=hidden.dtype),
            torch.arange(width, device=hidden.device, dtype=hidden.dtype),
            indexing="ij",
        )
        cell_centers = torch.stack(
            (
                (x + 0.5) / float(width),
                (y + 0.5) / float(height),
            ),
            dim=0,
        )
        anchors = torch.stack(
            (
                (x[None, None] + offsets[:, :, 0]) / float(width),
                (y[None, None] + offsets[:, :, 1]) / float(height),
            ),
            dim=2,
        )
        base_features = self.feature_projection(hidden)
        slot_features = (
            base_features[:, None]
            + self.slot_embedding.weight.to(base_features.dtype)[
                None, :, :, None, None
            ]
        )
        slot_features = self.slot_norm(
            slot_features.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3)
        return {
            "object_features": hidden,
            "presence_logits": self.presence(hidden),
            "anchor_offsets": offsets,
            "anchors": anchors,
            "cell_centers": cell_centers,
            "slot_features": slot_features,
            "count_logits": self.count(hidden),
        }


def _gather_flat_candidates(
    values: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    gather = indices.reshape(
        indices.size(0), indices.size(1), *([1] * (values.ndim - 2))
    ).expand(indices.size(0), indices.size(1), *values.shape[2:])
    return values.gather(1, gather)


def _flatten_dense_slot_output(values: torch.Tensor) -> torch.Tensor:
    """Move dense spatial axes beside the slot axis before flattening."""

    if values.ndim < 4:
        raise ValueError("G3 dense slot output must contain slot, height, and width")
    feature_axes = tuple(range(2, values.ndim - 2))
    permutation = (0, 1, values.ndim - 2, values.ndim - 1, *feature_axes)
    return values.permute(permutation).reshape(
        values.size(0),
        values.size(1) * values.size(-2) * values.size(-1),
        *values.shape[2:-2],
    )


def extract_g3_dense_bond_queries(
    dense_output: dict[str, torch.Tensor],
    *,
    maximum_queries: int,
) -> dict[str, torch.Tensor]:
    """Rank raw dense Bond evidence without NMS or geometric candidate deletion."""

    if maximum_queries < 1:
        raise ValueError("G3 Bond query capacity must be positive")
    required = {
        "heatmap_logits",
        "object_features",
        "endpoints",
        "display_logits",
        "chemical_logits",
        "direction_logits",
    }
    missing = sorted(required - dense_output.keys())
    if missing:
        raise ValueError(f"G3 dense Bond output is missing fields: {missing}")
    heatmap_logits = dense_output["heatmap_logits"]
    if heatmap_logits.ndim != 4:
        raise ValueError("G3 dense Bond heatmap must have shape [batch, slots, h, w]")
    batch, slots, height, width = heatmap_logits.shape
    expected_shapes = {
        "endpoints": (batch, slots, 2, 2, height, width),
        "display_logits": (batch, slots, len(BOND_TYPES), height, width),
        "chemical_logits": (batch, slots, len(BOND_TYPES), height, width),
        "direction_logits": (batch, slots, len(DIRECTIONS), height, width),
    }
    for name, expected in expected_shapes.items():
        if dense_output[name].shape != expected:
            raise ValueError(
                f"G3 dense Bond {name} has shape "
                f"{tuple(dense_output[name].shape)}, expected {expected}"
            )
    object_features = dense_output["object_features"]
    if (
        object_features.ndim != 4
        or object_features.size(0) != batch
        or object_features.shape[-2:] != (height, width)
    ):
        raise ValueError("G3 dense Bond features and heatmap differ")

    query_count = min(maximum_queries, slots * height * width)
    base_logits, indices = heatmap_logits.flatten(1).topk(query_count, dim=1)
    spatial_indices = indices.remainder(height * width)
    feature_map = object_features.flatten(2).transpose(1, 2)
    feature_indices = spatial_indices[..., None].expand(
        batch, query_count, feature_map.size(-1)
    )
    slot_indices = indices.div(height * width, rounding_mode="floor")
    y_indices = spatial_indices.div(width, rounding_mode="floor")
    x_indices = spatial_indices.remainder(width)
    return {
        "base_scores": base_logits.sigmoid(),
        "base_logits": base_logits,
        "base_endpoints": _gather_flat_candidates(
            _flatten_dense_slot_output(dense_output["endpoints"]), indices
        ),
        "base_display_logits": _gather_flat_candidates(
            _flatten_dense_slot_output(dense_output["display_logits"]), indices
        ),
        "base_chemical_logits": _gather_flat_candidates(
            _flatten_dense_slot_output(dense_output["chemical_logits"]), indices
        ),
        "base_direction_logits": _gather_flat_candidates(
            _flatten_dense_slot_output(dense_output["direction_logits"]), indices
        ),
        "base_features": feature_map.gather(1, feature_indices),
        "mask": torch.ones_like(base_logits, dtype=torch.bool),
        "indices": torch.stack((slot_indices, y_indices, x_indices), dim=-1),
    }


def build_gt_bond_slot_identity_oracle(
    proposals: dict[str, torch.Tensor],
    target_endpoints: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Route visual proposals into stable GT-prefix slots without replacing them.

    GT geometry is used only to compute a detached permutation. Every proposal
    value consumed by the decoder remains a gathered model prediction, so this
    causal oracle provides object identity but no target geometry or chemistry.
    """

    predicted_endpoints = proposals.get("base_endpoints")
    proposal_mask = proposals.get("mask")
    if predicted_endpoints is None or proposal_mask is None:
        raise ValueError("G3 Bond-slot oracle requires visual endpoints and a mask")
    if predicted_endpoints.ndim != 4 or predicted_endpoints.shape[-2:] != (2, 2):
        raise ValueError("G3 Bond-slot oracle endpoints must have shape [batch, queries, 2, 2]")
    if proposal_mask.shape != predicted_endpoints.shape[:2]:
        raise ValueError("G3 Bond-slot oracle proposal mask and endpoints differ")
    if target_endpoints.ndim != 4 or target_endpoints.shape[-2:] != (2, 2):
        raise ValueError("G3 Bond-slot target endpoints must have shape [batch, bonds, 2, 2]")
    if target_mask.shape != target_endpoints.shape[:2]:
        raise ValueError("G3 Bond-slot target mask and endpoints differ")
    if target_endpoints.size(0) != predicted_endpoints.size(0):
        raise ValueError("G3 Bond-slot oracle proposal and target batches differ")
    target_mask = target_mask.to(predicted_endpoints.device).bool()
    target_endpoints = target_endpoints.to(predicted_endpoints.device)
    expected_prefix = (
        torch.arange(target_mask.size(1), device=target_mask.device)[None]
        < target_mask.sum(dim=1)[:, None]
    )
    if not torch.equal(target_mask, expected_prefix):
        raise ValueError("G3 Bond-slot oracle targets must use canonical prefix order")
    if not bool(torch.isfinite(target_endpoints[target_mask]).all()):
        raise ValueError("G3 Bond-slot oracle target endpoints must be finite")

    batch, queries = predicted_endpoints.shape[:2]
    permutations = []
    target_identity = torch.full(
        (batch, queries), -1, device=predicted_endpoints.device, dtype=torch.long
    )
    positive_mask = torch.zeros(
        batch, queries, device=predicted_endpoints.device, dtype=torch.bool
    )
    matched_counts = torch.zeros(
        batch, device=predicted_endpoints.device, dtype=torch.long
    )
    with torch.no_grad():
        detached_endpoints = predicted_endpoints.detach().float()
        for batch_index in range(batch):
            valid_queries = torch.where(proposal_mask[batch_index].bool())[0]
            count = int(target_mask[batch_index].sum())
            if count > len(valid_queries):
                raise ValueError("G3 Bond-slot oracle targets exceed visual proposals")
            if count:
                expected = target_endpoints[batch_index, :count].detach().float()
                predicted = detached_endpoints[batch_index, valid_queries]
                endpoint_cost = unordered_endpoint_cost(predicted, expected)
                center_cost = torch.cdist(
                    predicted.mean(dim=1), expected.mean(dim=1), p=1
                )
                rows, columns = linear_sum_assignment(
                    (8.0 * endpoint_cost + 2.0 * center_cost).cpu().numpy()
                )
                matched_by_target = torch.empty(
                    count, device=predicted_endpoints.device, dtype=torch.long
                )
                matched_by_target[
                    torch.as_tensor(columns, device=predicted_endpoints.device)
                ] = valid_queries[
                    torch.as_tensor(rows, device=predicted_endpoints.device)
                ]
            else:
                matched_by_target = torch.empty(
                    0, device=predicted_endpoints.device, dtype=torch.long
                )
            matched_set = set(matched_by_target.tolist())
            remainder = torch.tensor(
                [index for index in range(queries) if index not in matched_set],
                device=predicted_endpoints.device,
                dtype=torch.long,
            )
            permutation = torch.cat((matched_by_target, remainder))
            if len(permutation) != queries or len(permutation.unique()) != queries:
                raise RuntimeError("G3 Bond-slot oracle did not build a permutation")
            permutations.append(permutation)
            target_identity[batch_index, :count] = torch.arange(
                count, device=predicted_endpoints.device
            )
            positive_mask[batch_index, :count] = True
            matched_counts[batch_index] = count

    permutation = torch.stack(permutations)
    routed: dict[str, torch.Tensor] = {}
    for name, value in proposals.items():
        routed[name] = (
            _gather_flat_candidates(value, permutation)
            if value.ndim >= 2 and value.shape[:2] == (batch, queries)
            else value
        )
    routed.update(
        {
            "bond_target_indices": target_identity,
            "oracle_positive_mask": positive_mask,
            "oracle_matched_target_count": matched_counts,
            "oracle_target_count": target_mask.sum(dim=1).long(),
            "oracle_full_coverage": matched_counts == target_mask.sum(dim=1),
            "oracle_permutation": permutation,
        }
    )
    return routed


def select_count_conditioned_atom_slots(
    output: dict[str, torch.Tensor],
    counts: torch.Tensor,
    *,
    maximum_atoms: int,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Select exactly ``count`` Atom proposals per image without threshold tuning."""

    logits = output["presence_logits"]
    anchors = output["anchors"]
    features = output["slot_features"]
    if logits.ndim != 4:
        raise ValueError("G3 Atom presence logits must have shape [batch, slots, h, w]")
    batch, slots, height, width = logits.shape
    if anchors.shape != (batch, slots, 2, height, width):
        raise ValueError("G3 Atom anchors and presence logits differ")
    if features.shape[:2] != (batch, slots) or features.shape[-2:] != (
        height,
        width,
    ):
        raise ValueError("G3 Atom slot features and presence logits differ")
    if (
        counts.shape != (batch,)
        or counts.dtype == torch.bool
        or counts.dtype.is_floating_point
        or counts.dtype.is_complex
    ):
        raise ValueError("G3 Atom counts must have shape [batch] and integer dtype")
    counts = counts.long()
    if bool(((counts < 0) | (counts > maximum_atoms)).any()):
        raise ValueError("G3 Atom count is outside configured capacity")
    candidates = slots * height * width
    if maximum_atoms > candidates:
        raise ValueError("G3 Atom capacity exceeds the dense proposal grid")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    if candidate_mask.shape != logits.shape:
        raise ValueError("G3 Atom candidate mask and logits differ")
    valid_counts = candidate_mask.flatten(1).sum(dim=1)
    if bool((counts > valid_counts).any()):
        raise ValueError("G3 Atom count exceeds valid dense candidates")

    ranked_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    selected_logits, indices = ranked_logits.flatten(1).topk(
        maximum_atoms, dim=1
    )
    flat_anchors = anchors.permute(0, 1, 3, 4, 2).reshape(batch, candidates, 2)
    flat_features = features.permute(0, 1, 3, 4, 2).reshape(
        batch, candidates, features.size(2)
    )
    flat_offsets = output["anchor_offsets"].permute(0, 1, 3, 4, 2).reshape(
        batch, candidates, 2
    )
    selected_anchors = _gather_flat_candidates(flat_anchors, indices)
    selected_features = _gather_flat_candidates(flat_features, indices)
    selected_offsets = _gather_flat_candidates(flat_offsets, indices)
    spatial = indices.remainder(height * width)
    slot_indices = indices.div(height * width, rounding_mode="floor")
    y_indices = spatial.div(width, rounding_mode="floor")
    x_indices = spatial.remainder(width)
    cell_centers = torch.stack(
        (
            (x_indices.to(selected_anchors.dtype) + 0.5) / float(width),
            (y_indices.to(selected_anchors.dtype) + 0.5) / float(height),
        ),
        dim=-1,
    )
    slot_scale = float(max(slots - 1, 1))
    slot_fraction = slot_indices.to(selected_anchors.dtype) / slot_scale
    atom_mask = (
        torch.arange(maximum_atoms, device=counts.device)[None] < counts[:, None]
    )
    selected_valid = _gather_flat_candidates(
        candidate_mask.flatten(1), indices
    ).bool()
    if not bool((selected_valid | ~atom_mask).all()):
        raise RuntimeError("G3 selected an invalid Atom inside the count budget")
    atom_geometry = torch.cat(
        (
            selected_anchors,
            cell_centers,
            selected_offsets,
            selected_logits.sigmoid()[..., None],
            slot_fraction[..., None],
        ),
        dim=-1,
    )
    return {
        "atom_counts": counts,
        "atom_mask": atom_mask,
        "atom_presence_logits": selected_logits,
        "atom_scores": selected_logits.sigmoid(),
        "atom_anchors": selected_anchors,
        "atom_features": selected_features,
        "atom_geometry": atom_geometry,
        "atom_slot_indices": torch.stack(
            (slot_indices, y_indices, x_indices), dim=-1
        ),
    }


def build_gt_atom_slot_oracle(
    output: dict[str, torch.Tensor],
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    maximum_atoms: int,
) -> dict[str, torch.Tensor]:
    """Build stable GT-ordered Atom slots with image-conditioned features."""

    logits = output["presence_logits"]
    features = output["slot_features"]
    if logits.ndim != 4:
        raise ValueError("G3 Atom presence logits must have shape [batch, slots, h, w]")
    batch, slots, height, width = logits.shape
    expected_shape = (batch, maximum_atoms)
    if target_anchors.shape != (*expected_shape, 2):
        raise ValueError("G3 oracle Atom anchors have an incompatible shape")
    if target_mask.shape != expected_shape:
        raise ValueError("G3 oracle Atom mask has an incompatible shape")
    target_mask = target_mask.bool()
    if bool((target_mask.sum(dim=1) > maximum_atoms).any()):
        raise ValueError("G3 oracle Atom count exceeds configured capacity")
    valid_anchors = target_anchors[target_mask]
    if not bool(torch.isfinite(valid_anchors).all()):
        raise ValueError("G3 oracle Atom anchors must be finite")
    if bool(((valid_anchors < 0.0) | (valid_anchors > 1.0)).any()):
        raise ValueError("G3 oracle Atom anchors must be normalized")
    if features.shape[:2] != (batch, slots) or features.shape[-2:] != (
        height,
        width,
    ):
        raise ValueError("G3 oracle Atom features and presence logits differ")

    x_indices = (target_anchors[..., 0] * width).long().clamp(0, width - 1)
    y_indices = (target_anchors[..., 1] * height).long().clamp(0, height - 1)
    flat_cells = y_indices * width + x_indices
    order = torch.arange(maximum_atoms, device=target_anchors.device)
    previous = order[None, :] < order[:, None]
    same_cell = flat_cells[:, :, None] == flat_cells[:, None, :]
    slot_indices = (
        same_cell & previous[None] & target_mask[:, None, :]
    ).sum(dim=-1)
    if bool((slot_indices[target_mask] >= slots).any()):
        raise ValueError("G3 oracle Atom slot capacity is smaller than a GT cell")

    batch_indices = torch.arange(batch, device=target_anchors.device)[:, None]
    spatial_features = features.permute(0, 1, 3, 4, 2)
    selected_features = spatial_features[
        batch_indices, slot_indices, y_indices, x_indices
    ]
    selected_logits = logits[batch_indices, slot_indices, y_indices, x_indices]
    cell_centers = torch.stack(
        (
            (x_indices.to(target_anchors.dtype) + 0.5) / float(width),
            (y_indices.to(target_anchors.dtype) + 0.5) / float(height),
        ),
        dim=-1,
    )
    offsets = torch.stack(
        (
            target_anchors[..., 0] * width - x_indices.to(target_anchors.dtype),
            target_anchors[..., 1] * height - y_indices.to(target_anchors.dtype),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    slot_scale = float(max(slots - 1, 1))
    slot_fraction = slot_indices.to(target_anchors.dtype) / slot_scale
    atom_geometry = torch.cat(
        (
            target_anchors,
            cell_centers,
            offsets,
            selected_logits.sigmoid()[..., None],
            slot_fraction[..., None],
        ),
        dim=-1,
    )
    target_indices = order[None].expand(batch, -1)
    invalid_indices = torch.full_like(target_indices, -1)
    target_indices = torch.where(target_mask, target_indices, invalid_indices)
    selected_features = selected_features.masked_fill(~target_mask[..., None], 0.0)
    atom_geometry = atom_geometry.masked_fill(~target_mask[..., None], 0.0)
    selected_logits = selected_logits.masked_fill(~target_mask, -1e4)
    invalid_slots = torch.full_like(slot_indices, -1)
    atom_slot_indices = torch.stack(
        (
            torch.where(target_mask, slot_indices, invalid_slots),
            torch.where(target_mask, y_indices, invalid_slots),
            torch.where(target_mask, x_indices, invalid_slots),
        ),
        dim=-1,
    )
    return {
        "atom_counts": target_mask.sum(dim=1).long(),
        "atom_mask": target_mask,
        "atom_presence_logits": selected_logits,
        "atom_scores": selected_logits.sigmoid(),
        "atom_anchors": target_anchors,
        "atom_features": selected_features,
        "atom_geometry": atom_geometry,
        "atom_slot_indices": atom_slot_indices,
        "atom_target_indices": target_indices,
    }


class AtomConditionedBondQueryDecoder(nn.Module):
    """Refine dense Bond groups using the selected Atom set as decoder memory."""

    GROUP_SIZE = 3

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.base = BondGroupQueryDetector(
            BondGroupQueryConfig(
                input_channels=config.input_channels,
                proposal_channels=config.hidden_channels,
                d_model=config.d_model,
                queries=config.maximum_bonds,
                layers=config.decoder_layers,
                heads=config.decoder_heads,
                ffn_dim=config.decoder_ffn_dim,
                memory_stride=config.memory_stride,
                dropout=config.dropout,
            )
        )
        self.atom_geometry_projection = nn.Sequential(
            nn.Linear(8, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.null_atom = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.atom_conditioner = nn.TransformerDecoder(
            layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.presence_residual = nn.Linear(config.d_model, 1)
        self.endpoint_residual = nn.Linear(config.d_model, 2)
        self.display_residual = nn.Linear(config.d_model, len(BOND_TYPES))
        self.chemical_residual = nn.Linear(config.d_model, len(BOND_TYPES))
        self.incidence = SpatialIncidenceHead(
            SpatialIncidenceConfig(
                atom_feature_dim=config.d_model,
                bond_feature_dim=config.d_model,
                bond_type_dim=len(BOND_TYPES),
                hidden_dim=config.incidence_hidden_dim,
                pair_hidden_dim=config.incidence_pair_hidden_dim,
            )
        )
        nn.init.normal_(self.null_atom, mean=0.0, std=0.02)
        for head in (
            self.presence_residual,
            self.endpoint_residual,
            self.display_residual,
            self.chemical_residual,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        features: torch.Tensor,
        proposals: dict[str, torch.Tensor],
        atom_features: torch.Tensor,
        atom_geometry: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if atom_features.ndim != 3 or atom_features.size(-1) != self.config.d_model:
            raise ValueError("G3 selected Atom features have an incompatible shape")
        if atom_geometry.shape != (*atom_features.shape[:2], 8):
            raise ValueError("G3 selected Atom geometry must contain eight values")
        if atom_mask.shape != atom_features.shape[:2]:
            raise ValueError("G3 selected Atom mask and features differ")

        base = self.base(features, proposals)
        safe_features = atom_features.masked_fill(~atom_mask[..., None], 0.0)
        safe_geometry = atom_geometry.masked_fill(~atom_mask[..., None], 0.0)
        atom_tokens = safe_features + self.atom_geometry_projection(safe_geometry)
        null_atom = self.null_atom.to(atom_tokens.dtype).expand(
            atom_tokens.size(0), -1, -1
        )
        atom_memory = torch.cat((atom_tokens, null_atom), dim=1)
        atom_padding = torch.cat(
            (
                ~atom_mask,
                torch.zeros(
                    atom_mask.size(0),
                    1,
                    device=atom_mask.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        group_tokens = torch.cat(
            (base["object_features"][:, :, None], base["endpoint_features"]),
            dim=2,
        )
        proposal_mask = base["proposal_mask"].bool()
        decoded = self.atom_conditioner(
            group_tokens.flatten(1, 2),
            atom_memory,
            tgt_key_padding_mask=~proposal_mask.repeat_interleave(
                self.GROUP_SIZE, dim=1
            ),
            memory_key_padding_mask=atom_padding,
        ).reshape(
            atom_features.size(0),
            group_tokens.size(1),
            self.GROUP_SIZE,
            self.config.d_model,
        )
        object_hidden = decoded[:, :, 0]
        endpoint_hidden = decoded[:, :, 1:]
        endpoints = (
            base["endpoints"]
            + self.endpoint_residual(endpoint_hidden).tanh()
            / float(max(features.shape[-2:]))
        ).clamp(0.0, 1.0)
        presence_logits = base["presence_logits"] + self.presence_residual(
            object_hidden
        ).squeeze(-1)
        display_logits = base["display_logits"] + self.display_residual(object_hidden)
        chemical_logits = base["chemical_logits"] + self.chemical_residual(
            object_hidden
        )
        incidence = self.incidence(
            safe_features,
            safe_geometry,
            atom_mask,
            object_hidden,
            endpoints,
            chemical_logits.float().softmax(-1).to(object_hidden.dtype),
            proposal_mask,
        )
        return {
            **base,
            "presence_logits": presence_logits,
            "endpoints": endpoints,
            "display_logits": display_logits,
            "chemical_logits": chemical_logits,
            "object_features": object_hidden,
            "endpoint_features": endpoint_hidden,
            "endpoint_pointer_logits": incidence["incidence_logits"],
            "atom_tokens": incidence["atom_tokens"],
        }


class FixedLearnedBondSlotDecoder(nn.Module):
    """Decode canonically ordered Bond slots without proposals or matching."""

    GROUP_SIZE = 3

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_input_projection = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(1, config.hidden_channels),
            nn.GELU(),
        )
        self.visual_context = nn.Sequential(
            *(
                DilatedResidualBlock(config.hidden_channels, dilation)
                for dilation in (1, 2, 4, 8)
            )
        )
        self.visual_memory_projection = nn.Conv2d(
            config.hidden_channels, config.d_model, kernel_size=1
        )
        self.slot_embedding = nn.Embedding(config.maximum_bonds, config.d_model)
        self.role_embedding = nn.Embedding(self.GROUP_SIZE, config.d_model)
        visual_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_decoder = nn.TransformerDecoder(
            visual_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.atom_geometry_projection = nn.Sequential(
            nn.Linear(8, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        atom_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.atom_conditioner = nn.TransformerDecoder(
            atom_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.endpoint_head = nn.Linear(config.d_model, 2)
        self.display_head = nn.Linear(config.d_model, len(BOND_TYPES))
        self.chemical_head = nn.Linear(config.d_model, len(BOND_TYPES))
        self.incidence = SpatialIncidenceHead(
            SpatialIncidenceConfig(
                atom_feature_dim=config.d_model,
                bond_feature_dim=config.d_model,
                bond_type_dim=len(BOND_TYPES),
                hidden_dim=config.incidence_hidden_dim,
                pair_hidden_dim=config.incidence_pair_hidden_dim,
            )
        )
        nn.init.normal_(self.slot_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        atom_features: torch.Tensor,
        atom_geometry: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.size(1) != self.config.input_channels:
            raise ValueError("G3 P0-E visual features have an incompatible shape")
        if atom_features.ndim != 3 or atom_features.size(-1) != self.config.d_model:
            raise ValueError("G3 P0-E Atom features have an incompatible shape")
        if atom_geometry.shape != (*atom_features.shape[:2], 8):
            raise ValueError("G3 P0-E Atom geometry must contain eight values")
        if atom_mask.shape != atom_features.shape[:2]:
            raise ValueError("G3 P0-E Atom mask and features differ")
        if not bool(atom_mask.any(dim=1).all()):
            raise ValueError("G3 P0-E requires at least one GT Atom in every image")

        visual_map = self.visual_memory_projection(
            self.visual_context(self.visual_input_projection(features))
        )
        if self.config.memory_stride > 1:
            visual_map = F.avg_pool2d(
                visual_map,
                kernel_size=self.config.memory_stride,
                stride=self.config.memory_stride,
                ceil_mode=True,
            )
        height, width = visual_map.shape[-2:]
        visual_memory = visual_map.flatten(2).transpose(1, 2)
        visual_memory = visual_memory + position_encoding_2d(
            height,
            width,
            self.config.d_model,
            visual_memory.device,
            visual_memory.dtype,
        ).unsqueeze(0)

        slots = self.slot_embedding.weight[:, None, :]
        roles = self.role_embedding.weight[None, :, :]
        group_tokens = (slots + roles).reshape(
            1, self.config.maximum_bonds * self.GROUP_SIZE, self.config.d_model
        )
        group_tokens = group_tokens.to(visual_memory.dtype).expand(
            features.size(0), -1, -1
        )
        visual_hidden = self.visual_decoder(group_tokens, visual_memory)

        safe_atom_features = atom_features.masked_fill(~atom_mask[..., None], 0.0)
        safe_atom_geometry = atom_geometry.masked_fill(~atom_mask[..., None], 0.0)
        atom_memory = safe_atom_features + self.atom_geometry_projection(
            safe_atom_geometry
        )
        decoded = self.atom_conditioner(
            visual_hidden,
            atom_memory,
            memory_key_padding_mask=~atom_mask,
        ).reshape(
            features.size(0),
            self.config.maximum_bonds,
            self.GROUP_SIZE,
            self.config.d_model,
        )
        object_hidden = decoded[:, :, 0]
        endpoint_hidden = decoded[:, :, 1:]
        endpoints = self.endpoint_head(endpoint_hidden).sigmoid()
        chemical_logits = self.chemical_head(object_hidden)
        slot_mask = torch.ones(
            object_hidden.shape[:2], device=features.device, dtype=torch.bool
        )
        incidence = self.incidence(
            safe_atom_features,
            safe_atom_geometry,
            atom_mask,
            object_hidden,
            endpoints,
            chemical_logits.float().softmax(-1).to(object_hidden.dtype),
            slot_mask,
        )
        return {
            "endpoints": endpoints,
            "display_logits": self.display_head(object_hidden),
            "chemical_logits": chemical_logits,
            "endpoint_pointer_logits": incidence["incidence_logits"],
            "object_features": object_hidden,
            "endpoint_features": endpoint_hidden,
            "atom_tokens": incidence["atom_tokens"],
            "visual_features": visual_map,
            "slot_indices": torch.arange(
                self.config.maximum_bonds, device=features.device
            )[None].expand(features.size(0), -1),
        }


def _count_conditioned_mask(
    logits: torch.Tensor, counts: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if logits.shape != valid_mask.shape or counts.shape != logits.shape[:1]:
        raise ValueError("G3 count-conditioned mask inputs differ")
    invalid_dtype = (
        counts.dtype == torch.bool
        or counts.dtype.is_floating_point
        or counts.dtype.is_complex
    )
    if invalid_dtype:
        raise ValueError("G3 query counts must use an integer dtype")
    counts = counts.long()
    if bool(((counts < 0) | (counts > valid_mask.sum(dim=1))).any()):
        raise ValueError("G3 count exceeds valid query capacity")
    order = logits.masked_fill(~valid_mask, -torch.inf).argsort(
        dim=1, descending=True
    )
    ranks = torch.empty_like(order)
    rank_values = torch.arange(logits.size(1), device=logits.device)[None].expand_as(
        order
    )
    ranks.scatter_(1, order, rank_values)
    return valid_mask & (ranks < counts[:, None])


class G3HierarchicalGraphPredictor(nn.Module):
    """Backbone-independent G3 head with image-anchored object identities."""

    model_type = "g3-hierarchical-graph-v0.1"

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_head = SpatialAtomSlotHead(config)
        self.dense_bond = DenseBondObjectDetector(
            DenseBondObjectConfig(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                slots=config.bond_slots_per_cell,
            )
        )
        self.bond_decoder = AtomConditionedBondQueryDecoder(config)
        self.bond_count = nn.Sequential(
            nn.LayerNorm(config.hidden_channels + config.d_model),
            nn.Linear(
                config.hidden_channels + config.d_model,
                config.decoder_ffn_dim,
            ),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, config.maximum_bonds + 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_counts: torch.Tensor | None = None,
        bond_counts: torch.Tensor | None = None,
        atom_candidate_mask: torch.Tensor | None = None,
        oracle_atom_anchors: torch.Tensor | None = None,
        oracle_atom_mask: torch.Tensor | None = None,
        oracle_bond_endpoints: torch.Tensor | None = None,
        oracle_bond_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Predict graph objects, optionally teacher-forcing only object counts."""

        atom_dense = self.atom_head(features)
        if (oracle_atom_anchors is None) != (oracle_atom_mask is None):
            raise ValueError("G3 oracle Atom anchors and mask must be provided together")
        selected_atom_counts = (
            atom_dense["count_logits"].argmax(dim=-1)
            if atom_counts is None
            else atom_counts.to(features.device)
        )
        if oracle_atom_anchors is None:
            atoms = select_count_conditioned_atom_slots(
                atom_dense,
                selected_atom_counts,
                maximum_atoms=self.config.maximum_atoms,
                candidate_mask=atom_candidate_mask,
            )
        else:
            if atom_candidate_mask is not None:
                raise ValueError("G3 oracle Atom slots do not accept a candidate mask")
            oracle_atom_anchors = oracle_atom_anchors.to(features.device)
            oracle_atom_mask = oracle_atom_mask.to(features.device)
            oracle_counts = oracle_atom_mask.sum(dim=1).long()
            if atom_counts is not None and not torch.equal(
                selected_atom_counts.long(), oracle_counts
            ):
                raise ValueError("G3 oracle Atom count differs from teacher-forced count")
            atoms = build_gt_atom_slot_oracle(
                atom_dense,
                oracle_atom_anchors,
                oracle_atom_mask,
                maximum_atoms=self.config.maximum_atoms,
            )

        dense_bond = self.dense_bond(features)
        proposals = extract_g3_dense_bond_queries(
            dense_bond, maximum_queries=self.config.maximum_bonds
        )
        if (oracle_bond_endpoints is None) != (oracle_bond_mask is None):
            raise ValueError("G3 oracle Bond endpoints and mask must be provided together")
        if oracle_bond_endpoints is not None:
            if oracle_atom_anchors is None:
                raise ValueError("G3 oracle Bond slots require oracle Atom slots")
            proposals = build_gt_bond_slot_identity_oracle(
                proposals,
                oracle_bond_endpoints,
                oracle_bond_mask,
            )
        bonds = self.bond_decoder(
            features,
            proposals,
            atoms["atom_features"],
            atoms["atom_geometry"],
            atoms["atom_mask"],
        )
        if oracle_bond_endpoints is not None:
            for name in (
                "bond_target_indices",
                "oracle_positive_mask",
                "oracle_matched_target_count",
                "oracle_target_count",
                "oracle_full_coverage",
                "oracle_permutation",
            ):
                bonds[name] = proposals[name]
        atom_weights = atoms["atom_mask"][..., None].to(
            atoms["atom_features"].dtype
        )
        pooled_atoms = (
            atoms["atom_features"] * atom_weights
        ).sum(dim=1) / atom_weights.sum(dim=1).clamp_min(1.0)
        pooled_visual = atom_dense["object_features"].mean(dim=(-2, -1))
        bond_count_logits = self.bond_count(
            torch.cat((pooled_visual, pooled_atoms), dim=-1)
        )
        selected_bond_counts = (
            bond_count_logits.argmax(dim=-1)
            if bond_counts is None
            else bond_counts.to(features.device)
        )
        if oracle_bond_endpoints is None:
            bond_mask = _count_conditioned_mask(
                bonds["presence_logits"],
                selected_bond_counts,
                bonds["proposal_mask"].bool(),
            )
        else:
            oracle_counts = oracle_bond_mask.to(features.device).sum(dim=1).long()
            if bond_counts is not None and not torch.equal(
                selected_bond_counts.long(), oracle_counts
            ):
                raise ValueError("G3 oracle Bond count differs from teacher-forced count")
            bond_mask = bonds["proposal_mask"].bool() & (
                torch.arange(self.config.maximum_bonds, device=features.device)[None]
                < selected_bond_counts[:, None]
            )
            if not bool((bond_mask.sum(dim=1) == selected_bond_counts).all()):
                raise ValueError("G3 oracle Bond prefix exceeds valid proposals")
        pointer_logits = bonds["endpoint_pointer_logits"].masked_fill(
            ~bond_mask[:, :, None, None], -1e4
        )
        return {
            **atoms,
            "atom_count_logits": atom_dense["count_logits"],
            "bond_counts": selected_bond_counts.long(),
            "bond_count_logits": bond_count_logits,
            "bond_mask": bond_mask,
            "bond_presence_logits": bonds["presence_logits"],
            "bond_endpoints": bonds["endpoints"],
            "bond_type_logits": bonds["chemical_logits"],
            "bond_display_logits": bonds["display_logits"],
            "bond_features": bonds["object_features"],
            "endpoint_pointer_logits": pointer_logits,
            "atom_dense_output": atom_dense,
            "bond_dense_output": dense_bond,
            "bond_query_output": bonds,
        }


class G3AtomOnlyPredictor(nn.Module):
    """P0-H Atom capacity model with no Bond modules or oracle inputs."""

    model_type = "g3-atom-only-capacity-p0h-v0.1"

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_head = SpatialAtomSlotHead(config)

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_counts: torch.Tensor | None = None,
        atom_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        atom_dense = self.atom_head(features)
        selected_counts = (
            atom_dense["count_logits"].argmax(dim=-1)
            if atom_counts is None
            else atom_counts.to(features.device).long()
        )
        atoms = select_count_conditioned_atom_slots(
            atom_dense,
            selected_counts,
            maximum_atoms=self.config.maximum_atoms,
            candidate_mask=atom_candidate_mask,
        )
        return {
            **atoms,
            "atom_count_logits": atom_dense["count_logits"],
            "atom_dense_output": atom_dense,
        }


class G3FixedLearnedBondSlotPredictor(nn.Module):
    """P0-E fixed Bond slots with direct canonical-prefix supervision."""

    model_type = "g3-fixed-learned-bond-slots-p0e-v0.1"

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_head = SpatialAtomSlotHead(config)
        self.bond_decoder = FixedLearnedBondSlotDecoder(config)
        self.bond_count = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, config.maximum_bonds + 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_counts: torch.Tensor | None = None,
        bond_counts: torch.Tensor | None = None,
        oracle_atom_anchors: torch.Tensor | None = None,
        oracle_atom_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Decode Bond prefix slots; GT Bond labels never enter this forward path."""

        if oracle_atom_anchors is None or oracle_atom_mask is None:
            raise ValueError("G3 P0-E requires GT Atom-slot oracle inputs")
        atom_dense = self.atom_head(features)
        oracle_atom_anchors = oracle_atom_anchors.to(features.device)
        oracle_atom_mask = oracle_atom_mask.to(features.device).bool()
        oracle_counts = oracle_atom_mask.sum(dim=1).long()
        selected_atom_counts = (
            atom_dense["count_logits"].argmax(dim=-1)
            if atom_counts is None
            else atom_counts.to(features.device).long()
        )
        if atom_counts is not None and not torch.equal(
            selected_atom_counts, oracle_counts
        ):
            raise ValueError("G3 P0-E oracle Atom count differs from forced count")
        atoms = build_gt_atom_slot_oracle(
            atom_dense,
            oracle_atom_anchors,
            oracle_atom_mask,
            maximum_atoms=self.config.maximum_atoms,
        )
        bonds = self.bond_decoder(
            features,
            atoms["atom_features"],
            atoms["atom_geometry"],
            atoms["atom_mask"],
        )
        atom_weights = atoms["atom_mask"][..., None].to(
            atoms["atom_features"].dtype
        )
        pooled_atoms = (
            atoms["atom_features"] * atom_weights
        ).sum(dim=1) / atom_weights.sum(dim=1).clamp_min(1.0)
        pooled_visual = bonds["visual_features"].mean(dim=(-2, -1))
        bond_count_logits = self.bond_count(
            torch.cat((pooled_visual, pooled_atoms), dim=-1)
        )
        selected_bond_counts = (
            bond_count_logits.argmax(dim=-1)
            if bond_counts is None
            else bond_counts.to(features.device).long()
        )
        if bool(
            ((selected_bond_counts < 0) |
             (selected_bond_counts > self.config.maximum_bonds)).any()
        ):
            raise ValueError("G3 P0-E Bond count exceeds fixed slot capacity")
        bond_mask = torch.arange(
            self.config.maximum_bonds, device=features.device
        )[None] < selected_bond_counts[:, None]
        pointer_logits = bonds["endpoint_pointer_logits"].masked_fill(
            ~bond_mask[:, :, None, None], -1e4
        )
        return {
            **atoms,
            "atom_counts": selected_atom_counts,
            "atom_count_logits": atom_dense["count_logits"],
            "bond_counts": selected_bond_counts,
            "bond_count_logits": bond_count_logits,
            "bond_mask": bond_mask,
            "bond_endpoints": bonds["endpoints"],
            "bond_type_logits": bonds["chemical_logits"],
            "bond_display_logits": bonds["display_logits"],
            "bond_features": bonds["object_features"],
            "endpoint_pointer_logits": pointer_logits,
            "atom_dense_output": atom_dense,
            "fixed_bond_slot_output": bonds,
        }


class G3SetLearnedBondSlotPredictor(nn.Module):
    """P0-F permutation-invariant set prediction over fixed learned Bond slots."""

    model_type = "g3-set-learned-bond-slots-p0f-v0.1"

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.base = G3FixedLearnedBondSlotPredictor(config)
        self.presence_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_counts: torch.Tensor | None = None,
        bond_counts: torch.Tensor | None = None,
        oracle_atom_anchors: torch.Tensor | None = None,
        oracle_atom_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Decode a Bond set; GT Bond labels never enter this forward path."""

        base = self.base(
            features,
            atom_counts=atom_counts,
            bond_counts=bond_counts,
            oracle_atom_anchors=oracle_atom_anchors,
            oracle_atom_mask=oracle_atom_mask,
        )
        slots = base["fixed_bond_slot_output"]
        presence_logits = self.presence_head(slots["object_features"]).squeeze(-1)
        valid = torch.ones_like(presence_logits, dtype=torch.bool)
        bond_mask = _count_conditioned_mask(
            presence_logits,
            base["bond_counts"],
            valid,
        )
        pointer_logits = slots["endpoint_pointer_logits"].masked_fill(
            ~bond_mask[:, :, None, None], -1e4
        )
        result = {
            name: value
            for name, value in base.items()
            if name not in {
                "bond_mask",
                "bond_endpoints",
                "bond_type_logits",
                "bond_display_logits",
                "bond_features",
                "endpoint_pointer_logits",
                "fixed_bond_slot_output",
            }
        }
        return {
            **result,
            "bond_mask": bond_mask,
            "bond_presence_logits": presence_logits,
            "bond_endpoints": slots["endpoints"],
            "bond_type_logits": slots["chemical_logits"],
            "bond_display_logits": slots["display_logits"],
            "bond_features": slots["object_features"],
            "endpoint_pointer_logits": pointer_logits,
            "set_bond_slot_output": {
                **slots,
                "presence_logits": presence_logits,
            },
        }


class G3AtomPairBondIdentityPredictor(nn.Module):
    """P0-G Bond prediction with each unordered AtomPair as a stable identity."""

    model_type = "g3-atom-pair-bond-identity-p0g-v0.1"

    def __init__(self, config: G3HierarchicalGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_head = SpatialAtomSlotHead(config)
        self.pair_bond = DirectAtomPairBond(
            DirectAtomPairBondConfig(
                feature_channels=config.input_channels,
                atom_feature_dim=config.d_model,
                longitudinal_samples=32,
                transverse_samples=9,
                ribbon_radius_pixels=12.0,
                image_size=768.0,
                hidden_dim=256,
                pair_chunk_size=512,
            )
        )
        self.bond_count = nn.Sequential(
            nn.LayerNorm(config.input_channels + config.d_model),
            nn.Linear(
                config.input_channels + config.d_model,
                config.decoder_ffn_dim,
            ),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, config.maximum_bonds + 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_counts: torch.Tensor | None = None,
        oracle_atom_anchors: torch.Tensor | None = None,
        oracle_atom_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Enumerate all AtomPairs; GT Bond labels never enter this path."""

        if oracle_atom_anchors is None or oracle_atom_mask is None:
            raise ValueError("G3 P0-G requires GT Atom-slot oracle inputs")
        atom_dense = self.atom_head(features)
        oracle_atom_anchors = oracle_atom_anchors.to(features.device)
        oracle_atom_mask = oracle_atom_mask.to(features.device).bool()
        oracle_counts = oracle_atom_mask.sum(dim=1).long()
        selected_atom_counts = (
            atom_dense["count_logits"].argmax(dim=-1)
            if atom_counts is None
            else atom_counts.to(features.device).long()
        )
        if atom_counts is not None and not torch.equal(
            selected_atom_counts, oracle_counts
        ):
            raise ValueError("G3 P0-G oracle Atom count differs from forced count")
        atoms = build_gt_atom_slot_oracle(
            atom_dense,
            oracle_atom_anchors,
            oracle_atom_mask,
            maximum_atoms=self.config.maximum_atoms,
        )
        pair_indices, sample_indices = enumerate_unordered_atom_pairs(
            atoms["atom_mask"]
        )
        flat_pairs = self.pair_bond(
            features,
            atoms["atom_features"],
            atoms["atom_anchors"],
            atoms["atom_scores"],
            pair_indices,
            sample_indices,
        )
        pair_counts = torch.bincount(
            sample_indices,
            minlength=features.size(0),
        )
        maximum_pairs = int(pair_counts.max()) if len(pair_counts) else 0
        if maximum_pairs < 1:
            raise ValueError("G3 P0-G requires at least one AtomPair per image")
        batch = features.size(0)
        packed_indices = torch.zeros(
            batch,
            maximum_pairs,
            2,
            device=features.device,
            dtype=torch.long,
        )
        pair_mask = torch.zeros(
            batch, maximum_pairs, device=features.device, dtype=torch.bool
        )
        presence_logits = features.new_full((batch, maximum_pairs), -1e4)
        chemical_logits = features.new_zeros(
            batch, maximum_pairs, len(BOND_TYPES)
        )
        for batch_index in range(batch):
            rows = torch.where(sample_indices == batch_index)[0]
            count = len(rows)
            packed_indices[batch_index, :count] = pair_indices[rows]
            pair_mask[batch_index, :count] = True
            presence_logits[batch_index, :count] = flat_pairs["presence_logits"][rows]
            chemical_logits[batch_index, :count] = flat_pairs["chemical_logits"][rows]

        atom_weights = atoms["atom_mask"][..., None].to(
            atoms["atom_features"].dtype
        )
        pooled_atoms = (
            atoms["atom_features"] * atom_weights
        ).sum(dim=1) / atom_weights.sum(dim=1).clamp_min(1.0)
        pooled_visual = features.mean(dim=(-2, -1))
        bond_count_logits = self.bond_count(
            torch.cat((pooled_visual, pooled_atoms), dim=-1)
        )
        selected_bond_counts = torch.minimum(
            bond_count_logits.argmax(dim=-1),
            pair_counts,
        )
        bond_mask = _count_conditioned_mask(
            presence_logits,
            selected_bond_counts,
            pair_mask,
        )

        gather = packed_indices[..., None].expand(-1, -1, -1, 2)
        expanded_anchors = atoms["atom_anchors"][:, None].expand(
            -1, maximum_pairs, -1, -1
        )
        bond_endpoints = torch.gather(expanded_anchors, 2, gather)
        pointer_logits = features.new_full(
            (
                batch,
                maximum_pairs,
                2,
                self.config.maximum_atoms,
            ),
            -1e4,
        )
        pointer_logits.scatter_(
            -1,
            packed_indices[..., None],
            torch.full(
                (*packed_indices.shape, 1),
                1e4,
                device=features.device,
                dtype=features.dtype,
            ),
        )
        selected_pointer_logits = pointer_logits.masked_fill(
            ~bond_mask[:, :, None, None], -1e4
        )
        return {
            **atoms,
            "atom_counts": selected_atom_counts,
            "atom_count_logits": atom_dense["count_logits"],
            "bond_counts": selected_bond_counts,
            "bond_count_logits": bond_count_logits,
            "bond_mask": bond_mask,
            "bond_presence_logits": presence_logits,
            "bond_endpoints": bond_endpoints,
            "bond_type_logits": chemical_logits,
            "bond_display_logits": chemical_logits,
            "endpoint_pointer_logits": selected_pointer_logits,
            "atom_pair_indices": packed_indices,
            "atom_pair_mask": pair_mask,
            "flat_atom_pair_indices": pair_indices,
            "flat_atom_pair_sample_indices": sample_indices,
            "flat_atom_pair_output": flat_pairs,
            "atom_pair_pointer_logits": pointer_logits,
            "atom_dense_output": atom_dense,
        }


def select_g3_atom_pair_bonds(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    bond_counts: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Apply GT- or deploy-count top-K outside the P0-G model forward path."""

    counts = bond_counts.to(output["bond_presence_logits"].device).long()
    mask = _count_conditioned_mask(
        output["bond_presence_logits"],
        counts,
        output["atom_pair_mask"].bool(),
    )
    selected = dict(output)
    selected["bond_counts"] = counts
    selected["bond_mask"] = mask
    selected["endpoint_pointer_logits"] = output[
        "atom_pair_pointer_logits"
    ].masked_fill(~mask[:, :, None, None], -1e4)
    return selected


def _build_atom_slot_targets(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    logits = output["presence_logits"]
    anchors = output["anchors"]
    target_anchors = targets["atom_anchors"]
    target_mask = targets["atom_mask"].bool()
    if target_anchors.ndim != 3 or target_anchors.shape[-1] != 2:
        raise ValueError("G3 target Atom anchors must have shape [batch, atoms, 2]")
    if target_mask.shape != target_anchors.shape[:2] or logits.size(0) != len(
        target_anchors
    ):
        raise ValueError("G3 target Atom anchors and masks differ")
    if not bool(torch.isfinite(target_anchors[target_mask]).all()):
        raise ValueError("G3 target Atom anchors must be finite")
    outside_canvas = (target_anchors[target_mask] < 0.0) | (
        target_anchors[target_mask] > 1.0
    )
    if bool(outside_canvas.any()):
        raise ValueError("G3 target Atom anchors must be normalized")

    batch, slots, height, width = logits.shape
    heatmap = torch.zeros_like(logits)
    expected_anchors = torch.zeros_like(anchors)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    occupancy = torch.zeros(
        batch, height, width, device=logits.device, dtype=torch.long
    )
    for batch_index in range(batch):
        cells: dict[tuple[int, int], list[int]] = {}
        for target_index in torch.where(target_mask[batch_index])[0].tolist():
            anchor = target_anchors[batch_index, target_index]
            cell_x = min(width - 1, max(0, int(float(anchor[0]) * width)))
            cell_y = min(height - 1, max(0, int(float(anchor[1]) * height)))
            cells.setdefault((cell_y, cell_x), []).append(target_index)
        for (cell_y, cell_x), indices in cells.items():
            indices.sort(
                key=lambda index: tuple(
                    target_anchors[batch_index, index].detach().float().cpu().tolist()
                )
            )
            count = len(indices)
            occupancy[batch_index, cell_y, cell_x] = count
            if count > slots:
                raise ValueError(
                    f"G3 Atom slot capacity {slots} is smaller than {count} "
                    f"Atoms in cell ({cell_x}, {cell_y})"
                )
            predicted = anchors[
                batch_index, :, :, cell_y, cell_x
            ].detach().float()
            expected = target_anchors[batch_index, indices].float()
            rows, columns = linear_sum_assignment(
                torch.cdist(predicted, expected, p=1).cpu().numpy()
            )
            for row, column in zip(rows, columns):
                target_index = indices[column]
                heatmap[batch_index, row, cell_y, cell_x] = 1.0
                positive[batch_index, row, cell_y, cell_x] = True
                expected_anchors[
                    batch_index, row, :, cell_y, cell_x
                ] = target_anchors[batch_index, target_index]
    return {
        "heatmap": heatmap,
        "anchors": expected_anchors,
        "positive": positive,
        "occupancy": occupancy,
    }


def build_canonical_atom_slot_targets(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Assign GT Atoms directly to stable coordinate-ordered cell slots."""

    logits = output["presence_logits"]
    anchors = output["anchors"]
    target_anchors = targets["atom_anchors"]
    target_mask = targets["atom_mask"].bool()
    if target_anchors.ndim != 3 or target_anchors.shape[-1] != 2:
        raise ValueError("G3 target Atom anchors must have shape [batch, atoms, 2]")
    if target_mask.shape != target_anchors.shape[:2] or logits.size(0) != len(
        target_anchors
    ):
        raise ValueError("G3 target Atom anchors and masks differ")
    if not bool(torch.isfinite(target_anchors[target_mask]).all()):
        raise ValueError("G3 target Atom anchors must be finite")
    if bool(
        (
            (target_anchors[target_mask] < 0.0)
            | (target_anchors[target_mask] > 1.0)
        ).any()
    ):
        raise ValueError("G3 target Atom anchors must be normalized")

    batch, slots, height, width = logits.shape
    if anchors.shape != (batch, slots, 2, height, width):
        raise ValueError("G3 Atom anchors and presence logits differ")
    heatmap = torch.zeros_like(logits)
    expected_anchors = torch.zeros_like(anchors)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    occupancy = torch.zeros(
        batch, height, width, device=logits.device, dtype=torch.long
    )
    for batch_index in range(batch):
        cells: dict[tuple[int, int], list[int]] = {}
        for target_index in torch.where(target_mask[batch_index])[0].tolist():
            anchor = target_anchors[batch_index, target_index]
            cell_x = min(width - 1, max(0, int(float(anchor[0]) * width)))
            cell_y = min(height - 1, max(0, int(float(anchor[1]) * height)))
            cells.setdefault((cell_y, cell_x), []).append(target_index)
        for (cell_y, cell_x), indices in cells.items():
            coordinates = [
                tuple(
                    target_anchors[batch_index, index]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                for index in indices
            ]
            if len(set(coordinates)) != len(coordinates):
                raise ValueError(
                    "G3 canonical Atom cell slots reject identical GT coordinates"
                )
            indices.sort(
                key=lambda index: tuple(
                    target_anchors[batch_index, index]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
            )
            count = len(indices)
            occupancy[batch_index, cell_y, cell_x] = count
            if count > slots:
                raise ValueError(
                    f"G3 Atom slot capacity {slots} is smaller than {count} "
                    f"Atoms in cell ({cell_x}, {cell_y})"
                )
            for slot, target_index in enumerate(indices):
                heatmap[batch_index, slot, cell_y, cell_x] = 1.0
                positive[batch_index, slot, cell_y, cell_x] = True
                expected_anchors[
                    batch_index, slot, :, cell_y, cell_x
                ] = target_anchors[batch_index, target_index]
    return {
        "heatmap": heatmap,
        "anchors": expected_anchors,
        "positive": positive,
        "occupancy": occupancy,
    }


def spatial_atom_slot_loss(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Supervise multi-slot Atom evidence and the explicit Atom count."""

    dense = _build_atom_slot_targets(output, targets)
    heatmap_loss = focal_heatmap_loss(output["presence_logits"], dense["heatmap"])
    positive = dense["positive"]
    zero = heatmap_loss * 0.0
    if bool(positive.any()):
        predicted = output["anchors"].permute(0, 1, 3, 4, 2)[positive]
        expected = dense["anchors"].permute(0, 1, 3, 4, 2)[positive]
        anchor_loss = F.smooth_l1_loss(predicted, expected)
    else:
        anchor_loss = zero
    count_targets = targets["atom_mask"].sum(dim=1).long()
    if bool((count_targets >= output["count_logits"].size(-1)).any()):
        raise ValueError("G3 target Atom count exceeds count-head capacity")
    count_loss = F.cross_entropy(output["count_logits"], count_targets)
    return {
        "loss": heatmap_loss + 8.0 * anchor_loss + 0.1 * count_loss,
        "heatmap_loss": heatmap_loss,
        "anchor_loss": anchor_loss,
        "count_loss": count_loss,
    }


def canonical_spatial_atom_slot_loss(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Apply the P0 Atom objective with canonical cell-slot targets."""

    dense = build_canonical_atom_slot_targets(output, targets)
    heatmap_loss = focal_heatmap_loss(output["presence_logits"], dense["heatmap"])
    positive = dense["positive"]
    zero = heatmap_loss * 0.0
    if bool(positive.any()):
        predicted = output["anchors"].permute(0, 1, 3, 4, 2)[positive]
        expected = dense["anchors"].permute(0, 1, 3, 4, 2)[positive]
        anchor_loss = F.smooth_l1_loss(predicted, expected)
    else:
        anchor_loss = zero
    count_targets = targets["atom_mask"].sum(dim=1).long()
    if bool((count_targets >= output["count_logits"].size(-1)).any()):
        raise ValueError("G3 target Atom count exceeds count-head capacity")
    count_loss = F.cross_entropy(output["count_logits"], count_targets)
    return {
        "loss": heatmap_loss + 8.0 * anchor_loss + 0.1 * count_loss,
        "heatmap_loss": heatmap_loss,
        "anchor_loss": anchor_loss,
        "count_loss": count_loss,
    }


def g3_atom_only_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply the unchanged P0 spatial Atom objective and nothing else."""

    dense_output = output.get("atom_dense_output")
    if not isinstance(dense_output, dict):
        raise ValueError("G3 P0-H output is missing the dense Atom head")
    return spatial_atom_slot_loss(dense_output, targets)


def g3_canonical_atom_only_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply only canonical cell-slot supervision for the P0-I branch."""

    dense_output = output.get("atom_dense_output")
    if not isinstance(dense_output, dict):
        raise ValueError("G3 P0-I output is missing the dense Atom head")
    return canonical_spatial_atom_slot_loss(dense_output, targets)


def _atom_slot_assignments(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    assignments = []
    target_identity = output.get("atom_target_indices")
    if target_identity is not None:
        if target_identity.shape != output["atom_mask"].shape:
            raise ValueError("G3 oracle Atom target identity has an incompatible shape")
        for batch_index in range(len(target_identity)):
            queries = torch.where(output["atom_mask"][batch_index])[0]
            expected = target_identity[batch_index, queries].long()
            target_mask = targets["atom_mask"][batch_index].bool()
            if (
                bool((expected < 0).any())
                or len(expected.unique()) != len(expected)
                or not bool(target_mask[expected].all())
                or set(expected.tolist()) != set(torch.where(target_mask)[0].tolist())
            ):
                raise ValueError("G3 oracle Atom identity differs from target slots")
            expected_anchors = targets["atom_anchors"][batch_index, expected]
            if not torch.equal(
                output["atom_anchors"][batch_index, queries], expected_anchors
            ):
                raise ValueError("G3 oracle Atom anchors differ from target slots")
            assignments.append((queries, expected))
        return assignments
    for batch_index in range(output["atom_anchors"].size(0)):
        queries = torch.where(output["atom_mask"][batch_index])[0]
        expected_indices = torch.where(targets["atom_mask"][batch_index])[0].tolist()
        expected_indices.sort(
            key=lambda index: tuple(
                targets["atom_anchors"][batch_index, index]
                .detach()
                .float()
                .cpu()
                .tolist()
            )
        )
        expected = torch.tensor(
            expected_indices,
            device=output["atom_anchors"].device,
            dtype=torch.long,
        )
        if not len(queries) or not len(expected):
            empty = torch.empty(
                0, device=output["atom_anchors"].device, dtype=torch.long
            )
            assignments.append((empty, empty))
            continue
        cost = torch.cdist(
            output["atom_anchors"][batch_index, queries].detach().float(),
            targets["atom_anchors"][batch_index, expected].float(),
            p=1,
        )
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        assignments.append(
            (
                queries[torch.as_tensor(rows, device=queries.device)],
                expected[torch.as_tensor(columns, device=expected.device)],
            )
        )
    return assignments


def build_g3_bond_targets(
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize padded graph targets for the Bond group loss."""

    atom_refs = targets["bond_atom_refs"].long()
    bond_mask = targets["bond_mask"].bool()
    if atom_refs.shape != (*bond_mask.shape, 2):
        raise ValueError("G3 target Bond references and masks differ")
    atom_mask = targets["atom_mask"].bool()
    valid_refs = atom_refs[bond_mask]
    if len(valid_refs):
        if bool((valid_refs < 0).any()) or bool(
            (valid_refs >= atom_mask.size(1)).any()
        ):
            raise ValueError("G3 target Bond references are outside Atom capacity")
        batch_indices = torch.arange(
            atom_refs.size(0), device=atom_refs.device
        )[:, None, None].expand_as(atom_refs)
        referenced_atoms = atom_mask[batch_indices[bond_mask], valid_refs]
        if not bool(referenced_atoms.all()):
            raise ValueError("G3 target Bond references a padded Atom")
        if bool((valid_refs[:, 0] == valid_refs[:, 1]).any()):
            raise ValueError("G3 target Bond endpoints must reference different Atoms")
    if "bond_endpoints" in targets:
        endpoints = targets["bond_endpoints"]
    else:
        safe_refs = torch.where(
            bond_mask[..., None], atom_refs, torch.zeros_like(atom_refs)
        )
        batch_indices = torch.arange(
            atom_refs.size(0), device=atom_refs.device
        )[:, None, None].expand_as(atom_refs)
        endpoints = targets["atom_anchors"][batch_indices, safe_refs]
    if endpoints.shape != (*bond_mask.shape, 2, 2):
        raise ValueError(
            "G3 target Bond endpoints must have shape [batch, bonds, 2, 2]"
        )
    if not bool(torch.isfinite(endpoints[bond_mask]).all()):
        raise ValueError("G3 target Bond endpoints must be finite")
    if bool(
        ((endpoints[bond_mask] < 0.0) | (endpoints[bond_mask] > 1.0)).any()
    ):
        raise ValueError("G3 target Bond endpoints must be normalized")
    canonical_endpoints = endpoints.clone()
    canonical_refs = atom_refs.clone()
    for batch_index, bond_index in torch.nonzero(bond_mask, as_tuple=False).tolist():
        refs = canonical_refs[batch_index, bond_index]
        ports = canonical_endpoints[batch_index, bond_index]
        atom_anchors = targets["atom_anchors"][batch_index, refs]
        keys = [
            (
                *ports[index].detach().float().cpu().tolist(),
                *atom_anchors[index].detach().float().cpu().tolist(),
            )
            for index in range(2)
        ]
        if keys[1] < keys[0]:
            canonical_endpoints[batch_index, bond_index] = ports.flip(0)
            canonical_refs[batch_index, bond_index] = refs.flip(0)

    bond_types = targets["bond_types"].long()
    if bond_types.shape != bond_mask.shape:
        raise ValueError("G3 target Bond types and masks differ")
    bond_display = targets.get("bond_display", bond_types).long()
    if bond_display.shape != bond_mask.shape:
        raise ValueError("G3 target Bond display types and masks differ")

    ordered_endpoints = torch.zeros_like(canonical_endpoints)
    ordered_refs = torch.zeros_like(canonical_refs)
    ordered_display = torch.zeros_like(bond_display)
    ordered_chemical = torch.zeros_like(bond_types)
    ordered_mask = torch.zeros_like(bond_mask)
    for batch_index in range(len(bond_mask)):
        indices = torch.where(bond_mask[batch_index])[0].tolist()
        indices.sort(
            key=lambda index: (
                *canonical_endpoints[batch_index, index]
                .detach()
                .float()
                .cpu()
                .flatten()
                .tolist(),
                int(bond_display[batch_index, index]),
                int(bond_types[batch_index, index]),
                *targets["atom_anchors"][
                    batch_index, canonical_refs[batch_index, index]
                ]
                .detach()
                .float()
                .cpu()
                .flatten()
                .tolist(),
            )
        )
        count = len(indices)
        if not count:
            continue
        source = torch.tensor(indices, device=bond_mask.device, dtype=torch.long)
        ordered_endpoints[batch_index, :count] = canonical_endpoints[
            batch_index, source
        ]
        ordered_refs[batch_index, :count] = canonical_refs[batch_index, source]
        ordered_display[batch_index, :count] = bond_display[batch_index, source]
        ordered_chemical[batch_index, :count] = bond_types[batch_index, source]
        ordered_mask[batch_index, :count] = True
    return {
        "endpoints": ordered_endpoints,
        "atom_refs": ordered_refs,
        "display": ordered_display,
        "chemical": ordered_chemical,
        "mask": ordered_mask,
    }


def g3_fixed_bond_slot_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    *,
    normalized_bond_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise P0-E slots by canonical prefix, without any set assignment."""

    normalized = (
        build_g3_bond_targets(targets)
        if normalized_bond_targets is None
        else normalized_bond_targets
    )
    slot_output = output.get("fixed_bond_slot_output")
    if not isinstance(slot_output, dict):
        raise ValueError("G3 P0-E output is missing fixed Bond slots")
    mask = normalized["mask"].bool()
    expected_shapes = {
        "endpoints": (*mask.shape, 2, 2),
        "display_logits": (*mask.shape, len(BOND_TYPES)),
        "chemical_logits": (*mask.shape, len(BOND_TYPES)),
        "endpoint_pointer_logits": (
            *mask.shape,
            2,
            targets["atom_mask"].size(1),
        ),
    }
    for name, expected in expected_shapes.items():
        if slot_output[name].shape != expected:
            raise ValueError(
                f"G3 P0-E {name} has shape {tuple(slot_output[name].shape)}, "
                f"expected {expected}"
            )
    prefix = torch.arange(mask.size(1), device=mask.device)[None] < mask.sum(
        dim=1, keepdim=True
    )
    if not torch.equal(mask, prefix):
        raise ValueError("G3 P0-E Bond targets must use a canonical prefix")
    atom_target_mask = targets["atom_mask"].bool()
    refs = normalized["atom_refs"].long()
    if bool(mask.any()):
        active_refs = refs[mask]
        if bool((active_refs < 0).any()) or bool(
            (active_refs >= atom_target_mask.size(1)).any()
        ):
            raise ValueError("G3 P0-E pointer target exceeds Atom capacity")
        endpoint_loss = F.smooth_l1_loss(
            slot_output["endpoints"][mask], normalized["endpoints"][mask]
        )
        display_loss = F.cross_entropy(
            slot_output["display_logits"][mask], normalized["display"][mask]
        )
        chemical_loss = F.cross_entropy(
            slot_output["chemical_logits"][mask], normalized["chemical"][mask]
        )
        pointer_logits = slot_output["endpoint_pointer_logits"][mask]
        pointer_loss = F.cross_entropy(
            pointer_logits.reshape(-1, pointer_logits.size(-1)),
            active_refs.reshape(-1),
        )
        pointer_predictions = pointer_logits.argmax(dim=-1)
        endpoint_accuracy = (pointer_predictions == active_refs).float().mean()
        bond_pointer_accuracy = (
            pointer_predictions == active_refs
        ).all(dim=-1).float().mean()
    else:
        zero = output["bond_count_logits"].sum() * 0.0
        endpoint_loss = zero
        display_loss = zero
        chemical_loss = zero
        pointer_loss = zero
        endpoint_accuracy = zero
        bond_pointer_accuracy = zero

    atom_count_targets = atom_target_mask.sum(dim=1).long()
    bond_count_targets = mask.sum(dim=1).long()
    if bool((atom_count_targets >= output["atom_count_logits"].size(-1)).any()):
        raise ValueError("G3 P0-E target Atom count exceeds count-head capacity")
    if bool((bond_count_targets >= output["bond_count_logits"].size(-1)).any()):
        raise ValueError("G3 P0-E target Bond count exceeds count-head capacity")
    atom_count_loss = F.cross_entropy(
        output["atom_count_logits"], atom_count_targets
    )
    bond_count_loss = F.cross_entropy(
        output["bond_count_logits"], bond_count_targets
    )
    total = (
        0.1 * atom_count_loss
        + 0.1 * bond_count_loss
        + 8.0 * endpoint_loss
        + display_loss
        + chemical_loss
        + pointer_loss
    )
    return {
        "loss": total,
        "atom_count_loss": atom_count_loss,
        "bond_count_loss": bond_count_loss,
        "endpoint_loss": endpoint_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
        "incidence_loss": pointer_loss,
        "endpoint_accuracy": endpoint_accuracy,
        "bond_incidence_accuracy": bond_pointer_accuracy,
    }


@torch.no_grad()
def g3_set_bond_slot_assignments(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    normalized_bond_targets: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match P0-F queries to the target Bond set once per image."""

    slots = output.get("set_bond_slot_output")
    if not isinstance(slots, dict):
        raise ValueError("G3 P0-F output is missing set Bond slots")
    target_mask = normalized_bond_targets["mask"].bool()
    batch, queries = slots["presence_logits"].shape
    if target_mask.shape != (batch, queries):
        raise ValueError("G3 P0-F prediction and target capacities differ")
    assignments = []
    for batch_index in range(batch):
        count = int(target_mask[batch_index].sum())
        if count == 0:
            empty = torch.empty(
                0, device=target_mask.device, dtype=torch.long
            )
            assignments.append((empty, empty))
            continue
        predicted_endpoints = slots["endpoints"][batch_index].float()
        target_endpoints = normalized_bond_targets["endpoints"][
            batch_index, :count
        ].float()
        forward_endpoint = (
            predicted_endpoints[:, None] - target_endpoints[None]
        ).abs().sum(dim=(-1, -2))
        reverse_endpoint = (
            predicted_endpoints[:, None] - target_endpoints.flip(-2)[None]
        ).abs().sum(dim=(-1, -2))
        reverse = reverse_endpoint < forward_endpoint
        endpoint_cost = torch.minimum(forward_endpoint, reverse_endpoint)

        refs = normalized_bond_targets["atom_refs"][batch_index, :count].long()
        pointer_log_probability = slots["endpoint_pointer_logits"][
            batch_index
        ].float().log_softmax(dim=-1)
        forward_pointer = -(
            pointer_log_probability[:, 0][:, refs[:, 0]]
            + pointer_log_probability[:, 1][:, refs[:, 1]]
        )
        reverse_pointer = -(
            pointer_log_probability[:, 0][:, refs[:, 1]]
            + pointer_log_probability[:, 1][:, refs[:, 0]]
        )
        pointer_cost = torch.where(reverse, reverse_pointer, forward_pointer)

        chemical = normalized_bond_targets["chemical"][
            batch_index, :count
        ].long()
        display = normalized_bond_targets["display"][batch_index, :count].long()
        chemical_cost = -slots["chemical_logits"][batch_index].float().log_softmax(
            dim=-1
        )[:, chemical]
        display_cost = -slots["display_logits"][batch_index].float().log_softmax(
            dim=-1
        )[:, display]
        presence_cost = -slots["presence_logits"][batch_index].float().sigmoid()[
            :, None
        ]
        cost = (
            8.0 * endpoint_cost
            + pointer_cost
            + chemical_cost
            + display_cost
            + presence_cost
        )
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        query_indices = torch.as_tensor(
            rows, device=target_mask.device, dtype=torch.long
        )
        target_indices = torch.as_tensor(
            columns, device=target_mask.device, dtype=torch.long
        )
        if len(query_indices) != count or len(query_indices.unique()) != count:
            raise RuntimeError("G3 P0-F Hungarian matching did not cover every target")
        assignments.append((query_indices, target_indices))
    return assignments


def g3_set_bond_slot_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    *,
    normalized_bond_targets: dict[str, torch.Tensor] | None = None,
    assignments: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Train P0-F with one shared set assignment for every Bond field."""

    normalized = (
        build_g3_bond_targets(targets)
        if normalized_bond_targets is None
        else normalized_bond_targets
    )
    slots = output.get("set_bond_slot_output")
    if not isinstance(slots, dict):
        raise ValueError("G3 P0-F output is missing set Bond slots")
    if assignments is None:
        assignments = g3_set_bond_slot_assignments(output, normalized)
    if len(assignments) != slots["presence_logits"].size(0):
        raise ValueError("G3 P0-F assignments have an incompatible batch")

    presence_targets = torch.zeros_like(slots["presence_logits"])
    predicted_endpoints = []
    expected_endpoints = []
    predicted_display = []
    expected_display = []
    predicted_chemical = []
    expected_chemical = []
    predicted_pointer = []
    expected_pointer = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if len(query_indices) != int(normalized["mask"][batch_index].sum()):
            raise ValueError("G3 P0-F assignment does not cover the target set")
        presence_targets[batch_index, query_indices] = 1.0
        if not len(query_indices):
            continue
        current_predicted_endpoints = slots["endpoints"][
            batch_index, query_indices
        ]
        current_expected_endpoints = normalized["endpoints"][
            batch_index, target_indices
        ]
        current_expected_refs = normalized["atom_refs"][
            batch_index, target_indices
        ].long()
        forward_cost = (
            current_predicted_endpoints.detach().float()
            - current_expected_endpoints.detach().float()
        ).abs().sum(dim=(-1, -2))
        reverse_cost = (
            current_predicted_endpoints.detach().float()
            - current_expected_endpoints.detach().float().flip(-2)
        ).abs().sum(dim=(-1, -2))
        reverse = reverse_cost < forward_cost
        current_expected_endpoints = torch.where(
            reverse[:, None, None],
            current_expected_endpoints.flip(-2),
            current_expected_endpoints,
        )
        current_expected_refs = torch.where(
            reverse[:, None],
            current_expected_refs.flip(-1),
            current_expected_refs,
        )
        predicted_endpoints.append(current_predicted_endpoints)
        expected_endpoints.append(current_expected_endpoints)
        predicted_display.append(slots["display_logits"][batch_index, query_indices])
        expected_display.append(
            normalized["display"][batch_index, target_indices].long()
        )
        predicted_chemical.append(
            slots["chemical_logits"][batch_index, query_indices]
        )
        expected_chemical.append(
            normalized["chemical"][batch_index, target_indices].long()
        )
        predicted_pointer.append(
            slots["endpoint_pointer_logits"][batch_index, query_indices]
        )
        expected_pointer.append(current_expected_refs)

    presence_loss = F.binary_cross_entropy_with_logits(
        slots["presence_logits"], presence_targets
    )
    if predicted_endpoints:
        predicted_endpoints_tensor = torch.cat(predicted_endpoints)
        expected_endpoints_tensor = torch.cat(expected_endpoints)
        predicted_display_tensor = torch.cat(predicted_display)
        expected_display_tensor = torch.cat(expected_display)
        predicted_chemical_tensor = torch.cat(predicted_chemical)
        expected_chemical_tensor = torch.cat(expected_chemical)
        predicted_pointer_tensor = torch.cat(predicted_pointer)
        expected_pointer_tensor = torch.cat(expected_pointer)
        endpoint_loss = F.smooth_l1_loss(
            predicted_endpoints_tensor, expected_endpoints_tensor
        )
        display_loss = F.cross_entropy(
            predicted_display_tensor, expected_display_tensor
        )
        chemical_loss = F.cross_entropy(
            predicted_chemical_tensor, expected_chemical_tensor
        )
        pointer_loss = F.cross_entropy(
            predicted_pointer_tensor.reshape(-1, predicted_pointer_tensor.size(-1)),
            expected_pointer_tensor.reshape(-1),
        )
        pointer_predictions = predicted_pointer_tensor.argmax(dim=-1)
        endpoint_accuracy = (
            pointer_predictions == expected_pointer_tensor
        ).float().mean()
        bond_pointer_accuracy = (
            pointer_predictions == expected_pointer_tensor
        ).all(dim=-1).float().mean()
    else:
        zero = presence_loss * 0.0
        endpoint_loss = zero
        display_loss = zero
        chemical_loss = zero
        pointer_loss = zero
        endpoint_accuracy = zero
        bond_pointer_accuracy = zero

    atom_count_targets = targets["atom_mask"].sum(dim=1).long()
    bond_count_targets = normalized["mask"].sum(dim=1).long()
    atom_count_loss = F.cross_entropy(
        output["atom_count_logits"], atom_count_targets
    )
    bond_count_loss = F.cross_entropy(
        output["bond_count_logits"], bond_count_targets
    )
    total = (
        0.1 * atom_count_loss
        + 0.1 * bond_count_loss
        + presence_loss
        + 8.0 * endpoint_loss
        + display_loss
        + chemical_loss
        + pointer_loss
    )
    return {
        "loss": total,
        "atom_count_loss": atom_count_loss,
        "bond_count_loss": bond_count_loss,
        "presence_loss": presence_loss,
        "endpoint_loss": endpoint_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
        "incidence_loss": pointer_loss,
        "endpoint_accuracy": endpoint_accuracy,
        "bond_incidence_accuracy": bond_pointer_accuracy,
    }


def build_g3_atom_pair_targets(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    *,
    normalized_bond_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | int]:
    """Map GT Bonds onto the complete post-forward P0-G AtomPair universe."""

    normalized = (
        build_g3_bond_targets(targets)
        if normalized_bond_targets is None
        else normalized_bond_targets
    )
    pairs = output["flat_atom_pair_indices"].long()
    samples = output["flat_atom_pair_sample_indices"].long()
    presence = torch.zeros(len(pairs), device=pairs.device, dtype=torch.bool)
    chemical = torch.zeros(len(pairs), device=pairs.device, dtype=torch.long)
    target_bonds = covered_bonds = 0
    for batch_index in range(normalized["mask"].size(0)):
        count = int(normalized["mask"][batch_index].sum())
        target_bonds += count
        target_map = {
            tuple(sorted((int(refs[0]), int(refs[1])))): int(kind)
            for refs, kind in zip(
                normalized["atom_refs"][batch_index, :count].tolist(),
                normalized["chemical"][batch_index, :count].tolist(),
            )
        }
        if len(target_map) != count or any(left == right for left, right in target_map):
            raise ValueError("G3 P0-G targets contain duplicate or self Bond refs")
        rows = torch.where(samples == batch_index)[0]
        universe = {}
        for row in rows.tolist():
            pair = tuple(int(value) for value in pairs[row].tolist())
            if pair[0] >= pair[1] or pair in universe:
                raise ValueError("G3 P0-G AtomPair universe is not unique unordered pairs")
            universe[pair] = row
        missing = set(target_map) - set(universe)
        if missing:
            raise ValueError("G3 P0-G AtomPair universe does not cover every GT Bond")
        for pair, kind in target_map.items():
            row = universe[pair]
            presence[row] = True
            chemical[row] = kind
            covered_bonds += 1
    if covered_bonds != target_bonds:
        raise RuntimeError("G3 P0-G target coverage accounting differs")
    return {
        "presence": presence,
        "chemical": chemical,
        "target_bonds": target_bonds,
        "covered_bonds": covered_bonds,
    }


def g3_atom_pair_bond_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    *,
    normalized_bond_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Train P0-G existence/type without Bond queries, matching, or pointer loss."""

    pair_targets = build_g3_atom_pair_targets(
        output,
        targets,
        normalized_bond_targets=normalized_bond_targets,
    )
    pair_losses = direct_atom_pair_loss(
        output["flat_atom_pair_output"],
        pair_targets["presence"],
        pair_targets["chemical"],
    )
    atom_count_targets = targets["atom_mask"].sum(dim=1).long()
    bond_count_targets = targets["bond_mask"].sum(dim=1).long()
    atom_count_loss = F.cross_entropy(
        output["atom_count_logits"], atom_count_targets
    )
    bond_count_loss = F.cross_entropy(
        output["bond_count_logits"], bond_count_targets
    )
    total = (
        0.1 * atom_count_loss
        + 0.1 * bond_count_loss
        + pair_losses["presence_loss"]
        + pair_losses["chemical_loss"]
    )
    one = pair_losses["presence_loss"] * 0.0 + 1.0
    return {
        "loss": total,
        "atom_count_loss": atom_count_loss,
        "bond_count_loss": bond_count_loss,
        "presence_loss": pair_losses["presence_loss"],
        "chemical_loss": pair_losses["chemical_loss"],
        "endpoint_accuracy": one,
        "bond_incidence_accuracy": one,
        "target_bond_coverage": one
        * (float(pair_targets["covered_bonds"]) / max(pair_targets["target_bonds"], 1)),
    }


def g3_confidence_aware_bond_assignments(
    output: dict[str, torch.Tensor],
    bond_targets: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return the pre-registered presence-aware P0-C Bond assignment."""

    return bond_group_query_assignments(
        output["bond_query_output"],
        bond_targets,
        presence_cost_weight=G3_CONFIDENCE_AWARE_BOND_MATCHING_WEIGHT,
    )


def gt_bond_slot_assignments(
    output: dict[str, torch.Tensor],
    bond_targets: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Validate and return the P0-D direct prefix Bond identity assignment."""

    query_output = output["bond_query_output"]
    target_identity = query_output.get("bond_target_indices")
    positive_mask = query_output.get("oracle_positive_mask")
    full_coverage = query_output.get("oracle_full_coverage")
    matched_counts = query_output.get("oracle_matched_target_count")
    target_counts = query_output.get("oracle_target_count")
    query_shape = query_output["presence_logits"].shape
    if target_identity is None or target_identity.shape != query_shape:
        raise ValueError("G3 oracle Bond target identity is missing or incompatible")
    if positive_mask is None or positive_mask.shape != query_shape:
        raise ValueError("G3 oracle Bond positive mask is missing or incompatible")
    batch = query_shape[0]
    if any(
        value is None or value.shape != (batch,)
        for value in (full_coverage, matched_counts, target_counts)
    ):
        raise ValueError("G3 oracle Bond coverage is missing or incompatible")
    if bond_targets["mask"].shape[0] != batch:
        raise ValueError("G3 oracle Bond target batch is incompatible")

    assignments = []
    for batch_index in range(batch):
        count = int(bond_targets["mask"][batch_index].sum())
        if count > query_shape[1]:
            raise ValueError("G3 oracle Bond targets exceed query capacity")
        expected_mask = torch.arange(
            query_shape[1], device=target_identity.device
        ) < count
        expected_target_mask = torch.arange(
            bond_targets["mask"].size(1), device=target_identity.device
        ) < count
        expected_targets = torch.full(
            (query_shape[1],), -1, device=target_identity.device, dtype=torch.long
        )
        expected_targets[:count] = torch.arange(count, device=target_identity.device)
        if (
            not torch.equal(
                bond_targets["mask"][batch_index].bool(), expected_target_mask
            )
            or not torch.equal(positive_mask[batch_index].bool(), expected_mask)
            or not torch.equal(target_identity[batch_index].long(), expected_targets)
            or int(matched_counts[batch_index]) != count
            or int(target_counts[batch_index]) != count
            or not bool(full_coverage[batch_index])
            or not bool(query_output["proposal_mask"][batch_index, :count].all())
        ):
            raise ValueError("G3 oracle Bond prefix identity or coverage differs from targets")
        identity = torch.arange(count, device=target_identity.device, dtype=torch.long)
        assignments.append((identity, identity))
    return assignments


def _incidence_targets(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    bond_targets: dict[str, torch.Tensor],
    *,
    bond_assignments: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pointer_targets = torch.zeros(
        output["bond_presence_logits"].shape + (2,),
        device=output["bond_presence_logits"].device,
        dtype=torch.long,
    )
    pointer_mask = torch.zeros_like(
        output["bond_presence_logits"], dtype=torch.bool
    )
    atom_assignments = _atom_slot_assignments(output, targets)
    if bond_assignments is None:
        bond_assignments = bond_group_query_assignments(
            output["bond_query_output"], bond_targets
        )
    paired_assignments = zip(atom_assignments, bond_assignments)
    for batch_index, (
        (atom_queries, atom_targets),
        (bond_queries, bond_indices),
    ) in enumerate(paired_assignments):
        target_to_query = {
            int(target): int(query)
            for query, target in zip(atom_queries.tolist(), atom_targets.tolist())
        }
        for query, target_index in zip(
            bond_queries.tolist(), bond_indices.tolist()
        ):
            refs = bond_targets["atom_refs"][batch_index, target_index].long()
            expected = bond_targets["endpoints"][batch_index, target_index]
            predicted = output["bond_query_output"]["endpoints"][batch_index, query]
            forward = (predicted - expected).abs().sum()
            reverse = (predicted - expected.flip(0)).abs().sum()
            tied = torch.isclose(forward, reverse, atol=1e-7, rtol=1e-6)
            if not bool(tied) and bool(reverse < forward):
                refs = refs.flip(0)
            if any(int(ref) not in target_to_query for ref in refs):
                continue
            pointer_targets[batch_index, query] = torch.tensor(
                [target_to_query[int(ref)] for ref in refs],
                device=pointer_targets.device,
                dtype=torch.long,
            )
            pointer_mask[batch_index, query] = True
    return pointer_targets, pointer_mask


def g3_graph_loss(
    output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    *,
    shared_confidence_aware_bond_assignment: bool = False,
    gt_bond_slot_identity_oracle: bool = False,
    normalized_bond_targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Train geometry, counts, Bond groups, and endpoint pointers jointly."""

    atom_losses = spatial_atom_slot_loss(output["atom_dense_output"], targets)
    if shared_confidence_aware_bond_assignment and gt_bond_slot_identity_oracle:
        raise ValueError("G3 P0-C and P0-D Bond identity protocols are mutually exclusive")
    normalized_bonds = (
        build_g3_bond_targets(targets)
        if normalized_bond_targets is None
        else normalized_bond_targets
    )
    dense_bond_targets = {
        **normalized_bonds,
        "direction": torch.zeros_like(targets["bond_types"]).long(),
    }
    dense_bond_losses = dense_bond_object_loss(
        output["bond_dense_output"],
        dense_bond_targets,
        slot_matching="hungarian",
    )
    # Direction/stereo canonicalization is outside the structural G3 prototype.
    dense_bond_losses = {
        **dense_bond_losses,
        "loss": dense_bond_losses["loss"]
        - 0.5 * dense_bond_losses["direction_loss"],
    }
    if gt_bond_slot_identity_oracle:
        shared_bond_assignments = gt_bond_slot_assignments(output, normalized_bonds)
    elif shared_confidence_aware_bond_assignment:
        shared_bond_assignments = g3_confidence_aware_bond_assignments(
            output, normalized_bonds
        )
    else:
        shared_bond_assignments = None
    bond_losses = bond_group_query_loss(
        output["bond_query_output"],
        normalized_bonds,
        assignments=shared_bond_assignments,
    )
    bond_count_targets = targets["bond_mask"].sum(dim=1).long()
    if bool((bond_count_targets >= output["bond_count_logits"].size(-1)).any()):
        raise ValueError("G3 target Bond count exceeds count-head capacity")
    bond_count_loss = F.cross_entropy(
        output["bond_count_logits"], bond_count_targets
    )
    pointer_targets, pointer_mask = _incidence_targets(
        output,
        targets,
        normalized_bonds,
        bond_assignments=shared_bond_assignments,
    )
    incidence_losses = spatial_incidence_loss(
        {
            "incidence_logits": output["bond_query_output"][
                "endpoint_pointer_logits"
            ]
        },
        pointer_targets,
        pointer_mask,
    )
    total = (
        atom_losses["loss"]
        + dense_bond_losses["loss"]
        + bond_losses["loss"]
        + 0.1 * bond_count_loss
        + incidence_losses["loss"]
    )
    result = {
        "loss": total,
        "bond_count_loss": bond_count_loss,
        "incidence_loss": incidence_losses["loss"],
        "endpoint_accuracy": incidence_losses["endpoint_accuracy"],
        "bond_incidence_accuracy": incidence_losses["bond_accuracy"],
    }
    result.update(
        {
            f"dense_bond_{name}": value
            for name, value in dense_bond_losses.items()
            if name not in {"loss", "direction_loss"}
        }
    )
    result.update(
        {
            f"atom_{name}": value
            for name, value in atom_losses.items()
            if name != "loss"
        }
    )
    result.update(
        {
            f"bond_{name}": value
            for name, value in bond_losses.items()
            if name != "loss"
        }
    )
    return result
