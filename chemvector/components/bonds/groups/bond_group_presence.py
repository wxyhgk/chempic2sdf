"""Presence selection for frozen Bond Group Query candidates."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.equivalence_bond_selector import equivalence_bond_selector_loss


GEOMETRY_FEATURE_DIM = 18


@dataclass(frozen=True, slots=True)
class BondGroupPresenceConfig:
    endpoint_local_dim: int = 800
    endpoint_ink_dim: int = 64
    object_dim: int = 192
    endpoint_context_dim: int = 0
    endpoint_embedding_dim: int = 64
    object_embedding_dim: int = 64
    d_model: int = 128
    set_layers: int = 0
    heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.endpoint_local_dim,
            self.endpoint_ink_dim,
            self.object_dim,
            self.endpoint_embedding_dim,
            self.object_embedding_dim,
            self.d_model,
            self.heads,
            self.ffn_dim,
        )
        if min(positive) < 1 or self.endpoint_context_dim < 0 or self.set_layers < 0:
            raise ValueError("Bond Group presence dimensions are invalid")
        if self.d_model % self.heads:
            raise ValueError("Bond Group presence heads must divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Bond Group presence dropout must be in [0, 1)")


def _segment_features(endpoints: torch.Tensor) -> torch.Tensor:
    center = endpoints.mean(dim=-2)
    delta = endpoints[..., 1, :] - endpoints[..., 0, :]
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    direction = delta / length.clamp_min(1e-6)
    double_angle = torch.stack(
        (
            direction[..., 0].square() - direction[..., 1].square(),
            2.0 * direction[..., 0] * direction[..., 1],
        ),
        dim=-1,
    )
    return torch.cat((center, delta.abs(), length, double_angle), dim=-1)


def _geometry_features(
    base_endpoints: torch.Tensor, refined_endpoints: torch.Tensor
) -> torch.Tensor:
    displacement = refined_endpoints - base_endpoints
    return torch.cat(
        (
            _segment_features(base_endpoints),
            _segment_features(refined_endpoints),
            displacement.mean(dim=-2),
            (displacement[..., 0, :] - displacement[..., 1, :]).abs(),
        ),
        dim=-1,
    )


class BondGroupPresenceSelector(nn.Module):
    """Rank frozen Bond proposals from local endpoint and set context."""

    def __init__(self, config: BondGroupPresenceConfig | None = None) -> None:
        super().__init__()
        self.config = config or BondGroupPresenceConfig()
        endpoint_dim = self.config.endpoint_embedding_dim
        self.endpoint_local_encoder = nn.Sequential(
            nn.LayerNorm(self.config.endpoint_local_dim),
            nn.Linear(self.config.endpoint_local_dim, endpoint_dim),
            nn.GELU(),
            nn.LayerNorm(endpoint_dim),
        )
        self.endpoint_ink_encoder = nn.Sequential(
            nn.LayerNorm(self.config.endpoint_ink_dim),
            nn.Linear(self.config.endpoint_ink_dim, endpoint_dim),
            nn.GELU(),
            nn.LayerNorm(endpoint_dim),
        )
        self.endpoint_context_encoder: nn.Module | None = None
        endpoint_sources = 2
        if self.config.endpoint_context_dim:
            self.endpoint_context_encoder = nn.Sequential(
                nn.LayerNorm(self.config.endpoint_context_dim),
                nn.Linear(self.config.endpoint_context_dim, endpoint_dim),
                nn.GELU(),
                nn.LayerNorm(endpoint_dim),
            )
            endpoint_sources += 1
        self.endpoint_fusion = nn.Sequential(
            nn.Linear(endpoint_sources * endpoint_dim, endpoint_dim),
            nn.GELU(),
            nn.LayerNorm(endpoint_dim),
        )
        self.object_encoder = nn.Sequential(
            nn.LayerNorm(self.config.object_dim),
            nn.Linear(self.config.object_dim, self.config.object_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.object_embedding_dim),
        )
        type_prior_dim = 2 * len(BOND_TYPES) + len(DIRECTIONS)
        candidate_dim = (
            2 * endpoint_dim
            + self.config.object_embedding_dim
            + GEOMETRY_FEATURE_DIM
            + 1
            + type_prior_dim
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(candidate_dim),
            nn.Linear(candidate_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        self.set_encoder: nn.Module | None = None
        if self.config.set_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.heads,
                dim_feedforward=self.config.ffn_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_encoder = nn.TransformerEncoder(
                layer,
                self.config.set_layers,
                norm=nn.LayerNorm(self.config.d_model),
                enable_nested_tensor=False,
            )
        self.presence_residual = nn.Linear(self.config.d_model, 1)
        nn.init.zeros_(self.presence_residual.weight)
        nn.init.zeros_(self.presence_residual.bias)

    def _validate_inputs(
        self,
        group_output: dict[str, torch.Tensor],
        proposals: dict[str, torch.Tensor],
    ) -> tuple[int, int, torch.Tensor]:
        required_group = {
            "endpoints",
            "endpoint_local_features",
            "endpoint_ink_line_features",
            "object_features",
            "display_logits",
            "chemical_logits",
            "direction_logits",
        }
        required_proposals = {"base_logits", "base_endpoints", "mask"}
        missing_group = sorted(required_group - group_output.keys())
        missing_proposals = sorted(required_proposals - proposals.keys())
        if missing_group or missing_proposals:
            raise ValueError(
                "Bond Group presence inputs are missing fields: "
                f"group={missing_group}, proposals={missing_proposals}"
            )
        base_logits = proposals["base_logits"]
        if base_logits.ndim != 2:
            raise ValueError("Bond Group base logits must have shape [batch, queries]")
        batch, queries = base_logits.shape
        mask = proposals["mask"].bool()
        expected = {
            "base_endpoints": (batch, queries, 2, 2),
            "endpoints": (batch, queries, 2, 2),
            "endpoint_local_features": (
                batch,
                queries,
                2,
                self.config.endpoint_local_dim,
            ),
            "endpoint_ink_line_features": (
                batch,
                queries,
                2,
                self.config.endpoint_ink_dim,
            ),
            "object_features": (batch, queries, self.config.object_dim),
            "display_logits": (batch, queries, len(BOND_TYPES)),
            "chemical_logits": (batch, queries, len(BOND_TYPES)),
            "direction_logits": (batch, queries, len(DIRECTIONS)),
        }
        values = {**group_output, **proposals}
        for name, shape in expected.items():
            value = values[name]
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(
                    f"Bond Group presence {name} has shape "
                    f"{getattr(value, 'shape', None)}, expected {shape}"
                )
        if mask.shape != (batch, queries):
            raise ValueError("Bond Group proposal mask has an incompatible shape")
        if "proposal_mask" in group_output and not torch.equal(
            group_output["proposal_mask"].bool(), mask
        ):
            raise ValueError("Bond Group output and proposal masks differ")
        if self.endpoint_context_encoder is not None:
            context = group_output.get("endpoint_features")
            expected_context = (
                batch,
                queries,
                2,
                self.config.endpoint_context_dim,
            )
            if not isinstance(context, torch.Tensor) or context.shape != expected_context:
                raise ValueError("Bond Group endpoint context has an incompatible shape")
        for name in expected:
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"Bond Group presence {name} contains nonfinite values")
        return batch, queries, mask

    def forward(
        self,
        group_output: dict[str, torch.Tensor],
        proposals: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        _, _, mask = self._validate_inputs(group_output, proposals)
        local = self.endpoint_local_encoder(
            group_output["endpoint_local_features"].float()
        )
        ink = self.endpoint_ink_encoder(
            group_output["endpoint_ink_line_features"].float()
        )
        endpoint_inputs = [local, ink]
        if self.endpoint_context_encoder is not None:
            endpoint_inputs.append(
                self.endpoint_context_encoder(group_output["endpoint_features"].float())
            )
        endpoint = self.endpoint_fusion(torch.cat(endpoint_inputs, dim=-1))
        endpoint_pair = torch.cat(
            (endpoint.mean(dim=2), (endpoint[:, :, 0] - endpoint[:, :, 1]).abs()),
            dim=-1,
        )
        object_features = self.object_encoder(group_output["object_features"].float())
        geometry = _geometry_features(
            proposals["base_endpoints"].float(), group_output["endpoints"].float()
        )
        type_priors = torch.cat(
            (
                group_output["display_logits"].float().softmax(dim=-1),
                group_output["chemical_logits"].float().softmax(dim=-1),
                group_output["direction_logits"].float().softmax(dim=-1),
            ),
            dim=-1,
        )
        base_probability = proposals["base_logits"].float().sigmoid().unsqueeze(-1)
        candidate = torch.cat(
            (endpoint_pair, object_features, geometry, base_probability, type_priors),
            dim=-1,
        )
        hidden = self.input_projection(candidate)
        hidden = hidden * mask[..., None].to(hidden.dtype)
        if self.set_encoder is not None:
            padding = ~mask
            all_padding = padding.all(dim=1)
            if bool(all_padding.any()):
                padding = padding.clone()
                padding[all_padding, 0] = False
            hidden = self.set_encoder(hidden, src_key_padding_mask=padding)
            hidden = hidden * mask[..., None].to(hidden.dtype)
        residual = self.presence_residual(hidden).squeeze(-1)
        residual = residual * mask.to(residual.dtype)
        keep_logits = proposals["base_logits"] + residual.to(
            proposals["base_logits"].dtype
        )
        return {
            "keep_logits": keep_logits,
            "presence_residual": residual,
            "candidate_features": hidden,
            "proposal_mask": mask,
            "display_logits": group_output["display_logits"],
            "chemical_logits": group_output["chemical_logits"],
        }


def _image_scale(
    image_size: float | tuple[float, float],
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(image_size, tuple):
        if len(image_size) != 2 or min(image_size) <= 0.0:
            raise ValueError("Bond Group label image dimensions must be positive")
        width, height = image_size
    else:
        if image_size <= 0.0:
            raise ValueError("Bond Group label image size must be positive")
        width = height = image_size
    return reference.new_tensor((float(width), float(height)))


def _oriented_endpoint_errors(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    forward = torch.linalg.vector_norm(
        (candidates[:, None] - targets[None]) * scale, dim=-1
    )
    reverse = torch.linalg.vector_norm(
        (candidates[:, None] - targets[None].flip(2)) * scale, dim=-1
    )
    use_reverse = reverse.sum(dim=-1) < forward.sum(dim=-1)
    return torch.where(use_reverse[..., None], reverse, forward)


def _compatible_assignment(
    compatible: torch.Tensor,
    geometry_cost: torch.Tensor,
) -> list[tuple[int, int]]:
    target_count = compatible.size(1)
    candidate_count = compatible.size(0)
    if not target_count or not candidate_count:
        return []
    eligible = torch.arange(candidate_count, device=compatible.device)
    eligible_compatible = compatible
    eligible_cost = geometry_cost.clamp(0.0, 1.0)
    unmatched_cost = float(target_count + 1)
    matrix = torch.full(
        (target_count, len(eligible) + target_count),
        2.0 * unmatched_cost,
        dtype=torch.float64,
    )
    real = matrix[:, : len(eligible)]
    transposed_mask = eligible_compatible.transpose(0, 1).cpu()
    real[transposed_mask] = eligible_cost.transpose(0, 1).double().cpu()[
        transposed_mask
    ]
    matrix[:, len(eligible) :] = unmatched_cost
    rows, columns = linear_sum_assignment(matrix.numpy())
    matches = []
    for target_index, column in zip(rows.tolist(), columns.tolist()):
        if column >= len(eligible):
            continue
        candidate_index = int(eligible[column])
        if bool(compatible[candidate_index, target_index]):
            matches.append((candidate_index, target_index))
    return matches


@torch.no_grad()
def build_geometry_equivalence_labels(
    refined_endpoints: torch.Tensor,
    candidate_mask: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    tolerance_pixels: float = 6.0,
    image_size: float | tuple[float, float] = 768.0,
) -> dict[str, torch.Tensor]:
    """Build training-only equivalence groups from refined endpoint geometry."""

    if refined_endpoints.ndim != 4 or refined_endpoints.shape[-2:] != (2, 2):
        raise ValueError("Refined Bond endpoints must have shape [batch, queries, 2, 2]")
    if candidate_mask.shape != refined_endpoints.shape[:2]:
        raise ValueError("Refined Bond mask has an incompatible shape")
    required = {"endpoints", "mask", "display", "chemical"}
    missing = sorted(required - targets.keys())
    if missing:
        raise ValueError(f"Bond Group geometry targets are missing fields: {missing}")
    batch, queries = candidate_mask.shape
    target_endpoints = targets["endpoints"]
    target_mask = targets["mask"].bool()
    if target_endpoints.ndim != 4 or target_endpoints.shape[0] != batch:
        raise ValueError("Bond Group target endpoints have an incompatible shape")
    if target_endpoints.shape[-2:] != (2, 2) or target_mask.shape != target_endpoints.shape[:2]:
        raise ValueError("Bond Group target endpoint mask has an incompatible shape")
    if targets["display"].shape != target_mask.shape:
        raise ValueError("Bond Group display targets have an incompatible shape")
    if targets["chemical"].shape != target_mask.shape:
        raise ValueError("Bond Group chemical targets have an incompatible shape")
    if tolerance_pixels <= 0.0:
        raise ValueError("Bond Group geometry tolerance must be positive")
    if not bool(torch.isfinite(refined_endpoints).all()) or not bool(
        torch.isfinite(target_endpoints).all()
    ):
        raise ValueError("Bond Group geometry labels require finite endpoints")

    device = refined_endpoints.device
    loss_mask = candidate_mask.bool().clone()
    ambiguous_mask = torch.zeros((batch, queries), dtype=torch.bool, device=device)
    group_ids = torch.full((batch, queries), -1, dtype=torch.long, device=device)
    group_mask = torch.zeros((batch, queries), dtype=torch.bool, device=device)
    group_targets = torch.zeros((batch, queries), dtype=torch.bool, device=device)
    canonical_targets = torch.zeros((batch, queries), dtype=torch.bool, device=device)
    group_display_targets = torch.zeros(
        (batch, queries), dtype=torch.long, device=device
    )
    group_chemical_targets = torch.zeros(
        (batch, queries), dtype=torch.long, device=device
    )
    representable_counts = torch.zeros(batch, dtype=torch.long, device=device)
    gt_counts = target_mask.sum(dim=1).long()
    target_representable = torch.zeros_like(target_mask, device=device)
    target_group_ids = torch.full_like(target_mask, -1, dtype=torch.long, device=device)
    compatible_counts = torch.zeros((batch, queries), dtype=torch.long, device=device)
    scale = _image_scale(image_size, refined_endpoints)

    for batch_index in range(batch):
        candidate_indices = torch.where(candidate_mask[batch_index].bool())[0]
        target_indices = torch.where(target_mask[batch_index])[0]
        if len(target_indices):
            candidates = refined_endpoints[batch_index, candidate_indices].float()
            expected = target_endpoints[batch_index, target_indices].float()
            errors = _oriented_endpoint_errors(candidates, expected, scale.float())
            compatible = errors.le(tolerance_pixels + 1e-6).all(dim=-1)
            geometry_cost = errors.mean(dim=-1) / tolerance_pixels
        else:
            compatible = torch.zeros(
                (len(candidate_indices), 0), dtype=torch.bool, device=device
            )
            geometry_cost = torch.zeros(
                (len(candidate_indices), 0), dtype=torch.float32, device=device
            )
        local_counts = compatible.sum(dim=1)
        ambiguous = local_counts.gt(1)
        compatible_counts[batch_index, candidate_indices] = local_counts
        ambiguous_mask[batch_index, candidate_indices] = ambiguous
        loss_mask[batch_index, candidate_indices[ambiguous]] = False
        matches = _compatible_assignment(compatible, geometry_cost)

        group_index = 0
        for candidate_local, target_local in sorted(matches, key=lambda item: item[1]):
            target_index = int(target_indices[target_local])
            canonical = int(candidate_indices[candidate_local])
            canonical_local_mask = torch.zeros_like(ambiguous)
            canonical_local_mask[candidate_local] = True
            members_local = torch.where(
                compatible[:, target_local] & (~ambiguous | canonical_local_mask)
            )[0]
            members = candidate_indices[members_local]
            loss_mask[batch_index, canonical] = True
            group_ids[batch_index, members] = group_index
            group_mask[batch_index, group_index] = True
            group_targets[batch_index, group_index] = True
            canonical_targets[batch_index, canonical] = True
            group_display_targets[batch_index, group_index] = targets["display"][
                batch_index, target_index
            ].long()
            group_chemical_targets[batch_index, group_index] = targets["chemical"][
                batch_index, target_index
            ].long()
            target_representable[batch_index, target_index] = True
            target_group_ids[batch_index, target_index] = group_index
            group_index += 1

        for candidate_index in candidate_indices.tolist():
            if not bool(loss_mask[batch_index, candidate_index]):
                continue
            if int(group_ids[batch_index, candidate_index]) >= 0:
                continue
            if group_index >= queries:
                raise RuntimeError("Bond Group geometry labels exceed candidate groups")
            group_ids[batch_index, candidate_index] = group_index
            group_mask[batch_index, group_index] = True
            group_index += 1
        representable_counts[batch_index] = int(
            target_representable[batch_index].sum()
        )

    if not torch.equal(canonical_targets.sum(dim=1), representable_counts):
        raise RuntimeError("Bond Group canonical count differs from representable count")
    return {
        "loss_mask": loss_mask,
        "ambiguous_mask": ambiguous_mask,
        "group_ids": group_ids,
        "group_mask": group_mask,
        "group_targets": group_targets,
        "canonical_targets": canonical_targets,
        "group_display_targets": group_display_targets,
        "group_chemical_targets": group_chemical_targets,
        "representable_counts": representable_counts,
        "gt_counts": gt_counts,
        "target_representable": target_representable,
        "target_group_ids": target_group_ids,
        "compatible_counts": compatible_counts,
    }


def bond_group_presence_loss(
    output: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Apply the existing winner-anchored equivalence selector loss."""

    options = {
        "positive_mode": "canonical",
        "negative_mode": "max",
        "cardinality_mode": "none",
        "type_weight": 0.0,
    }
    options.update(kwargs)
    return equivalence_bond_selector_loss(
        output,
        labels["loss_mask"],
        labels["group_ids"],
        labels["group_mask"],
        labels["group_targets"],
        labels["canonical_targets"],
        labels["representable_counts"],
        labels["group_display_targets"],
        labels["group_chemical_targets"],
        **options,
    )
