"""Learned presence and bond type for first-class raster line AtomPairs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES
from chemvector.line_segment_rescue import AtomPairLineCandidate


LINE_GEOMETRY_DIM = 6


@dataclass(frozen=True, slots=True)
class LineAtomPairHeadConfig:
    feature_channels: int = 128
    hidden_dim: int = 256
    line_samples: int = 7
    image_size: float = 768.0
    endpoint_scale_pixels: float = 16.0
    angle_scale_degrees: float = 12.0

    def __post_init__(self) -> None:
        if min(self.feature_channels, self.hidden_dim, self.line_samples) < 1:
            raise ValueError("Line AtomPair dimensions must be positive")
        if min(
            self.image_size,
            self.endpoint_scale_pixels,
            self.angle_scale_degrees,
        ) <= 0.0:
            raise ValueError("Line AtomPair geometry scales must be positive")

    @property
    def input_dim(self) -> int:
        return self.feature_channels * 4 + LINE_GEOMETRY_DIM


class LineAtomPairHead(nn.Module):
    def __init__(self, config: LineAtomPairHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.presence = nn.Linear(config.hidden_dim, 1)
        self.display = nn.Linear(config.hidden_dim, len(BOND_TYPES))
        self.chemical = nn.Linear(config.hidden_dim, len(BOND_TYPES))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        return {
            "presence_logits": self.presence(hidden).squeeze(-1),
            "display_logits": self.display(hidden),
            "chemical_logits": self.chemical(hidden),
        }


def load_line_atom_pair_head(
    path: str | Path, device: torch.device
) -> tuple[LineAtomPairHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "line-atom-pair-head-0.1":
        raise ValueError("Unsupported line AtomPair checkpoint schema")
    model = LineAtomPairHead(LineAtomPairHeadConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


def _sample_line_features(
    feature_map: torch.Tensor,
    candidates: tuple[AtomPairLineCandidate, ...],
    *,
    config: LineAtomPairHeadConfig,
) -> torch.Tensor:
    if not candidates:
        return feature_map.new_empty((0, config.feature_channels * 2))
    if feature_map.shape != (
        config.feature_channels,
        feature_map.shape[-2],
        feature_map.shape[-1],
    ):
        raise ValueError("Bond feature map channel count differs from line-head config")
    segments = feature_map.new_tensor(
        [candidate.representative_segment_pixels for candidate in candidates]
    )
    fractions = torch.linspace(
        0.1,
        0.9,
        config.line_samples,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    points = (
        segments[:, None, 0]
        + fractions[None, :, None] * (segments[:, None, 1] - segments[:, None, 0])
    )
    denominator = config.image_size - 1.0
    grid = points / denominator * 2.0 - 1.0
    sampled = F.grid_sample(
        feature_map[None],
        grid.reshape(1, len(candidates) * config.line_samples, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled[0, :, :, 0].T.reshape(
        len(candidates), config.line_samples, config.feature_channels
    )
    return torch.cat((sampled.mean(1), sampled.amax(1)), dim=-1)


def build_line_atom_pair_features(
    bond_feature_maps: torch.Tensor,
    atom_features: torch.Tensor,
    atom_anchors: torch.Tensor,
    candidates: list[tuple[AtomPairLineCandidate, ...]],
    *,
    config: LineAtomPairHeadConfig = LineAtomPairHeadConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build order-invariant line/AtomPair features and sample indices."""

    if bond_feature_maps.ndim != 4:
        raise ValueError("bond_feature_maps must have shape [batch,channels,height,width]")
    if atom_features.ndim != 3 or atom_anchors.shape[:2] != atom_features.shape[:2]:
        raise ValueError("Atom features and anchors must share [batch,atoms]")
    if bond_feature_maps.size(0) != atom_features.size(0) or len(candidates) != atom_features.size(0):
        raise ValueError("Line AtomPair batch dimensions differ")
    if bond_feature_maps.size(1) != config.feature_channels:
        raise ValueError("Bond feature channels differ from line-head config")
    if atom_features.size(-1) != config.feature_channels:
        raise ValueError("Atom feature channels differ from line-head config")

    rows = []
    sample_indices = []
    for batch_index, sample_candidates in enumerate(candidates):
        if not sample_candidates:
            continue
        pairs = torch.tensor(
            [candidate.atom_pair for candidate in sample_candidates],
            device=atom_features.device,
            dtype=torch.long,
        )
        first = atom_features[batch_index, pairs[:, 0]]
        second = atom_features[batch_index, pairs[:, 1]]
        symmetric_atoms = torch.cat(
            ((first + second) * 0.5, (first - second).abs()), dim=-1
        )
        line_visual = _sample_line_features(
            bond_feature_maps[batch_index], sample_candidates, config=config
        )
        anchors = atom_anchors[batch_index, pairs] * config.image_size
        atom_delta = anchors[:, 1] - anchors[:, 0]
        atom_distance = torch.linalg.vector_norm(atom_delta, dim=-1).clamp_min(1e-6)
        segments = anchors.new_tensor(
            [candidate.representative_segment_pixels for candidate in sample_candidates]
        )
        segment_delta = segments[:, 1] - segments[:, 0]
        segment_length = torch.linalg.vector_norm(segment_delta, dim=-1).clamp_min(1e-6)
        unit = segment_delta / segment_length[:, None]
        doubled_direction = torch.stack(
            (
                unit[:, 0].square() - unit[:, 1].square(),
                2.0 * unit[:, 0] * unit[:, 1],
            ),
            dim=-1,
        )
        endpoint_error = anchors.new_tensor(
            [candidate.endpoint_error_pixels for candidate in sample_candidates]
        )
        angle_error = anchors.new_tensor(
            [candidate.angle_error_degrees for candidate in sample_candidates]
        )
        support = anchors.new_tensor(
            [len(candidate.support_segment_indices) for candidate in sample_candidates]
        )
        geometry = torch.cat(
            (
                (segment_length / atom_distance)[:, None],
                (endpoint_error / config.endpoint_scale_pixels)[:, None],
                (angle_error / config.angle_scale_degrees)[:, None],
                (support.log1p() / math.log(9.0))[:, None],
                doubled_direction,
            ),
            dim=-1,
        )
        rows.append(torch.cat((line_visual, symmetric_atoms, geometry), dim=-1))
        sample_indices.extend([batch_index] * len(sample_candidates))
    if not rows:
        return (
            bond_feature_maps.new_empty((0, config.input_dim)),
            torch.empty(0, device=bond_feature_maps.device, dtype=torch.long),
        )
    features = torch.cat(rows, dim=0)
    if features.size(-1) != config.input_dim:
        raise RuntimeError("Line AtomPair feature dimension is inconsistent")
    return features, torch.tensor(
        sample_indices, device=features.device, dtype=torch.long
    )


def line_atom_pair_loss(
    output: dict[str, torch.Tensor],
    presence_targets: torch.Tensor,
    display_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    type_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    targets = presence_targets.float()
    positives = targets.sum().clamp_min(1.0)
    negatives = (1.0 - targets).sum().clamp_min(1.0)
    presence = F.binary_cross_entropy_with_logits(
        output["presence_logits"].float(),
        targets,
        pos_weight=(negatives / positives).clamp(1.0, 16.0),
    )
    mask = presence_targets.bool()
    zero = presence * 0.0
    if bool(mask.any()):
        display = F.cross_entropy(output["display_logits"][mask].float(), display_targets[mask])
        chemical = F.cross_entropy(
            output["chemical_logits"][mask].float(), chemical_targets[mask]
        )
    else:
        display = chemical = zero
    return {
        "loss": presence + type_weight * (display + chemical),
        "presence_loss": presence,
        "display_loss": display,
        "chemical_loss": chemical,
    }
