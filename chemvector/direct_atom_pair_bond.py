"""Direct semantic Bond classification over unordered predicted Atom pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES


@dataclass(frozen=True, slots=True)
class DirectAtomPairBondConfig:
    feature_channels: int = 96
    atom_feature_dim: int = 128
    longitudinal_samples: int = 32
    transverse_samples: int = 9
    ribbon_radius_pixels: float = 12.0
    image_size: float = 768.0
    hidden_dim: int = 256
    pair_chunk_size: int = 512

    def __post_init__(self) -> None:
        if min(
            self.feature_channels,
            self.atom_feature_dim,
            self.longitudinal_samples,
            self.transverse_samples,
            self.hidden_dim,
            self.pair_chunk_size,
        ) < 1:
            raise ValueError("Direct AtomPair dimensions must be positive")
        if min(self.ribbon_radius_pixels, self.image_size) <= 0.0:
            raise ValueError("Direct AtomPair geometry scales must be positive")


def enumerate_unordered_atom_pairs(
    atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all valid pairs without distance or neighborhood filtering."""

    if atom_mask.ndim != 2:
        raise ValueError("atom_mask must have shape [batch, atoms]")
    pairs = []
    samples = []
    for batch_index, mask in enumerate(atom_mask):
        valid = torch.where(mask)[0]
        current = torch.combinations(valid, r=2)
        if len(current):
            pairs.append(current)
            samples.append(
                torch.full(
                    (len(current),),
                    batch_index,
                    device=atom_mask.device,
                    dtype=torch.long,
                )
            )
    if not pairs:
        return (
            torch.empty((0, 2), device=atom_mask.device, dtype=torch.long),
            torch.empty(0, device=atom_mask.device, dtype=torch.long),
        )
    return torch.cat(pairs), torch.cat(samples)


class DirectAtomPairBond(nn.Module):
    """Classify final Bond semantics directly from stable AtomPair evidence."""

    def __init__(self, config: DirectAtomPairBondConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectAtomPairBondConfig()
        encoded_height = (self.config.transverse_samples + 1) // 2
        encoded_width = (self.config.longitudinal_samples + 3) // 4
        self.ribbon_encoder = nn.Sequential(
            nn.Conv2d(self.config.feature_channels, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=(1, 2), padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 160, 3, stride=2, padding=1),
            nn.GroupNorm(8, 160),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(160 * encoded_height * encoded_width, self.config.hidden_dim),
            nn.GELU(),
        )
        self.atom_encoder = nn.Sequential(
            nn.Linear(self.config.atom_feature_dim * 2, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.GELU(),
        )
        self.score_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(self.config.hidden_dim + 128 + 32, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        self.presence = nn.Linear(self.config.hidden_dim, 1)
        self.chemical = nn.Linear(self.config.hidden_dim, len(BOND_TYPES))

    def _sample_ribbons(
        self,
        feature_map: torch.Tensor,
        anchors: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        endpoints = anchors[pairs]
        begin = endpoints[:, 0]
        delta = endpoints[:, 1] - begin
        delta_pixels = delta * self.config.image_size
        lengths = torch.linalg.vector_norm(delta_pixels, dim=-1).clamp_min(1e-6)
        direction = delta_pixels / lengths[:, None]
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
        perpendicular = perpendicular / self.config.image_size
        longitudinal = torch.linspace(
            0.04,
            0.96,
            self.config.longitudinal_samples,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        transverse = torch.linspace(
            -self.config.ribbon_radius_pixels,
            self.config.ribbon_radius_pixels,
            self.config.transverse_samples,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        base = begin[:, None, :] + delta[:, None, :] * longitudinal[None, :, None]
        grid = base[:, None, :, :] + (
            perpendicular[:, None, None, :]
            * transverse[None, :, None, None]
        )
        grid = grid.mul(2.0).sub(1.0).reshape(
            1,
            len(pairs) * self.config.transverse_samples,
            self.config.longitudinal_samples,
            2,
        )
        sampled = F.grid_sample(
            feature_map[None],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(
            self.config.feature_channels,
            len(pairs),
            self.config.transverse_samples,
            self.config.longitudinal_samples,
        ).permute(1, 0, 2, 3)

    def _encode_visual(self, ribbons: torch.Tensor) -> torch.Tensor:
        forward = self.ribbon_encoder(ribbons)
        # Reversing a directed segment flips both its longitudinal axis and
        # the signed transverse axis.  Average both views so an unordered
        # AtomPair has one stable representation.
        reverse = self.ribbon_encoder(ribbons.flip((-2, -1)))
        return (forward + reverse) * 0.5

    def forward(
        self,
        feature_maps: torch.Tensor,
        atom_features: torch.Tensor,
        atom_anchors: torch.Tensor,
        atom_scores: torch.Tensor,
        pair_indices: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pair_indices.shape != (len(sample_indices), 2):
            raise ValueError("Direct AtomPair indices have an invalid shape")
        if not len(pair_indices):
            empty = feature_maps.new_empty((0, self.config.hidden_dim))
            return {
                "presence_logits": feature_maps.new_empty(0),
                "chemical_logits": feature_maps.new_empty((0, len(BOND_TYPES))),
                "hidden": empty,
            }
        visual_rows = []
        for sample_index in sample_indices.unique(sorted=True).tolist():
            selected = torch.where(sample_indices == sample_index)[0]
            for start in range(0, len(selected), self.config.pair_chunk_size):
                rows = selected[start : start + self.config.pair_chunk_size]
                ribbons = self._sample_ribbons(
                    feature_maps[sample_index],
                    atom_anchors[sample_index],
                    pair_indices[rows],
                )
                visual_rows.append((rows, self._encode_visual(ribbons)))
        encoded_rows = torch.cat([values for _, values in visual_rows])
        row_indices = torch.cat([rows for rows, _ in visual_rows])
        visual = encoded_rows[torch.argsort(row_indices)]

        first = atom_features[sample_indices, pair_indices[:, 0]]
        second = atom_features[sample_indices, pair_indices[:, 1]]
        atom_pair = torch.cat(((first + second) * 0.5, (first - second).abs()), dim=-1)
        first_score = atom_scores[sample_indices, pair_indices[:, 0]]
        second_score = atom_scores[sample_indices, pair_indices[:, 1]]
        score_pair = torch.stack(
            (
                (first_score + second_score) * 0.5,
                torch.minimum(first_score, second_score),
                torch.maximum(first_score, second_score),
            ),
            dim=-1,
        )
        hidden = self.shared(
            torch.cat(
                (
                    visual,
                    self.atom_encoder(atom_pair),
                    self.score_encoder(score_pair),
                ),
                dim=-1,
            )
        )
        return {
            "presence_logits": self.presence(hidden).squeeze(-1),
            "chemical_logits": self.chemical(hidden),
            "hidden": hidden,
        }


def direct_atom_pair_loss(
    output: dict[str, torch.Tensor],
    presence_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    positives = presence_targets.bool()
    positive_count = positives.sum().clamp_min(1)
    negative_count = (~positives).sum().clamp_min(1)
    presence_loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"].float(),
        presence_targets.float(),
        pos_weight=(negative_count / positive_count).clamp(1.0, 16.0),
    )
    zero = presence_loss * 0.0
    chemical_loss = (
        F.cross_entropy(
            output["chemical_logits"][positives].float(),
            chemical_targets[positives],
        )
        if bool(positives.any())
        else zero
    )
    return {
        "loss": presence_loss + chemical_loss,
        "presence_loss": presence_loss,
        "chemical_loss": chemical_loss,
    }
