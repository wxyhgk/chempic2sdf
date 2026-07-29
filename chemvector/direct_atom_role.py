"""Anchor-centered multi-scale TEXT/IMPLICIT AtomObject role classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DirectAtomRoleConfig:
    radii_pixels: tuple[float, ...] = (12.0, 20.0, 32.0)
    patch_size: int = 32
    image_size: float = 768.0
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if not self.radii_pixels or min(self.radii_pixels) <= 0.0:
            raise ValueError("Atom role radii must be positive")
        if min(self.patch_size, self.hidden_dim) < 1 or self.image_size <= 0.0:
            raise ValueError("Atom role dimensions must be positive")


def enumerate_selected_atoms(atom_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if atom_mask.ndim != 2:
        raise ValueError("atom_mask must have shape [batch, atoms]")
    atoms = []
    samples = []
    for batch_index, mask in enumerate(atom_mask):
        selected = torch.where(mask)[0]
        atoms.append(selected)
        samples.append(torch.full_like(selected, batch_index))
    if not atoms or not sum(len(row) for row in atoms):
        empty = torch.empty(0, device=atom_mask.device, dtype=torch.long)
        return empty, empty
    return torch.cat(atoms), torch.cat(samples)


class DirectAtomRoleGate(nn.Module):
    def __init__(self, config: DirectAtomRoleConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectAtomRoleConfig()
        channels = 3 * len(self.config.radii_pixels)
        encoded_size = (self.config.patch_size + 7) // 8
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(96 * encoded_size * encoded_size, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def _sample_scale(
        self, pixels: torch.Tensor, anchors: torch.Tensor, radius: float
    ) -> torch.Tensor:
        offsets = torch.linspace(
            -radius / self.config.image_size,
            radius / self.config.image_size,
            self.config.patch_size,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        y, x = torch.meshgrid(offsets, offsets, indexing="ij")
        local = torch.stack((x, y), dim=-1)
        grid = anchors[:, None, None, :] + local[None]
        grid = grid.mul(2.0).sub(1.0).reshape(
            1,
            len(anchors) * self.config.patch_size,
            self.config.patch_size,
            2,
        )
        sampled = F.grid_sample(
            pixels[None], grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(
            3, len(anchors), self.config.patch_size, self.config.patch_size
        ).permute(1, 0, 2, 3)

    def forward(
        self,
        pixels: torch.Tensor,
        anchors: torch.Tensor,
        atom_indices: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        if atom_indices.shape != sample_indices.shape:
            raise ValueError("Atom role indices have incompatible shapes")
        if not len(atom_indices):
            return pixels.new_empty(0)
        rows = []
        values = []
        for sample_index in sample_indices.unique(sorted=True).tolist():
            selected = torch.where(sample_indices == sample_index)[0]
            centers = anchors[sample_index, atom_indices[selected]]
            patches = torch.cat(
                [
                    self._sample_scale(pixels[sample_index], centers, radius)
                    for radius in self.config.radii_pixels
                ],
                dim=1,
            )
            rows.append(selected)
            values.append(self.encoder(patches).squeeze(-1))
        order = torch.argsort(torch.cat(rows))
        return torch.cat(values)[order]
