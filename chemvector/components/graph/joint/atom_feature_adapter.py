"""Residual adapter between P0-I native and P0-G same-slot Atom features."""

from __future__ import annotations

import torch
from torch import nn


ATOM_FEATURE_DIM = 384
ATOM_FEATURE_ADAPTER_PARAMETERS = 147_840


class AtomFeatureAdapter(nn.Module):
    """Map native Atom features into the frozen relation-head feature space."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(ATOM_FEATURE_DIM, elementwise_affine=False)
        self.projection = nn.Linear(ATOM_FEATURE_DIM, ATOM_FEATURE_DIM, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != ATOM_FEATURE_ADAPTER_PARAMETERS:
            raise RuntimeError("Atom feature adapter parameter count changed")

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        if atom_features.ndim != 3 or atom_features.size(-1) != ATOM_FEATURE_DIM:
            raise ValueError(
                "Atom feature adapter requires [batch, atoms, 384] input"
            )
        return atom_features + self.projection(self.norm(atom_features))
