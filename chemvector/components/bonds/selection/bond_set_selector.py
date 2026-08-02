"""Whole-set selection and type correction for Bond Object proposals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES


@dataclass(frozen=True, slots=True)
class BondSetSelectorConfig:
    input_dim: int = 154
    d_model: int = 96
    layers: int = 3
    heads: int = 4
    ffn_dim: int = 384
    maximum_bonds: int = 128
    type_prior: float = 2.0

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.d_model,
            self.layers,
            self.heads,
            self.ffn_dim,
            self.maximum_bonds,
        ) < 1:
            raise ValueError("Bond set selector dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("Bond set selector heads must divide d_model")
        if self.type_prior < 0.0:
            raise ValueError("Bond set selector type prior must be nonnegative")


class BondSetSelector(nn.Module):
    def __init__(self, config: BondSetSelectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or BondSetSelectorConfig()
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.heads,
            dim_feedforward=self.config.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, self.config.layers, norm=nn.LayerNorm(self.config.d_model)
        )
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.keep = nn.Linear(self.config.d_model, 1)
        self.count = nn.Linear(self.config.d_model, self.config.maximum_bonds + 1)
        self.display_residual = nn.Linear(self.config.d_model, len(BOND_TYPES))
        self.chemical_residual = nn.Linear(self.config.d_model, len(BOND_TYPES))
        for head in (self.display_residual, self.chemical_residual):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        display_ids: torch.Tensor,
        chemical_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.size(-1) != self.config.input_dim:
            raise ValueError("Bond candidate features have an invalid shape")
        if mask.shape != features.shape[:2]:
            raise ValueError("Bond candidate mask differs from features")
        if display_ids.shape != mask.shape or chemical_ids.shape != mask.shape:
            raise ValueError("Bond type IDs differ from candidate mask")
        hidden = self.input_projection(features)
        global_token = self.global_token.expand(features.size(0), -1, -1)
        hidden = torch.cat((global_token, hidden), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    (features.size(0), 1), device=mask.device, dtype=torch.bool
                ),
                ~mask,
            ),
            dim=1,
        )
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        global_hidden = hidden[:, 0]
        candidate_hidden = hidden[:, 1:]
        display_prior = F.one_hot(
            display_ids.long(), num_classes=len(BOND_TYPES)
        ).float() * self.config.type_prior
        chemical_prior = F.one_hot(
            chemical_ids.long(), num_classes=len(BOND_TYPES)
        ).float() * self.config.type_prior
        return {
            "keep_logits": self.keep(candidate_hidden).squeeze(-1),
            "count_logits": self.count(global_hidden),
            "display_logits": display_prior
            + self.display_residual(candidate_hidden),
            "chemical_logits": chemical_prior
            + self.chemical_residual(candidate_hidden),
        }


def bond_set_selector_loss(
    output: dict[str, torch.Tensor],
    mask: torch.Tensor,
    keep_targets: torch.Tensor,
    count_targets: torch.Tensor,
    display_targets: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    count_weight: float = 1.0,
    set_margin_weight: float = 0.5,
    type_weight: float = 1.0,
    set_margin: float = 1.0,
) -> dict[str, torch.Tensor]:
    valid_keep = output["keep_logits"][mask]
    valid_targets = keep_targets[mask].float()
    positives = valid_targets.sum().clamp_min(1.0)
    negatives = (len(valid_targets) - positives).clamp_min(1.0)
    keep_loss = F.binary_cross_entropy_with_logits(
        valid_keep, valid_targets, pos_weight=(negatives / positives).clamp_max(16.0)
    )
    count_loss = F.cross_entropy(output["count_logits"], count_targets.long())
    margin_losses = []
    for batch_index in range(len(mask)):
        positive = mask[batch_index] & keep_targets[batch_index]
        negative = mask[batch_index] & ~keep_targets[batch_index]
        if bool(positive.any()) and bool(negative.any()):
            margin_losses.append(
                F.relu(
                    set_margin
                    - output["keep_logits"][batch_index, positive].min()
                    + output["keep_logits"][batch_index, negative].max()
                )
            )
    margin_loss = (
        torch.stack(margin_losses).mean()
        if margin_losses
        else keep_loss * 0.0
    )
    positive = mask & keep_targets
    if bool(positive.any()):
        display_loss = F.cross_entropy(
            output["display_logits"][positive], display_targets[positive]
        )
        chemical_loss = F.cross_entropy(
            output["chemical_logits"][positive], chemical_targets[positive]
        )
    else:
        display_loss = output["display_logits"].sum() * 0.0
        chemical_loss = output["chemical_logits"].sum() * 0.0
    return {
        "loss": keep_loss
        + count_weight * count_loss
        + set_margin_weight * margin_loss
        + type_weight * (display_loss + chemical_loss),
        "keep_loss": keep_loss,
        "count_loss": count_loss,
        "set_margin_loss": margin_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
    }


@torch.no_grad()
def decode_bond_set(
    output: dict[str, torch.Tensor],
    pairs: torch.Tensor,
    mask: torch.Tensor,
    count_override: torch.Tensor | None = None,
) -> list[list[int]]:
    if pairs.shape != (*mask.shape, 2):
        raise ValueError("Bond candidate pairs differ from mask")
    decoded = []
    counts = (
        count_override
        if count_override is not None
        else output["count_logits"].argmax(-1)
    )
    if counts.shape != mask.shape[:1]:
        raise ValueError("Bond count override differs from candidate batch")
    for batch_index in range(len(mask)):
        desired = min(int(counts[batch_index]), int(mask[batch_index].sum()))
        order = output["keep_logits"][batch_index].argsort(descending=True)
        selected = []
        used_pairs: set[tuple[int, int]] = set()
        for candidate_index in order.tolist():
            if not bool(mask[batch_index, candidate_index]):
                continue
            begin, end = pairs[batch_index, candidate_index].tolist()
            pair = tuple(sorted((int(begin), int(end))))
            if pair[0] == pair[1] or pair in used_pairs:
                continue
            selected.append(candidate_index)
            used_pairs.add(pair)
            if len(selected) >= desired:
                break
        decoded.append(selected)
    return decoded


@torch.no_grad()
def decode_bond_set_threshold(
    keep_logits: torch.Tensor,
    pairs: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
) -> list[list[int]]:
    """Keep thresholded candidates while enforcing one Bond per Junction pair."""

    if keep_logits.shape != mask.shape or pairs.shape != (*mask.shape, 2):
        raise ValueError("Threshold Bond decode inputs have incompatible shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Bond keep threshold must be in [0, 1]")
    decoded = []
    probabilities = keep_logits.sigmoid()
    for batch_index in range(len(mask)):
        order = probabilities[batch_index].argsort(descending=True)
        selected = []
        used_pairs: set[tuple[int, int]] = set()
        for candidate_index in order.tolist():
            if not bool(mask[batch_index, candidate_index]):
                continue
            if float(probabilities[batch_index, candidate_index]) < threshold:
                break
            begin, end = pairs[batch_index, candidate_index].tolist()
            pair = tuple(sorted((int(begin), int(end))))
            if pair[0] == pair[1] or pair in used_pairs:
                continue
            selected.append(candidate_index)
            used_pairs.add(pair)
        decoded.append(selected)
    return decoded


@torch.no_grad()
def structured_unique_bond_counts(
    candidate_logits: torch.Tensor,
    pairs: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.25,
) -> torch.Tensor:
    """Count unique non-self Junction pairs above a detector threshold."""

    if candidate_logits.shape != mask.shape or pairs.shape != (*mask.shape, 2):
        raise ValueError("Structured Bond count inputs have incompatible shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Structured Bond count threshold must be in [0, 1]")
    keep = mask & (candidate_logits.sigmoid() >= threshold)
    counts = []
    for batch_index in range(len(mask)):
        unique_pairs = {
            tuple(sorted((int(begin), int(end))))
            for begin, end in pairs[batch_index, keep[batch_index]].tolist()
            if int(begin) != int(end)
        }
        counts.append(len(unique_pairs))
    return torch.tensor(counts, device=mask.device, dtype=torch.long)
