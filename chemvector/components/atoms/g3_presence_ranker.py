"""Frozen G3 Atom-head adapter for canonical presence ranking."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.components.graph.joint.g3_hierarchical_graph import (
    build_canonical_atom_slot_targets,
    select_count_conditioned_atom_slots,
)


G3_PRESENCE_HIDDEN_CHANNELS = 256
G3_PRESENCE_SLOTS = 3
G3_PRESENCE_BOTTLENECK_CHANNELS = 64
G3_PRESENCE_RANKER_PARAMETERS = 16_643


class _G3PresenceResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(
            G3_PRESENCE_HIDDEN_CHANNELS,
            G3_PRESENCE_BOTTLENECK_CHANNELS,
            kernel_size=1,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(
            G3_PRESENCE_BOTTLENECK_CHANNELS,
            G3_PRESENCE_SLOTS,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.activation(self.input_projection(hidden)))


class G3FrozenAtomPresenceRanker(nn.Module):
    """Add a trainable ranking residual to a fully frozen G3 Atom head."""

    def __init__(
        self,
        atom_head: nn.Module,
        config,
        bottleneck_channels: int = G3_PRESENCE_BOTTLENECK_CHANNELS,
    ) -> None:
        super().__init__()
        if int(config.hidden_channels) != G3_PRESENCE_HIDDEN_CHANNELS:
            raise ValueError("G3 presence ranker requires hidden_channels=256")
        if int(config.atom_slots_per_cell) != G3_PRESENCE_SLOTS:
            raise ValueError("G3 presence ranker requires atom_slots_per_cell=3")
        if bottleneck_channels != G3_PRESENCE_BOTTLENECK_CHANNELS:
            raise ValueError("G3 presence ranker requires bottleneck_channels=64")
        if int(config.maximum_atoms) < 1:
            raise ValueError("G3 presence ranker requires positive Atom capacity")

        self.atom_head = atom_head.requires_grad_(False).eval()
        self.config = config
        self.presence_ranker = _G3PresenceResidual()
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        if trainable_parameters != G3_PRESENCE_RANKER_PARAMETERS:
            raise RuntimeError("G3 presence ranker parameter count changed")

    def train(self, mode: bool = True):
        super().train(mode)
        self.atom_head.eval()
        return self

    def forward_dense(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the frozen Atom head and replace only its ranking logits."""

        self.atom_head.eval()
        with torch.no_grad():
            base = self.atom_head(features)
        required = {"object_features", "presence_logits", "anchors", "count_logits"}
        missing = required - base.keys()
        if missing:
            raise ValueError(
                f"G3 frozen Atom head is missing dense outputs: {sorted(missing)}"
            )
        hidden = base["object_features"]
        base_logits = base["presence_logits"]
        if hidden.ndim != 4 or hidden.size(1) != G3_PRESENCE_HIDDEN_CHANNELS:
            raise ValueError(
                "G3 presence ranker hidden must have shape [batch, 256, h, w]"
            )
        if base_logits.shape != (
            hidden.size(0),
            G3_PRESENCE_SLOTS,
            hidden.size(2),
            hidden.size(3),
        ):
            raise ValueError(
                "G3 base presence logits must have shape [batch, 3, h, w]"
            )
        residual = self.presence_ranker(hidden.detach())
        ranking_logits = base_logits.detach() + residual
        ranked = dict(base)
        ranked.update(
            {
                "base_presence_logits": base_logits.detach(),
                "presence_logits": ranking_logits,
                "presence_residual": residual,
            }
        )
        return ranked

    def forward(
        self,
        features: torch.Tensor,
        *,
        atom_counts: torch.Tensor | None = None,
        atom_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        dense = self.forward_dense(features)
        selected_counts = (
            dense["count_logits"].argmax(dim=-1)
            if atom_counts is None
            else atom_counts.to(features.device).long()
        )
        atoms = select_count_conditioned_atom_slots(
            dense,
            selected_counts,
            maximum_atoms=int(self.config.maximum_atoms),
            candidate_mask=atom_candidate_mask,
        )
        return {
            **atoms,
            "atom_count_logits": dense["count_logits"],
            "atom_dense_output": dense,
        }


def canonical_presence_ranking_loss(
    dense_output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    negative_multiplier: int = 4,
) -> dict[str, torch.Tensor]:
    """Rank every canonical positive above the hardest ``4 * K`` negatives."""

    if (
        isinstance(negative_multiplier, bool)
        or not isinstance(negative_multiplier, int)
        or negative_multiplier < 1
    ):
        raise ValueError("G3 presence negative_multiplier must be a positive integer")
    canonical = build_canonical_atom_slot_targets(dense_output, targets)
    logits = dense_output["presence_logits"]
    positives = canonical["positive"]
    counts = targets["atom_mask"].sum(dim=1).long()
    if logits.shape != positives.shape:
        raise ValueError("G3 presence logits and canonical targets differ")

    sample_losses = []
    hard_negative_count = 0
    for batch_index in range(logits.size(0)):
        count = int(counts[batch_index])
        positive_mask = positives[batch_index].flatten()
        if count < 1 or int(positive_mask.sum()) != count:
            raise ValueError("G3 canonical positive count must equal positive GT count")
        scores = logits[batch_index].float().flatten()
        positive_scores = scores[positive_mask]
        negative_scores = scores[~positive_mask]
        if not len(negative_scores):
            raise ValueError("G3 presence ranking requires negative candidates")
        selected_negatives = min(negative_multiplier * count, len(negative_scores))
        hard_negatives = negative_scores.topk(selected_negatives).values
        sample_losses.append(
            F.softplus(hard_negatives[:, None] - positive_scores[None, :]).mean()
        )
        hard_negative_count += selected_negatives

    loss = torch.stack(sample_losses).mean()
    return {
        "loss": loss,
        "positive_count": positives.sum().detach().to(dtype=torch.float32),
        "hard_negative_count": loss.new_tensor(float(hard_negative_count)),
    }
