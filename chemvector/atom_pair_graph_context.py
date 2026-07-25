"""Graph-context residual for joint AtomPair Bond presence decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_pair_bond_group import GROUP_FEATURE_DIM, atom_pair_group_loss


@dataclass(frozen=True, slots=True)
class AtomPairGraphContextConfig:
    input_dim: int = GROUP_FEATURE_DIM
    hidden_dim: int = 128
    maximum_atoms: int = 128

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.maximum_atoms) < 1:
            raise ValueError("Graph-context dimensions must be positive")


def flatten_atom_pairs(
    atom_pairs: list[list[tuple[int, int]]],
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = [pair for sample in atom_pairs for pair in sample]
    if not rows:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    return torch.tensor(rows, device=device, dtype=torch.long)


class AtomPairGraphContextHead(nn.Module):
    """Refine frozen local logits with endpoint and whole-set context."""

    def __init__(self, config: AtomPairGraphContextConfig) -> None:
        super().__init__()
        self.config = config
        self.local_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        context_dim = config.hidden_dim * 5 + 2
        self.residual = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        sample_indices: torch.Tensor,
        atom_pairs: torch.Tensor,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if not (
            len(features) == len(base_logits) == len(sample_indices) == len(atom_pairs)
        ):
            raise ValueError("Graph-context group tensors differ in length")
        if atom_pairs.ndim != 2 or atom_pairs.size(-1) != 2:
            raise ValueError("atom_pairs must have shape [groups,2]")
        if not len(features):
            return {
                "presence_logits": base_logits,
                "residual_logits": base_logits,
            }

        hidden = self.local_encoder(features)
        sample_mean = hidden.new_zeros((batch_size, self.config.hidden_dim))
        sample_max = hidden.new_full(
            (batch_size, self.config.hidden_dim), -torch.inf
        )
        sample_count = hidden.new_zeros((batch_size,))
        for sample_index in range(batch_size):
            mask = sample_indices == sample_index
            if not bool(mask.any()):
                sample_max[sample_index].zero_()
                continue
            values = hidden[mask]
            sample_mean[sample_index] = values.mean(0)
            sample_max[sample_index] = values.amax(0)
            sample_count[sample_index] = mask.sum()

        maximum_atoms = self.config.maximum_atoms
        if int(atom_pairs.max()) >= maximum_atoms or int(atom_pairs.min()) < 0:
            raise ValueError("AtomPair index exceeds graph-context atom capacity")
        first = hidden.new_zeros(hidden.shape)
        second = hidden.new_zeros(hidden.shape)
        for sample_index in range(batch_size):
            mask = sample_indices == sample_index
            if not bool(mask.any()):
                continue
            sample_pairs = atom_pairs[mask]
            incidence = (
                F.one_hot(sample_pairs[:, 0], maximum_atoms)
                + F.one_hot(sample_pairs[:, 1], maximum_atoms)
            ).to(hidden.dtype)
            endpoint_sum = incidence.T @ hidden[mask]
            endpoint_count = incidence.sum(0).clamp_min(1.0)
            endpoint_mean = endpoint_sum / endpoint_count[:, None]
            first[mask] = endpoint_mean[sample_pairs[:, 0]]
            second[mask] = endpoint_mean[sample_pairs[:, 1]]
        symmetric_endpoints = torch.cat(
            ((first + second) * 0.5, (first - second).abs()), dim=-1
        )
        normalized_group_count = (
            sample_count[sample_indices] / max(float(maximum_atoms), 1.0)
        )[:, None]
        context = torch.cat(
            (
                hidden,
                symmetric_endpoints,
                sample_mean[sample_indices],
                sample_max[sample_indices],
                base_logits[:, None],
                normalized_group_count,
            ),
            dim=-1,
        )
        residual = self.residual(context).squeeze(-1)
        return {
            "presence_logits": base_logits + residual,
            "residual_logits": residual,
        }


def load_atom_pair_graph_context_head(
    path: str | Path, device: torch.device
) -> tuple[AtomPairGraphContextHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "atom-pair-graph-context-0.1":
        raise ValueError("Unsupported AtomPair graph-context checkpoint schema")
    model = AtomPairGraphContextHead(
        AtomPairGraphContextConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


def refine_atom_pair_group_output(
    model: AtomPairGraphContextHead,
    group_output: dict[str, torch.Tensor],
    groups: dict,
    *,
    batch_size: int,
    residual_scale: float = 1.0,
    base_positive_protection_threshold: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    pairs = flatten_atom_pairs(
        groups["atom_pairs"], device=group_output["presence_logits"].device
    )
    context = model(
        groups["features"],
        group_output["presence_logits"],
        groups["sample_indices"],
        pairs,
        batch_size=batch_size,
    )
    refined_logits = (
        group_output["presence_logits"]
        + float(residual_scale) * context["residual_logits"]
    )
    if base_positive_protection_threshold is not None:
        protected = group_output["presence_logits"] >= float(
            base_positive_protection_threshold
        )
        refined_logits = torch.where(
            protected, group_output["presence_logits"], refined_logits
        )
    refined = {**group_output, "presence_logits": refined_logits}
    residual = context["residual_logits"].detach().float()
    diagnostics = {
        "groups": int(len(residual)),
        "residual_scale": float(residual_scale),
        "base_positive_protection_threshold": base_positive_protection_threshold,
        "mean_absolute_residual": (
            float(residual.abs().mean()) if len(residual) else 0.0
        ),
        "maximum_absolute_residual": (
            float(residual.abs().amax()) if len(residual) else 0.0
        ),
    }
    return refined, diagnostics


def atom_pair_graph_context_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    sample_indices: torch.Tensor,
    atom_pairs: torch.Tensor,
    *,
    batch_size: int,
    maximum_atoms: int,
    count_weight: float = 0.25,
    degree_weight: float = 0.25,
    residual_weight: float = 1e-4,
    positive_margin: float = 0.5,
    positive_margin_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    logits = output["presence_logits"]
    presence = atom_pair_group_loss(logits, targets, sample_indices)
    probabilities = logits.float().sigmoid()
    target_values = targets.float()

    predicted_counts = torch.stack(
        [probabilities[sample_indices == index].sum() for index in range(batch_size)]
    )
    target_counts = torch.stack(
        [target_values[sample_indices == index].sum() for index in range(batch_size)]
    )
    count_scale = target_counts.clamp_min(1.0)
    count = F.smooth_l1_loss(
        predicted_counts / count_scale,
        target_counts / count_scale,
    )

    degree_losses = []
    for sample_index in range(batch_size):
        mask = sample_indices == sample_index
        if not bool(mask.any()):
            continue
        sample_pairs = atom_pairs[mask]
        incidence = (
            F.one_hot(sample_pairs[:, 0], maximum_atoms)
            + F.one_hot(sample_pairs[:, 1], maximum_atoms)
        ).to(probabilities.dtype)
        predicted_degree = incidence.T @ probabilities[mask]
        target_degree = incidence.T @ target_values[mask]
        supervised_atoms = target_degree > 0
        if bool(supervised_atoms.any()):
            degree_losses.append(
                F.smooth_l1_loss(
                    predicted_degree[supervised_atoms],
                    target_degree[supervised_atoms],
                )
            )
    degree = torch.stack(degree_losses).mean() if degree_losses else presence * 0.0
    residual = output["residual_logits"].float().square().mean()
    positive_margin_loss = (
        F.relu(float(positive_margin) - logits[targets].float()).mean()
        if bool(targets.any())
        else presence * 0.0
    )
    total = (
        presence
        + count_weight * count
        + degree_weight * degree
        + residual_weight * residual
        + positive_margin_weight * positive_margin_loss
    )
    return {
        "loss": total,
        "presence_loss": presence,
        "count_loss": count,
        "degree_loss": degree,
        "residual_loss": residual,
        "positive_margin_loss": positive_margin_loss,
    }
