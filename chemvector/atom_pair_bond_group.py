"""Proposal-invariant Bond existence over predicted unordered AtomPairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES, DIRECTIONS


PROPOSAL_FEATURE_DIM = 147
GROUP_FEATURE_DIM = 2 * PROPOSAL_FEATURE_DIM + 1


@dataclass(frozen=True, slots=True)
class AtomPairBondGroupConfig:
    input_dim: int = GROUP_FEATURE_DIM
    hidden_dim: int = 128


class AtomPairBondGroupHead(nn.Module):
    def __init__(self, config: AtomPairBondGroupConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.display_head = nn.Linear(config.input_dim, len(BOND_TYPES))
        self.chemical_head = nn.Linear(config.input_dim, len(BOND_TYPES))
        self.direction_head = nn.Linear(config.input_dim, len(DIRECTIONS))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "presence_logits": self.network(features).squeeze(-1),
            "display_logits": self.display_head(features),
            "chemical_logits": self.chemical_head(features),
            "direction_logits": self.direction_head(features),
        }


def build_atom_pair_groups(
    output: dict[str, torch.Tensor],
    proposals: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | list[list[list[int]]]]:
    incidence = output["incidence_probabilities"]
    top = incidence.topk(min(2, incidence.size(-1)), dim=-1).values
    maximum = top[..., 0]
    second = top[..., 1] if top.size(-1) > 1 else torch.zeros_like(maximum)
    incidence_features = torch.stack((maximum, maximum - second), dim=-1).flatten(2)
    per_proposal = torch.cat(
        (
            proposals["bond_features"],
            proposals["bond_endpoints"].flatten(2),
            proposals["bond_types"],
            proposals["bond_scores"][..., None],
            output["bond_pre_presence_logits"][..., None],
            output["base_bond_presence_logits"][..., None],
            incidence_features,
        ),
        dim=-1,
    )
    if per_proposal.size(-1) != PROPOSAL_FEATURE_DIM:
        raise ValueError(
            f"AtomPair proposal feature dim {per_proposal.size(-1)} != "
            f"{PROPOSAL_FEATURE_DIM}"
        )

    group_features = []
    group_targets = []
    group_samples = []
    group_type_mask = []
    group_display_targets = []
    group_chemical_targets = []
    group_direction_targets = []
    group_members: list[list[list[int]]] = []
    group_atom_pairs: list[list[tuple[int, int]]] = []
    group_target_atom_pairs: list[list[tuple[int, int]]] = []
    predicted_refs = output["incidence_logits"].argmax(-1)
    for batch_index in range(proposals["bond_mask"].size(0)):
        groups: dict[tuple[int, int], list[int]] = {}
        for proposal_index in torch.where(proposals["bond_mask"][batch_index])[0].tolist():
            refs = predicted_refs[batch_index, proposal_index]
            key = tuple(sorted((int(refs[0]), int(refs[1]))))
            groups.setdefault(key, []).append(proposal_index)
        sample_members = []
        sample_atom_pairs = []
        sample_target_atom_pairs = []
        for atom_pair, members in groups.items():
            indices = torch.tensor(members, device=per_proposal.device)
            values = per_proposal[batch_index, indices]
            count = values.new_tensor([len(members)]).log1p()
            group_features.append(torch.cat((values.mean(0), values.amax(0), count)))
            group_targets.append(
                proposals["bond_presence_targets"][batch_index, indices].any()
                if "bond_presence_targets" in proposals
                else indices.new_tensor(False, dtype=torch.bool)
            )
            target_ids = (
                proposals["bond_target_indices"][batch_index, indices]
                if "bond_target_indices" in proposals
                else indices.new_full((len(indices),), -1)
            )
            target_ids = target_ids[target_ids >= 0].unique()
            supervised_type = targets is not None and len(target_ids) == 1
            group_type_mask.append(supervised_type)
            target_id = int(target_ids[0]) if supervised_type else 0
            group_display_targets.append(
                targets["bond_display"][batch_index, target_id]
                if supervised_type
                else indices.new_tensor(0)
            )
            group_chemical_targets.append(
                targets["bond_chemical"][batch_index, target_id]
                if supervised_type
                else indices.new_tensor(0)
            )
            group_direction_targets.append(
                targets["bond_direction"][batch_index, target_id]
                if supervised_type
                else indices.new_tensor(0)
            )
            group_samples.append(batch_index)
            sample_members.append(members)
            sample_atom_pairs.append(atom_pair)
            target_atom_pair = (-1, -1)
            if "atom_target_indices" in proposals:
                mapped = proposals["atom_target_indices"][
                    batch_index,
                    torch.tensor(atom_pair, device=per_proposal.device),
                ]
                if bool((mapped >= 0).all()) and int(mapped[0]) != int(mapped[1]):
                    target_atom_pair = tuple(sorted((int(mapped[0]), int(mapped[1]))))
            sample_target_atom_pairs.append(target_atom_pair)
        group_members.append(sample_members)
        group_atom_pairs.append(sample_atom_pairs)
        group_target_atom_pairs.append(sample_target_atom_pairs)
    if not group_features:
        empty = per_proposal.new_empty((0, GROUP_FEATURE_DIM))
        return {
            "features": empty,
            "targets": torch.empty(0, device=empty.device, dtype=torch.bool),
            "sample_indices": torch.empty(0, device=empty.device, dtype=torch.long),
            "type_mask": torch.empty(0, device=empty.device, dtype=torch.bool),
            "display_targets": torch.empty(0, device=empty.device, dtype=torch.long),
            "chemical_targets": torch.empty(0, device=empty.device, dtype=torch.long),
            "direction_targets": torch.empty(0, device=empty.device, dtype=torch.long),
            "members": group_members,
            "atom_pairs": group_atom_pairs,
            "target_atom_pairs": group_target_atom_pairs,
        }
    return {
        "features": torch.stack(group_features),
        "targets": torch.stack(group_targets).bool(),
        "sample_indices": torch.tensor(
            group_samples, device=per_proposal.device, dtype=torch.long
        ),
        "type_mask": torch.tensor(
            group_type_mask, device=per_proposal.device, dtype=torch.bool
        ),
        "display_targets": torch.stack(group_display_targets).long(),
        "chemical_targets": torch.stack(group_chemical_targets).long(),
        "direction_targets": torch.stack(group_direction_targets).long(),
        "members": group_members,
        "atom_pairs": group_atom_pairs,
        "target_atom_pairs": group_target_atom_pairs,
    }


def atom_pair_group_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    ranking_margin: float = 1.0,
) -> torch.Tensor:
    targets_float = targets.float()
    positives = targets_float.sum().clamp_min(1.0)
    negatives = (1.0 - targets_float).sum().clamp_min(1.0)
    positive_weight = (negatives / positives).clamp(1.0, 8.0)
    binary = F.binary_cross_entropy_with_logits(
        logits.float(), targets_float, pos_weight=positive_weight
    )
    ranking = []
    for sample_index in sample_indices.unique().tolist():
        sample = sample_indices == sample_index
        positive = logits[sample & targets]
        negative = logits[sample & ~targets]
        if len(positive) and len(negative):
            ranking.append(F.relu(ranking_margin - positive.min() + negative.max()))
    ranking_loss = torch.stack(ranking).mean() if ranking else logits.sum() * 0.0
    return binary + ranking_loss


def atom_pair_group_type_loss(
    output: dict[str, torch.Tensor],
    groups: dict[str, torch.Tensor],
    *,
    direction_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    mask = groups["type_mask"]
    zero = output["display_logits"].sum() * 0.0
    if not bool(mask.any()):
        return {
            "loss": zero,
            "display_loss": zero,
            "chemical_loss": zero,
            "direction_loss": zero,
        }
    display_loss = F.cross_entropy(
        output["display_logits"][mask].float(), groups["display_targets"][mask]
    )
    chemical_loss = F.cross_entropy(
        output["chemical_logits"][mask].float(), groups["chemical_targets"][mask]
    )
    direction_loss = F.cross_entropy(
        output["direction_logits"][mask].float(), groups["direction_targets"][mask]
    )
    return {
        "loss": display_loss + chemical_loss + direction_weight * direction_loss,
        "display_loss": display_loss,
        "chemical_loss": chemical_loss,
        "direction_loss": direction_loss,
    }


def collapse_atom_pair_group_outputs(
    proposals: dict[str, torch.Tensor],
    set_output: dict[str, torch.Tensor],
    group_output: dict[str, torch.Tensor],
    groups: dict[str, torch.Tensor | list[list[list[int]]]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict]]:
    """Expose one representative Bond proposal per predicted AtomPair."""

    bond_logits = torch.full_like(set_output["bond_presence_logits"], -1e4)
    bond_mask = torch.zeros_like(proposals["bond_mask"])
    bond_types = proposals["bond_types"].clone()
    diagnostics = []
    group_index = 0
    for batch_index, sample_groups in enumerate(groups["members"]):
        sample_diagnostics = {"groups": len(sample_groups), "representatives": []}
        for members in sample_groups:
            indices = torch.tensor(
                members,
                device=set_output["bond_presence_logits"].device,
                dtype=torch.long,
            )
            base_logits = set_output["bond_presence_logits"][batch_index, indices]
            representative = int(indices[base_logits.argmax()])
            bond_mask[batch_index, representative] = True
            bond_logits[batch_index, representative] = group_output[
                "presence_logits"
            ][group_index]
            bond_types[batch_index, representative, : len(BOND_TYPES)] = group_output[
                "display_logits"
            ][group_index].softmax(-1)
            bond_types[batch_index, representative, len(BOND_TYPES) :] = group_output[
                "chemical_logits"
            ][group_index].softmax(-1)
            sample_diagnostics["representatives"].append(
                {"proposal": representative, "members": list(members)}
            )
            group_index += 1
        diagnostics.append(sample_diagnostics)
    if group_index != len(group_output["presence_logits"]):
        raise ValueError("AtomPair group output count differs from decoded groups")
    collapsed_proposals = {
        **proposals,
        "bond_mask": bond_mask,
        "bond_types": bond_types,
    }
    collapsed_output = {
        **set_output,
        "bond_presence_logits": bond_logits,
    }
    return collapsed_proposals, collapsed_output, diagnostics
