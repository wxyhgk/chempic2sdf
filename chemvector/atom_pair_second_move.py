"""A residual graph edit after frozen AtomPair one-move inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chemvector.atom_pair_one_move import (
    EDGE_ACTION_FEATURE_DIM,
    SWAP_INTERACTION_DIM,
    AtomPairAction,
    AtomPairOneMoveHead,
    _swap_interactions,
)


SECOND_MOVE_FEATURE_DIM = EDGE_ACTION_FEATURE_DIM + 2


@dataclass(frozen=True, slots=True)
class AtomPairSecondMoveConfig:
    input_dim: int = SECOND_MOVE_FEATURE_DIM
    hidden_dim: int = 64
    delete_candidates: int = 4
    add_candidates: int = 16
    maximum_groups: int = 256

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.delete_candidates,
            self.add_candidates,
            self.maximum_groups,
        ) < 1:
            raise ValueError("Second-move dimensions must be positive")


def build_second_move_edge_features(
    relation_features: torch.Tensor,
    base_logits: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if not (len(relation_features) == len(base_logits) == len(selected)):
        raise ValueError("Second-move feature tensors differ in length")
    if selected.dtype != torch.bool:
        raise ValueError("Second-move selected mask must be boolean")
    changed = selected != (base_logits >= 0.0)
    return torch.cat(
        (
            relation_features,
            base_logits.float()[:, None],
            selected.float()[:, None],
            changed.float()[:, None],
        ),
        dim=-1,
    )


def candidate_second_action_indices(
    base_logits: torch.Tensor,
    selected: torch.Tensor,
    *,
    config: AtomPairSecondMoveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(base_logits) != len(selected):
        raise ValueError("Second-move logits and selection differ in length")
    selected_indices = torch.where(selected)[0]
    unselected_indices = torch.where(~selected)[0]
    if len(selected_indices):
        order = torch.argsort(base_logits[selected_indices], stable=True)
        delete_indices = selected_indices[order[: config.delete_candidates]]
    else:
        delete_indices = selected_indices
    if len(unselected_indices):
        order = torch.argsort(
            base_logits[unselected_indices], descending=True, stable=True
        )
        add_indices = unselected_indices[order[: config.add_candidates]]
    else:
        add_indices = unselected_indices
    return delete_indices, add_indices


class AtomPairSecondMoveHead(nn.Module):
    """Score a second bounded graph edit after a frozen first-stage action."""

    def __init__(self, config: AtomPairSecondMoveConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        global_dim = config.hidden_dim * 2 + 3
        self.keep_head = nn.Sequential(
            nn.Linear(global_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        edge_action_dim = config.hidden_dim + global_dim
        self.delete_head = nn.Sequential(
            nn.Linear(edge_action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.add_head = nn.Sequential(
            nn.Linear(edge_action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.swap_head = nn.Sequential(
            nn.Linear(
                config.hidden_dim * 2 + global_dim + SWAP_INTERACTION_DIM,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        for head in (self.keep_head, self.delete_head, self.add_head, self.swap_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def score_sample(
        self,
        edge_features: torch.Tensor,
        relation_features: torch.Tensor,
        base_logits: torch.Tensor,
        atom_pairs: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, list[AtomPairAction]]:
        if not (
            len(edge_features)
            == len(relation_features)
            == len(base_logits)
            == len(atom_pairs)
            == len(selected)
        ):
            raise ValueError("Second-move sample tensors differ in length")
        if not len(edge_features):
            raise ValueError("Second-move scoring requires at least one group")
        if len(edge_features) > self.config.maximum_groups:
            raise ValueError("Second-move group count exceeds configured capacity")
        if selected.dtype != torch.bool:
            raise ValueError("Second-move selected mask must be boolean")

        hidden = self.edge_encoder(edge_features)
        changed = selected != (base_logits >= 0.0)
        global_feature = torch.cat(
            (
                hidden.mean(0),
                hidden.amax(0),
                hidden.new_tensor(
                    (
                        float(selected.sum()) / self.config.maximum_groups,
                        float(len(selected)) / self.config.maximum_groups,
                        float(changed.sum()) / self.config.maximum_groups,
                    )
                ),
            )
        )
        scores = [self.keep_head(global_feature).squeeze(-1)]
        actions = [AtomPairAction("keep")]
        delete_indices, add_indices = candidate_second_action_indices(
            base_logits, selected, config=self.config
        )
        if len(delete_indices):
            delete_input = torch.cat(
                (
                    hidden[delete_indices],
                    global_feature.expand(len(delete_indices), -1),
                ),
                dim=-1,
            )
            scores.extend(self.delete_head(delete_input).squeeze(-1).unbind())
            actions.extend(
                AtomPairAction("delete", delete_index=int(index))
                for index in delete_indices.tolist()
            )
        if len(add_indices):
            add_input = torch.cat(
                (
                    hidden[add_indices],
                    global_feature.expand(len(add_indices), -1),
                ),
                dim=-1,
            )
            scores.extend(self.add_head(add_input).squeeze(-1).unbind())
            actions.extend(
                AtomPairAction("add", add_index=int(index))
                for index in add_indices.tolist()
            )
        if len(delete_indices) and len(add_indices):
            delete_hidden = hidden[delete_indices][:, None].expand(
                -1, len(add_indices), -1
            )
            add_hidden = hidden[add_indices][None].expand(
                len(delete_indices), -1, -1
            )
            interactions = _swap_interactions(
                relation_features,
                atom_pairs,
                base_logits,
                delete_indices,
                add_indices,
            ).reshape(len(delete_indices), len(add_indices), -1)
            swap_input = torch.cat(
                (
                    delete_hidden,
                    add_hidden,
                    global_feature.expand(
                        len(delete_indices), len(add_indices), -1
                    ),
                    interactions,
                ),
                dim=-1,
            )
            scores.extend(self.swap_head(swap_input).reshape(-1).unbind())
            actions.extend(
                AtomPairAction(
                    "swap",
                    delete_index=int(delete_index),
                    add_index=int(add_index),
                )
                for delete_index in delete_indices.tolist()
                for add_index in add_indices.tolist()
            )
        return torch.stack(scores), actions


@torch.no_grad()
def initialize_second_move_from_first(
    second: AtomPairSecondMoveHead,
    first: AtomPairOneMoveHead,
) -> None:
    if second.config.hidden_dim != first.config.hidden_dim:
        raise ValueError("First and second move hidden dimensions must match")
    if (
        second.config.delete_candidates != first.config.delete_candidates
        or second.config.add_candidates != first.config.add_candidates
        or second.config.maximum_groups != first.config.maximum_groups
    ):
        raise ValueError("First and second move candidate configurations must match")

    second.edge_encoder[0].weight.zero_()
    second.edge_encoder[0].weight[:, : first.config.input_dim].copy_(
        first.edge_encoder[0].weight
    )
    second.edge_encoder[0].bias.copy_(first.edge_encoder[0].bias)
    second.edge_encoder[2].load_state_dict(first.edge_encoder[2].state_dict())

    hidden = first.config.hidden_dim
    first_global = hidden * 2 + 2
    second_global = hidden * 2 + 3

    def copy_regular_head(second_head: nn.Sequential, first_head: nn.Sequential) -> None:
        second_head[0].weight.zero_()
        second_head[0].weight[:, : first_head[0].weight.shape[1]].copy_(
            first_head[0].weight
        )
        second_head[0].bias.copy_(first_head[0].bias)
        second_head[-1].load_state_dict(first_head[-1].state_dict())

    copy_regular_head(second.keep_head, first.keep_head)
    copy_regular_head(second.delete_head, first.delete_head)
    copy_regular_head(second.add_head, first.add_head)

    second.swap_head[0].weight.zero_()
    prefix = hidden * 2 + first_global
    second.swap_head[0].weight[:, :prefix].copy_(
        first.swap_head[0].weight[:, :prefix]
    )
    second.swap_head[0].weight[
        :, hidden * 2 + second_global : hidden * 2 + second_global + SWAP_INTERACTION_DIM
    ].copy_(first.swap_head[0].weight[:, prefix:])
    second.swap_head[0].bias.copy_(first.swap_head[0].bias)
    second.swap_head[-1].load_state_dict(first.swap_head[-1].state_dict())


def target_second_action_index(
    actions: list[AtomPairAction],
    selected: torch.Tensor,
    targets: torch.Tensor,
) -> int | None:
    false_positive = torch.where(selected & ~targets)[0].tolist()
    false_negative = torch.where(~selected & targets)[0].tolist()
    if not false_positive and not false_negative:
        target = AtomPairAction("keep")
    elif len(false_positive) == 1 and not false_negative:
        target = AtomPairAction("delete", delete_index=false_positive[0])
    elif not false_positive and len(false_negative) == 1:
        target = AtomPairAction("add", add_index=false_negative[0])
    elif len(false_positive) == 1 and len(false_negative) == 1:
        target = AtomPairAction(
            "swap",
            delete_index=false_positive[0],
            add_index=false_negative[0],
        )
    else:
        return None
    try:
        return actions.index(target)
    except ValueError:
        return None


def load_atom_pair_second_move_head(
    path: str | Path,
    device: torch.device,
) -> tuple[AtomPairSecondMoveHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "atom-pair-second-move-0.1":
        raise ValueError("Unsupported AtomPair second-move checkpoint schema")
    model = AtomPairSecondMoveHead(
        AtomPairSecondMoveConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload
