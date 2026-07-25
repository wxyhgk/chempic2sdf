"""Set-level keep/add/delete/swap actions over AtomPair bond groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_pair_deletion_veto import RELATION_FEATURE_DIM


EDGE_ACTION_FEATURE_DIM = RELATION_FEATURE_DIM + 1
SWAP_INTERACTION_DIM = 4


@dataclass(frozen=True, slots=True)
class AtomPairOneMoveConfig:
    input_dim: int = EDGE_ACTION_FEATURE_DIM
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
            raise ValueError("One-move dimensions must be positive")


@dataclass(frozen=True, slots=True)
class AtomPairAction:
    kind: str
    delete_index: int = -1
    add_index: int = -1

    def __post_init__(self) -> None:
        if self.kind not in {"keep", "delete", "add", "swap"}:
            raise ValueError(f"Unsupported AtomPair action: {self.kind}")


def candidate_action_indices(
    base_logits: torch.Tensor,
    *,
    config: AtomPairOneMoveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = base_logits >= 0.0
    selected_indices = torch.where(selected)[0]
    unselected_indices = torch.where(~selected)[0]
    if len(selected_indices):
        order = torch.argsort(base_logits[selected_indices], stable=True)
        delete_indices = selected_indices[order[: config.delete_candidates]]
    else:
        delete_indices = selected_indices
    if len(unselected_indices):
        order = torch.argsort(base_logits[unselected_indices], descending=True, stable=True)
        add_indices = unselected_indices[order[: config.add_candidates]]
    else:
        add_indices = unselected_indices
    return delete_indices, add_indices


def _swap_interactions(
    relation_features: torch.Tensor,
    atom_pairs: torch.Tensor,
    base_logits: torch.Tensor,
    delete_indices: torch.Tensor,
    add_indices: torch.Tensor,
) -> torch.Tensor:
    if not len(delete_indices) or not len(add_indices):
        return relation_features.new_empty((0, SWAP_INTERACTION_DIM))
    delete_pairs = atom_pairs[delete_indices][:, None]
    add_pairs = atom_pairs[add_indices][None]
    shared = (
        (delete_pairs[..., 0, None] == add_pairs[..., None, 0])
        | (delete_pairs[..., 0, None] == add_pairs[..., None, 1])
        | (delete_pairs[..., 1, None] == add_pairs[..., None, 0])
        | (delete_pairs[..., 1, None] == add_pairs[..., None, 1])
    ).any(-1).float()
    delete_direction = relation_features[delete_indices, 1:3]
    add_direction = relation_features[add_indices, 1:3]
    direction_alignment = (
        delete_direction[:, None] * add_direction[None]
    ).sum(-1).abs()
    delete_length = relation_features[delete_indices, 0][:, None].clamp_min(1e-4)
    add_length = relation_features[add_indices, 0][None].clamp_min(1e-4)
    length_ratio = torch.log(delete_length / add_length).abs().clamp_max(4.0) / 4.0
    logit_delta = (
        base_logits[delete_indices][:, None] - base_logits[add_indices][None]
    ).clamp(-8.0, 8.0) / 8.0
    return torch.stack(
        (shared, direction_alignment, length_ratio, logit_delta), dim=-1
    ).reshape(-1, SWAP_INTERACTION_DIM)


class AtomPairOneMoveHead(nn.Module):
    """Score one graph-level edit action against keeping the current graph."""

    def __init__(self, config: AtomPairOneMoveConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        global_dim = config.hidden_dim * 2 + 2
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
    ) -> tuple[torch.Tensor, list[AtomPairAction]]:
        if not (
            len(edge_features)
            == len(relation_features)
            == len(base_logits)
            == len(atom_pairs)
        ):
            raise ValueError("One-move sample tensors differ in length")
        if not len(edge_features):
            raise ValueError("One-move scoring requires at least one group")
        if len(edge_features) > self.config.maximum_groups:
            raise ValueError("One-move group count exceeds configured capacity")
        hidden = self.edge_encoder(edge_features)
        selected = base_logits >= 0.0
        global_feature = torch.cat(
            (
                hidden.mean(0),
                hidden.amax(0),
                hidden.new_tensor(
                    (
                        float(selected.sum()) / self.config.maximum_groups,
                        float(len(selected)) / self.config.maximum_groups,
                    )
                ),
            )
        )
        scores = [self.keep_head(global_feature).squeeze(-1)]
        actions = [AtomPairAction("keep")]
        delete_indices, add_indices = candidate_action_indices(
            base_logits, config=self.config
        )
        if len(delete_indices):
            delete_input = torch.cat(
                (
                    hidden[delete_indices],
                    global_feature.expand(len(delete_indices), -1),
                ),
                dim=-1,
            )
            delete_scores = self.delete_head(delete_input).squeeze(-1)
            scores.extend(delete_scores.unbind())
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
            add_scores = self.add_head(add_input).squeeze(-1)
            scores.extend(add_scores.unbind())
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
            swap_scores = self.swap_head(swap_input).reshape(-1)
            scores.extend(swap_scores.unbind())
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


def target_action_index(
    actions: list[AtomPairAction],
    base_logits: torch.Tensor,
    targets: torch.Tensor,
) -> int | None:
    selected = base_logits >= 0.0
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


def apply_atom_pair_action(
    selected: torch.Tensor,
    action: AtomPairAction,
) -> torch.Tensor:
    output = selected.clone()
    if action.kind in {"delete", "swap"}:
        output[action.delete_index] = False
    if action.kind in {"add", "swap"}:
        output[action.add_index] = True
    return output


def choose_atom_pair_action(
    scores: torch.Tensor,
    actions: list[AtomPairAction],
    *,
    minimum_margin: float,
) -> tuple[AtomPairAction, float]:
    if len(scores) != len(actions) or not len(actions):
        raise ValueError("One-move scores and actions differ in length")
    keep_score = scores[0]
    if len(scores) == 1:
        return actions[0], 0.0
    best_non_keep = int(scores[1:].argmax()) + 1
    margin = float((scores[best_non_keep] - keep_score).detach())
    if margin <= float(minimum_margin):
        return actions[0], margin
    return actions[best_non_keep], margin


def one_move_loss(
    scores: torch.Tensor,
    target_index: int,
    *,
    edit_weight: float = 8.0,
) -> torch.Tensor:
    target = torch.tensor((target_index,), device=scores.device, dtype=torch.long)
    loss = F.cross_entropy(scores[None].float(), target)
    return loss * (float(edit_weight) if target_index != 0 else 1.0)


def load_atom_pair_one_move_head(
    path: str | Path,
    device: torch.device,
) -> tuple[AtomPairOneMoveHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "atom-pair-one-move-0.1":
        raise ValueError("Unsupported AtomPair one-move checkpoint schema")
    model = AtomPairOneMoveHead(AtomPairOneMoveConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload
