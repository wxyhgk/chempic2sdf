"""CPU-only contract for group existence, ranking, and abstention.

This module deliberately contains no model, file, checkpoint, optimizer, or
GPU dependency.  It is a mathematical interface for a *future* group
selector.  In particular, a confirmed-negative group is supervised only by
the group-existence target ``0``; it never enters a winner/argmax target.

The contract separates three questions that the failed pointwise shadow
collapsed into one:

* ``existence_logit``: does this candidate group contain a recoverable node?
* ``candidate_logits``: if it exists, which candidate ranks first?
* abstention: is the existence evidence and the winner margin sufficient to
  emit a candidate at all?

Unknown members are not hard negatives by default.  The optional
``negative_slots`` field permits explicitly reviewed negatives.  A strict
positive group may therefore contribute an existence loss without silently
turning every unknown alternative into a negative.  No training is performed
by this module and every result is marked ``training_authorized=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class AbstainGroupSpec:
    """Ownership and reviewed labels for one candidate group.

    ``candidate_slots`` contains all candidates passed to the group scorer.
    ``positive_slots`` are reviewed recoverable candidates.  ``negative_slots``
    are reviewed non-candidates and are optional; all remaining candidates are
    unknown and receive no pairwise negative gradient.  A group with no
    positives must explicitly set ``existence_target=False``; this is the
    fail-closed representation of a confirmed-negative group.

    ``protected`` means an already selected baseline prefix.  It is hard
    locked and contributes no loss.  ``exclusion_reason`` excludes a group
    from supervision for an auditable reason.
    """

    node_id: Hashable
    candidate_slots: tuple[int, ...]
    positive_slots: tuple[int, ...] = ()
    negative_slots: tuple[int, ...] = ()
    existence_target: bool | None = None
    protected: bool = False
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GroupAbstainResult:
    total: torch.Tensor
    existence_loss: torch.Tensor
    ranking_loss: torch.Tensor
    existence_targets: tuple[tuple[Hashable, bool], ...]
    active_node_ids: tuple[Hashable, ...]
    protected_node_ids: tuple[Hashable, ...]
    excluded_nodes: tuple[tuple[Hashable, str], ...]
    supervised_slots: tuple[int, ...]
    unknown_slots: tuple[int, ...]
    training_authorized: bool = False


@dataclass(frozen=True, slots=True)
class AbstainPolicy:
    """Fixed inference gates; values are logits, not fitted probabilities."""

    existence_threshold: float = 0.0
    winner_threshold: float = 0.0
    margin_threshold: float = 0.0

    def __post_init__(self) -> None:
        for value in (
            self.existence_threshold,
            self.winner_threshold,
            self.margin_threshold,
        ):
            if not isinstance(value, (float, int)) or not torch.isfinite(
                torch.tensor(float(value))
            ):
                raise ValueError("Abstain thresholds must be finite scalars")
        if self.margin_threshold < 0:
            raise ValueError("margin_threshold must be nonnegative")


@dataclass(frozen=True, slots=True)
class GroupDecision:
    status: str
    winner_slot: int | None
    winner_logit: float | None
    margin: float | None
    training_authorized: bool = False


def _validate_logits(logits: torch.Tensor, *, name: str) -> None:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 1
        or logits.device.type != "cpu"
        or logits.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError(f"{name} must be finite one-dimensional CPU float32/float64")


def _validate_groups(
    candidate_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    groups: Sequence[AbstainGroupSpec],
) -> tuple[tuple[AbstainGroupSpec, ...], set[int]]:
    _validate_logits(candidate_logits, name="candidate_logits")
    _validate_logits(existence_logits, name="existence_logits")
    groups = tuple(groups)
    if existence_logits.numel() != len(groups):
        raise ValueError("One existence logit is required for every group")
    if any(not isinstance(g, AbstainGroupSpec) for g in groups):
        raise ValueError("Expected AbstainGroupSpec records")
    try:
        if len({g.node_id for g in groups}) != len(groups):
            raise ValueError("Group node identities must be distinct")
    except TypeError as exc:
        raise ValueError("Group node identities must be hashable") from exc
    occupied: set[int] = set()
    for group in groups:
        if not isinstance(group, AbstainGroupSpec):
            raise ValueError("Expected AbstainGroupSpec records")
        fields = {
            "candidate": group.candidate_slots,
            "positive": group.positive_slots,
            "negative": group.negative_slots,
        }
        for role, slots in fields.items():
            if not isinstance(slots, tuple) or len(slots) != len(set(slots)):
                raise ValueError(f"{role} slots must be unique tuples")
            if any(type(slot) is not int or slot < 0 or slot >= candidate_logits.numel() for slot in slots):
                raise ValueError(f"{role} slot is outside candidate logits")
        candidate = set(group.candidate_slots)
        positive = set(group.positive_slots)
        negative = set(group.negative_slots)
        if not candidate:
            raise ValueError("Candidate group must be nonempty")
        if not positive.issubset(candidate) or not negative.issubset(candidate):
            raise ValueError("Reviewed labels must be candidate members")
        if positive & negative:
            raise ValueError("Positive and negative slots overlap")
        if occupied & candidate:
            raise ValueError("Candidate ownership overlaps across groups")
        occupied.update(candidate)
        if group.protected:
            if group.exclusion_reason is not None or positive or negative or group.existence_target is not None:
                raise ValueError("Protected groups cannot carry supervision labels")
            continue
        if group.exclusion_reason is not None:
            if not isinstance(group.exclusion_reason, str) or not group.exclusion_reason.strip():
                raise ValueError("Exclusion requires a nonempty reason")
            if positive or negative or group.existence_target is not None:
                raise ValueError("Excluded groups cannot carry supervision labels")
            continue
        if positive:
            if group.existence_target is False:
                raise ValueError("A positive group cannot have existence target false")
        elif group.existence_target is not False:
            raise ValueError("No-positive group must explicitly set existence_target=False")
    return groups, occupied


def evaluate_group_abstain_objective(
    candidate_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    groups: Sequence[AbstainGroupSpec],
    *,
    ranking_margin: float = 1.0,
    explicit_negative_weight: float = 1.0,
) -> GroupAbstainResult:
    """Return a differentiable existence/ranking loss without optimization.

    Existence is a binary group-level BCE over strict-positive and
    confirmed-negative groups.  Ranking compares reviewed positives only with
    explicitly reviewed negatives; unknown alternatives are excluded rather
    than made pseudo-negatives.  This is intentionally conservative: if a
    dataset supplies no reviewed negative within a positive group, only its
    existence head receives a direct gradient.  ``explicit_negative_weight``
    is fixed by the caller and is not tuned here.
    """

    if not isinstance(ranking_margin, (float, int)) or not torch.isfinite(torch.tensor(float(ranking_margin))):
        raise ValueError("ranking_margin must be finite")
    if float(ranking_margin) < 0:
        raise ValueError("ranking_margin must be nonnegative")
    if not isinstance(explicit_negative_weight, (float, int)) or float(explicit_negative_weight) < 0:
        raise ValueError("explicit_negative_weight must be nonnegative")
    groups, _ = _validate_groups(candidate_logits, existence_logits, groups)
    zero = candidate_logits.sum() * 0.0 + existence_logits.sum() * 0.0
    existence_terms: list[torch.Tensor] = []
    ranking_terms: list[torch.Tensor] = []
    targets: list[tuple[Hashable, bool]] = []
    active, protected, excluded = [], [], []
    supervised: set[int] = set()
    unknown: set[int] = set()
    for index, group in enumerate(groups):
        candidate = set(group.candidate_slots)
        positive = set(group.positive_slots)
        negative = set(group.negative_slots)
        unknown.update(candidate - positive - negative)
        if group.protected:
            protected.append(group.node_id)
            continue
        if group.exclusion_reason is not None:
            excluded.append((group.node_id, group.exclusion_reason))
            continue
        target = bool(positive)
        targets.append((group.node_id, target))
        existence_terms.append(F.binary_cross_entropy_with_logits(existence_logits[index], existence_logits[index].new_tensor(float(target))))
        supervised.update(positive | negative)
        if not positive:
            continue
        active.append(group.node_id)
        if negative and explicit_negative_weight:
            positive_scores = candidate_logits[list(positive)]
            negative_scores = candidate_logits[list(negative)]
            hard_negative = negative_scores.max()
            ranking_terms.append(
                float(explicit_negative_weight)
                * F.softplus(float(ranking_margin) + hard_negative - positive_scores).mean()
            )
    existence_loss = torch.stack(existence_terms).mean() if existence_terms else zero
    ranking_loss = torch.stack(ranking_terms).mean() if ranking_terms else zero
    return GroupAbstainResult(
        total=existence_loss + ranking_loss,
        existence_loss=existence_loss,
        ranking_loss=ranking_loss,
        existence_targets=tuple(targets),
        active_node_ids=tuple(active),
        protected_node_ids=tuple(protected),
        excluded_nodes=tuple(excluded),
        supervised_slots=tuple(sorted(supervised)),
        unknown_slots=tuple(sorted(unknown)),
    )


def select_group_with_abstention(
    candidate_logits: torch.Tensor,
    existence_logit: torch.Tensor | float,
    *,
    candidate_slots: tuple[int, ...] | None = None,
    policy: AbstainPolicy = AbstainPolicy(),
    protected: bool = False,
) -> GroupDecision:
    """Select at most one candidate; low-existence groups return abstain.

    The existence gate runs before ``argmax``.  Consequently an all-negative
    group cannot be forced to emit its highest-scoring candidate.  The margin
    gate also abstains on ambiguous ties.  Protected groups are returned as a
    hard-locked status and must be handled by the caller's baseline prefix.
    """

    _validate_logits(candidate_logits, name="candidate_logits")
    if isinstance(existence_logit, torch.Tensor):
        if existence_logit.numel() != 1 or existence_logit.device.type != "cpu" or not torch.is_floating_point(existence_logit) or not bool(torch.isfinite(existence_logit).all()):
            raise ValueError("existence_logit must be one finite CPU scalar")
        existence_value = float(existence_logit.reshape(()))
    elif isinstance(existence_logit, (float, int)) and torch.isfinite(torch.tensor(float(existence_logit))):
        existence_value = float(existence_logit)
    else:
        raise ValueError("existence_logit must be one finite scalar")
    if protected:
        return GroupDecision("protected", None, None, None)
    slots = tuple(range(candidate_logits.numel())) if candidate_slots is None else candidate_slots
    if not slots or len(set(slots)) != len(slots) or any(type(slot) is not int or slot < 0 or slot >= candidate_logits.numel() for slot in slots):
        raise ValueError("candidate_slots must be a nonempty unique in-range tuple")
    if existence_value < policy.existence_threshold:
        return GroupDecision("abstain_existence", None, None, None)
    scores = candidate_logits[list(slots)]
    winner_index = int(torch.argmax(scores))  # deterministic first index on an exact tie
    winner_score = float(scores[winner_index])
    if winner_score < policy.winner_threshold:
        return GroupDecision("abstain_winner", None, winner_score, None)
    if len(scores) > 1:
        second_score = float(scores.topk(2).values[1])
        margin = winner_score - second_score
        if margin < policy.margin_threshold:
            return GroupDecision("abstain_margin", None, winner_score, margin)
    else:
        margin = float("inf")
    return GroupDecision("selected", int(slots[winner_index]), winner_score, margin)
