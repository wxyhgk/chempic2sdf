"""CPU-only, offline mathematical contract; NOT a production training loss.

No model, saved-data reader, optimizer, matching change, or fitted hyperparameter
is provided. A stable/reviewable node group defines interchangeable candidates;
the raw representative is an admission guard, NOT the desired winner. Calling
this function does not authorize using either the proposal or its source data
for training. Exact-one likelihood has symmetric stationary points and does NOT
guarantee that thresholding selects exactly one member. Worse, for three or more
similar scores just below threshold, it pushes EVERY valid member farther down.
This CPU counterexample rejects it as a missing-node recovery contract.
"""

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch
from torch.nn import functional as F


SELECTION_THRESHOLD = -0.25  # Existing selector rule: z >= -0.25.
TRAINING_AUTHORIZED = False


@dataclass(frozen=True)
class GroupSpec:
    node_id: Hashable
    acceptable_slots: tuple[int, ...]
    representative_slots: tuple[int, ...]
    fragment_slots: tuple[int, ...] = ()
    # Mandatory for unknown, sensitive, or no-representative groups. All their
    # referenced slots are excluded, including fragments; no pseudo-positive.
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    total: torch.Tensor
    node_loss: torch.Tensor
    fragment_loss: torch.Tensor
    node_losses: tuple[tuple[Hashable, torch.Tensor], ...]
    active_node_ids: tuple[Hashable, ...]
    excluded_nodes: tuple[tuple[Hashable, str], ...]
    supervised_slots: tuple[int, ...]
    training_authorized: bool = False


def _validate(logits: torch.Tensor, groups: Sequence[GroupSpec]) -> None:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 1
        or logits.device.type != "cpu"
        or logits.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("Expected finite, one-dimensional CPU float32/float64 logits")
    node_ids, claimed_slots = set(), set()
    for group in groups:
        if not isinstance(group, GroupSpec):
            raise ValueError("Expected GroupSpec records")
        try:
            if group.node_id in node_ids:
                raise ValueError("Duplicate node identity")
            node_ids.add(group.node_id)
        except TypeError as exc:
            raise ValueError("Node identity must be hashable") from exc
        for slots in (group.acceptable_slots, group.representative_slots, group.fragment_slots):
            if not isinstance(slots, tuple) or any(
                type(slot) is not int or slot < 0 or slot >= logits.numel() for slot in slots
            ):
                raise ValueError("Slots must be tuples of in-range integer indices")
            if len(slots) != len(set(slots)):
                raise ValueError("Duplicate slot within a role")
        acceptable, representatives, fragments = map(
            set, (group.acceptable_slots, group.representative_slots, group.fragment_slots)
        )
        if not representatives.issubset(acceptable):
            raise ValueError("Raw representatives must belong to the acceptable group")
        if acceptable & fragments:
            raise ValueError("A valid node candidate cannot also be a fragment")
        if claimed_slots & (acceptable | fragments):
            # This includes excluded groups: a supposedly ignored node must
            # not accidentally receive gradients through another node group.
            raise ValueError("Overlapping node/fragment ownership must remain unresolved")
        claimed_slots.update(acceptable | fragments)
        if group.exclusion_reason is not None:
            if not isinstance(group.exclusion_reason, str) or not group.exclusion_reason.strip():
                raise ValueError("Exclusion requires a nonempty reason")
        elif not acceptable or not representatives:
            raise ValueError("No valid representative: explicitly exclude this node")


def exactly_one_nll(centered_logits: torch.Tensor) -> torch.Tensor:
    """Stable exact-one Bernoulli NLL, with no winner or tie-breaking choice.

Equivalent to sum(softplus(u)) - logsumexp(u). A log-space two-state
    recurrence avoids subtracting large nearly equal terms at extreme logits.
    The stationary symmetric point is u_i = -log(k-1), for every k >= 2.
    For k >= 3, the open neighborhood near u_i=-0.1 is missing-all yet
    downward-all under gradient descent: this is not only an exact-tie issue.
"""
    if centered_logits.ndim != 1 or centered_logits.numel() == 0:
        raise ValueError("Expected a nonempty one-dimensional group")
    positive = F.logsigmoid(centered_logits)
    negative = F.logsigmoid(-centered_logits)
    log_zero, log_one = negative[0], positive[0]
    for index in range(1, centered_logits.numel()):
        log_one = torch.logaddexp(
            log_one + negative[index], log_zero + positive[index]
        )
        log_zero = log_zero + negative[index]
    return -log_one


def evaluate_group_objective(
    logits: torch.Tensor, groups: Sequence[GroupSpec]
) -> ObjectiveResult:
    """Mean exact-one node NLL + mean known-fragment softplus, weights 1 + 1.

The scales are declared for a single numerical contract, NOT tuned or admitted
as a training objective. Unknown/unreferenced/excluded slots have zero direct
gradient. Raw-vs-alternative membership never enters the loss. Direct logit
derivatives do not predict updates through shared model parameters or through
the pre-existing full loss. No optimization step is performed here.
    """
    groups = tuple(groups)
    _validate(logits, groups)
    # Remains differentiable (all-zero gradient) for an empty admitted subset.
    zero = logits.sum() * 0.0
    node_losses, excluded, fragments, supervised = [], [], [], set()
    centered = logits - SELECTION_THRESHOLD
    for group in groups:
        if group.exclusion_reason is not None:
            excluded.append((group.node_id, group.exclusion_reason))
            continue
        value = exactly_one_nll(centered[list(group.acceptable_slots)])
        node_losses.append((group.node_id, value))
        fragments.extend(group.fragment_slots)
        supervised.update(group.acceptable_slots + group.fragment_slots)
    node_loss = torch.stack([loss for _, loss in node_losses]).mean() if node_losses else zero
    fragment_loss = F.softplus(centered[fragments]).mean() if fragments else zero
    return ObjectiveResult(
        total=node_loss + fragment_loss,
        node_loss=node_loss,
        fragment_loss=fragment_loss,
        node_losses=tuple(node_losses),
        active_node_ids=tuple(node for node, _ in node_losses),
        excluded_nodes=tuple(excluded),
        supervised_slots=tuple(sorted(supervised)),
    )
