"""Grammar-aware decision-balanced loss for ChemTraverse v0.1.

This module is deliberately independent from the shared training entry point.
An integration can replace the model's ordinary sequence loss as follows::

    plan = build_chemtraverse_decision_plan(input_ids, tokenizer)
    result = grammar_aware_decision_balanced_loss(
        output["logits"],
        input_ids,
        tokenizer,
        token_weights=frozen_train_only_weights,
        plan=plan,
    )
    output["loss"] = result.loss
    output["loss_per_sample"] = result.loss_per_sample

``frozen_train_only_weights`` must be computed once from training targets with
``decision_token_frequencies`` and ``balanced_control_token_weights``. It must
not be recomputed per batch or from route-select data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from chemvector.generation_chemtraverse_v0_1 import ChemTraverseGrammarV01


TokenKey = int | str


@dataclass(frozen=True)
class ChemTraverseDecisionPosition:
    """One non-forced next-token decision under a ground-truth prefix."""

    batch_index: int
    logit_index: int
    target_token_id: int
    allowed_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ChemTraverseDecisionPlan:
    """CPU-side sparse grammar plan for one padded target batch."""

    batch_size: int
    logit_length: int
    vocabulary_size: int
    decisions: tuple[ChemTraverseDecisionPosition, ...]
    forced_logit_indices: tuple[tuple[int, int], ...]
    valid_positions_per_sample: tuple[int, ...]
    decision_positions_per_sample: tuple[int, ...]

    @property
    def valid_positions(self) -> int:
        return sum(self.valid_positions_per_sample)

    @property
    def decision_positions(self) -> int:
        return len(self.decisions)

    @property
    def forced_positions(self) -> int:
        return len(self.forced_logit_indices)


@dataclass(frozen=True)
class ChemTraverseDecisionLossOutput:
    """Loss tensors plus auditable position counts."""

    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    decision_positions: int
    forced_positions: int
    valid_positions: int
    weight_sum: torch.Tensor


def _unpadded_length(row: list[int], pad_token_id: int) -> int:
    try:
        length = row.index(pad_token_id)
    except ValueError:
        return len(row)
    if any(token_id != pad_token_id for token_id in row[length:]):
        raise ValueError("ChemTraverse padding must be one contiguous suffix")
    return length


def build_chemtraverse_decision_plan(
    input_ids: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
    *,
    pad_token_id: int | None = None,
) -> ChemTraverseDecisionPlan:
    """Replay each GT prefix and retain only positions with multiple legal tokens."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id
    rows = input_ids.detach().to(device="cpu", dtype=torch.long).tolist()
    decisions: list[ChemTraverseDecisionPosition] = []
    forced: list[tuple[int, int]] = []
    valid_per_sample: list[int] = []
    decision_per_sample: list[int] = []

    for batch_index, row in enumerate(rows):
        length = _unpadded_length(row, pad_token_id)
        if length < 2:
            raise ValueError("Each ChemTraverse target must contain BOS and at least one token")
        grammar = ChemTraverseGrammarV01(tokenizer)
        try:
            grammar.consume(row[0])
        except ChemMolSequenceError as error:
            raise ChemMolSequenceError(
                f"Invalid ChemTraverse BOS in batch row {batch_index}: {error}"
            ) from error

        sample_decisions = 0
        for input_index in range(1, length):
            target_token_id = row[input_index]
            logit_index = input_index - 1
            allowed = grammar.allowed_token_ids()
            if target_token_id not in allowed:
                target = (
                    tokenizer.id_to_token[target_token_id]
                    if 0 <= target_token_id < tokenizer.vocab_size
                    else str(target_token_id)
                )
                raise ChemMolSequenceError(
                    f"GT token {target!r} is illegal at batch row {batch_index}, "
                    f"logit position {logit_index}"
                )
            if len(allowed) == 1:
                forced.append((batch_index, logit_index))
            else:
                decisions.append(
                    ChemTraverseDecisionPosition(
                        batch_index=batch_index,
                        logit_index=logit_index,
                        target_token_id=target_token_id,
                        allowed_token_ids=tuple(allowed),
                    )
                )
                sample_decisions += 1
            grammar.consume(target_token_id)
        if grammar.phase != "done":
            raise ChemMolSequenceError(
                f"ChemTraverse GT row {batch_index} ended in grammar phase {grammar.phase!r}"
            )
        valid_per_sample.append(length - 1)
        decision_per_sample.append(sample_decisions)

    return ChemTraverseDecisionPlan(
        batch_size=len(rows),
        logit_length=max(input_ids.size(1) - 1, 0),
        vocabulary_size=tokenizer.vocab_size,
        decisions=tuple(decisions),
        forced_logit_indices=tuple(forced),
        valid_positions_per_sample=tuple(valid_per_sample),
        decision_positions_per_sample=tuple(decision_per_sample),
    )


def decision_token_frequencies(
    plan: ChemTraverseDecisionPlan,
) -> dict[int, int]:
    """Count targets only where the grammar presented a real choice."""

    return dict(Counter(item.target_token_id for item in plan.decisions))


def _token_id(tokenizer: ChemTraverseTokenizerV01, key: TokenKey) -> int:
    if isinstance(key, str):
        try:
            return tokenizer.token_to_id[key]
        except KeyError as error:
            raise ValueError(f"Unknown ChemTraverse token weight key: {key!r}") from error
    token_id = int(key)
    if not 0 <= token_id < tokenizer.vocab_size:
        raise ValueError(f"Token weight ID is outside the vocabulary: {token_id}")
    return token_id


def _normalized_weight_mapping(
    tokenizer: ChemTraverseTokenizerV01,
    token_weights: Mapping[TokenKey, float] | None,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in (token_weights or {}).items():
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Decision token weights must be finite and positive")
        result[_token_id(tokenizer, key)] = weight
    return result


def balanced_control_token_weights(
    tokenizer: ChemTraverseTokenizerV01,
    decision_frequencies: Mapping[TokenKey, int],
    *,
    exponent: float = 0.5,
    minimum: float = 1.0,
    maximum: float = 4.0,
    control_tokens: tuple[str, ...] | None = None,
) -> dict[int, float]:
    """Create the preregistered clipped inverse-sqrt control weights.

    Frequencies must describe non-forced decision positions from training data.
    A selected token with zero decision occurrences is omitted because its loss
    is always zero under the grammar plan.
    """

    if not 0.0 <= exponent <= 1.0:
        raise ValueError("exponent must be in [0, 1]")
    if not 0.0 < minimum <= maximum:
        raise ValueError("weight bounds must satisfy 0 < minimum <= maximum")
    control_tokens = control_tokens or (
        tokenizer.DOWN,
        tokenizer.UP,
        tokenizer.LINK,
        tokenizer.COMPONENT_END,
        tokenizer.MOL_END,
    )
    normalized_frequencies: dict[int, int] = {}
    for key, value in decision_frequencies.items():
        frequency = int(value)
        if frequency < 0:
            raise ValueError("Decision token frequencies must be nonnegative")
        normalized_frequencies[_token_id(tokenizer, key)] = frequency

    selected = {
        tokenizer.token_to_id[token]: normalized_frequencies.get(
            tokenizer.token_to_id[token], 0
        )
        for token in control_tokens
    }
    positive = [frequency for frequency in selected.values() if frequency > 0]
    if not positive:
        raise ValueError("No positive control decision frequencies were provided")
    reference = max(positive)
    return {
        token_id: min(
            maximum,
            max(minimum, (reference / frequency) ** exponent),
        )
        for token_id, frequency in selected.items()
        if frequency > 0
    }


def _validate_plan(
    plan: ChemTraverseDecisionPlan,
    logits: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
) -> None:
    expected = (plan.batch_size, plan.logit_length, plan.vocabulary_size)
    actual = tuple(logits.shape)
    if actual != expected:
        raise ValueError(f"Logits shape {actual} does not match decision plan {expected}")
    if plan.vocabulary_size != tokenizer.vocab_size:
        raise ValueError("Decision plan and tokenizer vocabulary sizes differ")


def grammar_aware_decision_balanced_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
    *,
    token_weights: Mapping[TokenKey, float] | None = None,
    pad_token_id: int | None = None,
    plan: ChemTraverseDecisionPlan | None = None,
) -> ChemTraverseDecisionLossOutput:
    """Compute differentiable CE over legal alternatives at decision positions."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence - 1, vocabulary]")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if logits.size(0) != input_ids.size(0) or logits.size(1) != input_ids.size(1) - 1:
        raise ValueError("logits and input_ids sequence shapes disagree")
    if logits.size(2) != tokenizer.vocab_size:
        raise ValueError("logits vocabulary does not match the ChemTraverse tokenizer")
    plan = plan or build_chemtraverse_decision_plan(
        input_ids,
        tokenizer,
        pad_token_id=pad_token_id,
    )
    _validate_plan(plan, logits, tokenizer)
    weights_by_token = _normalized_weight_mapping(tokenizer, token_weights)

    if not plan.decisions:
        zero = logits.sum() * 0.0
        return ChemTraverseDecisionLossOutput(
            loss=zero,
            loss_per_sample=zero.expand(input_ids.size(0)),
            decision_positions=0,
            forced_positions=plan.forced_positions,
            valid_positions=plan.valid_positions,
            weight_sum=zero.detach(),
        )

    device = logits.device
    batch_indices = torch.tensor(
        [item.batch_index for item in plan.decisions],
        dtype=torch.long,
        device=device,
    )
    logit_indices = torch.tensor(
        [item.logit_index for item in plan.decisions],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [item.target_token_id for item in plan.decisions],
        dtype=torch.long,
        device=device,
    )
    position_weights = torch.tensor(
        [
            weights_by_token.get(item.target_token_id, 1.0)
            for item in plan.decisions
        ],
        dtype=torch.float32,
        device=device,
    )

    decision_logits = logits[batch_indices, logit_indices].float()
    allowed_lengths = torch.tensor(
        [len(item.allowed_token_ids) for item in plan.decisions],
        dtype=torch.long,
        device=device,
    )
    allowed_rows = torch.repeat_interleave(
        torch.arange(len(plan.decisions), device=device),
        allowed_lengths,
    )
    allowed_tokens = torch.tensor(
        [
            token_id
            for item in plan.decisions
            for token_id in item.allowed_token_ids
        ],
        dtype=torch.long,
        device=device,
    )
    allowed_mask = torch.zeros(
        (len(plan.decisions), tokenizer.vocab_size),
        dtype=torch.bool,
        device=device,
    )
    allowed_mask[allowed_rows, allowed_tokens] = True
    masked_logits = decision_logits.masked_fill(~allowed_mask, -torch.inf)
    position_losses = F.cross_entropy(masked_logits, targets, reduction="none")
    weighted_losses = position_losses * position_weights
    weight_sum = position_weights.sum()
    loss = weighted_losses.sum() / weight_sum.clamp_min(1.0)

    sample_loss_sums = torch.zeros(
        input_ids.size(0), dtype=position_losses.dtype, device=device
    )
    sample_weight_sums = torch.zeros_like(sample_loss_sums)
    sample_loss_sums.scatter_add_(0, batch_indices, weighted_losses)
    sample_weight_sums.scatter_add_(0, batch_indices, position_weights)
    loss_per_sample = sample_loss_sums / sample_weight_sums.clamp_min(1.0)

    return ChemTraverseDecisionLossOutput(
        loss=loss,
        loss_per_sample=loss_per_sample,
        decision_positions=plan.decision_positions,
        forced_positions=plan.forced_positions,
        valid_positions=plan.valid_positions,
        weight_sum=weight_sum,
    )


__all__ = [
    "ChemTraverseDecisionLossOutput",
    "ChemTraverseDecisionPlan",
    "ChemTraverseDecisionPosition",
    "balanced_control_token_weights",
    "build_chemtraverse_decision_plan",
    "decision_token_frequencies",
    "grammar_aware_decision_balanced_loss",
]
