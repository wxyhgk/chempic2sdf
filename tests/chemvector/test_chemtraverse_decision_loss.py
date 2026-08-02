from __future__ import annotations

import math

import pytest
import torch

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import serialize_chemtraverse_v0_1
from chemvector.chemtraverse_decision_loss import (
    balanced_control_token_weights,
    build_chemtraverse_decision_plan,
    decision_token_frequencies,
    grammar_aware_decision_balanced_loss,
)
from tests.data_sdf.test_chemtraverse_v0_1 import chiral_scene, ring_scene


def _batch(tokenizer: ChemTraverseTokenizerV01) -> tuple[torch.Tensor, list[list[int]]]:
    sequences = [
        serialize_chemtraverse_v0_1(ring_scene()),
        serialize_chemtraverse_v0_1(chiral_scene()),
    ]
    encoded = [tokenizer.encode(sequence) for sequence in sequences]
    padded = tokenizer.encode_batch(sequences)["input_ids"]
    return torch.tensor(padded, dtype=torch.long), encoded


def test_plan_partitions_valid_targets_into_decision_and_forced_positions() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    input_ids, encoded = _batch(tokenizer)

    plan = build_chemtraverse_decision_plan(input_ids, tokenizer)

    assert plan.valid_positions == sum(len(row) - 1 for row in encoded)
    assert plan.valid_positions == plan.decision_positions + plan.forced_positions
    assert plan.forced_positions > 0
    assert plan.decision_positions > 0
    assert plan.valid_positions_per_sample == tuple(len(row) - 1 for row in encoded)
    assert plan.decision_positions_per_sample == tuple(
        sum(item.batch_index == batch_index for item in plan.decisions)
        for batch_index in range(2)
    )


def test_masked_ce_matches_manual_weighted_legal_softmax() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    input_ids, _ = _batch(tokenizer)
    plan = build_chemtraverse_decision_plan(input_ids, tokenizer)
    logits = torch.zeros(
        input_ids.size(0),
        input_ids.size(1) - 1,
        tokenizer.vocab_size,
        requires_grad=True,
    )
    # PAD is illegal at every valid ChemTraverse position and must not affect CE.
    with torch.no_grad():
        logits[:, :, tokenizer.pad_token_id] = 100.0
    weights = {
        tokenizer.DOWN: 1.0,
        tokenizer.LINK: 2.5,
        tokenizer.COMPONENT_END: 4.0,
    }

    output = grammar_aware_decision_balanced_loss(
        logits,
        input_ids,
        tokenizer,
        token_weights=weights,
        plan=plan,
    )

    token_weights = {
        tokenizer.token_to_id[token]: value for token, value in weights.items()
    }
    manual_losses = []
    manual_weights = []
    per_sample = [[] for _ in range(input_ids.size(0))]
    for item in plan.decisions:
        weight = token_weights.get(item.target_token_id, 1.0)
        loss = math.log(len(item.allowed_token_ids))
        manual_losses.append(loss * weight)
        manual_weights.append(weight)
        per_sample[item.batch_index].append((loss, weight))
    expected = sum(manual_losses) / sum(manual_weights)
    assert float(output.loss.detach()) == pytest.approx(expected, rel=1e-6)
    for batch_index, values in enumerate(per_sample):
        expected_sample = sum(loss * weight for loss, weight in values) / sum(
            weight for _, weight in values
        )
        assert float(output.loss_per_sample[batch_index].detach()) == pytest.approx(
            expected_sample, rel=1e-6
        )


def test_gradients_exist_only_for_legal_decision_logits() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    input_ids, encoded = _batch(tokenizer)
    plan = build_chemtraverse_decision_plan(input_ids, tokenizer)
    logits = torch.zeros(
        input_ids.size(0),
        input_ids.size(1) - 1,
        tokenizer.vocab_size,
        requires_grad=True,
    )

    output = grammar_aware_decision_balanced_loss(
        logits, input_ids, tokenizer, plan=plan
    )
    output.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    for batch_index, logit_index in plan.forced_logit_indices:
        assert torch.count_nonzero(logits.grad[batch_index, logit_index]) == 0
    for batch_index, row in enumerate(encoded):
        assert torch.count_nonzero(logits.grad[batch_index, len(row) - 1 :]) == 0

    decision = plan.decisions[0]
    allowed = set(decision.allowed_token_ids)
    illegal = next(
        token_id for token_id in range(tokenizer.vocab_size) if token_id not in allowed
    )
    gradient = logits.grad[decision.batch_index, decision.logit_index]
    assert gradient[illegal] == 0
    assert gradient[decision.target_token_id] != 0
    assert any(gradient[token_id] != 0 for token_id in allowed)


def test_balanced_weights_use_clipped_inverse_sqrt_decision_frequency() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    frequencies = {
        tokenizer.DOWN: 100,
        tokenizer.UP: 100,
        tokenizer.LINK: 25,
        tokenizer.COMPONENT_END: 1,
        tokenizer.MOL_END: 0,
    }

    weights = balanced_control_token_weights(tokenizer, frequencies)

    assert weights[tokenizer.token_to_id[tokenizer.DOWN]] == pytest.approx(1.0)
    assert weights[tokenizer.token_to_id[tokenizer.UP]] == pytest.approx(1.0)
    assert weights[tokenizer.token_to_id[tokenizer.LINK]] == pytest.approx(2.0)
    assert weights[tokenizer.token_to_id[tokenizer.COMPONENT_END]] == pytest.approx(4.0)
    assert tokenizer.token_to_id[tokenizer.MOL_END] not in weights


def test_decision_frequencies_exclude_forced_targets() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    input_ids, _ = _batch(tokenizer)
    plan = build_chemtraverse_decision_plan(input_ids, tokenizer)

    frequencies = decision_token_frequencies(plan)

    assert sum(frequencies.values()) == plan.decision_positions
    assert tokenizer.token_to_id[tokenizer.MOL] not in frequencies
    assert frequencies[tokenizer.token_to_id[tokenizer.DOWN]] > 0


def test_invalid_gt_and_noncontiguous_padding_are_rejected() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    sequence = tokenizer.encode(serialize_chemtraverse_v0_1(ring_scene()))
    invalid = torch.tensor([sequence], dtype=torch.long)
    invalid[0, 1] = tokenizer.token_to_id[tokenizer.ATOM]
    with pytest.raises(ChemMolSequenceError, match="illegal"):
        build_chemtraverse_decision_plan(invalid, tokenizer)

    padded = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="contiguous suffix"):
        build_chemtraverse_decision_plan(padded, tokenizer)


def test_precomputed_plan_shape_is_validated() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    input_ids, _ = _batch(tokenizer)
    plan = build_chemtraverse_decision_plan(input_ids, tokenizer)
    wrong_logits = torch.zeros(
        input_ids.size(0), input_ids.size(1) - 2, tokenizer.vocab_size
    )

    with pytest.raises(ValueError, match="sequence shapes disagree"):
        grammar_aware_decision_balanced_loss(
            wrong_logits,
            input_ids,
            tokenizer,
            plan=plan,
        )
