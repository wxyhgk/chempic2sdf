"""Auxiliary losses and diagnostics for Visual SVG coordinate tokens."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


def _coordinate_token_tensors(
    tokenizer: VisualSVGTokenizerV01,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    coordinate_tokens = sorted(
        (
            (int(token.removeprefix("c:")), token_id)
            for token, token_id in tokenizer.token_to_id.items()
            if token.startswith("c:")
        ),
        key=lambda item: item[0],
    )
    if not coordinate_tokens or coordinate_tokens[0][0] != 0:
        raise ValueError("Visual SVG tokenizer must contain coordinates starting at c:0")
    coordinate_values = [value for value, _ in coordinate_tokens]
    if coordinate_values != list(range(coordinate_values[-1] + 1)):
        raise ValueError("Visual SVG coordinate tokens must form a contiguous range")
    coordinate_max = float(max(coordinate_values[-1], 1))
    coordinate_ids = torch.tensor(
        [token_id for _, token_id in coordinate_tokens],
        dtype=torch.long,
        device=device,
    )
    coordinate_values_q = torch.tensor(
        coordinate_values,
        dtype=torch.float32,
        device=device,
    )
    target_lookup_q = torch.full(
        (tokenizer.vocab_size,),
        -1.0,
        dtype=torch.float32,
        device=device,
    )
    target_lookup_q[coordinate_ids] = coordinate_values_q
    return coordinate_ids, coordinate_values_q, target_lookup_q, coordinate_max


def coordinate_expectation_per_sample(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: VisualSVGTokenizerV01,
) -> dict[str, torch.Tensor]:
    """Return coordinate-only loss and metrics for each sample.

    The softmax is conditional on the ``c:*`` vocabulary. Cross entropy remains
    responsible for predicting the token type, while the normalized Smooth L1
    objective gives larger errors a larger penalty. ``soft_mae_q`` uses the
    Visual SVG quantization units (q2047 in the current protocol). Hard metrics
    use the argmax over the complete vocabulary, so ``hard_valid_rate`` detects
    predictions that are not coordinate tokens at all.
    """

    if logits.ndim != 3:
        raise ValueError("Coordinate loss expects logits shaped [batch, length, vocab]")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("Coordinate targets must match the batch and length of logits")
    if target_ids.device != logits.device:
        raise ValueError("Coordinate targets and logits must be on the same device")
    if logits.size(-1) != tokenizer.vocab_size:
        raise ValueError("Coordinate loss logits do not match the tokenizer vocabulary")

    coordinate_ids, coordinate_values_q, target_lookup_q, coordinate_max = (
        _coordinate_token_tensors(tokenizer, logits.device)
    )
    target_coordinates_q = target_lookup_q[target_ids]
    coordinate_mask = target_coordinates_q >= 0
    selected_logits = logits[coordinate_mask]
    conditional_logits = selected_logits.index_select(-1, coordinate_ids).float()
    selected_targets_q = target_coordinates_q[coordinate_mask]
    expected_coordinates_q = F.softmax(conditional_logits, dim=-1) @ coordinate_values_q
    soft_absolute_errors_q = (expected_coordinates_q - selected_targets_q).abs()
    token_losses = F.smooth_l1_loss(
        expected_coordinates_q / coordinate_max,
        selected_targets_q / coordinate_max,
        reduction="none",
        beta=1.0 / coordinate_max,
    )

    selected_target_ids = target_ids[coordinate_mask]
    hard_token_ids = selected_logits.argmax(dim=-1)
    hard_valid = target_lookup_q[hard_token_ids] >= 0
    hard_exact = hard_token_ids == selected_target_ids
    batch_indices = coordinate_mask.nonzero(as_tuple=False)[:, 0]
    token_counts = torch.bincount(batch_indices, minlength=logits.size(0))
    differentiable_zeros = logits[..., 0].float().sum(dim=1) * 0.0
    metric_zeros = torch.zeros_like(differentiable_zeros)

    def per_sample_mean(
        values: torch.Tensor,
        *,
        differentiable: bool = True,
    ) -> torch.Tensor:
        base = differentiable_zeros if differentiable else metric_zeros
        sums = base.scatter_add(
            0,
            batch_indices,
            values.to(dtype=torch.float32),
        )
        return sums / token_counts.clamp_min(1).to(dtype=torch.float32)

    return {
        "loss": per_sample_mean(token_losses),
        "soft_mae_q": per_sample_mean(soft_absolute_errors_q),
        "hard_exact": per_sample_mean(hard_exact, differentiable=False),
        "hard_valid_rate": per_sample_mean(hard_valid, differentiable=False),
        "tokens": token_counts,
    }


def coordinate_expectation_metrics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: VisualSVGTokenizerV01,
) -> dict[str, torch.Tensor]:
    """Aggregate coordinate expectation metrics over all coordinate tokens."""

    per_sample = coordinate_expectation_per_sample(logits, target_ids, tokenizer)
    token_counts = per_sample["tokens"]
    denominator = token_counts.sum().clamp_min(1).to(dtype=torch.float32)
    weights = token_counts.to(dtype=torch.float32)

    def weighted_mean(name: str) -> torch.Tensor:
        return (per_sample[name] * weights).sum() / denominator

    return {
        "loss": weighted_mean("loss"),
        "soft_mae_q": weighted_mean("soft_mae_q"),
        "hard_exact": weighted_mean("hard_exact"),
        "hard_valid_rate": weighted_mean("hard_valid_rate"),
        "tokens": token_counts.sum(),
    }
