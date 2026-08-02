"""Grammar-masked teacher-forcing metrics for ChemTraverse v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from chemvector.generation_chemtraverse_v0_1 import ChemTraverseGrammarV01


_CONTROL_TOKENS = (
    ChemTraverseTokenizerV01.COMPONENT,
    ChemTraverseTokenizerV01.DOWN,
    ChemTraverseTokenizerV01.UP,
    ChemTraverseTokenizerV01.LINK,
    ChemTraverseTokenizerV01.COMPONENT_END,
    ChemTraverseTokenizerV01.MOL_END,
    ChemTraverseTokenizerV01.EOS,
)


@dataclass
class _AccuracyCounts:
    correct: int = 0
    total: int = 0

    def add(self, matched: bool) -> None:
        self.correct += int(matched)
        self.total += 1

    def merge(self, other: "_AccuracyCounts") -> None:
        self.correct += other.correct
        self.total += other.total

    def metrics(self) -> dict[str, int | float]:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.correct / max(self.total, 1),
        }


@dataclass(frozen=True)
class _TeacherPosition:
    batch_index: int
    logit_index: int
    target_token_id: int
    allowed_token_ids: tuple[int, ...]
    byte_context: str | None

    @property
    def forced(self) -> bool:
        return len(self.allowed_token_ids) == 1


@dataclass
class ChemTraverseGrammarMaskedTeacherCounts:
    """Mergeable integer counts; convert to JSON metrics only after all batches."""

    valid_positions: int = 0
    forced_positions: int = 0
    non_forced_decisions: _AccuracyCounts = field(default_factory=_AccuracyCounts)
    control_tokens: dict[str, _AccuracyCounts] = field(default_factory=dict)
    byte_contexts: dict[str, _AccuracyCounts] = field(default_factory=dict)

    def update(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer: ChemTraverseTokenizerV01,
        *,
        pad_token_id: int | None = None,
    ) -> None:
        positions = _replay_teacher_positions(
            input_ids,
            tokenizer,
            pad_token_id=pad_token_id,
        )
        decision_positions = [position for position in positions if not position.forced]
        decision_matches = _masked_decision_matches(
            logits,
            input_ids,
            tokenizer,
            decision_positions,
        )
        decision_cursor = 0
        for position in positions:
            self.valid_positions += 1
            if position.forced:
                self.forced_positions += 1
                matched = True
            else:
                matched = decision_matches[decision_cursor]
                decision_cursor += 1
                self.non_forced_decisions.add(matched)

            token = tokenizer.id_to_token[position.target_token_id]
            if token in _CONTROL_TOKENS:
                self.control_tokens.setdefault(token, _AccuracyCounts()).add(matched)
            if position.byte_context is not None:
                self.byte_contexts.setdefault(
                    position.byte_context,
                    _AccuracyCounts(),
                ).add(matched)

    def merge(self, other: "ChemTraverseGrammarMaskedTeacherCounts") -> None:
        self.valid_positions += other.valid_positions
        self.forced_positions += other.forced_positions
        self.non_forced_decisions.merge(other.non_forced_decisions)
        for name, counts in other.control_tokens.items():
            self.control_tokens.setdefault(name, _AccuracyCounts()).merge(counts)
        for name, counts in other.byte_contexts.items():
            self.byte_contexts.setdefault(name, _AccuracyCounts()).merge(counts)

    def metrics(self) -> dict:
        decision_total = self.non_forced_decisions.total
        decision_correct = self.non_forced_decisions.correct
        all_correct = self.forced_positions + decision_correct
        return {
            "schema_version": "chemtraverse-grammar-masked-teacher-0.1",
            "valid_positions": self.valid_positions,
            "forced_positions": self.forced_positions,
            "decision_positions": decision_total,
            "decision_fraction": decision_total / max(self.valid_positions, 1),
            "grammar_masked_token_accuracy": all_correct
            / max(self.valid_positions, 1),
            "non_forced_decision_correct": decision_correct,
            "non_forced_decision_total": decision_total,
            "non_forced_decision_accuracy": decision_correct
            / max(decision_total, 1),
            "control_tokens": {
                token: self.control_tokens.get(token, _AccuracyCounts()).metrics()
                for token in _CONTROL_TOKENS
            },
            "byte_contexts": {
                context: self.byte_contexts.get(context, _AccuracyCounts()).metrics()
                for context in ("meta_molecule_id", "text_object")
            },
        }


def _unpadded_length(row: list[int], pad_token_id: int) -> int:
    try:
        length = row.index(pad_token_id)
    except ValueError:
        return len(row)
    if any(token_id != pad_token_id for token_id in row[length:]):
        raise ValueError("ChemTraverse padding must be one contiguous suffix")
    return length


def _replay_teacher_positions(
    input_ids: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
    *,
    pad_token_id: int | None,
) -> list[_TeacherPosition]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id
    rows = input_ids.detach().to(device="cpu", dtype=torch.long).tolist()
    positions = []
    for batch_index, row in enumerate(rows):
        length = _unpadded_length(row, pad_token_id)
        if length < 2:
            raise ValueError("Each ChemTraverse target must contain at least BOS and EOS")
        grammar = ChemTraverseGrammarV01(tokenizer)
        grammar.consume(row[0])
        for input_index in range(1, length):
            target_token_id = row[input_index]
            allowed = grammar.allowed_token_ids()
            if target_token_id not in allowed:
                token = (
                    tokenizer.id_to_token[target_token_id]
                    if 0 <= target_token_id < tokenizer.vocab_size
                    else str(target_token_id)
                )
                raise ChemMolSequenceError(
                    f"GT token {token!r} is illegal at batch row {batch_index}, "
                    f"logit position {input_index - 1}"
                )
            token = tokenizer.id_to_token[target_token_id]
            byte_context = None
            if token.startswith("byte:"):
                if grammar.record_type == "meta":
                    byte_context = "meta_molecule_id"
                elif grammar.record_type == "text":
                    byte_context = "text_object"
            positions.append(
                _TeacherPosition(
                    batch_index=batch_index,
                    logit_index=input_index - 1,
                    target_token_id=target_token_id,
                    allowed_token_ids=tuple(allowed),
                    byte_context=byte_context,
                )
            )
            grammar.consume(target_token_id)
        if grammar.phase != "done":
            raise ChemMolSequenceError(
                f"ChemTraverse GT row {batch_index} ended in grammar phase "
                f"{grammar.phase!r}"
            )
    return positions


def _masked_decision_matches(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
    positions: list[_TeacherPosition],
    *,
    chunk_size: int = 1024,
) -> list[bool]:
    expected_shape = (
        input_ids.size(0),
        max(input_ids.size(1) - 1, 0),
        tokenizer.vocab_size,
    )
    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape)} does not match ChemTraverse target "
            f"shape {expected_shape}"
        )
    matches = []
    for start in range(0, len(positions), chunk_size):
        chunk = positions[start : start + chunk_size]
        batch_indices = torch.tensor(
            [position.batch_index for position in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        logit_indices = torch.tensor(
            [position.logit_index for position in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits[batch_indices, logit_indices].float()
        allowed_mask = torch.zeros_like(selected, dtype=torch.bool)
        allowed_lengths = torch.tensor(
            [len(position.allowed_token_ids) for position in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        allowed_rows = torch.repeat_interleave(
            torch.arange(len(chunk), device=logits.device),
            allowed_lengths,
        )
        allowed_tokens = torch.tensor(
            [
                token_id
                for position in chunk
                for token_id in position.allowed_token_ids
            ],
            dtype=torch.long,
            device=logits.device,
        )
        allowed_mask[allowed_rows, allowed_tokens] = True
        predictions = selected.masked_fill(~allowed_mask, -torch.inf).argmax(dim=-1)
        targets = torch.tensor(
            [position.target_token_id for position in chunk],
            dtype=torch.long,
            device=logits.device,
        )
        matches.extend((predictions == targets).detach().cpu().tolist())
    return matches


__all__ = ["ChemTraverseGrammarMaskedTeacherCounts"]
