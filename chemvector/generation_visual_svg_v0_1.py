"""Grammar-constrained generation for Visual SVG v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from Data.sdf.visual_svg_sequence_v0_1 import (
    QuadRecordV01,
    TriangleRecordV01,
    VisualSVGSequenceError,
    _validate_quad,
    _validate_triangle,
    parse_visual_svg_v0_1,
)
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


@dataclass
class VisualSVGGrammarV01:
    tokenizer: VisualSVGTokenizerV01
    phase: str = "bos"
    minimum_kind: int = 0
    record_kind: int | None = None
    field_index: int = 0
    current_values: list[int] = field(default_factory=list)
    structural_primitives: int = 0

    LINE_KIND = 0
    TRIANGLE_KIND = 1
    QUAD_KIND = 2
    TEXT_KIND = 3

    def __post_init__(self) -> None:
        self._coordinates = tuple(
            token_id
            for token, token_id in self.tokenizer.token_to_id.items()
            if token.startswith("c:")
        )
        self._widths = tuple(
            token_id
            for token, token_id in self.tokenizer.token_to_id.items()
            if token.startswith("w:")
        )

    def _id(self, token: str) -> int:
        return self.tokenizer.token_to_id[token]

    def _coordinate_ids(
        self,
        *,
        greater_than: int | None = None,
        maximum: int | None = None,
    ) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id in self._coordinates
            if (greater_than is None or int(self.tokenizer.id_to_token[token_id][2:]) > greater_than)
            and (maximum is None or int(self.tokenizer.id_to_token[token_id][2:]) <= maximum)
        )

    def _coordinate_ids_excluding(self, excluded: int) -> tuple[int, ...]:
        return tuple(
            token_id
            for token_id in self._coordinates
            if int(self.tokenizer.id_to_token[token_id][2:]) != excluded
        )

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.phase == "bos":
            return (self.tokenizer.bos_token_id,)
        if self.phase == "document":
            return (self._id(self.tokenizer.DOCUMENT),)
        if self.phase == "record":
            allowed = []
            markers = (
                self.tokenizer.LINE,
                self.tokenizer.TRIANGLE,
                self.tokenizer.QUAD,
                self.tokenizer.TEXT_BOX,
            )
            for kind in range(self.minimum_kind, len(markers)):
                if kind == self.TEXT_KIND and not self.structural_primitives:
                    continue
                allowed.append(self._id(markers[kind]))
            if self.structural_primitives:
                allowed.append(self._id(self.tokenizer.DOCUMENT_END))
            return tuple(allowed)
        if self.phase == "end":
            return (self.tokenizer.eos_token_id,)
        if self.phase == "done":
            return ()
        if self.phase != "fields" or self.record_kind is None:
            raise VisualSVGSequenceError(f"Unknown grammar phase: {self.phase}")
        arities = {
            self.LINE_KIND: 5,
            self.TRIANGLE_KIND: 7,
            self.QUAD_KIND: 9,
            self.TEXT_KIND: 4,
        }
        arity = arities[self.record_kind]
        if self.field_index == arity - 1 and self.record_kind != self.TEXT_KIND:
            return self._widths
        if self.record_kind == self.TEXT_KIND and self.field_index in {0, 1}:
            return self._coordinate_ids(maximum=2046)
        if self.record_kind == self.TEXT_KIND and self.field_index == 2:
            return self._coordinate_ids(greater_than=self.current_values[0])
        if self.record_kind == self.TEXT_KIND and self.field_index == 3:
            return self._coordinate_ids(greater_than=self.current_values[1])
        if self.record_kind == self.LINE_KIND and self.field_index == 3:
            x0, y0, x1 = self.current_values
            return tuple(
                token_id
                for token_id in self._coordinates
                if (
                    int(self.tokenizer.id_to_token[token_id][2:]),
                    x1,
                )
                > (y0, x0)
            )
        if self.record_kind == self.TRIANGLE_KIND and self.field_index == 5:
            x0, y0, x1, y1, x2 = self.current_values
            allowed = []
            for token_id in self._coordinates:
                y2 = int(self.tokenizer.id_to_token[token_id][2:])
                try:
                    _validate_triangle(TriangleRecordV01(x0, y0, x1, y1, x2, y2, 1))
                except VisualSVGSequenceError:
                    continue
                allowed.append(token_id)
            return tuple(allowed)
        if self.record_kind == self.QUAD_KIND and self.field_index == 7:
            x0, y0, x1, y1, x2, y2, x3 = self.current_values
            allowed = []
            for token_id in self._coordinates:
                y3 = int(self.tokenizer.id_to_token[token_id][2:])
                try:
                    _validate_quad(
                        QuadRecordV01(x0, y0, x1, y1, x2, y2, x3, y3, 1)
                    )
                except VisualSVGSequenceError:
                    continue
                allowed.append(token_id)
            return tuple(allowed)
        return self._coordinates

    def consume(self, token_id: int) -> None:
        if token_id not in self.allowed_token_ids():
            token = self.tokenizer.id_to_token[token_id]
            raise VisualSVGSequenceError(
                f"Token {token!r} is invalid in phase {self.phase}"
            )
        token = self.tokenizer.id_to_token[token_id]
        if self.phase == "bos":
            self.phase = "document"
            return
        if self.phase == "document":
            self.phase = "record"
            return
        if self.phase == "record":
            if token == self.tokenizer.DOCUMENT_END:
                self.phase = "end"
                return
            kind_by_token = {
                self.tokenizer.LINE: self.LINE_KIND,
                self.tokenizer.TRIANGLE: self.TRIANGLE_KIND,
                self.tokenizer.QUAD: self.QUAD_KIND,
                self.tokenizer.TEXT_BOX: self.TEXT_KIND,
            }
            self.record_kind = kind_by_token[token]
            self.minimum_kind = self.record_kind
            self.field_index = 0
            self.current_values = []
            self.phase = "fields"
            return
        if self.phase == "end":
            self.phase = "done"
            return

        prefix = "w:" if token.startswith("w:") else "c:"
        self.current_values.append(int(token[len(prefix) :]))
        self.field_index += 1
        arities = {
            self.LINE_KIND: 5,
            self.TRIANGLE_KIND: 7,
            self.QUAD_KIND: 9,
            self.TEXT_KIND: 4,
        }
        if self.field_index == arities[self.record_kind]:
            if self.record_kind != self.TEXT_KIND:
                self.structural_primitives += 1
            self.record_kind = None
            self.phase = "record"


@dataclass(frozen=True, slots=True)
class VisualSVGGenerationResultV01:
    input_ids: tuple[int, ...]
    sequence: str | None
    completed: bool
    grammar_valid: bool
    error: str | None = None


@torch.inference_mode()
def generate_visual_svg_v0_1(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    tokenizer: VisualSVGTokenizerV01,
    *,
    max_length: int | None = None,
) -> VisualSVGGenerationResultV01:
    if pixel_values.size(0) != 1:
        raise ValueError("Visual SVG generation currently supports batch size 1")
    model.eval()
    max_length = max_length or model.config.max_target_length
    if max_length > model.config.max_target_length:
        raise ValueError(
            f"max_length={max_length} exceeds model context {model.config.max_target_length}"
        )
    grammar = VisualSVGGrammarV01(tokenizer)
    generated = [tokenizer.bos_token_id]
    grammar.consume(generated[0])
    caches = None
    dead_end = False
    with torch.autocast(
        device_type=pixel_values.device.type,
        dtype=torch.bfloat16,
        enabled=pixel_values.device.type == "cuda",
    ):
        memory = model.encode_images(pixel_values)
        while len(generated) < max_length:
            token = torch.tensor([[generated[-1]]], device=pixel_values.device)
            logits, caches = model.decode_step(
                token,
                len(generated) - 1,
                memory,
                caches,
            )
            allowed = grammar.allowed_token_ids()
            if not allowed:
                dead_end = True
                break
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            scores = logits[0, 0].index_select(0, allowed_tensor)
            next_id = int(allowed_tensor[scores.argmax()])
            generated.append(next_id)
            grammar.consume(next_id)
            if grammar.phase == "done":
                try:
                    sequence = tokenizer.decode(generated)
                    parse_visual_svg_v0_1(sequence)
                except VisualSVGSequenceError as error:
                    return VisualSVGGenerationResultV01(
                        tuple(generated),
                        None,
                        True,
                        False,
                        str(error),
                    )
                return VisualSVGGenerationResultV01(
                    tuple(generated),
                    sequence,
                    True,
                    True,
                )
    error = (
        f"Grammar dead-end in phase={grammar.phase} field={grammar.field_index}"
        if dead_end
        else f"Generation did not finish within {max_length} tokens"
    )
    return VisualSVGGenerationResultV01(
        tuple(generated),
        None,
        False,
        False,
        error,
    )
