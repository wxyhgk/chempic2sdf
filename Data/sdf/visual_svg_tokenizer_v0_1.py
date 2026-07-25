#!/usr/bin/env python3
"""Tokenizer for the restricted Visual SVG v0.1 grammar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    from .visual_svg_sequence_v0_1 import (
        COORDINATE_MAX,
        SCHEMA_VERSION,
        STROKE_WIDTH_MAX,
        STROKE_WIDTH_MIN,
        LineRecordV01,
        ParsedVisualSVGSequenceV01,
        QuadRecordV01,
        TextBoxRecordV01,
        TriangleRecordV01,
        VisualSVGSequenceError,
        parse_visual_svg_v0_1,
        serialize_parsed_visual_svg_v0_1,
    )
except ImportError:
    from visual_svg_sequence_v0_1 import (
        COORDINATE_MAX,
        SCHEMA_VERSION,
        STROKE_WIDTH_MAX,
        STROKE_WIDTH_MIN,
        LineRecordV01,
        ParsedVisualSVGSequenceV01,
        QuadRecordV01,
        TextBoxRecordV01,
        TriangleRecordV01,
        VisualSVGSequenceError,
        parse_visual_svg_v0_1,
        serialize_parsed_visual_svg_v0_1,
    )


TOKENIZER_SCHEMA_VERSION = "visual-svg-tokenizer-0.1"


class VisualSVGTokenizerV01:
    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    UNK = "[UNK]"
    DOCUMENT = "[VISUAL_SVG_V01]"
    LINE = "[LINE]"
    TRIANGLE = "[TRIANGLE]"
    QUAD = "[QUAD]"
    TEXT_BOX = "[TEXT_BOX]"
    DOCUMENT_END = "[VISUAL_SVG_END]"

    SPECIAL_TOKENS = (
        PAD,
        BOS,
        EOS,
        UNK,
        DOCUMENT,
        LINE,
        TRIANGLE,
        QUAD,
        TEXT_BOX,
        DOCUMENT_END,
    )
    sequence_schema_version = SCHEMA_VERSION

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary or self.build_vocabulary()
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_token = tuple(self.vocabulary)

    @classmethod
    def build_vocabulary(cls) -> list[str]:
        return [
            *cls.SPECIAL_TOKENS,
            *(f"c:{value}" for value in range(COORDINATE_MAX + 1)),
            *(f"w:{value}" for value in range(STROKE_WIDTH_MIN, STROKE_WIDTH_MAX + 1)),
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK]

    def _coordinate_token(self, value: int) -> str:
        token = f"c:{value}"
        if token not in self.token_to_id:
            raise VisualSVGSequenceError(f"Coordinate is outside the tokenizer: {value}")
        return token

    def _width_token(self, value: int) -> str:
        token = f"w:{value}"
        if token not in self.token_to_id:
            raise VisualSVGSequenceError(f"Stroke width is outside the tokenizer: {value}")
        return token

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> list[int]:
        parsed = parse_visual_svg_v0_1(sequence)
        tokens = [self.DOCUMENT]
        for record in parsed.lines:
            tokens.extend(
                (
                    self.LINE,
                    self._coordinate_token(record.x0),
                    self._coordinate_token(record.y0),
                    self._coordinate_token(record.x1),
                    self._coordinate_token(record.y1),
                    self._width_token(record.stroke_width),
                )
            )
        for record in parsed.triangles:
            tokens.extend(
                (
                    self.TRIANGLE,
                    self._coordinate_token(record.x0),
                    self._coordinate_token(record.y0),
                    self._coordinate_token(record.x1),
                    self._coordinate_token(record.y1),
                    self._coordinate_token(record.x2),
                    self._coordinate_token(record.y2),
                    self._width_token(record.stroke_width),
                )
            )
        for record in parsed.quads:
            tokens.extend(
                (
                    self.QUAD,
                    self._coordinate_token(record.x0),
                    self._coordinate_token(record.y0),
                    self._coordinate_token(record.x1),
                    self._coordinate_token(record.y1),
                    self._coordinate_token(record.x2),
                    self._coordinate_token(record.y2),
                    self._coordinate_token(record.x3),
                    self._coordinate_token(record.y3),
                    self._width_token(record.stroke_width),
                )
            )
        for record in parsed.text_boxes:
            tokens.extend(
                (
                    self.TEXT_BOX,
                    self._coordinate_token(record.x0),
                    self._coordinate_token(record.y0),
                    self._coordinate_token(record.x1),
                    self._coordinate_token(record.y1),
                )
            )
        tokens.append(self.DOCUMENT_END)
        if add_special_tokens:
            tokens = [self.BOS, *tokens, self.EOS]
        return [self.token_to_id[token] for token in tokens]

    @staticmethod
    def _value(token: str, prefix: str, field: str) -> int:
        if not token.startswith(prefix):
            raise VisualSVGSequenceError(f"Expected {field}, got {token}")
        return int(token[len(prefix) :])

    def decode(self, input_ids: Iterable[int], *, skip_padding: bool = True) -> str:
        tokens = []
        for token_id in input_ids:
            token_id = int(token_id)
            if not 0 <= token_id < len(self.id_to_token):
                raise VisualSVGSequenceError(f"Token ID is outside the vocabulary: {token_id}")
            token = self.id_to_token[token_id]
            if not (skip_padding and token == self.PAD):
                tokens.append(token)
        if tokens and tokens[0] == self.BOS:
            tokens.pop(0)
        if tokens and tokens[-1] == self.EOS:
            tokens.pop()
        if not tokens or tokens[0] != self.DOCUMENT or tokens[-1] != self.DOCUMENT_END:
            raise VisualSVGSequenceError("Token stream lacks Visual SVG document markers")

        lines = []
        triangles = []
        quads = []
        text_boxes = []
        cursor = 1
        while cursor < len(tokens) - 1:
            marker = tokens[cursor]
            cursor += 1
            if marker == self.LINE:
                fields = tokens[cursor : cursor + 5]
                if len(fields) != 5:
                    raise VisualSVGSequenceError("Truncated line token record")
                cursor += 5
                lines.append(
                    LineRecordV01(
                        *(self._value(value, "c:", "coordinate") for value in fields[:4]),
                        self._value(fields[4], "w:", "stroke width"),
                    )
                )
            elif marker == self.TRIANGLE:
                fields = tokens[cursor : cursor + 7]
                if len(fields) != 7:
                    raise VisualSVGSequenceError("Truncated triangle token record")
                cursor += 7
                triangles.append(
                    TriangleRecordV01(
                        *(self._value(value, "c:", "coordinate") for value in fields[:6]),
                        self._value(fields[6], "w:", "stroke width"),
                    )
                )
            elif marker == self.TEXT_BOX:
                fields = tokens[cursor : cursor + 4]
                if len(fields) != 4:
                    raise VisualSVGSequenceError("Truncated text-box token record")
                cursor += 4
                text_boxes.append(
                    TextBoxRecordV01(
                        *(self._value(value, "c:", "coordinate") for value in fields)
                    )
                )
            elif marker == self.QUAD:
                fields = tokens[cursor : cursor + 9]
                if len(fields) != 9:
                    raise VisualSVGSequenceError("Truncated quad token record")
                cursor += 9
                quads.append(
                    QuadRecordV01(
                        *(self._value(value, "c:", "coordinate") for value in fields[:8]),
                        self._value(fields[8], "w:", "stroke width"),
                    )
                )
            else:
                raise VisualSVGSequenceError(f"Unexpected Visual SVG marker: {marker}")
        return serialize_parsed_visual_svg_v0_1(
            ParsedVisualSVGSequenceV01(
                tuple(lines), tuple(triangles), tuple(quads), tuple(text_boxes)
            )
        )

    def encode_batch(
        self,
        sequences: Iterable[str],
        *,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> dict[str, list[list[int]]]:
        encoded = [self.encode(sequence) for sequence in sequences]
        required = max((len(ids) for ids in encoded), default=0)
        if max_length is not None and required > max_length:
            raise VisualSVGSequenceError(
                f"Batch requires {required} tokens but max_length is {max_length}"
            )
        target_length = required
        if pad_to_multiple_of and target_length % pad_to_multiple_of:
            target_length += pad_to_multiple_of - target_length % pad_to_multiple_of
        input_ids = []
        attention_mask = []
        for ids in encoded:
            padding = target_length - len(ids)
            input_ids.append([*ids, *([self.pad_token_id] * padding)])
            attention_mask.append([*([1] * len(ids)), *([0] * padding)])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "sequence_schema_version": self.sequence_schema_version,
            "coordinate_max": COORDINATE_MAX,
            "stroke_width_min": STROKE_WIDTH_MIN,
            "stroke_width_max": STROKE_WIDTH_MAX,
            "vocabulary": self.vocabulary,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VisualSVGTokenizerV01":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported tokenizer schema: {payload.get('schema_version')}")
        return cls(vocabulary=payload["vocabulary"])
