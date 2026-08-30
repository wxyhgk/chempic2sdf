#!/usr/bin/env python3
"""Structured tokenizer for ChemTraverse v0.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rdkit import Chem

try:
    from .chemmol_sequence import ChemMolSequenceError
    from .chemtraverse_v0_1 import END, HEADER, parse_chemtraverse_v0_1
except ImportError:
    from chemmol_sequence import ChemMolSequenceError
    from chemtraverse_v0_1 import END, HEADER, parse_chemtraverse_v0_1


TOKENIZER_SCHEMA_VERSION = "chemtraverse-tokenizer-0.1"


class ChemTraverseTokenizerV01:
    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    UNK = "[UNK]"
    MOL = "[CHEMTRAVERSE_V01]"
    META = "[META]"
    COMPONENT = "[COMPONENT]"
    ATOM = "[ATOM]"
    DOWN = "[DOWN]"
    UP = "[UP]"
    LINK = "[LINK]"
    COMPONENT_END = "[COMPONENT_END]"
    TEXT = "[TEXT]"
    STEREO = "[STEREO]"
    END_RECORD = "[END_RECORD]"
    MOL_END = "[CHEMTRAVERSE_END]"
    UTF8 = "[UTF8]"
    UTF8_END = "[UTF8_END]"
    BBOX_NONE = "[BBOX_NONE]"
    BBOX = "[BBOX]"

    SPECIAL_TOKENS = (
        PAD,
        BOS,
        EOS,
        UNK,
        MOL,
        META,
        COMPONENT,
        ATOM,
        DOWN,
        UP,
        LINK,
        COMPONENT_END,
        TEXT,
        STEREO,
        END_RECORD,
        MOL_END,
        UTF8,
        UTF8_END,
        BBOX_NONE,
        BBOX,
    )
    INTEGER_MIN = -16
    INTEGER_MAX = 2047
    sequence_schema_version = "chemtraverse-0.1"

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary or self.build_vocabulary()
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_token = tuple(self.vocabulary)

    @classmethod
    def build_vocabulary(cls) -> list[str]:
        table = Chem.GetPeriodicTable()
        elements = ["*", *(table.GetElementSymbol(value) for value in range(1, 119))]
        bond_types = [name for name in dir(Chem.BondType) if name.isupper()]
        directions = [name for name in dir(Chem.BondDir) if name.isupper()]
        stereo = [name for name in dir(Chem.BondStereo) if name.isupper()]
        return [
            *cls.SPECIAL_TOKENS,
            *(f"n:{value}" for value in range(cls.INTEGER_MIN, cls.INTEGER_MAX + 1)),
            *(f"el:{value}" for value in elements),
            "role:IMPLICIT",
            "role:TEXT",
            "chi:-",
            "chi:TCW",
            "chi:TCCW",
            "ori:F",
            "ori:R",
            *(f"vis:{value}" for value in sorted(bond_types)),
            *(f"bt:{value}" for value in sorted(bond_types)),
            *(f"dir:{value}" for value in sorted(directions)),
            *(f"st:{value}" for value in sorted(stereo)),
            *(f"byte:{value:02x}" for value in range(256)),
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

    def _lookup(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as error:
            raise ChemMolSequenceError(
                f"Token is outside the ChemTraverse v0.1 vocabulary: {token}"
            ) from error

    def _number_token(self, value: str | int) -> str:
        try:
            number = int(value)
        except ValueError as error:
            raise ChemMolSequenceError(f"Expected integer, got {value!r}") from error
        if not self.INTEGER_MIN <= number <= self.INTEGER_MAX:
            raise ChemMolSequenceError(f"Integer {number} is outside the tokenizer range")
        return f"n:{number}"

    @staticmethod
    def _strip(token: str, prefix: str, field: str) -> str:
        if not token.startswith(prefix):
            raise ChemMolSequenceError(f"Expected {field} token, got {token}")
        return token[len(prefix) :]

    def _number(self, token: str, field: str) -> str:
        return str(int(self._strip(token, "n:", field)))

    def _utf8_tokens(self, value: str) -> list[str]:
        if not value.startswith("utf8:") or len(value) == 5:
            raise ChemMolSequenceError("Expected non-empty utf8:<hex> field")
        try:
            payload = bytes.fromhex(value[5:])
            payload.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise ChemMolSequenceError("Invalid UTF-8 field") from error
        return [self.UTF8, *(f"byte:{item:02x}" for item in payload), self.UTF8_END]

    def _utf8_field(self, tokens: list[str], cursor: int) -> tuple[str, int]:
        if cursor >= len(tokens) or tokens[cursor] != self.UTF8:
            raise ChemMolSequenceError("UTF-8 field is missing its start marker")
        cursor += 1
        payload = bytearray()
        while cursor < len(tokens) and tokens[cursor] != self.UTF8_END:
            value = self._strip(tokens[cursor], "byte:", "UTF-8 byte")
            try:
                payload.append(int(value, 16))
            except ValueError as error:
                raise ChemMolSequenceError("Invalid UTF-8 byte token") from error
            cursor += 1
        if cursor >= len(tokens) or not payload:
            raise ChemMolSequenceError("UTF-8 field is empty or unterminated")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ChemMolSequenceError("Token bytes are not valid UTF-8") from error
        return "utf8:" + payload.hex(), cursor + 1

    @staticmethod
    def _record_fields(line: str) -> list[str]:
        if not line.startswith("<") or not line.endswith(">"):
            raise ChemMolSequenceError(f"Malformed ChemTraverse record: {line!r}")
        return line[1:-1].split()

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> list[int]:
        parse_chemtraverse_v0_1(sequence)
        lines = [line.strip() for line in sequence.splitlines() if line.strip()]
        if lines[0] != HEADER or lines[-1] != END:
            raise ChemMolSequenceError("ChemTraverse markers are missing")
        tokens = [self.MOL]
        markers = {
            "meta": self.META,
            "component": self.COMPONENT,
            "atom": self.ATOM,
            "down": self.DOWN,
            "up": self.UP,
            "link": self.LINK,
            "component_end": self.COMPONENT_END,
            "text": self.TEXT,
            "stereo": self.STEREO,
        }
        for line in lines[1:-1]:
            fields = self._record_fields(line)
            name = fields[0]
            if name not in markers:
                raise ChemMolSequenceError(f"Unknown ChemTraverse record: {name}")
            record = [markers[name]]
            if name == "meta":
                record.extend((self._number_token(fields[1]), self._number_token(fields[2])))
                record.extend(self._utf8_tokens(fields[3]))
            elif name == "atom":
                record.extend(
                    (
                        self._number_token(fields[1]),
                        self._number_token(fields[2]),
                        f"role:{fields[3]}",
                        f"el:{fields[4]}",
                        *(self._number_token(value) for value in fields[5:11]),
                        f"chi:{fields[11]}",
                    )
                )
            elif name == "down":
                record.extend(
                    (
                        f"ori:{fields[1]}",
                        f"vis:{fields[2]}",
                        f"bt:{fields[3]}",
                        f"dir:{fields[4]}",
                        f"st:{fields[5]}",
                    )
                )
            elif name == "link":
                record.extend(
                    (
                        self._number_token(fields[1]),
                        f"ori:{fields[2]}",
                        f"vis:{fields[3]}",
                        f"bt:{fields[4]}",
                        f"dir:{fields[5]}",
                        f"st:{fields[6]}",
                    )
                )
            elif name == "text":
                record.extend(self._number_token(value) for value in fields[1:4])
                if fields[4] == "-":
                    record.append(self.BBOX_NONE)
                else:
                    bbox = fields[4].split(",")
                    if len(bbox) != 4:
                        raise ChemMolSequenceError("Text bbox has the wrong arity")
                    record.append(self.BBOX)
                    record.extend(self._number_token(value) for value in bbox)
                record.extend(self._utf8_tokens(fields[5]))
            elif name == "stereo":
                record.extend(self._number_token(value) for value in fields[1:4])
            tokens.extend((*record, self.END_RECORD))
        tokens.append(self.MOL_END)
        if add_special_tokens:
            tokens = [self.BOS, *tokens, self.EOS]
        return [self._lookup(token) for token in tokens]

    def decode(self, input_ids: Iterable[int], *, skip_padding: bool = True) -> str:
        tokens: list[str] = []
        for token_id in input_ids:
            if not 0 <= int(token_id) < len(self.id_to_token):
                raise ChemMolSequenceError(f"Token ID is outside the vocabulary: {token_id}")
            token = self.id_to_token[int(token_id)]
            if not (skip_padding and token == self.PAD):
                tokens.append(token)
        if tokens and tokens[0] == self.BOS:
            tokens.pop(0)
        if tokens and tokens[-1] == self.EOS:
            tokens.pop()
        if not tokens or tokens[0] != self.MOL or tokens[-1] != self.MOL_END:
            raise ChemMolSequenceError("Token stream must contain ChemTraverse markers")

        marker_names = {
            self.META: "meta",
            self.COMPONENT: "component",
            self.ATOM: "atom",
            self.DOWN: "down",
            self.UP: "up",
            self.LINK: "link",
            self.COMPONENT_END: "component_end",
            self.TEXT: "text",
            self.STEREO: "stereo",
        }
        lines = [HEADER]
        cursor = 1
        while cursor < len(tokens) - 1:
            marker = tokens[cursor]
            cursor += 1
            try:
                record_end = tokens.index(self.END_RECORD, cursor)
            except ValueError as error:
                raise ChemMolSequenceError("Token record is missing END_RECORD") from error
            fields = tokens[cursor:record_end]
            cursor = record_end + 1
            name = marker_names.get(marker)
            if name is None:
                raise ChemMolSequenceError(f"Unexpected ChemTraverse marker: {marker}")
            values: list[str] = [name]
            if name == "meta":
                if len(fields) < 5:
                    raise ChemMolSequenceError("Invalid meta token record")
                values.extend((self._number(fields[0], "width"), self._number(fields[1], "height")))
                utf8, end = self._utf8_field(fields, 2)
                if end != len(fields):
                    raise ChemMolSequenceError("Trailing tokens in meta record")
                values.append(utf8)
            elif name == "atom":
                if len(fields) != 11:
                    raise ChemMolSequenceError("Invalid atom token record")
                values.extend(
                    (
                        self._number(fields[0], "atom.x"),
                        self._number(fields[1], "atom.y"),
                        self._strip(fields[2], "role:", "role"),
                        self._strip(fields[3], "el:", "element"),
                        *(self._number(value, "atom integer") for value in fields[4:10]),
                        self._strip(fields[10], "chi:", "chiral"),
                    )
                )
            elif name == "down":
                if len(fields) != 5:
                    raise ChemMolSequenceError("Invalid down token record")
                values.extend(
                    self._strip(value, prefix, label)
                    for value, prefix, label in zip(
                        fields,
                        ("ori:", "vis:", "bt:", "dir:", "st:"),
                        ("orientation", "display", "bond", "direction", "stereo"),
                    )
                )
            elif name == "link":
                if len(fields) != 6:
                    raise ChemMolSequenceError("Invalid link token record")
                values.append(self._number(fields[0], "link reference"))
                values.extend(
                    self._strip(value, prefix, label)
                    for value, prefix, label in zip(
                        fields[1:],
                        ("ori:", "vis:", "bt:", "dir:", "st:"),
                        ("orientation", "display", "bond", "direction", "stereo"),
                    )
                )
            elif name == "text":
                if len(fields) < 5:
                    raise ChemMolSequenceError("Invalid text token record")
                values.extend(self._number(value, "text integer") for value in fields[:3])
                position = 3
                if fields[position] == self.BBOX_NONE:
                    values.append("-")
                    position += 1
                elif fields[position] == self.BBOX:
                    if len(fields) < position + 6:
                        raise ChemMolSequenceError("Incomplete text bbox")
                    bbox = [self._number(value, "bbox") for value in fields[position + 1 : position + 5]]
                    values.append(",".join(bbox))
                    position += 5
                else:
                    raise ChemMolSequenceError("Text bbox marker is missing")
                utf8, position = self._utf8_field(fields, position)
                if position != len(fields):
                    raise ChemMolSequenceError("Trailing tokens in text record")
                values.append(utf8)
            elif name == "stereo":
                if len(fields) != 3:
                    raise ChemMolSequenceError("Invalid stereo token record")
                values.extend(self._number(value, "stereo reference") for value in fields)
            elif fields:
                raise ChemMolSequenceError(f"{name} record must not contain fields")
            lines.append("<" + " ".join(values) + ">")
        lines.append(END)
        sequence = "\n".join(lines) + "\n"
        parse_chemtraverse_v0_1(sequence)
        return sequence

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
            raise ChemMolSequenceError(
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
            "integer_min": self.INTEGER_MIN,
            "integer_max": self.INTEGER_MAX,
            "vocabulary": self.vocabulary,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChemTraverseTokenizerV01":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported tokenizer schema: {payload.get('schema_version')}")
        return cls(vocabulary=payload["vocabulary"])
