#!/usr/bin/env python3
"""Compact tokenizer for coordinate-grounded ChemSVG v0.4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rdkit import Chem

try:
    from .chemmol_sequence import ChemMolSequenceError
    from .chemsvg_sequence_v4 import (
        END_TOKEN,
        START_TOKEN,
        ParsedChemSVGSequenceV4,
        StrokeRecordV4,
        VertexRecordV4,
        parse_chemsvg_v4,
    )
except ImportError:
    from chemmol_sequence import ChemMolSequenceError
    from chemsvg_sequence_v4 import (
        END_TOKEN,
        START_TOKEN,
        ParsedChemSVGSequenceV4,
        StrokeRecordV4,
        VertexRecordV4,
        parse_chemsvg_v4,
    )


TOKENIZER_SCHEMA_VERSION = "chemsvg-tokenizer-0.1"


class ChemSVGTokenizerV4:
    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    UNK = "[UNK]"
    MOL = "[CHEMSVG_V04]"
    VERTEX = "[VERTEX]"
    STROKE = "[STROKE]"
    END_RECORD = "[END_RECORD]"
    MOL_END = "[CHEMSVG_END]"

    SPECIAL_TOKENS = (
        PAD,
        BOS,
        EOS,
        UNK,
        MOL,
        VERTEX,
        STROKE,
        END_RECORD,
        MOL_END,
    )
    ELEMENTS = ("B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si")
    INTEGER_MIN = -16
    INTEGER_MAX = 2047
    sequence_schema_version = "chemsvg-0.4"

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary or self.build_vocabulary()
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_token = tuple(self.vocabulary)

    @classmethod
    def build_vocabulary(cls) -> list[str]:
        chiral = [f"chi:{name}" for name in dir(Chem.ChiralType) if name.isupper()]
        bond_types = [name for name in dir(Chem.BondType) if name.isupper()]
        directions = [f"dir:{name}" for name in dir(Chem.BondDir) if name.isupper()]
        stereo = [f"st:{name}" for name in dir(Chem.BondStereo) if name.isupper()]
        return [
            *cls.SPECIAL_TOKENS,
            *(f"el:{element}" for element in cls.ELEMENTS),
            *(f"n:{value}" for value in range(cls.INTEGER_MIN, cls.INTEGER_MAX + 1)),
            *sorted(chiral),
            *(f"vis:{name}" for name in sorted(bond_types)),
            *(f"bt:{name}" for name in sorted(bond_types)),
            *sorted(directions),
            *sorted(stereo),
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
            raise ChemMolSequenceError(f"Token is outside the v0.4 vocabulary: {token}") from error

    def _number_token(self, value: int) -> str:
        if not self.INTEGER_MIN <= value <= self.INTEGER_MAX:
            raise ChemMolSequenceError(f"Integer {value} is outside the tokenizer range")
        return f"n:{value}"

    @staticmethod
    def _strip(token: str, prefix: str, field: str) -> str:
        if not token.startswith(prefix):
            raise ChemMolSequenceError(f"Expected {field} token, got {token}")
        return token[len(prefix) :]

    def _number(self, token: str, field: str) -> int:
        return int(self._strip(token, "n:", field))

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> list[int]:
        parsed = parse_chemsvg_v4(sequence)
        tokens = [self.MOL]
        for vertex in parsed.vertices:
            tokens.extend(
                (
                    self.VERTEX,
                    self._number_token(vertex.x),
                    self._number_token(vertex.y),
                    f"el:{vertex.element}",
                    self._number_token(vertex.formal_charge),
                    self._number_token(vertex.isotope),
                    self._number_token(int(vertex.aromatic)),
                    self._number_token(vertex.explicit_h_count),
                    self._number_token(int(vertex.no_implicit)),
                    self._number_token(vertex.radical_electrons),
                    f"chi:{vertex.canonical_chiral_tag}",
                    self.END_RECORD,
                )
            )
        for stroke in parsed.strokes:
            tokens.extend(
                (
                    self.STROKE,
                    self._number_token(stroke.begin_x),
                    self._number_token(stroke.begin_y),
                    self._number_token(stroke.end_x),
                    self._number_token(stroke.end_y),
                    f"vis:{stroke.display_type}",
                    f"bt:{stroke.bond_type}",
                    f"dir:{stroke.direction}",
                    f"st:{stroke.stereo}",
                )
            )
            if stroke.stereo_atom_0 is not None:
                tokens.extend(
                    (
                        self._number_token(stroke.stereo_atom_0_x),
                        self._number_token(stroke.stereo_atom_0_y),
                        self._number_token(stroke.stereo_atom_1_x),
                        self._number_token(stroke.stereo_atom_1_y),
                    )
                )
            tokens.append(self.END_RECORD)
        tokens.append(self.MOL_END)
        if add_special_tokens:
            tokens = [self.BOS, *tokens, self.EOS]
        return [self._lookup(token) for token in tokens]

    def decode(self, input_ids: Iterable[int], *, skip_padding: bool = True) -> str:
        tokens = []
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
            raise ChemMolSequenceError("Token stream must contain ChemSVG v0.4 markers")

        vertices = []
        strokes = []
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
            if marker == self.VERTEX:
                if strokes or len(fields) != 10:
                    raise ChemMolSequenceError("Invalid ChemSVG vertex token record")
                vertices.append(
                    VertexRecordV4(
                        x=self._number(fields[0], "vertex.x"),
                        y=self._number(fields[1], "vertex.y"),
                        element=self._strip(fields[2], "el:", "element"),
                        formal_charge=self._number(fields[3], "formal_charge"),
                        isotope=self._number(fields[4], "isotope"),
                        aromatic=bool(self._number(fields[5], "aromatic")),
                        explicit_h_count=self._number(fields[6], "explicit_h_count"),
                        no_implicit=bool(self._number(fields[7], "no_implicit")),
                        radical_electrons=self._number(fields[8], "radical_electrons"),
                        canonical_chiral_tag=self._strip(
                            fields[9],
                            "chi:",
                            "chiral_tag",
                        ),
                    )
                )
            elif marker == self.STROKE:
                if len(fields) not in {8, 12}:
                    raise ChemMolSequenceError("Invalid ChemSVG stroke token record")
                stereo_values = (
                    tuple(self._number(value, "stereo_coordinate") for value in fields[8:])
                    if len(fields) == 12
                    else (-1, -1, -1, -1)
                )
                strokes.append(
                    StrokeRecordV4(
                        begin_x=self._number(fields[0], "stroke.begin_x"),
                        begin_y=self._number(fields[1], "stroke.begin_y"),
                        end_x=self._number(fields[2], "stroke.end_x"),
                        end_y=self._number(fields[3], "stroke.end_y"),
                        display_type=self._strip(fields[4], "vis:", "display_type"),
                        bond_type=self._strip(fields[5], "bt:", "bond_type"),
                        direction=self._strip(fields[6], "dir:", "direction"),
                        stereo=self._strip(fields[7], "st:", "stereo"),
                        stereo_atom_0_x=stereo_values[0],
                        stereo_atom_0_y=stereo_values[1],
                        stereo_atom_1_x=stereo_values[2],
                        stereo_atom_1_y=stereo_values[3],
                    )
                )
            else:
                raise ChemMolSequenceError(f"Unexpected ChemSVG token marker: {marker}")

        lines = [START_TOKEN]
        for vertex in vertices:
            lines.append(
                f"<v {vertex.x} {vertex.y} {vertex.element} {vertex.formal_charge} "
                f"{vertex.isotope} {int(vertex.aromatic)} {vertex.explicit_h_count} "
                f"{int(vertex.no_implicit)} {vertex.radical_electrons} "
                f"{vertex.canonical_chiral_tag}>"
            )
        for stroke in strokes:
            line = (
                f"<s {stroke.begin_x} {stroke.begin_y} {stroke.end_x} {stroke.end_y} "
                f"{stroke.display_type} {stroke.bond_type} {stroke.direction} {stroke.stereo}"
            )
            if stroke.stereo_atom_0 is not None:
                line += (
                    f" {stroke.stereo_atom_0_x} {stroke.stereo_atom_0_y} "
                    f"{stroke.stereo_atom_1_x} {stroke.stereo_atom_1_y}"
                )
            lines.append(line + ">")
        lines.append(END_TOKEN)
        sequence = "\n".join(lines) + "\n"
        parse_chemsvg_v4(sequence)
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
    def load(cls, path: str | Path) -> "ChemSVGTokenizerV4":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported tokenizer schema: {payload.get('schema_version')}")
        return cls(vocabulary=payload["vocabulary"])
