#!/usr/bin/env python3
"""Compact grammar tokenizer for ChemMolSeq v0.3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rdkit import Chem

try:
    from .chemmol_sequence import ChemMolSequenceError
    from .chemmol_sequence_v3 import (
        END_TOKEN,
        START_TOKEN,
        COORDS_END_TOKEN,
        COORDS_START_TOKEN,
        AtomRecordV3,
        BondRecordV3,
        ParsedChemMolSequenceV3,
        parse_chemmolseq_v3,
    )
except ImportError:
    from chemmol_sequence import ChemMolSequenceError
    from chemmol_sequence_v3 import (
        END_TOKEN,
        START_TOKEN,
        COORDS_END_TOKEN,
        COORDS_START_TOKEN,
        AtomRecordV3,
        BondRecordV3,
        ParsedChemMolSequenceV3,
        parse_chemmolseq_v3,
    )


TOKENIZER_SCHEMA_VERSION = "chemmol-tokenizer-0.2"


class ChemMolTokenizerV3:
    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    UNK = "[UNK]"
    MOL = "[MOL_V03]"
    ATOM = "[ATOM]"
    BOND = "[BOND]"
    END_RECORD = "[END_RECORD]"
    MOL_END = "[MOL_END]"
    COORDS = "[COORDS]"
    COORDS_END = "[COORDS_END]"

    SPECIAL_TOKENS = (
        PAD,
        BOS,
        EOS,
        UNK,
        MOL,
        ATOM,
        BOND,
        END_RECORD,
        MOL_END,
        COORDS,
        COORDS_END,
    )
    ELEMENTS = ("B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si")
    INTEGER_MIN = -16
    INTEGER_MAX = 2047
    sequence_schema_version = "chemmolseq-0.3"

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
        stereo = [f"st:{name}" for name in dir(Chem.BondStereo) if name.isupper()]
        return [
            *cls.SPECIAL_TOKENS,
            *(f"el:{element}" for element in cls.ELEMENTS),
            *(f"n:{value}" for value in range(cls.INTEGER_MIN, cls.INTEGER_MAX + 1)),
            *sorted(chiral),
            *(f"bt:{name}" for name in sorted(bond_types)),
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
            raise ChemMolSequenceError(f"Token is outside the v0.3 vocabulary: {token}") from error

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
        parsed = parse_chemmolseq_v3(sequence)
        tokens = [self.MOL]
        for atom in parsed.atoms:
            tokens.extend(
                (
                    self.ATOM,
                    f"el:{atom.element}",
                    self._number_token(atom.formal_charge),
                    self._number_token(atom.isotope),
                    self._number_token(int(atom.aromatic)),
                    self._number_token(atom.explicit_h_count),
                    self._number_token(int(atom.no_implicit)),
                    self._number_token(atom.radical_electrons),
                    f"chi:{atom.canonical_chiral_tag}",
                    self.END_RECORD,
                )
            )
        for bond in parsed.bonds:
            tokens.extend(
                (
                    self.BOND,
                    self._number_token(bond.begin_atom),
                    self._number_token(bond.end_atom),
                    f"bt:{bond.bond_type}",
                    f"st:{bond.stereo}",
                    self._number_token(bond.stereo_atom_0),
                    self._number_token(bond.stereo_atom_1),
                    self.END_RECORD,
                )
            )
        tokens.append(self.COORDS)
        for x, y in parsed.positions:
            tokens.extend((self._number_token(x), self._number_token(y)))
        tokens.append(self.COORDS_END)
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
            raise ChemMolSequenceError("Token stream must contain v0.3 molecule markers")

        atoms = []
        bonds = []
        positions = []
        cursor = 1
        while cursor < len(tokens) - 1:
            marker = tokens[cursor]
            cursor += 1
            if marker == self.COORDS:
                for _ in atoms:
                    if cursor + 1 >= len(tokens):
                        raise ChemMolSequenceError("Coordinate token section is truncated")
                    positions.append(
                        (
                            self._number(tokens[cursor], "position.x"),
                            self._number(tokens[cursor + 1], "position.y"),
                        )
                    )
                    cursor += 2
                if cursor >= len(tokens) or tokens[cursor] != self.COORDS_END:
                    raise ChemMolSequenceError("Coordinate token section has no end marker")
                cursor += 1
                continue
            try:
                record_end = tokens.index(self.END_RECORD, cursor)
            except ValueError as error:
                raise ChemMolSequenceError("Token record is missing END_RECORD") from error
            fields = tokens[cursor:record_end]
            cursor = record_end + 1
            if marker == self.ATOM:
                if bonds or len(fields) != 8:
                    raise ChemMolSequenceError("Invalid v0.3 atom token record")
                atoms.append(
                    AtomRecordV3(
                        element=self._strip(fields[0], "el:", "element"),
                        formal_charge=self._number(fields[1], "formal_charge"),
                        isotope=self._number(fields[2], "isotope"),
                        aromatic=bool(self._number(fields[3], "aromatic")),
                        explicit_h_count=self._number(fields[4], "explicit_h_count"),
                        no_implicit=bool(self._number(fields[5], "no_implicit")),
                        radical_electrons=self._number(fields[6], "radical_electrons"),
                        canonical_chiral_tag=self._strip(fields[7], "chi:", "chiral_tag"),
                    )
                )
            elif marker == self.BOND:
                if len(fields) != 6:
                    raise ChemMolSequenceError("Invalid v0.3 bond token record")
                bonds.append(
                    BondRecordV3(
                        begin_atom=self._number(fields[0], "begin_atom"),
                        end_atom=self._number(fields[1], "end_atom"),
                        bond_type=self._strip(fields[2], "bt:", "bond_type"),
                        stereo=self._strip(fields[3], "st:", "stereo"),
                        stereo_atom_0=self._number(fields[4], "stereo_atom_0"),
                        stereo_atom_1=self._number(fields[5], "stereo_atom_1"),
                    )
                )
            else:
                raise ChemMolSequenceError(f"Unexpected record marker: {marker}")

        parsed = ParsedChemMolSequenceV3(tuple(atoms), tuple(bonds), tuple(positions))
        lines = [START_TOKEN]
        for atom in parsed.atoms:
            lines.append(
                f"<a {atom.element} {atom.formal_charge} {atom.isotope} "
                f"{int(atom.aromatic)} {atom.explicit_h_count} {int(atom.no_implicit)} "
                f"{atom.radical_electrons} {atom.canonical_chiral_tag}>"
            )
        for bond in parsed.bonds:
            lines.append(
                f"<b {bond.begin_atom} {bond.end_atom} {bond.bond_type} {bond.stereo} "
                f"{bond.stereo_atom_0} {bond.stereo_atom_1}>"
            )
        lines.append(COORDS_START_TOKEN)
        for x, y in parsed.positions:
            lines.append(f"<p {x} {y}>")
        lines.append(COORDS_END_TOKEN)
        lines.append(END_TOKEN)
        sequence = "\n".join(lines) + "\n"
        parse_chemmolseq_v3(sequence)
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
    def load(cls, path: str | Path) -> "ChemMolTokenizerV3":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported tokenizer schema: {payload.get('schema_version')}")
        return cls(vocabulary=payload["vocabulary"])
