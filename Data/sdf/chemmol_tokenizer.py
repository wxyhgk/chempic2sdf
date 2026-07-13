#!/usr/bin/env python3
"""A compact grammar tokenizer for ChemMolSeq v0.2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rdkit import Chem

try:
    from .chemmol_sequence import (
        END_TOKEN,
        START_TOKEN,
        AtomRecord,
        BondRecord,
        ChemMolSequenceError,
        ParsedChemMolSequence,
        parse_chemmolseq,
    )
except ImportError:
    from chemmol_sequence import (
        END_TOKEN,
        START_TOKEN,
        AtomRecord,
        BondRecord,
        ChemMolSequenceError,
        ParsedChemMolSequence,
        parse_chemmolseq,
    )


TOKENIZER_SCHEMA_VERSION = "chemmol-tokenizer-0.1"


class ChemMolTokenizer:
    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    UNK = "[UNK]"
    MOL = "[MOL_V02]"
    ATOM = "[ATOM]"
    BOND = "[BOND]"
    END_RECORD = "[END_RECORD]"
    MOL_END = "[MOL_END]"

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
    )
    ELEMENTS = ("B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S", "Se", "Si")
    INTEGER_MIN = -16
    INTEGER_MAX = 2047
    sequence_schema_version = "chemmolseq-0.2"

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary or self.build_vocabulary()
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_token = tuple(self.vocabulary)
        for token in self.SPECIAL_TOKENS:
            if token not in self.token_to_id:
                raise ValueError(f"Tokenizer vocabulary is missing {token}")

    @classmethod
    def build_vocabulary(cls) -> list[str]:
        chiral = [f"chi:{name}" for name in dir(Chem.ChiralType) if name.isupper()]
        bond_types = [name for name in dir(Chem.BondType) if name.isupper()]
        stereo = [f"st:{name}" for name in dir(Chem.BondStereo) if name.isupper()]
        directions = [f"dir:{name}" for name in dir(Chem.BondDir) if name.isupper()]
        return [
            *cls.SPECIAL_TOKENS,
            *(f"el:{element}" for element in cls.ELEMENTS),
            *(f"n:{value}" for value in range(cls.INTEGER_MIN, cls.INTEGER_MAX + 1)),
            *sorted(chiral),
            *(f"bt:{name}" for name in sorted(bond_types)),
            *(f"dt:{name}" for name in sorted(bond_types)),
            *sorted(stereo),
            *sorted(directions),
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
            raise ChemMolSequenceError(f"Token is outside the v0 vocabulary: {token}") from error

    def _number_token(self, value: int) -> str:
        if not self.INTEGER_MIN <= value <= self.INTEGER_MAX:
            raise ChemMolSequenceError(
                f"Integer {value} is outside [{self.INTEGER_MIN}, {self.INTEGER_MAX}]"
            )
        return f"n:{value}"

    @staticmethod
    def _strip_prefix(token: str, prefix: str, field: str) -> str:
        if not token.startswith(prefix):
            raise ChemMolSequenceError(f"Expected {field} token with prefix {prefix}, got {token}")
        return token[len(prefix) :]

    def _decode_number(self, token: str, field: str) -> int:
        value = self._strip_prefix(token, "n:", field)
        try:
            return int(value)
        except ValueError as error:
            raise ChemMolSequenceError(f"Invalid integer token for {field}: {token}") from error

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> list[int]:
        parsed = parse_chemmolseq(sequence)
        tokens: list[str] = [self.MOL]
        for atom in parsed.atoms:
            tokens.extend(
                (
                    self.ATOM,
                    self._number_token(atom.atom_id),
                    f"el:{atom.element}",
                    self._number_token(atom.formal_charge),
                    self._number_token(atom.isotope),
                    self._number_token(int(atom.aromatic)),
                    self._number_token(atom.explicit_h_count),
                    self._number_token(int(atom.no_implicit)),
                    self._number_token(atom.radical_electrons),
                    f"chi:{atom.chiral_tag}",
                    self._number_token(atom.quantized_x),
                    self._number_token(atom.quantized_y),
                    self.END_RECORD,
                )
            )
        for bond in parsed.bonds:
            tokens.extend(
                (
                    self.BOND,
                    self._number_token(bond.bond_id),
                    self._number_token(bond.begin_atom),
                    self._number_token(bond.end_atom),
                    f"bt:{bond.bond_type}",
                    f"dt:{bond.display_type}",
                    f"st:{bond.stereo}",
                    self._number_token(bond.stereo_atom_0),
                    self._number_token(bond.stereo_atom_1),
                    f"dir:{bond.direction}",
                    self.END_RECORD,
                )
            )
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
            if skip_padding and token == self.PAD:
                continue
            tokens.append(token)
        if tokens and tokens[0] == self.BOS:
            tokens.pop(0)
        if tokens and tokens[-1] == self.EOS:
            tokens.pop()
        if not tokens or tokens[0] != self.MOL or tokens[-1] != self.MOL_END:
            raise ChemMolSequenceError("Token stream must contain MOL and MOL_END markers")

        atoms: list[AtomRecord] = []
        bonds: list[BondRecord] = []
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
            if marker == self.ATOM:
                if bonds:
                    raise ChemMolSequenceError("Atom token records must precede bonds")
                if len(fields) != 11:
                    raise ChemMolSequenceError(f"ATOM record has {len(fields)} fields, expected 11")
                atoms.append(
                    AtomRecord(
                        atom_id=self._decode_number(fields[0], "atom.id"),
                        element=self._strip_prefix(fields[1], "el:", "element"),
                        formal_charge=self._decode_number(fields[2], "formal_charge"),
                        isotope=self._decode_number(fields[3], "isotope"),
                        aromatic=bool(self._decode_number(fields[4], "aromatic")),
                        explicit_h_count=self._decode_number(fields[5], "explicit_h_count"),
                        no_implicit=bool(self._decode_number(fields[6], "no_implicit")),
                        radical_electrons=self._decode_number(fields[7], "radical_electrons"),
                        chiral_tag=self._strip_prefix(fields[8], "chi:", "chiral_tag"),
                        quantized_x=self._decode_number(fields[9], "quantized_x"),
                        quantized_y=self._decode_number(fields[10], "quantized_y"),
                    )
                )
            elif marker == self.BOND:
                if len(fields) != 9:
                    raise ChemMolSequenceError(f"BOND record has {len(fields)} fields, expected 9")
                bonds.append(
                    BondRecord(
                        bond_id=self._decode_number(fields[0], "bond.id"),
                        begin_atom=self._decode_number(fields[1], "begin_atom"),
                        end_atom=self._decode_number(fields[2], "end_atom"),
                        bond_type=self._strip_prefix(fields[3], "bt:", "bond_type"),
                        display_type=self._strip_prefix(fields[4], "dt:", "display_type"),
                        stereo=self._strip_prefix(fields[5], "st:", "stereo"),
                        stereo_atom_0=self._decode_number(fields[6], "stereo_atom_0"),
                        stereo_atom_1=self._decode_number(fields[7], "stereo_atom_1"),
                        direction=self._strip_prefix(fields[8], "dir:", "direction"),
                    )
                )
            else:
                raise ChemMolSequenceError(f"Unexpected record marker: {marker}")

        parsed = ParsedChemMolSequence(tuple(atoms), tuple(bonds))
        lines = [START_TOKEN]
        for atom in parsed.atoms:
            lines.append(
                f"<a {atom.atom_id} {atom.element} {atom.formal_charge} {atom.isotope} "
                f"{int(atom.aromatic)} {atom.explicit_h_count} {int(atom.no_implicit)} "
                f"{atom.radical_electrons} {atom.chiral_tag} "
                f"{atom.quantized_x} {atom.quantized_y}>"
            )
        for bond in parsed.bonds:
            lines.append(
                f"<b {bond.bond_id} {bond.begin_atom} {bond.end_atom} "
                f"{bond.bond_type} {bond.display_type} {bond.stereo} "
                f"{bond.stereo_atom_0} {bond.stereo_atom_1} {bond.direction}>"
            )
        lines.append(END_TOKEN)
        sequence = "\n".join(lines) + "\n"
        parse_chemmolseq(sequence)
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
        if pad_to_multiple_of:
            remainder = target_length % pad_to_multiple_of
            if remainder:
                target_length += pad_to_multiple_of - remainder
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
            "sequence_schema_version": "chemmolseq-0.2",
            "integer_min": self.INTEGER_MIN,
            "integer_max": self.INTEGER_MAX,
            "vocabulary": self.vocabulary,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChemMolTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported tokenizer schema: {payload.get('schema_version')}")
        return cls(vocabulary=payload["vocabulary"])
