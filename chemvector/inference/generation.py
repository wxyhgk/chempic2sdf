"""Grammar-constrained ChemMolSeq generation and SDF reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from rdkit import Chem

from Data.sdf.chemmol_sequence import ChemMolSequenceError, mol_from_chemmolseq
from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from chemvector.model import ChemVectorModel


@dataclass
class ChemMolGrammar:
    """Incremental grammar state for one ChemMolSeq token stream."""

    tokenizer: ChemMolTokenizer
    phase: str = "bos"
    record_type: str | None = None
    field_index: int = 0
    atom_count: int = 0
    bond_count: int = 0
    bond_started: bool = False
    current_begin_atom: int | None = None
    current_end_atom: int | None = None
    current_stereo_atom_0: int | None = None
    existing_bonds: set[tuple[int, int]] = field(default_factory=set)
    atoms: list[dict] = field(default_factory=list)
    current_atom: dict = field(default_factory=dict)
    used_valence: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        lookup = self.tokenizer.token_to_id
        self._numbers = {
            int(token[2:]): token_id
            for token, token_id in lookup.items()
            if token.startswith("n:")
        }
        self._all_numbers = tuple(self._numbers.values())
        self._nonnegative = tuple(
            token_id for value, token_id in self._numbers.items() if value >= 0
        )
        self._coordinates = tuple(
            token_id for value, token_id in self._numbers.items() if 0 <= value <= 1023
        )
        self._elements = self._prefix_ids("el:")
        self._chiral = self._prefix_ids("chi:")
        supported_bonds = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
        self._bond_types = tuple(self._id(f"bt:{name}") for name in supported_bonds)
        self._display_types = tuple(self._id(f"dt:{name}") for name in supported_bonds)
        self._stereo = self._prefix_ids("st:")
        self._directions = self._prefix_ids("dir:")

    def _prefix_ids(self, prefix: str) -> tuple[int, ...]:
        return tuple(
            token_id
            for token, token_id in self.tokenizer.token_to_id.items()
            if token.startswith(prefix)
        )

    def _id(self, token: str) -> int:
        return self.tokenizer.token_to_id[token]

    @staticmethod
    def _maximum_valence(atom: dict) -> float:
        element = atom["element"]
        charge = atom["formal_charge"]
        if element == "B":
            return 4.0 if charge < 0 else 3.0
        if element == "C":
            if atom["aromatic"]:
                return 5.0
            return 3.0 if charge < 0 else 4.0
        if element == "N":
            if atom["aromatic"] or charge > 0:
                return 4.0
            return 2.0 if charge < 0 else 3.0
        if element == "O":
            return 3.0 if charge > 0 else (1.0 if charge < 0 else 2.0)
        if element in {"F", "Cl", "Br", "I"}:
            return 1.0
        if element == "Si":
            return 4.0
        if element in {"P", "S", "Se"}:
            return 6.0
        return 4.0

    def _remaining_valence(self, atom_id: int) -> float:
        atom = self.atoms[atom_id]
        return (
            self._maximum_valence(atom)
            - atom["explicit_h_count"]
            - self.used_valence[atom_id]
        )

    def _has_available_bond(self) -> bool:
        for begin in range(self.atom_count):
            if self._remaining_valence(begin) < 1.0:
                continue
            for end in range(begin + 1, self.atom_count):
                if self._remaining_valence(end) < 1.0:
                    continue
                if (begin, end) not in self.existing_bonds:
                    return True
        return False

    def _endpoint_ids(self, *, exclude_begin: bool) -> tuple[int, ...]:
        result = []
        for atom_id in range(self.atom_count):
            if self._remaining_valence(atom_id) < 1.0:
                continue
            if exclude_begin and atom_id == self.current_begin_atom:
                continue
            if exclude_begin and self.current_begin_atom is not None:
                edge = tuple(sorted((self.current_begin_atom, atom_id)))
                if edge in self.existing_bonds:
                    continue
            result.append(self._numbers[atom_id])
        return tuple(result)

    def _begin_atom_ids(self) -> tuple[int, ...]:
        result = []
        for begin in range(self.atom_count):
            if self._remaining_valence(begin) < 1.0:
                continue
            for end in range(self.atom_count):
                if begin == end or self._remaining_valence(end) < 1.0:
                    continue
                if tuple(sorted((begin, end))) not in self.existing_bonds:
                    result.append(self._numbers[begin])
                    break
        return tuple(result)

    def _atom_reference_ids(self) -> tuple[int, ...]:
        return tuple(self._numbers[atom_id] for atom_id in range(self.atom_count))

    def _allowed_bond_types(self) -> tuple[int, ...]:
        assert self.current_begin_atom is not None and self.current_end_atom is not None
        begin = self.atoms[self.current_begin_atom]
        end = self.atoms[self.current_end_atom]
        capacity = min(
            self._remaining_valence(self.current_begin_atom),
            self._remaining_valence(self.current_end_atom),
        )
        allowed = []
        for name, order in (("SINGLE", 1.0), ("DOUBLE", 2.0), ("TRIPLE", 3.0)):
            if capacity >= order:
                allowed.append(self._id(f"bt:{name}"))
        if capacity >= 1.0 and begin["aromatic"] and end["aromatic"]:
            allowed.append(self._id("bt:AROMATIC"))
        return tuple(allowed)

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.phase == "bos":
            return (self.tokenizer.bos_token_id,)
        if self.phase == "mol":
            return (self._id(self.tokenizer.MOL),)
        if self.phase == "record":
            if self.bond_started:
                return (self._id(self.tokenizer.BOND), self._id(self.tokenizer.MOL_END))
            allowed = [self._id(self.tokenizer.ATOM)]
            if self.atom_count:
                allowed.append(self._id(self.tokenizer.MOL_END))
            if self.atom_count >= 2 and self._has_available_bond():
                allowed.append(self._id(self.tokenizer.BOND))
            return tuple(allowed)
        if self.phase == "mol_end":
            return (self.tokenizer.eos_token_id,)
        if self.phase == "done":
            return ()
        if self.phase != "fields" or self.record_type is None:
            raise ChemMolSequenceError(f"Unknown grammar phase: {self.phase}")

        if self.record_type == "atom":
            fields = (
                (self._numbers[self.atom_count],),
                self._elements,
                self._all_numbers,
                self._nonnegative,
                (self._numbers[0], self._numbers[1]),
                self._nonnegative,
                (self._numbers[0], self._numbers[1]),
                self._nonnegative,
                self._chiral,
                self._coordinates,
                self._coordinates,
                (self._id(self.tokenizer.END_RECORD),),
            )
            return fields[self.field_index]

        if self.field_index == 0:
            return (self._numbers[self.bond_count],)
        if self.field_index == 1:
            return self._begin_atom_ids()
        if self.field_index == 2:
            return self._endpoint_ids(exclude_begin=True)
        if self.field_index == 3:
            return self._allowed_bond_types()
        if self.field_index == 4:
            return self._display_types
        if self.field_index == 5:
            return self._stereo
        if self.field_index == 6:
            return (self._numbers[-1], *self._atom_reference_ids())
        if self.field_index == 7:
            if self.current_stereo_atom_0 == -1:
                return (self._numbers[-1],)
            return self._atom_reference_ids()
        if self.field_index == 8:
            return self._directions
        return (self._id(self.tokenizer.END_RECORD),)

    def consume(self, token_id: int) -> None:
        if token_id not in self.allowed_token_ids():
            token = self.tokenizer.id_to_token[token_id]
            raise ChemMolSequenceError(
                f"Token {token!r} is not valid in phase {self.phase} field {self.field_index}"
            )
        token = self.tokenizer.id_to_token[token_id]
        if self.phase == "bos":
            self.phase = "mol"
        elif self.phase == "mol":
            self.phase = "record"
        elif self.phase == "record":
            if token == self.tokenizer.MOL_END:
                self.phase = "mol_end"
            else:
                self.record_type = "atom" if token == self.tokenizer.ATOM else "bond"
                self.bond_started = self.bond_started or self.record_type == "bond"
                self.field_index = 0
                self.current_begin_atom = None
                self.current_end_atom = None
                self.current_stereo_atom_0 = None
                self.current_atom = {}
                self.phase = "fields"
        elif self.phase == "mol_end":
            self.phase = "done"
        elif self.phase == "fields":
            value = int(token[2:]) if token.startswith("n:") else None
            if self.record_type == "atom":
                if self.field_index == 1:
                    self.current_atom["element"] = token[3:]
                elif self.field_index == 2:
                    self.current_atom["formal_charge"] = value
                elif self.field_index == 4:
                    self.current_atom["aromatic"] = bool(value)
                elif self.field_index == 5:
                    self.current_atom["explicit_h_count"] = value
            else:
                if self.field_index == 1:
                    self.current_begin_atom = value
                elif self.field_index == 2:
                    assert self.current_begin_atom is not None and value is not None
                    self.current_end_atom = value
                    self.existing_bonds.add(tuple(sorted((self.current_begin_atom, value))))
                elif self.field_index == 3:
                    assert self.current_begin_atom is not None and self.current_end_atom is not None
                    order = {
                        "bt:SINGLE": 1.0,
                        "bt:DOUBLE": 2.0,
                        "bt:TRIPLE": 3.0,
                        "bt:AROMATIC": 1.0,
                    }[token]
                    self.used_valence[self.current_begin_atom] += order
                    self.used_valence[self.current_end_atom] += order
                elif self.field_index == 6:
                    self.current_stereo_atom_0 = value
            self.field_index += 1
            final_field = 12 if self.record_type == "atom" else 10
            if self.field_index == final_field:
                if self.record_type == "atom":
                    self.atoms.append(self.current_atom)
                    self.used_valence.append(0.0)
                    self.atom_count += 1
                else:
                    self.bond_count += 1
                self.record_type = None
                self.field_index = 0
                self.phase = "record"


@dataclass(frozen=True)
class GenerationResult:
    input_ids: tuple[int, ...]
    sequence: str | None
    completed: bool
    chemically_valid: bool
    error: str | None = None


@torch.inference_mode()
def generate_chemmolseq(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    tokenizer: ChemMolTokenizer,
    *,
    max_length: int | None = None,
) -> GenerationResult:
    """Greedily generate one syntactically constrained ChemMolSeq document."""
    if pixel_values.size(0) != 1:
        raise ValueError("Generation currently supports batch size 1")
    model.eval()
    max_length = max_length or model.config.max_target_length
    with torch.autocast(
        device_type=pixel_values.device.type,
        dtype=torch.bfloat16,
        enabled=pixel_values.device.type == "cuda",
    ):
        memory = model.encode_images(pixel_values)
        grammar = ChemMolGrammar(tokenizer)
        generated = [tokenizer.bos_token_id]
        grammar.consume(generated[0])
        caches = None

        while len(generated) < max_length:
            token = torch.tensor([[generated[-1]]], device=pixel_values.device)
            logits, caches = model.decode_step(token, len(generated) - 1, memory, caches)
            allowed = grammar.allowed_token_ids()
            if not allowed:
                break
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            scores = logits[0, 0].index_select(0, allowed_tensor)
            next_id = int(allowed_tensor[scores.argmax()])
            generated.append(next_id)
            grammar.consume(next_id)
            if grammar.phase == "done":
                try:
                    sequence = tokenizer.decode(generated)
                    mol_from_chemmolseq(sequence)
                    return GenerationResult(tuple(generated), sequence, True, True)
                except ChemMolSequenceError as error:
                    sequence = tokenizer.decode(generated)
                    return GenerationResult(
                        tuple(generated), sequence, True, False, str(error)
                    )

    return GenerationResult(
        tuple(generated),
        None,
        False,
        False,
        f"Generation did not finish within {max_length} tokens",
    )


def write_sdf(sequence: str, path: str | Path) -> None:
    """Reconstruct a generated sequence and write a single-record SDF."""
    mol = mol_from_chemmolseq(sequence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()
