"""Grammar-constrained generation for spatial ChemMolSeq v0.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from rdkit import Chem

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemmol_sequence_v3 import mol_from_chemmolseq_v3
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from chemvector.model import ChemVectorModel


@dataclass
class ChemMolGrammarV3:
    tokenizer: ChemMolTokenizerV3
    phase: str = "bos"
    record_type: str | None = None
    field_index: int = 0
    atom_count: int = 0
    bond_started: bool = False
    current_begin_atom: int | None = None
    current_stereo_atom_0: int | None = None
    existing_bonds: set[tuple[int, int]] = field(default_factory=set)
    coordinate_values: int = 0

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
        self._bond_types = tuple(
            self._id(f"bt:{name}")
            for name in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
        )
        self._stereo = self._prefix_ids("st:")

    def _prefix_ids(self, prefix: str) -> tuple[int, ...]:
        return tuple(
            token_id
            for token, token_id in self.tokenizer.token_to_id.items()
            if token.startswith(prefix)
        )

    def _id(self, token: str) -> int:
        return self.tokenizer.token_to_id[token]

    def _begin_ids(self) -> tuple[int, ...]:
        result = []
        for begin in range(self.atom_count):
            if any(
                end != begin
                and tuple(sorted((begin, end))) not in self.existing_bonds
                for end in range(self.atom_count)
            ):
                result.append(self._numbers[begin])
        return tuple(result)

    def _end_ids(self) -> tuple[int, ...]:
        if self.current_begin_atom is None:
            return ()
        return tuple(
            self._numbers[end]
            for end in range(self.atom_count)
            if end != self.current_begin_atom
            and tuple(sorted((self.current_begin_atom, end))) not in self.existing_bonds
        )

    def _atom_reference_ids(self) -> tuple[int, ...]:
        return tuple(self._numbers[index] for index in range(self.atom_count))

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.phase == "bos":
            return (self.tokenizer.bos_token_id,)
        if self.phase == "mol":
            return (self._id(self.tokenizer.MOL),)
        if self.phase == "record":
            if self.bond_started:
                allowed = [self._id(self.tokenizer.COORDS)]
                if self._begin_ids():
                    allowed.insert(0, self._id(self.tokenizer.BOND))
                return tuple(allowed)
            allowed = [self._id(self.tokenizer.ATOM)]
            if self.atom_count:
                allowed.append(self._id(self.tokenizer.COORDS))
            if self.atom_count >= 2:
                allowed.append(self._id(self.tokenizer.BOND))
            return tuple(allowed)
        if self.phase == "coords":
            if self.coordinate_values == self.atom_count * 2:
                return (self._id(self.tokenizer.COORDS_END),)
            return self._coordinates
        if self.phase == "molecule_end":
            return (self._id(self.tokenizer.MOL_END),)
        if self.phase == "eos":
            return (self.tokenizer.eos_token_id,)
        if self.phase == "done":
            return ()
        if self.phase != "fields" or self.record_type is None:
            raise ChemMolSequenceError(f"Unknown v0.3 grammar phase: {self.phase}")

        if self.record_type == "atom":
            fields = (
                self._elements,
                self._all_numbers,
                self._nonnegative,
                (self._numbers[0], self._numbers[1]),
                self._nonnegative,
                (self._numbers[0], self._numbers[1]),
                self._nonnegative,
                self._chiral,
                (self._id(self.tokenizer.END_RECORD),),
            )
            return fields[self.field_index]

        if self.field_index == 0:
            return self._begin_ids()
        if self.field_index == 1:
            return self._end_ids()
        if self.field_index == 2:
            return self._bond_types
        if self.field_index == 3:
            return self._stereo
        if self.field_index == 4:
            return (self._numbers[-1], *self._atom_reference_ids())
        if self.field_index == 5:
            if self.current_stereo_atom_0 == -1:
                return (self._numbers[-1],)
            return self._atom_reference_ids()
        return (self._id(self.tokenizer.END_RECORD),)

    def consume(self, token_id: int) -> None:
        if token_id not in self.allowed_token_ids():
            token = self.tokenizer.id_to_token[token_id]
            raise ChemMolSequenceError(
                f"Token {token!r} is invalid in v0.3 phase {self.phase} field {self.field_index}"
            )
        token = self.tokenizer.id_to_token[token_id]
        if self.phase == "bos":
            self.phase = "mol"
        elif self.phase == "mol":
            self.phase = "record"
        elif self.phase == "record":
            if token == self.tokenizer.COORDS:
                self.coordinate_values = 0
                self.phase = "coords"
            else:
                self.record_type = "atom" if token == self.tokenizer.ATOM else "bond"
                self.bond_started = self.bond_started or self.record_type == "bond"
                self.field_index = 0
                self.current_begin_atom = None
                self.current_stereo_atom_0 = None
                self.phase = "fields"
        elif self.phase == "coords":
            if token == self.tokenizer.COORDS_END:
                self.phase = "molecule_end"
            else:
                self.coordinate_values += 1
        elif self.phase == "molecule_end":
            self.phase = "eos"
        elif self.phase == "eos":
            self.phase = "done"
        elif self.phase == "fields":
            value = int(token[2:]) if token.startswith("n:") else None
            if self.record_type == "bond":
                if self.field_index == 0:
                    self.current_begin_atom = value
                elif self.field_index == 1:
                    assert self.current_begin_atom is not None and value is not None
                    self.existing_bonds.add(tuple(sorted((self.current_begin_atom, value))))
                elif self.field_index == 4:
                    self.current_stereo_atom_0 = value
            self.field_index += 1
            final_field = 9 if self.record_type == "atom" else 7
            if self.field_index == final_field:
                if self.record_type == "atom":
                    self.atom_count += 1
                self.record_type = None
                self.field_index = 0
                self.phase = "record"


@dataclass(frozen=True)
class GenerationResultV3:
    input_ids: tuple[int, ...]
    sequence: str | None
    completed: bool
    chemically_valid: bool
    error: str | None = None


@torch.inference_mode()
def generate_chemmolseq_v3(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    tokenizer: ChemMolTokenizerV3,
    *,
    max_length: int | None = None,
) -> GenerationResultV3:
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
        grammar = ChemMolGrammarV3(tokenizer)
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
                    mol_from_chemmolseq_v3(sequence)
                    return GenerationResultV3(tuple(generated), sequence, True, True)
                except ChemMolSequenceError as error:
                    sequence = tokenizer.decode(generated)
                    return GenerationResultV3(
                        tuple(generated), sequence, True, False, str(error)
                    )
    return GenerationResultV3(
        tuple(generated),
        None,
        False,
        False,
        f"Generation did not finish within {max_length} tokens",
    )


def write_sdf_v3(sequence: str, path: str | Path) -> None:
    chemical_mol = mol_from_chemmolseq_v3(sequence, apply_render_coordinates=False)
    mol = mol_from_chemmolseq_v3(sequence, apply_render_coordinates=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(temporary_path))
    try:
        writer.write(mol)
    finally:
        writer.close()
    supplier = Chem.SDMolSupplier(str(temporary_path), removeHs=False)
    written_mol = next((item for item in supplier if item is not None), None)
    expected_key = Chem.MolToInchiKey(chemical_mol)
    actual_key = Chem.MolToInchiKey(written_mol) if written_mol is not None else None
    if actual_key != expected_key:
        temporary_path.unlink(missing_ok=True)
        raise ChemMolSequenceError(
            f"SDF depiction stereo mismatch: expected {expected_key}, got {actual_key}"
        )
    temporary_path.replace(path)
