"""Grammar-constrained generation for the ChemTraverse v0.1 protocol."""

from __future__ import annotations

import codecs
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rdkit import Chem, rdBase

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import (
    mol_from_chemtraverse_v0_1,
    parse_chemtraverse_v0_1,
    sdf_from_chemtraverse_v0_1,
)

if TYPE_CHECKING:
    from chemvector.model import ChemVectorModel


_NO_REFERENCE_STEREO = frozenset({"STEREONONE", "STEREOANY"})


def _validate_expected_counts(
    expected_atom_count: int | None,
    expected_bond_count: int | None,
) -> None:
    if (expected_atom_count is None) != (expected_bond_count is None):
        raise ValueError(
            "expected_atom_count and expected_bond_count must be provided together"
        )
    if expected_atom_count is None:
        return
    if isinstance(expected_atom_count, bool) or not isinstance(expected_atom_count, int):
        raise ValueError("expected_atom_count must be an integer")
    if isinstance(expected_bond_count, bool) or not isinstance(expected_bond_count, int):
        raise ValueError("expected_bond_count must be an integer")
    if expected_atom_count < 1:
        raise ValueError("expected_atom_count must be positive")
    if expected_bond_count < expected_atom_count - 1:
        raise ValueError(
            "Single-component expected_bond_count must be at least atom_count - 1"
        )
    maximum_bonds = expected_atom_count * (expected_atom_count - 1) // 2
    if expected_bond_count > maximum_bonds:
        raise ValueError(
            "expected_bond_count exceeds the maximum for a simple graph"
        )


@dataclass(frozen=True)
class _BondState:
    begin_atom: int
    end_atom: int
    stereo: str


@dataclass
class ChemTraverseGrammarV01:
    """Incrementally enforce syntax and references, but not chemical valence."""

    tokenizer: ChemTraverseTokenizerV01
    expected_atom_count: int | None = field(default=None, kw_only=True)
    expected_bond_count: int | None = field(default=None, kw_only=True)
    phase: str = "bos"
    record_type: str | None = None
    current_fields: list[str] = field(default_factory=list)
    meta_complete: bool = False
    component_open: bool = False
    component_count: int = 0
    postamble_started: bool = False
    current_atom: int | None = None
    parent_stack: list[int] = field(default_factory=list)
    pending_down: tuple[int, tuple[str, ...]] | None = None
    component_atoms: set[int] = field(default_factory=set)
    atom_roles: list[str] = field(default_factory=list)
    atom_coordinates: list[tuple[int, int]] = field(default_factory=list)
    used_coordinates: set[tuple[int, int]] = field(default_factory=set)
    coordinate_count_by_x: dict[int, int] = field(default_factory=dict)
    bonds: list[_BondState] = field(default_factory=list)
    existing_bonds: set[tuple[int, int]] = field(default_factory=set)
    adjacency: list[set[int]] = field(default_factory=list)
    emitted_text_references: set[int] = field(default_factory=set)
    emitted_stereo_references: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        _validate_expected_counts(
            self.expected_atom_count,
            self.expected_bond_count,
        )
        lookup = self.tokenizer.token_to_id
        self._numbers = {
            int(token[2:]): token_id
            for token, token_id in lookup.items()
            if token.startswith("n:")
        }
        self._all_numbers = self._number_ids()
        self._positive = self._number_ids(minimum=1)
        self._nonnegative = self._number_ids(minimum=0)
        self._coordinates = self._number_ids(minimum=0, maximum=1023)
        self._roles = self._ids("role:IMPLICIT", "role:TEXT")
        self._elements = self._prefix_ids("el:")
        self._chiral = self._ids("chi:-", "chi:TCW", "chi:TCCW")
        self._orientations = self._ids("ori:F", "ori:R")
        self._display_types = self._prefix_ids("vis:")
        self._bond_types = self._prefix_ids("bt:")
        self._directions = self._prefix_ids("dir:")
        self._stereo = self._prefix_ids("st:")
        self._bytes = tuple(
            (value, lookup[f"byte:{value:02x}"])
            for value in range(256)
            if f"byte:{value:02x}" in lookup
        )
        self._flags = tuple(
            self._numbers[value] for value in (0, 1) if value in self._numbers
        )

    def _id(self, token: str) -> int:
        return self.tokenizer.token_to_id[token]

    def _ids(self, *tokens: str) -> tuple[int, ...]:
        return tuple(self._id(token) for token in tokens)

    def _prefix_ids(self, prefix: str) -> tuple[int, ...]:
        return tuple(
            token_id
            for token, token_id in self.tokenizer.token_to_id.items()
            if token.startswith(prefix)
        )

    def _number_ids(
        self,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> tuple[int, ...]:
        return tuple(
            token_id
            for value, token_id in sorted(self._numbers.items())
            if (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum)
        )

    def _reference_ids(self, references: set[int] | list[int]) -> tuple[int, ...]:
        return tuple(
            self._numbers[reference]
            for reference in sorted(references)
            if reference in self._numbers
        )

    @staticmethod
    def _number(token: str) -> int:
        return int(token.removeprefix("n:"))

    @staticmethod
    def _name(token: str) -> str:
        return token.split(":", 1)[1]

    def _has_unused_coordinate(self) -> bool:
        return len(self.used_coordinates) < 1024 * 1024

    def _count_constraints_enabled(self) -> bool:
        return self.expected_atom_count is not None

    def _can_add_atom(self) -> bool:
        return (
            self.expected_atom_count is None
            or len(self.atom_roles) < self.expected_atom_count
        )

    def _can_add_bond(self) -> bool:
        return (
            self.expected_bond_count is None
            or len(self.bonds) < self.expected_bond_count
        )

    def _expected_counts_reached(self) -> bool:
        return (
            not self._count_constraints_enabled()
            or (
                len(self.atom_roles) == self.expected_atom_count
                and len(self.bonds) == self.expected_bond_count
            )
        )

    def _atom_x_ids(self) -> tuple[int, ...]:
        return tuple(
            self._numbers[x]
            for x in range(1024)
            if self.coordinate_count_by_x.get(x, 0) < 1024 and x in self._numbers
        )

    def _atom_y_ids(self, x: int) -> tuple[int, ...]:
        return tuple(
            self._numbers[y]
            for y in range(1024)
            if y in self._numbers and (x, y) not in self.used_coordinates
        )

    def _link_references(self) -> set[int]:
        if self.current_atom is None or len(self.atom_roles) < 2:
            return set()
        latest_allowed = len(self.atom_roles) - 2
        return {
            reference
            for reference in self.component_atoms
            if reference <= latest_allowed
            and reference != self.current_atom
            and tuple(sorted((self.current_atom, reference))) not in self.existing_bonds
        }

    def _missing_text_references(self) -> set[int]:
        return {
            atom_id
            for atom_id, role in enumerate(self.atom_roles)
            if role == "TEXT" and atom_id not in self.emitted_text_references
        }

    def _missing_stereo_references(self) -> set[int]:
        return {
            bond_id
            for bond_id, bond in enumerate(self.bonds)
            if bond.stereo not in _NO_REFERENCE_STEREO
            and bond_id not in self.emitted_stereo_references
        }

    def _stereo_bond_ids(self) -> tuple[int, ...]:
        result = []
        for bond_id in sorted(self._missing_stereo_references()):
            if bond_id not in self._numbers:
                continue
            bond = self.bonds[bond_id]
            if self._reference_ids(self.adjacency[bond.begin_atom]) and self._reference_ids(
                self.adjacency[bond.end_atom]
            ):
                result.append(self._numbers[bond_id])
        return tuple(result)

    def _allowed_record_ids(self) -> tuple[int, ...]:
        if not self.meta_complete:
            return (self._id(self.tokenizer.META),)

        if self.component_open:
            if self.current_atom is None or self.pending_down is not None:
                if not self._can_add_atom():
                    return ()
                return (self._id(self.tokenizer.ATOM),)
            allowed: list[int] = []
            if (
                self._has_unused_coordinate()
                and self._can_add_atom()
                and self._can_add_bond()
            ):
                allowed.append(self._id(self.tokenizer.DOWN))
            if self._can_add_bond() and self._link_references():
                allowed.append(self._id(self.tokenizer.LINK))
            if self.parent_stack:
                allowed.append(self._id(self.tokenizer.UP))
            elif self._expected_counts_reached():
                allowed.append(self._id(self.tokenizer.COMPONENT_END))
            return tuple(allowed)

        missing_text = self._missing_text_references()
        missing_stereo = self._missing_stereo_references()
        allowed = []
        component_allowed = (
            not self._count_constraints_enabled() or self.component_count == 0
        )
        if (
            not self.postamble_started
            and self._has_unused_coordinate()
            and self._can_add_atom()
            and component_allowed
        ):
            allowed.append(self._id(self.tokenizer.COMPONENT))
        if self._reference_ids(missing_text):
            allowed.append(self._id(self.tokenizer.TEXT))
        if self._stereo_bond_ids():
            allowed.append(self._id(self.tokenizer.STEREO))
        if (
            self.component_count
            and not missing_text
            and not missing_stereo
            and self._expected_counts_reached()
        ):
            allowed.append(self._id(self.tokenizer.MOL_END))
        return tuple(allowed)

    @staticmethod
    def _valid_utf8_prefix(payload: bytes) -> bool:
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            decoder.decode(payload, final=False)
        except UnicodeDecodeError:
            return False
        return True

    @staticmethod
    def _complete_utf8(payload: bytes) -> bool:
        if not payload:
            return False
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    def _allowed_utf8_ids(self, marker_index: int) -> tuple[int, ...]:
        fields = self.current_fields
        if len(fields) == marker_index:
            return (self._id(self.tokenizer.UTF8),)
        if fields[marker_index] != self.tokenizer.UTF8:
            return ()
        payload_tokens = fields[marker_index + 1 :]
        if payload_tokens and payload_tokens[-1] == self.tokenizer.UTF8_END:
            return (self._id(self.tokenizer.END_RECORD),)
        payload = bytes(int(token.removeprefix("byte:"), 16) for token in payload_tokens)
        allowed = [
            token_id
            for value, token_id in self._bytes
            if self._valid_utf8_prefix(payload + bytes((value,)))
        ]
        if self._complete_utf8(payload):
            allowed.append(self._id(self.tokenizer.UTF8_END))
        return tuple(allowed)

    def _allowed_atom_fields(self) -> tuple[int, ...]:
        index = len(self.current_fields)
        if index == 0:
            return self._atom_x_ids()
        if index == 1:
            return self._atom_y_ids(self._number(self.current_fields[0]))
        fields = (
            self._roles,
            self._elements,
            self._all_numbers,
            self._nonnegative,
            self._flags,
            self._nonnegative,
            self._flags,
            self._nonnegative,
            self._chiral,
            (self._id(self.tokenizer.END_RECORD),),
        )
        return fields[index - 2]

    def _allowed_link_fields(self) -> tuple[int, ...]:
        index = len(self.current_fields)
        fields = (
            self._reference_ids(self._link_references()),
            self._orientations,
            self._display_types,
            self._bond_types,
            self._directions,
            self._stereo,
            (self._id(self.tokenizer.END_RECORD),),
        )
        return fields[index]

    def _allowed_text_fields(self) -> tuple[int, ...]:
        index = len(self.current_fields)
        if index == 0:
            return self._reference_ids(self._missing_text_references())
        if index in {1, 2}:
            return self._coordinates
        if index == 3:
            return self._ids(self.tokenizer.BBOX_NONE, self.tokenizer.BBOX)

        bbox_marker = self.current_fields[3]
        if bbox_marker == self.tokenizer.BBOX_NONE:
            return self._allowed_utf8_ids(4)
        if index in {4, 5}:
            return self._coordinates
        if index == 6:
            return self._number_ids(
                minimum=self._number(self.current_fields[4]), maximum=1023
            )
        if index == 7:
            return self._number_ids(
                minimum=self._number(self.current_fields[5]), maximum=1023
            )
        return self._allowed_utf8_ids(8)

    def _allowed_stereo_fields(self) -> tuple[int, ...]:
        index = len(self.current_fields)
        if index == 0:
            return self._stereo_bond_ids()
        bond = self.bonds[self._number(self.current_fields[0])]
        if index == 1:
            return self._reference_ids(self.adjacency[bond.begin_atom])
        if index == 2:
            return self._reference_ids(self.adjacency[bond.end_atom])
        return (self._id(self.tokenizer.END_RECORD),)

    def _allowed_field_ids(self) -> tuple[int, ...]:
        record_type = self.record_type
        index = len(self.current_fields)
        if record_type == "meta":
            if index in {0, 1}:
                return self._positive
            return self._allowed_utf8_ids(2)
        if record_type == "atom":
            return self._allowed_atom_fields()
        if record_type == "down":
            fields = (
                self._orientations,
                self._display_types,
                self._bond_types,
                self._directions,
                self._stereo,
                (self._id(self.tokenizer.END_RECORD),),
            )
            return fields[index]
        if record_type == "link":
            return self._allowed_link_fields()
        if record_type == "text":
            return self._allowed_text_fields()
        if record_type == "stereo":
            return self._allowed_stereo_fields()
        if record_type in {"component", "up", "component_end"}:
            return (self._id(self.tokenizer.END_RECORD),)
        raise ChemMolSequenceError(f"Unknown ChemTraverse record type: {record_type}")

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.phase == "bos":
            return (self.tokenizer.bos_token_id,)
        if self.phase == "mol":
            return (self._id(self.tokenizer.MOL),)
        if self.phase == "record":
            return self._allowed_record_ids()
        if self.phase == "fields":
            return self._allowed_field_ids()
        if self.phase == "eos":
            return (self.tokenizer.eos_token_id,)
        if self.phase == "done":
            return ()
        raise ChemMolSequenceError(f"Unknown ChemTraverse grammar phase: {self.phase}")

    def _add_bond(self, left: int, right: int, payload: tuple[str, ...]) -> None:
        if not self._can_add_bond():
            raise ChemMolSequenceError(
                "ChemTraverse generation exceeded expected_bond_count"
            )
        pair = tuple(sorted((left, right)))
        if pair in self.existing_bonds:
            raise ChemMolSequenceError("ChemTraverse generation created a duplicate bond")
        orientation, _display, _chemical, _direction, stereo = payload
        begin, end = (left, right) if self._name(orientation) == "F" else (right, left)
        self.bonds.append(
            _BondState(begin_atom=begin, end_atom=end, stereo=self._name(stereo))
        )
        self.existing_bonds.add(pair)
        self.adjacency[left].add(right)
        self.adjacency[right].add(left)

    def _finish_atom(self) -> None:
        if not self._can_add_atom():
            raise ChemMolSequenceError(
                "ChemTraverse generation exceeded expected_atom_count"
            )
        if self.pending_down is not None and not self._can_add_bond():
            raise ChemMolSequenceError(
                "ChemTraverse generation exceeded expected_bond_count"
            )
        fields = self.current_fields
        coordinate = (self._number(fields[0]), self._number(fields[1]))
        atom_id = len(self.atom_roles)
        self.atom_coordinates.append(coordinate)
        self.atom_roles.append(self._name(fields[2]))
        self.used_coordinates.add(coordinate)
        self.coordinate_count_by_x[coordinate[0]] = (
            self.coordinate_count_by_x.get(coordinate[0], 0) + 1
        )
        self.adjacency.append(set())
        self.component_atoms.add(atom_id)
        if self.pending_down is None:
            self.current_atom = atom_id
            return
        parent, payload = self.pending_down
        self._add_bond(parent, atom_id, payload)
        self.parent_stack.append(parent)
        self.current_atom = atom_id
        self.pending_down = None

    def _finish_record(self) -> None:
        record_type = self.record_type
        fields = self.current_fields
        if record_type == "meta":
            self.meta_complete = True
        elif record_type == "component":
            if self._count_constraints_enabled() and self.component_count:
                raise ChemMolSequenceError(
                    "Count-constrained ChemTraverse supports one component"
                )
            self.component_open = True
            self.component_count += 1
            self.current_atom = None
            self.component_atoms = set()
        elif record_type == "atom":
            self._finish_atom()
        elif record_type == "down":
            if self.current_atom is None:
                raise ChemMolSequenceError("Down record has no current atom")
            self.pending_down = (self.current_atom, tuple(fields))
        elif record_type == "link":
            if self.current_atom is None:
                raise ChemMolSequenceError("Link record has no current atom")
            self._add_bond(
                self.current_atom,
                self._number(fields[0]),
                tuple(fields[1:]),
            )
        elif record_type == "up":
            self.current_atom = self.parent_stack.pop()
        elif record_type == "component_end":
            if not self._expected_counts_reached():
                raise ChemMolSequenceError(
                    "ChemTraverse component ended before expected counts were reached"
                )
            self.component_open = False
            self.current_atom = None
            self.component_atoms = set()
        elif record_type == "text":
            self.emitted_text_references.add(self._number(fields[0]))
            self.postamble_started = True
        elif record_type == "stereo":
            self.emitted_stereo_references.add(self._number(fields[0]))
            self.postamble_started = True
        else:
            raise ChemMolSequenceError(f"Unknown ChemTraverse record type: {record_type}")
        self.record_type = None
        self.current_fields = []
        self.phase = "record"

    def consume(self, token_id: int) -> None:
        allowed = self.allowed_token_ids()
        if token_id not in allowed:
            token = (
                self.tokenizer.id_to_token[token_id]
                if 0 <= token_id < len(self.tokenizer.id_to_token)
                else str(token_id)
            )
            raise ChemMolSequenceError(
                f"Token {token!r} is invalid in ChemTraverse phase {self.phase} "
                f"record {self.record_type!r} field {len(self.current_fields)}"
            )
        token = self.tokenizer.id_to_token[token_id]
        if self.phase == "bos":
            self.phase = "mol"
            return
        if self.phase == "mol":
            self.phase = "record"
            return
        if self.phase == "eos":
            self.phase = "done"
            return
        if self.phase == "record":
            if token == self.tokenizer.MOL_END:
                self.phase = "eos"
                return
            marker_to_record = {
                self.tokenizer.META: "meta",
                self.tokenizer.COMPONENT: "component",
                self.tokenizer.ATOM: "atom",
                self.tokenizer.DOWN: "down",
                self.tokenizer.UP: "up",
                self.tokenizer.LINK: "link",
                self.tokenizer.COMPONENT_END: "component_end",
                self.tokenizer.TEXT: "text",
                self.tokenizer.STEREO: "stereo",
            }
            self.record_type = marker_to_record[token]
            self.current_fields = []
            self.phase = "fields"
            return
        if token == self.tokenizer.END_RECORD:
            self._finish_record()
            return
        self.current_fields.append(token)


@dataclass(frozen=True)
class GenerationResultChemTraverseV01:
    input_ids: tuple[int, ...]
    sequence: str | None
    completed: bool
    parseable: bool
    chemically_valid: bool
    error: str | None = None


def _completed_result_chemtraverse_v0_1(
    input_ids: tuple[int, ...],
    tokenizer: ChemTraverseTokenizerV01,
) -> GenerationResultChemTraverseV01:
    try:
        sequence = tokenizer.decode(input_ids)
        parse_chemtraverse_v0_1(sequence)
    except (ChemMolSequenceError, RuntimeError, ValueError) as error:
        return GenerationResultChemTraverseV01(
            input_ids=input_ids,
            sequence=None,
            completed=True,
            parseable=False,
            chemically_valid=False,
            error=str(error),
        )
    try:
        with rdBase.BlockLogs():
            mol = mol_from_chemtraverse_v0_1(sequence)
            Chem.SanitizeMol(Chem.Mol(mol))
    except (ChemMolSequenceError, RuntimeError, ValueError) as error:
        return GenerationResultChemTraverseV01(
            input_ids=input_ids,
            sequence=sequence,
            completed=True,
            parseable=True,
            chemically_valid=False,
            error=str(error),
        )
    return GenerationResultChemTraverseV01(
        input_ids=input_ids,
        sequence=sequence,
        completed=True,
        parseable=True,
        chemically_valid=True,
    )


@dataclass
class _BeamStateChemTraverseV01:
    input_ids: tuple[int, ...]
    grammar: ChemTraverseGrammarV01
    caches: list[torch.Tensor] | None
    log_probability: float


def _beam_rank(state: _BeamStateChemTraverseV01) -> float:
    decision_length = max(len(state.input_ids) - 1, 1)
    return state.log_probability / (decision_length**0.6)


def _generate_beam_chemtraverse_v0_1(
    model: "ChemVectorModel",
    memory: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
    *,
    max_length: int,
    beam_width: int,
    expected_atom_count: int | None,
    expected_bond_count: int | None,
) -> GenerationResultChemTraverseV01:
    grammar = ChemTraverseGrammarV01(
        tokenizer,
        expected_atom_count=expected_atom_count,
        expected_bond_count=expected_bond_count,
    )
    grammar.consume(tokenizer.bos_token_id)
    active = [
        _BeamStateChemTraverseV01(
            input_ids=(tokenizer.bos_token_id,),
            grammar=grammar,
            caches=None,
            log_probability=0.0,
        )
    ]
    completed: list[_BeamStateChemTraverseV01] = []
    best_partial = active[0]
    while active and len(active[0].input_ids) < max_length:
        tokens = torch.tensor(
            [[state.input_ids[-1]] for state in active], device=memory.device
        )
        batched_caches = None
        if active[0].caches is not None:
            batched_caches = [
                torch.cat(
                    [state.caches[layer_index] for state in active], dim=0
                )
                for layer_index in range(len(active[0].caches))
            ]
        logits, next_batched_caches = model.decode_step(
            tokens,
            len(active[0].input_ids) - 1,
            memory.expand(len(active), -1, -1),
            batched_caches,
        )
        candidates: list[_BeamStateChemTraverseV01] = []
        for state_index, state in enumerate(active):
            allowed = state.grammar.allowed_token_ids()
            if not allowed:
                continue
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            allowed_logits = logits[state_index, 0].float().index_select(0, allowed_tensor)
            log_probabilities = torch.log_softmax(allowed_logits, dim=0)
            count = min(beam_width, len(allowed))
            top_scores, top_indices = log_probabilities.topk(count)
            next_caches = [
                cache[state_index : state_index + 1]
                for cache in next_batched_caches
            ]
            for score, index in zip(top_scores.tolist(), top_indices.tolist()):
                next_id = int(allowed_tensor[index])
                next_grammar = copy.deepcopy(state.grammar)
                next_grammar.consume(next_id)
                child = _BeamStateChemTraverseV01(
                    input_ids=(*state.input_ids, next_id),
                    grammar=next_grammar,
                    caches=next_caches,
                    log_probability=state.log_probability + score,
                )
                if next_grammar.phase == "done":
                    completed.append(child)
                else:
                    candidates.append(child)
                if len(child.input_ids) > len(best_partial.input_ids) or (
                    len(child.input_ids) == len(best_partial.input_ids)
                    and _beam_rank(child) > _beam_rank(best_partial)
                ):
                    best_partial = child
        if completed:
            best_completed = max(completed, key=_beam_rank)
            best_active = max(candidates, key=_beam_rank, default=None)
            if best_active is None or _beam_rank(best_completed) >= _beam_rank(best_active):
                return _completed_result_chemtraverse_v0_1(
                    best_completed.input_ids, tokenizer
                )
        active = sorted(candidates, key=_beam_rank, reverse=True)[:beam_width]
    if completed:
        best_completed = max(completed, key=_beam_rank)
        return _completed_result_chemtraverse_v0_1(best_completed.input_ids, tokenizer)
    error = (
        "Beam generation exhausted all legal continuations"
        if not active
        else f"Beam generation did not finish within {max_length} tokens"
    )
    return GenerationResultChemTraverseV01(
        input_ids=best_partial.input_ids,
        sequence=None,
        completed=False,
        parseable=False,
        chemically_valid=False,
        error=error,
    )


@torch.inference_mode()
def generate_chemtraverse_v0_1(
    model: "ChemVectorModel",
    pixels: torch.Tensor,
    tokenizer: ChemTraverseTokenizerV01,
    max_length: int | None = None,
    *,
    beam_width: int = 1,
    expected_atom_count: int | None = None,
    expected_bond_count: int | None = None,
) -> GenerationResultChemTraverseV01:
    """Generate one ChemTraverse sequence while masking every illegal token."""

    _validate_expected_counts(expected_atom_count, expected_bond_count)
    if pixels.size(0) != 1:
        raise ValueError("Generation currently supports batch size 1")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    model.eval()
    max_length = max_length if max_length is not None else model.config.max_target_length
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    with torch.autocast(
        device_type=pixels.device.type,
        dtype=torch.bfloat16,
        enabled=pixels.device.type == "cuda",
    ):
        memory = model.encode_images(pixels)
        if beam_width > 1:
            return _generate_beam_chemtraverse_v0_1(
                model,
                memory,
                tokenizer,
                max_length=max_length,
                beam_width=beam_width,
                expected_atom_count=expected_atom_count,
                expected_bond_count=expected_bond_count,
            )
        grammar = ChemTraverseGrammarV01(
            tokenizer,
            expected_atom_count=expected_atom_count,
            expected_bond_count=expected_bond_count,
        )
        generated = [tokenizer.bos_token_id]
        grammar.consume(generated[0])
        caches = None
        exhausted = False
        while len(generated) < max_length:
            token = torch.tensor([[generated[-1]]], device=pixels.device)
            logits, caches = model.decode_step(
                token, len(generated) - 1, memory, caches
            )
            allowed = grammar.allowed_token_ids()
            if not allowed:
                exhausted = True
                break
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            scores = logits[0, 0].index_select(0, allowed_tensor)
            next_id = int(allowed_tensor[scores.argmax()])
            generated.append(next_id)
            grammar.consume(next_id)
            if grammar.phase == "done":
                return _completed_result_chemtraverse_v0_1(
                    tuple(generated), tokenizer
                )
    error = (
        "Generation exhausted all legal continuations"
        if exhausted
        else f"Generation did not finish within {max_length} tokens"
    )
    return GenerationResultChemTraverseV01(
        input_ids=tuple(generated),
        sequence=None,
        completed=False,
        parseable=False,
        chemically_valid=False,
        error=error,
    )


def write_sdf(sequence: str, path: str | Path) -> None:
    """Compile one ChemTraverse sequence and atomically write its SDF."""

    sdf = sdf_from_chemtraverse_v0_1(sequence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(sdf, encoding="utf-8")
    temporary_path.replace(path)


__all__ = [
    "ChemTraverseGrammarV01",
    "GenerationResultChemTraverseV01",
    "generate_chemtraverse_v0_1",
    "write_sdf",
]
