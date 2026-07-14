"""Grammar-constrained generation for coordinate-grounded ChemSVG v0.4."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import torch
from rdkit import Chem, rdBase

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemsvg_sequence_v4 import (
    StrokeRecordV4,
    VertexRecordV4,
    mol_from_chemsvg_v4,
)
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from chemvector.model import ChemVectorModel
from chemvector.ocr import OCRTextCodec, parse_atom_display_text
from chemvector.detection import detect_vertex_records
from chemvector.sdf_io import write_sdf_v4
from chemvector.stroke import all_pair_indices


Coordinate = tuple[int, int]


@dataclass
class ChemSVGGrammarV4:
    tokenizer: ChemSVGTokenizerV4
    visual_first: bool = False
    allowed_stroke_pairs: set[tuple[Coordinate, Coordinate]] | None = None
    phase: str = "bos"
    record_type: str | None = None
    field_index: int = 0
    vertices: list[VertexRecordV4] = field(default_factory=list)
    strokes: list[StrokeRecordV4] = field(default_factory=list)
    stroke_started: bool = False
    current_fields: list[str] = field(default_factory=list)
    current_vertex_x: int | None = None
    current_begin_x: int | None = None
    current_begin: Coordinate | None = None
    current_end_x: int | None = None
    current_end: Coordinate | None = None
    current_display_type: str | None = None
    current_stereo: str | None = None
    current_stereo_x: int | None = None
    existing_strokes: set[tuple[Coordinate, Coordinate]] = field(default_factory=set)
    _partial_mol_cache: Chem.Mol | None = field(default=None, init=False, repr=False)
    _open_coordinate_cache: tuple[Coordinate, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

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
        self._display_types = tuple(
            self._id(f"vis:{name}")
            for name in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
        )
        self._directions = self._prefix_ids("dir:")
        self._stereo = self._prefix_ids("st:")

    def _prefix_ids(self, prefix: str) -> tuple[int, ...]:
        return tuple(
            token_id
            for token, token_id in self.tokenizer.token_to_id.items()
            if token.startswith(prefix)
        )

    def _id(self, token: str) -> int:
        return self.tokenizer.token_to_id[token]

    @property
    def coordinate_to_atom(self) -> dict[Coordinate, int]:
        return {(vertex.x, vertex.y): index for index, vertex in enumerate(self.vertices)}

    def _invalidate_chemical_cache(self) -> None:
        self._partial_mol_cache = None
        self._open_coordinate_cache = None

    def _partial_mol(self) -> Chem.Mol:
        if self._partial_mol_cache is not None:
            return self._partial_mol_cache
        rw_mol = Chem.RWMol()
        coordinate_to_atom = self.coordinate_to_atom
        for record in self.vertices:
            atom = Chem.Atom(record.element)
            atom.SetFormalCharge(record.formal_charge)
            atom.SetIsotope(record.isotope)
            atom.SetIsAromatic(record.aromatic)
            atom.SetNumExplicitHs(record.explicit_h_count)
            atom.SetNoImplicit(record.no_implicit)
            atom.SetNumRadicalElectrons(record.radical_electrons)
            rw_mol.AddAtom(atom)
        for record in self.strokes:
            begin_atom = coordinate_to_atom[record.begin]
            end_atom = coordinate_to_atom[record.end]
            rw_mol.AddBond(begin_atom, end_atom, getattr(Chem.BondType, record.bond_type))
            if record.bond_type == "AROMATIC":
                rw_mol.GetBondBetweenAtoms(begin_atom, end_atom).SetIsAromatic(True)
        rw_mol.UpdatePropertyCache(strict=True)
        self._partial_mol_cache = rw_mol.GetMol()
        return self._partial_mol_cache

    def _coordinate_has_open_valence(self, coordinate: Coordinate) -> bool:
        rw_mol = Chem.RWMol(self._partial_mol())
        hydrogen_id = rw_mol.AddAtom(Chem.Atom("H"))
        rw_mol.AddBond(
            self.coordinate_to_atom[coordinate],
            hydrogen_id,
            Chem.BondType.SINGLE,
        )
        try:
            with rdBase.BlockLogs():
                rw_mol.UpdatePropertyCache(strict=True)
        except (RuntimeError, ValueError):
            return False
        return True

    def _open_coordinates(self) -> tuple[Coordinate, ...]:
        if self._open_coordinate_cache is None:
            self._open_coordinate_cache = tuple(
                coordinate
                for coordinate in self.coordinate_to_atom
                if self._coordinate_has_open_valence(coordinate)
            )
        return self._open_coordinate_cache

    def _pair(self, begin: Coordinate, end: Coordinate) -> tuple[Coordinate, Coordinate]:
        return tuple(sorted((begin, end)))

    def _valid_bond_type_ids(
        self,
        begin: Coordinate,
        end: Coordinate,
        display_type: str | None = None,
    ) -> tuple[int, ...]:
        allowed = []
        for name in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"):
            if display_type in {"TRIPLE", "AROMATIC"} and name != display_type:
                continue
            if display_type in {"SINGLE", "DOUBLE"} and name not in {
                display_type,
                "AROMATIC",
            }:
                continue
            if self.visual_first:
                allowed.append(self._id(f"bt:{name}"))
                continue
            coordinate_to_atom = self.coordinate_to_atom
            begin_atom = coordinate_to_atom[begin]
            end_atom = coordinate_to_atom[end]
            begin_aromatic = self.vertices[begin_atom].aromatic
            end_aromatic = self.vertices[end_atom].aromatic
            if name == "AROMATIC" and not (begin_aromatic and end_aromatic):
                continue
            rw_mol = Chem.RWMol(self._partial_mol())
            rw_mol.AddBond(begin_atom, end_atom, getattr(Chem.BondType, name))
            if name == "AROMATIC":
                rw_mol.GetBondBetweenAtoms(begin_atom, end_atom).SetIsAromatic(True)
            try:
                with rdBase.BlockLogs():
                    rw_mol.UpdatePropertyCache(strict=True)
            except (RuntimeError, ValueError):
                continue
            allowed.append(self._id(f"bt:{name}"))
        return tuple(allowed)

    def _can_finish_molecule(self) -> bool:
        if not self.vertices:
            return False
        if self.visual_first:
            return bool(self.strokes)
        try:
            mol = Chem.Mol(self._partial_mol())
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            return False
        return True

    def _vertex_x_ids(self) -> tuple[int, ...]:
        if not self.vertices or self.vertices[-1].y < 1023:
            return self._coordinates
        return tuple(
            self._numbers[x]
            for x in range(self.vertices[-1].x + 1, 1024)
        )

    def _vertex_y_ids(self) -> tuple[int, ...]:
        if self.current_vertex_x is None:
            return ()
        used = self.coordinate_to_atom
        last = (self.vertices[-1].y, self.vertices[-1].x) if self.vertices else None
        return tuple(
            self._numbers[y]
            for y in range(1024)
            if (self.current_vertex_x, y) not in used
            and (last is None or (y, self.current_vertex_x) > last)
        )

    def _begin_coordinates(self) -> tuple[Coordinate, ...]:
        open_coordinates = set(
            self.coordinate_to_atom if self.visual_first else self._open_coordinates()
        )
        return tuple(
            begin
            for begin in open_coordinates
            if any(
                end != begin
                and end in open_coordinates
                and self._pair(begin, end) not in self.existing_strokes
                and (
                    self.allowed_stroke_pairs is None
                    or self._pair(begin, end) in self.allowed_stroke_pairs
                )
                for end in self.coordinate_to_atom
            )
        )

    def _begin_x_ids(self) -> tuple[int, ...]:
        return tuple(self._numbers[x] for x in sorted({x for x, _ in self._begin_coordinates()}))

    def _begin_y_ids(self) -> tuple[int, ...]:
        if self.current_begin_x is None:
            return ()
        return tuple(
            self._numbers[y]
            for x, y in self._begin_coordinates()
            if x == self.current_begin_x
        )

    def _end_coordinates(self) -> tuple[Coordinate, ...]:
        if self.current_begin is None:
            return ()
        coordinates = (
            self.coordinate_to_atom if self.visual_first else self._open_coordinates()
        )
        return tuple(
            end
            for end in coordinates
            if end != self.current_begin
            and self._pair(self.current_begin, end) not in self.existing_strokes
            and (
                self.allowed_stroke_pairs is None
                or self._pair(self.current_begin, end) in self.allowed_stroke_pairs
            )
            and self._valid_bond_type_ids(self.current_begin, end)
        )

    def _end_x_ids(self) -> tuple[int, ...]:
        return tuple(self._numbers[x] for x in sorted({x for x, _ in self._end_coordinates()}))

    def _end_y_ids(self) -> tuple[int, ...]:
        if self.current_end_x is None:
            return ()
        return tuple(
            self._numbers[y]
            for x, y in self._end_coordinates()
            if x == self.current_end_x
        )

    def _coordinate_x_ids(self) -> tuple[int, ...]:
        return tuple(
            self._numbers[x] for x in sorted({x for x, _ in self.coordinate_to_atom})
        )

    def _coordinate_y_ids(self, x: int | None) -> tuple[int, ...]:
        if x is None:
            return ()
        return tuple(
            self._numbers[y]
            for coordinate_x, y in self.coordinate_to_atom
            if coordinate_x == x
        )

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.phase == "bos":
            return (self.tokenizer.bos_token_id,)
        if self.phase == "mol":
            return (self._id(self.tokenizer.MOL),)
        if self.phase == "record":
            allowed = [] if self.stroke_started else [self._id(self.tokenizer.VERTEX)]
            if len(self.vertices) >= 2 and self._begin_coordinates():
                allowed.append(self._id(self.tokenizer.STROKE))
            if self._can_finish_molecule():
                allowed.append(self._id(self.tokenizer.MOL_END))
            return tuple(allowed)
        if self.phase == "eos":
            return (self.tokenizer.eos_token_id,)
        if self.phase == "done":
            return ()
        if self.phase != "fields" or self.record_type is None:
            raise ChemMolSequenceError(f"Unknown v0.4 grammar phase: {self.phase}")

        if self.record_type == "vertex":
            fields = (
                self._vertex_x_ids(),
                self._vertex_y_ids(),
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
            return self._begin_x_ids()
        if self.field_index == 1:
            return self._begin_y_ids()
        if self.field_index == 2:
            return self._end_x_ids()
        if self.field_index == 3:
            return self._end_y_ids()
        if self.field_index == 4:
            return self._display_types
        if self.field_index == 5:
            if self.current_begin is None or self.current_end is None:
                return ()
            return self._valid_bond_type_ids(
                self.current_begin,
                self.current_end,
                self.current_display_type,
            )
        if self.field_index == 6:
            return self._directions
        if self.field_index == 7:
            return self._stereo
        if self.current_stereo in {"STEREONONE", "STEREOANY"}:
            return (self._id(self.tokenizer.END_RECORD),)
        if self.field_index in {8, 10}:
            return self._coordinate_x_ids()
        if self.field_index in {9, 11}:
            return self._coordinate_y_ids(self.current_stereo_x)
        return (self._id(self.tokenizer.END_RECORD),)

    def consume(self, token_id: int) -> None:
        if token_id not in self.allowed_token_ids():
            token = self.tokenizer.id_to_token[token_id]
            raise ChemMolSequenceError(
                f"Token {token!r} is invalid in v0.4 phase {self.phase} field {self.field_index}"
            )
        token = self.tokenizer.id_to_token[token_id]
        if self.phase == "bos":
            self.phase = "mol"
            return
        if self.phase == "mol":
            self.phase = "record"
            return
        if self.phase == "record":
            if token == self.tokenizer.MOL_END:
                self.phase = "eos"
                return
            self.record_type = "vertex" if token == self.tokenizer.VERTEX else "stroke"
            self.stroke_started = self.stroke_started or self.record_type == "stroke"
            self.field_index = 0
            self.current_fields = []
            self.current_vertex_x = None
            self.current_begin_x = None
            self.current_begin = None
            self.current_end_x = None
            self.current_end = None
            self.current_display_type = None
            self.current_stereo = None
            self.current_stereo_x = None
            self.phase = "fields"
            return
        if self.phase == "eos":
            self.phase = "done"
            return

        value = int(token[2:]) if token.startswith("n:") else None
        self.current_fields.append(token)
        if self.record_type == "vertex":
            if self.field_index == 0:
                self.current_vertex_x = value
            self.field_index += 1
            if self.field_index == 11:
                fields = self.current_fields
                self.vertices.append(
                    VertexRecordV4(
                        x=int(fields[0][2:]),
                        y=int(fields[1][2:]),
                        element=fields[2].removeprefix("el:"),
                        formal_charge=int(fields[3][2:]),
                        isotope=int(fields[4][2:]),
                        aromatic=bool(int(fields[5][2:])),
                        explicit_h_count=int(fields[6][2:]),
                        no_implicit=bool(int(fields[7][2:])),
                        radical_electrons=int(fields[8][2:]),
                        canonical_chiral_tag=fields[9].removeprefix("chi:"),
                    )
                )
                self._invalidate_chemical_cache()
                self.record_type = None
                self.phase = "record"
            return

        if self.field_index == 0:
            self.current_begin_x = value
        elif self.field_index == 1:
            self.current_begin = (self.current_begin_x, value)
        elif self.field_index == 2:
            self.current_end_x = value
        elif self.field_index == 3:
            self.current_end = (self.current_end_x, value)
            self.existing_strokes.add(self._pair(self.current_begin, self.current_end))
        elif self.field_index == 4:
            self.current_display_type = token.removeprefix("vis:")
        elif self.field_index == 7:
            self.current_stereo = token.removeprefix("st:")
        elif self.field_index in {8, 10}:
            self.current_stereo_x = value
        self.field_index += 1
        stereo_has_references = self.current_stereo not in {"STEREONONE", "STEREOANY"}
        final_field = 13 if stereo_has_references else 9
        if self.field_index == final_field:
            fields = self.current_fields
            stereo_values = (
                tuple(int(item[2:]) for item in fields[8:12])
                if stereo_has_references
                else (-1, -1, -1, -1)
            )
            self.strokes.append(
                StrokeRecordV4(
                    begin_x=int(fields[0][2:]),
                    begin_y=int(fields[1][2:]),
                    end_x=int(fields[2][2:]),
                    end_y=int(fields[3][2:]),
                    display_type=fields[4].removeprefix("vis:"),
                    bond_type=fields[5].removeprefix("bt:"),
                    direction=fields[6].removeprefix("dir:"),
                    stereo=fields[7].removeprefix("st:"),
                    stereo_atom_0_x=stereo_values[0],
                    stereo_atom_0_y=stereo_values[1],
                    stereo_atom_1_x=stereo_values[2],
                    stereo_atom_1_y=stereo_values[3],
                )
            )
            self._invalidate_chemical_cache()
            self.record_type = None
            self.phase = "record"


@dataclass(frozen=True)
class GenerationResultV4:
    input_ids: tuple[int, ...]
    sequence: str | None
    completed: bool
    chemically_valid: bool
    error: str | None = None


def _recognize_vertex_labels(
    model: ChemVectorModel,
    ocr_features: torch.Tensor,
    vertices: list[VertexRecordV4],
    pixel_size: tuple[int, int],
    cache: dict[tuple[Coordinate, ...], tuple[str, ...]],
) -> tuple[str, ...]:
    coordinates = tuple((vertex.x, vertex.y) for vertex in vertices)
    if coordinates in cache:
        return cache[coordinates]
    if model.ocr_head is None:
        raise ValueError("OCR vertex fusion requires an OCR-enabled model")
    points = torch.tensor(
        [[[x / 1023.0, y / 1023.0] for x, y in coordinates]],
        device=ocr_features.device,
        dtype=torch.float32,
    )
    mask = torch.ones((1, len(vertices)), device=ocr_features.device, dtype=torch.bool)
    token_ids = model.ocr_head.generate(
        ocr_features,
        points,
        mask,
        pixel_size,
    )
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    labels = tuple(codec.decode(row.tolist()) for row in token_ids.cpu())
    cache[coordinates] = labels
    return labels


def _fuse_ocr_vertex_prefix(
    input_ids: tuple[int, ...],
    grammar: ChemSVGGrammarV4,
    labels: tuple[str, ...],
    tokenizer: ChemSVGTokenizerV4,
) -> tuple[int, ...]:
    if len(labels) != len(grammar.vertices):
        raise ValueError("OCR label count does not match generated vertices")
    fused_vertices = []
    for vertex, label in zip(grammar.vertices, labels):
        parsed = parse_atom_display_text(label)
        if parsed is None or parsed["element"] == "C":
            fused_vertices.append(vertex)
            continue
        explicit_h_count = parsed["explicit_h_count"]
        fused_vertices.append(
            replace(
                vertex,
                element=parsed["element"],
                formal_charge=parsed["formal_charge"],
                isotope=parsed["isotope"],
                explicit_h_count=explicit_h_count,
                no_implicit=True if explicit_h_count else vertex.no_implicit,
                radical_electrons=parsed["radical_electrons"],
            )
        )
    grammar.vertices = fused_vertices
    grammar._invalidate_chemical_cache()

    fused_ids = list(input_ids)
    vertex_index = 0
    for index, token_id in enumerate(fused_ids):
        if tokenizer.id_to_token[token_id] != tokenizer.VERTEX:
            continue
        vertex = fused_vertices[vertex_index]
        fused_ids[index + 3] = tokenizer.token_to_id[f"el:{vertex.element}"]
        fused_ids[index + 4] = tokenizer.token_to_id[f"n:{vertex.formal_charge}"]
        fused_ids[index + 5] = tokenizer.token_to_id[f"n:{vertex.isotope}"]
        fused_ids[index + 7] = tokenizer.token_to_id[f"n:{vertex.explicit_h_count}"]
        fused_ids[index + 8] = tokenizer.token_to_id[f"n:{int(vertex.no_implicit)}"]
        fused_ids[index + 9] = tokenizer.token_to_id[f"n:{vertex.radical_electrons}"]
        vertex_index += 1
    return tuple(fused_ids)


def _score_vertex_pairs(
    model: ChemVectorModel,
    stroke_features: torch.Tensor,
    vertices: list[VertexRecordV4],
    pixel_size: tuple[int, int],
    threshold: float,
    cache: dict[tuple[Coordinate, ...], set[tuple[Coordinate, Coordinate]]],
) -> set[tuple[Coordinate, Coordinate]]:
    coordinates = tuple((vertex.x, vertex.y) for vertex in vertices)
    if coordinates in cache:
        return cache[coordinates]
    if model.stroke_head is None:
        raise ValueError("Stroke scoring requires a stroke-enabled model")
    index_pairs = all_pair_indices(len(vertices))
    anchors = torch.tensor(
        [[[x / 1023.0, y / 1023.0] for x, y in coordinates]],
        device=stroke_features.device,
        dtype=torch.float32,
    )
    pair_indices = torch.tensor(
        [index_pairs],
        device=stroke_features.device,
        dtype=torch.long,
    )
    pair_mask = torch.ones(
        (1, len(index_pairs)),
        device=stroke_features.device,
        dtype=torch.bool,
    )
    logits = model.stroke_head(
        stroke_features,
        anchors,
        pair_indices,
        pair_mask,
        pixel_size,
    )
    edge_scores = 1.0 - logits.float().softmax(dim=-1)[:, 0]
    allowed = {
        tuple(sorted((coordinates[left], coordinates[right])))
        for (left, right), score in zip(index_pairs, edge_scores.tolist())
        if score >= threshold
    }
    cache[coordinates] = allowed
    return allowed


def _rebuild_prefix_caches(
    model: ChemVectorModel,
    input_ids: tuple[int, ...],
    memory: torch.Tensor,
) -> list[torch.Tensor]:
    caches = None
    for position, token_id in enumerate(input_ids):
        token = torch.tensor([[token_id]], device=memory.device)
        _, caches = model.decode_step(token, position, memory, caches)
    return caches


def _vertex_prefix_ids(
    tokenizer: ChemSVGTokenizerV4,
    vertices: list[VertexRecordV4],
) -> tuple[int, ...]:
    tokens = [tokenizer.BOS, tokenizer.MOL]
    for vertex in vertices:
        tokens.extend(
            (
                tokenizer.VERTEX,
                f"n:{vertex.x}",
                f"n:{vertex.y}",
                f"el:{vertex.element}",
                f"n:{vertex.formal_charge}",
                f"n:{vertex.isotope}",
                f"n:{int(vertex.aromatic)}",
                f"n:{vertex.explicit_h_count}",
                f"n:{int(vertex.no_implicit)}",
                f"n:{vertex.radical_electrons}",
                f"chi:{vertex.canonical_chiral_tag}",
                tokenizer.END_RECORD,
            )
        )
    return tuple(tokenizer.token_to_id[token] for token in tokens)


@dataclass
class _BeamStateV4:
    input_ids: tuple[int, ...]
    grammar: ChemSVGGrammarV4
    caches: list[torch.Tensor] | None
    log_probability: float


def _completed_result_v4(
    input_ids: tuple[int, ...],
    tokenizer: ChemSVGTokenizerV4,
) -> GenerationResultV4:
    sequence = None
    try:
        sequence = tokenizer.decode(input_ids)
        mol_from_chemsvg_v4(sequence)
    except ChemMolSequenceError as error:
        return GenerationResultV4(input_ids, sequence, True, False, str(error))
    return GenerationResultV4(input_ids, sequence, True, True)


def _beam_rank(state: _BeamStateV4) -> float:
    decision_length = max(len(state.input_ids) - 1, 1)
    return state.log_probability / (decision_length**0.6)


def _generate_beam_v4(
    model: ChemVectorModel,
    memory: torch.Tensor,
    tokenizer: ChemSVGTokenizerV4,
    *,
    max_length: int,
    beam_width: int,
    visual_first: bool = False,
    ocr_features: torch.Tensor | None = None,
    stroke_features: torch.Tensor | None = None,
    pixel_size: tuple[int, int] | None = None,
    stroke_threshold: float = 0.5,
) -> GenerationResultV4:
    grammar = ChemSVGGrammarV4(tokenizer, visual_first=visual_first)
    grammar.consume(tokenizer.bos_token_id)
    active = [
        _BeamStateV4(
            input_ids=(tokenizer.bos_token_id,),
            grammar=grammar,
            caches=None,
            log_probability=0.0,
        )
    ]
    completed: list[_BeamStateV4] = []
    best_partial = active[0]
    ocr_cache: dict[tuple[Coordinate, ...], tuple[str, ...]] = {}
    stroke_cache: dict[
        tuple[Coordinate, ...], set[tuple[Coordinate, Coordinate]]
    ] = {}
    stroke_token_id = tokenizer.token_to_id[tokenizer.STROKE]
    while active and len(active[0].input_ids) < max_length:
        candidates: list[_BeamStateV4] = []
        tokens = torch.tensor(
            [[state.input_ids[-1]] for state in active],
            device=memory.device,
        )
        batched_caches = None
        if active[0].caches is not None:
            batched_caches = [
                torch.cat(
                    [state.caches[layer_index] for state in active],
                    dim=0,
                )
                for layer_index in range(len(active[0].caches))
            ]
        logits, next_batched_caches = model.decode_step(
            tokens,
            len(active[0].input_ids) - 1,
            memory.expand(len(active), -1, -1),
            batched_caches,
        )
        for state_index, state in enumerate(active):
            allowed = state.grammar.allowed_token_ids()
            if not allowed:
                continue
            next_caches = [
                cache[state_index : state_index + 1]
                for cache in next_batched_caches
            ]
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            allowed_logits = logits[state_index, 0].float().index_select(
                0,
                allowed_tensor,
            )
            log_probabilities = torch.log_softmax(allowed_logits, dim=0)
            count = min(beam_width, len(allowed))
            top_scores, top_indices = log_probabilities.topk(count)
            for score, index in zip(top_scores.tolist(), top_indices.tolist()):
                next_id = int(allowed_tensor[index])
                next_grammar = copy.deepcopy(state.grammar)
                child_input_ids = state.input_ids
                child_caches = next_caches
                if (
                    ocr_features is not None
                    and next_id == stroke_token_id
                    and not state.grammar.stroke_started
                ):
                    labels = _recognize_vertex_labels(
                        model,
                        ocr_features,
                        next_grammar.vertices,
                        pixel_size,
                        ocr_cache,
                    )
                    child_input_ids = _fuse_ocr_vertex_prefix(
                        state.input_ids,
                        next_grammar,
                        labels,
                        tokenizer,
                    )
                    child_caches = _rebuild_prefix_caches(
                        model,
                        child_input_ids,
                        memory,
                    )
                if (
                    stroke_features is not None
                    and next_id == stroke_token_id
                    and not state.grammar.stroke_started
                ):
                    next_grammar.allowed_stroke_pairs = _score_vertex_pairs(
                        model,
                        stroke_features,
                        next_grammar.vertices,
                        pixel_size,
                        stroke_threshold,
                        stroke_cache,
                    )
                    if not next_grammar.allowed_stroke_pairs:
                        continue
                next_grammar.consume(next_id)
                child = _BeamStateV4(
                    input_ids=(*child_input_ids, next_id),
                    grammar=next_grammar,
                    caches=child_caches,
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
                return _completed_result_v4(best_completed.input_ids, tokenizer)
        active = sorted(candidates, key=_beam_rank, reverse=True)[:beam_width]
    if completed:
        best_completed = max(completed, key=_beam_rank)
        return _completed_result_v4(best_completed.input_ids, tokenizer)
    error = (
        "Beam generation exhausted all legal continuations"
        if not active
        else f"Beam generation did not finish within {max_length} tokens"
    )
    return GenerationResultV4(
        best_partial.input_ids,
        None,
        False,
        False,
        error,
    )


@torch.inference_mode()
def generate_chemsvg_v4(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    tokenizer: ChemSVGTokenizerV4,
    *,
    max_length: int | None = None,
    beam_width: int = 1,
    visual_first: bool = False,
    ocr_fusion: bool = False,
    stroke_scoring: bool = False,
    stroke_threshold: float = 0.5,
) -> GenerationResultV4:
    if pixel_values.size(0) != 1:
        raise ValueError("Generation currently supports batch size 1")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if ocr_fusion and model.ocr_head is None:
        raise ValueError("OCR fusion requires an OCR-enabled model")
    if stroke_scoring and model.stroke_head is None:
        raise ValueError("Stroke scoring requires a stroke-enabled model")
    if not 0.0 <= stroke_threshold <= 1.0:
        raise ValueError("stroke_threshold must be in [0, 1]")
    model.eval()
    max_length = max_length or model.config.max_target_length
    with torch.autocast(
        device_type=pixel_values.device.type,
        dtype=torch.bfloat16,
        enabled=pixel_values.device.type == "cuda",
    ):
        ocr_features = None
        if ocr_fusion:
            memory, ocr_features = model.encode_images_with_ocr_features(pixel_values)
        else:
            memory = model.encode_images(pixel_values)
        stroke_features = (
            model.vision_encoder.forward_stroke_resolution(pixel_values)
            if stroke_scoring
            else None
        )
        if beam_width > 1:
            return _generate_beam_v4(
                model,
                memory,
                tokenizer,
                max_length=max_length,
                beam_width=beam_width,
                visual_first=visual_first,
                ocr_features=ocr_features,
                stroke_features=stroke_features,
                pixel_size=pixel_values.shape[-2:],
                stroke_threshold=stroke_threshold,
            )
        grammar = ChemSVGGrammarV4(tokenizer, visual_first=visual_first)
        generated = [tokenizer.bos_token_id]
        grammar.consume(generated[0])
        caches = None
        exhausted = False
        ocr_cache: dict[tuple[Coordinate, ...], tuple[str, ...]] = {}
        stroke_cache: dict[
            tuple[Coordinate, ...], set[tuple[Coordinate, Coordinate]]
        ] = {}
        stroke_token_id = tokenizer.token_to_id[tokenizer.STROKE]
        while len(generated) < max_length:
            token = torch.tensor([[generated[-1]]], device=pixel_values.device)
            logits, caches = model.decode_step(token, len(generated) - 1, memory, caches)
            allowed = grammar.allowed_token_ids()
            if not allowed:
                exhausted = True
                break
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            scores = logits[0, 0].index_select(0, allowed_tensor)
            next_id = int(allowed_tensor[scores.argmax()])
            if (
                ocr_features is not None
                and next_id == stroke_token_id
                and not grammar.stroke_started
            ):
                labels = _recognize_vertex_labels(
                    model,
                    ocr_features,
                    grammar.vertices,
                    pixel_values.shape[-2:],
                    ocr_cache,
                )
                fused_ids = _fuse_ocr_vertex_prefix(
                    tuple(generated),
                    grammar,
                    labels,
                    tokenizer,
                )
                generated = list(fused_ids)
                caches = _rebuild_prefix_caches(model, fused_ids, memory)
            if (
                stroke_features is not None
                and next_id == stroke_token_id
                and not grammar.stroke_started
            ):
                grammar.allowed_stroke_pairs = _score_vertex_pairs(
                    model,
                    stroke_features,
                    grammar.vertices,
                    pixel_values.shape[-2:],
                    stroke_threshold,
                    stroke_cache,
                )
                if not grammar.allowed_stroke_pairs:
                    exhausted = True
                    break
            generated.append(next_id)
            grammar.consume(next_id)
            if grammar.phase == "done":
                return _completed_result_v4(tuple(generated), tokenizer)
    error = (
        "Generation exhausted all legal continuations"
        if exhausted
        else f"Generation did not finish within {max_length} tokens"
    )
    return GenerationResultV4(
        tuple(generated),
        None,
        False,
        False,
        error,
    )


@torch.inference_mode()
def generate_detected_chemsvg_v4(
    model: ChemVectorModel,
    pixel_values: torch.Tensor,
    tokenizer: ChemSVGTokenizerV4,
    *,
    vertex_threshold: float = 0.3,
    maximum_vertices: int = 128,
    stroke_threshold: float = 0.5,
    ocr_roi_radius: float = 20.0,
    max_length: int | None = None,
) -> GenerationResultV4:
    """Detect a fixed visual vertex prefix, then autoregress only over strokes."""

    if pixel_values.size(0) != 1:
        raise ValueError("Detected generation currently supports batch size 1")
    if not 0.0 <= vertex_threshold <= 1.0 or not 0.0 <= stroke_threshold <= 1.0:
        raise ValueError("Detection thresholds must be in [0, 1]")
    if ocr_roi_radius <= 0:
        raise ValueError("ocr_roi_radius must be positive")
    if model.vertex_head is None or model.stroke_head is None or model.ocr_head is None:
        raise ValueError("Detected generation requires vertex, stroke, and OCR heads")
    model.eval()
    max_length = max_length or model.config.max_target_length
    with torch.autocast(
        device_type=pixel_values.device.type,
        dtype=torch.bfloat16,
        enabled=pixel_values.device.type == "cuda",
    ):
        vertices, stroke_features = detect_vertex_records(
            model,
            pixel_values,
            threshold=vertex_threshold,
            maximum_vertices=maximum_vertices,
            ocr_roi_radius=ocr_roi_radius,
        )
        memory = model.encode_images(pixel_values)
        grammar = ChemSVGGrammarV4(tokenizer, visual_first=True)
        prefix = _vertex_prefix_ids(tokenizer, vertices)
        for token_id in prefix:
            grammar.consume(token_id)
        grammar.allowed_stroke_pairs = _score_vertex_pairs(
            model,
            stroke_features,
            vertices,
            pixel_values.shape[-2:],
            stroke_threshold,
            {},
        )
        if not grammar.allowed_stroke_pairs:
            return GenerationResultV4(
                prefix,
                None,
                False,
                False,
                "Stroke scorer found no visible vertex pairs",
            )

        generated = list(prefix)
        caches = None
        for position, token_id in enumerate(prefix[:-1]):
            token = torch.tensor([[token_id]], device=pixel_values.device)
            _, caches = model.decode_step(token, position, memory, caches)
        exhausted = False
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
                exhausted = True
                break
            allowed_tensor = torch.tensor(allowed, device=logits.device)
            scores = logits[0, 0].index_select(0, allowed_tensor)
            next_id = int(allowed_tensor[scores.argmax()])
            generated.append(next_id)
            grammar.consume(next_id)
            if grammar.phase == "done":
                return _completed_result_v4(tuple(generated), tokenizer)
    error = (
        "Detected generation exhausted all legal continuations"
        if exhausted
        else f"Detected generation did not finish within {max_length} tokens"
    )
    return GenerationResultV4(tuple(generated), None, False, False, error)
