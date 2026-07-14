#!/usr/bin/env python3
"""ChemSVG v0.4 visual primitives and deterministic RDKit compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

try:
    from .chemmol_sequence import (
        ChemMolSequenceError,
        mol_from_chemmolseq as mol_from_chemmolseq_v2,
        serialize_chemmolseq as serialize_chemmolseq_v2,
    )
    from .chemmol_sequence_v3 import canonical_chiral_tag, spatial_atom_order
except ImportError:
    from chemmol_sequence import (
        ChemMolSequenceError,
        mol_from_chemmolseq as mol_from_chemmolseq_v2,
        serialize_chemmolseq as serialize_chemmolseq_v2,
    )
    from chemmol_sequence_v3 import canonical_chiral_tag, spatial_atom_order


SCHEMA_VERSION = "chemsvg-0.4"
START_TOKEN = "<chemsvg v=0.4 units=q1023>"
END_TOKEN = "<chemsvg_end>"
QUANTIZATION_MAX = 1023


class ChemSVGCoordinateCollisionError(ChemMolSequenceError):
    """Raised when two visual atoms occupy the same quantized coordinate."""


@dataclass(frozen=True, slots=True)
class VertexRecordV4:
    x: int
    y: int
    element: str
    formal_charge: int
    isotope: int
    aromatic: bool
    explicit_h_count: int
    no_implicit: bool
    radical_electrons: int
    canonical_chiral_tag: str


@dataclass(frozen=True, slots=True)
class StrokeRecordV4:
    begin_x: int
    begin_y: int
    end_x: int
    end_y: int
    display_type: str
    bond_type: str
    direction: str
    stereo: str
    stereo_atom_0_x: int = -1
    stereo_atom_0_y: int = -1
    stereo_atom_1_x: int = -1
    stereo_atom_1_y: int = -1

    @property
    def begin(self) -> tuple[int, int]:
        return self.begin_x, self.begin_y

    @property
    def end(self) -> tuple[int, int]:
        return self.end_x, self.end_y

    @property
    def stereo_atom_0(self) -> tuple[int, int] | None:
        if self.stereo_atom_0_x < 0:
            return None
        return self.stereo_atom_0_x, self.stereo_atom_0_y

    @property
    def stereo_atom_1(self) -> tuple[int, int] | None:
        if self.stereo_atom_1_x < 0:
            return None
        return self.stereo_atom_1_x, self.stereo_atom_1_y


@dataclass(frozen=True, slots=True)
class ParsedChemSVGSequenceV4:
    vertices: tuple[VertexRecordV4, ...]
    strokes: tuple[StrokeRecordV4, ...]
    schema_version: str = SCHEMA_VERSION


def serialize_parsed_chemsvg_v4(parsed: ParsedChemSVGSequenceV4) -> str:
    """Serialize already constructed visual records in canonical record order."""

    lines = [START_TOKEN]
    for vertex in parsed.vertices:
        lines.append(
            f"<v {vertex.x} {vertex.y} {vertex.element} {vertex.formal_charge} "
            f"{vertex.isotope} {int(vertex.aromatic)} {vertex.explicit_h_count} "
            f"{int(vertex.no_implicit)} {vertex.radical_electrons} "
            f"{vertex.canonical_chiral_tag}>"
        )
    for stroke in sorted(parsed.strokes, key=_stroke_sort_key):
        line = (
            f"<s {stroke.begin_x} {stroke.begin_y} {stroke.end_x} {stroke.end_y} "
            f"{stroke.display_type} {stroke.bond_type} {stroke.direction} "
            f"{stroke.stereo}"
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


def _enum_member(enum_type: Any, name: str, field: str) -> Any:
    try:
        return getattr(enum_type, name)
    except AttributeError as error:
        raise ChemMolSequenceError(f"Unknown {field}: {name!r}") from error


def _parse_int(value: str, field: str, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ChemMolSequenceError(f"{field} must be an integer, got {value!r}") from error
    if minimum is not None and result < minimum:
        raise ChemMolSequenceError(f"{field} must be >= {minimum}, got {result}")
    return result


def _parse_coordinate(value: str, field: str) -> int:
    result = _parse_int(value, field, 0)
    if result > QUANTIZATION_MAX:
        raise ChemMolSequenceError(f"{field} is outside the quantized canvas")
    return result


def _parse_bool(value: str, field: str) -> bool:
    if value == "0":
        return False
    if value == "1":
        return True
    raise ChemMolSequenceError(f"{field} must be 0 or 1, got {value!r}")


def _permutation_is_odd(values: list[int]) -> bool:
    ranks = {value: index for index, value in enumerate(sorted(values))}
    permutation = [ranks[value] for value in values]
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return bool(inversions % 2)


def _flip_tetrahedral_tag(tag: Any) -> Any:
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return Chem.ChiralType.CHI_TETRAHEDRAL_CW
    return tag


def _stroke_sort_key(
    record: StrokeRecordV4,
) -> tuple[tuple[int, int], tuple[int, int], str, str]:
    endpoints = sorted(((record.begin_y, record.begin_x), (record.end_y, record.end_x)))
    return endpoints[0], endpoints[1], record.display_type, record.bond_type


def serialize_chemsvg_v4(chemmol: dict[str, Any], render: dict[str, Any]) -> str:
    """Serialize visual vertices and strokes without atom-index bond references."""
    if chemmol.get("molecule_id") != render.get("molecule_id"):
        raise ChemMolSequenceError("ChemMol and Render molecule_id values differ")

    source_sequence = serialize_chemmolseq_v2(chemmol, render)
    source_mol = mol_from_chemmolseq_v2(source_sequence)
    order = spatial_atom_order(render, source_mol.GetNumAtoms())
    spatial_mol = Chem.RenumberAtoms(source_mol, order)
    positions = {item["atom_id"]: item for item in render["atoms"]}
    coordinates = {
        atom_id: (positions[atom_id]["quantized_x"], positions[atom_id]["quantized_y"])
        for atom_id in range(source_mol.GetNumAtoms())
    }
    if len(set(coordinates.values())) != len(coordinates):
        raise ChemSVGCoordinateCollisionError(
            "ChemSVG v0.4 cannot represent overlapping visual atom coordinates"
        )

    vertices = []
    for spatial_atom_id, source_atom_id in enumerate(order):
        atom = spatial_mol.GetAtomWithIdx(spatial_atom_id)
        x, y = coordinates[source_atom_id]
        vertices.append(
            VertexRecordV4(
                x=x,
                y=y,
                element=atom.GetSymbol(),
                formal_charge=atom.GetFormalCharge(),
                isotope=atom.GetIsotope(),
                aromatic=atom.GetIsAromatic(),
                explicit_h_count=atom.GetNumExplicitHs(),
                no_implicit=atom.GetNoImplicit(),
                radical_electrons=atom.GetNumRadicalElectrons(),
                canonical_chiral_tag=canonical_chiral_tag(atom),
            )
        )

    render_bonds = {item["bond_id"]: item for item in render["bonds"]}
    chemical_bonds = {item["id"]: item for item in chemmol["bonds"]}
    if set(render_bonds) != set(chemical_bonds):
        raise ChemMolSequenceError("Render and ChemMol bond IDs differ")
    strokes = []
    for bond_id in sorted(chemical_bonds):
        chemical = chemical_bonds[bond_id]
        visual = render_bonds[bond_id]
        begin = coordinates[chemical["begin_atom"]]
        end = coordinates[chemical["end_atom"]]
        stereo_atoms = chemical.get("stereo_atoms", [])
        stereo_coordinates = [coordinates[atom_id] for atom_id in stereo_atoms]
        if stereo_coordinates:
            if len(stereo_coordinates) != 2:
                raise ChemMolSequenceError("Bond stereo must reference exactly two vertices")
            stereo_0_x, stereo_0_y = stereo_coordinates[0]
            stereo_1_x, stereo_1_y = stereo_coordinates[1]
        else:
            stereo_0_x = stereo_0_y = stereo_1_x = stereo_1_y = -1
        strokes.append(
            StrokeRecordV4(
                begin_x=begin[0],
                begin_y=begin[1],
                end_x=end[0],
                end_y=end[1],
                display_type=visual["display_type"],
                bond_type=chemical["type"],
                direction=visual["direction"],
                stereo=chemical["stereo"],
                stereo_atom_0_x=stereo_0_x,
                stereo_atom_0_y=stereo_0_y,
                stereo_atom_1_x=stereo_1_x,
                stereo_atom_1_y=stereo_1_y,
            )
        )
    strokes.sort(key=_stroke_sort_key)

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


def parse_chemsvg_v4(sequence: str) -> ParsedChemSVGSequenceV4:
    if not isinstance(sequence, str):
        raise ChemMolSequenceError("ChemSVG input must be text")
    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != START_TOKEN or lines[-1] != END_TOKEN:
        raise ChemMolSequenceError("ChemSVG v0.4 has an invalid document boundary")

    vertices = []
    strokes = []
    seen_stroke = False
    coordinates: set[tuple[int, int]] = set()
    endpoint_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for line_number, line in enumerate(lines[1:-1], 2):
        if not line.startswith("<") or not line.endswith(">"):
            raise ChemMolSequenceError(f"Line {line_number} is not a record")
        fields = line[1:-1].split()
        if fields[0] == "v":
            if seen_stroke or len(fields) != 11:
                raise ChemMolSequenceError(f"Invalid vertex record on line {line_number}")
            coordinate = (
                _parse_coordinate(fields[1], "vertex.x"),
                _parse_coordinate(fields[2], "vertex.y"),
            )
            if coordinate in coordinates:
                raise ChemSVGCoordinateCollisionError(
                    f"Duplicate visual vertex coordinate on line {line_number}"
                )
            coordinates.add(coordinate)
            _enum_member(Chem.ChiralType, fields[10], "canonical chiral tag")
            vertices.append(
                VertexRecordV4(
                    x=coordinate[0],
                    y=coordinate[1],
                    element=fields[3],
                    formal_charge=_parse_int(fields[4], "vertex.formal_charge"),
                    isotope=_parse_int(fields[5], "vertex.isotope", 0),
                    aromatic=_parse_bool(fields[6], "vertex.aromatic"),
                    explicit_h_count=_parse_int(fields[7], "vertex.explicit_h_count", 0),
                    no_implicit=_parse_bool(fields[8], "vertex.no_implicit"),
                    radical_electrons=_parse_int(fields[9], "vertex.radical_electrons", 0),
                    canonical_chiral_tag=fields[10],
                )
            )
            continue
        if fields[0] != "s" or len(fields) not in {9, 13}:
            raise ChemMolSequenceError(f"Invalid stroke record on line {line_number}")
        seen_stroke = True
        begin = (
            _parse_coordinate(fields[1], "stroke.begin_x"),
            _parse_coordinate(fields[2], "stroke.begin_y"),
        )
        end = (
            _parse_coordinate(fields[3], "stroke.end_x"),
            _parse_coordinate(fields[4], "stroke.end_y"),
        )
        if begin == end or begin not in coordinates or end not in coordinates:
            raise ChemMolSequenceError(f"Stroke endpoints are invalid on line {line_number}")
        pair = tuple(sorted((begin, end)))
        if pair in endpoint_pairs:
            raise ChemMolSequenceError(f"Duplicate stroke endpoints on line {line_number}")
        endpoint_pairs.add(pair)
        _enum_member(Chem.BondType, fields[5], "display bond type")
        _enum_member(Chem.BondType, fields[6], "chemical bond type")
        _enum_member(Chem.BondDir, fields[7], "bond direction")
        _enum_member(Chem.BondStereo, fields[8], "bond stereo")
        if len(fields) == 13:
            stereo_0 = (
                _parse_coordinate(fields[9], "stroke.stereo_atom_0_x"),
                _parse_coordinate(fields[10], "stroke.stereo_atom_0_y"),
            )
            stereo_1 = (
                _parse_coordinate(fields[11], "stroke.stereo_atom_1_x"),
                _parse_coordinate(fields[12], "stroke.stereo_atom_1_y"),
            )
            if stereo_0 not in coordinates or stereo_1 not in coordinates:
                raise ChemMolSequenceError("Stroke stereo reference is not a visual vertex")
        else:
            stereo_0 = stereo_1 = (-1, -1)
        stereo_requires_references = fields[8] not in {"STEREONONE", "STEREOANY"}
        if stereo_requires_references != (len(fields) == 13):
            raise ChemMolSequenceError("Stroke stereo fields do not match the stereo value")
        strokes.append(
            StrokeRecordV4(
                begin_x=begin[0],
                begin_y=begin[1],
                end_x=end[0],
                end_y=end[1],
                display_type=fields[5],
                bond_type=fields[6],
                direction=fields[7],
                stereo=fields[8],
                stereo_atom_0_x=stereo_0[0],
                stereo_atom_0_y=stereo_0[1],
                stereo_atom_1_x=stereo_1[0],
                stereo_atom_1_y=stereo_1[1],
            )
        )
    if not vertices:
        raise ChemMolSequenceError("ChemSVG v0.4 contains no visual vertices")
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {
        coordinate: set() for coordinate in coordinates
    }
    for stroke in strokes:
        adjacency[stroke.begin].add(stroke.end)
        adjacency[stroke.end].add(stroke.begin)
    for stroke in strokes:
        if stroke.stereo_atom_0 is None:
            continue
        if stroke.stereo_atom_0 not in adjacency[stroke.begin]:
            raise ChemMolSequenceError(
                "First stroke stereo reference is not connected to the begin vertex"
            )
        if stroke.stereo_atom_1 not in adjacency[stroke.end]:
            raise ChemMolSequenceError(
                "Second stroke stereo reference is not connected to the end vertex"
            )
    return ParsedChemSVGSequenceV4(tuple(vertices), tuple(strokes))


def mol_from_parsed_chemsvg_v4(
    parsed: ParsedChemSVGSequenceV4,
    *,
    apply_display_directions: bool = False,
) -> Chem.Mol:
    rw_mol = Chem.RWMol()
    coordinate_to_atom = {}
    canonical_tags = []
    for atom_id, record in enumerate(parsed.vertices):
        atom = Chem.Atom(record.element)
        atom.SetFormalCharge(record.formal_charge)
        atom.SetIsotope(record.isotope)
        atom.SetIsAromatic(record.aromatic)
        atom.SetNumExplicitHs(record.explicit_h_count)
        atom.SetNoImplicit(record.no_implicit)
        atom.SetNumRadicalElectrons(record.radical_electrons)
        rw_mol.AddAtom(atom)
        coordinate_to_atom[(record.x, record.y)] = atom_id
        canonical_tags.append(
            _enum_member(
                Chem.ChiralType,
                record.canonical_chiral_tag,
                "canonical chiral tag",
            )
        )

    stroke_bonds = []
    for record in parsed.strokes:
        begin_atom = coordinate_to_atom[record.begin]
        end_atom = coordinate_to_atom[record.end]
        rw_mol.AddBond(
            begin_atom,
            end_atom,
            _enum_member(Chem.BondType, record.bond_type, "chemical bond type"),
        )
        bond = rw_mol.GetBondBetweenAtoms(begin_atom, end_atom)
        if record.bond_type == "AROMATIC":
            bond.SetIsAromatic(True)
        stroke_bonds.append((record, begin_atom, end_atom))

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as error:
        raise ChemMolSequenceError(f"RDKit sanitization failed: {error}") from error

    conformer = Chem.Conformer(len(parsed.vertices))
    conformer.Set3D(False)
    for atom_id, record in enumerate(parsed.vertices):
        conformer.SetAtomPosition(atom_id, (record.x / 100.0, -record.y / 100.0, 0.0))
    mol.AddConformer(conformer, assignId=True)

    for atom_id, canonical_tag in enumerate(canonical_tags):
        atom = mol.GetAtomWithIdx(atom_id)
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        actual_tag = canonical_tag
        if canonical_tag in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ) and _permutation_is_odd(neighbors):
            actual_tag = _flip_tetrahedral_tag(canonical_tag)
        atom.SetChiralTag(actual_tag)

    for record, begin_atom, end_atom in stroke_bonds:
        bond = mol.GetBondBetweenAtoms(begin_atom, end_atom)
        if record.stereo_atom_0 is not None:
            try:
                bond.SetStereoAtoms(
                    coordinate_to_atom[record.stereo_atom_0],
                    coordinate_to_atom[record.stereo_atom_1],
                )
            except RuntimeError as error:
                raise ChemMolSequenceError(
                    f"RDKit rejected stroke stereo references: {error}"
                ) from error
        bond.SetStereo(_enum_member(Chem.BondStereo, record.stereo, "bond stereo"))
        if apply_display_directions:
            bond.SetBondDir(_enum_member(Chem.BondDir, record.direction, "bond direction"))

    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def mol_from_chemsvg_v4(
    sequence: str,
    *,
    apply_display_directions: bool = False,
) -> Chem.Mol:
    return mol_from_parsed_chemsvg_v4(
        parse_chemsvg_v4(sequence),
        apply_display_directions=apply_display_directions,
    )


def validate_chemsvg_v4_roundtrip(
    sequence: str,
    expected_smiles: str,
    expected_inchikey: str,
) -> Chem.Mol:
    mol = mol_from_chemsvg_v4(sequence)
    actual_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    actual_inchikey = Chem.MolToInchiKey(mol)
    expected_mol = Chem.MolFromSmiles(expected_smiles)
    actual_mol = Chem.MolFromSmiles(actual_smiles)
    if expected_mol is None or actual_mol is None:
        raise ChemMolSequenceError("Cannot normalize a ChemSVG roundtrip SMILES value")
    normalized_expected = Chem.MolToSmiles(
        expected_mol,
        canonical=True,
        isomericSmiles=True,
    )
    normalized_actual = Chem.MolToSmiles(
        actual_mol,
        canonical=True,
        isomericSmiles=True,
    )
    if normalized_actual != normalized_expected or actual_inchikey != expected_inchikey:
        raise ChemMolSequenceError(
            f"ChemSVG roundtrip mismatch: {normalized_actual} / {actual_inchikey}"
        )
    return mol
