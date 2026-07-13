#!/usr/bin/env python3
"""ChemMolSeq v0.3 spatial ordering and RDKit reconstruction."""

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
except ImportError:
    from chemmol_sequence import (
        ChemMolSequenceError,
        mol_from_chemmolseq as mol_from_chemmolseq_v2,
        serialize_chemmolseq as serialize_chemmolseq_v2,
    )


SCHEMA_VERSION = "chemmolseq-0.3"
START_TOKEN = "<mol v=0.3 order=spatial-yx>"
END_TOKEN = "<mol_end>"
COORDS_START_TOKEN = "<coords>"
COORDS_END_TOKEN = "<coords_end>"
QUANTIZATION_MAX = 1023


@dataclass(frozen=True, slots=True)
class AtomRecordV3:
    element: str
    formal_charge: int
    isotope: int
    aromatic: bool
    explicit_h_count: int
    no_implicit: bool
    radical_electrons: int
    canonical_chiral_tag: str


@dataclass(frozen=True, slots=True)
class BondRecordV3:
    begin_atom: int
    end_atom: int
    bond_type: str
    stereo: str
    stereo_atom_0: int
    stereo_atom_1: int


@dataclass(frozen=True, slots=True)
class ParsedChemMolSequenceV3:
    atoms: tuple[AtomRecordV3, ...]
    bonds: tuple[BondRecordV3, ...]
    positions: tuple[tuple[int, int], ...]
    schema_version: str = SCHEMA_VERSION


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


def canonical_chiral_tag(atom: Chem.Atom) -> str:
    """Express tetrahedral parity relative to neighbors sorted by spatial atom ID."""
    tag = atom.GetChiralTag()
    neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
    if tag in (
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ) and _permutation_is_odd(neighbors):
        tag = _flip_tetrahedral_tag(tag)
    return str(tag)


def spatial_atom_order(render: dict[str, Any], atom_count: int) -> list[int]:
    positions = {item.get("atom_id"): item for item in render.get("atoms", [])}
    if set(positions) != set(range(atom_count)):
        raise ChemMolSequenceError("Render atom IDs do not match the molecule")
    return sorted(
        range(atom_count),
        key=lambda atom_id: (
            positions[atom_id].get("quantized_y"),
            positions[atom_id].get("quantized_x"),
            atom_id,
        ),
    )


def serialize_chemmolseq_v3(chemmol: dict[str, Any], render: dict[str, Any]) -> str:
    """Create a minimal chemical sequence ordered by visible image coordinates."""
    if chemmol.get("molecule_id") != render.get("molecule_id"):
        raise ChemMolSequenceError("ChemMol and Render molecule_id values differ")

    source_sequence = serialize_chemmolseq_v2(chemmol, render)
    source_mol = mol_from_chemmolseq_v2(source_sequence)
    order = spatial_atom_order(render, source_mol.GetNumAtoms())
    spatial_mol = Chem.RenumberAtoms(source_mol, order)

    lines = [START_TOKEN]
    for atom in spatial_mol.GetAtoms():
        lines.append(
            f"<a {atom.GetSymbol()} {atom.GetFormalCharge()} {atom.GetIsotope()} "
            f"{int(atom.GetIsAromatic())} {atom.GetNumExplicitHs()} "
            f"{int(atom.GetNoImplicit())} {atom.GetNumRadicalElectrons()} "
            f"{canonical_chiral_tag(atom)}>"
        )

    bonds = list(spatial_mol.GetBonds())
    for bond in bonds:
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        stereo_atoms = list(bond.GetStereoAtoms())
        if stereo_atoms:
            stereo_atom_0, stereo_atom_1 = stereo_atoms
        else:
            stereo_atom_0 = stereo_atom_1 = -1
        lines.append(
            f"<b {begin_atom} {end_atom} {bond.GetBondType()} {bond.GetStereo()} "
            f"{stereo_atom_0} {stereo_atom_1}>"
        )

    positions = {item["atom_id"]: item for item in render["atoms"]}
    lines.append(COORDS_START_TOKEN)
    for source_atom_id in order:
        position = positions[source_atom_id]
        lines.append(f"<p {position['quantized_x']} {position['quantized_y']}>")
    lines.append(COORDS_END_TOKEN)
    lines.append(END_TOKEN)
    sequence = "\n".join(lines) + "\n"
    parse_chemmolseq_v3(sequence)
    return sequence


def parse_chemmolseq_v3(sequence: str) -> ParsedChemMolSequenceV3:
    if not isinstance(sequence, str):
        raise ChemMolSequenceError("ChemMolSeq input must be text")
    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != START_TOKEN or lines[-1] != END_TOKEN:
        raise ChemMolSequenceError("ChemMolSeq v0.3 has an invalid document boundary")

    atoms: list[AtomRecordV3] = []
    bonds: list[BondRecordV3] = []
    positions: list[tuple[int, int]] = []
    seen_bond = False
    seen_coords = False
    coords_complete = False
    seen_bond_pairs: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines[1:-1], 2):
        if line == COORDS_START_TOKEN:
            if seen_coords or coords_complete:
                raise ChemMolSequenceError("ChemMolSeq v0.3 has duplicate coordinate sections")
            seen_coords = True
            continue
        if line == COORDS_END_TOKEN:
            if not seen_coords or coords_complete:
                raise ChemMolSequenceError("ChemMolSeq v0.3 has an invalid coordinate boundary")
            coords_complete = True
            continue
        if not line.startswith("<") or not line.endswith(">"):
            raise ChemMolSequenceError(f"Line {line_number} is not a record")
        fields = line[1:-1].split()
        if seen_coords:
            if coords_complete or fields[0] != "p" or len(fields) != 3:
                raise ChemMolSequenceError(f"Invalid coordinate record on line {line_number}")
            x = _parse_int(fields[1], "position.x", 0)
            y = _parse_int(fields[2], "position.y", 0)
            if x > QUANTIZATION_MAX or y > QUANTIZATION_MAX:
                raise ChemMolSequenceError("Coordinate is outside the quantized range")
            positions.append((x, y))
            continue
        if fields[0] == "a":
            if seen_bond:
                raise ChemMolSequenceError("Atom records must precede bond records")
            if len(fields) != 9:
                raise ChemMolSequenceError(
                    f"Atom line {line_number} has {len(fields) - 1} fields, expected 8"
                )
            element = fields[1]
            if not element.isalpha():
                raise ChemMolSequenceError(f"Invalid element token {element!r}")
            _enum_member(Chem.ChiralType, fields[8], "canonical chiral tag")
            atoms.append(
                AtomRecordV3(
                    element=element,
                    formal_charge=_parse_int(fields[2], "atom.formal_charge"),
                    isotope=_parse_int(fields[3], "atom.isotope", 0),
                    aromatic=_parse_bool(fields[4], "atom.aromatic"),
                    explicit_h_count=_parse_int(fields[5], "atom.explicit_h_count", 0),
                    no_implicit=_parse_bool(fields[6], "atom.no_implicit"),
                    radical_electrons=_parse_int(fields[7], "atom.radical_electrons", 0),
                    canonical_chiral_tag=fields[8],
                )
            )
        elif fields[0] == "b":
            seen_bond = True
            if len(fields) != 7:
                raise ChemMolSequenceError(
                    f"Bond line {line_number} has {len(fields) - 1} fields, expected 6"
                )
            begin_atom = _parse_int(fields[1], "bond.begin_atom", 0)
            end_atom = _parse_int(fields[2], "bond.end_atom", 0)
            if (
                begin_atom == end_atom
                or begin_atom >= len(atoms)
                or end_atom >= len(atoms)
            ):
                raise ChemMolSequenceError(f"Invalid endpoints on line {line_number}")
            bond_type = fields[3]
            stereo = fields[4]
            _enum_member(Chem.BondType, bond_type, "bond type")
            _enum_member(Chem.BondStereo, stereo, "bond stereo")
            stereo_atom_0 = _parse_int(fields[5], "bond.stereo_atom_0")
            stereo_atom_1 = _parse_int(fields[6], "bond.stereo_atom_1")
            if (stereo_atom_0, stereo_atom_1) != (-1, -1):
                if min(stereo_atom_0, stereo_atom_1) < 0:
                    raise ChemMolSequenceError("Bond stereo references must both be present")
                if max(stereo_atom_0, stereo_atom_1) >= len(atoms):
                    raise ChemMolSequenceError("Bond stereo reference is outside the atom range")
            endpoints = tuple(sorted((begin_atom, end_atom)))
            if endpoints in seen_bond_pairs:
                raise ChemMolSequenceError("Bond records must contain unique endpoint pairs")
            seen_bond_pairs.add(endpoints)
            bonds.append(
                BondRecordV3(
                    begin_atom=begin_atom,
                    end_atom=end_atom,
                    bond_type=bond_type,
                    stereo=stereo,
                    stereo_atom_0=stereo_atom_0,
                    stereo_atom_1=stereo_atom_1,
                )
            )
        else:
            raise ChemMolSequenceError(f"Unknown record type on line {line_number}")

    if not atoms:
        raise ChemMolSequenceError("ChemMolSeq v0.3 contains no atoms")
    if not coords_complete or len(positions) != len(atoms):
        raise ChemMolSequenceError("Coordinate count must equal atom count")
    return ParsedChemMolSequenceV3(tuple(atoms), tuple(bonds), tuple(positions))


def mol_from_parsed_sequence_v3(
    parsed: ParsedChemMolSequenceV3,
    *,
    apply_render_coordinates: bool = False,
) -> Chem.Mol:
    rw_mol = Chem.RWMol()
    canonical_tags = []
    for record in parsed.atoms:
        atom = Chem.Atom(record.element)
        atom.SetFormalCharge(record.formal_charge)
        atom.SetIsotope(record.isotope)
        atom.SetIsAromatic(record.aromatic)
        atom.SetNumExplicitHs(record.explicit_h_count)
        atom.SetNoImplicit(record.no_implicit)
        atom.SetNumRadicalElectrons(record.radical_electrons)
        rw_mol.AddAtom(atom)
        canonical_tags.append(
            _enum_member(
                Chem.ChiralType,
                record.canonical_chiral_tag,
                "canonical chiral tag",
            )
        )

    for record in parsed.bonds:
        rw_mol.AddBond(
            record.begin_atom,
            record.end_atom,
            _enum_member(Chem.BondType, record.bond_type, "bond type"),
        )
        bond = rw_mol.GetBondBetweenAtoms(record.begin_atom, record.end_atom)
        if record.bond_type == "AROMATIC":
            bond.SetIsAromatic(True)

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as error:
        raise ChemMolSequenceError(f"RDKit sanitization failed: {error}") from error

    if apply_render_coordinates:
        conformer = Chem.Conformer(len(parsed.atoms))
        conformer.Set3D(False)
        for atom_id, (x, y) in enumerate(parsed.positions):
            conformer.SetAtomPosition(atom_id, (x / 100.0, -y / 100.0, 0.0))
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

    for bond_id, record in enumerate(parsed.bonds):
        bond = mol.GetBondWithIdx(bond_id)
        if record.stereo_atom_0 >= 0:
            bond.SetStereoAtoms(record.stereo_atom_0, record.stereo_atom_1)
        bond.SetStereo(_enum_member(Chem.BondStereo, record.stereo, "bond stereo"))

    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def mol_from_chemmolseq_v3(
    sequence: str,
    *,
    apply_render_coordinates: bool = False,
) -> Chem.Mol:
    return mol_from_parsed_sequence_v3(
        parse_chemmolseq_v3(sequence),
        apply_render_coordinates=apply_render_coordinates,
    )


def validate_chemmolseq_v3_roundtrip(
    sequence: str,
    expected_smiles: str,
    expected_inchikey: str,
) -> Chem.Mol:
    mol = mol_from_chemmolseq_v3(sequence)
    actual_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    actual_inchikey = Chem.MolToInchiKey(mol)
    expected_mol = Chem.MolFromSmiles(expected_smiles)
    actual_mol = Chem.MolFromSmiles(actual_smiles)
    if expected_mol is None or actual_mol is None:
        raise ChemMolSequenceError("Cannot normalize a roundtrip SMILES value")
    normalized_expected = Chem.MolToSmiles(
        expected_mol, canonical=True, isomericSmiles=True
    )
    normalized_actual = Chem.MolToSmiles(
        actual_mol, canonical=True, isomericSmiles=True
    )
    if normalized_actual != normalized_expected or actual_inchikey != expected_inchikey:
        raise ChemMolSequenceError(
            f"Chemical roundtrip mismatch: {normalized_actual} / {actual_inchikey}"
        )
    return mol
