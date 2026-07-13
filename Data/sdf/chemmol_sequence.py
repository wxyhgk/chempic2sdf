#!/usr/bin/env python3
"""ChemMolSeq v0.2 serialization and RDKit reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem


SCHEMA_VERSION = "chemmolseq-0.2"
START_TOKEN = "<mol v=0.2>"
END_TOKEN = "<mol_end>"
QUANTIZATION_MAX = 1023


class ChemMolSequenceError(ValueError):
    """Raised when a ChemMolSeq document is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class AtomRecord:
    atom_id: int
    element: str
    formal_charge: int
    isotope: int
    aromatic: bool
    explicit_h_count: int
    no_implicit: bool
    radical_electrons: int
    chiral_tag: str
    quantized_x: int
    quantized_y: int


@dataclass(frozen=True, slots=True)
class BondRecord:
    bond_id: int
    begin_atom: int
    end_atom: int
    bond_type: str
    display_type: str
    stereo: str
    stereo_atom_0: int
    stereo_atom_1: int
    direction: str


@dataclass(frozen=True, slots=True)
class ParsedChemMolSequence:
    atoms: tuple[AtomRecord, ...]
    bonds: tuple[BondRecord, ...]
    schema_version: str = SCHEMA_VERSION


def _bool_token(value: Any, field: str) -> str:
    if not isinstance(value, bool):
        raise ChemMolSequenceError(f"{field} must be a boolean")
    return "1" if value else "0"


def _parse_bool(value: str, field: str) -> bool:
    if value == "0":
        return False
    if value == "1":
        return True
    raise ChemMolSequenceError(f"{field} must be 0 or 1, got {value!r}")


def _require_int(value: Any, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChemMolSequenceError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ChemMolSequenceError(f"{field} must be >= {minimum}, got {value}")
    return value


def _parse_int(value: str, field: str, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ChemMolSequenceError(f"{field} must be an integer, got {value!r}") from error
    if minimum is not None and parsed < minimum:
        raise ChemMolSequenceError(f"{field} must be >= {minimum}, got {parsed}")
    return parsed


def _enum_member(enum_type: Any, name: str, field: str) -> Any:
    try:
        return getattr(enum_type, name)
    except AttributeError as error:
        raise ChemMolSequenceError(f"Unknown {field}: {name!r}") from error


def _validate_atom_order(atoms: list[dict[str, Any]]) -> None:
    ids = [_require_int(atom.get("id"), "atom.id", 0) for atom in atoms]
    expected = list(range(len(atoms)))
    if ids != expected:
        raise ChemMolSequenceError(
            f"Atom IDs must be contiguous and ordered, expected {expected[:5]}, got {ids[:5]}"
        )


def _validate_bond_order(bonds: list[dict[str, Any]]) -> None:
    ids = [_require_int(bond.get("id"), "bond.id", 0) for bond in bonds]
    expected = list(range(len(bonds)))
    if ids != expected:
        raise ChemMolSequenceError(
            f"Bond IDs must be contiguous and ordered, expected {expected[:5]}, got {ids[:5]}"
        )


def serialize_chemmolseq(chemmol: dict[str, Any], render: dict[str, Any]) -> str:
    """Serialize ChemMol and Render JSON objects into ChemMolSeq v0.2."""

    if chemmol.get("molecule_id") != render.get("molecule_id"):
        raise ChemMolSequenceError("ChemMol and Render molecule_id values differ")

    atoms = chemmol.get("atoms")
    bonds = chemmol.get("bonds")
    render_atoms = render.get("atoms")
    render_bonds = render.get("bonds")
    if not isinstance(atoms, list) or not atoms:
        raise ChemMolSequenceError("ChemMol atoms must be a non-empty list")
    if not isinstance(bonds, list):
        raise ChemMolSequenceError("ChemMol bonds must be a list")
    if not isinstance(render_atoms, list) or not isinstance(render_bonds, list):
        raise ChemMolSequenceError("Render atoms and bonds must be lists")

    _validate_atom_order(atoms)
    _validate_bond_order(bonds)

    positions = {atom.get("atom_id"): atom for atom in render_atoms}
    displays = {bond.get("bond_id"): bond for bond in render_bonds}
    if set(positions) != set(range(len(atoms))):
        raise ChemMolSequenceError("Render atom IDs do not match ChemMol atom IDs")
    if set(displays) != set(range(len(bonds))):
        raise ChemMolSequenceError("Render bond IDs do not match ChemMol bond IDs")

    lines = [START_TOKEN]
    for atom in atoms:
        atom_id = atom["id"]
        position = positions[atom_id]
        element = atom.get("element")
        if not isinstance(element, str) or not element or not element.isalpha():
            raise ChemMolSequenceError(f"Invalid element for atom {atom_id}: {element!r}")
        formal_charge = _require_int(atom.get("formal_charge"), "atom.formal_charge")
        isotope = _require_int(atom.get("isotope"), "atom.isotope", 0)
        explicit_h = _require_int(atom.get("explicit_h_count"), "atom.explicit_h_count", 0)
        radical = _require_int(atom.get("radical_electrons"), "atom.radical_electrons", 0)
        chiral_tag = atom.get("chiral_tag")
        if not isinstance(chiral_tag, str):
            raise ChemMolSequenceError(f"Invalid chiral tag for atom {atom_id}")
        _enum_member(Chem.ChiralType, chiral_tag, "chiral tag")
        quantized_x = _require_int(position.get("quantized_x"), "render.quantized_x", 0)
        quantized_y = _require_int(position.get("quantized_y"), "render.quantized_y", 0)
        if quantized_x > QUANTIZATION_MAX or quantized_y > QUANTIZATION_MAX:
            raise ChemMolSequenceError(
                f"Quantized coordinates exceed {QUANTIZATION_MAX} for atom {atom_id}"
            )
        lines.append(
            f"<a {atom_id} {element} {formal_charge} {isotope} "
            f"{_bool_token(atom.get('aromatic'), 'atom.aromatic')} {explicit_h} "
            f"{_bool_token(atom.get('no_implicit'), 'atom.no_implicit')} {radical} "
            f"{chiral_tag} {quantized_x} {quantized_y}>"
        )

    for bond in bonds:
        bond_id = bond["id"]
        display = displays[bond_id]
        begin_atom = _require_int(bond.get("begin_atom"), "bond.begin_atom", 0)
        end_atom = _require_int(bond.get("end_atom"), "bond.end_atom", 0)
        if begin_atom >= len(atoms) or end_atom >= len(atoms) or begin_atom == end_atom:
            raise ChemMolSequenceError(f"Invalid endpoints for bond {bond_id}")
        bond_type = bond.get("type")
        display_type = display.get("display_type")
        stereo = bond.get("stereo")
        direction = display.get("direction")
        for enum_type, value, field in (
            (Chem.BondType, bond_type, "bond type"),
            (Chem.BondType, display_type, "display bond type"),
            (Chem.BondStereo, stereo, "bond stereo"),
            (Chem.BondDir, direction, "bond direction"),
        ):
            if not isinstance(value, str):
                raise ChemMolSequenceError(f"Invalid {field} for bond {bond_id}")
            _enum_member(enum_type, value, field)
        stereo_atoms = bond.get("stereo_atoms")
        if not isinstance(stereo_atoms, list) or len(stereo_atoms) not in (0, 2):
            raise ChemMolSequenceError(f"Bond {bond_id} stereo_atoms must have length 0 or 2")
        if stereo_atoms:
            stereo_atom_0 = _require_int(stereo_atoms[0], "bond.stereo_atom_0", 0)
            stereo_atom_1 = _require_int(stereo_atoms[1], "bond.stereo_atom_1", 0)
            if stereo_atom_0 >= len(atoms) or stereo_atom_1 >= len(atoms):
                raise ChemMolSequenceError(f"Invalid stereo atom for bond {bond_id}")
        else:
            stereo_atom_0 = -1
            stereo_atom_1 = -1
        lines.append(
            f"<b {bond_id} {begin_atom} {end_atom} {bond_type} {display_type} "
            f"{stereo} {stereo_atom_0} {stereo_atom_1} {direction}>"
        )

    lines.append(END_TOKEN)
    return "\n".join(lines) + "\n"


def parse_chemmolseq(sequence: str) -> ParsedChemMolSequence:
    """Parse and strictly validate a ChemMolSeq v0.2 document."""

    if not isinstance(sequence, str):
        raise ChemMolSequenceError("ChemMolSeq input must be text")
    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != START_TOKEN or lines[-1] != END_TOKEN:
        raise ChemMolSequenceError("ChemMolSeq must start with v0.2 header and end with <mol_end>")

    atoms: list[AtomRecord] = []
    bonds: list[BondRecord] = []
    seen_bond = False
    for line_number, line in enumerate(lines[1:-1], 2):
        if not line.startswith("<") or not line.endswith(">"):
            raise ChemMolSequenceError(f"Line {line_number} is not a record")
        fields = line[1:-1].split()
        if not fields:
            raise ChemMolSequenceError(f"Line {line_number} is empty")
        if fields[0] == "a":
            if seen_bond:
                raise ChemMolSequenceError("Atom records must precede bond records")
            if len(fields) != 12:
                raise ChemMolSequenceError(
                    f"Atom line {line_number} has {len(fields) - 1} fields, expected 11"
                )
            atom_id = _parse_int(fields[1], "atom.id", 0)
            if atom_id != len(atoms):
                raise ChemMolSequenceError(f"Expected atom ID {len(atoms)}, got {atom_id}")
            element = fields[2]
            if not element.isalpha():
                raise ChemMolSequenceError(f"Invalid element token {element!r}")
            chiral_tag = fields[9]
            _enum_member(Chem.ChiralType, chiral_tag, "chiral tag")
            quantized_x = _parse_int(fields[10], "atom.quantized_x", 0)
            quantized_y = _parse_int(fields[11], "atom.quantized_y", 0)
            if quantized_x > QUANTIZATION_MAX or quantized_y > QUANTIZATION_MAX:
                raise ChemMolSequenceError("Atom coordinate is outside the quantized range")
            atoms.append(
                AtomRecord(
                    atom_id=atom_id,
                    element=element,
                    formal_charge=_parse_int(fields[3], "atom.formal_charge"),
                    isotope=_parse_int(fields[4], "atom.isotope", 0),
                    aromatic=_parse_bool(fields[5], "atom.aromatic"),
                    explicit_h_count=_parse_int(fields[6], "atom.explicit_h_count", 0),
                    no_implicit=_parse_bool(fields[7], "atom.no_implicit"),
                    radical_electrons=_parse_int(fields[8], "atom.radical_electrons", 0),
                    chiral_tag=chiral_tag,
                    quantized_x=quantized_x,
                    quantized_y=quantized_y,
                )
            )
        elif fields[0] == "b":
            seen_bond = True
            if len(fields) != 10:
                raise ChemMolSequenceError(
                    f"Bond line {line_number} has {len(fields) - 1} fields, expected 9"
                )
            bond_id = _parse_int(fields[1], "bond.id", 0)
            if bond_id != len(bonds):
                raise ChemMolSequenceError(f"Expected bond ID {len(bonds)}, got {bond_id}")
            begin_atom = _parse_int(fields[2], "bond.begin_atom", 0)
            end_atom = _parse_int(fields[3], "bond.end_atom", 0)
            if begin_atom >= len(atoms) or end_atom >= len(atoms) or begin_atom == end_atom:
                raise ChemMolSequenceError(f"Invalid endpoints for bond {bond_id}")
            bond_type, display_type, stereo = fields[4:7]
            stereo_atom_0 = _parse_int(fields[7], "bond.stereo_atom_0")
            stereo_atom_1 = _parse_int(fields[8], "bond.stereo_atom_1")
            direction = fields[9]
            for enum_type, value, field in (
                (Chem.BondType, bond_type, "bond type"),
                (Chem.BondType, display_type, "display bond type"),
                (Chem.BondStereo, stereo, "bond stereo"),
                (Chem.BondDir, direction, "bond direction"),
            ):
                _enum_member(enum_type, value, field)
            if (stereo_atom_0, stereo_atom_1) != (-1, -1):
                if stereo_atom_0 < 0 or stereo_atom_1 < 0:
                    raise ChemMolSequenceError(f"Bond {bond_id} has incomplete stereo atom references")
                if stereo_atom_0 >= len(atoms) or stereo_atom_1 >= len(atoms):
                    raise ChemMolSequenceError(f"Bond {bond_id} has invalid stereo atom references")
            bonds.append(
                BondRecord(
                    bond_id=bond_id,
                    begin_atom=begin_atom,
                    end_atom=end_atom,
                    bond_type=bond_type,
                    display_type=display_type,
                    stereo=stereo,
                    stereo_atom_0=stereo_atom_0,
                    stereo_atom_1=stereo_atom_1,
                    direction=direction,
                )
            )
        else:
            raise ChemMolSequenceError(f"Unknown record type on line {line_number}: {fields[0]!r}")

    if not atoms:
        raise ChemMolSequenceError("ChemMolSeq contains no atoms")
    return ParsedChemMolSequence(atoms=tuple(atoms), bonds=tuple(bonds))


def mol_from_parsed_sequence(
    parsed: ParsedChemMolSequence,
    coordinate_scale: float = 100.0,
    *,
    apply_display_directions: bool = False,
) -> Chem.Mol:
    """Build an RDKit molecule and 2D conformer from parsed ChemMolSeq data.

    Render bond directions are intentionally opt-in. Wedge and dash directions
    describe how one image was drawn; applying them while also restoring atom
    chiral tags can change RDKit/InChI stereo perception for complex molecules.
    """

    if coordinate_scale <= 0:
        raise ChemMolSequenceError("coordinate_scale must be positive")
    rw_mol = Chem.RWMol()
    for record in parsed.atoms:
        try:
            atom = Chem.Atom(record.element)
        except Exception as error:
            raise ChemMolSequenceError(f"Unknown element: {record.element!r}") from error
        atom.SetFormalCharge(record.formal_charge)
        atom.SetIsotope(record.isotope)
        atom.SetIsAromatic(record.aromatic)
        atom.SetNumExplicitHs(record.explicit_h_count)
        atom.SetNoImplicit(record.no_implicit)
        atom.SetNumRadicalElectrons(record.radical_electrons)
        atom.SetChiralTag(_enum_member(Chem.ChiralType, record.chiral_tag, "chiral tag"))
        rw_mol.AddAtom(atom)

    for record in parsed.bonds:
        try:
            rw_mol.AddBond(
                record.begin_atom,
                record.end_atom,
                _enum_member(Chem.BondType, record.bond_type, "bond type"),
            )
        except Exception as error:
            raise ChemMolSequenceError(f"Cannot add bond {record.bond_id}: {error}") from error
        bond = rw_mol.GetBondBetweenAtoms(record.begin_atom, record.end_atom)
        if bond is None:
            raise ChemMolSequenceError(f"Cannot retrieve bond {record.bond_id}")
        if record.bond_type == "AROMATIC":
            bond.SetIsAromatic(True)

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as error:
        raise ChemMolSequenceError(f"RDKit sanitization failed: {error}") from error

    for record in parsed.bonds:
        bond = mol.GetBondWithIdx(record.bond_id)
        if record.stereo_atom_0 >= 0:
            bond.SetStereoAtoms(record.stereo_atom_0, record.stereo_atom_1)
        bond.SetStereo(_enum_member(Chem.BondStereo, record.stereo, "bond stereo"))
        if apply_display_directions:
            bond.SetBondDir(_enum_member(Chem.BondDir, record.direction, "bond direction"))

    conformer = Chem.Conformer(len(parsed.atoms))
    conformer.Set3D(False)
    for record in parsed.atoms:
        conformer.SetAtomPosition(
            record.atom_id,
            (
                record.quantized_x / coordinate_scale,
                -record.quantized_y / coordinate_scale,
                0.0,
            ),
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def mol_from_chemmolseq(
    sequence: str,
    coordinate_scale: float = 100.0,
    *,
    apply_display_directions: bool = False,
) -> Chem.Mol:
    """Parse ChemMolSeq text and reconstruct an RDKit molecule."""

    return mol_from_parsed_sequence(
        parse_chemmolseq(sequence),
        coordinate_scale,
        apply_display_directions=apply_display_directions,
    )


def validate_chemmolseq_roundtrip(
    sequence: str,
    expected_smiles: str,
    expected_inchikey: str,
) -> Chem.Mol:
    """Require a sequence to reconstruct the expected chemical structure exactly."""

    mol = mol_from_chemmolseq(sequence)
    actual_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    actual_inchikey = Chem.MolToInchiKey(mol)
    expected_mol = Chem.MolFromSmiles(expected_smiles)
    if expected_mol is None:
        raise ChemMolSequenceError(f"Expected SMILES cannot be parsed: {expected_smiles}")
    normalized_expected = Chem.MolToSmiles(expected_mol, canonical=True, isomericSmiles=True)
    normalized_actual = Chem.MolToSmiles(
        Chem.MolFromSmiles(actual_smiles), canonical=True, isomericSmiles=True
    )
    if normalized_actual != normalized_expected or actual_inchikey != expected_inchikey:
        raise ChemMolSequenceError(
            "ChemMolSeq chemical roundtrip mismatch: "
            f"actual={actual_smiles}/{actual_inchikey}, "
            f"expected={normalized_expected}/{expected_inchikey}"
        )
    return mol
