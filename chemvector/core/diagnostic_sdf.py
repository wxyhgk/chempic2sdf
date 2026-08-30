"""Write valid and diagnostic-invalid SDF prediction artifacts."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from Data.sdf.chemsvg_sequence_v4 import (
    diagnostic_mol_from_chemsvg_v4,
    parse_chemsvg_v4,
)


def write_prediction_sdf(
    mol: Chem.Mol,
    path: Path,
    *,
    sanitize: bool,
) -> Chem.Mol:
    """Write the delivered SDF and return the molecule read back from that file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        writer = Chem.SDWriter(str(temporary))
        if not sanitize:
            writer.SetKekulize(False)
        try:
            writer.write(mol)
        finally:
            writer.close()
        written = Chem.MolFromMolFile(
            str(temporary),
            sanitize=sanitize,
            removeHs=False,
            strictParsing=False,
        )
        if written is None:
            raise ValueError("RDKit could not read back the generated SDF")
        temporary.replace(path)
        return written
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_manual_diagnostic_sdf(sequence: str, path: Path) -> tuple[int, int]:
    """Write a V3000 diagnostic SDF when RDKit cannot serialize an invalid graph."""

    parsed = parse_chemsvg_v4(sequence)
    coordinate_to_atom = {
        (vertex.x, vertex.y): index
        for index, vertex in enumerate(parsed.vertices, start=1)
    }
    bond_types = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    lines = [
        "ChemVector diagnostic prediction",
        "  ChemVector",
        "",
        "  0  0  0  0  0  0  0  0  0  0999 V3000",
        "M  V30 BEGIN CTAB",
        f"M  V30 COUNTS {len(parsed.vertices)} {len(parsed.strokes)} 0 0 0",
        "M  V30 BEGIN ATOM",
    ]
    for atom_id, vertex in enumerate(parsed.vertices, start=1):
        properties = []
        if vertex.formal_charge:
            properties.append(f"CHG={vertex.formal_charge}")
        if vertex.isotope:
            properties.append(f"MASS={vertex.isotope}")
        if vertex.radical_electrons:
            properties.append(f"RAD={min(vertex.radical_electrons + 1, 3)}")
        suffix = f" {' '.join(properties)}" if properties else ""
        lines.append(
            f"M  V30 {atom_id} {vertex.element} {vertex.x / 100.0:.6f} "
            f"{-vertex.y / 100.0:.6f} 0.000000 0{suffix}"
        )
    lines.extend(("M  V30 END ATOM", "M  V30 BEGIN BOND"))
    for bond_id, stroke in enumerate(parsed.strokes, start=1):
        lines.append(
            f"M  V30 {bond_id} {bond_types[stroke.bond_type]} "
            f"{coordinate_to_atom[stroke.begin]} {coordinate_to_atom[stroke.end]}"
        )
    lines.extend(
        (
            "M  V30 END BOND",
            "M  V30 END CTAB",
            "M  END",
            ">  <CHEMVECTOR_STATUS>",
            "invalid",
            "",
            ">  <CHEMSVG_SEQUENCE>",
            sequence.rstrip(),
            "",
            "$$$$",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return len(parsed.vertices), len(parsed.strokes)


def write_invalid_prediction_sdf(sequence: str, path: Path) -> tuple[int, int]:
    try:
        diagnostic_mol = diagnostic_mol_from_chemsvg_v4(sequence)
        written = write_prediction_sdf(diagnostic_mol, path, sanitize=False)
        return written.GetNumAtoms(), written.GetNumBonds()
    except Exception:
        return write_manual_diagnostic_sdf(sequence, path)
