"""SDF serialization with a strict stereochemistry roundtrip check."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemsvg_sequence_v4 import mol_from_chemsvg_v4


def write_sdf_v4(sequence: str, path: str | Path) -> None:
    mol = mol_from_chemsvg_v4(sequence, apply_display_directions=True)
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
    expected_key = Chem.MolToInchiKey(mol)
    actual_key = Chem.MolToInchiKey(written_mol) if written_mol is not None else None
    if actual_key != expected_key:
        temporary_path.unlink(missing_ok=True)
        raise ChemMolSequenceError(
            f"SDF ChemSVG stereo mismatch: expected {expected_key}, got {actual_key}"
        )
    temporary_path.replace(path)
