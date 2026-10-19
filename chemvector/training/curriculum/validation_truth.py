"""Score-only molecular identity helpers; no model or dataset access."""

from rdkit import Chem


def canonical_molblock(block):
    if not block:
        return None
    molecule = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False, strictParsing=True)
    if molecule is None or not molecule.GetNumAtoms():
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def expected_identity(sample):
    """Accept exactly one original SDF record or the legacy graph truth."""
    has_sdf, has_graph = "target_molblock" in sample, "graph" in sample
    if has_sdf == has_graph:
        raise ValueError("Exactly one validation truth source is required")
    if has_sdf:
        block = sample["target_molblock"]
        if not isinstance(block, str) or not block.strip():
            raise ValueError("Invalid validation SDF truth")
        parts = block.split("$$$$")
        if len(parts) > 2 or (len(parts) == 2 and parts[1].strip()):
            raise ValueError("Exactly one validation SDF record is required")
        expected = canonical_molblock(parts[0])
        if expected is None:
            raise ValueError("Invalid validation SDF truth")
        return expected
    target = Chem.MolFromSmiles(sample["graph"]["canonical_smiles"])
    if target is None or not target.GetNumAtoms():
        raise ValueError("Invalid released validation truth")
    return Chem.MolToSmiles(target, canonical=True, isomericSmiles=True)
