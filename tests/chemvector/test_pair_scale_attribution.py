from rdkit import Chem

from tools.audit_pair_scale_attribution import _error_category, _mol_features, compare


def molblock(atom_line: str, bond_lines: str, atoms: int = 2, bonds: int = 1) -> str:
    return "\n  RDKit\n\n  %d %d  0  0  0  0  0  0  0  0999 V2000\n%s\n%s\nM  END\n" % (
        atoms, bonds, atom_line, bond_lines)


def record(**values):
    base = dict(key="full:a", domain="full_molecule", valid=True, exact=False,
                fallback=False, predicted_smiles="CC", expected_smiles="CC",
                molblock=molblock(
                    "    0.0 0.0 0.0 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
                    "    1.0 0.0 0.0 C   0  0  0  0  0  0  0  0  0  0  0  0",
                    "  1  2  1  0"), error=None)
    base.update(values)
    return base


def test_mol_features_count_components_and_aromaticity():
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("c1ccccc1"))
    features = _mol_features(block)
    assert features["parsed"]
    assert features["aromatic_atoms"] == 6
    assert features["aromatic_bonds"] == 6
    assert features["components"] == 1


def test_compare_classifies_broken_aromatic_and_new_fallback():
    before = record(exact=True, predicted_smiles="c1ccccc1", molblock=None)
    after = record(exact=False, fallback=True, valid=False, molblock=None,
                   error="Explicit valence for atom # 4 C, 5")
    result = compare(before, after)
    assert {"broken_exact", "new_fallback", "became_invalid", "error_category_changed"} <= set(result["categories"])
    assert _error_category(after["error"]) == "explicit_valence"


def test_compare_requires_matching_keys():
    before = record()
    after = record(key="full:b")
    try:
        compare(before, after)
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("mismatched saved keys must be rejected")
