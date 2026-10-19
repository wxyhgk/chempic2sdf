from rdkit import Chem
from rdkit.Geometry import Point3D

from Data.sdf.build_chemvision_layers import explicit_atom_text
from chemvector.audit_text_failure_label_probe import visible_svg_labels


def test_linear_carbon_glyph_is_not_chemical_hydrogen_expansion():
    mol = Chem.MolFromSmiles("CCC")
    conformer = Chem.Conformer(3)
    for index in range(3):
        conformer.SetAtomPosition(index, Point3D(index * 1.5, 0, 0))
    mol.AddConformer(conformer)
    labels = visible_svg_labels(mol, {})
    assert labels[1] == ["C"]
    assert mol.GetAtomWithIdx(1).GetTotalNumHs() == 2
    # This is the current builder's chemical fallback, not a correct OCR target.
    assert explicit_atom_text({"element": "C", "implicit_h_count": 2}) == "CH2"
