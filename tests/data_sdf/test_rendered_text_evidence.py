"""Synthetic renderer checks; no training or evaluation corpus is opened."""

import ast
from pathlib import Path

from rdkit import Chem
from rdkit.Geometry import Point3D

from Data.sdf.rendered_text_evidence import no_freetype_text_diagnostic


def test_diagnostic_exposes_visible_carbon_without_claiming_source_evidence():
    mol = Chem.MolFromSmiles("CCC")
    conformer = Chem.Conformer(3)
    for index in range(3):
        conformer.SetAtomPosition(index, Point3D(index * 1.5, 0, 0))
    mol.AddConformer(conformer)
    before = Chem.MolToMolBlock(mol)
    result = no_freetype_text_diagnostic(mol, {})
    assert result["labels"][1] == ["C"]
    assert mol.GetAtomWithIdx(1).GetTotalNumHs() == 2
    assert result["evidence_kind"] == "nofreetype_diagnostic"
    assert result["same_source_raster"] is False
    assert result["revision_authorized"] is False
    assert Chem.MolToMolBlock(mol) == before


def test_data_helpers_do_not_import_model_or_audit_modules():
    for path in ("Data/sdf/rendered_text_evidence.py", "Data/sdf/visible_text_revision.py"):
        tree = ast.parse(Path(path).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(name.startswith("chemvector") or "audit" in name for name in imports)
