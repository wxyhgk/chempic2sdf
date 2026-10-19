from tools.run_pair_scale_trace_diagnostic import compare_prediction, scene_summary
from rdkit import Chem


def test_scene_summary_counts_aromatic_and_types():
    scene = {
        "atom_objects": [{"chemical": {"aromatic": True}}, {"chemical": {"aromatic": False}}],
        "bond_objects": [{"chemical_type": "AROMATIC"}, {"chemical_type": "SINGLE"}],
    }
    result = scene_summary(scene)
    assert result["atoms"] == 2
    assert result["aromatic_atoms"] == 1
    assert result["aromatic_bonds"] == 1
    assert result["bond_types"]["SINGLE"] == 1


def test_compare_prediction_requires_saved_molblock_and_metrics():
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CC"))
    saved = {"predicted_smiles": "CC", "valid": True, "exact": True,
             "fallback": False, "error": None, "expected_smiles": "CC",
             "molblock": block}
    result = compare_prediction({"molblock": block, "fallback": False, "error": None}, saved)
    assert result["matches_saved"]


def test_compare_prediction_rejects_metric_mismatch():
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CC"))
    saved = {"predicted_smiles": "CC", "valid": True, "exact": True,
             "fallback": False, "error": None, "expected_smiles": "CC",
             "molblock": block}
    result = compare_prediction({"molblock": block, "fallback": True, "error": None}, saved)
    assert not result["matches_saved"]
