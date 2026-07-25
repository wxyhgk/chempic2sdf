import unittest

from rdkit import Chem

from chemvector.benchmark_joint_mr_tadf import (
    SceneCompilationFailure,
    _gt_scene,
    _summary,
)


class BenchmarkJointMrTadfTests(unittest.TestCase):
    def test_scene_compilation_failure_preserves_diagnostics(self) -> None:
        error = SceneCompilationFailure("invalid", {"atom_objects": []}, {"joint": True})

        self.assertEqual(error.scene, {"atom_objects": []})
        self.assertEqual(error.diagnostics, {"joint": True})

    def test_gt_scene_preserves_atom_chemistry_and_coordinates(self) -> None:
        mol = Chem.MolFromSmiles("BN")
        annotation = {
            "molecule_id": "test",
            "width": 768,
            "height": 768,
            "atoms": [
                {"atom_id": 0, "image_x": 100.0, "image_y": 200.0},
                {"atom_id": 1, "image_x": 300.0, "image_y": 400.0},
            ],
            "bonds": [
                {
                    "bond_id": 0,
                    "begin_atom": 0,
                    "end_atom": 1,
                    "bond_type": "SINGLE",
                }
            ],
        }

        scene = _gt_scene(annotation, mol)

        self.assertEqual([atom["chemical"]["element"] for atom in scene["atom_objects"]], ["B", "N"])
        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "SINGLE")
        self.assertEqual(scene["atom_objects"][0]["role"], "TEXT")

    def test_summary_groups_difficulty(self) -> None:
        summary = _summary(
            [
                {
                    "difficulty": "long",
                    "sdf_valid": True,
                    "atom_count_match": True,
                    "bond_count_match": True,
                    "nonstereo_structure_match": True,
                    "chemical_match": False,
                    "decode_path": "primary_joint",
                    "seconds": 0.1,
                }
            ]
        )

        self.assertEqual(summary["nonstereo_structure_match_rate"], 1.0)
        self.assertEqual(summary["difficulty_rates"]["long"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
