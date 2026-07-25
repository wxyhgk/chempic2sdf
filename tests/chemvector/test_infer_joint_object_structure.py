import unittest

from chemvector.infer_joint_object_structure import (
    compile_carbon_placeholder,
    placeholder_scene,
)


class JointObjectStructureInferenceTests(unittest.TestCase):
    def test_placeholder_scene_contains_no_oracle_atom_chemistry(self) -> None:
        scene = placeholder_scene("upload", 768, 768)

        self.assertEqual(scene["canvas"], {"width": 768, "height": 768})
        self.assertEqual(scene["atom_objects"], [])
        self.assertEqual(scene["bond_objects"], [])

    def test_carbon_placeholder_demotes_aromatic_bonds(self) -> None:
        scene = placeholder_scene("upload", 768, 768)
        scene["atom_objects"] = [
            {
                "id": index,
                "anchor": [100 + index * 100, 100],
                "role": "IMPLICIT",
                "chemical": {
                    "element": "C",
                    "formal_charge": 0,
                    "isotope": 0,
                    "aromatic": True,
                    "explicit_h_count": 0,
                    "no_implicit": False,
                    "radical_electrons": 0,
                    "canonical_chiral_tag": "CHI_UNSPECIFIED",
                },
                "text": None,
                "incident_ports": [],
            }
            for index in range(2)
        ]
        scene["bond_objects"] = [
            {
                "id": 0,
                "atom_refs": [0, 1],
                "ports": [[100, 100], [200, 100]],
                "display_type": "DOUBLE",
                "chemical_type": "AROMATIC",
                "direction": "NONE",
                "stereo": "STEREONONE",
                "bbox": None,
            }
        ]

        repaired, mol, demoted = compile_carbon_placeholder(scene)

        self.assertEqual(demoted, 1)
        self.assertEqual(repaired["bond_objects"][0]["chemical_type"], "DOUBLE")
        self.assertEqual(mol.GetNumAtoms(), 2)


if __name__ == "__main__":
    unittest.main()
