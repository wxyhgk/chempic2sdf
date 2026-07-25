import unittest

import torch

from chemvector.pretrain_chemscene_graph_corrector import (
    build_graph_targets,
    element_id,
)


class ChemSceneGraphCorrectorTargetsTest(unittest.TestCase):
    def test_element_groups_are_open_string_fallbacks(self) -> None:
        self.assertEqual(element_id("C", 0), 0)
        self.assertNotEqual(element_id("Cl", 1), element_id("R17", 1))

    def test_bond_object_targets_include_types(self) -> None:
        sample = {
            "layers": {
                "width": 100,
                "height": 100,
                "atoms": [
                    {"atom_id": 0, "center": [10.0, 10.0]},
                    {"atom_id": 1, "center": [90.0, 10.0]},
                ],
                "bonds": [
                    {
                        "begin": [10.0, 10.0],
                        "end": [90.0, 10.0],
                        "begin_atom": 0,
                        "end_atom": 1,
                        "display_type": "DOUBLE",
                        "chemical_type": "DOUBLE",
                    }
                ],
            }
        }
        targets = build_graph_targets(
            torch.tensor(((0.1, 0.1), (0.9, 0.1))),
            [{"endpoints": torch.tensor(((0.1, 0.1), (0.9, 0.1)))}],
            sample,
        )
        self.assertEqual(targets["edge_targets"].tolist(), [1.0])
        self.assertEqual(targets["display_targets"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
