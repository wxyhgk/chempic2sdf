import unittest

import torch

from chemvector.pretrain_junction_bond_assembler import build_assembler_targets


class JunctionBondAssemblerTargetsTest(unittest.TestCase):
    def sample(self) -> dict:
        return {
            "layers": {
                "width": 100,
                "height": 100,
                "atoms": [
                    {"atom_id": 4, "center": [20.0, 20.0]},
                    {"atom_id": 9, "center": [80.0, 20.0]},
                    {"atom_id": 12, "center": [50.0, 80.0]},
                ],
                "bonds": [
                    {
                        "begin": [25.0, 20.0],
                        "end": [75.0, 20.0],
                        "begin_atom": 4,
                        "end_atom": 9,
                    },
                    {
                        "begin": [80.0, 20.0],
                        "end": [50.0, 80.0],
                        "begin_atom": 9,
                        "end_atom": 12,
                    },
                ],
            }
        }

    def test_maps_visual_line_endpoints_to_atom_center_junctions(self) -> None:
        points = torch.tensor(((0.2, 0.2), (0.8, 0.2), (0.5, 0.8)))
        objects = [
            {"endpoints": torch.tensor(((0.25, 0.2), (0.75, 0.2)))},
            {"endpoints": torch.tensor(((0.5, 0.8), (0.8, 0.2)))},
        ]
        targets, diagnostics = build_assembler_targets(points, objects, self.sample())
        self.assertEqual(targets.tolist(), [[0, 1], [2, 1]])
        self.assertEqual(diagnostics["matched_bond_objects"], 2)

    def test_false_object_and_missing_junction_use_null(self) -> None:
        points = torch.tensor(((0.2, 0.2), (0.8, 0.2)))
        objects = [
            {"endpoints": torch.tensor(((0.25, 0.2), (0.75, 0.2)))},
            {"endpoints": torch.tensor(((0.05, 0.9), (0.95, 0.9)))},
        ]
        targets, _ = build_assembler_targets(points, objects, self.sample())
        self.assertEqual(targets.tolist(), [[0, 1], [2, 2]])


if __name__ == "__main__":
    unittest.main()
