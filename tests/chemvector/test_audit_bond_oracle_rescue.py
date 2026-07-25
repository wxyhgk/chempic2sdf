import unittest

import torch

from chemvector.audit_bond_oracle_rescue import oracle_select_objects


class BondOracleRescueTest(unittest.TestCase):
    def test_oracle_selects_one_object_per_true_pair(self) -> None:
        sample = {
            "key": "x",
            "layers": {
                "width": 100,
                "height": 100,
                "atoms": [
                    {"atom_id": 0, "center": [10.0, 50.0]},
                    {"atom_id": 1, "center": [50.0, 50.0]},
                    {"atom_id": 2, "center": [90.0, 50.0]},
                ],
                "bonds": [
                    {"begin_atom": 0, "end_atom": 1, "begin": [10.0, 50.0], "end": [50.0, 50.0], "display_type": "SINGLE", "chemical_type": "SINGLE", "direction": "NONE"},
                    {"begin_atom": 1, "end_atom": 2, "begin": [50.0, 50.0], "end": [90.0, 50.0], "display_type": "DOUBLE", "chemical_type": "DOUBLE", "direction": "NONE"},
                ],
            },
        }
        example = {
            "endpoints": torch.tensor(
                (
                    ((0.1, 0.5), (0.5, 0.5)),
                    ((0.11, 0.5), (0.49, 0.5)),
                    ((0.5, 0.5), (0.9, 0.5)),
                    ((0.1, 0.1), (0.9, 0.9)),
                )
            ),
            "scores": torch.tensor((0.8, 0.7, 0.6, 0.9)),
            "display_ids": torch.tensor((0, 0, 1, 0)),
            "chemical_ids": torch.tensor((0, 0, 1, 0)),
            "direction_ids": torch.zeros(4, dtype=torch.long),
        }
        selected, covered = oracle_select_objects(sample, example, threshold=0.2)
        self.assertTrue(covered)
        self.assertEqual(len(selected), 2)
        self.assertTrue(torch.equal(selected[0]["endpoints"], example["endpoints"][0]))


if __name__ == "__main__":
    unittest.main()
