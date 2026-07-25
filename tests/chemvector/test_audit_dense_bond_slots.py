import unittest

from chemvector.audit_dense_bond_slots import maximum_cell_occupancy


class DenseBondSlotAuditTest(unittest.TestCase):
    def test_parallel_bonds_with_same_center_use_multiple_slots(self) -> None:
        render = {
            "width": 768,
            "height": 768,
            "bonds": [
                {
                    "begin_image_x": 100 + offset,
                    "begin_image_y": 100,
                    "end_image_x": 140 - offset,
                    "end_image_y": 100,
                }
                for offset in (0, 1, 2, 3)
            ],
        }

        self.assertEqual(maximum_cell_occupancy(render, 192), 4)


if __name__ == "__main__":
    unittest.main()
