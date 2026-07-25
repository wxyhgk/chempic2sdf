import unittest

from chemvector.train_spatial_atom_object_full import (
    _atom_target_cells,
    freeze_atom_object_features,
    spatial_cell_collision_count,
)
from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
)


class FullSpatialAtomObjectTrainingTests(unittest.TestCase):
    def test_collision_count_detects_two_atoms_in_one_stride4_cell(self) -> None:
        sample = {
            "layers": {
                "atoms": [
                    {"atom_id": 0, "center": [8.1, 8.1]},
                    {"atom_id": 1, "center": [10.9, 10.9]},
                    {"atom_id": 2, "center": [20.0, 20.0]},
                ],
                "bonds": [],
                "width": 768,
                "height": 768,
            }
        }

        self.assertEqual(spatial_cell_collision_count(sample), 1)

    def test_collision_count_matches_q1023_target_cell_boundary(self) -> None:
        sample = {
            "render": {
                "atoms": [
                    {"quantized_x": 20, "quantized_y": 20},
                    {"quantized_x": 21, "quantized_y": 21},
                    {"quantized_x": 40, "quantized_y": 40},
                ]
            }
        }

        self.assertEqual(_atom_target_cells(sample), [(3, 3), (3, 3), (7, 7)])
        self.assertEqual(spatial_cell_collision_count(sample), 1)

    def test_raw_pixels_are_quantized_before_target_cell_assignment(self) -> None:
        sample = {
            "layers": {
                "atoms": [
                    {"atom_id": 0, "center": [6.0, 6.0]},
                    {"atom_id": 1, "center": [7.0, 7.0]},
                ],
                "bonds": [],
                "width": 768,
                "height": 768,
            }
        }

        self.assertEqual(spatial_cell_collision_count(sample), 1)

    def test_feature_stable_mode_only_freezes_proposal_feature_modules(self) -> None:
        head = SpatialAtomObjectDetector(SpatialAtomObjectConfig())

        freeze_atom_object_features(head)

        self.assertFalse(any(p.requires_grad for p in head.input_projection.parameters()))
        self.assertFalse(any(p.requires_grad for p in head.context.parameters()))
        self.assertTrue(all(p.requires_grad for p in head.heatmap.parameters()))
        self.assertTrue(all(p.requires_grad for p in head.anchor_offset.parameters()))


if __name__ == "__main__":
    unittest.main()
