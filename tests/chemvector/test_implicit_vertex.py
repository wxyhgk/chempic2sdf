"""Tests for implicit-only vertex batching."""

import unittest

from PIL import Image

from chemvector.pretrain_implicit_vertex import ImplicitVertexCollator


class ImplicitVertexTests(unittest.TestCase):
    def test_visible_atoms_are_removed_from_targets(self) -> None:
        sample = {
            "key": "sample",
            "image": Image.new("RGB", (768, 768), "white"),
            "render": {
                "width": 768,
                "height": 768,
                "atoms": [
                    {"atom_id": 0, "image_x": 100, "image_y": 100},
                    {"atom_id": 1, "image_x": 200, "image_y": 200},
                ],
            },
            "chemmol": {
                "atoms": [
                    {"element": "C", "aromatic": False, "implicit_h_count": 0},
                    {"element": "N", "aromatic": False, "implicit_h_count": 0},
                ]
            },
        }
        batch = ImplicitVertexCollator()([sample])
        self.assertEqual(batch["anchor_mask"].tolist(), [[True, False]])


if __name__ == "__main__":
    unittest.main()
