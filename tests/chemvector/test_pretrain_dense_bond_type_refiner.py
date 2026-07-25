import unittest

from PIL import Image
import torch

from chemvector.pretrain_dense_bond_type_refiner import (
    PositiveBondTypeCollator,
    class_balancing_weights,
)


def sample() -> dict:
    return {
        "key": "example",
        "image": Image.new("RGB", (768, 768), "white"),
        "layers": {
            "width": 768,
            "height": 768,
            "bonds": [
                {
                    "begin": [76.8, 153.6],
                    "end": [230.4, 307.2],
                    "display_type": "DOUBLE",
                    "chemical_type": "AROMATIC",
                },
                {
                    "begin": [384.0, 384.0],
                    "end": [614.4, 614.4],
                    "display_type": "SINGLE",
                    "chemical_type": "SINGLE",
                },
            ],
        },
    }


class PositiveBondTypeCollatorTest(unittest.TestCase):
    def test_collator_packs_only_true_bonds(self) -> None:
        batch = PositiveBondTypeCollator()([sample()])
        self.assertEqual(batch["points"].shape, (1, 4, 2))
        self.assertTrue(
            torch.equal(batch["pair_indices"][0], torch.tensor(((0, 1), (2, 3))))
        )
        self.assertEqual(batch["display_targets"][0].tolist(), [1, 0])
        self.assertEqual(batch["chemical_targets"][0].tolist(), [3, 0])

    def test_class_weights_are_finite_for_absent_rare_classes(self) -> None:
        weights = class_balancing_weights([sample()], "display_type", 0.5, 4.0)
        self.assertEqual(weights.shape, (4,))
        self.assertTrue(torch.isfinite(weights).all())


if __name__ == "__main__":
    unittest.main()
