import unittest

import torch

from chemvector.ocr_utility_router import (
    OCRUtilityRouter,
    conservative_frontier,
    ocr_utility_features,
)


class OCRUtilityRouterTests(unittest.TestCase):
    def test_features_include_parseable_atom_state(self) -> None:
        values = ocr_utility_features("NH", -1.2, "N", -0.4)
        self.assertEqual(values["old_parseable"], 1.0)
        self.assertEqual(values["new_parseable"], 1.0)
        self.assertEqual(values["old_explicit_h"], 1.0)
        self.assertEqual(values["new_element_N"], 1.0)
        self.assertAlmostEqual(values["score_delta_new_minus_old"], 0.8)

    def test_router_rejects_wrong_feature_shape(self) -> None:
        model = OCRUtilityRouter(4, 8)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3))

    def test_zero_break_frontier_keeps_safe_rescue(self) -> None:
        frontier = conservative_frontier(
            torch.tensor((0.9, 0.8, 0.1)),
            ["new_rescue", "new_rescue", "new_break"],
            ["a", "b", "a"],
        )
        self.assertEqual(frontier["0"]["overall"]["new_rescue"], 2)
        self.assertEqual(frontier["0"]["overall"]["new_break"], 0)


if __name__ == "__main__":
    unittest.main()
