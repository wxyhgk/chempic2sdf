import unittest

import torch

from chemvector.bond_expert_utility import combined_utility_features


def bond(score, begin, end, chemical=0, display=0):
    return {
        "score": score,
        "endpoints": torch.tensor((begin, end), dtype=torch.float32),
        "chemical": chemical,
        "display": display,
        "direction": 0,
    }


class BondExpertUtilityFeaturesTest(unittest.TestCase):
    def test_features_are_finite_and_capture_expert_disagreement(self) -> None:
        base = [bond(0.9, (0.1, 0.1), (0.5, 0.1))]
        expert = [
            bond(0.8, (0.1, 0.1), (0.5, 0.1), chemical=1, display=1),
            bond(0.7, (0.5, 0.1), (0.9, 0.1)),
        ]

        features = combined_utility_features(
            base, expert, style_probability=0.75
        )

        self.assertEqual(features["style_probability"], 0.75)
        self.assertEqual(features["base_selected_count"], 1.0)
        self.assertEqual(features["expert_selected_count"], 2.0)
        self.assertEqual(features["selected_count_difference"], 1.0)
        self.assertEqual(features["chemical_type_agreement"], 0.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
