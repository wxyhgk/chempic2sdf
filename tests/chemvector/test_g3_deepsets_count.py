"""Tests for the independent G3 DeepSets Count model."""

import unittest

import torch

from chemvector.components.atoms.objects.g3_deepsets_count import (
    G3DeepSetsCount,
    g3_deepsets_count_features,
)


class G3DeepSetsCountTests(unittest.TestCase):
    def test_feature_and_model_shapes(self) -> None:
        output = {
            "existence_logits": torch.randn(2, 128),
            "pointer_logits": torch.randn(2, 128, 128),
        }
        query, candidate, global_features = g3_deepsets_count_features(
            output,
            output,
            torch.randn(2, 128, 32),
            torch.randn(2, 128),
            torch.randn(2, 128),
            torch.arange(128).float().repeat(2, 1),
        )
        self.assertEqual(tuple(query.shape), (2, 128, 8))
        self.assertEqual(tuple(candidate.shape), (2, 128, 41))
        self.assertEqual(tuple(global_features.shape), (2, 256))
        logits = G3DeepSetsCount()(query, candidate, global_features)
        self.assertEqual(tuple(logits.shape), (2, 129))

    def test_candidate_permutation_preserves_set_aggregate(self) -> None:
        model = G3DeepSetsCount().eval()
        query = torch.randn(2, 128, 8)
        candidate = torch.randn(2, 128, 41)
        global_features = torch.randn(2, 256)
        permutation = torch.randperm(128)
        with torch.no_grad():
            expected = model(query, candidate, global_features)
            current = model(query, candidate[:, permutation], global_features)
        self.assertTrue(torch.allclose(expected, current, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
