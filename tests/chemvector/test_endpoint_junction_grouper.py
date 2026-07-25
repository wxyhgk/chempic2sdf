import unittest

import torch

from chemvector.endpoint_junction_grouper import (
    EndpointJunctionGrouper,
    EndpointJunctionGrouperConfig,
    VisualEndpointJunctionGrouper,
    VisualEndpointJunctionGrouperConfig,
    decode_endpoint_groups,
    endpoint_pair_features,
    sample_aligned_endpoint_ink_patches,
)


class EndpointJunctionGrouperTest(unittest.TestCase):
    def test_features_and_constrained_group_decode(self) -> None:
        objects = [
            {"endpoints": torch.tensor(((0.1, 0.5), (0.45, 0.5))), "score": 0.9},
            {"endpoints": torch.tensor(((0.55, 0.5), (0.9, 0.5))), "score": 0.8},
        ]
        pairs, features = endpoint_pair_features(objects)
        model = EndpointJunctionGrouper(
            EndpointJunctionGrouperConfig(hidden_dim=16)
        )
        self.assertEqual(model(features).shape, (len(pairs),))
        logits = torch.full((len(pairs),), -10.0)
        target_pair = torch.where((pairs == torch.tensor((1, 2))).all(-1))[0]
        logits[target_pair] = 10.0
        groups = decode_endpoint_groups(
            logits, pairs, 4, threshold=0.5
        )
        self.assertEqual(groups.tolist(), [0, 1, 1, 2])
        groups = decode_endpoint_groups(
            logits, pairs, 4, threshold=0.5, linkage="single"
        )
        self.assertEqual(groups.tolist(), [0, 1, 1, 2])

    def test_aligned_endpoint_patches_and_visual_model(self) -> None:
        pixels = torch.ones((1, 1, 32, 32))
        pixels[:, :, 16, 8:25] = -1.0
        endpoints = torch.tensor([[[8.0 / 32.0, 16.0 / 32.0], [24.0 / 32.0, 16.0 / 32.0]]])
        patches = sample_aligned_endpoint_ink_patches(
            pixels, endpoints, patch_size=16
        )
        self.assertEqual(patches.shape, (2, 1, 16, 16))
        self.assertGreater(float(patches[0, 0, 7:9, 12:].max()), 0.5)
        self.assertGreater(float(patches[1, 0, 7:9, 12:].max()), 0.5)

        model = VisualEndpointJunctionGrouper(
            VisualEndpointJunctionGrouperConfig(
                patch_embedding_dim=8,
                hidden_dim=16,
                base_channels=4,
            )
        )
        geometry = torch.zeros((3, 15))
        logits = model(
            geometry,
            patches[[0, 1, 0]],
            patches[[1, 0, 1]],
        )
        self.assertEqual(logits.shape, (3,))
        embeddings = model.encode_patches(patches)
        embedded_logits = model.score_embeddings(
            geometry,
            embeddings[[0, 1, 0]],
            embeddings[[1, 0, 1]],
        )
        torch.testing.assert_close(logits, embedded_logits)


if __name__ == "__main__":
    unittest.main()
