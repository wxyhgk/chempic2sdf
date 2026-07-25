from __future__ import annotations

import unittest

import torch

from chemvector.line_atom_pair_head import (
    LineAtomPairHead,
    LineAtomPairHeadConfig,
    build_line_atom_pair_features,
    line_atom_pair_loss,
)
from chemvector.line_segment_rescue import AtomPairLineCandidate


def candidate(pair: tuple[int, int]) -> AtomPairLineCandidate:
    return AtomPairLineCandidate(
        atom_pair=pair,
        representative_segment_pixels=((10.0, 20.0), (90.0, 60.0)),
        support_segment_indices=(0, 1),
        endpoint_error_pixels=2.0,
        angle_error_degrees=1.0,
        maximum_support_length_pixels=89.4,
    )


class LineAtomPairHeadTest(unittest.TestCase):
    def test_feature_builder_is_atom_order_invariant(self) -> None:
        config = LineAtomPairHeadConfig(feature_channels=4, hidden_dim=8)
        feature_map = torch.randn(1, 4, 16, 16)
        atom_features = torch.randn(1, 2, 4)
        atom_anchors = torch.tensor([[[10.0 / 768.0, 20.0 / 768.0], [90.0 / 768.0, 60.0 / 768.0]]])

        forward, _ = build_line_atom_pair_features(
            feature_map,
            atom_features,
            atom_anchors,
            [(candidate((0, 1)),)],
            config=config,
        )
        reverse, _ = build_line_atom_pair_features(
            feature_map,
            atom_features,
            atom_anchors,
            [(candidate((1, 0)),)],
            config=config,
        )

        torch.testing.assert_close(forward, reverse)
        self.assertEqual(forward.shape, (1, config.input_dim))

    def test_constant_visual_features_do_not_encode_absolute_bond_length(self) -> None:
        config = LineAtomPairHeadConfig(feature_channels=4, hidden_dim=8)
        feature_map = torch.ones(1, 4, 16, 16)
        atom_features = torch.randn(1, 2, 4)

        def features(end_x: float) -> torch.Tensor:
            anchors = torch.tensor(
                [[[10.0 / 768.0, 20.0 / 768.0], [end_x / 768.0, 20.0 / 768.0]]]
            )
            item = AtomPairLineCandidate(
                atom_pair=(0, 1),
                representative_segment_pixels=((10.0, 20.0), (end_x, 20.0)),
                support_segment_indices=(0, 1),
                endpoint_error_pixels=2.0,
                angle_error_degrees=1.0,
                maximum_support_length_pixels=end_x - 10.0,
            )
            result, _ = build_line_atom_pair_features(
                feature_map,
                atom_features,
                anchors,
                [(item,)],
                config=config,
            )
            return result

        torch.testing.assert_close(features(90.0), features(490.0))

    def test_head_and_loss_are_finite(self) -> None:
        config = LineAtomPairHeadConfig(feature_channels=4, hidden_dim=8)
        head = LineAtomPairHead(config)
        output = head(torch.randn(3, config.input_dim))
        losses = line_atom_pair_loss(
            output,
            torch.tensor((True, False, True)),
            torch.tensor((0, 0, 1)),
            torch.tensor((0, 0, 1)),
        )

        self.assertEqual(output["presence_logits"].shape, (3,))
        self.assertTrue(torch.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
