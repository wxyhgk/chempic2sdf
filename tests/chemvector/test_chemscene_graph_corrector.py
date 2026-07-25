import unittest

import torch

from chemvector.chemscene_graph_corrector import (
    ChemSceneGraphCorrector,
    ChemSceneGraphCorrectorConfig,
    graph_corrector_loss,
    nearest_bond_pairs,
)


class ChemSceneGraphCorrectorTest(unittest.TestCase):
    def test_forward_and_loss(self) -> None:
        model = ChemSceneGraphCorrector(
            ChemSceneGraphCorrectorConfig(d_model=24, layers=2, element_dim=8)
        )
        points = torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.1)))
        endpoints = torch.tensor(
            (((0.1, 0.1), (0.5, 0.5)), ((0.5, 0.5), (0.9, 0.1)))
        )
        output = model(
            points,
            torch.tensor((0, 1, 0)),
            torch.tensor((0, 1, 0)),
            endpoints,
            torch.tensor((0.9, 0.8)),
            torch.tensor((0, 1)),
            torch.tensor((0, 1)),
        )
        self.assertEqual(output["edge_keep_logits"].shape, (2,))
        self.assertEqual(output["display_logits"].shape, (2, 4))
        losses = graph_corrector_loss(
            output,
            torch.tensor((1, 1, 1)),
            torch.tensor((1, 0)),
            torch.tensor((0, 1)),
            torch.tensor((0, 1)),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_nearest_pairs_preserve_long_edges(self) -> None:
        points = torch.tensor(((0.1, 0.1), (0.9, 0.9)))
        pairs, _ = nearest_bond_pairs(
            points, torch.tensor(([[0.1, 0.1], [0.9, 0.9]],))
        )
        self.assertEqual(pairs.tolist(), [[0, 1]])


if __name__ == "__main__":
    unittest.main()
