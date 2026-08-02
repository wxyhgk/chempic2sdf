import unittest

import torch

from chemvector.audit_g3_p0j_composition_residuals import (
    gather_slot_scores,
    replace_with_mapped_gt_anchors,
    residual_interpretation,
)


class G3P0JCompositionResidualTests(unittest.TestCase):
    def test_gather_slot_scores_uses_predicted_identity(self) -> None:
        dense = {
            "presence_logits": torch.tensor(
                [[[[0.0, 1.0]], [[2.0, 3.0]]]]
            )
        }
        identities = torch.tensor([[[1, 0, 0], [0, 0, 1]]])
        scores = gather_slot_scores(
            dense, identities, torch.tensor([[True, True]])
        )

        torch.testing.assert_close(scores[0, 0], torch.tensor(2.0).sigmoid())
        torch.testing.assert_close(scores[0, 1], torch.tensor(1.0).sigmoid())

    def test_gt_anchor_oracle_preserves_predicted_query_order(self) -> None:
        atoms = {
            "atom_mask": torch.tensor([[True, True]]),
            "atom_anchors": torch.tensor([[[0.795, 0.50], [0.205, 0.50]]]),
        }
        targets = {
            "atom_mask": torch.tensor([[True, True]]),
            "atom_anchors": torch.tensor([[[0.20, 0.50], [0.80, 0.50]]]),
        }

        anchors = replace_with_mapped_gt_anchors(atoms, targets)

        torch.testing.assert_close(
            anchors, torch.tensor([[[0.80, 0.50], [0.20, 0.50]]])
        )

    def test_interpretation_isolates_geometry(self) -> None:
        modes = {
            "pred_anchor_p0i_score": {
                "gt_bond_count": {"typed_graph_exact_images": 26}
            },
            "pred_anchor_p0g_score": {
                "gt_bond_count": {"typed_graph_exact_images": 27}
            },
            "gt_anchor_p0i_score": {
                "gt_bond_count": {"typed_graph_exact_images": 32}
            },
            "gt_anchor_p0g_score": {
                "gt_bond_count": {"typed_graph_exact_images": 32}
            },
        }

        result = residual_interpretation(modes)

        self.assertEqual(result["primary_cause"], "predicted_anchor_geometry_shift")
        self.assertTrue(result["anchor_oracle_passed"])
        self.assertFalse(result["score_oracle_passed"])


if __name__ == "__main__":
    unittest.main()
