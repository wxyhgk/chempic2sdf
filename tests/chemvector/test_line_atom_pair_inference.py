from __future__ import annotations

import unittest

import torch

from chemvector.line_atom_pair_inference import append_line_atom_pair_proposals
from chemvector.line_segment_rescue import AtomPairLineCandidate


class LineAtomPairInferenceTest(unittest.TestCase):
    def test_appended_candidate_has_fixed_incidence_and_shifted_threshold(self) -> None:
        proposals = {
            "atom_geometry": torch.tensor([[[0.1, 0.2], [0.8, 0.7]]]),
            "bond_mask": torch.tensor([[True]]),
            "bond_features": torch.zeros((1, 1, 4)),
            "bond_endpoints": torch.zeros((1, 1, 2, 2)),
            "bond_types": torch.zeros((1, 1, 8)),
            "bond_scores": torch.zeros((1, 1)),
        }
        set_output = {
            "bond_presence_logits": torch.tensor([[1.0]]),
            "incidence_logits": torch.zeros((1, 1, 2, 2)),
        }
        line_output = {
            "presence_logits": torch.tensor([0.75]),
            "display_logits": torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
            "chemical_logits": torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        }
        candidate = AtomPairLineCandidate(
            atom_pair=(0, 1),
            representative_segment_pixels=((76.8, 153.6), (614.4, 537.6)),
            support_segment_indices=(0, 1),
            endpoint_error_pixels=1.0,
            angle_error_degrees=0.0,
            maximum_support_length_pixels=660.0,
        )

        combined, output, diagnostics = append_line_atom_pair_proposals(
            proposals,
            set_output,
            line_output,
            [(candidate,)],
            torch.tensor([0]),
            line_threshold=0.5,
        )

        self.assertEqual(combined["bond_mask"].tolist(), [[True, True]])
        self.assertAlmostEqual(float(output["bond_presence_logits"][0, 1]), 0.25)
        self.assertEqual(
            output["incidence_logits"][0, 1].argmax(-1).tolist(), [0, 1]
        )
        torch.testing.assert_close(
            combined["bond_endpoints"][0, 1], proposals["atom_geometry"][0, :, :2]
        )
        self.assertEqual(diagnostics[0]["selected"], 1)


if __name__ == "__main__":
    unittest.main()
