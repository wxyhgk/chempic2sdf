from __future__ import annotations

import unittest

import torch

from chemvector.line_segment_group_rescue import rescue_disconnected_atom_pair_groups
from chemvector.line_segment_rescue import AtomPairLineCandidate


def candidate(pair: tuple[int, int], support: int = 2) -> AtomPairLineCandidate:
    return AtomPairLineCandidate(
        atom_pair=pair,
        representative_segment_pixels=((0.0, 0.0), (10.0, 0.0)),
        support_segment_indices=tuple(range(support)),
        endpoint_error_pixels=1.0,
        angle_error_degrees=0.0,
        maximum_support_length_pixels=10.0,
    )


class LineSegmentGroupRescueTest(unittest.TestCase):
    def test_rescue_connects_components_without_adding_cycle(self) -> None:
        output = {
            "presence_logits": torch.tensor((2.0, 2.0, -3.0, -2.0)),
            "chemical_logits": torch.zeros((4, 4)),
        }
        groups = {
            "members": [[[0], [1], [2], [3]]],
            "atom_pairs": [[(0, 1), (2, 3), (1, 2), (0, 1)]],
        }

        rescued, diagnostics = rescue_disconnected_atom_pair_groups(
            output,
            groups,
            atom_presence_logits=torch.ones((1, 4)),
            atom_mask=torch.ones((1, 4), dtype=torch.bool),
            line_candidates=[(candidate((1, 2)), candidate((0, 1)))],
            atom_threshold=0.0,
            group_threshold=0.0,
        )

        self.assertGreater(float(rescued["presence_logits"][2]), 0.0)
        self.assertLess(float(rescued["presence_logits"][3]), 0.0)
        self.assertEqual(
            diagnostics[0]["rescued_groups"][0]["atom_pair"], [1, 2]
        )

    def test_single_lsd_boundary_is_not_enough(self) -> None:
        output = {"presence_logits": torch.tensor((-1.0,))}
        groups = {"members": [[[0]]], "atom_pairs": [[(0, 1)]]}

        rescued, diagnostics = rescue_disconnected_atom_pair_groups(
            output,
            groups,
            atom_presence_logits=torch.ones((1, 2)),
            atom_mask=torch.ones((1, 2), dtype=torch.bool),
            line_candidates=[(candidate((0, 1), support=1),)],
            atom_threshold=0.0,
            group_threshold=0.0,
        )

        self.assertEqual(float(rescued["presence_logits"][0]), -1.0)
        self.assertEqual(diagnostics[0]["supported_groups"], 0)


if __name__ == "__main__":
    unittest.main()
