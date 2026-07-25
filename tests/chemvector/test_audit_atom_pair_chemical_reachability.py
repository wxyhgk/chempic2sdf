from __future__ import annotations

import unittest

import torch

from chemvector.audit_atom_pair_chemical_reachability import (
    ABSENT_STATE,
    bounded_typed_reachable,
    chemical_confidence_margin,
    target_typed_states,
    typed_candidate_indices,
    typed_edit_kinds,
    typed_states,
)


class AtomPairChemicalReachabilityTest(unittest.TestCase):
    def test_states_combine_presence_and_chemical_type(self) -> None:
        logits = torch.tensor(
            ((3.0, 1.0, 0.0, -1.0), (0.0, 2.0, 1.0, -1.0))
        )

        states = typed_states(torch.tensor((True, False)), logits)
        targets = target_typed_states(
            torch.tensor((True, False)), torch.tensor((1, 0))
        )

        self.assertEqual(states.tolist(), [0, ABSENT_STATE])
        self.assertEqual(targets.tolist(), [1, ABSENT_STATE])

    def test_edit_kinds_cover_delete_add_and_retype(self) -> None:
        current = torch.tensor((0, ABSENT_STATE, 1, 2))
        expected = torch.tensor((ABSENT_STATE, 3, 0, 2))

        self.assertEqual(
            typed_edit_kinds(current, expected),
            [(0, "delete"), (1, "add"), (2, "retype")],
        )

    def test_candidates_separate_presence_and_type_uncertainty(self) -> None:
        base = torch.tensor((2.0, 0.1, -0.2, -3.0, 1.0, -0.1))
        selected = torch.tensor((True, True, False, False, True, False))
        chemical = torch.tensor(
            (
                (5.0, 0.0, 0.0, 0.0),
                (1.0, 0.9, 0.0, 0.0),
                (2.0, 0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.95, 0.0),
                (2.0, 0.0, 0.0, 0.0),
            )
        )

        delete, add, retype = typed_candidate_indices(
            base,
            chemical,
            selected,
            delete_candidates=2,
            add_candidates=2,
            retype_candidates=2,
        )

        self.assertEqual(delete, {1, 4})
        self.assertEqual(add, {2, 5})
        self.assertEqual(retype, {1, 4})
        self.assertTrue(
            torch.allclose(
                chemical_confidence_margin(chemical)[torch.tensor((1, 4))],
                torch.tensor((0.1, 0.05)),
                atol=1e-6,
            )
        )

    def test_bounded_reachability_requires_every_typed_candidate(self) -> None:
        candidates = ({2}, {3}, {4})
        self.assertTrue(
            bounded_typed_reachable(
                [(2, "delete"), (4, "retype")],
                candidates,
                maximum_edits=2,
            )
        )
        self.assertFalse(
            bounded_typed_reachable(
                [(2, "delete"), (5, "add")],
                candidates,
                maximum_edits=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
