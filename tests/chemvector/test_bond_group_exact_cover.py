from __future__ import annotations

import unittest

from chemvector.bond_group_exact_cover import (
    BondGroupCandidate,
    BondGroupExactCoverValidationError,
    solve_bond_group_exact_cover,
)
from chemvector.bond_group_set import PARALLEL_2


def _candidate(
    kind: str,
    members: tuple[int, ...],
    score: float,
) -> BondGroupCandidate:
    return BondGroupCandidate(
        kind=kind,
        proposal_indices=members,
        score=score,
    )


def _partition(result) -> set[tuple[str, tuple[int, ...]]]:
    return {
        (candidate.kind, candidate.proposal_indices)
        for candidate in result.selected_candidates
    }


class BondGroupExactCoverTest(unittest.TestCase):
    def test_parallel_two_beats_two_singletons_and_ties_choose_fewer_groups(
        self,
    ) -> None:
        candidates = (
            _candidate("single_line", (0,), 1.0),
            _candidate("parallel_2", (0, 1), 2.0),
            _candidate("single_line", (1,), 1.0),
        )
        result = solve_bond_group_exact_cover(candidates, ("line", "line"))

        self.assertTrue(result.feasible)
        self.assertEqual(result.solver_status, "optimal")
        self.assertAlmostEqual(result.objective, 2.0)
        self.assertEqual(
            _partition(result),
            {("parallel_2", (0, 1))},
        )
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_parallel_three_is_one_group_over_three_lines(self) -> None:
        candidates = (
            _candidate("parallel_3", (0, 1, 2), 8.0),
            _candidate("single_line", (0,), 1.0),
            _candidate("single_line", (1,), 1.0),
            _candidate("single_line", (2,), 1.0),
        )
        result = solve_bond_group_exact_cover(candidates, ("line",) * 3)

        self.assertTrue(result.feasible)
        self.assertEqual(
            _partition(result),
            {("parallel_3", (0, 1, 2))},
        )

    def test_hashed_wedge_can_own_variable_number_of_line_proposals(self) -> None:
        candidates = (
            _candidate("hashed_wedge", (0, 1, 2, 3), 10.0),
            *(_candidate("single_line", (index,), 1.0) for index in range(4)),
        )
        result = solve_bond_group_exact_cover(candidates, ("line",) * 4)

        self.assertTrue(result.feasible)
        self.assertEqual(
            _partition(result),
            {("hashed_wedge", (0, 1, 2, 3))},
        )

    def test_triangle_and_quad_are_physical_singletons(self) -> None:
        candidates = (
            _candidate("solid_triangle", (4,), 3.0),
            _candidate("solid_quad", (9,), 4.0),
        )
        result = solve_bond_group_exact_cover(
            candidates,
            {4: "triangle", 9: "quad"},
        )

        self.assertTrue(result.feasible)
        self.assertEqual(_partition(result), {
            ("solid_triangle", (4,)),
            ("solid_quad", (9,)),
        })

    def test_group_type_can_be_used_instead_of_kind(self) -> None:
        candidate = BondGroupCandidate(
            group_type=PARALLEL_2,
            proposal_indices=(1, 0),
            score=2.0,
        )
        self.assertEqual(candidate.kind, "parallel_2")
        self.assertEqual(candidate.proposal_indices, (0, 1))

        result = solve_bond_group_exact_cover((candidate,), ("line", "line"))
        self.assertTrue(result.feasible)

    def test_known_count_can_choose_a_lower_scoring_feasible_partition(self) -> None:
        candidates = (
            _candidate("parallel_2", (0, 1), 5.0),
            _candidate("single_line", (0,), 1.0),
            _candidate("single_line", (1,), 1.0),
        )
        one_group = solve_bond_group_exact_cover(
            candidates,
            ("line", "line"),
            known_group_count=1,
        )
        two_groups = solve_bond_group_exact_cover(
            candidates,
            ("line", "line"),
            known_group_count=2,
        )

        self.assertEqual(_partition(one_group), {("parallel_2", (0, 1))})
        self.assertEqual(_partition(two_groups), {
            ("single_line", (0,)),
            ("single_line", (1,)),
        })

    def test_drop_allows_uncovered_proposals_without_returning_typed_drop(
        self,
    ) -> None:
        result = solve_bond_group_exact_cover(
            (_candidate("single_line", (0,), 1.0),),
            ("line", "line"),
            allow_drop=True,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(_partition(result), {("single_line", (0,))})
        self.assertEqual(result.dropped_proposal_indices, (1,))
        self.assertAlmostEqual(result.objective, 1.0)

    def test_drop_scores_can_enable_drop_and_change_primary_optimum(self) -> None:
        result = solve_bond_group_exact_cover(
            (_candidate("single_line", (0,), 1.0),),
            ("line",),
            drop_scores={0: 2.0},
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.selected_candidates, ())
        self.assertEqual(result.dropped_proposal_indices, (0,))
        self.assertAlmostEqual(result.objective, 2.0)

    def test_empty_candidate_pool_can_drop_every_active_proposal(self) -> None:
        result = solve_bond_group_exact_cover(
            (),
            ("line", "triangle"),
            known_group_count=0,
            allow_drop=True,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.selected_candidates, ())
        self.assertEqual(result.dropped_proposal_indices, (0, 1))
        self.assertAlmostEqual(result.objective, 0.0)

    def test_ties_prefer_fewer_drops_then_fewer_chemical_groups(self) -> None:
        candidates = (
            _candidate("parallel_2", (0, 1), 0.0),
            _candidate("single_line", (0,), 0.0),
            _candidate("single_line", (1,), 0.0),
            _candidate("single_line", (2,), 0.0),
        )
        result = solve_bond_group_exact_cover(
            candidates,
            ("line", "line", "line"),
            allow_drop=True,
        )

        self.assertEqual(result.dropped_proposal_indices, ())
        self.assertEqual(_partition(result), {
            ("parallel_2", (0, 1)),
            ("single_line", (2,)),
        })

    def test_known_count_counts_only_chemical_groups_when_drop_is_enabled(
        self,
    ) -> None:
        result = solve_bond_group_exact_cover(
            (_candidate("single_line", (0,), 1.0),),
            ("line", "line"),
            known_group_count=1,
            allow_drop=True,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(len(result.selected_candidates), 1)
        self.assertEqual(result.dropped_proposal_indices, (1,))

    def test_known_count_infeasible_returns_normal_solver_result(self) -> None:
        candidates = (
            _candidate("parallel_2", (0, 1), 5.0),
            _candidate("single_line", (0,), 1.0),
            _candidate("single_line", (1,), 1.0),
        )
        result = solve_bond_group_exact_cover(
            candidates,
            ("line", "line"),
            known_group_count=3,
        )

        self.assertFalse(result.feasible)
        self.assertIsNone(result.objective)
        self.assertEqual(result.selected_candidates, ())
        self.assertEqual(result.solver_status, "infeasible")
        self.assertEqual(result.solver_status_code, 2)

    def test_fully_covered_but_conflicting_pool_can_have_no_solution(self) -> None:
        candidates = (
            _candidate("parallel_2", (0, 1), 2.0),
            _candidate("parallel_2", (1, 2), 2.0),
        )
        result = solve_bond_group_exact_cover(candidates, ("line",) * 3)

        self.assertFalse(result.feasible)
        self.assertEqual(result.solver_status, "infeasible")

    def test_candidate_order_permutation_does_not_change_partition(self) -> None:
        candidates = (
            _candidate("single_line", (0,), 1.0),
            _candidate("single_line", (1,), 1.0),
            _candidate("parallel_2", (0, 1), 2.0),
            _candidate("single_line", (2,), 1.0),
        )
        baseline = solve_bond_group_exact_cover(candidates, ("line",) * 3)
        permuted = solve_bond_group_exact_cover(
            tuple(reversed(candidates)),
            ("line",) * 3,
        )

        self.assertEqual(_partition(baseline), _partition(permuted))
        self.assertAlmostEqual(baseline.objective, permuted.objective)

    def test_equal_score_equal_count_tie_is_stable_under_input_permutation(
        self,
    ) -> None:
        candidates = (
            _candidate("parallel_2", (0, 1), 1.0),
            _candidate("parallel_2", (2, 3), 1.0),
            _candidate("parallel_2", (0, 2), 1.0),
            _candidate("parallel_2", (1, 3), 1.0),
        )
        expected = None
        for ordering in (
            candidates,
            tuple(reversed(candidates)),
            (candidates[2], candidates[0], candidates[3], candidates[1]),
        ):
            result = solve_bond_group_exact_cover(ordering, ("line",) * 4)
            self.assertTrue(result.feasible)
            if expected is None:
                expected = _partition(result)
            self.assertEqual(_partition(result), expected)

    def test_selected_partition_owns_every_proposal_exactly_once(self) -> None:
        candidates = (
            _candidate("parallel_2", (0, 1), 5.0),
            _candidate("single_line", (0,), 1.0),
            _candidate("single_line", (1,), 1.0),
            _candidate("hashed_wedge", (2, 3, 4), 6.0),
            _candidate("single_line", (2,), 1.0),
            _candidate("single_line", (3,), 1.0),
            _candidate("single_line", (4,), 1.0),
        )
        result = solve_bond_group_exact_cover(candidates, ("line",) * 5)

        ownership = [0] * 5
        for candidate in result.selected_candidates:
            for index in candidate.proposal_indices:
                ownership[index] += 1
        self.assertEqual(ownership, [1, 1, 1, 1, 1])

    def test_rejects_illegal_parallel_and_hashed_member_counts(self) -> None:
        invalid = (
            ("parallel_2", (0,)),
            ("parallel_2", (0, 1, 2)),
            ("parallel_3", (0, 1)),
            ("parallel_3", (0, 1, 2, 3)),
            ("hashed_wedge", (0,)),
        )
        for kind, members in invalid:
            with self.subTest(kind=kind, members=members):
                with self.assertRaises(BondGroupExactCoverValidationError):
                    _candidate(kind, members, 1.0)

    def test_rejects_duplicate_out_of_bounds_uncovered_and_wrong_physical_type(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "same proposal",
        ):
            _candidate("parallel_2", (0, 0), 1.0)

        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "out of bounds",
        ):
            solve_bond_group_exact_cover(
                (_candidate("single_line", (2,), 1.0),),
                ("line", "line"),
            )

        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "not covered",
        ):
            solve_bond_group_exact_cover(
                (_candidate("single_line", (0,), 1.0),),
                ("line", "line"),
            )

        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "requires line",
        ):
            solve_bond_group_exact_cover(
                (_candidate("parallel_2", (0, 1), 1.0),),
                ("line", "triangle"),
            )

    def test_rejects_duplicate_semantic_candidate(self) -> None:
        candidates = (
            _candidate("single_line", (0,), 1.0),
            _candidate("single_line", (0,), 2.0),
        )
        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "Duplicate typed candidate",
        ):
            solve_bond_group_exact_cover(candidates, ("line",))

    def test_rejects_disagreeing_kind_and_group_type(self) -> None:
        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "disagree",
        ):
            BondGroupCandidate(
                kind="single_line",
                group_type=PARALLEL_2,
                proposal_indices=(0,),
                score=1.0,
            )

    def test_rejects_invalid_drop_configuration(self) -> None:
        candidate = _candidate("single_line", (0,), 1.0)
        for drop_scores in ({0: float("nan")}, {0: float("inf")}, {0: True}):
            with self.subTest(drop_scores=drop_scores):
                with self.assertRaises(BondGroupExactCoverValidationError):
                    solve_bond_group_exact_cover(
                        (candidate,),
                        ("line",),
                        allow_drop=True,
                        drop_scores=drop_scores,
                    )

        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "out of bounds",
        ):
            solve_bond_group_exact_cover(
                (candidate,),
                ("line",),
                allow_drop=True,
                drop_scores={1: 0.0},
            )

        with self.assertRaisesRegex(
            BondGroupExactCoverValidationError,
            "missing",
        ):
            solve_bond_group_exact_cover(
                (candidate,),
                ("line", "line"),
                allow_drop=True,
                drop_scores=(0.0,),
            )


if __name__ == "__main__":
    unittest.main()
