from __future__ import annotations

from collections import Counter
import unittest

import torch

from chemvector.audit_typed_candidate_exact_cover import (
    CandidateGeometryThresholds,
    TypedCandidateGroup,
    candidate_recall_counts,
    generate_typed_candidate_groups,
    generate_typed_candidate_groups_from_affinities,
    oracle_pair_affinities_from_targets,
    solve_typed_exact_cover,
)
from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
    SOLID_TRIANGLE,
)
from chemvector.primitive_set import LINE, TRIANGLE


def _line(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0 / 2047.0, y0 / 2047.0, x1 / 2047.0, y1 / 2047.0, 0, 0, 0, 0]


class TypedCandidateExactCoverAuditTest(unittest.TestCase):
    def test_parallel_singleton_and_polygon_candidates_need_no_gt_identity(self) -> None:
        geometry = torch.tensor(
            [
                _line(100, 100, 300, 100),
                _line(100, 120, 300, 120),
                _line(500, 100, 700, 100),
                _line(500, 120, 700, 120),
                _line(500, 140, 700, 140),
                [0.7, 0.7, 0.75, 0.7, 0.72, 0.75, 0, 0],
            ],
            dtype=torch.float32,
        )
        types = torch.tensor([LINE, LINE, LINE, LINE, LINE, TRIANGLE])
        usable = torch.ones(6, dtype=torch.bool)

        candidates = set(generate_typed_candidate_groups(geometry, types, usable))

        self.assertIn(TypedCandidateGroup(PARALLEL_2, (0, 1)), candidates)
        self.assertIn(TypedCandidateGroup(PARALLEL_3, (2, 3, 4)), candidates)
        self.assertIn(TypedCandidateGroup(SINGLE_LINE, (0,)), candidates)
        self.assertIn(TypedCandidateGroup(SOLID_TRIANGLE, (5,)), candidates)

    def test_far_or_perpendicular_lines_are_not_parallel_candidates(self) -> None:
        geometry = torch.tensor(
            [
                _line(100, 100, 300, 100),
                _line(100, 350, 300, 350),
                _line(200, 80, 200, 280),
            ],
            dtype=torch.float32,
        )
        candidates = set(
            generate_typed_candidate_groups(
                geometry,
                torch.full((3,), LINE),
                torch.ones(3, dtype=torch.bool),
            )
        )

        self.assertNotIn(TypedCandidateGroup(PARALLEL_2, (0, 1)), candidates)
        self.assertNotIn(TypedCandidateGroup(PARALLEL_2, (0, 2)), candidates)

    def test_hashed_corridor_emits_exact_ordered_contiguous_subset(self) -> None:
        geometry = torch.tensor(
            [
                _line(100, 100, 110, 100),
                _line(98, 120, 112, 120),
                _line(96, 140, 114, 140),
                _line(94, 160, 116, 160),
                _line(500, 500, 700, 500),
            ],
            dtype=torch.float32,
        )
        candidates = set(
            generate_typed_candidate_groups(
                geometry,
                torch.full((5,), LINE),
                torch.ones(5, dtype=torch.bool),
            )
        )

        self.assertIn(TypedCandidateGroup(HASHED_WEDGE, (0, 1, 2, 3)), candidates)
        self.assertIn(TypedCandidateGroup(HASHED_WEDGE, (1, 2, 3)), candidates)

    def test_recall_is_measured_only_after_candidates_exist(self) -> None:
        candidates = (
            TypedCandidateGroup(PARALLEL_2, (0, 1)),
            TypedCandidateGroup(SINGLE_LINE, (2,)),
        )
        targets = (
            TypedCandidateGroup(PARALLEL_2, (0, 1)),
            TypedCandidateGroup(HASHED_WEDGE, (2, 3)),
        )

        result = candidate_recall_counts(candidates, targets)

        self.assertEqual(result["expected"], Counter({PARALLEL_2: 1, HASHED_WEDGE: 1}))
        self.assertEqual(result["recalled"], Counter({PARALLEL_2: 1}))
        self.assertEqual(result["missing"], (targets[1],))

    def test_perfect_pair_oracle_still_uses_shared_candidate_generator(self) -> None:
        geometry = torch.tensor(
            [
                _line(100, 100, 110, 100),
                _line(98, 120, 112, 120),
                _line(96, 140, 114, 140),
                _line(500, 100, 700, 100),
                _line(500, 120, 700, 120),
                _line(900, 500, 1100, 500),
            ],
            dtype=torch.float32,
        )
        types = torch.full((6,), LINE)
        usable = torch.ones(6, dtype=torch.bool)
        targets = (
            TypedCandidateGroup(HASHED_WEDGE, (0, 1, 2)),
            TypedCandidateGroup(PARALLEL_2, (3, 4)),
            TypedCandidateGroup(SINGLE_LINE, (5,)),
        )

        affinities = oracle_pair_affinities_from_targets(targets, types, usable)
        candidates = set(
            generate_typed_candidate_groups_from_affinities(
                geometry,
                types,
                usable,
                affinities,
            )
        )

        self.assertTrue(set(targets).issubset(candidates))
        self.assertTrue(affinities[HASHED_WEDGE][0, 1])
        self.assertTrue(affinities[PARALLEL_2][3, 4])
        self.assertFalse(affinities[HASHED_WEDGE][0, 3])

    def test_typed_milp_finds_exact_cover_without_target_memberships(self) -> None:
        candidates = (
            TypedCandidateGroup(SINGLE_LINE, (0,)),
            TypedCandidateGroup(SINGLE_LINE, (1,)),
            TypedCandidateGroup(SINGLE_LINE, (2,)),
            TypedCandidateGroup(PARALLEL_2, (0, 1)),
        )

        result = solve_typed_exact_cover(
            candidates,
            (0, 1, 2),
            Counter({PARALLEL_2: 1, SINGLE_LINE: 1}),
        )

        self.assertTrue(result["feasible"])
        self.assertEqual(
            set(result["selected"]),
            {
                TypedCandidateGroup(PARALLEL_2, (0, 1)),
                TypedCandidateGroup(SINGLE_LINE, (2,)),
            },
        )

    def test_threshold_validation_rejects_invalid_geometry_gate(self) -> None:
        with self.assertRaises(ValueError):
            CandidateGeometryThresholds(parallel_max_angle_degrees=91.0)


if __name__ == "__main__":
    unittest.main()
