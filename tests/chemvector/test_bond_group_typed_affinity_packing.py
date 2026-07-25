from __future__ import annotations

import unittest

import torch

from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
    SOLID_QUAD,
    SOLID_TRIANGLE,
)
from chemvector.bond_group_typed_affinity import (
    PAIR_CLASS_FOR_GROUP_TYPE,
    PAIR_NONE,
)
from chemvector.bond_group_typed_affinity_packing import (
    TypedAffinityPackingValidationError,
    build_typed_affinity_candidate_pool,
    decode_typed_affinity_packing,
    solve_typed_affinity_candidates,
)
from chemvector.primitive_set import LINE, QUAD, TRIANGLE


def _line(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0 / 2047.0, y0 / 2047.0, x1 / 2047.0, y1 / 2047.0, 0, 0, 0, 0]


def _partition(rows) -> set[tuple[int, tuple[int, ...]]]:
    return {
        (int(row["type"]), tuple(sorted(row["proposal_indices"])))
        for row in rows
    }


class TypedAffinityPackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = torch.tensor(
            [
                _line(100, 100, 280, 100),
                _line(400, 100, 600, 100),
                _line(400, 120, 600, 120),
                _line(800, 100, 1000, 100),
                _line(800, 120, 1000, 120),
                _line(800, 140, 1000, 140),
                _line(1200, 100, 1210, 100),
                _line(1198, 120, 1212, 120),
                _line(1196, 140, 1214, 140),
                _line(1194, 160, 1216, 160),
                [0.70, 0.55, 0.76, 0.55, 0.73, 0.62, 0, 0],
                [0.82, 0.55, 0.88, 0.55, 0.88, 0.62, 0.82, 0.62],
            ],
            dtype=torch.float32,
        )
        self.proposal_types = torch.tensor(
            [LINE] * 10 + [TRIANGLE, QUAD],
            dtype=torch.long,
        )
        self.usable = torch.ones(12, dtype=torch.bool)
        self.expected = {
            (SINGLE_LINE, (0,)),
            (PARALLEL_2, (1, 2)),
            (PARALLEL_3, (3, 4, 5)),
            (HASHED_WEDGE, (6, 7, 8, 9)),
            (SOLID_TRIANGLE, (10,)),
            (SOLID_QUAD, (11,)),
        }
        self.output = self._perfect_output()

    def _perfect_output(self):
        node = torch.full((12, 7), -9.0)
        pair = torch.full((12, 12, 4), -9.0)
        pair[..., PAIR_NONE] = 9.0
        for group_type, members in self.expected:
            node[list(members), group_type] = 9.0
            pair_class = PAIR_CLASS_FOR_GROUP_TYPE.get(group_type)
            if pair_class is None:
                continue
            for offset, left in enumerate(members):
                for right in members[offset + 1 :]:
                    pair[left, right] = -9.0
                    pair[right, left] = -9.0
                    pair[left, right, pair_class] = 9.0
                    pair[right, left, pair_class] = 9.0
        return {
            "node_group_type_logits": node,
            "pair_logits": pair,
        }

    def test_perfect_logits_recover_every_supported_group_type(self) -> None:
        result = decode_typed_affinity_packing(
            self.output,
            self.geometry,
            self.proposal_types,
            self.usable,
            known_group_count=6,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.solver_status, "optimal")
        self.assertEqual(_partition(result.typed_rows), self.expected)
        self.assertEqual(len(result.typed_rows), 6)
        self.assertGreater(result.candidate_counts["geometry"], 0)
        self.assertGreater(result.candidate_counts["learned"], 0)
        self.assertGreaterEqual(
            result.candidate_counts["union"],
            result.candidate_counts["geometry"],
        )
        self.assertGreaterEqual(result.solver_latency_ms, 0.0)

        forbidden = {"root_index", "free_slot", "carrier_points", "geometry"}
        ownership = [0] * len(self.usable)
        for row in result.typed_rows:
            self.assertTrue(row["valid_membership"])
            self.assertTrue(forbidden.isdisjoint(row))
            for proposal_index in row["proposal_indices"]:
                ownership[proposal_index] += 1
        self.assertEqual(ownership, [1] * len(self.usable))

    def test_candidate_input_permutation_cannot_change_exact_cover(self) -> None:
        pool = build_typed_affinity_candidate_pool(
            self.output,
            self.geometry,
            self.proposal_types,
            self.usable,
        )
        baseline = solve_typed_affinity_candidates(
            pool.candidates,
            self.output,
            self.proposal_types,
            self.usable,
            6,
        )
        reversed_result = solve_typed_affinity_candidates(
            tuple(reversed(pool.candidates)),
            self.output,
            self.proposal_types,
            self.usable,
            6,
        )

        def selected(result):
            return {
                (candidate.group_type, candidate.proposal_indices)
                for candidate in result.selected_candidates
            }

        self.assertTrue(baseline.feasible)
        self.assertEqual(selected(baseline), selected(reversed_result))
        self.assertAlmostEqual(baseline.objective, reversed_result.objective)

    def test_joint_proposal_permutation_is_equivariant(self) -> None:
        permutation = torch.tensor([7, 1, 10, 4, 0, 11, 6, 3, 9, 2, 5, 8])
        node = self.output["node_group_type_logits"][permutation]
        pair = self.output["pair_logits"][permutation][:, permutation]
        result = decode_typed_affinity_packing(
            {"node_group_type_logits": node, "pair_logits": pair},
            self.geometry[permutation],
            self.proposal_types[permutation],
            self.usable[permutation],
            known_group_count=6,
        )
        remapped = {
            (
                int(row["type"]),
                tuple(sorted(int(permutation[index]) for index in row["proposal_indices"])),
            )
            for row in result.typed_rows
        }
        self.assertEqual(remapped, self.expected)

    def test_invalid_logits_and_infeasible_known_count_are_distinct(self) -> None:
        asymmetric = {
            "node_group_type_logits": self.output["node_group_type_logits"],
            "pair_logits": self.output["pair_logits"].clone(),
        }
        asymmetric["pair_logits"][0, 1, PAIR_NONE] += 1.0
        with self.assertRaisesRegex(
            TypedAffinityPackingValidationError,
            "symmetric",
        ):
            decode_typed_affinity_packing(
                asymmetric,
                self.geometry,
                self.proposal_types,
                self.usable,
                6,
            )

        one_geometry = self.geometry[:1]
        one_output = {
            "node_group_type_logits": self.output["node_group_type_logits"][:1],
            "pair_logits": self.output["pair_logits"][:1, :1],
        }
        infeasible = decode_typed_affinity_packing(
            one_output,
            one_geometry,
            self.proposal_types[:1],
            self.usable[:1],
            known_group_count=0,
        )
        self.assertFalse(infeasible.feasible)
        self.assertEqual(infeasible.solver_status, "infeasible")
        self.assertEqual(infeasible.typed_rows, ())

    def test_empty_partition_and_unsupported_active_type_boundaries(self) -> None:
        empty = decode_typed_affinity_packing(
            {
                "node_group_type_logits": torch.zeros((1, 7)),
                "pair_logits": torch.zeros((1, 1, 4)),
            },
            torch.zeros((1, 8)),
            torch.tensor([0]),
            torch.tensor([False]),
            known_group_count=0,
        )
        self.assertTrue(empty.feasible)
        self.assertEqual(empty.typed_rows, ())

        with self.assertRaisesRegex(
            TypedAffinityPackingValidationError,
            "unsupported physical types",
        ):
            decode_typed_affinity_packing(
                {
                    "node_group_type_logits": torch.zeros((1, 7)),
                    "pair_logits": torch.zeros((1, 1, 4)),
                },
                torch.zeros((1, 8)),
                torch.tensor([4]),
                torch.tensor([True]),
                known_group_count=1,
            )


if __name__ == "__main__":
    unittest.main()
