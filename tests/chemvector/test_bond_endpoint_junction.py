import unittest

import torch

from chemvector.bond_endpoint_junction import (
    junctions_from_bond_endpoints,
    junctions_from_endpoint_groups,
    project_unique_junction_pairs,
)


def bond(begin, end, *, score=0.9, chemical=0):
    return {
        "endpoints": torch.tensor((begin, end), dtype=torch.float32),
        "score": score,
        "display": chemical,
        "chemical": chemical,
        "direction": 0,
    }


class BondEndpointJunctionTest(unittest.TestCase):
    def test_unique_pair_projection_uses_explicit_ranking_scores(self) -> None:
        objects = [
            bond((0.1, 0.2), (0.9, 0.2), score=0.95, chemical=0),
            bond((0.2, 0.7), (0.8, 0.7), score=0.20, chemical=2),
        ]

        result = project_unique_junction_pairs(
            objects,
            torch.tensor((4, 9, 4, 9)),
            ranking_scores=torch.tensor((0.1, 0.9)),
        )

        self.assertEqual(len(result.bond_objects), 1)
        self.assertEqual(result.bond_objects[0]["chemical"], 2)
        self.assertEqual(result.duplicate_pairs, 1)
        self.assertTrue(
            torch.allclose(
                result.points,
                torch.tensor(((0.2, 0.7), (0.8, 0.7))),
            )
        )

    def test_unique_pair_projection_counts_self_loops_and_stable_ties(self) -> None:
        objects = [
            bond((0.1, 0.1), (0.2, 0.2), chemical=1),
            bond((0.2, 0.5), (0.8, 0.5), chemical=2),
            bond((0.3, 0.6), (0.7, 0.6), chemical=3),
        ]

        result = project_unique_junction_pairs(
            objects,
            torch.tensor((3, 3, 4, 5, 4, 5)),
            ranking_scores=torch.tensor((0.9, 0.5, 0.5)),
        )

        self.assertEqual(result.self_loops, 1)
        self.assertEqual(result.duplicate_pairs, 1)
        self.assertEqual(len(result.bond_objects), 1)
        self.assertEqual(result.bond_objects[0]["chemical"], 2)

    def test_unique_pair_projection_accepts_empty_input(self) -> None:
        result = project_unique_junction_pairs(
            [],
            torch.empty((0,), dtype=torch.long),
            ranking_scores=torch.empty((0,), dtype=torch.float32),
        )

        self.assertEqual(result.points.shape, (0, 2))
        self.assertEqual(result.bond_objects, ())

    def test_explicit_endpoint_groups_define_shared_junctions(self) -> None:
        result = junctions_from_endpoint_groups(
            [
                bond((0.1, 0.5), (0.45, 0.5)),
                bond((0.55, 0.5), (0.9, 0.5)),
            ],
            torch.tensor((0, 1, 1, 2)),
        )
        self.assertEqual(result.points.shape, (3, 2))
        self.assertEqual(
            [item["junction_pair"] for item in result.bond_objects],
            [(0, 1), (1, 2)],
        )

    def test_parallel_strokes_remain_one_semantic_bond_with_two_junctions(self) -> None:
        result = junctions_from_bond_endpoints(
            [bond((0.1, 0.5), (0.9, 0.5), chemical=1)],
            width=100,
            height=100,
            radius_pixels=6.0,
        )
        self.assertEqual(result.points.shape, (2, 2))
        self.assertEqual(len(result.bond_objects), 1)
        self.assertEqual(result.bond_objects[0]["junction_pair"], (0, 1))

    def test_shared_noisy_endpoints_form_one_junction(self) -> None:
        result = junctions_from_bond_endpoints(
            [
                bond((0.1, 0.5), (0.49, 0.5)),
                bond((0.51, 0.5), (0.9, 0.5)),
            ],
            width=100,
            height=100,
            radius_pixels=3.0,
        )
        self.assertEqual(result.points.shape, (3, 2))
        self.assertEqual(
            [item["junction_pair"] for item in result.bond_objects],
            [(0, 1), (1, 2)],
        )

    def test_identical_large_coordinates_merge_at_subpixel_radius(self) -> None:
        shared = (647.5 / 768.0, 623.5 / 768.0)
        result = junctions_from_bond_endpoints(
            [
                bond((0.1, 0.1), shared),
                bond(shared, (0.9, 0.9)),
            ],
            width=768,
            height=768,
            radius_pixels=0.25,
        )
        self.assertEqual(result.points.shape, (3, 2))
        self.assertEqual(
            [item["junction_pair"] for item in result.bond_objects],
            [(0, 1), (1, 2)],
        )

    def test_duplicate_bond_objects_do_not_create_extra_junctions(self) -> None:
        result = junctions_from_bond_endpoints(
            [
                bond((0.1, 0.5), (0.9, 0.5), score=0.9),
                bond((0.11, 0.5), (0.89, 0.5), score=0.7),
            ],
            width=100,
            height=100,
            radius_pixels=3.0,
        )
        self.assertEqual(result.points.shape, (2, 2))
        self.assertEqual(len(result.bond_objects), 1)
        self.assertEqual(result.duplicate_pairs, 1)


if __name__ == "__main__":
    unittest.main()
