from __future__ import annotations

import json
from pathlib import Path
import unittest

from formal.chemgraph.audit.constrained_recovery_v0_1 import (
    ScoreInterval,
    bounded_score_values,
    certify_class_intervals,
    graph_equivalence_classes,
)
from formal.chemgraph.evidence_core_v0_4.candidate.candidate import build_candidate_set


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evidence_core_v0_4" / "candidate" / "fixtures"


def candidate_result(name: str) -> dict:
    value = json.loads((FIXTURES / f"{name}.evidence.json").read_bytes())
    return build_candidate_set(value)


def edge_pairs(graph: dict) -> set[tuple[int, int]]:
    return {(edge["begin"], edge["end"]) for edge in graph["edges"]}


class StructuralConstraintTests(unittest.TestCase):
    def test_path_has_no_unsupported_endpoint_shortcut(self) -> None:
        graph = candidate_result("chain")["candidates"][0]
        self.assertEqual(edge_pairs(graph), {(0, 1), (1, 2)})
        self.assertNotIn((0, 2), edge_pairs(graph))

    def test_fan_has_no_outer_to_outer_edges(self) -> None:
        graph = candidate_result("fan")["candidates"][0]
        degrees = {node["id"]: 0 for node in graph["nodes"]}
        for edge in graph["edges"]:
            degrees[edge["begin"]] += 1
            degrees[edge["end"]] += 1
        center = next(node for node, degree in degrees.items() if degree == 3)
        self.assertTrue(all(center in pair for pair in edge_pairs(graph)))

    def test_double_and_triple_bundles_have_two_nodes_and_one_edge(self) -> None:
        for fixture, order in (
            ("double_parallel", 2),
            ("triple_parallel", 3),
            ("noisy_triple_parallel", 3),
        ):
            with self.subTest(fixture=fixture):
                candidates = candidate_result(fixture)["candidates"]
                bundle = next(
                    graph for graph in candidates
                    if [edge["order"] for edge in graph["edges"]] == [order]
                )
                self.assertEqual(len(bundle["nodes"]), 2)
                self.assertEqual(len(bundle["edges"]), 1)

    def test_known_collision_fixtures_remain_ambiguous(self) -> None:
        for fixture in (
            "crossing",
            "double_parallel",
            "triple_parallel",
            "noisy_triple_parallel",
            "glyph_like",
        ):
            with self.subTest(fixture=fixture):
                result = candidate_result(fixture)
                classes = graph_equivalence_classes(result["candidates"])
                self.assertGreaterEqual(len(classes), 2)
                intervals = [ScoreInterval(0, 0) for _ in classes]
                self.assertEqual(
                    certify_class_intervals(result["candidates"], intervals)["decision"],
                    "Ambiguous",
                )


class MarginCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = candidate_result("double_parallel")["candidates"]

    def test_strict_interval_separation_certifies_one_graph_class(self) -> None:
        result = certify_class_intervals(
            self.candidates,
            [ScoreInterval(0, 3), ScoreInterval(4, 10)],
        )
        self.assertEqual(result["decision"], "Graph")
        self.assertEqual(result["winner_class"], 0)

    def test_touching_intervals_do_not_certify(self) -> None:
        result = certify_class_intervals(
            self.candidates,
            [ScoreInterval(0, 3), ScoreInterval(3, 10)],
        )
        self.assertEqual(result["decision"], "Ambiguous")

    def test_two_epsilon_margin_is_tight_by_exhaustive_enumeration(self) -> None:
        epsilon = 3
        for observed_truth in bounded_score_values(0, epsilon):
            for observed_alternative in bounded_score_values(7, epsilon):
                self.assertLess(observed_truth, observed_alternative)

        tied = False
        for observed_truth in bounded_score_values(0, epsilon):
            for observed_alternative in bounded_score_values(6, epsilon):
                tied |= observed_truth == observed_alternative
        self.assertTrue(tied)

    def test_unique_graph_class_needs_no_arbitrary_candidate_order(self) -> None:
        graph = candidate_result("chain")["candidates"][0]
        duplicate = json.loads(json.dumps(graph))
        duplicate["nodes"].reverse()
        result = certify_class_intervals([graph, duplicate])
        self.assertEqual(result, {
            "decision": "Graph",
            "class_count": 1,
            "winner_class": 0,
        })

    def test_invalid_score_intervals_are_rejected(self) -> None:
        for bounds in ((-1, 0), (2, 1), (False, 1), (0.5, 1)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    ScoreInterval(*bounds)


if __name__ == "__main__":
    unittest.main()
