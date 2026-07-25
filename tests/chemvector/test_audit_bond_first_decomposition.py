import unittest

import torch

from chemvector.audit_bond_first_decomposition import (
    analyze_objects,
    failure_stage,
    sample_keys_sha256,
    select_routed_object_batches,
    sample_slices,
    summarize,
)


def sample() -> dict:
    return {
        "key": "sample",
        "layers": {
            "width": 100,
            "height": 100,
            "variant": "stress",
            "render_parameters": {"stress_operation": "crossed_bonds"},
            "atoms": [
                {"atom_id": 0, "center": [10.0, 10.0], "visible_text": False, "text": ""},
                {"atom_id": 1, "center": [50.0, 10.0], "visible_text": True, "text": "N"},
                {"atom_id": 2, "center": [90.0, 10.0], "visible_text": True, "text": "CH3"},
            ],
            "bonds": [
                {
                    "begin_atom": 0,
                    "end_atom": 1,
                    "begin": [10.0, 10.0],
                    "end": [50.0, 10.0],
                    "display_type": "SINGLE",
                    "chemical_type": "SINGLE",
                    "direction": "NONE",
                },
                {
                    "begin_atom": 1,
                    "end_atom": 2,
                    "begin": [50.0, 10.0],
                    "end": [90.0, 10.0],
                    "display_type": "DOUBLE",
                    "chemical_type": "DOUBLE",
                    "direction": "NONE",
                },
            ],
        },
    }


def bond(begin, end, display=0, chemical=0, direction=0, score=0.9):
    return {
        "endpoints": torch.tensor((begin, end), dtype=torch.float32),
        "display": display,
        "chemical": chemical,
        "direction": direction,
        "score": score,
    }


class BondFirstDecompositionTest(unittest.TestCase):
    def test_style_router_selects_complete_per_image_expert_outputs(self) -> None:
        base = [[{"source": "base-0"}], [{"source": "base-1"}]]
        expert = [[{"source": "expert-0"}], [{"source": "expert-1"}]]

        selected, routes = select_routed_object_batches(
            base, expert, torch.tensor((0.98, 0.995)), threshold=0.99
        )

        self.assertEqual(routes, [False, True])
        self.assertEqual(selected, [base[0], expert[1]])

    def test_sample_keys_sha256_is_order_sensitive(self) -> None:
        first = sample_keys_sha256([{"key": "a"}, {"key": "b"}])
        second = sample_keys_sha256([{"key": "b"}, {"key": "a"}])

        self.assertNotEqual(first, second)
        self.assertEqual(first, sample_keys_sha256([{"key": "a"}, {"key": "b"}]))

    def test_exact_objects_pass_all_stages(self) -> None:
        objects = [
            bond((0.1, 0.1), (0.5, 0.1)),
            bond((0.5, 0.1), (0.9, 0.1), display=1, chemical=1),
        ]
        metrics = analyze_objects(sample(), objects)
        self.assertTrue(metrics["geometry_exact"])
        self.assertTrue(metrics["topology_exact"])
        self.assertTrue(metrics["topology_candidate_oracle"])
        self.assertTrue(metrics["chemical_candidate_oracle"])
        self.assertTrue(metrics["chemical_exact"])
        self.assertEqual(failure_stage(metrics, True), "exact_sdf")

    def test_duplicate_pair_is_not_topology_exact(self) -> None:
        objects = [
            bond((0.1, 0.1), (0.5, 0.1)),
            bond((0.11, 0.1), (0.49, 0.1), score=0.8),
        ]
        metrics = analyze_objects(sample(), objects)
        self.assertEqual(metrics["duplicate_pairs"], 1)
        self.assertEqual(metrics["topology_false_negatives"], 1)
        self.assertFalse(metrics["topology_candidate_oracle"])
        self.assertFalse(metrics["recoverable_topology_exact"])
        self.assertFalse(metrics["topology_exact"])
        self.assertEqual(
            failure_stage(metrics, False), "topology_or_endpoint_assignment"
        )

    def test_slices_and_summary(self) -> None:
        names = sample_slices(sample())
        self.assertIn("display_double", names)
        self.assertIn("has_multichar_text", names)
        self.assertIn("stress_crossed_bonds", names)
        objects = [
            bond((0.1, 0.1), (0.5, 0.1)),
            bond((0.5, 0.1), (0.9, 0.1), display=1, chemical=1),
        ]
        metrics = analyze_objects(sample(), objects)
        result = summarize(
            [
                {
                    "known_count": metrics,
                    "predicted": metrics,
                    "compiled_sdf_exact": True,
                    "failure_stage": "exact",
                }
            ]
        )
        self.assertEqual(result["predicted_topology_exact"], 1.0)
        self.assertEqual(result["topology_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
