import unittest

import torch

from chemvector.audit_bond_endpoint_junction import analyze_endpoint_junctions


def sample():
    return {
        "key": "sample",
        "chemmol": {"inchikey": "unused"},
        "layers": {
            "width": 100,
            "height": 100,
            "variant": "standard",
            "render_parameters": {},
            "atoms": [
                {"atom_id": 0, "center": [10.0, 50.0], "visible_text": False, "text": ""},
                {"atom_id": 1, "center": [50.0, 50.0], "visible_text": False, "text": ""},
                {"atom_id": 2, "center": [90.0, 50.0], "visible_text": False, "text": ""},
            ],
            "bonds": [
                {"begin_atom": 0, "end_atom": 1, "begin": [10.0, 50.0], "end": [50.0, 50.0], "display_type": "DOUBLE", "chemical_type": "DOUBLE", "direction": "NONE"},
                {"begin_atom": 1, "end_atom": 2, "begin": [50.0, 50.0], "end": [90.0, 50.0], "display_type": "SINGLE", "chemical_type": "SINGLE", "direction": "NONE"},
            ],
        },
    }


def example():
    return {
        "endpoints": torch.tensor(
            (((0.1, 0.5), (0.49, 0.5)), ((0.51, 0.5), (0.9, 0.5)))
        ),
        "scores": torch.tensor((0.9, 0.9)),
        "display_ids": torch.tensor((1, 0)),
        "chemical_ids": torch.tensor((1, 0)),
        "direction_ids": torch.tensor((0, 0)),
    }


class BondEndpointJunctionAuditTest(unittest.TestCase):
    def test_double_bond_is_one_object_and_two_semantic_endpoints(self) -> None:
        metrics = analyze_endpoint_junctions(
            sample(),
            example(),
            threshold=0.25,
            cluster_radius_pixels=3.0,
            match_radius_pixels=3.0,
            compile_sdf=False,
        )
        self.assertEqual(metrics["raw_endpoint_hypotheses"], 4)
        self.assertEqual(metrics["predicted_nodes"], 3)
        self.assertTrue(metrics["vertex_exact"])
        self.assertTrue(metrics["topology_exact"])
        self.assertTrue(metrics["chemical_exact"])


if __name__ == "__main__":
    unittest.main()
