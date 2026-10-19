"""Unit checks for the audit-only explicit-H edge guard.

The guard is intentionally not production inference.  These tests protect
the small, deterministic replay transform while broader chemistry rules are
still under evaluation.
"""

from __future__ import annotations

import copy
import unittest

from tools.audit_valence_guard_public import filter_explicit_hydrogen_overbonded_edges


def _scene(*, second_type: str = "AROMATIC", explicit_h: int = 1) -> dict:
    return {
        "atom_objects": [
            {
                "id": 0,
                "chemical": {
                    "element": "O",
                    "formal_charge": 0,
                    "explicit_h_count": explicit_h,
                    "radical_electrons": 0,
                },
                "incident_ports": [
                    {"bond_id": 0, "endpoint": 0},
                    {"bond_id": 1, "endpoint": 0},
                ],
            },
            {"id": 1, "chemical": {"element": "C"}, "incident_ports": []},
            {"id": 2, "chemical": {"element": "C"}, "incident_ports": []},
        ],
        "bond_objects": [
            {"id": 0, "atom_refs": [0, 1], "chemical_type": "SINGLE"},
            {"id": 1, "atom_refs": [0, 2], "chemical_type": second_type},
        ],
    }


class ValenceGuardAuditTests(unittest.TestCase):
    def test_drops_only_second_aromatic_heavy_edge_on_neutral_explicit_h_oxygen(self):
        scene = _scene()
        original = copy.deepcopy(scene)
        guarded, report = filter_explicit_hydrogen_overbonded_edges(scene)

        self.assertEqual(report["removed_count"], 1)
        self.assertEqual(report["removed"][0]["atom_refs"], [0, 2])
        self.assertEqual([bond["id"] for bond in guarded["bond_objects"]], [0])
        self.assertEqual(guarded["atom_objects"][0]["incident_ports"][0]["bond_id"], 0)
        self.assertEqual(scene, original, "audit guard must not mutate its input")

    def test_does_not_filter_non_aromatic_or_implicit_h_edges(self):
        for second_type, explicit_h in (("SINGLE", 1), ("AROMATIC", 0)):
            with self.subTest(second_type=second_type, explicit_h=explicit_h):
                guarded, report = filter_explicit_hydrogen_overbonded_edges(
                    _scene(second_type=second_type, explicit_h=explicit_h)
                )
                self.assertEqual(report["removed_count"], 0)
                self.assertEqual(len(guarded["bond_objects"]), 2)


if __name__ == "__main__":
    unittest.main()
