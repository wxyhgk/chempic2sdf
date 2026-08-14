from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from formal.chemgraph.audit.finite_micro_raster_v0_1 import (
    observation_for_scene,
    render_scene,
)
from formal.chemgraph.audit.finite_micro_scene_v0_1 import (
    EXPECTED_CORE_SCENES,
    GRID_EDGES,
    canonical_graph_signature,
    crossing_junction_fixtures,
    double_parallel_fixtures,
    iter_core_scenes,
    relabel_scene,
)
from formal.chemgraph.audit.run_finite_micro_scene_v0_1 import run_audit


class FiniteMicroSceneV01Tests(unittest.TestCase):
    def test_core_catalog_and_count(self) -> None:
        self.assertEqual(len(GRID_EDGES), 36)
        self.assertEqual(EXPECTED_CORE_SCENES, 66_712)
        self.assertEqual(sum(1 for _ in iter_core_scenes(limit=50)), 50)

    def test_graph_signature_ignores_atom_ids_and_bond_order(self) -> None:
        scene = list(iter_core_scenes(limit=100))[-1]
        atom_ids = [node.atom_id for node in scene.nodes]
        mapping = dict(zip(atom_ids, reversed([value + 100 for value in atom_ids])))
        relabeled = relabel_scene(scene, mapping, reverse_bonds=True)
        self.assertEqual(
            canonical_graph_signature(scene), canonical_graph_signature(relabeled)
        )

    def test_crossing_junction_fixture(self) -> None:
        crossing, junction = crossing_junction_fixtures()
        self.assertEqual(render_scene(crossing).tobytes(), render_scene(junction).tobytes())
        self.assertNotEqual(
            canonical_graph_signature(crossing), canonical_graph_signature(junction)
        )
        self.assertEqual(
            observation_for_scene(crossing)["production_tensor_sha256"],
            observation_for_scene(junction)["production_tensor_sha256"],
        )

    def test_double_parallel_fixture(self) -> None:
        double, parallel = double_parallel_fixtures()
        self.assertEqual(render_scene(double).tobytes(), render_scene(parallel).tobytes())
        self.assertNotEqual(
            canonical_graph_signature(double), canonical_graph_signature(parallel)
        )

    def test_default_and_prepared_paths_are_equivalent(self) -> None:
        scene = next(iter_core_scenes(limit=1))
        direct = observation_for_scene(scene)
        prepared = observation_for_scene(scene, image_is_prepared=True)
        self.assertEqual(
            direct["production_tensor_sha256"], prepared["production_tensor_sha256"]
        )

    def test_smoke_audit_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_audit(Path(directory), core_limit=12)
            self.assertEqual(result["validation"]["status"], "passed_smoke_audit")
            self.assertFalse(result["validation"]["training_authorized"])
            self.assertEqual(result["summary"]["processed_core_scene_count"], 12)
            self.assertEqual(result["summary"]["processed_fixture_scene_count"], 4)
            self.assertGreaterEqual(
                result["summary"]["graph_collision_class_count"], 2
            )
            for name in (
                "summary.json",
                "validation.json",
                "collision_certificates.json",
                "runtime.json",
            ):
                self.assertTrue((Path(directory) / name).is_file())


if __name__ == "__main__":
    unittest.main()
