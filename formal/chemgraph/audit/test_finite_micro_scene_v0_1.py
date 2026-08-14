from __future__ import annotations

from collections import Counter
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from formal.chemgraph.audit.finite_geometry_grammar_v0_1 import (
    ADVERSARIAL_INVALID,
    BOND_THROUGH_ACTIVE_NONINCIDENT_ATOM,
    CROSSING_ALLOWED,
    NONSEMANTIC_T_TOUCH,
    PHYSICAL_MEMBER_SEMANTIC_MISMATCH,
    POSITIVE_LENGTH_COLLINEAR_OVERLAP,
    SIMPLE_LINE_ANGLE_VALID,
    UNMARKED_180_DEGREE2,
    UNSUPPORTED_CROSSING_MULTIPLICITY,
    UNSUPPORTED_TEXT_NOT_RENDERED,
    classify_geometry,
)
from formal.chemgraph.audit.finite_micro_raster_v0_1 import (
    observation_for_scene,
    render_scene,
)
from formal.chemgraph.audit.finite_micro_scene_v0_1 import (
    EXPECTED_CORE_SCENES,
    Bond,
    GRID_EDGES,
    GRID_NODES,
    Node,
    PhysicalMember,
    Scene,
    canonical_graph_signature,
    crossing_junction_fixtures,
    double_parallel_fixtures,
    iter_core_scenes,
    relabel_scene,
)
from formal.chemgraph.audit.run_finite_micro_scene_v0_1 import run_audit
from formal.chemgraph.audit.verify_finite_micro_scene_v0_1 import verify_certificates


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

    def test_preregistered_grammar_minimal_counterexamples(self) -> None:
        def member(begin: Node, end: Node) -> PhysicalMember:
            return PhysicalMember((begin.x, begin.y), (end.x, end.y))

        left, middle, right = (
            Node(0, 12, 32),
            Node(1, 32, 32),
            Node(2, 52, 32),
        )
        top = Node(3, 32, 12)
        through = Scene(
            "counterexample-through",
            "test",
            (left, middle, right, top),
            (
                Bond(0, 2, members=(member(left, right),)),
                Bond(1, 3, members=(member(middle, top),)),
            ),
        )
        through_result = classify_geometry(through)
        self.assertEqual(through_result.layer, ADVERSARIAL_INVALID)
        self.assertIn(BOND_THROUGH_ACTIVE_NONINCIDENT_ATOM, through_result.reasons)
        self.assertIn(NONSEMANTIC_T_TOUCH, through_result.reasons)

        overlap = Scene(
            "counterexample-overlap",
            "test",
            (left, middle, right),
            (
                Bond(0, 2, members=(member(left, right),)),
                Bond(0, 1, members=(member(left, middle),)),
            ),
        )
        self.assertIn(
            POSITIVE_LENGTH_COLLINEAR_OVERLAP,
            classify_geometry(overlap).reasons,
        )

        straight_chain = Scene(
            "counterexample-straight-chain",
            "test",
            (left, middle, right),
            (
                Bond(0, 1, members=(member(left, middle),)),
                Bond(1, 2, members=(member(middle, right),)),
            ),
        )
        self.assertIn(UNMARKED_180_DEGREE2, classify_geometry(straight_chain).reasons)

        diagonal = Scene(
            "counterexample-proper-crossing",
            "test",
            (GRID_NODES[0], GRID_NODES[2], GRID_NODES[6], GRID_NODES[8]),
            (
                Bond(0, 8, members=(member(GRID_NODES[0], GRID_NODES[8]),)),
                Bond(2, 6, members=(member(GRID_NODES[2], GRID_NODES[6]),)),
            ),
        )
        self.assertEqual(classify_geometry(diagonal).layer, CROSSING_ALLOWED)

        nodes = tuple(GRID_NODES[index] for index in range(7))
        multiway = Scene(
            "counterexample-multiway",
            "test",
            nodes,
            tuple(
                Bond(begin, end, members=(member(GRID_NODES[begin], GRID_NODES[end]),))
                for begin, end in ((0, 4), (1, 6), (2, 3))
            ),
        )
        self.assertIn(
            UNSUPPORTED_CROSSING_MULTIPLICITY,
            classify_geometry(multiway).reasons,
        )

    def test_render_and_chemical_signature_boundaries(self) -> None:
        double, _ = double_parallel_fixtures()
        self.assertEqual(classify_geometry(double).layer, SIMPLE_LINE_ANGLE_VALID)
        shifted_members = (
            PhysicalMember((8, 24), (56, 24)),
            PhysicalMember((8, 28), (56, 28)),
        )
        shifted = replace(
            double,
            scene_id="counterexample-asymmetric-double",
            bonds=(replace(double.bonds[0], members=shifted_members),),
        )
        self.assertIn(
            PHYSICAL_MEMBER_SEMANTIC_MISMATCH,
            classify_geometry(shifted).reasons,
        )
        fake_display = replace(
            double,
            scene_id="counterexample-fake-display",
            bonds=(replace(double.bonds[0], display_type="FOO"),),
        )
        self.assertEqual(
            canonical_graph_signature(double), canonical_graph_signature(fake_display)
        )
        self.assertEqual(classify_geometry(fake_display).layer, ADVERSARIAL_INVALID)
        text_only = Scene(
            "counterexample-unrendered-text",
            "test",
            (Node(0, 32, 32, "TEXT:N"),),
            (),
        )
        self.assertIn(
            UNSUPPORTED_TEXT_NOT_RENDERED, classify_geometry(text_only).reasons
        )

    def test_full_core_grammar_counts(self) -> None:
        counts = Counter(classify_geometry(scene).layer for scene in iter_core_scenes())
        self.assertEqual(
            counts,
            {
                ADVERSARIAL_INVALID: 30_059,
                CROSSING_ALLOWED: 16_832,
                SIMPLE_LINE_ANGLE_VALID: 19_821,
            },
        )

    def test_grammar_is_invariant_to_representation_and_exact_geometry(self) -> None:
        crossing, _ = crossing_junction_fixtures()
        expected = classify_geometry(crossing)
        atom_ids = [node.atom_id for node in crossing.nodes]
        relabeled = relabel_scene(
            crossing,
            dict(zip(atom_ids, reversed([atom_id + 100 for atom_id in atom_ids]))),
            reverse_bonds=True,
        )
        self.assertEqual(classify_geometry(relabeled), expected)

        swapped_bonds = tuple(
            replace(
                bond,
                begin_atom=bond.end_atom,
                end_atom=bond.begin_atom,
                members=tuple(
                    PhysicalMember(member.end, member.begin)
                    for member in reversed(bond.members)
                ),
            )
            for bond in reversed(crossing.bonds)
        )
        self.assertEqual(
            classify_geometry(
                replace(crossing, scene_id="crossing-swapped", bonds=swapped_bonds)
            ),
            expected,
        )

        def transform_scene(scene: Scene, name: str, transform) -> Scene:
            return replace(
                scene,
                scene_id=name,
                nodes=tuple(
                    replace(node, x=transform((node.x, node.y))[0], y=transform((node.x, node.y))[1])
                    for node in scene.nodes
                ),
                bonds=tuple(
                    replace(
                        bond,
                        members=tuple(
                            PhysicalMember(transform(member.begin), transform(member.end))
                            for member in bond.members
                        ),
                    )
                    for bond in scene.bonds
                ),
            )

        transforms = {
            "translated": lambda point: (point[0] + 2, point[1] + 2),
            "rotated90": lambda point: (64 - point[1], point[0]),
            "reflected": lambda point: (64 - point[0], point[1]),
        }
        for name, transform in transforms.items():
            with self.subTest(name=name):
                self.assertEqual(
                    classify_geometry(transform_scene(crossing, name, transform)),
                    expected,
                )

        small = Scene(
            "small-angle",
            "test",
            (Node(0, 10, 10), Node(1, 20, 10), Node(2, 20, 20)),
            (
                Bond(0, 1, members=(PhysicalMember((10, 10), (20, 10)),)),
                Bond(1, 2, members=(PhysicalMember((20, 10), (20, 20)),)),
            ),
        )
        scaled = transform_scene(small, "small-angle-scaled", lambda point: (2 * point[0], 2 * point[1]))
        self.assertEqual(classify_geometry(scaled), classify_geometry(small))

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
                "scene_manifest.jsonl",
            ):
                self.assertTrue((Path(directory) / name).is_file())
            self.assertEqual(len(result["manifest_rows"]), 16)
            self.assertIn(
                "canonical_graph_signature_full", result["manifest_rows"][0]
            )
            self.assertIn("exclusive_grammar_layer", result["manifest_rows"][0])
            self.assertIn("active_prior_id", result["manifest_rows"][0])
            self.assertIn("simple_domain", result["summary"]["cumulative_domains"])
            self.assertIn("all_domain", result["summary"]["cumulative_domains"])
            self.assertEqual(
                result["validation"]["identifiability_gate_status"], "no_go"
            )
            review = verify_certificates(
                Path(directory) / "collision_certificates.json"
            )
            self.assertEqual(review["status"], "passed")
            self.assertEqual(
                review["different_signature_pairs_checked"],
                2,
            )
            self.assertEqual(
                review["isomorphic_pairs_misclassified_as_different"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
