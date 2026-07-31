import unittest

import torch

from chemvector.infer_chemscene_hybrid import (
    apply_graph_correction,
    assemble_chemscene,
    attach_text_to_bond_junctions,
    compile_scene_with_aromatic_fallback,
    repair_impossible_explicit_h,
    suppress_text_interior_bond_objects,
)
from chemvector.evaluate_detected_ctc_ocr import match_text_predictions


class ChemSceneHybridAssemblerTest(unittest.TestCase):
    def test_text_interior_veto_removes_only_fully_contained_bond(self) -> None:
        interior = {
            "score": 0.8,
            "endpoints": torch.tensor(((0.46, 0.47), (0.54, 0.53))),
        }
        attached = {
            "score": 0.9,
            "endpoints": torch.tensor(((0.50, 0.50), (0.80, 0.50))),
        }
        kept, removed = suppress_text_interior_bond_objects(
            [interior, attached],
            [
                {
                    "x": 0.50,
                    "y": 0.50,
                    "bbox_x": 0.50,
                    "bbox_y": 0.50,
                    "half_width": 0.10,
                    "half_height": 0.10,
                }
            ],
            image_size=100,
        )
        self.assertEqual(kept, [attached])
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["bond_index"], 0)

    def test_text_interior_veto_uses_shifted_bbox_center(self) -> None:
        bond = {
            "score": 0.8,
            "endpoints": torch.tensor(((0.66, 0.47), (0.74, 0.53))),
        }
        kept, removed = suppress_text_interior_bond_objects(
            [bond],
            [
                {
                    "x": 0.50,
                    "y": 0.50,
                    "bbox_x": 0.70,
                    "bbox_y": 0.50,
                    "half_width": 0.10,
                    "half_height": 0.10,
                }
            ],
            image_size=100,
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(removed), 1)

    def test_text_attachment_cannot_create_structural_junctions(self) -> None:
        roles, labels = attach_text_to_bond_junctions(
            torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.1))),
            torch.tensor(((0.11, 0.1), (0.12, 0.1), (0.8, 0.8))),
            torch.tensor((1, 1, 1)),
            ["N", "O", "F"],
            image_size=100,
            maximum_distance_pixels=5.0,
        )
        self.assertEqual(len(roles), 3)
        self.assertEqual(int((roles == 1).sum()), 1)
        self.assertIn(labels[0], {"N", "O"})

    def test_detected_ocr_matching_is_one_to_one(self) -> None:
        matches = match_text_predictions(
            torch.tensor(((0.10, 0.10), (0.11, 0.10), (0.80, 0.80))),
            torch.tensor((1, 1, 0)),
            ["N", "O", "C"],
            [{"center": [10.0, 10.0], "visible_text": True, "text": "N"}],
            image_size=100,
            tolerance_pixels=8.0,
        )
        self.assertEqual(matches, [("N", "N")])

    def test_valence_repair_drops_only_impossible_ocr_h(self) -> None:
        points = torch.tensor(((0.5, 0.5), (0.2, 0.5), (0.8, 0.5), (0.5, 0.8)))
        roles = torch.tensor((1, 0, 0, 0))
        objects = []
        for endpoint in (1, 2, 3):
            objects.append(
                {
                    "score": 0.9,
                    "endpoints": torch.stack((points[0], points[endpoint])),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                }
            )
        scene, _ = assemble_chemscene(
            "nh-hallucination", points, roles, ["NH", "C", "C", "C"], objects
        )
        repaired, changes = repair_impossible_explicit_h(scene)
        self.assertEqual(repaired["junctions"][0]["atom"]["explicit_h_count"], 0)
        self.assertEqual(repaired["text_objects"][0]["text"], "N")
        self.assertEqual(changes[0]["junction_index"], 0)

    def test_valence_repair_preserves_valid_nh(self) -> None:
        points = torch.tensor(((0.5, 0.5), (0.2, 0.5), (0.8, 0.5)))
        roles = torch.tensor((1, 0, 0))
        objects = [
            {
                "score": 0.9,
                "endpoints": torch.stack((points[0], points[index])),
                "display": 0,
                "chemical": 0,
                "direction": 0,
            }
            for index in (1, 2)
        ]
        scene, _ = assemble_chemscene(
            "valid-nh", points, roles, ["NH", "C", "C"], objects
        )
        repaired, changes = repair_impossible_explicit_h(scene)
        self.assertEqual(repaired["junctions"][0]["atom"]["explicit_h_count"], 1)
        self.assertEqual(changes, [])

    def test_graph_type_only_preserves_objects_and_updates_types(self) -> None:
        objects = [
            {"display": 0, "chemical": 0, "score": 0.9},
            {"display": 1, "chemical": 1, "score": 0.8},
        ]
        output = {
            "edge_keep_logits": torch.tensor((5.0, -5.0)),
            "display_logits": torch.tensor(((0.0, 4.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0))),
            "chemical_logits": torch.tensor(((0.0, 4.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0))),
        }
        corrected, diagnostics = apply_graph_correction(
            objects, output, mode="type_only", keep_threshold=0.5
        )
        self.assertEqual(len(corrected), 2)
        self.assertEqual(corrected[0]["chemical"], 1)
        self.assertEqual(diagnostics["graph_removed_bond_objects"], 0)

    def test_explicit_learned_assignments_can_reject_false_objects(self) -> None:
        points = torch.tensor(((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)))
        roles = torch.tensor((0, 0, 0))
        objects = [
            {
                "score": 0.9,
                "endpoints": torch.tensor(((0.1, 0.1), (0.9, 0.1))),
                "display": 0,
                "chemical": 0,
                "direction": 0,
            },
            {
                "score": 0.8,
                "endpoints": torch.tensor(((0.2, 0.8), (0.8, 0.8))),
                "display": 0,
                "chemical": 0,
                "direction": 0,
            },
        ]
        scene, diagnostics = assemble_chemscene(
            "learned",
            points,
            roles,
            ["C", "C", "C"],
            objects,
            assignment_mode="learned",
            junction_assignments=[(0, 1), None],
        )
        self.assertEqual(len(scene["bond_objects"]), 1)
        self.assertEqual(diagnostics["assembled_bonds"], 1)

    def test_bond_endpoint_assignments_are_used_directly(self) -> None:
        points = torch.tensor(((0.1, 0.1), (0.5, 0.9), (0.9, 0.1)))
        objects = [
            {
                "score": 0.9,
                "endpoints": torch.tensor(((0.1, 0.1), (0.9, 0.1))),
                "display": 1,
                "chemical": 1,
                "direction": 0,
            }
        ]
        scene, diagnostics = assemble_chemscene(
            "bond-endpoints",
            points,
            torch.zeros(3, dtype=torch.long),
            ["C", "C", "C"],
            objects,
            assignment_mode="explicit",
            junction_assignments=[(0, 2)],
        )
        self.assertEqual(len(scene["bond_objects"]), 1)
        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "DOUBLE")
        self.assertEqual(diagnostics["junction_assignment_mode"], "explicit")

    def test_assembler_maps_bond_endpoints_to_predicted_junctions(self) -> None:
        scene, diagnostics = assemble_chemscene(
            "example",
            torch.tensor(((0.1, 0.2), (0.8, 0.7))),
            torch.tensor((1, 0)),
            ["N", "C"],
            [
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.11, 0.21), (0.79, 0.69))),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                }
            ],
        )
        self.assertEqual(scene["junctions"][0]["atom"]["element"], "N")
        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "SINGLE")
        self.assertEqual(diagnostics["assembled_bonds"], 1)

    def test_unparsed_text_is_preserved_as_dummy_atom_instead_of_carbon(self) -> None:
        scene, _ = assemble_chemscene(
            "alias",
            torch.tensor(((0.1, 0.2), (0.8, 0.7))),
            torch.tensor((1, 0)),
            ["Ph", "C"],
            [
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.11, 0.21), (0.79, 0.69))),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                }
            ],
            unresolved_as_dummy=True,
        )
        self.assertEqual(scene["junctions"][0]["atom"]["element"], "*")
        self.assertEqual(scene["text_objects"][0]["text"], "Ph")

    def test_greedy_protocol_keeps_legacy_unknown_text_fallback(self) -> None:
        scene, _ = assemble_chemscene(
            "legacy-alias",
            torch.tensor(((0.1, 0.2), (0.8, 0.7))),
            torch.tensor((1, 0)),
            ["Ph", "C"],
            [
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.11, 0.21), (0.79, 0.69))),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                }
            ],
        )
        self.assertEqual(scene["junctions"][0]["atom"]["element"], "C")

    def test_aromatic_bond_marks_both_endpoint_atoms_aromatic(self) -> None:
        scene, _ = assemble_chemscene(
            "aromatic",
            torch.tensor(((0.2, 0.2), (0.8, 0.8))),
            torch.tensor((0, 0)),
            ["C", "C"],
            [
                {
                    "score": 0.8,
                    "endpoints": torch.tensor(((0.2, 0.2), (0.8, 0.8))),
                    "display": 3,
                    "chemical": 3,
                    "direction": 0,
                }
            ],
        )
        self.assertTrue(all(item["atom"]["aromatic"] for item in scene["junctions"]))

    def test_duplicate_mapped_pair_keeps_the_higher_score_object(self) -> None:
        scene, diagnostics = assemble_chemscene(
            "duplicate",
            torch.tensor(((0.1, 0.1), (0.9, 0.9))),
            torch.tensor((0, 0)),
            ["C", "C"],
            [
                {
                    "score": 0.3,
                    "endpoints": torch.tensor(((0.1, 0.1), (0.9, 0.9))),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                },
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.11, 0.1), (0.89, 0.9))),
                    "display": 1,
                    "chemical": 1,
                    "direction": 0,
                },
            ],
        )
        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "DOUBLE")
        self.assertEqual(diagnostics["duplicate_pair_objects"], 1)

    def test_invalid_aromatic_set_falls_back_to_visible_bond_types(self) -> None:
        scene, _ = assemble_chemscene(
            "invalid-aromatic",
            torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.1))),
            torch.tensor((0, 0, 0)),
            ["C", "C", "C"],
            [
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.1, 0.1), (0.5, 0.5))),
                    "display": 0,
                    "chemical": 3,
                    "direction": 0,
                },
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.5, 0.5), (0.9, 0.1))),
                    "display": 1,
                    "chemical": 3,
                    "direction": 0,
                },
            ],
        )
        sdf, compiled, used_fallback = compile_scene_with_aromatic_fallback(scene)
        self.assertTrue(used_fallback)
        self.assertIn("$$$$", sdf)
        self.assertEqual(
            [bond["chemical_type"] for bond in compiled["bond_objects"]],
            ["SINGLE", "DOUBLE"],
        )

    def test_global_unique_assignment_avoids_self_loops_and_duplicate_pairs(self) -> None:
        scene, diagnostics = assemble_chemscene(
            "global",
            torch.tensor(((0.1, 0.1), (0.5, 0.5), (0.9, 0.9))),
            torch.tensor((0, 0, 0)),
            ["C", "C", "C"],
            [
                {
                    "score": 0.9,
                    "endpoints": torch.tensor(((0.1, 0.1), (0.11, 0.11))),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                },
                {
                    "score": 0.8,
                    "endpoints": torch.tensor(((0.1, 0.1), (0.5, 0.5))),
                    "display": 0,
                    "chemical": 0,
                    "direction": 0,
                },
            ],
            assignment_mode="global_unique",
        )
        pairs = {
            tuple(sorted(tuple(endpoint) for endpoint in bond["endpoints"]))
            for bond in scene["bond_objects"]
        }
        self.assertEqual(len(pairs), 2)
        self.assertEqual(diagnostics["self_loop_objects"], 0)
        self.assertEqual(diagnostics["duplicate_pair_objects"], 0)
        self.assertGreaterEqual(diagnostics["reassigned_bond_objects"], 1)


if __name__ == "__main__":
    unittest.main()
