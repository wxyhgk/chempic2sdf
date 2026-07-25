"""Tests for fixed-topology structured atom-state decoding."""

from __future__ import annotations

import unittest

import torch

from Data.sdf.chemscene_v0_1 import mol_from_chemscene_v0_1
from chemvector.infer_chemscene_hybrid import assemble_chemscene
from chemvector.structured_atom_decoder import (
    atom_state_candidate,
    failure_atom_indices,
    merge_ctc_atom_candidates,
    rerank_invalid_scene_atom_states,
    scene_topology_signature,
)


def quaternary_nitrogen_scene() -> dict:
    points = torch.tensor(
        ((0.5, 0.5), (0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8))
    )
    objects = []
    for end in range(1, 5):
        objects.append(
            {
                "score": 0.9,
                "endpoints": torch.stack((points[0], points[end])),
                "display": 0,
                "chemical": 0,
                "direction": 0,
                "junction_pair": (0, end),
            }
        )
    scene, _ = assemble_chemscene(
        "quat-n",
        points,
        torch.tensor((1, 0, 0, 0, 0)),
        ["N", "C", "C", "C", "C"],
        objects,
        assignment_mode="explicit",
        junction_assignments=[item["junction_pair"] for item in objects],
    )
    return scene


class StructuredAtomDecoderTests(unittest.TestCase):
    def test_shifted_and_anchor_beams_are_merged(self) -> None:
        shifted = torch.full((3, 257), -10.0)
        shifted[:, 0] = 2.0
        shifted[1, ord("N") + 1] = 8.0
        anchor = shifted.clone()
        anchor[2, ord("+") + 1] = 7.0
        candidates = merge_ctc_atom_candidates(
            (("shifted", shifted), ("anchor", anchor)),
            current_text="N",
            beam_size=8,
            maximum_candidates=8,
        )
        texts = {item.text for item in candidates}
        self.assertIn("N", texts)
        self.assertIn("N+", texts)

    def test_failure_atom_indices_handles_valence_and_kekulize_errors(self) -> None:
        self.assertEqual(
            failure_atom_indices("Explicit valence for atom # 2 N, 4", 5), (2,)
        )
        self.assertEqual(
            failure_atom_indices("Unkekulized atoms: 1 3 8", 5), (1, 3)
        )

    def test_invalid_neutral_quaternary_nitrogen_selects_n_plus(self) -> None:
        scene = quaternary_nitrogen_scene()
        signature = scene_topology_signature(scene)
        with self.assertRaisesRegex(Exception, "valence"):
            mol_from_chemscene_v0_1(scene)
        candidates = [
            [
                atom_state_candidate("N", -0.1, source="shifted"),
                atom_state_candidate("N+", -0.2, source="anchor", rank=1),
            ],
            [],
            [],
            [],
            [],
        ]
        repaired, diagnostics = rerank_invalid_scene_atom_states(scene, candidates)
        self.assertTrue(diagnostics["final_sanitize_valid"])
        self.assertEqual(repaired["junctions"][0]["atom"]["formal_charge"], 1)
        self.assertEqual(scene_topology_signature(repaired), signature)
        self.assertEqual(diagnostics["changes"][0]["to"], "N+")

    def test_valid_scene_is_never_changed(self) -> None:
        scene = quaternary_nitrogen_scene()
        scene["junctions"][0]["atom"]["formal_charge"] = 1
        scene["text_objects"][0]["text"] = "N+"
        candidates = [
            [
                atom_state_candidate("N", -0.01),
                atom_state_candidate("N+", -2.0),
            ],
            [],
            [],
            [],
            [],
        ]
        repaired, diagnostics = rerank_invalid_scene_atom_states(scene, candidates)
        self.assertIs(repaired, scene)
        self.assertTrue(diagnostics["base_sanitize_valid"])
        self.assertEqual(diagnostics["changes"], [])

    def test_unparsed_candidate_cannot_launder_invalid_scene(self) -> None:
        scene = quaternary_nitrogen_scene()
        candidates = [
            [
                atom_state_candidate("N", -0.2),
                atom_state_candidate("Ph", -0.1),
            ],
            [],
            [],
            [],
            [],
        ]
        repaired, diagnostics = rerank_invalid_scene_atom_states(scene, candidates)
        self.assertIs(repaired, scene)
        self.assertFalse(diagnostics["final_sanitize_valid"])


if __name__ == "__main__":
    unittest.main()
