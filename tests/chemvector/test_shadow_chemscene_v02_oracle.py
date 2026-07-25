import unittest
from copy import deepcopy

import torch

from Data.sdf.chemscene_v0_2 import parse_chemscene_v0_2
from chemvector.shadow_chemscene_v02_oracle import (
    apply_aromatic_consistency,
    apply_oracle_object_controls,
    build_oracle_chemistry_shadow_scene,
    clear_noncycle_aromatic_atoms,
    demote_noncycle_aromatic_bonds,
)


class ShadowChemSceneV02OracleTests(unittest.TestCase):
    @staticmethod
    def _graph_scene(edges: list[tuple[int, int, str, str]]) -> dict:
        atom_count = 1 + max(max(begin, end) for begin, end, _, _ in edges)
        return {
            "schema_version": "chemscene-0.2",
            "molecule_id": "graph",
            "units": "q1023",
            "canvas": {"width": 768, "height": 768},
            "atom_objects": [
                {
                    "id": index,
                    "anchor": [index * 100, 100],
                    "chemical": {"aromatic": True},
                }
                for index in range(atom_count)
            ],
            "bond_objects": [
                {
                    "id": index,
                    "atom_refs": [begin, end],
                    "display_type": display,
                    "chemical_type": chemical,
                }
                for index, (begin, end, display, chemical) in enumerate(edges)
            ],
        }

    def test_noncycle_aromatic_bond_is_demoted_without_mutating_input(self) -> None:
        scene = self._graph_scene([(0, 1, "DOUBLE", "AROMATIC")])
        original = deepcopy(scene)

        repaired, diagnostics = demote_noncycle_aromatic_bonds(scene)

        self.assertEqual(scene, original)
        self.assertEqual(repaired["bond_objects"][0]["chemical_type"], "DOUBLE")
        self.assertEqual(diagnostics["demoted_aromatic_bond_count"], 1)
        self.assertEqual(diagnostics["demoted_aromatic_bond_ids"], [0])

    def test_cycle_aromatic_bonds_are_preserved(self) -> None:
        scene = self._graph_scene(
            [
                (0, 1, "SINGLE", "AROMATIC"),
                (1, 2, "SINGLE", "AROMATIC"),
                (2, 0, "SINGLE", "AROMATIC"),
            ]
        )

        repaired, diagnostics = demote_noncycle_aromatic_bonds(scene)

        self.assertTrue(
            all(
                bond["chemical_type"] == "AROMATIC"
                for bond in repaired["bond_objects"]
            )
        )
        self.assertEqual(diagnostics["cycle_bond_count"], 3)
        self.assertEqual(diagnostics["demoted_aromatic_bond_count"], 0)

    def test_nonaromatic_bridge_is_unchanged(self) -> None:
        scene = self._graph_scene([(0, 1, "SINGLE", "SINGLE")])

        repaired, diagnostics = demote_noncycle_aromatic_bonds(scene)

        self.assertEqual(repaired["bond_objects"][0]["chemical_type"], "SINGLE")
        self.assertEqual(diagnostics["demoted_aromatic_bond_count"], 0)

    def test_none_mode_returns_unmodified_copy(self) -> None:
        scene = self._graph_scene([(0, 1, "SINGLE", "AROMATIC")])

        repaired, diagnostics = apply_aromatic_consistency(scene, "none")
        repaired["bond_objects"][0]["chemical_type"] = "SINGLE"

        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "AROMATIC")
        self.assertEqual(diagnostics["mode"], "none")

    def test_clear_aromatic_flag_only_for_atom_outside_cycle(self) -> None:
        scene = self._graph_scene(
            [
                (0, 1, "SINGLE", "AROMATIC"),
                (1, 2, "SINGLE", "AROMATIC"),
                (2, 0, "SINGLE", "AROMATIC"),
                (2, 3, "SINGLE", "AROMATIC"),
            ]
        )

        repaired, diagnostics = clear_noncycle_aromatic_atoms(scene)

        self.assertTrue(
            all(
                repaired["atom_objects"][index]["chemical"]["aromatic"]
                for index in (0, 1, 2)
            )
        )
        self.assertFalse(repaired["atom_objects"][3]["chemical"]["aromatic"])
        self.assertEqual(diagnostics["cleared_noncycle_aromatic_atom_ids"], [3])
        self.assertTrue(scene["atom_objects"][3]["chemical"]["aromatic"])

    def test_combined_mode_demotes_bond_and_clears_atom(self) -> None:
        scene = self._graph_scene([(0, 1, "SINGLE", "AROMATIC")])

        repaired, diagnostics = apply_aromatic_consistency(
            scene, "demote-noncycle-clear-atoms"
        )

        self.assertEqual(repaired["bond_objects"][0]["chemical_type"], "SINGLE")
        self.assertFalse(repaired["atom_objects"][0]["chemical"]["aromatic"])
        self.assertFalse(repaired["atom_objects"][1]["chemical"]["aromatic"])
        self.assertEqual(diagnostics["demoted_aromatic_bond_count"], 1)
        self.assertEqual(diagnostics["cleared_noncycle_aromatic_atom_count"], 2)

    def test_selected_objects_compile_with_oracle_atom_chemistry(self) -> None:
        gt_scene = {
            "schema_version": "chemscene-0.2",
            "molecule_id": "test",
            "units": "q1023",
            "canvas": {"width": 768, "height": 768},
            "atom_objects": [
                {
                    "id": 0,
                    "anchor": [205, 512],
                    "role": "IMPLICIT",
                    "chemical": {
                        "element": "C",
                        "formal_charge": 0,
                        "isotope": 0,
                        "aromatic": False,
                        "explicit_h_count": 0,
                        "no_implicit": False,
                        "radical_electrons": 0,
                        "canonical_chiral_tag": "CHI_UNSPECIFIED",
                    },
                    "text": None,
                    "incident_ports": [],
                },
                {
                    "id": 1,
                    "anchor": [818, 512],
                    "role": "TEXT",
                    "chemical": {
                        "element": "O",
                        "formal_charge": 0,
                        "isotope": 0,
                        "aromatic": False,
                        "explicit_h_count": 0,
                        "no_implicit": False,
                        "radical_electrons": 0,
                        "canonical_chiral_tag": "CHI_UNSPECIFIED",
                    },
                    "text": {"value": "O", "center": [818, 512], "bbox": [790, 484, 846, 540]},
                    "incident_ports": [],
                },
            ],
            "bond_objects": [],
        }
        atom_geometry = torch.tensor(
            [
                [0.20, 0.50, 1, 0, 0.20, 0.50, 0.02, 0.02],
                [0.80, 0.50, 0, 1, 0.80, 0.50, 0.03, 0.03],
                [0.50, 0.20, 1, 0, 0.50, 0.20, 0.02, 0.02],
            ]
        )
        bond_types = torch.tensor(
            [[1.0, 0, 0, 0, 1.0, 0, 0, 0], [1.0, 0, 0, 0, 1.0, 0, 0, 0]]
        )
        incidence = torch.tensor(
            [
                [[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )

        scene, diagnostics = build_oracle_chemistry_shadow_scene(
            molecule_id="test",
            gt_scene=gt_scene,
            atom_geometry=atom_geometry,
            atom_mask=torch.tensor([True, True, True]),
            atom_presence_logits=torch.tensor([5.0, 5.0, -5.0]),
            bond_types=bond_types,
            bond_mask=torch.tensor([True, True]),
            bond_presence_logits=torch.tensor([5.0, -5.0]),
            incidence_logits=incidence,
            atom_threshold=0.0,
            bond_threshold=0.0,
        )

        parsed = parse_chemscene_v0_2(scene)
        self.assertEqual([atom.element for atom in parsed.atom_objects], ["C", "O"])
        self.assertEqual(len(parsed.bond_objects), 1)
        self.assertEqual(diagnostics["oracle_matched_atoms"], 2)

    def test_oracle_matched_bond_chemistry_changes_only_matched_gt_pair(self) -> None:
        gt_scene = {
            "schema_version": "chemscene-0.2",
            "molecule_id": "test",
            "units": "q1023",
            "canvas": {"width": 768, "height": 768},
            "atom_objects": [
                {
                    "id": index,
                    "anchor": anchor,
                    "role": "IMPLICIT",
                    "chemical": {
                        "element": "C",
                        "formal_charge": 0,
                        "isotope": 0,
                        "aromatic": True,
                        "explicit_h_count": 0,
                        "no_implicit": False,
                        "radical_electrons": 0,
                        "canonical_chiral_tag": "CHI_UNSPECIFIED",
                    },
                    "text": None,
                    "incident_ports": [],
                }
                for index, anchor in enumerate(([205, 512], [818, 512]))
            ],
            "bond_objects": [
                {"id": 0, "atom_refs": [0, 1], "chemical_type": "AROMATIC"}
            ],
        }
        scene, diagnostics = build_oracle_chemistry_shadow_scene(
            molecule_id="test",
            gt_scene=gt_scene,
            atom_geometry=torch.tensor(
                [
                    [0.20, 0.50, 1, 0, 0.20, 0.50, 0.02, 0.02],
                    [0.80, 0.50, 1, 0, 0.80, 0.50, 0.02, 0.02],
                ]
            ),
            atom_mask=torch.tensor([True, True]),
            atom_presence_logits=torch.tensor([5.0, 5.0]),
            bond_types=torch.tensor([[1.0, 0, 0, 0, 1.0, 0, 0, 0]]),
            bond_mask=torch.tensor([True]),
            bond_presence_logits=torch.tensor([5.0]),
            incidence_logits=torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]]),
            atom_threshold=0.0,
            bond_threshold=0.0,
            bond_chemistry_mode="oracle-matched",
        )

        self.assertEqual(scene["bond_objects"][0]["display_type"], "SINGLE")
        self.assertEqual(scene["bond_objects"][0]["chemical_type"], "AROMATIC")
        self.assertEqual(diagnostics["oracle_matched_bond_chemistry"], 1)

    def test_oracle_object_controls_replace_only_requested_predictions(self) -> None:
        atom, bond, incidence = apply_oracle_object_controls(
            atom_presence_logits=torch.tensor([0.2, 0.3]),
            bond_presence_logits=torch.tensor([0.4]),
            incidence_logits=torch.zeros(1, 2, 2),
            batch={
                "atom_presence_targets": torch.tensor([[True, False]]),
                "bond_presence_targets": torch.tensor([[True]]),
                "bond_supervision_mask": torch.tensor([[True]]),
                "target_atom_refs": torch.tensor([[[1, 0]]]),
            },
            batch_index=0,
            object_presence_mode="oracle",
            incidence_mode="oracle-eligible",
        )

        self.assertEqual(atom.tolist(), [10.0, -10.0])
        self.assertEqual(bond.tolist(), [10.0])
        self.assertEqual(incidence[0].argmax(-1).tolist(), [1, 0])

    def test_selected_bond_can_rescue_borderline_endpoint_atom(self) -> None:
        gt_scene = {
            "schema_version": "chemscene-0.2",
            "molecule_id": "test",
            "units": "q1023",
            "canvas": {"width": 768, "height": 768},
            "atom_objects": [
                {
                    "id": index,
                    "anchor": anchor,
                    "role": "IMPLICIT",
                    "chemical": {
                        "element": "C",
                        "formal_charge": 0,
                        "isotope": 0,
                        "aromatic": False,
                        "explicit_h_count": 0,
                        "no_implicit": False,
                        "radical_electrons": 0,
                        "canonical_chiral_tag": "CHI_UNSPECIFIED",
                    },
                    "text": None,
                    "incident_ports": [],
                }
                for index, anchor in enumerate(([205, 512], [818, 512]))
            ],
            "bond_objects": [],
        }
        scene, diagnostics = build_oracle_chemistry_shadow_scene(
            molecule_id="test",
            gt_scene=gt_scene,
            atom_geometry=torch.tensor(
                [
                    [0.20, 0.50, 1, 0, 0.20, 0.50, 0.02, 0.02],
                    [0.80, 0.50, 1, 0, 0.80, 0.50, 0.02, 0.02],
                ]
            ),
            atom_mask=torch.tensor([True, True]),
            atom_presence_logits=torch.tensor([1.0, -0.1]),
            bond_types=torch.tensor([[1.0, 0, 0, 0, 1.0, 0, 0, 0]]),
            bond_mask=torch.tensor([True]),
            bond_presence_logits=torch.tensor([1.0]),
            incidence_logits=torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]]),
            atom_threshold=0.0,
            bond_threshold=0.0,
            endpoint_atom_rescue_margin=0.2,
        )

        self.assertEqual(len(scene["atom_objects"]), 2)
        self.assertEqual(len(scene["bond_objects"]), 1)
        self.assertEqual(diagnostics["rescued_endpoint_atom_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
