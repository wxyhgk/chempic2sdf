"""Tests for shared ChemVector data helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chemvector.data import (
    ChemVectorIterableDataset,
    load_reject_keys,
    rendered_atom_labels,
    structure_labels,
    synthesize_structure_labels,
)


class DataHelperTests(unittest.TestCase):
    def test_rendered_label_fallback_uses_explicit_atom_ids(self) -> None:
        sample = {
            "chemmol": {
                "atoms": [
                    {
                        "id": 4,
                        "element": "N",
                        "formal_charge": 0,
                        "implicit_h_count": 0,
                    },
                    {
                        "id": 1,
                        "element": "C",
                        "formal_charge": 0,
                        "implicit_h_count": 3,
                        "no_implicit": False,
                    },
                ]
            }
        }

        self.assertEqual(rendered_atom_labels(sample), {4: ("N", True), 1: ("", False)})

    def test_synthesized_structure_labels_join_render_and_chemmol_by_id(self) -> None:
        sample = {
            "key": "train_000001_original",
            "meta": {"split": "train"},
            "chemmol": {
                "molecule_id": "chembl:test",
                "atoms": [
                    {
                        "id": 0,
                        "element": "C",
                        "formal_charge": 0,
                        "implicit_h_count": 3,
                        "explicit_h_count": 0,
                        "isotope": 0,
                        "radical_electrons": 0,
                        "no_implicit": False,
                        "aromatic": False,
                    },
                    {
                        "id": 1,
                        "element": "N",
                        "formal_charge": 0,
                        "implicit_h_count": 1,
                        "explicit_h_count": 0,
                        "isotope": 0,
                        "radical_electrons": 0,
                        "no_implicit": False,
                        "aromatic": False,
                    },
                ],
                "bonds": [
                    {
                        "id": 7,
                        "begin_atom": 0,
                        "end_atom": 1,
                        "type": "SINGLE",
                    }
                ],
            },
            "render": {
                "molecule_id": "chembl:test",
                "variant": "original",
                "width": 768,
                "height": 768,
                "render_parameters": {"stress_operation": None},
                "atoms": [
                    {"atom_id": 1, "image_x": 30.0, "image_y": 40.0},
                    {"atom_id": 0, "image_x": 10.0, "image_y": 20.0},
                ],
                "bonds": [
                    {
                        "bond_id": 7,
                        "begin_image_x": 10.0,
                        "begin_image_y": 20.0,
                        "end_image_x": 30.0,
                        "end_image_y": 40.0,
                        "display_type": "SINGLE",
                        "direction": "NONE",
                    }
                ],
            },
        }

        labels = synthesize_structure_labels(sample)

        self.assertEqual(labels["atoms"][0]["center"], [10.0, 20.0])
        self.assertNotIn("text", labels["atoms"][0])
        self.assertNotIn("visible_text", labels["atoms"][1])
        self.assertEqual(labels["bonds"][0]["begin_atom"], 0)
        self.assertEqual(labels["bonds"][0]["end_atom"], 1)
        self.assertEqual(labels["bonds"][0]["chemical_type"], "SINGLE")
        self.assertIsNone(labels["bonds"][0]["bbox"])
        self.assertFalse(labels["text_supervision_available"])
        self.assertFalse(labels["text_bbox_supervision_available"])

    def test_structure_labels_prefers_existing_semantic_layers(self) -> None:
        layers = {"schema_version": "chemvision-layers-0.1"}
        self.assertIs(structure_labels({"layers": layers}), layers)

    def test_rendered_labels_override_semantic_display_approximation(self) -> None:
        sample = {
            "chemmol": {
                "atoms": [
                    {
                        "element": "C",
                        "formal_charge": 0,
                        "implicit_h_count": 1,
                        "no_implicit": True,
                    }
                ]
            },
            "layers": {
                "atoms": [
                    {
                        "atom_id": 0,
                        "visible_text": False,
                        "text": "",
                    }
                ]
            },
        }

        self.assertEqual(rendered_atom_labels(sample), {0: ("", False)})

    def test_variant_filter_keeps_only_requested_samples(self) -> None:
        dataset = ChemVectorIterableDataset(
            "unused.tar",
            allowed_variants={"stress"},
        )

        samples = [
            {"key": "easy", "meta": {"variant": "original"}},
            {"key": "hard", "meta": {"variant": "stress"}},
        ]
        with patch("chemvector.data.ChemMolWebDataset", return_value=iter(samples)):
            selected = list(dataset)

        self.assertEqual([sample["key"] for sample in selected], ["hard"])

    def test_iterable_dataset_forwards_deterministic_shuffle_seed(self) -> None:
        dataset = ChemVectorIterableDataset(
            "unused.tar",
            sample_shuffle_buffer=32,
            sample_shuffle_seed=17,
        )

        with patch("chemvector.data.ChemMolWebDataset", return_value=iter(())) as loader:
            list(dataset)

        self.assertEqual(loader.call_args.kwargs["sample_shuffle_buffer"], 32)
        self.assertEqual(loader.call_args.kwargs["sample_shuffle_seed"], 17)

    def test_reject_key_file_ignores_comments_blanks_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reject_keys.txt"
            path.write_text(
                "# generated audit\ntrain_000001\n\ntrain_000002\ntrain_000001\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_reject_keys(path),
                {"train_000001", "train_000002"},
            )

    def test_missing_reject_key_argument_means_no_filter(self) -> None:
        self.assertEqual(load_reject_keys(None), set())


if __name__ == "__main__":
    unittest.main()
