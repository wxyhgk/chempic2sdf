"""Tests for the route-independent external ChemScene Junction adapter."""

import unittest

import torch

from chemvector.inference.infer_chemscene_hybrid import (
    IMPLICIT_ROLE,
    components_from_external_junctions,
)


class ExternalChemSceneJunctionTests(unittest.TestCase):
    def test_without_ocr_preserves_points_and_bonds(self) -> None:
        points = torch.tensor([[0.1, 0.2], [0.8, 0.9]])
        bonds = [{"begin": [0.1, 0.2], "end": [0.8, 0.9]}]
        actual_points, roles, labels, actual_bonds = components_from_external_junctions(
            points, bonds, include_ocr=False
        )
        self.assertTrue(torch.equal(actual_points, points))
        self.assertEqual(roles.tolist(), [IMPLICIT_ROLE, IMPLICIT_ROLE])
        self.assertEqual(labels, ["C", "C"])
        self.assertEqual(actual_bonds, bonds)

    def test_ocr_attaches_detected_text_without_moving_points(self) -> None:
        points = torch.tensor([[0.1, 0.2], [0.8, 0.9]])
        detected_points = torch.tensor([[0.101, 0.201]])
        detected_roles = torch.tensor([1])
        actual_points, roles, labels, _ = components_from_external_junctions(
            points,
            [],
            include_ocr=True,
            detected_points=detected_points,
            detected_roles=detected_roles,
            detected_labels=["N"],
            image_size=768,
        )
        self.assertTrue(torch.equal(actual_points, points))
        self.assertEqual(labels[0], "N")
        self.assertEqual(int(roles[0]), 1)

    def test_invalid_external_points_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            components_from_external_junctions(
                torch.tensor([[1.1, 0.2]]), [], include_ocr=False
            )


if __name__ == "__main__":
    unittest.main()
