"""Tests for graph-level benchmark error diagnostics."""

from __future__ import annotations

import unittest

from chemvector.benchmark_diagnostics import diagnose_chemsvg


def annotation(
    points: list[tuple[int, int]],
    elements: list[str],
    edges: list[tuple[int, int, str]],
) -> dict:
    return {
        "width": 1023,
        "height": 1023,
        "atoms": [
            {
                "atom_id": atom_id,
                "element": element,
                "formal_charge": 0,
                "isotope": 0,
                "image_x": x,
                "image_y": y,
            }
            for atom_id, ((x, y), element) in enumerate(zip(points, elements))
        ],
        "bonds": [
            {
                "bond_id": bond_id,
                "begin_atom": begin,
                "end_atom": end,
                "bond_type": bond_type,
            }
            for bond_id, (begin, end, bond_type) in enumerate(edges)
        ],
    }


def sequence(
    points: list[tuple[int, int]],
    elements: list[str],
    edges: list[tuple[int, int, str]],
) -> str:
    lines = ["<chemsvg v=0.4 units=q1023>"]
    lines.extend(
        f"<v {x} {y} {element} 0 0 0 0 0 0 CHI_UNSPECIFIED>"
        for (x, y), element in zip(points, elements)
    )
    lines.extend(
        f"<s {points[begin][0]} {points[begin][1]} "
        f"{points[end][0]} {points[end][1]} SINGLE {bond_type} NONE STEREONONE>"
        for begin, end, bond_type in edges
    )
    lines.append("<chemsvg_end>")
    return "\n".join(lines) + "\n"


class BenchmarkDiagnosticsTests(unittest.TestCase):
    def test_exact_graph_has_no_error_flags(self) -> None:
        points = [(100, 100), (200, 100), (300, 100)]
        edges = [(0, 1, "SINGLE"), (1, 2, "DOUBLE")]

        result = diagnose_chemsvg(
            annotation(points, ["C", "N", "C"], edges),
            sequence(points, ["C", "N", "C"], edges),
            sdf_valid=True,
        )

        self.assertTrue(result["graph_exact"])
        self.assertEqual(result["primary_error"], "exact")
        self.assertEqual(result["error_flags"], [])

    def test_element_substitution_is_classified_as_ocr(self) -> None:
        points = [(100, 100), (200, 100)]
        edges = [(0, 1, "SINGLE")]

        result = diagnose_chemsvg(
            annotation(points, ["C", "N"], edges),
            sequence(points, ["C", "C"], edges),
            sdf_valid=True,
        )

        self.assertEqual(result["element_mismatches"], 1)
        self.assertEqual(result["visible_element_mismatches"], 1)
        self.assertEqual(result["primary_error"], "ocr")

    def test_crossed_extra_bonds_are_counted(self) -> None:
        points = [(100, 100), (300, 100), (200, 200), (100, 300), (300, 300)]
        expected_edges = [
            (0, 2, "SINGLE"),
            (1, 2, "SINGLE"),
            (2, 3, "SINGLE"),
            (2, 4, "SINGLE"),
        ]
        predicted_edges = expected_edges + [
            (0, 4, "SINGLE"),
            (1, 3, "SINGLE"),
        ]

        result = diagnose_chemsvg(
            annotation(points, ["C"] * len(points), expected_edges),
            sequence(points, ["C"] * len(points), predicted_edges),
            sdf_valid=False,
            failure_category="valence",
        )

        self.assertEqual(result["extra_bonds"], 2)
        self.assertEqual(result["crossing_extra_bonds"], 2)
        self.assertEqual(result["extra_bonds_through_atom"], 2)
        self.assertEqual(result["primary_error"], "crossing_extra_bond")
        self.assertIn("invalid_valence", result["error_flags"])


if __name__ == "__main__":
    unittest.main()
