"""Tests for P3-E17 Count/router decomposition."""

import unittest

from chemvector.training.atoms.audit_g3_p3e17_p3e16_count_router import (
    POLICY,
    count_summary,
    decompose,
)


def row(key: str, *, count: int, atom_count: int, base: bool, expert: bool, score: float):
    return {
        "sample_key": key,
        "atom_count": atom_count,
        "base": {"decoded_count": count, "correct": base},
        "expert": {"decoded_count": count, "correct": expert},
        "features": {POLICY["feature"]: score},
    }


class G3P3E17CountRouterTests(unittest.TestCase):
    def test_count_summary(self) -> None:
        rows = [
            row("a", count=10, atom_count=10, base=True, expert=True, score=0.0),
            row("b", count=8, atom_count=10, base=False, expert=False, score=0.0),
        ]
        metrics = count_summary(rows)
        self.assertEqual(metrics["count_exact"], 1)
        self.assertEqual(metrics["count_mae"], 1.0)

    def test_decomposition_attributes_cardinality_loss(self) -> None:
        known = [
            row("a", count=10, atom_count=10, base=True, expert=True, score=0.0),
            row("b", count=10, atom_count=10, base=True, expert=True, score=0.0),
        ]
        predicted = [
            row("a", count=10, atom_count=10, base=True, expert=True, score=0.0),
            row("b", count=9, atom_count=10, base=False, expert=False, score=0.0),
        ]
        metrics = decompose(known, predicted, POLICY)
        self.assertEqual(metrics["known_router_exact_and_count_correct"], 1)
        self.assertEqual(metrics["known_router_exact_lost_to_count"], 1)
        self.assertTrue(metrics["predicted_router_exact_keys_equal_intersection"])


if __name__ == "__main__":
    unittest.main()
