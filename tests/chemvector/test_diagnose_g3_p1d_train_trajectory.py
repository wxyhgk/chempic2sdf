import unittest

from chemvector.evaluation.audits.atoms.diagnose_g3_p1d_train_trajectory import (
    _add_totals,
    _empty_totals,
    _rates,
)


class G3P1DTrajectoryTests(unittest.TestCase):
    def test_totals_and_rates_preserve_set_metrics(self) -> None:
        totals = _empty_totals()
        _add_totals(
            totals,
            {
                "images": 4,
                "positive_count": 10,
                "matched_positive_count": 8,
                "whole_set_exact_images": 2,
                "boundary_exact_images": 1,
            },
        )

        result = _rates(totals)

        self.assertEqual(result["hybrid_topm_recall"], 0.8)
        self.assertEqual(result["hybrid_topm_whole_set_rate"], 0.5)
        self.assertEqual(result["boundary_exact_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
