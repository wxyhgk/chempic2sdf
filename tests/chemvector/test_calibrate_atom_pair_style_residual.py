from __future__ import annotations

import unittest

from chemvector.calibrate_atom_pair_style_residual import selection_key


def _result(scale: float, chemvision: int, full_v0: int, all_exact: int) -> dict:
    return {
        "adapter_scale": scale,
        "official": {
            "chemvision": {
                "presence_exact": chemvision,
                "chemical_exact": chemvision,
            },
            "full_v0": {
                "presence_exact": full_v0,
                "chemical_exact": full_v0,
            },
        },
        "style": {
            "overall": {"presence_exact": 300},
            "paired_renderings": {
                "presence_exact": {
                    "all_renderings_exact": all_exact,
                    "mixed_outcome": 20,
                }
            },
        },
    }


class CalibrateAtomPairStyleResidualTest(unittest.TestCase):
    def test_safe_scale_beats_style_gain_with_official_regression(self) -> None:
        baseline = _result(0.0, 100, 100, 20)
        unsafe = _result(1.0, 99, 101, 30)
        safe = _result(0.5, 100, 100, 21)
        self.assertGreater(selection_key(safe, baseline), selection_key(unsafe, baseline))


if __name__ == "__main__":
    unittest.main()
