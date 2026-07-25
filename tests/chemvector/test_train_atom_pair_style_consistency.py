from __future__ import annotations

import unittest

from chemvector.train_atom_pair_style_consistency import selection_key


def _official(chemvision: int, full_v0: int) -> dict:
    return {
        "domains": {
            "chemvision": {
                "deployed_exact": chemvision,
                "chemical_deployed_exact": chemvision,
            },
            "full_v0": {
                "deployed_exact": full_v0,
                "chemical_deployed_exact": full_v0,
            },
        },
        "pooled_deployed_exact_rate": (chemvision + full_v0) / 1000,
    }


def _style(all_exact: int, mixed: int, exact: int) -> dict:
    return {
        "candidate_style": {
            "paired_renderings": {
                "presence_exact": {
                    "all_renderings_exact": all_exact,
                    "mixed_outcome": mixed,
                }
            },
            "overall": {"presence_exact": exact},
        }
    }


class TrainAtomPairStyleConsistencyTest(unittest.TestCase):
    def test_selection_rejects_official_domain_regression(self) -> None:
        baseline = _official(350, 360)
        unsafe = selection_key(_official(349, 370), _style(40, 20, 300), baseline)
        safe = selection_key(_official(350, 360), _style(30, 30, 280), baseline)
        self.assertGreater(safe, unsafe)

    def test_selection_prefers_more_all_rendering_exact_when_safe(self) -> None:
        baseline = _official(350, 360)
        lower = selection_key(baseline, _style(30, 30, 280), baseline)
        higher = selection_key(baseline, _style(31, 35, 279), baseline)
        self.assertGreater(higher, lower)


if __name__ == "__main__":
    unittest.main()
