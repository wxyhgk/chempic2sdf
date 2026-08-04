"""Tests for the deterministic P3-E7 zero-break threshold router."""

import unittest

from chemvector.training.atoms.audit_g3_p3e7_base_expert_router import (
    apply_policy,
    select_zero_break_policy,
)


def row(key, feature, base, expert):
    return {
        "sample_key": key,
        "features": {name: feature for name in (
            "expert_minus_base_joint_score_mean",
            "expert_joint_score_mean",
            "base_joint_score_mean",
            "expert_minus_base_existence_mean",
            "expert_minus_base_pointer_logprob_mean",
            "candidate_overlap_fraction",
            "candidate_replacements",
            "expert_assignment_stable",
            "base_assignment_stable",
        )},
        "base": {"correct": base},
        "expert": {"correct": expert},
    }


class G3P3E7RouterTests(unittest.TestCase):
    def test_select_policy_maximizes_rescue_without_break(self) -> None:
        rows = [
            row("rescue-high", 3.0, False, True),
            row("rescue-mid", 2.0, False, True),
            row("break-low", 1.0, True, False),
            row("both", 0.0, True, True),
        ]
        policy, metrics = select_zero_break_policy(rows)
        self.assertIsNotNone(policy)
        self.assertEqual(metrics["rescues"], 2)
        self.assertEqual(metrics["breaks"], 0)
        self.assertEqual(apply_policy(rows, policy), metrics)

    def test_apply_none_keeps_base(self) -> None:
        rows = [row("base", 1.0, True, False), row("expert", 2.0, False, True)]
        metrics = apply_policy(rows, None)
        self.assertEqual(metrics["exact"], 1)
        self.assertEqual(metrics["routed"], 0)


if __name__ == "__main__":
    unittest.main()
