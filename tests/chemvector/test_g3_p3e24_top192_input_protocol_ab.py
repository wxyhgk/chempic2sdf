from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p3e24_top192_input_protocol_ab import (
    arm_checks,
    prefix_normalized_scores,
    prefix_scale_ranks,
)


def test_prefix_protocol_preserves_first128_normalization() -> None:
    scores = torch.linspace(0.99, 0.01, 192)
    current = prefix_normalized_scores(scores)
    logits = torch.logit(scores[:128].clamp(1e-6, 1 - 1e-6))
    expected = (logits - logits.mean()) / logits.std(unbiased=False).clamp_min(1e-4)
    assert torch.equal(current[:128], expected)
    assert torch.equal(prefix_scale_ranks()[:128], torch.arange(128).float() / 127)
    assert prefix_scale_ranks()[-1] > 1


def test_arm_checks_apply_preregistered_dev_limits() -> None:
    baseline = {
        "router": {"images": 100, "exact": 50, "matched_atoms": 980},
        "base": {"exact": 40},
        "expert": {"exact": 45},
    }
    passing = {
        "router": {"images": 100, "exact": 48, "matched_atoms": 1050},
        "base": {"exact": 37},
        "expert": {"exact": 42},
    }
    utility = {
        "matched_atom_delta": 70,
        "exact_rescues": 3,
        "exact_breaks": 2,
        "top128_oracle_protected_breaks": 2,
        "correct_tail_hits": 1,
        "newly_reachable_exact_rescues": 1,
    }
    checks = arm_checks(passing, baseline, {"atom_recall": 0.99}, utility)
    assert all(checks.values())
    passing["router"]["exact"] = 37
    assert not arm_checks(passing, baseline, {"atom_recall": 0.99}, utility)[
        "router_exact_net_loss_max12"
    ]
