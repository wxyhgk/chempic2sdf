from __future__ import annotations

from chemvector.evaluation.audits.atoms.audit_g3_p4e5_peak_budget_dev1024 import (
    BUDGETS,
    decision_gate,
)


def _metrics(top1024_recall: float, replay: bool = True) -> dict:
    output = {
        str(budget): {"recall": 0.998, "matched": 1, "exact": 1}
        for budget in BUDGETS
    }
    output["128"].update(
        {"matched": 32_195 if replay else 32_194, "exact": 550}
    )
    output["1024"]["recall"] = top1024_recall
    return output


def test_gate_accepts_tail_ranking_capacity_at_top1024() -> None:
    decision, checks, minimum = decision_gate(_metrics(0.999))

    assert decision == "tail_ranking_capacity_go"
    assert all(checks.values())
    assert minimum == 1024


def test_gate_rejects_missing_regional_maxima_capacity() -> None:
    decision, checks, minimum = decision_gate(_metrics(0.9989))

    assert decision == "regional_maxima_representation_no_go"
    assert checks["top1024_recall_min999"] is False
    assert minimum is None


def test_gate_rejects_non_reproducible_top128() -> None:
    decision, checks, _ = decision_gate(_metrics(0.999, replay=False))

    assert decision == "regional_maxima_representation_no_go"
    assert checks["top128_replays_p4e3_final"] is False
