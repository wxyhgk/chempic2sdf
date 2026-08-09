from chemvector.evaluation.audits.atoms.evaluate_g3_p7e4_pairwise_holdout import (
    MINIMUM_RESCUES,
    THRESHOLD,
    fixed_threshold_metrics,
)


def test_p7_holdout_protocol_is_fixed() -> None:
    assert MINIMUM_RESCUES == 100
    assert THRESHOLD == 1.6568973064422607


def test_p7_holdout_uses_one_frozen_threshold() -> None:
    records = [
        {"score": 2.0, "prior_good": False, "alternative_good": True},
        {"score": 1.8, "prior_good": True, "alternative_good": True},
        {"score": 1.7, "prior_good": True, "alternative_good": False},
        {"score": 1.0, "prior_good": False, "alternative_good": True},
    ]
    metrics = fixed_threshold_metrics(records, THRESHOLD)
    assert metrics == {
        "threshold": THRESHOLD,
        "executed": 3,
        "rescues": 1,
        "harmful": 1,
        "useless": 1,
    }
