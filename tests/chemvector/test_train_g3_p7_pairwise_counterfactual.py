import math

from chemvector.training.atoms.train_g3_p7_pairwise_counterfactual import (
    CAPACITY_ACCURACY,
    CAPACITY_DIRECTED,
    FOLDS,
    MINIMUM_RESCUES,
    calibrate_zero_harmful,
    scaffold_fold,
)


def test_p7_training_protocol_is_fixed() -> None:
    assert FOLDS == 5
    assert CAPACITY_DIRECTED == 256
    assert CAPACITY_ACCURACY == 0.99
    assert MINIMUM_RESCUES == 100
    assert scaffold_fold("abc") == scaffold_fold("abc")


def test_p7_zero_harmful_calibration_prefers_rescues() -> None:
    records = [
        {"score": 0.9, "prior_good": False, "alternative_good": True},
        {"score": 0.8, "prior_good": False, "alternative_good": True},
        {"score": 0.7, "prior_good": True, "alternative_good": False},
        {"score": 0.6, "prior_good": True, "alternative_good": True},
    ]
    winner = calibrate_zero_harmful(records)
    assert winner == {
        "threshold": 0.8,
        "executed": 2,
        "rescues": 2,
        "harmful": 0,
        "useless": 0,
    }


def test_p7_empty_calibration_selects_no_action() -> None:
    winner = calibrate_zero_harmful([])
    assert math.isinf(winner["threshold"])
    assert winner["executed"] == 0
