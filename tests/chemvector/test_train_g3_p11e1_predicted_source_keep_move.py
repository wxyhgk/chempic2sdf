from chemvector.training.atoms.train_g3_p11e1_predicted_source_keep_move import (
    EXPECTED_DIRECTED_PER_CLASS,
    MINIMUM_CALIBRATION_RESCUES,
    calibrate_nonnegative,
    scaffold_split,
)


def test_p11e1_protocol_is_fixed() -> None:
    assert EXPECTED_DIRECTED_PER_CLASS == 281
    assert MINIMUM_CALIBRATION_RESCUES == 20
    assert scaffold_split("x") == scaffold_split("x")


def test_p11e1_calibration_never_uses_negative_threshold() -> None:
    records = [
        {"score": 1.0, "prior_good": False, "alternative_good": True},
        {"score": 0.5, "prior_good": True, "alternative_good": False},
        {"score": -0.1, "prior_good": False, "alternative_good": True},
    ]
    winner = calibrate_nonnegative(records)
    assert winner["threshold"] == 1.0
    assert winner["rescues"] == 1
    assert winner["harmful"] == 0
