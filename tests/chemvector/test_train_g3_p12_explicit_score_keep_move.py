from chemvector.training.atoms.train_g3_p12_explicit_score_keep_move import (
    MINIMUM_CALIBRATION_RESCUES,
    STEPS,
)


def test_p12_protocol_is_fixed() -> None:
    assert STEPS == 8000
    assert MINIMUM_CALIBRATION_RESCUES == 40
