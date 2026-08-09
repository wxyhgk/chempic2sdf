from chemvector.training.atoms.train_g3_p11e3_predicted_source_keep_move_full import (
    EQUIVALENT_POOL,
    EXPECTED_PER_CLASS,
    MINIMUM_CALIBRATION_RESCUES,
)


def test_p11e3_protocol_is_fixed() -> None:
    assert EXPECTED_PER_CLASS == 1464
    assert EQUIVALENT_POOL == 2048
    assert MINIMUM_CALIBRATION_RESCUES == 40
