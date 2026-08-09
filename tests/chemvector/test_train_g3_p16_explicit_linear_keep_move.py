from chemvector.training.atoms.train_g3_p16_explicit_linear_keep_move import (
    EXPECTED_PER_CLASS,
    L2,
    MINIMUM_CALIBRATION_RESCUES,
    MODEL_TYPE,
    NEWTON_STEPS,
)


def test_p16_protocol_is_fixed() -> None:
    assert MODEL_TYPE == "chemvector-g3-p16-explicit-linear-keep-move-fit562"
    assert EXPECTED_PER_CLASS == 281
    assert L2 == 0.01
    assert NEWTON_STEPS == 64
    assert MINIMUM_CALIBRATION_RESCUES == 20
