from chemvector.training.atoms.train_g3_p14_full_top128_context_keep_move import (
    CAPACITY_STEPS,
    EXPECTED_DIRECTED_PER_CLASS,
    FULL_STEPS,
    MINIMUM_CALIBRATION_RESCUES,
    MODEL_TYPE,
)


def test_p14_protocol_is_fixed() -> None:
    assert MODEL_TYPE == "chemvector-g3-p14-full-top128-context-keep-move-fit562"
    assert CAPACITY_STEPS == 1500
    assert FULL_STEPS == 6000
    assert EXPECTED_DIRECTED_PER_CLASS == 281
    assert MINIMUM_CALIBRATION_RESCUES == 20
