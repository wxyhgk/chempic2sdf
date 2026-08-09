from chemvector.training.atoms.train_g3_p13_local_peak_set_keep_move import (
    LOCAL_PEAK_COUNT,
    LOCAL_PEAK_KERNEL,
    MINIMUM_CALIBRATION_RESCUES,
    MODEL_TYPE,
    STEPS,
)


def test_p13_protocol_is_fixed() -> None:
    assert MODEL_TYPE == "chemvector-g3-p13-local-peak-set-keep-move-full2928"
    assert STEPS == 8000
    assert LOCAL_PEAK_COUNT == 16
    assert LOCAL_PEAK_KERNEL == 5
    assert MINIMUM_CALIBRATION_RESCUES == 40
