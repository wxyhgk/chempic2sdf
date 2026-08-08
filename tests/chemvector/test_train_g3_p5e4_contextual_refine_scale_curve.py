from chemvector.training.atoms.train_g3_p5e4_contextual_refine_scale_curve import (
    FULL_MODEL,
    FULL_NEW_ACTIONS,
    FULL_STEPS,
    MID_MODEL,
    MID_NEW_ACTIONS,
    MID_STEPS,
)


def test_p5e4_scale_curve_is_frozen() -> None:
    assert MID_MODEL == "chemvector-g3-p5e4-contextual-secondary-peak-fit528"
    assert FULL_MODEL == "chemvector-g3-p5e5-contextual-secondary-peak-fit1378"
    assert MID_NEW_ACTIONS == 264
    assert FULL_NEW_ACTIONS == 1_114
    assert MID_STEPS == 6_000
    assert FULL_STEPS == 15_660
