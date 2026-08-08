from chemvector.training.atoms.train_g3_p6e5_peak_blind_balanced import (
    EASY_ACTIONS, HARD_ACTIONS, HARD_GENERIC, HARD_REAL, MODEL_TYPE, STEPS,
)


def test_p6e5_protocol_is_frozen() -> None:
    assert MODEL_TYPE == "chemvector-g3-p6e5-peak-blind-balanced-hard294-easy294"
    assert HARD_GENERIC == 169
    assert HARD_REAL == 125
    assert HARD_ACTIONS == EASY_ACTIONS == 294
    assert STEPS == 5_000
