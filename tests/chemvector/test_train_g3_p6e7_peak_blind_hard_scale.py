from chemvector.training.atoms.train_g3_p6e7_peak_blind_hard_scale import (
    ARMS, EXPOSURES_PER_HARD, HARD_BATCH,
)


def test_p6e7_scale_protocol_is_frozen() -> None:
    assert ARMS == (500, 1339)
    assert EXPOSURES_PER_HARD == 192
    assert 500 * EXPOSURES_PER_HARD // HARD_BATCH == 3_000
    assert 1339 * EXPOSURES_PER_HARD // HARD_BATCH == 8_034
