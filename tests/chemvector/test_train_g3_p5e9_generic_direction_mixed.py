from chemvector.training.atoms.train_g3_p5e9_generic_direction_mixed import (
    BATCH_SIZE,
    GENERIC_ACTIONS,
    MIXED_HALF_BATCH,
    MIXED_STEPS,
    MODEL_TYPE,
    PRETRAIN_STEPS,
    REAL_ACTIONS,
    _schedule,
)


def test_p5e9_protocol_and_schedule_are_frozen() -> None:
    assert MODEL_TYPE == "chemvector-g3-p5e9-generic-direction-mixed50k-fit1378"
    assert GENERIC_ACTIONS == 50_000
    assert REAL_ACTIONS == 1_378
    assert PRETRAIN_STEPS == 10_000
    assert MIXED_STEPS == 5_000
    assert BATCH_SIZE == 64
    assert MIXED_HALF_BATCH == 32
    first = _schedule(17, 5, 8, 123)
    second = _schedule(17, 5, 8, 123)
    assert [row.tolist() for row in first] == [row.tolist() for row in second]
