from chemvector.training.atoms.train_g3_p9_stratified_pairwise_anchor import (
    EXPECTED_PER_CLASS,
    MINIMUM_PAIR_DISTANCE_PIXELS,
    _distance_band,
)


def test_p9_distance_strata_are_fixed() -> None:
    assert MINIMUM_PAIR_DISTANCE_PIXELS == 8.0
    assert _distance_band(0) is None
    assert _distance_band(1) is None
    assert _distance_band(2) == 0
    assert _distance_band(3) == 0
    assert _distance_band(4) == 1
    assert EXPECTED_PER_CLASS == 4960
