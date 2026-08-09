from chemvector.evaluation.audits.atoms.build_g3_p11e2_predicted_source_full80k import (
    KEEP_PER_IMAGE,
    MINIMUM_MOVE,
    scaffold_split,
)


def test_p11e2_protocol_is_fixed() -> None:
    assert KEEP_PER_IMAGE == 8
    assert MINIMUM_MOVE == 1000
    assert scaffold_split("x") == scaffold_split("x")
    assert scaffold_split("x") in {"train", "calibration", "holdout"}
