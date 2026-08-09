from chemvector.evaluation.audits.atoms.build_g3_p6e8_residual_cycle_data import (
    choose_residual_cap, scaffold_split,
)


def test_p6e8_cap_selects_smallest_95pct_option() -> None:
    assert choose_residual_cap([0.5] * 95 + [7.0] * 5) == 1.0
    assert choose_residual_cap([8.1] * 96 + [9.0] * 4) is None


def test_p6e8_scaffold_split_is_deterministic() -> None:
    assert scaffold_split("c1ccccc1") == scaffold_split("c1ccccc1")
    assert scaffold_split("c1ccccc1") in {"train", "calibration", "holdout"}
