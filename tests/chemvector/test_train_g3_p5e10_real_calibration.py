from chemvector.training.atoms.train_g3_p5e10_real_calibration import (
    CANDIDATE_STEPS,
    MODEL_TYPE,
    select_candidate,
)


def test_p5e10_candidate_selection_is_frozen() -> None:
    assert MODEL_TYPE == "chemvector-g3-p5e10-real-calibrated-fit1378"
    assert CANDIDATE_STEPS == (0, 250, 500, 1_000, 2_000)
    metrics = {
        0: {"recovery_rate_6px": 0.70, "mae_pixels": 4.0},
        250: {"recovery_rate_6px": 0.82, "mae_pixels": 2.9},
        500: {"recovery_rate_6px": 0.85, "mae_pixels": 2.5},
    }
    assert select_candidate(metrics) == 250
    metrics[250] = {"recovery_rate_6px": 0.75, "mae_pixels": 3.2}
    metrics[500] = {"recovery_rate_6px": 0.78, "mae_pixels": 3.5}
    assert select_candidate(metrics) == 500
