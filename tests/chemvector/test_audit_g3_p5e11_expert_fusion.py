from chemvector.evaluation.audits.atoms.audit_g3_p5e11_expert_fusion import (
    REAL_WEIGHTS,
    select_weight,
)


def test_p5e11_weight_selection_is_frozen() -> None:
    assert REAL_WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0)
    metrics = {
        0.0: {"recovery_rate_6px": 0.75, "mae_pixels": 3.5},
        0.25: {"recovery_rate_6px": 0.82, "mae_pixels": 2.9},
        0.5: {"recovery_rate_6px": 0.85, "mae_pixels": 2.8},
    }
    assert select_weight(metrics) == 0.5
    metrics[0.25] = {"recovery_rate_6px": 0.85, "mae_pixels": 2.8}
    assert select_weight(metrics) == 0.25
