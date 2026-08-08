from chemvector.components.recovery.structural_context_peak import StructuralContextPeakHead
from chemvector.training.atoms.train_g3_p6e1_structural_context96_mixed import (
    GENERIC_ACTIONS,
    MIXED_STEPS,
    MODEL_TYPE,
    PRETRAIN_STEPS,
    REAL_ACTIONS,
)


def test_p6e1_training_protocol_is_frozen() -> None:
    assert MODEL_TYPE == "chemvector-g3-p6e1-structural-context96-mixed50k-fit1378"
    assert GENERIC_ACTIONS == 50_000
    assert REAL_ACTIONS == 1_378
    assert PRETRAIN_STEPS == 10_000
    assert MIXED_STEPS == 5_000
    assert StructuralContextPeakHead().parameter_count == 150_365
