from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
)
from chemvector.training.atoms.train_g3_p5e2_contextual_secondary_peak_fit264 import (
    BATCH_SIZE,
    MODEL_TYPE,
    STEPS,
)


def test_p5e2_frozen_capacity_and_identity() -> None:
    model = ContextualSecondaryPeakHead()
    assert MODEL_TYPE == "chemvector-g3-p5e2-contextual-secondary-peak-fit264"
    assert model.parameter_count < 100_000
    assert BATCH_SIZE == 64
    assert STEPS == 3_000
