from __future__ import annotations

from chemvector.training.atoms.train_g3_p3e27_query_slot_occupancy_micro160 import (
    BATCH_SIZE,
    EXPOSURE,
    STEPS,
    training_batches,
)


def test_query_slot_schedule_is_fixed() -> None:
    batches, audit = training_batches()
    assert len(batches) == STEPS == 4000
    assert audit["exposure"] == EXPOSURE == 16000
    assert all(len(batch) == BATCH_SIZE == 4 for batch in batches)
    assert len(audit["epoch_order_sha256"]) == 100
