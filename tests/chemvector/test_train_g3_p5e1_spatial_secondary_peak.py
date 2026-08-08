import torch

from chemvector.training.atoms.train_g3_p5e1_spatial_secondary_peak_fit264 import (
    BATCH_SIZE,
    STEPS,
    _distance_metrics,
    _schedule,
)


def test_schedule_is_fixed_and_full_batch() -> None:
    first = _schedule(264)
    second = _schedule(264)
    assert len(first) == STEPS
    assert all(len(value) == BATCH_SIZE for value in first)
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_distance_metrics_report_recovery_and_variant_buckets() -> None:
    metrics = _distance_metrics(
        [1.0, 5.0, 7.0], ["rdkit", "rdkit", "stress"]
    )
    assert metrics["within_6px"] == 2
    assert metrics["recovery_rate_6px"] == 2 / 3
    assert metrics["by_variant"]["stress"]["within_6px"] == 0
