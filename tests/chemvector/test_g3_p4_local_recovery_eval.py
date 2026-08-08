import torch

from chemvector.evaluation.evaluators.atoms.g3_p4_local_recovery import (
    calibrate_zero_useless_threshold,
    evaluate_threshold,
    sample_recovery_metrics,
)


def _record(logits=(5.0, 4.0, 3.0)) -> dict:
    return {
        "sample_key": "sample",
        "variant": "rdkit",
        "base_anchors": torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]),
        "source_ids": torch.tensor([10, 20, 30]),
        "target_anchors": torch.tensor([[0.1, 0.1], [0.51, 0.5]]),
        "logits": torch.tensor(logits),
        "offsets": torch.tensor([[0.3, 0.0], [0.01, 0.0], [-0.2, 0.0]]),
        "oracle_actions": [
            {"base_source_id": 20, "target_anchor": [0.51, 0.5]}
        ],
    }


def test_sample_metrics_count_only_net_useful_additions() -> None:
    record = _record()
    useful = sample_recovery_metrics(record, [1])
    assert useful["matched_gain"] == 1
    assert useful["useless_actions"] == 0
    assert useful["exact_gain"] == 1
    assert useful["recovered_oracle"] == 1
    mixed = sample_recovery_metrics(record, [0, 1])
    assert mixed["matched_gain"] == 1
    assert mixed["useless_actions"] == 1


def test_calibration_stops_before_first_useless_high_score_action() -> None:
    record = _record(logits=(4.0, 5.0, 3.0))
    threshold, audit = calibrate_zero_useless_threshold([record])
    assert threshold == 5.0
    assert audit["metrics"]["actions"] == 1
    assert audit["metrics"]["matched_gain"] == 1
    assert audit["metrics"]["useless_actions"] == 0
    fixed = evaluate_threshold([record], threshold)
    assert fixed["by_variant"]["rdkit"]["exact_gain"] == 1


def test_calibration_uses_safe_no_action_state_when_top_action_is_false() -> None:
    record = _record(logits=(5.0, 4.0, 3.0))
    threshold, audit = calibrate_zero_useless_threshold([record])
    assert threshold > 5.0
    assert audit["metrics"]["actions"] == 0
    assert audit["metrics"]["zero_useless"]
