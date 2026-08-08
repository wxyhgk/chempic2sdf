import torch

from chemvector.evaluation.audits.atoms.audit_g3_p5e0_action_decomposition import (
    calibrate_image_zero_fp,
    decompose_records,
    evaluate_image_threshold,
)


def _record(key: str, logits: list[float], action_index: int | None) -> dict:
    sources = torch.arange(len(logits))
    anchors = torch.zeros(len(logits), 2)
    offsets = torch.zeros(len(logits), 2)
    actions = []
    if action_index is not None:
        offsets[action_index, 0] = 4.0 / 768.0
        actions.append(
            {
                "base_source_id": int(sources[action_index]),
                "target_anchor": [4.0 / 768.0, 0.0],
            }
        )
    return {
        "sample_key": key,
        "logits": torch.tensor(logits),
        "offsets": offsets,
        "base_anchors": anchors,
        "source_ids": sources,
        "oracle_actions": actions,
    }


def test_decomposition_separates_trigger_pointer_and_offset() -> None:
    records = [
        _record("negative", [1.0, 0.0, -1.0], None),
        _record("positive", [3.0, 2.0, 1.0], 1),
    ]
    summary = decompose_records(records)
    assert summary["action_images"] == 1
    assert summary["oracle_actions"] == 1
    assert summary["pointer_rank"]["top1"] == 0
    assert summary["pointer_rank"]["top2"] == 1
    assert summary["forced_correct_source_offset"]["within_6px"] == 1
    assert summary["pointer_and_offset"]["top2_and_offset6px"] == 1


def test_image_zero_false_positive_threshold_is_replayed() -> None:
    calibration = decompose_records(
        [
            _record("negative", [1.0, 0.0], None),
            _record("positive", [2.0, 0.0], 0),
        ]
    )
    threshold = calibrate_image_zero_fp(calibration)
    assert threshold > 1.0
    result = evaluate_image_threshold(calibration, threshold)
    assert result["zero_false_positive"]
    assert result["true_positive_images"] == 1
