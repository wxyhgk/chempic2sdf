from chemvector.evaluation.audits.vision.audit_v1p1r_semantic_ink_threshold_confirm import (
    THRESHOLDS,
    select_iou_threshold,
)


def test_threshold_grid_is_fixed_and_ordered():
    assert THRESHOLDS[0] == 0.01
    assert THRESHOLDS[-1] == 0.50
    assert len(THRESHOLDS) == 26
    assert tuple(sorted(set(THRESHOLDS))) == THRESHOLDS


def test_threshold_selection_uses_iou_then_precision_then_conservatism():
    rows = [
        {"threshold": 0.03, "iou": 0.91, "precision": 0.95},
        {"threshold": 0.04, "iou": 0.91, "precision": 0.97},
        {"threshold": 0.05, "iou": 0.91, "precision": 0.97},
    ]
    assert select_iou_threshold(rows)["threshold"] == 0.05
