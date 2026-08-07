from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p4e6_missing_peak_sources import (
    classify_missing_peak,
    maximum_cardinality_assignments,
)


def test_assignments_preserve_maximum_cardinality_under_shared_candidates() -> None:
    expected = torch.tensor([[0.0, 0.0], [5.0 / 768.0, 0.0]])
    predicted = torch.tensor([[1.0 / 768.0, 0.0], [6.0 / 768.0, 0.0]])

    assignments = maximum_cardinality_assignments(predicted, expected)

    assert assignments == [0, 1]


def test_assignments_return_none_for_unreachable_gt() -> None:
    expected = torch.tensor([[0.0, 0.0], [100.0 / 768.0, 0.0]])
    predicted = torch.tensor([[1.0 / 768.0, 0.0]])

    assignments = maximum_cardinality_assignments(predicted, expected)

    assert assignments == [0, None]


def test_failure_classification_has_fixed_precedence() -> None:
    assert classify_missing_peak(
        top1024_within_radius=True,
        all_peaks_within_radius=True,
        gt_cell_is_regional_maximum=False,
    ) == "matching_conflict"
    assert classify_missing_peak(
        top1024_within_radius=False,
        all_peaks_within_radius=True,
        gt_cell_is_regional_maximum=False,
    ) == "below_top1024"
    assert classify_missing_peak(
        top1024_within_radius=False,
        all_peaks_within_radius=False,
        gt_cell_is_regional_maximum=False,
    ) == "gt_cell_not_regional_maximum"
    assert classify_missing_peak(
        top1024_within_radius=False,
        all_peaks_within_radius=False,
        gt_cell_is_regional_maximum=True,
    ) == "regional_peak_anchor_outside_radius"
