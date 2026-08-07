from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p4e7_offset_radius_ab import (
    cell_center_anchors,
    choose_decision,
)


def test_cell_center_anchors_decode_stride1_source_ids() -> None:
    anchors = cell_center_anchors(torch.tensor([0, 767, 768], dtype=torch.long))

    torch.testing.assert_close(
        anchors,
        torch.tensor(
            [
                [0.5 / 768, 0.5 / 768],
                [767.5 / 768, 0.5 / 768],
                [0.5 / 768, 1.5 / 768],
            ]
        ),
    )


def _metrics(center_matched: int, center_exact: int) -> dict:
    value = {
        mode: {
            "1024": {
                "6": {"matched": 32_342, "exact": 597},
            }
        }
        for mode in ("predicted_offset", "cell_center")
    }
    value["cell_center"]["1024"]["6"] = {
        "matched": center_matched,
        "exact": center_exact,
    }
    return value


def test_decision_requires_material_safe_cell_center_gain() -> None:
    decision, checks = choose_decision(
        _metrics(32_357, 600),
        {"cell_center_vs_offset_radius6": {"exact_break": 0}},
    )

    assert decision == "offset_head_harmful_go"
    assert all(checks.values())


def test_decision_rejects_paired_break_even_with_aggregate_gain() -> None:
    decision, checks = choose_decision(
        _metrics(32_360, 600),
        {"cell_center_vs_offset_radius6": {"exact_break": 1}},
    )

    assert decision == "offset_head_not_primary"
    assert checks["cell_center_no_paired_exact_break"] is False
