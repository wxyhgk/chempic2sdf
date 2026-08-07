from __future__ import annotations

from chemvector.training.atoms.train_g3_p4e3_global_peak_rank_pilot20k import (
    GLOBAL_PEAK_BOUNDARY_WINDOW,
    GLOBAL_PEAK_CANDIDATE_BUDGET,
    GLOBAL_PEAK_RANK_WEIGHT,
    capability_gate,
)


def _metrics(recall: float, exact: int, known_recall: float = 0.98) -> dict:
    return {
        "top128_reachability": {"atom_recall": recall, "exact_images": exact},
        "known_full_count": {"recall_6px": known_recall},
    }


def test_protocol_uses_a_material_global_boundary_objective() -> None:
    assert GLOBAL_PEAK_CANDIDATE_BUDGET == 128
    assert GLOBAL_PEAK_BOUNDARY_WINDOW == 16
    assert GLOBAL_PEAK_RANK_WEIGHT == 10.0


def test_capability_gate_requires_recall_and_exact_gain() -> None:
    decision, checks = capability_gate(
        _metrics(0.990, 500),
        _metrics(0.991, 505),
    )
    assert decision == "global_peak_rank_pilot20k_go"
    assert all(checks.values())


def test_capability_gate_rejects_known_count_regression() -> None:
    decision, checks = capability_gate(
        _metrics(0.990, 500, 0.98),
        _metrics(0.992, 510, 0.978),
    )
    assert decision == "global_peak_rank_pilot20k_no_go"
    assert checks["known_count_recall_no_regression"] is False
