from __future__ import annotations

from chemvector.training.atoms.train_g3_p4e2_center_field_pilot20k import (
    capability_gate,
)


def _metrics(recall: float, exact: int, known_recall: float = 0.98) -> dict:
    return {
        "top128_reachability": {"atom_recall": recall, "exact_images": exact},
        "known_full_count": {"recall_6px": known_recall},
    }


def test_capability_gate_requires_recall_and_exact_gain() -> None:
    decision, checks = capability_gate(
        _metrics(0.990, 500),
        _metrics(0.991, 505),
    )
    assert decision == "center_field_pilot20k_go"
    assert all(checks.values())


def test_capability_gate_rejects_exact_only_gain() -> None:
    decision, checks = capability_gate(
        _metrics(0.990, 500),
        _metrics(0.9905, 510),
    )
    assert decision == "center_field_pilot20k_no_go"
    assert checks["dev_top128_recall_gain_min001"] is False
