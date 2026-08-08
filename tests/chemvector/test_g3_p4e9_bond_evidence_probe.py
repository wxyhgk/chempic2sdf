from __future__ import annotations

import pytest
import torch

from chemvector.evaluation.audits.atoms.audit_g3_p4e9_bond_evidence_probe import (
    BASELINE_DIM,
    INCIDENT_DIM,
    build_route_dataset,
    endpoint_target_coverage,
    evidence_gate,
    scaffold_fold,
    three_fold_logistic_oof,
)


def _record(index: int, *, scaffold: str) -> dict:
    sources = torch.arange(index * 10, index * 10 + 8, dtype=torch.long)
    baseline = torch.zeros((8, BASELINE_DIM))
    incident = torch.zeros((8, INCIDENT_DIM))
    incident[0, 0] = 4.0
    return {
        "sample_key": f"sample-{index}",
        "molecule_id": f"molecule-{index}",
        "scaffold": scaffold,
        "source_ids": sources,
        "baseline_features": baseline,
        "incident_features": incident,
        "actions": {
            int(sources[0]): "ADD_REFINE",
            int(sources[1]): "ADD_SPLIT",
        },
        "unsupported_source_ids": [int(sources[2])],
    }


def test_scaffold_fold_keeps_variants_together_and_has_fallback() -> None:
    first = scaffold_fold("c1ccccc1", "mol-a")
    assert first == scaffold_fold("c1ccccc1", "mol-b")
    assert scaffold_fold("", "mol-a") == scaffold_fold("", "mol-a")
    assert 0 <= first < 3
    with pytest.raises(ValueError, match="at least two"):
        scaffold_fold("x", "m", folds=1)


def test_route_dataset_uses_only_none_negatives_and_exact_ratio() -> None:
    records = []
    # Find one deterministic scaffold in each fold so OOF is structurally complete.
    for fold in range(3):
        suffix = 0
        while True:
            scaffold = f"scaffold-{fold}-{suffix}"
            if scaffold_fold(scaffold, f"molecule-{fold}") == fold:
                break
            suffix += 1
        records.append(_record(fold, scaffold=scaffold))
    dataset = build_route_dataset(records, route="ADD_REFINE")
    assert dataset["positives"] == 3
    assert dataset["none_negatives"] == 9
    assert dataset["excluded_counts"] == {"ADD_SPLIT": 3, "UNSUPPORTED": 3}
    assert all(status in {"ADD_REFINE", "NONE"} for _, _, status in dataset["identities"])
    assert dataset["baseline_features"].shape == (12, BASELINE_DIM)
    assert dataset["augmented_features"].shape == (12, BASELINE_DIM + INCIDENT_DIM)
    assert all(value["none_negatives"] == 3 for value in dataset["fold_counts"].values())


def test_three_fold_logistic_detects_incremental_incident_signal() -> None:
    folds = torch.arange(3).repeat_interleave(8)
    labels = torch.tensor(([True, True] + [False] * 6) * 3)
    baseline = torch.zeros((24, 2))
    incident = labels.float().mul(2.0).sub(1.0)[:, None]
    baseline_result = three_fold_logistic_oof(baseline, labels, folds)
    augmented_result = three_fold_logistic_oof(
        torch.cat((baseline, incident), dim=1), labels, folds
    )
    assert baseline_result["auc"] == pytest.approx(0.5)
    assert augmented_result["auc"] > 0.99
    assert set(augmented_result["per_fold"]) == {"0", "1", "2"}


def test_endpoint_coverage_uses_only_valid_endpoints_and_inclusive_radius() -> None:
    targets = torch.tensor(((0.10, 0.10), (0.50, 0.50), (0.90, 0.90)))
    endpoints = torch.tensor(
        (
            ((0.10, 0.10), (0.20, 0.20)),
            ((0.90, 0.90), (0.80, 0.80)),
            ((0.50 + 6.0 / 768.0, 0.50), (0.60, 0.60)),
        )
    )
    result = endpoint_target_coverage(
        targets, endpoints, torch.tensor((True, False, True))
    )
    assert result["covered"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["covered_mask"] == [True, True, False]


def test_evidence_gate_is_conjunctive_and_inclusive() -> None:
    assert evidence_gate(0.02, 0.90)
    assert not evidence_gate(0.0199, 1.0)
    assert not evidence_gate(0.20, 0.8999)
