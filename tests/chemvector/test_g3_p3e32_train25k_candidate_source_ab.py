from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p3e32_train25k_candidate_source_ab import (
    compare_sources,
    reachability_metrics,
)


def _candidate(points) -> dict:
    anchors = torch.full((256, 2), 0.95)
    anchors[: len(points)] = torch.tensor(points)
    return {"candidate_anchors": anchors}


def _target(key: str, points) -> dict:
    anchors = torch.zeros((128, 2))
    mask = torch.zeros(128, dtype=torch.bool)
    anchors[: len(points)] = torch.tensor(points)
    mask[: len(points)] = True
    return {
        "sample_key": key,
        "atom_anchors": anchors,
        "full_atom_mask": mask,
    }


def test_reachability_metrics_reports_each_budget() -> None:
    targets = [_target("x_stress", ((0.1, 0.1), (0.2, 0.2)))]
    points = [(0.1, 0.1), *[(0.8, 0.8)] * 127, (0.2, 0.2)]
    metrics = reachability_metrics({"x_stress": _candidate(points)}, targets)
    assert metrics["128"]["exact_images"] == 0
    assert metrics["192"]["exact_images"] == 1
    assert metrics["192"]["by_variant"]["stress"]["exact"] == 1


def test_compare_sources_counts_tail_recovery_and_regression() -> None:
    targets = [
        _target("tail_stress", ((0.1, 0.1), (0.2, 0.2))),
        _target("old_rdkit", ((0.3, 0.3),)),
    ]
    old = {
        "tail_stress": _candidate([(0.1, 0.1), *[(0.8, 0.8)] * 127, (0.2, 0.2)]),
        "old_rdkit": _candidate([(0.3, 0.3)]),
    }
    new = {
        "tail_stress": _candidate(((0.1, 0.1), (0.2, 0.2))),
        "old_rdkit": _candidate(((0.8, 0.8),)),
    }
    result = compare_sources(old, new, targets)
    assert result["old_tail_images"] == 1
    assert result["recovered_old_tail_images"] == 1
    assert result["regressed_old_reachable_images"] == 1
    assert result["old_tail_targets"] == 1
    assert result["recovered_old_tail_targets_top128"] == 1
