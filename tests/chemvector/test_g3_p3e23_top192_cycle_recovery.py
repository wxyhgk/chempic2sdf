from __future__ import annotations

import pytest
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
)
from chemvector.evaluation.audits.atoms.audit_g3_p3e23_top192_cycle_recovery import (
    CANDIDATES,
    count_metrics,
    format_top192_record,
    reachability_metrics,
)


def source_record() -> dict:
    return {
        "source_ids": torch.arange(256),
        "candidate_features": torch.zeros(256, 32),
        "candidate_anchors": torch.stack(
            (torch.linspace(0, 1, 256), torch.linspace(0, 1, 256)), dim=-1
        ),
        "raw_scores": torch.linspace(1, 0, 256),
        "normalized_scores": torch.linspace(1, 0, 256),
        "atom_anchors": torch.zeros(128, 2),
        "atom_mask": torch.zeros(128, dtype=torch.bool),
    }


def test_top192_record_is_a_fixed_prefix() -> None:
    record = format_top192_record("sample", source_record())
    assert record["candidate_anchors"].shape == (CANDIDATES, 2)
    assert record["candidate_features"].shape == (CANDIDATES, 32)
    assert record["candidate_ids"].tolist() == list(range(CANDIDATES))
    assert record["scale_ranks"][0] == 0
    assert record["scale_ranks"][-1] == 1


def test_top192_record_rejects_duplicate_candidate_ids() -> None:
    source = source_record()
    source["source_ids"][1] = 0
    with pytest.raises(ValueError, match="not unique"):
        format_top192_record("duplicate", source)


def test_dynamic_candidate_error_reports_configured_width() -> None:
    model = G3AtomQuerySet(G3AtomQuerySetConfig(candidates=192))
    candidates = {
        "candidate_features": torch.zeros(1, 128, 32),
        "candidate_anchors": torch.zeros(1, 128, 2),
        "raw_scores": torch.zeros(1, 128),
        "normalized_scores": torch.zeros(1, 128),
        "scale_ranks": torch.zeros(1, 128),
    }
    with pytest.raises(ValueError, match=r"\[batch,192,2\]"):
        model(candidates)


def test_reachability_and_count_metrics() -> None:
    records = [
        {
            "atom_mask": torch.tensor([True, True]),
            "full_atom_mask": torch.tensor([True, True]),
            "p3e16_count_logits": torch.tensor([0.0, 0.0, 1.0]),
            "label": 2,
        },
        {
            "atom_mask": torch.tensor([True, False]),
            "full_atom_mask": torch.tensor([True, True]),
            "p3e16_count_logits": torch.tensor([0.0, 1.0, 0.0]),
            "label": 2,
        },
    ]
    reachability = reachability_metrics(records)
    count = count_metrics(records)
    assert reachability["matched_atoms"] == 3
    assert reachability["exact_images"] == 1
    assert count["exact_images"] == 1
    assert count["mae"] == 0.5
