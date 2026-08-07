from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.build_g3_p4e1_pilot_union_cache import (
    FEATURE_DIM,
    source_aware_union_record,
)


def _record(offset: int) -> dict:
    ids = torch.arange(256) + offset
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:2] = True
    return {
        "source_ids": ids,
        "candidate_features": torch.stack([(ids + 1).float()] * 32, dim=1),
        "candidate_anchors": torch.stack((ids.float(), ids.float()), dim=1) / 1024,
        "raw_scores": -ids.float(),
        "normalized_scores": torch.linspace(2, -2, 256),
        "atom_anchors": torch.zeros((128, 2)),
        "atom_mask": mask,
    }


def test_source_aware_union_has_fixed_shape_and_unique_ids() -> None:
    old = _record(64)
    new = _record(0)
    result = source_aware_union_record(old, new, "sample_stress")
    assert result["candidate_features"].shape == (256, FEATURE_DIM)
    assert len(result["candidate_ids"].unique()) == 256
    assert result["label"] == 2


def test_source_aware_metadata_marks_missing_source() -> None:
    old = _record(256)
    new = _record(0)
    result = source_aware_union_record(old, new, "sample_rdkit")
    first = result["candidate_features"][0].float()
    assert torch.count_nonzero(first[:32]) == 0
    assert torch.count_nonzero(first[32:64]) == 32
    assert first[68:72].tolist() == [0.0, 1.0, 0.0, 1.0]
