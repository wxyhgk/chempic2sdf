from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.build_g3_p4e4_peak_rank_union_audit import (
    FEATURE_DIM,
    choose_winner,
    merge_candidate_records,
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


def _metrics(recall: float, exact: int) -> dict:
    return {
        split: {
            str(budget): {"recall": recall, "exact": exact}
            for budget in (128, 192, 256)
        }
        for split in ("pilot_train", "dev")
    }


def test_merge_prioritizes_fixed_new_prefix_and_deduplicates_ids() -> None:
    old = _record(64)
    new = _record(0)

    merged = merge_candidate_records(old, new, 128)

    assert merged["source_ids"][:128].tolist() == list(range(128))
    assert len(merged["source_ids"].unique()) == 256
    assert int(merged["source_codes"][:128].sum()) == 128


def test_source_aware_union_has_fixed_feature_shape() -> None:
    result = source_aware_union_record(_record(256), _record(0), "sample", 160)

    assert result["candidate_features"].shape == (256, FEATURE_DIM)
    assert result["candidate_features"][0, 68:72].float().tolist() == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]


def test_winner_prefers_smallest_passing_budget_then_dev_exact() -> None:
    first = _metrics(0.999, 10)
    second = _metrics(0.999, 20)
    first["pilot_train"]["128"]["recall"] = 0.998
    first["dev"]["128"]["recall"] = 0.998
    results = {"128": first, "160": second, "192": _metrics(0.998, 100)}

    assert choose_winner(results) == ("160", 128)


def test_winner_returns_none_when_no_budget_reaches_gate() -> None:
    results = {name: _metrics(0.998, 100) for name in ("128", "160", "192")}

    assert choose_winner(results) is None
