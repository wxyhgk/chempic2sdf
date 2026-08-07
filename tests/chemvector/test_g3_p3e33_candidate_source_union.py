from __future__ import annotations

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p3e33_candidate_source_union import (
    choose_winner,
    merge_candidate_records,
)


def _record(offset: int) -> dict:
    ids = torch.arange(256) + offset
    return {
        "source_ids": ids,
        "candidate_features": torch.stack((ids.float(), ids.float()), dim=1),
        "candidate_anchors": torch.stack((ids.float(), ids.float()), dim=1) / 1024,
        "raw_scores": -ids.float(),
    }


def test_primary_prefix_is_preserved_and_source_ids_are_unique() -> None:
    old = _record(64)
    new = _record(0)
    merged = merge_candidate_records(old, new, "train25k128_then_train5k")
    assert merged["source_ids"][:128].tolist() == list(range(128))
    assert len(merged["source_ids"].unique()) == 256
    assert int(merged["source_codes"][:128].sum()) == 128


def test_alternating_rank_interleaves_novel_sources() -> None:
    old = _record(256)
    new = _record(0)
    merged = merge_candidate_records(old, new, "alternating_rank")
    assert merged["source_ids"][:4].tolist() == [0, 256, 1, 257]


def test_choose_winner_prioritizes_top192_then_top128() -> None:
    def result(top192, top128):
        return {
            "reachability": {"192": {"exact_images": top192}, "128": {"exact_images": top128}},
            "comparison": {"recovered_old_tail_images": 1, "regressed_old_reachable_images": 0},
            "known_count_raw": {"exact_images": 1},
        }

    name, _ = choose_winner({
        "train25k128_then_train5k": result(160, 140),
        "train25k160_then_train5k": result(159, 150),
    })
    assert name == "train25k128_then_train5k"
