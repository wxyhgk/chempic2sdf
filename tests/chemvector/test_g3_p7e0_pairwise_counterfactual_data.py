import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    PAIR_EQUIVALENT,
    PAIR_HARMFUL,
    PAIR_RESCUE,
)
from chemvector.evaluation.audits.atoms.build_g3_p7e0_pairwise_counterfactual_data import (
    SOURCE_HARD,
    build_pair_rows,
    classify_pair,
)


def test_p7e0_pair_classes_are_fixed() -> None:
    assert classify_pair(7.0, 2.0) == PAIR_RESCUE
    assert classify_pair(2.0, 7.0) == PAIR_HARMFUL
    assert classify_pair(2.0, 3.0) == PAIR_EQUIVALENT
    assert classify_pair(7.0, 8.0) is None


def test_p7e0_builds_indices_without_copying_rasters() -> None:
    cache = {
        "candidate_offsets": torch.tensor(
            [[[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]]], dtype=torch.float16
        ),
        "candidate_peak_logits": torch.tensor([[3.0, 2.0, 1.0]]),
        "candidate_mask": torch.ones(1, 3, dtype=torch.bool),
        "offset_targets": torch.tensor([[0.01, 0.0]]),
        "scaffold": ["s0"],
        "variant": ["original"],
        "target_role": ["IMPLICIT"],
        "degree_bucket": ["2"],
        "image_crops": torch.rand(1, 1, 96, 96),
    }
    rows, counts = build_pair_rows(cache, SOURCE_HARD)
    assert counts["rescue"] == 1
    assert counts["both_bad_excluded"] == 1
    assert rows["label"].tolist() == [PAIR_RESCUE]
    assert "image_crops" not in rows
