import pytest
import torch

from chemvector.evaluation.audits.atoms.build_g3_p14e0_top128_source_context import (
    TOP_K,
    source_context,
)


def test_source_context_is_relative_and_marks_source() -> None:
    anchors = torch.rand(TOP_K, 2)
    scores = torch.linspace(1.0, 0.0, TOP_K)
    source_ids = torch.arange(TOP_K, dtype=torch.long) + 100
    offsets, output_scores, mask, rank = source_context(
        anchors, scores, source_ids, 117
    )
    assert rank == 17
    assert mask.sum() == 1 and mask[17]
    torch.testing.assert_close(offsets[17], torch.zeros(2))
    torch.testing.assert_close(output_scores, scores)


def test_source_context_rejects_missing_source() -> None:
    with pytest.raises(ValueError, match="not unique"):
        source_context(
            torch.rand(TOP_K, 2),
            torch.rand(TOP_K),
            torch.arange(TOP_K, dtype=torch.long),
            TOP_K + 1,
        )
