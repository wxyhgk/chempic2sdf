import pytest
import torch

from chemvector.evaluation.audits.atoms.build_g3_p14e0_top128_source_context import (
    TOP_K,
    SET_TOKENS,
    source_context,
)


def test_source_context_is_relative_and_marks_source() -> None:
    anchors = torch.rand(TOP_K, 2)
    scores = torch.linspace(1.0, 0.0, TOP_K)
    source_anchor = torch.tensor([0.25, 0.75])
    source_score = torch.tensor(0.42)
    offsets, output_scores, mask = source_context(
        anchors, scores, source_anchor, source_score
    )
    assert offsets.shape == (SET_TOKENS, 2)
    assert mask.sum() == 1 and mask[-1]
    torch.testing.assert_close(offsets[-1], torch.zeros(2))
    torch.testing.assert_close(offsets[:TOP_K], anchors - source_anchor)
    torch.testing.assert_close(output_scores[:TOP_K], scores)
    torch.testing.assert_close(output_scores[-1], source_score)


def test_source_context_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="explicit source"):
        source_context(
            torch.rand(TOP_K, 2),
            torch.rand(TOP_K),
            torch.rand(3),
            torch.tensor(0.5),
        )
