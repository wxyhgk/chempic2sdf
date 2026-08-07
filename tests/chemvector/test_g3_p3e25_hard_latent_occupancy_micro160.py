from __future__ import annotations

import torch
import pytest

from chemvector.training.atoms.train_g3_p3e25_hard_latent_occupancy_micro160 import (
    COUNT_CLASSES,
    bounded_count_log_probs,
    count_bucket,
    hard_presence_targets,
    stable_topk_indices,
)


def test_count_bucket_boundaries() -> None:
    assert count_bucket(1) == "1-20"
    assert count_bucket(20) == "1-20"
    assert count_bucket(21) == "21-40"
    assert count_bucket(100) == "81-100"


def test_bounded_count_log_probs_are_129_classes() -> None:
    source = torch.randn(2, 193).log_softmax(dim=-1)
    result = bounded_count_log_probs({"count_log_probs": source})
    assert result.shape == (2, COUNT_CLASSES)
    assert torch.allclose(result.logsumexp(dim=-1), torch.zeros(2), atol=1e-6)


def test_hard_targets_choose_unique_compatible_candidates() -> None:
    record = {
        "candidate_anchors": torch.tensor(
            [[0.1, 0.1], [0.102, 0.1], [0.5, 0.5], [0.502, 0.5]]
        ),
        "atom_anchors": torch.tensor([[0.1, 0.1], [0.5, 0.5]]),
        "full_atom_mask": torch.tensor([True, True]),
    }
    logits = torch.tensor([[0.0, 2.0, 3.0, 1.0]])
    targets = hard_presence_targets(logits, [record])
    assert targets.tolist() == [[False, True, True, False]]
    assert int(targets.sum()) == 2


def test_hard_targets_reject_infeasible_cover() -> None:
    record = {
        "candidate_anchors": torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
        "atom_anchors": torch.tensor([[0.9, 0.9]]),
        "full_atom_mask": torch.tensor([True]),
    }
    with pytest.raises(RuntimeError, match="infeasible"):
        hard_presence_targets(torch.zeros(1, 2), [record])


def test_stable_topk_uses_index_order_for_ties() -> None:
    scores = torch.tensor([0.5, 0.7, 0.7, 0.1])
    assert stable_topk_indices(scores, 3).tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="outside"):
        stable_topk_indices(scores, 129)
