from __future__ import annotations

import torch

from chemvector.training.atoms.train_g3_p3e26_soft_coverage_occupancy_micro160 import (
    compatibility_components,
    soft_coverage_loss,
)


def test_compatibility_components_merge_shared_candidate() -> None:
    compatibility = torch.tensor([
        [True, True, False, False],
        [False, True, True, False],
        [False, False, False, True],
    ])
    components = compatibility_components(compatibility)
    assert [len(value["gt_indices"]) for value in components] == [2, 1]
    assert components[0]["candidate_mask"].tolist() == [True, True, True, False]
    assert components[1]["candidate_mask"].tolist() == [False, False, False, True]


def test_soft_loss_is_finite_and_identity_invariant() -> None:
    record = {
        "candidate_anchors": torch.tensor(
            [[0.1, 0.1], [0.102, 0.1], [0.5, 0.5], [0.9, 0.9]]
        ),
        "atom_anchors": torch.tensor([[0.1, 0.1], [0.5, 0.5]]),
        "full_atom_mask": torch.tensor([True, True]),
    }
    output = {
        "presence_logits": torch.tensor([[1.0, 0.5, 2.0, -1.0]], requires_grad=True),
        "count_log_probs": torch.randn(1, 193).log_softmax(dim=-1),
    }
    loss = soft_coverage_loss(output, [record])
    assert all(torch.isfinite(value) for value in loss.values())
    loss["loss"].backward()
    assert output["presence_logits"].grad is not None

    permutation = torch.tensor([1, 0, 2, 3])
    permuted = {
        **record,
        "candidate_anchors": record["candidate_anchors"][permutation],
    }
    permuted_output = {
        "presence_logits": output["presence_logits"].detach()[:, permutation],
        "count_log_probs": output["count_log_probs"],
    }
    repeated = soft_coverage_loss(permuted_output, [permuted])
    assert torch.allclose(loss["loss"].detach(), repeated["loss"], atol=1e-6)
