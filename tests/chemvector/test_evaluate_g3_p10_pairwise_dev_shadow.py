import torch

from chemvector.evaluation.evaluators.atoms.evaluate_g3_p10_pairwise_dev_shadow import (
    THRESHOLD,
    choose_overrides,
    summarize_records,
)


def test_p10_override_requires_frozen_threshold() -> None:
    scores = torch.tensor([[1.7, 1.0], [1.6, 1.5]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    winners, execute = choose_overrides(scores, mask, THRESHOLD)
    assert winners.tolist() == [0, 0]
    assert execute.tolist() == [True, False]


def test_p10_summary_uses_whole_image_paired_metrics() -> None:
    metrics = summarize_records(
        [
            {"variant": "stress", "targets": 2, "raw_matched": 1, "prior_matched": 1, "p9_matched": 2, "raw_exact": False, "prior_exact": False, "p9_exact": True, "actions": 1},
            {"variant": "stress", "targets": 2, "raw_matched": 2, "prior_matched": 2, "p9_matched": 1, "raw_exact": True, "prior_exact": True, "p9_exact": False, "actions": 1},
            {"variant": "rdkit", "targets": 1, "raw_matched": 1, "prior_matched": 1, "p9_matched": 1, "raw_exact": True, "prior_exact": True, "p9_exact": True, "actions": 0},
        ]
    )
    assert metrics["matched_net_gain"] == 0
    assert metrics["exact_net_gain"] == 0
    assert metrics["paired"] == {
        "exact_rescues": 1,
        "matched_gain_images": 1,
        "exact_breaks": 1,
        "matched_loss_images": 1,
    }
