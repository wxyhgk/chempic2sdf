import torch

from chemvector.training.graph.relations import (
    train_g3_p19e1_direct_atom_pair_capacity64 as p19e1,
)


def test_pair_targets_cover_only_true_unordered_pairs():
    pairs = torch.combinations(torch.arange(4), r=2)
    presence, chemical = p19e1.build_pair_targets(
        pairs,
        torch.tensor([[0, 2], [2, 3]]),
        torch.tensor([1, 4]),
    )
    assert pairs.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert presence.tolist() == [False, True, False, False, False, True]
    assert chemical[presence].tolist() == [1, 4]


def test_pair_targets_reject_duplicate_target_edges():
    pairs = torch.combinations(torch.arange(3), r=2)
    try:
        p19e1.build_pair_targets(
            pairs,
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([1, 1]),
        )
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate target edge was accepted")


def test_sample_metrics_require_the_entire_pair_set():
    metrics = p19e1._sample_metrics(
        torch.tensor([0.9, 0.1, 0.7]),
        torch.tensor([1, 0, 2]),
        torch.tensor([True, False, False]),
        torch.tensor([1, 0, 0]),
        threshold=0.5,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert not metrics["skeleton_exact"]


def test_capacity_summary_keeps_difficulty_and_operation_buckets():
    rows = [
        {
            "difficulty": "easy",
            "stress_operation": "long_bond",
            "target_bonds": 2,
            "selected_bonds": 2,
            "true_positive": 2,
            "false_positive": 0,
            "false_negative": 0,
            "skeleton_exact": True,
            "typed_exact": True,
        },
        {
            "difficulty": "hard",
            "stress_operation": "crossed_bonds",
            "target_bonds": 2,
            "selected_bonds": 1,
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 1,
            "skeleton_exact": False,
            "typed_exact": False,
        },
    ]
    summary = p19e1._capacity_summary(rows)
    assert summary["overall"]["skeleton_exact"] == 1
    assert summary["by_difficulty"]["easy"]["skeleton_exact"] == 1
    assert summary["hard_by_operation"]["crossed_bonds"]["images"] == 1
    assert summary["hard_by_operation"]["near_nonbonded"]["images"] == 0


def test_metric_key_prioritizes_easy_zero_break():
    base = {
        "overall": {
            "true_positive": 100,
            "false_positive": 2,
            "false_negative": 3,
        },
        "by_difficulty": {
            "easy": {"skeleton_exact": 32},
            "hard": {"skeleton_exact": 20},
        },
    }
    more_hard_but_easy_break = {
        "overall": {
            "true_positive": 200,
            "false_positive": 0,
            "false_negative": 0,
        },
        "by_difficulty": {
            "easy": {"skeleton_exact": 31},
            "hard": {"skeleton_exact": 32},
        },
    }
    assert p19e1._metric_key(base) > p19e1._metric_key(more_hard_but_easy_break)


def test_balanced_presence_loss_backpropagates_both_classes():
    logits = torch.tensor([0.2, -0.1, 0.5, -0.7], requires_grad=True)
    targets = torch.tensor([True, True, False, False])
    losses = p19e1.balanced_presence_loss(logits, targets)
    losses["loss"].backward()
    assert float(losses["positive_loss"].detach()) > 0
    assert float(losses["negative_loss"].detach()) > 0
    assert float(losses["listwise_margin_loss"].detach()) > 0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == len(logits)
