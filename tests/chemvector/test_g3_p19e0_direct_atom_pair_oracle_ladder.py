import torch

from chemvector.evaluation.audits.graph.audit_g3_p19e0_direct_atom_pair_oracle_ladder import (
    _sample_metrics,
)


def test_sample_metrics_uses_atom_pair_identity_without_endpoints():
    metrics = _sample_metrics(
        probabilities=torch.tensor([0.99, 0.01, 0.98]),
        chemical=torch.tensor([0, 0, 1]),
        pairs=torch.tensor([[0, 1], [0, 2], [1, 2]]),
        target_ids=torch.tensor([2, 0, 1, -1]),
        target_atom_count=3,
        target_refs=torch.tensor([[0, 2], [0, 1]]),
        target_types=torch.tensor([0, 1]),
        threshold=0.85,
    )
    assert metrics["atom_exact"]
    assert metrics["skeleton_exact"]
    assert metrics["typed_exact"]


def test_sample_metrics_penalizes_missing_atom_even_if_representable_edges_match():
    metrics = _sample_metrics(
        probabilities=torch.tensor([0.99]),
        chemical=torch.tensor([0]),
        pairs=torch.tensor([[0, 1]]),
        target_ids=torch.tensor([0, 1, -1]),
        target_atom_count=3,
        target_refs=torch.tensor([[0, 1]]),
        target_types=torch.tensor([0]),
        threshold=0.85,
    )
    assert not metrics["atom_exact"]
    assert not metrics["skeleton_exact"]


def test_sample_metrics_penalizes_false_pair():
    metrics = _sample_metrics(
        probabilities=torch.tensor([0.99, 0.99, 0.98]),
        chemical=torch.tensor([0, 0, 1]),
        pairs=torch.tensor([[0, 1], [0, 2], [1, 2]]),
        target_ids=torch.tensor([2, 0, 1]),
        target_atom_count=3,
        target_refs=torch.tensor([[0, 2], [0, 1]]),
        target_types=torch.tensor([0, 1]),
        threshold=0.85,
    )
    assert metrics["false_positive"] == 1
    assert not metrics["skeleton_exact"]
