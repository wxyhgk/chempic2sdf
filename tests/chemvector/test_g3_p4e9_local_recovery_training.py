import torch

from chemvector.components.recovery.local_atom_recovery import LocalAtomRecoveryHead
from chemvector.training.atoms.train_g3_p4e9_local_recovery_pilot20k import (
    BATCH_SIZE,
    NEGATIVE_BATCH_SIZE,
    POSITIVE_BATCH_SIZE,
    balanced_batch_schedule,
    capacity_subset,
    route_parameters,
)


def _cache(positives: int = 160, negatives: int = 480) -> dict:
    rows = positives + negatives
    return {
        "action_targets": torch.tensor([True] * positives + [False] * negatives),
        "offset_targets": torch.zeros(rows, 2),
        "source_ids": torch.arange(rows),
        "sample_keys": [f"sample-{index // 4}" for index in range(rows)],
        "candidate_scores": torch.zeros(rows),
    }


def test_capacity_subset_and_batches_follow_frozen_ratio() -> None:
    selected = capacity_subset(_cache())
    assert int(selected["action_targets"].sum()) == 128
    assert len(selected["action_targets"]) == 512
    first = balanced_batch_schedule(selected, steps=4, seed=8)
    second = balanced_batch_schedule(selected, steps=4, seed=8)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    for indices in first:
        targets = selected["action_targets"][indices]
        assert len(indices) == BATCH_SIZE
        assert int(targets.sum()) == POSITIVE_BATCH_SIZE
        assert int((~targets).sum()) == NEGATIVE_BATCH_SIZE


def test_route_training_freezes_only_the_other_output_heads() -> None:
    model = LocalAtomRecoveryHead()
    parameters = route_parameters(model, "split")
    parameter_ids = {id(value) for value in parameters}
    assert all(not value.requires_grad for value in model.refine_action.parameters())
    assert all(not value.requires_grad for value in model.refine_offset.parameters())
    assert all(id(value) not in parameter_ids for value in model.refine_action.parameters())
    assert all(value.requires_grad for value in model.split_action.parameters())
    assert all(value.requires_grad for value in model.split_offset.parameters())
