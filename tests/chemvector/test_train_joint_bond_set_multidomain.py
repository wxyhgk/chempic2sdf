from unittest.mock import patch

import pytest

from chemvector.train_joint_bond_set_multidomain import (
    include_forced_exact_failures,
    mixed_batches,
)


DOMAINS = {"style_train": ["style"], "chemvision_replay": ["cv"], "font_v2_replay": ["font"]}


def test_mixed_batch_counts_must_sum_to_batch_size() -> None:
    with patch(
        "chemvector.train_joint_bond_set_multidomain.TRAIN_DOMAINS", DOMAINS
    ):
        with pytest.raises(ValueError, match="sum to batch size"):
            next(
                mixed_batches(
                    domain_sample_counts={name: 1 for name in DOMAINS},
                    domain_batch_counts={name: 1 for name in DOMAINS},
                    batch_size=4,
                    config=None,
                    seed=1,
                    shuffle_buffer=1,
                )
            )


def test_mixed_batches_honor_exact_domain_totals() -> None:
    def stream(shards, **_kwargs):
        domain = shards[0]
        for index in range(20):
            yield (domain, index)

    with (
        patch("chemvector.train_joint_bond_set_multidomain.TRAIN_DOMAINS", DOMAINS),
        patch("chemvector.train_joint_bond_set_multidomain.valid_stream", stream),
    ):
        batches = list(
            mixed_batches(
                domain_sample_counts={
                    "style_train": 10,
                    "chemvision_replay": 2,
                    "font_v2_replay": 2,
                },
                domain_batch_counts={
                    "style_train": 4,
                    "chemvision_replay": 1,
                    "font_v2_replay": 1,
                },
                batch_size=6,
                config=None,
                seed=1,
                shuffle_buffer=1,
            )
        )

    flattened = [sample for batch, _seen in batches for sample in batch]
    assert len(flattened) == 14
    assert sum(sample[0] == "style" for sample in flattened) == 10
    assert batches[-1][1] == {
        "style_train": 10,
        "chemvision_replay": 2,
        "font_v2_replay": 2,
    }


def test_forced_failures_are_included_in_exact_denominators() -> None:
    metrics = {
        "samples": 8,
        "atom_candidate_exact": 6,
        "atom_deployed_exact": 5,
        "bond_candidate_exact": 4,
        "bond_deployed_exact": 3,
        "incidence_exact": 7,
        "full_object_exact": 2,
    }

    result = include_forced_exact_failures(metrics, 2)

    assert result["samples"] == 10
    assert result["evaluable_samples"] == 8
    assert result["forced_failure_samples"] == 2
    assert result["bond_deployed_exact_rate"] == 0.3
    assert result["full_object_exact_rate"] == 0.2
