"""Tests for target-isolated P4 local recovery data and losses."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from chemvector.training.atoms.g3_p4_local_recovery_data import (
    anchor_centered_crops,
    build_local_recovery_observations,
    build_local_recovery_targets,
    deterministic_recovery_sample_mask,
    local_recovery_loss,
    normalized_candidate_ranks,
    normalized_candidate_scores,
)


def fake_observable_inputs(size: int = 32):
    pixels = torch.ones(1, 3, size, size)
    pixels[:, :, 15:17, 15:17] = -1.0
    center_logits = torch.arange(size * size).float().reshape(1, 1, size, size)
    features = torch.stack(
        [torch.full((size, size), float(channel)) for channel in range(32)]
    )[None]
    source_ids = torch.arange(128, dtype=torch.long)
    anchors = torch.stack(
        (
            (source_ids.remainder(size).float() + 0.5) / size,
            (torch.div(source_ids, size, rounding_mode="floor").float() + 0.5)
            / size,
        ),
        dim=-1,
    )
    scores = torch.linspace(0.99, 0.01, 128)
    peaks = [{"anchors": anchors, "scores": scores, "source_ids": source_ids}]
    return pixels, {"center_logits": center_logits, "stride1_features": features}, peaks


def test_anchor_crops_are_symmetric_grid_samples() -> None:
    values = torch.arange(64 * 64).float().reshape(1, 1, 64, 64)
    anchors = torch.tensor([[[0.5, 0.5]]])
    crops = anchor_centered_crops(values, anchors)

    assert crops.shape == (1, 1, 1, 32, 32)
    torch.testing.assert_close(crops[0, 0, 0], values[0, 0, 16:48, 16:48])


def test_observations_use_only_image_field_and_peak_sources() -> None:
    pixels, field, peaks = fake_observable_inputs()
    observations = build_local_recovery_observations(pixels, field, peaks)

    assert observations["image_crops"].shape == (1, 128, 1, 32, 32)
    assert observations["score_crops"].shape == (1, 128, 1, 32, 32)
    assert observations["candidate_features"].shape == (1, 128, 32)
    torch.testing.assert_close(
        observations["candidate_features"][0, 17], torch.arange(32).float()
    )
    assert observations["candidate_scores"].mean().abs() < 1e-6
    assert observations["candidate_scores"].std(unbiased=False) == pytest.approx(1.0)
    assert observations["candidate_ranks"][0, 0] == 0.0
    assert observations["candidate_ranks"][0, -1] == 1.0
    assert observations["image_crops"].min() >= 0.0
    assert observations["image_crops"].max() <= 1.0

    corrupted = copy.deepcopy(peaks)
    corrupted[0]["target_anchors"] = torch.randn(100, 2)
    repeated = build_local_recovery_observations(pixels, field, corrupted)
    for name in observations:
        assert torch.equal(observations[name], repeated[name])


def test_score_normalization_and_rank_tie_break_are_per_image() -> None:
    scores = torch.tensor([[0.9, 0.5, 0.5], [0.2, 0.4, 0.8]])
    source_ids = torch.tensor([[9, 7, 8], [2, 1, 0]])
    normalized = normalized_candidate_scores(scores)
    ranks = normalized_candidate_ranks(scores, source_ids)

    torch.testing.assert_close(normalized.mean(1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        normalized.std(1, unbiased=False), torch.ones(2), atol=1e-6, rtol=0
    )
    assert ranks[0].tolist() == [0.0, 0.5, 1.0]
    assert ranks[1].tolist() == [1.0, 0.5, 0.0]


def test_target_builder_keeps_gt_out_of_observations_and_normalizes_offsets() -> None:
    _, _, peaks = fake_observable_inputs()
    anchors = peaks[0]["anchors"][None]
    source_ids = peaks[0]["source_ids"][None]
    mask = torch.ones(1, 128, dtype=torch.bool)
    split_target = anchors[0, 7] + torch.tensor([4.0, 0.0]) / 768.0
    refine_target = anchors[0, 11] + torch.tensor([0.0, -10.0]) / 768.0
    targets = build_local_recovery_targets(
        ["sample"],
        anchors,
        source_ids,
        mask,
        [
            {
                "sample_key": "sample",
                "route": "ADD_SPLIT",
                "base_source_id": 7,
                "target_anchor": split_target.tolist(),
            },
            {
                "sample_key": "sample",
                "route": "ADD_REFINE",
                "base_source_id": 11,
                "target_anchor": refine_target.tolist(),
            },
        ],
    )

    assert targets["split_targets"][0, 7]
    assert targets["refine_targets"][0, 11]
    torch.testing.assert_close(
        targets["split_offsets"][0, 7] * 768.0,
        torch.tensor([4.0, 0.0]),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        targets["refine_offsets"][0, 11] * 768.0,
        torch.tensor([0.0, -10.0]),
        atol=1e-4,
        rtol=0,
    )


def test_deterministic_sampler_is_exactly_one_to_three() -> None:
    targets = torch.zeros(2, 128, dtype=torch.bool)
    targets[0, 3] = True
    targets[1, 4] = True
    mask = torch.ones_like(targets)
    source_ids = torch.arange(128, dtype=torch.long)[None].expand(2, -1).clone()
    keys = ["left", "right"]

    first = deterministic_recovery_sample_mask(
        targets, mask, source_ids, keys, round_index=2
    )
    second = deterministic_recovery_sample_mask(
        targets, mask, source_ids, keys, round_index=2
    )
    changed = deterministic_recovery_sample_mask(
        targets, mask, source_ids, keys, round_index=3
    )

    assert torch.equal(first, second)
    assert int(first.sum()) == 8
    assert int((first & targets).sum()) == 2
    assert int((first & ~targets).sum()) == 6
    assert not torch.equal(first, changed)
    with pytest.raises(ValueError, match="fixed 1:3"):
        deterministic_recovery_sample_mask(
            targets, mask, source_ids, keys, negative_ratio=2
        )


def test_route_loss_uses_bce_and_positive_offsets() -> None:
    logits = torch.zeros(1, 128, requires_grad=True)
    offsets = torch.zeros(1, 128, 2, requires_grad=True)
    output = {
        "split_logits": logits,
        "split_offsets": offsets,
        "refine_logits": torch.zeros(1, 128),
        "refine_offsets": torch.zeros(1, 128, 2),
    }
    target_mask = torch.zeros(1, 128, dtype=torch.bool)
    target_mask[0, 0] = True
    targets = {
        "split_targets": target_mask,
        "split_offsets": torch.zeros(1, 128, 2),
        "refine_targets": torch.zeros(1, 128, dtype=torch.bool),
        "refine_offsets": torch.zeros(1, 128, 2),
        "candidate_mask": torch.ones(1, 128, dtype=torch.bool),
    }
    targets["split_offsets"][0, 0, 0] = 4.0 / 768.0
    sampled = torch.zeros(1, 128, dtype=torch.bool)
    sampled[0, :4] = True

    losses = local_recovery_loss(
        output, targets, sampled, route="ADD_SPLIT"
    )
    losses["loss"].backward()

    assert float(losses["action_loss"].detach()) == pytest.approx(math.log(2.0))
    assert float(losses["offset_loss"].detach()) == pytest.approx(1.75)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert offsets.grad is not None and torch.isfinite(offsets.grad).all()


def test_target_radius_and_sampler_positive_coverage_are_enforced() -> None:
    _, _, peaks = fake_observable_inputs()
    anchors = peaks[0]["anchors"][None]
    source_ids = peaks[0]["source_ids"][None]
    mask = torch.ones(1, 128, dtype=torch.bool)
    with pytest.raises(ValueError, match="exceeds"):
        build_local_recovery_targets(
            ["sample"],
            anchors,
            source_ids,
            mask,
            [
                {
                    "sample_key": "sample",
                    "route": "ADD_SPLIT",
                    "base_source_id": 0,
                    "target_anchor": (
                        anchors[0, 0] + torch.tensor([9.0, 0.0]) / 768.0
                    ).tolist(),
                }
            ],
        )

    output = {
        "split_logits": torch.zeros(1, 128),
        "split_offsets": torch.zeros(1, 128, 2),
    }
    targets = {
        "split_targets": torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=128
        ).bool(),
        "split_offsets": torch.zeros(1, 128, 2),
        "candidate_mask": mask,
    }
    with pytest.raises(ValueError, match="omitted"):
        local_recovery_loss(
            output,
            targets,
            torch.zeros(1, 128, dtype=torch.bool),
            route="ADD_SPLIT",
        )
