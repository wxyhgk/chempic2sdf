"""Tests for the bounded P4-E9 local Atom recovery head."""

from __future__ import annotations

import pytest
import torch

from chemvector.components.recovery.local_atom_recovery import (
    BASE_ACTION,
    REFINE_ACTION,
    SPLIT_ACTION,
    LocalAtomRecoveryConfig,
    LocalAtomRecoveryHead,
    decode_local_atom_recovery,
)


def model_inputs(batch: int = 1):
    return (
        torch.randn(batch, 128, 1, 32, 32),
        torch.randn(batch, 128, 1, 32, 32),
        torch.randn(batch, 128, 32),
        torch.randn(batch, 128),
        torch.arange(128).float()[None].expand(batch, -1) / 127.0,
        torch.ones(batch, 128, dtype=torch.bool),
    )


def test_head_contract_initialization_and_parameter_budget() -> None:
    config = LocalAtomRecoveryConfig()
    model = LocalAtomRecoveryHead(config).eval()
    inputs = model_inputs(batch=2)
    inputs[-1][0, -1] = False

    with torch.no_grad():
        output = model(*inputs)

    assert model.parameter_count < 100_000
    assert output["split_logits"].shape == (2, 128)
    assert output["refine_logits"].shape == (2, 128)
    assert output["split_offsets"].shape == (2, 128, 2)
    assert output["refine_offsets"].shape == (2, 128, 2)
    torch.testing.assert_close(
        output["split_logits"][inputs[-1]],
        torch.full((255,), -6.0),
    )
    torch.testing.assert_close(
        output["refine_logits"][inputs[-1]],
        torch.full((255,), -6.0),
    )
    assert output["split_logits"][0, -1] == -30.0
    assert output["refine_logits"][0, -1] == -30.0
    assert torch.count_nonzero(output["split_offsets"]) == 0
    assert torch.count_nonzero(output["refine_offsets"]) == 0
    assert torch.count_nonzero(model.split_offset.weight) == 0
    assert torch.count_nonzero(model.refine_offset.weight) == 0


def test_head_offsets_use_fixed_tanh_pixel_ranges() -> None:
    model = LocalAtomRecoveryHead().eval()
    with torch.no_grad():
        model.split_offset.bias.fill_(100.0)
        model.refine_offset.bias.fill_(-100.0)
        output = model(*model_inputs())

    torch.testing.assert_close(
        output["split_offsets"],
        torch.full_like(output["split_offsets"], 8.0 / 768.0),
    )
    torch.testing.assert_close(
        output["refine_offsets"],
        torch.full_like(output["refine_offsets"], -12.0 / 768.0),
    )


def test_decode_preserves_base_and_enforces_route_budgets_and_ids() -> None:
    config = LocalAtomRecoveryConfig()
    base_anchors = torch.rand(1, 128, 2) * 0.8 + 0.1
    base_ids = torch.arange(128, dtype=torch.long)[None]
    base_mask = torch.ones(1, 128, dtype=torch.bool)
    output = {
        "split_logits": torch.arange(128).float()[None],
        "refine_logits": torch.arange(128).float()[None] - 256.0,
        "split_offsets": torch.zeros(1, 128, 2),
        "refine_offsets": torch.zeros(1, 128, 2),
    }
    output["split_offsets"][0, :, 0] = 4.0 / 768.0
    output["refine_offsets"][0, :, 1] = 5.0 / 768.0

    decoded = decode_local_atom_recovery(
        output, base_anchors, base_ids, base_mask, config=config
    )

    assert decoded["anchors"].shape == (1, 160, 2)
    assert torch.equal(decoded["anchors"][:, :128], base_anchors)
    assert torch.equal(decoded["source_ids"][:, :128], base_ids)
    assert torch.equal(decoded["mask"][:, :128], base_mask)
    assert torch.equal(
        decoded["actions"][:, :128],
        torch.full((1, 128), BASE_ACTION, dtype=torch.int8),
    )
    assert decoded["split_candidate_counts"].tolist() == [16]
    assert decoded["refine_candidate_counts"].tolist() == [0]
    assert int(decoded["mask"].sum()) == 144

    namespace = 768 * 768
    split_parents = list(range(127, 111, -1))
    expected_split_ids = [namespace + parent for parent in split_parents]
    assert decoded["source_ids"][0, 128:144].tolist() == expected_split_ids
    assert decoded["actions"][0, 128:144].tolist() == [SPLIT_ACTION] * 16
    assert not decoded["mask"][0, 144:].any()


def test_decode_routes_compete_per_source_before_route_budgets() -> None:
    base_anchors = torch.rand(1, 128, 2)
    base_ids = torch.arange(128, dtype=torch.long)[None]
    base_mask = torch.ones(1, 128, dtype=torch.bool)
    output = {
        "split_logits": torch.full((1, 128), -20.0),
        "refine_logits": torch.full((1, 128), -20.0),
        "split_offsets": torch.zeros(1, 128, 2),
        "refine_offsets": torch.zeros(1, 128, 2),
    }
    output["split_logits"][0, :20] = 5.0
    output["refine_logits"][0, :20] = 6.0
    output["split_logits"][0, 20:40] = 7.0
    output["refine_logits"][0, 20:40] = 6.0
    decoded = decode_local_atom_recovery(
        output, base_anchors, base_ids, base_mask, action_threshold=0.0
    )
    assert decoded["split_candidate_counts"].tolist() == [16]
    assert decoded["refine_candidate_counts"].tolist() == [16]
    parents = decoded["parent_indices"][0, 128:160].tolist()
    assert len(parents) == len(set(parents))
    assert all(parent >= 20 for parent in parents[:16])
    assert all(parent < 20 for parent in parents[16:])


def test_decode_threshold_can_leave_recovery_empty() -> None:
    model = LocalAtomRecoveryHead().eval()
    inputs = model_inputs()
    base_anchors = torch.rand(1, 128, 2)
    base_ids = torch.arange(128, dtype=torch.long)[None]
    with torch.no_grad():
        output = model(*inputs)
        decoded = decode_local_atom_recovery(
            output, base_anchors, base_ids, inputs[-1]
        )

    assert int(decoded["mask"].sum()) == 128
    assert decoded["split_candidate_counts"].tolist() == [0]
    assert decoded["refine_candidate_counts"].tolist() == [0]
    assert torch.equal(decoded["anchors"][:, :128], base_anchors)


def test_invalid_inputs_and_duplicate_source_ids_are_rejected() -> None:
    model = LocalAtomRecoveryHead()
    inputs = list(model_inputs())
    inputs[0] = inputs[0][:, :, :, :-1]
    with pytest.raises(ValueError, match="image and score crops"):
        model(*inputs)

    output = {
        "split_logits": torch.zeros(1, 128),
        "refine_logits": torch.zeros(1, 128),
        "split_offsets": torch.zeros(1, 128, 2),
        "refine_offsets": torch.zeros(1, 128, 2),
    }
    base_ids = torch.arange(128, dtype=torch.long)[None]
    base_ids[0, 1] = 0
    with pytest.raises(ValueError, match="must be unique"):
        decode_local_atom_recovery(
            output,
            torch.rand(1, 128, 2),
            base_ids,
            torch.ones(1, 128, dtype=torch.bool),
        )


def test_protocol_fields_cannot_drift() -> None:
    with pytest.raises(ValueError, match="crop_size must be 32"):
        LocalAtomRecoveryConfig(crop_size=24)
    with pytest.raises(ValueError, match="split radius must be 8"):
        LocalAtomRecoveryConfig(split_radius_pixels=9.0)
    with pytest.raises(ValueError, match="action bias must be -6"):
        LocalAtomRecoveryConfig(action_bias=-4.0)
