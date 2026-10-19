import pytest
import torch

from chemvector.components.graph.joint.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
)
from chemvector.training.curriculum.chemical_head_stage import (
    ChemicalHeadOnlyStage,
    chemical_head_loss,
    chemical_head_loss_with_aromatic_safety,
)
from chemvector.bond_object import BOND_TYPE_TO_ID


def make_pair():
    return DirectAtomPairBond(DirectAtomPairBondConfig(
        feature_channels=8,
        atom_feature_dim=4,
        longitudinal_samples=4,
        transverse_samples=2,
        hidden_dim=8,
        pair_chunk_size=4,
    ))


def make_inputs():
    torch.manual_seed(17)
    return (
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 2, 4),
        torch.tensor([[[0.2, 0.3], [0.7, 0.6]]]),
        torch.tensor([[0.8, 0.9]]),
        torch.tensor([[0, 1]]),
        torch.tensor([0]),
    )


def test_chemical_update_changes_only_output_head_and_preserves_presence():
    torch.manual_seed(23)
    pair = make_pair()
    stage = ChemicalHeadOnlyStage(pair)
    inputs = make_inputs()
    before = {name: value.detach().clone() for name, value in pair.state_dict().items()}
    optimizer = torch.optim.SGD(stage.trainable_parameters(), lr=0.05)

    initial = stage(*inputs)
    loss = chemical_head_loss(
        initial,
        torch.tensor([True]),
        torch.tensor([1]),
        torch.tensor([3]),
    )
    assert loss is not None
    loss.backward()
    optimizer.step()
    final = stage(*inputs)

    assert torch.equal(initial["presence_logits"], final["presence_logits"])
    assert not torch.equal(initial["chemical_logits"], final["chemical_logits"])
    for name, value in pair.state_dict().items():
        if not name.startswith("chemical."):
            assert torch.equal(value, before[name]), name
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in pair.chemical.parameters()
    }


def test_stage_rejects_train_mode_and_unknown_pair_rows():
    stage = ChemicalHeadOnlyStage(make_pair())
    with pytest.raises(ValueError, match="eval mode"):
        stage.train()

    output = {"presence_logits": torch.zeros(2), "chemical_logits": torch.zeros(2, 4)}
    with pytest.raises(ValueError, match="Unknown pair"):
        chemical_head_loss(output, torch.tensor([True, False]), torch.tensor([1, 0]), torch.tensor([0, 0]))


def test_no_positive_pair_has_no_chemical_update():
    output = {"presence_logits": torch.zeros(2), "chemical_logits": torch.zeros(2, 4)}
    loss = chemical_head_loss(
        output,
        torch.tensor([True, True]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
    )
    assert loss is None


def test_optional_aromatic_safety_branch_is_off_by_default_and_adds_ring_gradient():
    aromatic = BOND_TYPE_TO_ID["AROMATIC"]
    pairs = torch.tensor([[i, (i + 1) % 6] for i in range(6)], dtype=torch.long)
    output = {
        "presence_logits": torch.ones(6),
        "chemical_logits": torch.zeros(6, 4, requires_grad=True),
    }
    known = torch.ones(6, dtype=torch.bool)
    presence = torch.ones(6, dtype=torch.long)
    targets = torch.full((6,), aromatic, dtype=torch.long)
    off, off_info = chemical_head_loss_with_aromatic_safety(
        output, known, presence, targets, pairs, torch.zeros(6, dtype=torch.long)
    )
    on, on_info = chemical_head_loss_with_aromatic_safety(
        output, known, presence, targets, pairs, torch.zeros(6, dtype=torch.long),
        safety_weight=0.5,
    )
    assert off is not None and on is not None and float(on.detach()) > float(off.detach())
    assert off_info["safety_weight"] == 0.0
    assert on_info["rings"] == 1 and on_info["aromatic_safety_loss"] > 0.0


def test_trust_region_requires_reference_and_adds_distillation_penalty():
    output = {
        "presence_logits": torch.ones(2),
        "chemical_logits": torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], requires_grad=True),
    }
    known = torch.ones(2, dtype=torch.bool)
    presence = torch.ones(2, dtype=torch.long)
    targets = torch.tensor([0, 1], dtype=torch.long)
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    samples = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="required"):
        chemical_head_loss_with_aromatic_safety(
            output, known, presence, targets, pairs, samples, trust_region_weight=1.0,
        )
    reference = torch.zeros(2, 4)
    loss, info = chemical_head_loss_with_aromatic_safety(
        output, known, presence, targets, pairs, samples,
        reference_chemical_logits=reference, trust_region_weight=0.5,
    )
    assert loss is not None and info["trust_region_loss"] > 0.0
    loss.backward()
    assert output["chemical_logits"].grad is not None
