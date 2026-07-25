import torch

from chemvector.atom_pair_bond_group import (
    GROUP_FEATURE_DIM,
    AtomPairBondGroupConfig,
    AtomPairBondGroupHead,
    atom_pair_group_loss,
    atom_pair_group_type_loss,
    collapse_atom_pair_group_outputs,
)


def test_atom_pair_group_head_and_loss() -> None:
    head = AtomPairBondGroupHead(AtomPairBondGroupConfig())
    features = torch.randn(4, GROUP_FEATURE_DIM)
    targets = torch.tensor([True, False, True, False])
    samples = torch.tensor([0, 0, 1, 1])

    output = head(features)
    logits = output["presence_logits"]
    loss = atom_pair_group_loss(logits, targets, samples)

    assert logits.shape == (4,)
    assert output["display_logits"].shape == (4, 4)
    assert output["chemical_logits"].shape == (4, 4)
    assert output["direction_logits"].shape == (4, 5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_atom_pair_group_type_loss() -> None:
    head = AtomPairBondGroupHead(AtomPairBondGroupConfig())
    output = head(torch.randn(3, GROUP_FEATURE_DIM))
    groups = {
        "type_mask": torch.tensor([True, True, False]),
        "display_targets": torch.tensor([0, 1, 0]),
        "chemical_targets": torch.tensor([0, 3, 0]),
        "direction_targets": torch.tensor([0, 1, 0]),
    }

    losses = atom_pair_group_type_loss(output, groups)

    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_collapse_atom_pair_group_outputs_keeps_one_representative() -> None:
    proposals = {
        "bond_mask": torch.tensor([[True, True, True]]),
        "bond_types": torch.zeros(1, 3, 8),
    }
    set_output = {"bond_presence_logits": torch.tensor([[0.2, 0.8, -0.1]])}
    group_output = {
        "presence_logits": torch.tensor([1.5, -0.5]),
        "display_logits": torch.tensor([[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "chemical_logits": torch.tensor([[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "direction_logits": torch.zeros(2, 5),
    }
    groups = {"members": [[[0, 1], [2]]]}

    collapsed, output, diagnostics = collapse_atom_pair_group_outputs(
        proposals, set_output, group_output, groups
    )

    assert collapsed["bond_mask"].tolist() == [[False, True, True]]
    assert output["bond_presence_logits"][0, 1].item() == 1.5
    assert diagnostics[0]["groups"] == 2
