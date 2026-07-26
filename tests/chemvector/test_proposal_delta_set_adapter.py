import torch

from chemvector.proposal_delta_set_adapter import (
    ProposalDeltaSetAdapter,
    ProposalDeltaSetConfig,
    gather_original_bond_evidence,
)


def test_gather_original_evidence_uses_dense_slot_cell_index() -> None:
    logits = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    endpoints = torch.arange(48, dtype=torch.float32).reshape(1, 3, 2, 2, 2, 2)
    proposals = {"bond_indices": torch.tensor([[0, 5, 11]])}

    result = gather_original_bond_evidence(
        proposals, {"heatmap_logits": logits, "endpoints": endpoints}
    )

    expected_scores = logits.sigmoid().flatten(1)[0, [0, 5, 11]]
    torch.testing.assert_close(result["scores"][0], expected_scores)
    expected = endpoints.permute(0, 1, 4, 5, 2, 3).flatten(1, 3)[0, [0, 5, 11]]
    torch.testing.assert_close(result["endpoints"][0], expected)


def test_zero_adapter_preserves_base_outputs_exactly() -> None:
    batch, bonds, atoms = 2, 3, 4
    base = {
        "atom_presence_logits": torch.randn(batch, atoms),
        "bond_presence_logits": torch.randn(batch, bonds),
        "incidence_logits": torch.randn(batch, bonds, 2, atoms),
        "incidence_probabilities": torch.randn(batch, bonds, 2, atoms),
    }
    proposals = {
        "atom_mask": torch.ones(batch, atoms, dtype=torch.bool),
        "bond_mask": torch.ones(batch, bonds, dtype=torch.bool),
        "atom_geometry": torch.randn(batch, atoms, 8),
        "atom_scores": torch.rand(batch, atoms),
        "bond_endpoints": torch.rand(batch, bonds, 2, 2),
        "bond_scores": torch.rand(batch, bonds),
    }
    deltas = {
        "score": torch.randn(batch, bonds),
        "endpoints": torch.randn(batch, bonds, 2, 2),
        "original_endpoints": torch.rand(batch, bonds, 2, 2),
    }
    adapter = ProposalDeltaSetAdapter(ProposalDeltaSetConfig())

    result = adapter(base, proposals, deltas)

    torch.testing.assert_close(
        result["bond_presence_logits"], base["bond_presence_logits"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["incidence_logits"], base["incidence_logits"], rtol=0, atol=0
    )
    assert result["incidence_probabilities"].dtype == base["atom_presence_logits"].dtype
