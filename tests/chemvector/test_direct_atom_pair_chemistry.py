import torch

from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings


def test_promotes_a_complete_six_member_ring_as_one_object() -> None:
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5], [0, 3]])
    presence = torch.tensor([0.99] * 6 + [0.1])
    logits = torch.tensor([[2.0, 1.0, 0.0, 2.5]] * 6 + [[4.0, 0.0, 0.0, 0.0]])

    chemical, diagnostics = promote_complete_aromatic_rings(
        pairs,
        presence,
        logits,
        presence_threshold=0.85,
        mean_threshold=0.5,
        minimum_threshold=0.4,
    )

    assert chemical[:6].tolist() == [3] * 6
    assert chemical[6].item() == 0
    assert diagnostics["accepted_cycles"] == 1
    assert diagnostics["promoted_bonds"] == 6


def test_rejects_ring_when_one_edge_lacks_aromatic_support() -> None:
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]])
    presence = torch.full((6,), 0.99)
    logits = torch.tensor([[2.0, 1.0, 0.0, 2.5]] * 5 + [[5.0, 0.0, 0.0, -2.0]])

    chemical, diagnostics = promote_complete_aromatic_rings(
        pairs,
        presence,
        logits,
        presence_threshold=0.85,
        mean_threshold=0.5,
        minimum_threshold=0.2,
    )

    assert chemical.tolist() != [3] * 6
    assert diagnostics["accepted_cycles"] == 0


def test_enumerates_all_fused_chordless_rings_not_only_a_cycle_basis() -> None:
    # Three length-three paths between the same junctions form three chordless
    # six-member rings, although an arbitrary cycle basis contains only two.
    pairs = torch.tensor(
        [
            [0, 2], [2, 3], [3, 1],
            [0, 4], [4, 5], [5, 1],
            [0, 6], [6, 7], [7, 1],
        ]
    )
    presence = torch.full((len(pairs),), 0.99)
    logits = torch.tensor([[2.0, 1.0, 0.0, 2.5]] * len(pairs))

    _, diagnostics = promote_complete_aromatic_rings(
        pairs,
        presence,
        logits,
        presence_threshold=0.85,
        mean_threshold=0.5,
        minimum_threshold=0.4,
    )

    assert diagnostics["candidate_cycles"] == 3
    assert diagnostics["accepted_cycles"] == 3
