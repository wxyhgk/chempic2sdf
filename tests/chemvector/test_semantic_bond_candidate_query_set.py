import torch

from chemvector.components.bonds.objects.semantic_bond_candidate_query_set import (
    SemanticBondCandidateQuerySet,
    SemanticBondCandidateQuerySetConfig,
    _group_pointer_log_probability,
    semantic_bond_candidate_query_set_loss,
)
from chemvector.components.bonds.objects.semantic_endpoint_candidate_graph import (
    NODE_FEATURE_DIM,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    PAIR_FEATURE_DIM,
)


def _model() -> SemanticBondCandidateQuerySet:
    torch.manual_seed(11)
    return SemanticBondCandidateQuerySet(
        SemanticBondCandidateQuerySetConfig(
            d_model=32,
            graph_message_passing_steps=1,
            decoder_layers=1,
            attention_heads=4,
            feedforward_dim=64,
            query_count=8,
        )
    )


def test_bond_query_set_outputs_fixed_queries_over_variable_edges() -> None:
    model = _model()
    output = model(
        torch.rand((5, NODE_FEATURE_DIM)),
        torch.rand((6, PAIR_FEATURE_DIM)),
        torch.tensor(((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4))),
    )
    assert output["existence_logits"].shape == (8,)
    assert output["pointer_logits"].shape == (8, 6)


def test_bond_query_set_hungarian_loss_matches_each_bond_once() -> None:
    model = _model().train()
    output = model(
        torch.rand((5, NODE_FEATURE_DIM)),
        torch.rand((6, PAIR_FEATURE_DIM)),
        torch.tensor(((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4))),
    )
    losses = semantic_bond_candidate_query_set_loss(
        output, torch.tensor((0, 0, -1, 1, 1, -1)), bond_count=2
    )
    assert torch.isfinite(losses["loss"])
    assert len(losses["matched_queries"]) == 2
    assert sorted(losses["matched_bonds"].tolist()) == [0, 1]
    losses["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_bond_query_set_rejects_unreachable_target_group() -> None:
    model = _model()
    output = model(
        torch.rand((3, NODE_FEATURE_DIM)),
        torch.rand((2, PAIR_FEATURE_DIM)),
        torch.tensor(((0, 1), (1, 2))),
    )
    try:
        semantic_bond_candidate_query_set_loss(
            output, torch.tensor((0, -1)), bond_count=2
        )
    except ValueError as error:
        assert "unreachable" in str(error)
    else:
        raise AssertionError("Query-set loss should reject an unreachable target Bond")


def test_bond_query_group_score_matches_argmax_instead_of_probability_mass() -> None:
    logits = torch.log(torch.tensor(((0.30, 0.30, 0.40),)))
    scores = _group_pointer_log_probability(
        logits, torch.tensor((0, 0, 1)), bond_count=2
    )
    assert scores.shape == (1, 2)
    assert float(scores[0, 1]) > float(scores[0, 0])
