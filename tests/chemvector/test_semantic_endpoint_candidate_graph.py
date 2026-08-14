import torch

from chemvector.components.bonds.objects.semantic_endpoint_candidate_graph import (
    NODE_FEATURE_DIM,
    SemanticEndpointCandidateGraph,
    SemanticEndpointCandidateGraphConfig,
    candidate_graph_node_features,
    semantic_endpoint_candidate_graph_loss,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    PAIR_FEATURE_DIM,
)


def _model() -> SemanticEndpointCandidateGraph:
    torch.manual_seed(7)
    return SemanticEndpointCandidateGraph(
        SemanticEndpointCandidateGraphConfig(hidden_dim=32, message_passing_steps=2)
    ).eval()


def test_candidate_graph_node_features_include_shared_incidence() -> None:
    points = torch.tensor(((0.1, 0.2), (0.5, 0.5), (0.9, 0.2)))
    scores = torch.tensor((0.8, 0.9, 0.7))
    pairs = torch.tensor(((0, 1), (1, 2)))
    features = candidate_graph_node_features(
        points, scores, pairs, incidence_count_scale=2.0
    )
    assert features.shape == (3, NODE_FEATURE_DIM)
    assert torch.allclose(features[:, :3], torch.cat((points, scores[:, None]), -1))
    assert features[:, 3].tolist() == [0.5, 1.0, 0.5]


def test_candidate_graph_is_equivariant_to_node_and_edge_permutations() -> None:
    model = _model()
    node_features = torch.rand((5, NODE_FEATURE_DIM))
    edge_features = torch.rand((4, PAIR_FEATURE_DIM))
    pairs = torch.tensor(((0, 1), (1, 2), (1, 3), (3, 4)))
    original = model(node_features, edge_features, pairs)

    node_order = torch.tensor((3, 0, 4, 1, 2))
    inverse = torch.empty_like(node_order)
    inverse[node_order] = torch.arange(len(node_order))
    edge_order = torch.tensor((2, 0, 3, 1))
    permuted = model(
        node_features[node_order],
        edge_features[edge_order],
        inverse[pairs[edge_order]],
    )
    assert torch.allclose(
        original["edge_logits"][edge_order], permuted["edge_logits"], atol=1e-6
    )
    assert torch.allclose(
        original["node_keep_logits"][node_order],
        permuted["node_keep_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        original["graph_bond_count"], permuted["graph_bond_count"], atol=1e-6
    )


def test_candidate_graph_set_loss_selects_one_equivalent_edge_per_bond() -> None:
    model = _model().train()
    node_features = torch.rand((5, NODE_FEATURE_DIM))
    edge_features = torch.rand((6, PAIR_FEATURE_DIM))
    pairs = torch.tensor(((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)))
    output = model(node_features, edge_features, pairs)
    losses = semantic_endpoint_candidate_graph_loss(
        output,
        torch.tensor((0, 0, -1, 1, 1, -1)),
        torch.tensor((1, 0, 2, 1, 0)),
        bond_count=2,
    )
    assert torch.isfinite(losses["loss"])
    assert int(losses["latent_positive_edges"]) == 2
    assert int(losses["reachable_bonds"]) == 2
    losses["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_candidate_graph_rejects_out_of_range_pair_index() -> None:
    model = _model()
    try:
        model(
            torch.rand((2, NODE_FEATURE_DIM)),
            torch.rand((1, PAIR_FEATURE_DIM)),
            torch.tensor(((0, 2),)),
        )
    except ValueError as error:
        assert "outside the node set" in str(error)
    else:
        raise AssertionError("Candidate graph should reject an invalid pair index")


def test_candidate_graph_supports_cuda_bfloat16_autocast() -> None:
    if not torch.cuda.is_available():
        return
    model = _model().cuda()
    node_features = torch.rand((5, NODE_FEATURE_DIM), device="cuda")
    edge_features = torch.rand((4, PAIR_FEATURE_DIM), device="cuda")
    pairs = torch.tensor(((0, 1), (1, 2), (1, 3), (3, 4)), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(node_features, edge_features, pairs)
    assert output["edge_logits"].shape == (4,)
    assert torch.isfinite(output["graph_bond_count"])
