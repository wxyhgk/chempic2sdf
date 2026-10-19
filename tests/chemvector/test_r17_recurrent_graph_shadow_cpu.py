import torch

from tools.train_r17_recurrent_graph_shadow_cpu import LoopedGraph


def test_looped_graph_zero_residual_preserves_frozen_presence_for_valid_edges() -> None:
    torch.manual_seed(20261008)
    model = LoopedGraph(edge_dim=16, hidden=8, iterations=3)
    node = torch.randn(2, 5, 4)
    edge = torch.randn(2, 7, 16)
    edge[..., 0] = torch.sigmoid(edge[..., 0])
    indices = torch.tensor([[ [0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [0, 4], [1, 4] ],
                            [ [0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [0, 4], [1, 4] ]])
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[0, -1] = False
    base = edge[..., 0]
    output = model(node, edge, indices, valid, base)
    expected = torch.logit(base.clamp(1e-4, 1 - 1e-4))
    assert torch.allclose(output[valid], expected[valid])
    assert torch.all(output[~valid] == -30)

