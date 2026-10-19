import torch


def test_zero_penalty_projection_is_frozen_argmax_and_includes_aromatic():
    from tools.probe_r17_t3f_graph_type_correction_cpu import project

    class Zero:
        def __call__(self, x):
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    item = {
        "base_logits": torch.tensor([8.0, 8.0]),
        "features": torch.zeros((2, 1)),
        "pairs": torch.tensor([[0, 1], [1, 2]]),
        "chemical_logits": torch.tensor([
            [-3.0, -2.0, -1.0, 5.0],  # AROMATIC must remain reachable
            [4.0, 1.0, 0.0, -2.0],
        ]),
    }
    ids, edges, assignment = project(item, Zero(), 0.0)
    assert ids == [0, 1]
    assert edges == [(0, 1), (1, 2)]
    assert assignment == {0: 3, 1: 0}
