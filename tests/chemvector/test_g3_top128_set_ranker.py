from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from chemvector.components.atoms.g3_top128_set_ranker import (
    G3_TOP128_SET_RANKER_PARAMETERS,
    G3Top128SetRanker,
    build_hybrid_top128_targets,
    extract_frozen_top128,
    select_ranked_top128,
    top128_set_ranking_loss,
)


def config():
    return SimpleNamespace(d_model=384, maximum_atoms=128, atom_slots_per_cell=3)


def fake_dense(*, height: int = 8, width: int = 8) -> dict[str, torch.Tensor]:
    batch = 1
    logits = torch.zeros(batch, 3, height, width)
    features = torch.randn(batch, 3, 384, height, width)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    anchors = torch.stack(((x + 0.5) / width, (y + 0.5) / height))
    anchors = anchors[None, None].expand(batch, 3, 2, height, width).clone()
    return {"presence_logits": logits, "slot_features": features, "anchors": anchors}


def permute_candidates(candidates: dict[str, torch.Tensor], permutation: torch.Tensor):
    output = {}
    for name, value in candidates.items():
        output[name] = (
            value[:, permutation]
            if value.ndim >= 2 and value.shape[:2] == (1, 128)
            else value
        )
    return output


class G3Top128SetRankerTests(unittest.TestCase):
    def test_stable_top128_uses_flat_id_to_break_boundary_ties(self) -> None:
        dense = fake_dense()
        dense["presence_logits"].flatten()[150] = 1.0

        candidates = extract_frozen_top128(dense)

        expected = torch.tensor(list(range(127)) + [150])
        torch.testing.assert_close(candidates["candidate_ids"][0], expected)
        self.assertTrue(
            bool(
                (
                    candidates["candidate_ids"][0, 1:]
                    > candidates["candidate_ids"][0, :-1]
                ).all()
            )
        )

    def test_parameter_count_zero_parity_and_membership_is_fixed(self) -> None:
        candidates = extract_frozen_top128(fake_dense())
        model = G3Top128SetRanker(config())

        ranked = model(candidates)

        self.assertEqual(model.parameter_count(), G3_TOP128_SET_RANKER_PARAMETERS)
        self.assertEqual(model.parameter_count(), 60_545)
        torch.testing.assert_close(
            ranked["ranking_logits"], ranked["base_logits"], rtol=0, atol=0
        )
        torch.testing.assert_close(ranked["candidate_ids"], candidates["candidate_ids"])
        with torch.no_grad():
            model.residual_head.weight.normal_()
            model.residual_head.bias.fill_(1.0)
        reranked = model(candidates)
        torch.testing.assert_close(reranked["candidate_ids"], candidates["candidate_ids"])
        self.assertFalse(torch.equal(reranked["ranking_logits"], ranked["ranking_logits"]))

    def test_ranker_is_permutation_equivariant(self) -> None:
        torch.manual_seed(7)
        candidates = extract_frozen_top128(fake_dense())
        model = G3Top128SetRanker(config()).eval()
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.1)
        permutation = torch.randperm(128)
        inverse = permutation.argsort()

        original = model(candidates)
        shuffled = model(permute_candidates(candidates, permutation))

        torch.testing.assert_close(
            shuffled["ranking_logits"][:, inverse],
            original["ranking_logits"],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_token_branches_apply_gelu_before_layer_norm(self) -> None:
        candidates = extract_frozen_top128(fake_dense())
        model = G3Top128SetRanker(config()).eval()
        captured = {}

        def capture_input(_module, args):
            captured["hidden"] = args[0].detach().clone()

        handle = model.set_encoder.register_forward_pre_hook(capture_input)
        try:
            model(candidates)
        finally:
            handle.remove()
        features = candidates["candidate_features"].float()
        geometry_score = torch.cat(
            (
                candidates["candidate_anchors"].float(),
                candidates["base_logits"].float()[..., None],
            ),
            dim=-1,
        )
        expected = model.token_norm(
            F.gelu(
                model.feature_projection(model.feature_norm(features))
                + model.geometry_projection(geometry_score)
            )
        )

        torch.testing.assert_close(captured["hidden"], expected)

    def test_hybrid_targets_prefer_canonical_then_rescue_frozen_geometry(self) -> None:
        dense = fake_dense()
        candidates = extract_frozen_top128(dense)
        anchors = candidates["candidate_anchors"].clone()
        target_anchors = torch.tensor(
            [[[0.010, 0.010], [0.012, 0.010], [0.014, 0.010]]]
        )
        anchors[0, 0] = target_anchors[0, 0]
        anchors[0, 64] = target_anchors[0, 1]
        anchors[0, 10] = target_anchors[0, 2]
        candidates["candidate_anchors"] = anchors
        targets = {
            "atom_anchors": target_anchors,
            "atom_mask": torch.tensor([[True, True, True]]),
        }

        hybrid = build_hybrid_top128_targets(candidates, targets)

        positive_ids = set(
            candidates["candidate_ids"][0, hybrid["positive_mask"][0]].tolist()
        )
        canonical_ids = set(
            candidates["candidate_ids"][0, hybrid["canonical_positive_mask"][0]].tolist()
        )
        self.assertEqual(canonical_ids, {0, 64})
        self.assertEqual(positive_ids, {0, 10, 64})
        self.assertEqual(int(hybrid["representable_counts"][0]), 3)
        self.assertEqual(int(hybrid["canonical_counts"][0]), 2)

    def test_loss_uses_top_four_m_and_does_not_backpropagate_base(self) -> None:
        torch.manual_seed(11)
        features = torch.randn(1, 128, 384, requires_grad=True)
        anchors = torch.rand(1, 128, 2, requires_grad=True)
        base = torch.zeros(1, 128, requires_grad=True)
        candidates = {
            "candidate_ids": torch.arange(128)[None],
            "candidate_features": features,
            "candidate_anchors": anchors,
            "base_logits": base,
            "grid_height": torch.tensor([8]),
            "grid_width": torch.tensor([8]),
        }
        model = G3Top128SetRanker(config())
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.1)
        ranked = model(candidates)
        positive = torch.zeros(1, 128, dtype=torch.bool)
        positive[0, :2] = True
        hybrid = {"positive_mask": positive}

        result = top128_set_ranking_loss(ranked, hybrid)
        expected_negatives = ranked["ranking_logits"][0, 2:].topk(8).values
        expected = F.softplus(
            expected_negatives[:, None] - ranked["ranking_logits"][0, :2][None]
        ).mean()
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(float(result["positive_count"]), 2.0)
        self.assertEqual(float(result["hard_negative_count"]), 8.0)
        result["loss"].backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(anchors.grad)
        self.assertIsNone(base.grad)
        self.assertIsNotNone(model.residual_head.weight.grad)

    def test_select_applies_count_only_after_ranking_with_stable_ties(self) -> None:
        candidates = extract_frozen_top128(fake_dense())
        ranked = G3Top128SetRanker(config())(candidates)

        selected = select_ranked_top128(ranked, torch.tensor([3]))

        self.assertEqual(int(selected["atom_mask"].sum()), 3)
        self.assertEqual(selected["atom_flat_ids"][0, :3].tolist(), [0, 1, 2])
        self.assertEqual(selected["atom_counts"].tolist(), [3])

    def test_hybrid_rejects_identical_target_coordinates(self) -> None:
        candidates = extract_frozen_top128(fake_dense())
        targets = {
            "atom_anchors": torch.tensor([[[0.01, 0.01], [0.01, 0.01]]]),
            "atom_mask": torch.tensor([[True, True]]),
        }

        with self.assertRaisesRegex(ValueError, "identical Atom coordinates"):
            build_hybrid_top128_targets(candidates, targets)


if __name__ == "__main__":
    unittest.main()
