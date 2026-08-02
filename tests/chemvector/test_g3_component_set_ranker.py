import unittest

import torch

from chemvector.components.atoms.g3_component_set_ranker import (
    G3ComponentSetRanker,
    build_target_free_components,
    component_boundary_band_loss,
    select_ranked_components,
    target_free_component_labels,
)


def candidates():
    flat = torch.arange(128)
    anchors = torch.stack(
        (((flat % 16).float() + 0.5) / 16.0, ((flat // 16).float() + 0.5) / 8.0),
        dim=1,
    )[None]
    anchors[0, 0] = torch.tensor([0.100, 0.100])
    anchors[0, 1] = torch.tensor([0.102, 0.100])
    values = {
        "candidate_ids": torch.arange(128)[None],
        "candidate_features": torch.randn(1, 128, 384),
        "candidate_anchors": anchors,
        "base_logits": torch.full((1, 128), -2.0),
    }
    values["base_logits"][0, :2] = torch.tensor([0.0, 2.0])
    return values


class G3ComponentSetRankerTests(unittest.TestCase):
    def test_three_pixel_labels_merge_near_candidates(self) -> None:
        labels = target_free_component_labels(candidates()["candidate_anchors"][0])

        self.assertEqual(int(labels[0]), int(labels[1]))
        self.assertNotEqual(int(labels[0]), int(labels[2]))

    def test_component_builder_uses_base_softmax_representative(self) -> None:
        values = candidates()
        components = build_target_free_components(values)
        weights = torch.tensor([0.0, 2.0]).softmax(dim=0)
        expected = (
            values["candidate_anchors"][0, :2] * weights[:, None]
        ).sum(dim=0)

        self.assertEqual(int(components["component_counts"][0]), 127)
        torch.testing.assert_close(components["candidate_anchors"][0, 0], expected)
        self.assertEqual(float(components["component_sizes"][0, 0]), 2.0)
        self.assertTrue(bool(components["component_mask"][0, :127].all()))
        self.assertFalse(bool(components["component_mask"][0, 127]))

    def test_ranker_zero_initialization_preserves_component_base_scores(self) -> None:
        components = build_target_free_components(candidates())
        ranker = G3ComponentSetRanker().eval()

        ranked = ranker(components)

        torch.testing.assert_close(
            ranked["ranking_logits"], components["base_logits"], rtol=0, atol=0
        )
        self.assertEqual(int(torch.count_nonzero(ranked["ranking_residual"])), 0)

    def test_selection_never_exposes_padding(self) -> None:
        components = build_target_free_components(candidates())
        ranked = G3ComponentSetRanker().eval()(components)

        selected = select_ranked_components(ranked, torch.tensor([2]))

        self.assertEqual(int(selected["atom_mask"].sum()), 2)
        self.assertTrue(
            bool((selected["atom_component_ids"][0, :2] < 127).all())
        )

    def test_boundary_loss_ignores_padding(self) -> None:
        components = build_target_free_components(candidates())
        ranked = G3ComponentSetRanker().eval()(components)
        positive = torch.zeros(1, 128, dtype=torch.bool)
        positive[0, 0] = True

        result = component_boundary_band_loss(
            ranked, {"positive_mask": positive}
        )
        result["loss"].backward()

        self.assertEqual(float(result["positive_count"]), 1.0)
        self.assertEqual(float(result["negative_count"]), 126.0)


if __name__ == "__main__":
    unittest.main()
