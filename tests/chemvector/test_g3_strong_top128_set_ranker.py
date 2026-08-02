import inspect
from types import SimpleNamespace
import unittest

import torch

from chemvector.components.atoms.g3_strong_top128_set_ranker import (
    G3_STRONG_SET_RANKER_PARAMETERS,
    G3StrongTop128SetRanker,
    align_candidate_patches,
    cap_zero_candidate_patches,
    deterministic_sham_permutations,
    extract_predicted_anchor_stroke_patches,
    sham_candidate_patches,
)


def config():
    return SimpleNamespace(d_model=384, maximum_atoms=128, atom_slots_per_cell=3)


def candidates(batch: int = 2):
    return {
        "candidate_ids": torch.arange(128)[None].expand(batch, -1).clone(),
        "candidate_features": torch.randn(batch, 128, 384),
        "candidate_anchors": torch.rand(batch, 128, 2),
        "base_logits": torch.randn(batch, 128),
    }


def permute_candidates(values: dict, permutation: torch.Tensor):
    return {
        name: (
            value.gather(
                1,
                permutation.reshape(*permutation.shape, *([1] * (value.ndim - 2))).expand(
                    *permutation.shape, *value.shape[2:]
                ),
            )
            if value.ndim >= 2 and value.shape[:2] == permutation.shape
            else value
        )
        for name, value in values.items()
    }


class G3StrongTop128SetRankerTests(unittest.TestCase):
    def test_parameter_total_and_zero_parity(self) -> None:
        model = G3StrongTop128SetRanker(config()).eval()
        values = candidates(batch=1)
        patches = torch.randn(1, 128, 96, 9, 9)

        output = model(values, patches)

        self.assertEqual(model.num_parameters(), G3_STRONG_SET_RANKER_PARAMETERS)
        self.assertEqual(model.num_parameters(), 2_427_713)
        torch.testing.assert_close(
            output["ranking_logits"], output["base_logits"], rtol=0, atol=0
        )
        self.assertEqual(int(torch.count_nonzero(output["ranking_residual"])), 0)

    def test_predicted_anchor_patch_shape_and_frozen_inputs(self) -> None:
        values = candidates(batch=2)
        stroke = torch.randn(2, 96, 16, 20, requires_grad=True)
        values["candidate_anchors"].requires_grad_()

        patches = extract_predicted_anchor_stroke_patches(stroke, values)

        self.assertEqual(patches.shape, (2, 128, 96, 9, 9))
        self.assertFalse(patches.requires_grad)
        self.assertIsNone(stroke.grad)
        self.assertIsNone(values["candidate_anchors"].grad)

    def test_patch_coordinates_match_non_square_feature_cells(self) -> None:
        height, width = 13, 17
        values = candidates(batch=1)
        values["candidate_anchors"].fill_(0.5)
        values["candidate_anchors"][0, 0] = torch.tensor(
            ((8.5 / width), (6.5 / height))
        )
        values["candidate_anchors"][0, 1] = torch.tensor(
            ((0.5 / width), (0.5 / height))
        )
        values["candidate_anchors"][0, 2] = torch.tensor(
            ((4.75 + 0.5) / width, (3.25 + 0.5) / height)
        )
        y = torch.arange(height, dtype=torch.float32)[:, None]
        x = torch.arange(width, dtype=torch.float32)[None, :]
        coordinate_map = 100.0 * y + x
        stroke = coordinate_map[None, None].expand(1, 96, -1, -1).clone()

        patches = extract_predicted_anchor_stroke_patches(stroke, values)

        center = patches[0, 0, 0]
        self.assertAlmostEqual(float(center[4, 4]), 608.0, places=4)
        self.assertAlmostEqual(float(center[4, 3]), 607.0, places=4)
        self.assertAlmostEqual(float(center[4, 5]), 609.0, places=4)
        self.assertAlmostEqual(float(center[3, 4]), 508.0, places=4)
        self.assertAlmostEqual(float(center[5, 4]), 708.0, places=4)
        boundary = patches[0, 1, 0]
        self.assertAlmostEqual(float(boundary[4, 4]), 0.0, places=4)
        self.assertAlmostEqual(float(boundary[4, 3]), 0.0, places=4)
        self.assertAlmostEqual(float(boundary[3, 4]), 0.0, places=4)
        self.assertAlmostEqual(float(boundary[4, 5]), 1.0, places=4)
        self.assertAlmostEqual(float(boundary[5, 4]), 100.0, places=4)
        fractional = patches[0, 2, 0]
        self.assertAlmostEqual(float(fractional[4, 4]), 329.75, places=4)
        self.assertAlmostEqual(float(fractional[4, 5]), 330.75, places=4)
        self.assertAlmostEqual(float(fractional[5, 4]), 429.75, places=4)

    def test_out_of_canvas_candidate_anchor_is_rejected(self) -> None:
        values = candidates(batch=1)
        values["candidate_anchors"][0, 0, 0] = 1.01

        with self.assertRaisesRegex(ValueError, "in \\[0,1\\]"):
            extract_predicted_anchor_stroke_patches(
                torch.randn(1, 96, 13, 17), values
            )

    def test_shape_and_gradient_reach_only_model(self) -> None:
        values = candidates(batch=1)
        for name in ("candidate_features", "candidate_anchors", "base_logits"):
            values[name].requires_grad_()
        patches = torch.randn(1, 128, 96, 9, 9, requires_grad=True)
        model = G3StrongTop128SetRanker(config()).train()
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.01)

        output = model(values, patches)
        output["ranking_logits"].sum().backward()

        self.assertEqual(output["ranking_logits"].shape, (1, 128))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertIsNone(values["candidate_features"].grad)
        self.assertIsNone(values["candidate_anchors"].grad)
        self.assertIsNone(values["base_logits"].grad)
        self.assertIsNone(patches.grad)

    def test_cap_align_and_sham_preserve_required_patch_contracts(self) -> None:
        patches = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128, 1, 1, 1)
        patches = patches.expand(-1, -1, 96, 9, 9).clone()
        aligned = align_candidate_patches(patches)
        cap = cap_zero_candidate_patches(patches)
        first = sham_candidate_patches(
            patches, ["sample-a", "sample-b"], seed=20260713
        )
        second = sham_candidate_patches(
            patches, ["sample-a", "sample-b"], seed=20260713
        )

        torch.testing.assert_close(aligned, patches, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(cap)), 0)
        torch.testing.assert_close(
            first["candidate_patches"].sort(dim=1).values,
            aligned.sort(dim=1).values,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(first["permutation"], second["permutation"])
        identity = torch.arange(128)[None]
        self.assertFalse(bool((first["permutation"] == identity).any()))

    def test_sham_permutation_depends_on_key_and_seed(self) -> None:
        first = deterministic_sham_permutations(["a"], seed=1)
        repeated = deterministic_sham_permutations(["a"], seed=1)
        changed_key = deterministic_sham_permutations(["b"], seed=1)
        changed_seed = deterministic_sham_permutations(["a"], seed=2)

        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.equal(first, changed_key))
        self.assertFalse(torch.equal(first, changed_seed))

    def test_ranker_is_permutation_equivariant(self) -> None:
        torch.manual_seed(17)
        values = candidates(batch=1)
        patches = torch.randn(1, 128, 96, 9, 9)
        model = G3StrongTop128SetRanker(config()).eval()
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.01)
        permutation = torch.randperm(128)[None]
        inverse = permutation.argsort(dim=1)

        original = model(values, patches)
        shuffled_values = permute_candidates(values, permutation)
        patch_index = permutation[..., None, None, None].expand_as(patches)
        shuffled = model(shuffled_values, patches.gather(1, patch_index))

        torch.testing.assert_close(
            shuffled["ranking_logits"].gather(1, inverse),
            original["ranking_logits"],
            rtol=2e-5,
            atol=2e-5,
        )

    def test_public_forward_has_no_supervision_or_cardinality_argument(self) -> None:
        signature = tuple(
            inspect.signature(G3StrongTop128SetRanker(config()).forward).parameters
        )
        source = inspect.getsource(G3StrongTop128SetRanker.forward)

        self.assertEqual(signature, ("candidates", "candidate_patches"))
        self.assertNotIn("target", source.lower())
        self.assertNotIn("count", source.lower())


if __name__ == "__main__":
    unittest.main()
