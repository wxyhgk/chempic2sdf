from argparse import Namespace
from types import SimpleNamespace
import inspect
import unittest
from unittest import mock

import pytest
import torch
from torch import nn

from chemvector.components.atoms.g3_top128_set_ranker import extract_frozen_top128
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms import (
    train_g3_p1e_aligned_stroke_feature_sufficiency as p1e,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    module_state_sha256,
)


def ranker_config():
    return SimpleNamespace(d_model=384, maximum_atoms=128, atom_slots_per_cell=3)


def tiny_config():
    return G3HierarchicalGraphConfig(
        input_channels=96,
        hidden_channels=16,
        d_model=384,
        atom_slots_per_cell=3,
        maximum_atoms=128,
        bond_slots_per_cell=1,
        maximum_bonds=4,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=64,
        memory_stride=1,
        incidence_hidden_dim=16,
        incidence_pair_hidden_dim=8,
        dropout=0.0,
    )


class CountingVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 96, kernel_size=1)
        self.calls = 0

    def forward_stroke_resolution(self, pixels):
        self.calls += 1
        return self.projection(pixels)


class IdentityRanker(nn.Module):
    def forward(self, candidates, candidate_patches):
        residual = torch.zeros_like(candidates["base_logits"])
        return {
            **candidates,
            "ranking_residual": residual,
            "ranking_logits": candidates["base_logits"] + residual,
        }


def synthetic_frozen_pipeline():
    config = tiny_config()
    atom = G3AtomOnlyPredictor(config).atom_head.requires_grad_(False).eval()
    with torch.no_grad():
        atom.presence.weight.zero_()
        atom.presence.bias.zero_()
        atom.anchor_offset.weight.zero_()
        atom.anchor_offset.bias.zero_()
    vision = CountingVision().requires_grad_(False).eval()
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        dense = atom(vision.forward_stroke_resolution(pixels))
        candidates = extract_frozen_top128(dense)
    vision.calls = 0
    anchors = torch.zeros(1, 128, 2)
    anchors[0, :2] = candidates["candidate_anchors"][0, :2]
    batch = {
        "pixel_values": pixels,
        "atom_anchors": anchors,
        "atom_mask": torch.tensor([[True, True] + [False] * 126]),
        "keys": ["synthetic-a"],
    }
    return atom, vision, batch, config


class G3P1ETrainerTests(unittest.TestCase):
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "superseded frozen hash chain: dependency g3_boundary_band_loss "
            "was revised to 032b829189a8 at ec2b846 while the preregistered "
            "pin records the unrecoverable training version 7ea06a351b43; "
            "P1-E is terminal NO-GO. strict canary: clear after a formal "
            "re-pin or source restoration."
        ),
    )
    def test_dependency_sources_are_preregistered(self):
        self.assertEqual(
            p1e.validate_dependency_sources(), p1e.EXPECTED_DEPENDENCY_SHA256
        )

    def test_identical_arms_have_equal_state_and_disjoint_storage(self):
        arms, hashes = p1e.build_identical_arms(
            ranker_config(), torch.device("cpu")
        )

        self.assertEqual(len(set(hashes.values())), 1)
        self.assertTrue(p1e.arm_parameter_storage_is_disjoint(arms))
        self.assertTrue(
            all(model.num_parameters() == 2_427_713 for model in arms.values())
        )
        identities = [
            {id(value) for value in (*model.parameters(), *model.buffers())}
            for model in arms.values()
        ]
        self.assertFalse(identities[0] & identities[1])
        self.assertFalse(identities[0] & identities[2])
        self.assertFalse(identities[1] & identities[2])

    def test_optimizers_are_exclusive_and_reject_cross_arm_parameter(self):
        atom, _, _, _ = synthetic_frozen_pipeline()
        arms, _ = p1e.build_identical_arms(ranker_config(), torch.device("cpu"))
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=3e-4)
            for name, model in arms.items()
        }
        p1e.validate_optimizers(atom, arms, optimizers)

        optimizers["cap"].param_groups[0]["params"].append(
            next(arms["sham"].parameters())
        )
        with self.assertRaisesRegex(ValueError, "optimizer parameters differ"):
            p1e.validate_optimizers(atom, arms, optimizers)

    def test_arm_patch_contract_and_epoch_seed_schedule(self):
        aligned = torch.arange(128, dtype=torch.float32).reshape(1, 128, 1, 1, 1)
        aligned = aligned.expand(-1, -1, 96, 9, 9).clone()
        first_patches, first_permutations = p1e.build_arm_patches(
            aligned, ["sample"], seed=20_260_714
        )
        second_patches, second_permutations = p1e.build_arm_patches(
            aligned, ["sample"], seed=20_260_715
        )

        self.assertEqual(int(torch.count_nonzero(first_patches["cap"])), 0)
        torch.testing.assert_close(first_patches["align"], aligned, rtol=0, atol=0)
        self.assertFalse(
            torch.equal(first_permutations["sham"], second_permutations["sham"])
        )
        identity = torch.arange(128)[None]
        self.assertFalse(bool((first_permutations["sham"] == identity).any()))
        inverse = first_permutations["sham"].argsort(dim=1)
        restored = first_patches["sham"].gather(
            1, inverse[..., None, None, None].expand_as(aligned)
        )
        torch.testing.assert_close(restored, aligned, rtol=0, atol=0)
        self.assertEqual(
            [20_260_713 + epoch for epoch in range(1, 6)],
            [20_260_714, 20_260_715, 20_260_716, 20_260_717, 20_260_718],
        )

    def test_training_step_uses_one_frozen_forward_and_updates_all_arms(self):
        atom, vision, batch, config = synthetic_frozen_pipeline()
        arms, _ = p1e.build_identical_arms(config, torch.device("cpu"))
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            for name, model in arms.items()
        }
        atom_before = module_state_sha256(atom)
        vision_before = module_state_sha256(vision)
        arm_before = {name: module_state_sha256(model) for name, model in arms.items()}

        metrics, evidence = p1e.p1e_training_step(
            atom,
            arms,
            vision,
            optimizers,
            batch,
            device=torch.device("cpu"),
            precision="fp32",
            max_grad_norm=1.0,
            seed=20_260_714,
        )

        self.assertEqual(vision.calls, 1)
        self.assertEqual(atom_before, module_state_sha256(atom))
        self.assertEqual(vision_before, module_state_sha256(vision))
        for name in p1e.ARMS:
            self.assertGreater(metrics[name]["gradient_norm"], 0.0)
            self.assertNotEqual(arm_before[name], module_state_sha256(arms[name]))
        self.assertFalse(bool((
            evidence["permutations"]["sham"] == torch.arange(128)[None]
        ).any()))
        p1e.validate_optimizers(atom, arms, optimizers)

    def test_seed_bank_summary_and_tie_break_are_deterministic(self):
        per_seed = {
            31_013: {"score": 0.8, "other": 0.2},
            31_001: {"score": 0.8, "other": 0.4},
            31_019: {"score": 0.7, "other": 0.3},
        }
        result = p1e.summarize_seed_bank(
            per_seed,
            metric_paths={"score": ("score",), "other": ("other",)},
            selection_path=("score",),
        )

        self.assertEqual(result["best_seed"], 31_001)
        self.assertEqual(result["best"], per_seed[31_001])
        self.assertAlmostEqual(result["summary"]["score"]["mean"], 2.3 / 3)
        self.assertEqual(result["summary"]["score"]["min"], 0.7)
        self.assertEqual(result["summary"]["score"]["max"], 0.8)
        self.assertEqual(result["summary"]["other"]["best"], 0.4)
        locked = p1e.summarize_seed_bank(
            per_seed,
            metric_paths={"score": ("score",)},
            selection_path=("score",),
            selected_seed=31_013,
        )
        self.assertEqual(locked["best_seed"], 31_013)

    def test_train_and_route_evaluators_use_one_stream_for_five_seed_bank(self):
        atom, vision, batch, _ = synthetic_frozen_pipeline()
        arms = {name: IdentityRanker() for name in p1e.ARMS}
        calls = []

        def one_stream(*args, **kwargs):
            calls.append((args, kwargs))
            return iter([batch])

        with mock.patch.object(p1e.p1a, "iter_manifest_batches", side_effect=one_stream):
            train = p1e.evaluate_train_all_arms(
                atom,
                arms,
                vision,
                ["unused"],
                object(),
                device=torch.device("cpu"),
                precision="fp32",
                batch_size=1,
                sham_seeds=p1e.SHAM_EVAL_SEEDS,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(train["sham_eval_bank"]["seeds"], list(p1e.SHAM_EVAL_SEEDS))
        self.assertEqual(
            train["align_on_sham_eval_bank"]["seeds"], list(p1e.SHAM_EVAL_SEEDS)
        )
        for bank_name in ("sham_eval_bank", "align_on_sham_eval_bank"):
            for value in train[bank_name]["per_seed"].values():
                self.assertEqual(
                    value["input_audit"]["permutation_fixed_points"], 0
                )

        calls.clear()
        with mock.patch.object(p1e.p1a, "iter_manifest_batches", side_effect=one_stream):
            route = p1e.evaluate_route_all_arms(
                atom,
                arms,
                vision,
                ["unused"],
                object(),
                device=torch.device("cpu"),
                precision="fp32",
                batch_size=1,
                sham_seeds=p1e.SHAM_EVAL_SEEDS,
                include_counterfactual=True,
                selected_sham_seed=train["sham_eval_bank"]["best_seed"],
                selected_counterfactual_seed=train[
                    "align_on_sham_eval_bank"
                ]["best_seed"],
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(route["sham_eval_bank"]["seeds"], list(p1e.SHAM_EVAL_SEEDS))
        self.assertEqual(
            route["sham_eval_bank"]["best_seed"],
            train["sham_eval_bank"]["best_seed"],
        )
        self.assertEqual(
            route["align_on_sham_eval_bank"]["best_seed"],
            train["align_on_sham_eval_bank"]["best_seed"],
        )

    def test_causal_gate_is_mutually_exclusive(self):
        def train_arm(rate):
            return {
                "boundary_exact_rate": rate,
                "hybrid_topm_whole_set_rate": rate,
                "hybrid_topm_recall": 1.0 if rate >= 0.9 else rate,
            }

        route_arm = {
            "known_count": {
                "precision_6px": 1.0,
                "recall_6px": 1.0,
                "f1_6px": 1.0,
                "matched_atoms": 3_080,
                "whole_set_exact_images": 100,
            },
            "canonical_identity": {"recall": 1.0},
        }
        route = {
            "arms": {name: route_arm for name in p1e.ARMS},
            "top128_coverage": {"matched": 3_080, "total": 3_080},
            "fixed_positive_localization": {"matched": 3_080, "total": 3_080},
        }
        train = {
            "arms": {
                "cap": train_arm(0.5),
                "sham": train_arm(0.6),
                "align": train_arm(0.95),
                "align_on_sham": train_arm(0.7),
            }
        }

        result = p1e.arm_gates(train, route)

        self.assertEqual(
            result["causal_decision"],
            "candidate_aligned_stroke_information_necessary",
        )
        self.assertTrue(result["causal_checks"]["align_minus_sham_15pp"])
        self.assertTrue(
            result["causal_checks"]["align_counterfactual_drop_15pp"]
        )

    def test_checkpoint_and_cli_are_protocol_fixed_and_arm_local(self):
        args = p1e.build_parser().parse_args([])
        p1e.validate_args(args)
        with self.assertRaisesRegex(ValueError, "fixed max_steps"):
            p1e.validate_args(
                p1e.build_parser().parse_args(["--max-steps", "6251"])
            )
        model = p1e.G3StrongTop128SetRanker(ranker_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        payload = p1e.checkpoint_payload(
            arm="cap",
            model=model,
            optimizer=optimizer,
            args=Namespace(batch_size=4),
            config=tiny_config(),
            lineage={"p1c_checkpoint_loaded": False, "p1d_checkpoint_loaded": False},
            audit={},
            step=1_250,
            epoch=1,
            exposure=5_000,
            losses={"loss": 1.0},
            step0={},
            train_final=None,
            route_final=None,
            gate=None,
            epoch_audits=[],
        )

        self.assertEqual(payload["arm"], "cap")
        self.assertIn("strong_ranker", payload)
        self.assertIn("optimizer", payload)
        self.assertNotIn("atom_head", payload)
        self.assertNotIn("vision", payload)
        self.assertNotIn("p1c_checkpoint", payload)
        self.assertNotIn("p1d_checkpoint", payload)
        main_source = inspect.getsource(p1e.main)
        self.assertIn("sham_seeds=(args.seed,)", main_source)
        self.assertIn("sham_seeds=SHAM_EVAL_SEEDS", main_source)
        self.assertIn("selected_sham_seed=selected_sham_seed", main_source)
        self.assertIn(
            "selected_counterfactual_seed=selected_counterfactual_seed",
            main_source,
        )


if __name__ == "__main__":
    unittest.main()
