from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.audit_g3_p0m_frozen_route_select import (
    DEFAULT_OUTPUT_DIR,
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT_SHA256,
    ROUTE_SELECT_MANIFEST_SHA256,
    build_parser,
    canonical_occupancy_audit,
    evaluate_paired_composition,
    evaluator_static_contract,
    frozen_paired_deploy_forward,
    load_p0k_overlay,
    load_route_select_manifest,
    manifest_disjointness_audit,
    p0m_gate,
    scaffold_overlap_audit,
    validate_args,
    validate_deploy_forward_contract,
)
from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3AtomPairBondIdentityPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
)
from chemvector.train_g3_hierarchical_graph import module_state_sha256


def tiny_models():
    config = G3HierarchicalGraphConfig(
        input_channels=8,
        hidden_channels=16,
        d_model=16,
        atom_slots_per_cell=2,
        maximum_atoms=4,
        bond_slots_per_cell=2,
        maximum_bonds=3,
        decoder_layers=1,
        decoder_heads=4,
        decoder_ffn_dim=32,
        memory_stride=1,
    )
    p0i = G3AtomOnlyPredictor(config)
    p0g = G3AtomPairBondIdentityPredictor(config)
    with torch.no_grad():
        count = p0i.atom_head.count[-1]
        count.weight.zero_()
        count.bias.fill_(-10.0)
        count.bias[3] = 10.0
    p0i.requires_grad_(False).eval()
    p0g.requires_grad_(False).eval()
    return p0i, p0g


def pass_metrics():
    same_slot = {
        "atom_recall": 0.91,
        "bond_representable_recall": 0.92,
        "positive_type_accuracy": 0.10,
        "positive_type_accuracy_representable": 0.96,
        "gt_bond_count": {"f1": 0.96, "skeleton_exact_images": 5},
    }
    native = {
        "positive_type_accuracy": 0.99,
        "positive_type_accuracy_representable": 0.90,
        "gt_bond_count": {"skeleton_exact_images": 4},
    }
    oracle = {
        "gt_count": {
            "atom_pair_presence_f1": 0.96,
            "atom_pair_positive_type_accuracy": 0.96,
        }
    }
    occupancy = {
        "cell_occupancy": {"overflow_gt_3": {"cells": 0}},
        "identical_normalized_anchor_collision": {"groups": 0},
    }
    return same_slot, native, oracle, occupancy


class G3P0MFrozenRouteSelectTests(unittest.TestCase):
    def test_fixed_manifest_sha_and_three_identity_namespaces_are_disjoint(self) -> None:
        route, audit = load_route_select_manifest()
        overfit, _ = load_g3_overfit32_manifest()

        result = manifest_disjointness_audit(route, overfit)

        self.assertEqual(audit["sha256"], ROUTE_SELECT_MANIFEST_SHA256)
        self.assertEqual(len(route), 100)
        self.assertEqual(result["route_molecules"], 25)
        self.assertTrue(result["all_three_zero_overlap"])
        self.assertEqual(
            result["overlap"], {"sample_key": [], "molecule_id": [], "inchikey": []}
        )

    def test_occupancy_reports_slots_overflow_and_identical_coordinates(self) -> None:
        anchors = torch.tensor(
            [
                [[0.1000, 0.1000], [0.1005, 0.1000], [0.1010, 0.1000], [0.1015, 0.1000]],
                [[0.2000, 0.2000], [0.2000, 0.2000], [0.8000, 0.8000], [0.0, 0.0]],
            ]
        )
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

        result = canonical_occupancy_audit(
            anchors,
            mask,
            ["overflow", "collision"],
            slots=3,
            height=10,
            width=10,
        )

        self.assertEqual(result["cell_occupancy"]["overflow_gt_3"]["cells"], 1)
        self.assertEqual(result["overflow_atoms"], 1)
        self.assertEqual(
            result["identical_normalized_anchor_collision"]["groups"], 1
        )
        self.assertEqual(result["canonical_slot_atom_counts"]["3"], 1)
        self.assertFalse(
            result["samples"][0]["canonical_identities"][3]["supported"]
        )

    def test_real_loader_cpu_preflight_locks_occupancy_and_reports_scaffold(self) -> None:
        route, _ = load_route_select_manifest()
        overfit, _ = load_g3_overfit32_manifest()
        route_samples = load_g3_manifest_samples(route)
        overfit_samples = load_g3_manifest_samples(overfit)
        collator = G3StructuralCollator()
        anchors = []
        masks = []
        keys = []
        for start in range(0, len(route_samples), 4):
            batch = collator(route_samples[start : start + 4])
            anchors.append(batch["atom_anchors"])
            masks.append(batch["atom_mask"])
            keys.extend(batch["keys"])

        occupancy = canonical_occupancy_audit(
            torch.cat(anchors),
            torch.cat(masks),
            keys,
            slots=3,
            height=192,
            width=192,
        )
        scaffold = scaffold_overlap_audit(route_samples, overfit_samples)

        self.assertEqual(occupancy["cell_occupancy"]["1"]["cells"], 3078)
        self.assertEqual(occupancy["cell_occupancy"]["2"]["cells"], 1)
        self.assertEqual(occupancy["cell_occupancy"]["3"]["cells"], 0)
        self.assertEqual(
            occupancy["cell_occupancy"]["overflow_gt_3"]["cells"], 0
        )
        self.assertEqual(
            occupancy["identical_normalized_anchor_collision"]["groups"], 0
        )
        self.assertFalse(scaffold["hard_gate"])
        self.assertFalse(scaffold["generator_guaranteed_disjoint"])

    def test_deploy_forward_has_no_gt_and_preserves_all_pairs_and_state(self) -> None:
        p0i, p0g = tiny_models()
        before = {
            "p0i": module_state_sha256(p0i),
            "p0g": module_state_sha256(p0g),
        }

        atoms, native, same_slot = frozen_paired_deploy_forward(
            p0i, p0g, torch.randn(2, 8, 8, 8), precision="fp32"
        )

        contract = validate_deploy_forward_contract()
        self.assertFalse(contract["accepts_targets"])
        self.assertEqual(contract["compose_pair_forward_calls"], 2)
        self.assertEqual(atoms["atom_mask"].sum(dim=1).tolist(), [3, 3])
        self.assertEqual(len(same_slot["pair_indices"]), 6)
        self.assertTrue(torch.equal(native["pair_indices"], same_slot["pair_indices"]))
        self.assertTrue(
            bool((same_slot["pair_indices"][:, 0] < same_slot["pair_indices"][:, 1]).all())
        )
        self.assertEqual(
            before,
            {"p0i": module_state_sha256(p0i), "p0g": module_state_sha256(p0g)},
        )

    def test_paired_evaluator_reports_variants_molecules_and_occupancy(self) -> None:
        p0i, p0g = tiny_models()
        features = torch.randn(4, 8, 8, 8)
        with torch.no_grad():
            atoms = p0i(features)
        targets = {
            "atom_mask": atoms["atom_mask"].clone(),
            "atom_anchors": atoms["atom_anchors"].clone(),
            "bond_mask": torch.tensor([[True, True, False]] * 4),
            "bond_atom_refs": torch.tensor([[[0, 1], [1, 2], [0, 0]]] * 4),
            "bond_types": torch.tensor([[1, 2, 0]] * 4),
            "bond_display": torch.tensor([[1, 2, 0]] * 4),
        }
        keys = [f"sample_{index}" for index in range(4)]
        variants = ["coordgen", "original", "rdkit", "stress"]
        occupancy = {
            "samples": [
                {"sample_key": key, "maximum_occupancy": 1} for key in keys
            ]
        }

        result = evaluate_paired_composition(
            p0i,
            p0g,
            features,
            targets,
            keys,
            ["molecule"] * 4,
            variants,
            occupancy,
            batch_size=4,
            precision="fp32",
        )

        for branch in ("native", "same_slot"):
            self.assertEqual(set(result[branch]["per_variant"]), set(variants))
            self.assertEqual(result[branch]["per_occupancy_bucket"]["1"]["images"], 4)
            molecules = result[branch]["molecule_four_variant_exact"]
            self.assertEqual(molecules["molecules"], 1)
            self.assertEqual(molecules["complete_four_variant_molecules"], 1)
            self.assertEqual(len(result[branch]["per_sample_exact"]), 4)
        self.assertEqual(
            result["native"]["pair_identity_sha256"],
            result["same_slot"]["pair_identity_sha256"],
        )

    def test_gate_requires_protocol_oracle_same_slot_and_paired_checks(self) -> None:
        same_slot, native, oracle, occupancy = pass_metrics()

        passed = p0m_gate(
            protocol_checks={"fixed": True},
            same_slot=same_slot,
            native=native,
            oracle=oracle,
            occupancy=occupancy,
        )

        self.assertEqual(passed["decision"], "GO")
        self.assertTrue(passed["confirm400_may_be_read"])
        same_slot["positive_type_accuracy_representable"] = 0.94
        failed = p0m_gate(
            protocol_checks={"fixed": True},
            same_slot=same_slot,
            native=native,
            oracle=oracle,
            occupancy=occupancy,
        )
        self.assertEqual(failed["decision"], "NO-GO")
        self.assertFalse(failed["confirm400_may_be_read"])

    def test_p0k_overlay_is_strict_and_leaves_all_parameters_frozen(self) -> None:
        _, source = tiny_models()
        _, target = tiny_models()
        payload = {
            "model_type": "g3-predicted-anchor-relation-alignment-p0k-v0.1",
            "global_step": 400,
            "source_p0i_checkpoint": {"sha256": P0I_CHECKPOINT_SHA256},
            "source_p0g_checkpoint": {"sha256": P0G_CHECKPOINT_SHA256},
            "pair_bond": source.pair_bond.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p0k.pt"
            torch.save(payload, path)
            with patch(
                "chemvector.audit_g3_p0m_frozen_route_select.file_sha256",
                return_value="expected",
            ), patch(
                "chemvector.audit_g3_p0m_frozen_route_select.P0K_CHECKPOINT_SHA256",
                "expected",
            ):
                restored = load_p0k_overlay(target, path)

        self.assertEqual(restored["global_step"], 400)
        for name, value in source.pair_bond.state_dict().items():
            torch.testing.assert_close(target.pair_bond.state_dict()[name], value)
        self.assertFalse(any(parameter.requires_grad for parameter in target.parameters()))

    def test_cli_and_static_read_only_contract_are_fixed(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(Path(args.output_dir), DEFAULT_OUTPUT_DIR)
        with self.assertRaisesRegex(ValueError, "fixed CUDA BF16"):
            validate_args(build_parser().parse_args(["--device", "cpu", "--precision", "fp32"]))

        contract = evaluator_static_contract()
        self.assertEqual(contract["optimizer_references"], 0)
        self.assertEqual(contract["backward_calls"], 0)
        self.assertEqual(contract["updates"], 0)
        self.assertEqual(contract["forbidden_data_path_references"], 0)


if __name__ == "__main__":
    unittest.main()
