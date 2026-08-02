from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from torch import nn

from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.g3_structural_data import G3SampleManifestRecord
from chemvector.train_g3_hierarchical_graph import module_state_sha256
from chemvector.train_g3_p1a_canonical_atom_selector import (
    DEFAULT_OUTPUT,
    MODEL_TYPE,
    P0I_CHECKPOINT,
    TRAIN_MANIFEST_SHA256,
    TRAIN_SEQUENCE_SHA256,
    CanonicalOccupancyAccumulator,
    G3AtomCollator,
    atom_capability_gate,
    atom_optimizer_is_exact,
    atom_sample_statistics,
    build_parser,
    checkpoint_payload,
    checkpoint_step_files,
    epoch_audits_are_exact,
    evaluate_route_streaming,
    identity_overlap_audit,
    iter_manifest_batches,
    iter_manifest_samples,
    load_train_manifest,
    load_warmstart_p0i,
    p1a_gate,
    p1a_training_step,
    static_training_contract,
    validate_args,
    validate_atom_optimizer,
    validate_checkpoint_files,
    validate_occupancy,
)


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward_stroke_resolution(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels)


class FixedAtomModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = len(features)
        anchors = features.new_zeros(batch, 4, 2)
        anchors[:, 0] = features.new_tensor([0.5, 0.5])
        mask = torch.zeros(batch, 4, device=features.device, dtype=torch.bool)
        mask[:, 0] = True
        return {"atom_anchors": anchors, "atom_mask": mask}


def tiny_config() -> G3HierarchicalGraphConfig:
    return G3HierarchicalGraphConfig(
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


def fake_sample(key: str, molecule: str, inchikey: str, variant: str) -> dict:
    return {
        "key": key,
        "image": Image.new("RGB", (32, 32), "white"),
        "chemmol": {
            "molecule_id": molecule,
            "inchikey": inchikey,
        },
        "render": {
            "variant": variant,
            "atoms": [
                {
                    "atom_id": 0,
                    "image_x": 15.5,
                    "image_y": 15.5,
                    "quantized_x": 512,
                    "quantized_y": 512,
                }
            ],
        },
        "layers": {
            "width": 32,
            "height": 32,
            "atoms": [{"atom_id": 0, "center": [15.5, 15.5]}],
        },
        "meta": {"split": "train"},
    }


def fake_manifest_and_samples(count: int = 4):
    variants = ["coordgen", "original", "rdkit", "stress"]
    records = []
    samples = []
    for index in range(count):
        key = f"sample_{index}"
        variant = variants[index % 4]
        records.append(
            G3SampleManifestRecord(
                sample_key=key,
                molecule_id="molecule",
                inchikey="KEY",
                shard="fake.tar",
                variant=variant,
            )
        )
        samples.append(fake_sample(key, "molecule", "KEY", variant))
    return records, samples


class G3P1ACanonicalAtomSelectorTests(unittest.TestCase):
    def test_fixed_train_manifest_contract(self) -> None:
        manifest, audit = load_train_manifest()

        self.assertEqual(audit["sha256"], TRAIN_MANIFEST_SHA256)
        self.assertEqual(audit["sample_keys_sha256"], TRAIN_SEQUENCE_SHA256)
        self.assertEqual(len(manifest), 5_000)
        self.assertEqual(audit["molecule_count"], 1_250)

    def test_stream_preserves_order_and_batches_without_residual(self) -> None:
        manifest, samples = fake_manifest_and_samples()

        def factory(shards, include_keys):
            self.assertEqual(shards, ["fake.tar"])
            self.assertEqual(include_keys, {item.sample_key for item in manifest})
            return iter(samples)

        streamed = list(iter_manifest_samples(manifest, dataset_factory=factory))
        batches = list(
            iter_manifest_batches(
                manifest,
                G3AtomCollator(maximum_atoms=4, image_size=32),
                batch_size=4,
                dataset_factory=factory,
            )
        )

        self.assertEqual([sample["key"] for sample in streamed], [item.sample_key for item in manifest])
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["keys"], [item.sample_key for item in manifest])
        self.assertEqual(
            set(batches[0]),
            {"pixel_values", "atom_anchors", "atom_mask", "keys", "molecule_ids", "variants"},
        )

    def test_stream_exposes_residual_batch_for_protocol_rejection(self) -> None:
        manifest, samples = fake_manifest_and_samples(count=5)

        def factory(shards, include_keys):
            return iter(samples)

        batches = list(
            iter_manifest_batches(
                manifest,
                G3AtomCollator(maximum_atoms=4, image_size=32),
                batch_size=4,
                dataset_factory=factory,
            )
        )

        self.assertEqual([len(batch["keys"]) for batch in batches], [4, 1])

    def test_occupancy_rejects_overflow_and_identical_coordinates(self) -> None:
        accumulator = CanonicalOccupancyAccumulator(slots=3, height=10, width=10)
        accumulator.add(
            "overflow",
            [(0.1000, 0.1), (0.1005, 0.1), (0.1010, 0.1), (0.1015, 0.1)],
        )
        accumulator.add("collision", [(0.2, 0.2), (0.2, 0.2)])
        result = accumulator.finalize()

        self.assertEqual(result["overflow_gt_3"]["cells"], 1)
        self.assertEqual(result["identical_coordinate_collision"]["groups"], 1)
        with self.assertRaisesRegex(ValueError, "occupancy above three"):
            validate_occupancy(result, name="test")

    def test_scaffold_is_report_only_but_three_identities_are_hard_gate(self) -> None:
        left = [G3SampleManifestRecord("a", "m1", "k1", "s", "coordgen")]
        right = [G3SampleManifestRecord("b", "m2", "k2", "s", "coordgen")]
        result = identity_overlap_audit(
            left,
            right,
            {"scaffolds": ["same"]},
            {"scaffolds": ["same"]},
        )

        self.assertTrue(result["all_three_identity_zero"])
        self.assertEqual(result["scaffold_overlap_count"], 1)
        self.assertFalse(result["scaffold_hard_gate"])

    def test_atom_matching_metrics_use_same_micro_denominators(self) -> None:
        current = atom_sample_statistics(
            torch.tensor([[0.10, 0.10], [0.90, 0.90]]),
            torch.tensor([[0.10, 0.10], [0.50, 0.50], [0.90, 0.90]]),
        )

        self.assertEqual(current["matched_atoms"], 2.0)
        self.assertEqual(current["predicted_atoms"], 2.0)
        self.assertEqual(current["target_atoms"], 3.0)
        self.assertEqual(current["count_error_distribution"], {-1: 1})
        self.assertEqual(current["whole_set_exact_images"], 0.0)

    def test_streaming_route_evaluator_reports_all_fixed_buckets(self) -> None:
        manifest, samples = fake_manifest_and_samples()

        def factory(shards, include_keys):
            return iter(samples)

        occupancy = {
            "sample_maximum_occupancy": {
                item.sample_key: 1 for item in manifest
            }
        }
        model = FixedAtomModel().eval()
        vision = TinyVision().requires_grad_(False).eval()

        metrics = evaluate_route_streaming(
            model,
            vision,
            manifest,
            G3AtomCollator(maximum_atoms=4, image_size=32),
            occupancy,
            device=torch.device("cpu"),
            precision="fp32",
            batch_size=4,
            dataset_factory=factory,
        )

        self.assertEqual(metrics["images"], 4)
        self.assertEqual(metrics["count_exact_images"], 4)
        self.assertEqual(metrics["whole_set_exact_images"], 4)
        self.assertEqual(metrics["precision_6px"], 1.0)
        self.assertEqual(set(metrics["per_variant"]), {"coordgen", "original", "rdkit", "stress"})
        self.assertEqual(metrics["per_occupancy"]["1"]["images"], 4)
        self.assertEqual(metrics["per_atom_count"]["1-20"]["images"], 4)

    def test_training_step_updates_only_atom_head_and_keeps_vision_frozen(self) -> None:
        model = G3AtomOnlyPredictor(tiny_config())
        vision = TinyVision().requires_grad_(False).eval()
        optimizer = torch.optim.AdamW(
            model.atom_head.parameters(), lr=1e-4, weight_decay=0.01
        )
        batch = {
            "pixel_values": torch.randn(2, 3, 32, 32),
            "atom_anchors": torch.tensor(
                [[[0.2, 0.5], [0.8, 0.5], [0.0, 0.0], [0.0, 0.0]]] * 2
            ),
            "atom_mask": torch.tensor([[True, True, False, False]] * 2),
            "keys": ["a", "b"],
        }
        atom_before = module_state_sha256(model.atom_head)
        vision_before = module_state_sha256(vision)

        losses = p1a_training_step(
            model,
            vision,
            optimizer,
            batch,
            device=torch.device("cpu"),
            precision="fp32",
            max_grad_norm=1.0,
        )

        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertNotEqual(atom_before, module_state_sha256(model.atom_head))
        self.assertEqual(vision_before, module_state_sha256(vision))
        self.assertFalse(any(parameter.grad is not None for parameter in vision.parameters()))
        invalid = torch.optim.AdamW(
            list(model.atom_head.parameters()) + list(vision.parameters()), lr=1e-4
        )
        with self.assertRaisesRegex(ValueError, "only Atom-head"):
            validate_atom_optimizer(model, invalid)
        self.assertTrue(atom_optimizer_is_exact(model, optimizer))
        self.assertFalse(atom_optimizer_is_exact(model, invalid))

    def test_real_p0i_warmstart_strict_loads_on_cpu(self) -> None:
        model, payload, config = load_warmstart_p0i(
            Path(P0I_CHECKPOINT), torch.device("cpu")
        )

        self.assertEqual(payload["global_step"], 2_400)
        self.assertEqual(config.atom_slots_per_cell, 3)
        self.assertEqual(config.maximum_atoms, 128)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_gate_uses_fixed_final_atom_thresholds(self) -> None:
        metrics = {
            "count_exact_images": 80,
            "count_mae": 1.0,
            "precision_6px": 0.97,
            "recall_6px": 0.97,
            "f1_6px": 0.97,
            "whole_set_exact_images": 50,
        }

        self.assertEqual(atom_capability_gate(metrics)["status"], "passed")
        self.assertEqual(
            p1a_gate({"protocol": True}, metrics)["decision"], "GO"
        )
        metrics["whole_set_exact_images"] = 49
        self.assertEqual(
            p1a_gate({"protocol": True}, metrics)["decision"], "NO-GO"
        )

    def test_cli_source_and_checkpoint_contracts_are_fixed(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(Path(args.output_dir), DEFAULT_OUTPUT)
        with self.assertRaisesRegex(ValueError, "fixed max_steps"):
            validate_args(build_parser().parse_args(["--max-steps", "6251"]))
        source = static_training_contract()
        self.assertEqual(source["forbidden_data_path_references"], 0)
        self.assertEqual(source["forbidden_model_identifiers"], 0)
        self.assertEqual(source["vision_calls_per_training_step"], 1)
        self.assertEqual(source["teacher_forced_count_references"], 0)

        model = G3AtomOnlyPredictor(tiny_config())
        optimizer = torch.optim.AdamW(model.atom_head.parameters(), lr=1e-4)
        completed_epoch_audits = [
            {
                "epoch": 1,
                "images": 5_000,
                "steps": 1_250,
                "sample_keys_sha256": "sequence",
            }
        ]
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=Namespace(batch_size=4),
            config=tiny_config(),
            lineage={},
            audit={},
            step=1_250,
            epoch=1,
            exposure=5_000,
            losses={"loss": 1.0},
            step0_metrics={},
            final_metrics=None,
            gate=None,
            completed_epoch_audits=completed_epoch_audits,
        )
        self.assertEqual(payload["model_type"], MODEL_TYPE)
        self.assertIn("atom_head", payload)
        self.assertNotIn("model", payload)
        self.assertNotIn("vision", payload)
        self.assertEqual(payload["completed_epoch_audits"], completed_epoch_audits)
        self.assertIsNone(payload["trainer_sha256"])

    def test_checkpoint_file_set_is_exact(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for step in (1_250, 2_500):
                (output_dir / f"step-{step:06d}.pt").touch()

            self.assertEqual(
                checkpoint_step_files(output_dir),
                ["step-001250.pt", "step-002500.pt"],
            )
            validate_checkpoint_files(output_dir, [1_250, 2_500])
            (output_dir / "step-003750.pt").touch()
            with self.assertRaisesRegex(RuntimeError, "checkpoint file set differs"):
                validate_checkpoint_files(output_dir, [1_250, 2_500])

    def test_epoch_audit_requires_all_five_fixed_sequences(self) -> None:
        audits = [
            {
                "epoch": epoch,
                "images": 5_000,
                "steps": 1_250,
                "sample_keys_sha256": TRAIN_SEQUENCE_SHA256,
            }
            for epoch in range(1, 6)
        ]

        self.assertTrue(epoch_audits_are_exact(audits))
        self.assertFalse(epoch_audits_are_exact([]))
        audits[-1]["sample_keys_sha256"] = "wrong"
        self.assertFalse(epoch_audits_are_exact(audits))


if __name__ == "__main__":
    unittest.main()
