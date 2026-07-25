from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.dense_bond_object import DenseBondObjectConfig, DenseBondObjectDetector
from chemvector.pretrain_dense_bond_object import (
    dual_domain_validation_score,
    MixedSamplingAudit,
    expand_compatible_state_dict,
    prepare_output_directory,
    select_stratified_replay_indices,
    stream_mixed_training_batches,
    stream_training_batches,
    validation_score,
)


class DenseBondObjectStreamingTest(unittest.TestCase):
    def test_output_directory_rejects_stale_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "best.pt").write_bytes(b"stale")

            with self.assertRaisesRegex(FileExistsError, "best.pt"):
                prepare_output_directory(output_dir)

    def test_output_directory_rejects_stale_sampling_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "sampling_audit.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "sampling_audit.json"):
                prepare_output_directory(output_dir)

    def test_output_directory_allows_empty_or_unrelated_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "notes.txt").write_text("kept", encoding="utf-8")

            prepare_output_directory(output_dir)

            self.assertTrue((output_dir / "notes.txt").is_file())

    def test_validation_score_can_keep_initial_checkpoint(self) -> None:
        initial = {
            "oracle_junction_sdf_exact": 0.04,
            "known_object_exact": 0.05,
            "image_max_endpoint_error_p95": 300.0,
        }
        regressed = {
            "oracle_junction_sdf_exact": 0.03,
            "known_object_exact": 0.06,
            "image_max_endpoint_error_p95": 250.0,
        }

        self.assertGreater(validation_score(initial), validation_score(regressed))

    def test_dual_domain_score_prioritizes_the_weaker_domain(self) -> None:
        balanced_primary = {
            "known_object_exact": 0.20,
            "known_object_accuracy": 0.70,
            "image_max_endpoint_error_p95": 300.0,
        }
        balanced_retention = {
            "known_object_exact": 0.18,
            "known_object_accuracy": 0.68,
            "image_max_endpoint_error_p95": 310.0,
        }
        specialized_primary = {
            "known_object_exact": 0.30,
            "known_object_accuracy": 0.80,
            "image_max_endpoint_error_p95": 250.0,
        }
        forgotten_retention = {
            "known_object_exact": 0.10,
            "known_object_accuracy": 0.60,
            "image_max_endpoint_error_p95": 250.0,
        }

        self.assertGreater(
            dual_domain_validation_score(balanced_primary, balanced_retention),
            dual_domain_validation_score(specialized_primary, forgotten_retention),
        )

    def test_compatible_state_expands_output_channel_prefix(self) -> None:
        target = {"head.weight": torch.full((5, 2), 9.0)}
        source = {"head.weight": torch.arange(6).reshape(3, 2).float()}

        expanded = expand_compatible_state_dict(target, source)

        self.assertTrue(torch.equal(expanded["head.weight"][:3], source["head.weight"]))
        self.assertTrue(torch.equal(expanded["head.weight"][3:], target["head.weight"][3:]))

    def test_slot_expansion_preserves_old_detector_outputs(self) -> None:
        old = DenseBondObjectDetector(
            DenseBondObjectConfig(input_channels=8, hidden_channels=16, slots=3)
        ).eval()
        torch.manual_seed(17)
        expanded = DenseBondObjectDetector(
            DenseBondObjectConfig(input_channels=8, hidden_channels=16, slots=5)
        ).eval()
        expanded.load_state_dict(
            expand_compatible_state_dict(expanded.state_dict(), old.state_dict())
        )
        features = torch.randn(2, 8, 12, 12)

        expected = old(features)
        actual = expanded(features)

        self.assertTrue(torch.equal(expected["object_features"], actual["object_features"]))
        for name in (
            "heatmap_logits",
            "center_offsets",
            "centers",
            "endpoints",
            "doubled_orientation",
            "half_length",
            "display_logits",
            "chemical_logits",
            "direction_logits",
        ):
            self.assertTrue(torch.equal(expected[name], actual[name][:, :3]), name)

    def test_streaming_batches_preserve_exact_nested_sample_pool(self) -> None:
        records = [{"key": str(index)} for index in range(8)]
        with patch(
            "chemvector.pretrain_dense_bond_object.expand_shards",
            return_value=["unused.tar"],
        ), patch(
            "chemvector.pretrain_dense_bond_object.ChemVectorIterableDataset",
            return_value=records,
        ):
            batches = list(
                stream_training_batches(
                    ["unused.tar"],
                    samples=7,
                    batch_size=3,
                    shuffle_buffer=4,
                    seed=11,
                )
            )

        selected = [record["key"] for batch in batches for record in batch]
        self.assertEqual(sorted(selected), [str(index) for index in range(7)])
        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])

    def test_streaming_batches_reject_incomplete_source(self) -> None:
        with patch(
            "chemvector.pretrain_dense_bond_object.expand_shards",
            return_value=["unused.tar"],
        ), patch(
            "chemvector.pretrain_dense_bond_object.ChemVectorIterableDataset",
            return_value=[{"key": "only"}],
        ):
            with self.assertRaisesRegex(ValueError, "loaded and emitted 1"):
                list(
                    stream_training_batches(
                        ["unused.tar"],
                        samples=2,
                        batch_size=2,
                        shuffle_buffer=2,
                        seed=11,
                    )
                )

    @staticmethod
    def _mixed_records(count: int) -> list[dict]:
        variants = ("original", "rdkit", "coordgen", "stress")
        return [
            {
                "key": f"sample-{index:03d}",
                "meta": {
                    "source": "chembl" if index % 3 else "coconut",
                    "variant": variants[index % len(variants)],
                },
            }
            for index in range(count)
        ]

    def _run_mixed_sampler(self) -> tuple[list[list[dict]], dict]:
        records = self._mixed_records(10)
        audit = MixedSamplingAudit(
            coverage_samples=10,
            replay_pool_samples=4,
            replay_samples=10,
            batch_size=4,
            seed=11,
        )
        with patch(
            "chemvector.pretrain_dense_bond_object.expand_shards",
            return_value=["unused.tar"],
        ), patch(
            "chemvector.pretrain_dense_bond_object.ChemVectorIterableDataset",
            return_value=records,
        ):
            batches = list(
                stream_mixed_training_batches(
                    ["unused.tar"],
                    coverage_samples=10,
                    replay_pool_samples=4,
                    replay_samples=10,
                    batch_size=4,
                    shuffle_buffer=8,
                    seed=11,
                    audit=audit,
                )
            )
        return batches, audit.to_dict()

    def test_mixed_sampler_is_deterministic_and_never_emits_tail_batches(self) -> None:
        first_batches, first_audit = self._run_mixed_sampler()
        second_batches, second_audit = self._run_mixed_sampler()

        first_keys = [[record["key"] for record in batch] for batch in first_batches]
        second_keys = [[record["key"] for record in batch] for batch in second_batches]
        self.assertEqual(first_keys, second_keys)
        self.assertEqual([len(batch) for batch in first_batches], [4] * 5)
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(first_audit["status"], "passed")
        self.assertEqual(first_audit["actual"]["optimizer_steps"], 5)
        self.assertEqual(
            first_audit["batch_composition_histogram"],
            {"coverage=2,replay=2": 5},
        )

    def test_mixed_sampler_audits_unique_keys_and_exposures(self) -> None:
        _, audit = self._run_mixed_sampler()

        self.assertEqual(audit["actual"]["coverage_unique_keys"], 10)
        self.assertEqual(audit["actual"]["replay_unique_keys"], 4)
        self.assertEqual(audit["actual"]["coverage_replay_intersection"], 4)
        self.assertEqual(audit["actual"]["total_unique_keys"], 10)
        self.assertEqual(audit["exposure_histogram"], {"1": 6, "3": 2, "4": 2})
        self.assertEqual(audit["replay_exposure_histogram"], {"2": 2, "3": 2})
        self.assertEqual(
            sum(audit["attribute_counts"]["coverage"]["variant"].values()),
            10,
        )
        self.assertEqual(
            sum(audit["attribute_counts"]["replay"]["source"].values()),
            10,
        )
        for digest in audit["key_sha256"].values():
            self.assertEqual(len(digest), 64)

    def test_requested_50k_replay_schedule_has_exact_6250_steps(self) -> None:
        records = self._mixed_records(50_000)
        audit = MixedSamplingAudit(
            coverage_samples=50_000,
            replay_pool_samples=8_974,
            replay_samples=50_000,
            batch_size=16,
            seed=20260718,
        )
        with patch(
            "chemvector.pretrain_dense_bond_object.expand_shards",
            return_value=["unused.tar"],
        ), patch(
            "chemvector.pretrain_dense_bond_object.ChemVectorIterableDataset",
            return_value=records,
        ):
            batches = sum(
                1
                for _ in stream_mixed_training_batches(
                    ["unused.tar"],
                    coverage_samples=50_000,
                    replay_pool_samples=8_974,
                    replay_samples=50_000,
                    batch_size=16,
                    shuffle_buffer=512,
                    seed=20260718,
                    audit=audit,
                )
            )
        payload = audit.to_dict()

        self.assertEqual(batches, 6_250)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["actual"]["optimizer_steps"], 6_250)
        self.assertEqual(
            payload["batch_composition_histogram"],
            {"coverage=8,replay=8": 6_250},
        )
        self.assertEqual(
            payload["exposure_histogram"],
            {"1": 41_026, "6": 3_844, "7": 5_130},
        )
        self.assertEqual(
            payload["replay_exposure_histogram"],
            {"5": 3_844, "6": 5_130},
        )

    def test_mixed_sampler_rejects_configuration_that_would_make_tail_batches(self) -> None:
        records = self._mixed_records(12)
        audit = MixedSamplingAudit(
            coverage_samples=11,
            replay_pool_samples=4,
            replay_samples=11,
            batch_size=4,
            seed=11,
        )
        with patch(
            "chemvector.pretrain_dense_bond_object.expand_shards",
            return_value=["unused.tar"],
        ), patch(
            "chemvector.pretrain_dense_bond_object.ChemVectorIterableDataset",
            return_value=records,
        ):
            with self.assertRaisesRegex(ValueError, "divisible"):
                list(
                    stream_mixed_training_batches(
                        ["unused.tar"],
                        coverage_samples=11,
                        replay_pool_samples=4,
                        replay_samples=11,
                        batch_size=4,
                        shuffle_buffer=8,
                        seed=11,
                        audit=audit,
                    )
                )

    def test_stratified_replay_subset_preserves_equal_strata(self) -> None:
        descriptors = [
            {
                "index": index,
                "key": f"sample-{index}",
                "source": "chembl" if index < 4 else "coconut",
                "variant": "original" if index % 2 == 0 else "stress",
            }
            for index in range(8)
        ]

        selected = select_stratified_replay_indices(descriptors, 4, seed=17)
        selected_strata = [
            (descriptors[index]["source"], descriptors[index]["variant"])
            for index in selected
        ]

        self.assertEqual(
            sorted(selected_strata),
            [
                ("chembl", "original"),
                ("chembl", "stress"),
                ("coconut", "original"),
                ("coconut", "stress"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
