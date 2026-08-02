"""Tests for ChemVector training data mixing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import serialize_chemtraverse_v0_1
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.ocr import OCRTextCodec
from chemvector.synthetic_ocr import SyntheticOCRDataset, collate_synthetic_ocr
from chemvector.train import (
    add_chemtraverse_decision_loss_arguments,
    apply_training_sequence_loss,
    capture_rng_state,
    iter_mixed_batches,
    load_sample_keys,
    load_chemtraverse_decision_objective,
    load_vision_encoder_checkpoint,
    ocr_sequence_accuracy,
    restore_rng_state,
    save_checkpoint,
    synthetic_ocr_replay_forward,
    traversal_control_token_ids,
    validate_chemtraverse_decision_loss_arguments,
    validate_resume_training_objective,
    weighted_token_loss,
)
from tests.data_sdf.test_chemtraverse_v0_1 import ring_scene


class SequenceRandom:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def random(self) -> float:
        return next(self.values)


class MixedBatchTests(unittest.TestCase):
    def test_primary_batches_are_unchanged_without_replay(self) -> None:
        batches = list(iter_mixed_batches(["p1", "p2"]))
        self.assertEqual(batches, [("p1", False), ("p2", False)])

    def test_replay_stream_restarts_when_exhausted(self) -> None:
        rng = SequenceRandom([0.1, 0.1, 0.9, 0.9, 0.9])
        batches = list(
            iter_mixed_batches(
                ["p1", "p2"],
                ["r1"],
                replay_probability=0.2,
                rng=rng,
            )
        )
        self.assertEqual(
            batches,
            [("r1", True), ("r1", True), ("p1", False), ("p2", False)],
        )

    def test_sample_manifest_loads_only_declared_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"sample_key": key}) for key in ("sample-a", "sample-b")
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_sample_keys(path), {"sample-a", "sample-b"})
        self.assertIsNone(load_sample_keys(None))

    def test_checkpoint_preserves_mid_epoch_resume_position(self) -> None:
        config = ChemVectorConfig(
            vocab_size=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            max_target_length=16,
            vision_dims=(8, 16, 24, 32),
            vision_depths=(1, 1, 1, 1),
        )
        model = ChemVectorModel(config)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=0,
                global_step=7,
                epoch_complete=False,
                samples_in_epoch=28,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertFalse(checkpoint["epoch_complete"])
        self.assertEqual(checkpoint["samples_in_epoch"], 28)
        self.assertIn("rng_state", checkpoint)
        self.assertNotIn("training_objective", checkpoint)

    def test_vision_only_initialization_does_not_replace_decoder(self) -> None:
        config = ChemVectorConfig(
            vocab_size=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            max_target_length=16,
            vision_dims=(8, 16, 24, 32),
            vision_depths=(1, 1, 1, 1),
        )
        source = ChemVectorModel(config)
        target = ChemVectorModel(config)
        original_decoder = target.token_embedding.weight.detach().clone()
        with torch.no_grad():
            source.vision_encoder.stem[0].weight.fill_(0.25)
            source.token_embedding.weight.fill_(0.75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            torch.save({"model": source.state_dict()}, path)
            loaded = load_vision_encoder_checkpoint(target, path)
        self.assertGreater(loaded, 0)
        self.assertTrue(
            torch.equal(
                target.vision_encoder.stem[0].weight,
                source.vision_encoder.stem[0].weight,
            )
        )
        self.assertTrue(torch.equal(target.token_embedding.weight, original_decoder))

    def test_rng_state_restores_python_numpy_and_torch(self) -> None:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_traversal_control_weight_emphasizes_control_error(self) -> None:
        tokenizer = ChemTraverseTokenizerV01()
        controls = traversal_control_token_ids(tokenizer)
        control = tokenizer.token_to_id[tokenizer.COMPONENT_END]
        ordinary = tokenizer.token_to_id["el:C"]
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id, control, ordinary, tokenizer.pad_token_id]]
        )
        logits = torch.zeros(1, 3, tokenizer.vocab_size)
        logits[0, 0, ordinary] = 5
        logits[0, 1, ordinary] = 5

        unweighted, _ = weighted_token_loss(
            logits,
            input_ids,
            tokenizer.pad_token_id,
            controls,
            1.0,
        )
        weighted, _ = weighted_token_loss(
            logits,
            input_ids,
            tokenizer.pad_token_id,
            controls,
            4.0,
        )

        self.assertIn(control, controls)
        self.assertGreater(float(weighted), float(unweighted))


class ChemTraverseDecisionLossIntegrationTests(unittest.TestCase):
    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sequence-version", default="0.2")
        parser.add_argument("--target-molecule-id")
        parser.add_argument("--traversal-control-loss-weight", type=float, default=1.0)
        add_chemtraverse_decision_loss_arguments(parser)
        return parser

    def test_cli_is_opt_in_and_requires_train_only_frequencies(self) -> None:
        parser = self._parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.chemtraverse_decision_loss)
        self.assertIsNone(defaults.chemtraverse_decision_frequencies)
        validate_chemtraverse_decision_loss_arguments(defaults)

        enabled = parser.parse_args(
            [
                "--sequence-version",
                "traverse-0.1",
                "--target-molecule-id",
                "sdf",
                "--chemtraverse-decision-loss",
                "--chemtraverse-decision-frequencies",
                "train-frequencies.json",
            ]
        )
        validate_chemtraverse_decision_loss_arguments(enabled)

        missing = parser.parse_args(
            [
                "--sequence-version",
                "traverse-0.1",
                "--target-molecule-id",
                "sdf",
                "--chemtraverse-decision-loss",
            ]
        )
        with self.assertRaisesRegex(ValueError, "frequencies is required"):
            validate_chemtraverse_decision_loss_arguments(missing)

    def test_frequency_artifact_freezes_weights_and_provenance(self) -> None:
        tokenizer = ChemTraverseTokenizerV01()
        payload = {
            "schema_version": "chemtraverse-decision-frequencies-0.1",
            "sequence_schema_version": "chemtraverse-0.1",
            "target_molecule_id": "sdf",
            "decision_frequencies": {
                tokenizer.DOWN: 100,
                tokenizer.UP: 100,
                tokenizer.LINK: 25,
                tokenizer.COMPONENT_END: 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequencies.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            weights, objective = load_chemtraverse_decision_objective(path, tokenizer)

        self.assertEqual(objective["name"], "grammar-aware-decision-balanced-ce")
        self.assertTrue(objective["masked_softmax"])
        self.assertEqual(len(objective["frequency_source"]["sha256"]), 64)
        self.assertEqual(weights[tokenizer.token_to_id[tokenizer.DOWN]], 1.0)
        self.assertEqual(weights[tokenizer.token_to_id[tokenizer.LINK]], 2.0)
        self.assertEqual(weights[tokenizer.token_to_id[tokenizer.COMPONENT_END]], 4.0)

    def test_frequency_artifact_must_match_target_molecule_id(self) -> None:
        tokenizer = ChemTraverseTokenizerV01()
        payload = {
            "schema_version": "chemtraverse-decision-frequencies-0.1",
            "sequence_schema_version": "chemtraverse-0.1",
            "target_molecule_id": "sdf",
            "decision_frequencies": {tokenizer.DOWN: 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequencies.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target-molecule-id differ"):
                load_chemtraverse_decision_objective(
                    path,
                    tokenizer,
                    expected_target_molecule_id="other",
                )

    def test_default_sequence_loss_is_untouched_and_opt_in_loss_is_applied(self) -> None:
        tokenizer = ChemTraverseTokenizerV01()
        input_ids = torch.tensor(
            [tokenizer.encode(serialize_chemtraverse_v0_1(ring_scene()))]
        )
        logits = torch.zeros(
            1,
            input_ids.size(1) - 1,
            tokenizer.vocab_size,
            requires_grad=True,
        )
        original_loss = logits.sum() + 7.0
        original_per_sample = original_loss.expand(1)
        output = {
            "logits": logits,
            "loss": original_loss,
            "loss_per_sample": original_per_sample,
        }

        result = apply_training_sequence_loss(output, input_ids, tokenizer)
        self.assertIsNone(result)
        self.assertIs(output["loss"], original_loss)
        self.assertIs(output["loss_per_sample"], original_per_sample)

        result = apply_training_sequence_loss(
            output,
            input_ids,
            tokenizer,
            decision_token_weights={tokenizer.token_to_id[tokenizer.LINK]: 2.0},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.valid_positions, input_ids.size(1) - 1)
        self.assertGreater(result.decision_positions, 0)
        self.assertGreater(result.forced_positions, 0)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_checkpoint_and_resume_bind_the_exact_objective(self) -> None:
        config = ChemVectorConfig(
            vocab_size=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            max_target_length=16,
            vision_dims=(8, 16, 24, 32),
            vision_depths=(1, 1, 1, 1),
        )
        model = ChemVectorModel(config)
        optimizer = torch.optim.AdamW(model.parameters())
        objective = {
            "schema_version": "chemtraverse-decision-loss-0.1",
            "target_molecule_id": "sdf",
            "control_token_weights": {"[DOWN]": 1.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=0,
                global_step=1,
                epoch_complete=False,
                samples_in_epoch=4,
                training_objective=objective,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["training_objective"], objective)
        validate_resume_training_objective(checkpoint, objective)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_resume_training_objective(checkpoint, None)


class OCRMetricTests(unittest.TestCase):
    def test_visible_and_blank_labels_are_reported_separately(self) -> None:
        labels = torch.tensor(
            [
                [OCRTextCodec.EOS_TOKEN_ID, 0],
                [ord("N") + OCRTextCodec.BYTE_OFFSET, OCRTextCodec.EOS_TOKEN_ID],
            ]
        )
        logits = torch.zeros(2, 2, OCRTextCodec.VOCAB_SIZE)
        logits[0, 0, OCRTextCodec.EOS_TOKEN_ID] = 1
        logits[1, 0, ord("B") + OCRTextCodec.BYTE_OFFSET] = 1
        logits[1, 1, OCRTextCodec.EOS_TOKEN_ID] = 1

        metrics = ocr_sequence_accuracy(
            {"ocr_labels": labels, "ocr_logits": logits}
        )

        self.assertEqual(metrics["exact"], 1)
        self.assertEqual(metrics["blank_exact"], 1)
        self.assertEqual(metrics["visible_exact"], 0)

    def test_synthetic_replay_only_updates_the_ocr_head(self) -> None:
        config = ChemVectorConfig(
            vocab_size=64,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            max_target_length=32,
            vision_dims=(16, 32, 48, 64),
            vision_depths=(1, 1, 1, 1),
            ocr_enabled=True,
            ocr_d_model=32,
            ocr_heads=4,
            ocr_ffn_dim=64,
        )
        model = ChemVectorModel(config)
        dataset = SyntheticOCRDataset(2, seed=9)
        batch = collate_synthetic_ocr(
            [dataset[0], dataset[1]],
            OCRTextCodec(max_length=config.ocr_max_length),
        )

        output = synthetic_ocr_replay_forward(model, batch, torch.device("cpu"))
        output["ocr_loss"].backward()

        self.assertIsNone(model.vision_encoder.stem[0].weight.grad)
        self.assertIsNotNone(model.ocr_head.roi_projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
