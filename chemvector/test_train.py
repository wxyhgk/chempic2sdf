"""Tests for ChemVector training data mixing."""

from __future__ import annotations

import unittest

import torch

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.ocr import OCRTextCodec
from chemvector.synthetic_ocr import SyntheticOCRDataset, collate_synthetic_ocr
from chemvector.train import (
    iter_mixed_batches,
    ocr_sequence_accuracy,
    synthetic_ocr_replay_forward,
)


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
