import unittest

import torch
from PIL import Image

from Data.sdf.visual_svg_sequence_v0_1 import (
    LineRecordV01,
    ParsedVisualSVGSequenceV01,
    TextBoxRecordV01,
    serialize_parsed_visual_svg_v0_1,
)
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from chemvector.train_visual_svg import corrupt_teacher_forcing_ids
from chemvector.visual_svg_losses import (
    coordinate_expectation_metrics,
    coordinate_expectation_per_sample,
)
from chemvector.visual_svg_data import VisualSVGCollator


class VisualSVGCorruptionTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = VisualSVGTokenizerV01()
        self.sequence = serialize_parsed_visual_svg_v0_1(
            ParsedVisualSVGSequenceV01(
                (LineRecordV01(1, 2, 3, 4, 15),),
                (),
                (),
                (TextBoxRecordV01(10, 10, 20, 20),),
            )
        )
        ids = self.tokenizer.encode(self.sequence)
        self.input_ids = torch.tensor([ids], dtype=torch.long)
        self.mask = torch.ones_like(self.input_ids)

    def test_whole_record_corruption_hides_markers_and_fields(self):
        corrupted = corrupt_teacher_forcing_ids(
            self.input_ids,
            self.mask,
            self.tokenizer,
            token_probability=0.0,
            record_probability=1.0,
        )
        self.assertEqual(int(corrupted[0, 0]), self.tokenizer.bos_token_id)
        self.assertEqual(
            int(corrupted[0, 1]),
            self.tokenizer.token_to_id[self.tokenizer.DOCUMENT],
        )
        document_end = (
            self.input_ids[0]
            == self.tokenizer.token_to_id[self.tokenizer.DOCUMENT_END]
        ).nonzero()[0, 0]
        self.assertTrue(
            (corrupted[0, 2:document_end] == self.tokenizer.unk_token_id).all()
        )
        self.assertEqual(
            int(corrupted[0, document_end]),
            self.tokenizer.token_to_id[self.tokenizer.DOCUMENT_END],
        )

    def test_token_corruption_keeps_record_markers(self):
        corrupted = corrupt_teacher_forcing_ids(
            self.input_ids,
            self.mask,
            self.tokenizer,
            token_probability=1.0,
            record_probability=0.0,
        )
        for marker in (self.tokenizer.LINE, self.tokenizer.TEXT_BOX):
            token_id = self.tokenizer.token_to_id[marker]
            positions = (self.input_ids[0] == token_id).nonzero()
            self.assertGreater(len(positions), 0)
            for position in positions[:, 0]:
                self.assertEqual(int(corrupted[0, position]), token_id)


class VisualSVGCoordinateLossTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = VisualSVGTokenizerV01()

    def _peaked_logits(
        self,
        targets: torch.Tensor,
        predicted_coordinates: dict[tuple[int, int], int],
    ) -> torch.Tensor:
        logits = torch.full(
            (*targets.shape, self.tokenizer.vocab_size),
            -30.0,
        )
        for (batch_index, position), coordinate in predicted_coordinates.items():
            logits[
                batch_index,
                position,
                self.tokenizer.token_to_id[f"c:{coordinate}"],
            ] = 30.0
        return logits.requires_grad_()

    def test_far_coordinate_has_larger_huber_loss_than_near_coordinate(self):
        target_id = self.tokenizer.token_to_id["c:500"]
        targets = torch.tensor([[target_id], [target_id]])
        logits = self._peaked_logits(
            targets,
            {(0, 0): 501, (1, 0): 1500},
        )

        metrics = coordinate_expectation_per_sample(
            logits,
            targets,
            self.tokenizer,
        )

        self.assertGreater(
            float(metrics["loss"][1].detach()),
            float(metrics["loss"][0].detach()),
        )
        self.assertGreater(
            float(metrics["soft_mae_q"][1].detach()),
            float(metrics["soft_mae_q"][0].detach()),
        )
        torch.testing.assert_close(metrics["tokens"], torch.tensor([1, 1]))

    def test_only_coordinate_targets_contribute_and_batch_is_token_weighted(self):
        coordinate_100 = self.tokenizer.token_to_id["c:100"]
        coordinate_200 = self.tokenizer.token_to_id["c:200"]
        coordinate_300 = self.tokenizer.token_to_id["c:300"]
        line = self.tokenizer.token_to_id[self.tokenizer.LINE]
        width = self.tokenizer.token_to_id["w:10"]
        targets = torch.tensor(
            [
                [coordinate_100, line, width],
                [coordinate_200, coordinate_300, line],
            ]
        )
        logits = self._peaked_logits(
            targets,
            {(0, 0): 110, (1, 0): 220, (1, 1): 330},
        )

        per_sample = coordinate_expectation_per_sample(
            logits,
            targets,
            self.tokenizer,
        )
        batch = coordinate_expectation_metrics(logits, targets, self.tokenizer)

        torch.testing.assert_close(per_sample["tokens"], torch.tensor([1, 2]))
        expected_loss = (per_sample["loss"][0] + 2 * per_sample["loss"][1]) / 3
        expected_mae = (
            per_sample["soft_mae_q"][0] + 2 * per_sample["soft_mae_q"][1]
        ) / 3
        torch.testing.assert_close(batch["loss"], expected_loss)
        torch.testing.assert_close(batch["soft_mae_q"], expected_mae)
        self.assertEqual(float(batch["hard_exact"]), 0.0)
        self.assertEqual(float(batch["hard_valid_rate"]), 1.0)
        self.assertEqual(int(batch["tokens"]), 3)

    def test_full_vocabulary_argmax_reports_invalid_coordinate(self):
        coordinate = self.tokenizer.token_to_id["c:100"]
        targets = torch.tensor([[coordinate]])
        logits = self._peaked_logits(targets, {})
        with torch.no_grad():
            logits[0, 0, self.tokenizer.token_to_id[self.tokenizer.LINE]] = 40.0

        batch = coordinate_expectation_metrics(logits, targets, self.tokenizer)

        self.assertEqual(float(batch["hard_exact"]), 0.0)
        self.assertEqual(float(batch["hard_valid_rate"]), 0.0)
        self.assertEqual(int(batch["tokens"]), 1)

    def test_no_coordinate_targets_return_differentiable_zero(self):
        targets = torch.tensor(
            [[
                self.tokenizer.token_to_id[self.tokenizer.LINE],
                self.tokenizer.token_to_id["w:10"],
            ]]
        )
        logits = self._peaked_logits(targets, {})

        batch = coordinate_expectation_metrics(logits, targets, self.tokenizer)
        batch["loss"].backward()

        self.assertEqual(float(batch["loss"].detach()), 0.0)
        self.assertEqual(float(batch["soft_mae_q"].detach()), 0.0)
        self.assertEqual(float(batch["hard_exact"]), 0.0)
        self.assertEqual(float(batch["hard_valid_rate"]), 0.0)
        self.assertEqual(int(batch["tokens"]), 0)
        self.assertIsNotNone(logits.grad)

    def test_non_coordinate_target_positions_receive_no_auxiliary_gradient(self):
        coordinate = self.tokenizer.token_to_id["c:100"]
        line = self.tokenizer.token_to_id[self.tokenizer.LINE]
        targets = torch.tensor([[coordinate, line]])
        torch.manual_seed(20260721)
        logits = torch.randn(
            (1, 2, self.tokenizer.vocab_size),
            requires_grad=True,
        )

        coordinate_expectation_metrics(logits, targets, self.tokenizer)["loss"].backward()

        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0.0)


class VisualSVGCollatorTest(unittest.TestCase):
    def test_primitive_counts_use_fixed_protocol_order(self):
        tokenizer = VisualSVGTokenizerV01()
        sequence = serialize_parsed_visual_svg_v0_1(
            ParsedVisualSVGSequenceV01(
                (LineRecordV01(1, 2, 3, 4, 15),),
                (),
                (),
                (TextBoxRecordV01(10, 10, 20, 20),),
            )
        )
        sample = {
            "key": "count-order",
            "image": Image.new("RGB", (8, 8), "white"),
            "bond_layer": Image.new("L", (8, 8), 0),
            "target_sequence": sequence,
            "primitives": {},
            "meta": {
                "primitive_counts": {
                    "text_box": 44,
                    "quad": 33,
                    "line": 11,
                    "triangle": 22,
                }
            },
        }
        collator = VisualSVGCollator(
            tokenizer,
            image_size=8,
            max_length=64,
        )

        batch = collator([sample])

        self.assertEqual(
            VisualSVGCollator.PRIMITIVE_COUNT_ORDER,
            ("line", "triangle", "quad", "text_box"),
        )
        torch.testing.assert_close(
            batch["primitive_counts"],
            torch.tensor([[11.0, 22.0, 33.0, 44.0]]),
        )


if __name__ == "__main__":
    unittest.main()
