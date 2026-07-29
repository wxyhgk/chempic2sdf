import math
import unittest

import torch

from chemvector.ctc_ocr import (
    ChemicalCTCConfig,
    ChemicalCTCOCR,
    ConditionedPixelCTCConfig,
    ConditionedPixelCTCOCR,
    PretrainedPixelCTCConfig,
    PretrainedPixelCTCOCR,
    detected_text_crops,
    ctc_loss,
    ctc_text_log_score,
    decode_ctc_prefix_beam,
    decode_ctc_with_score,
    decode_ctc,
    encode_text,
)


class ChemicalCTCOCRTest(unittest.TestCase):
    def test_byte_roundtrip_and_repeat_collapse(self) -> None:
        encoded = encode_text("NH2+")
        stream = torch.tensor([encoded[0], encoded[0], 0, *encoded[1:].tolist()])
        self.assertEqual(decode_ctc(stream), "NH2+")

    def test_ctc_sequence_score_prefers_confident_path(self) -> None:
        logits = torch.full((4, 257), -5.0)
        logits[:, 0] = 2.0
        logits[1, ord("N") + 1] = 8.0
        text, score = decode_ctc_with_score(logits)
        self.assertEqual(text, "N")
        self.assertLess(score, 0.0)

    def test_prefix_beam_sums_ctc_alignments(self) -> None:
        logits = torch.full((2, 257), -1000.0)
        for row in logits:
            row[0] = math.log(0.40)
            row[ord("A") + 1] = math.log(0.35)
            row[ord("B") + 1] = math.log(0.25)

        self.assertEqual(decode_ctc(logits.argmax(-1)), "")
        candidates = decode_ctc_prefix_beam(logits, beam_size=8, top_paths=3)
        self.assertEqual(candidates[0].text, "A")
        self.assertAlmostEqual(candidates[0].log_score, math.log(0.4025), places=5)
        self.assertGreater(candidates[0].log_score, candidates[1].log_score)

    def test_prefix_beam_keeps_blank_separated_repeated_bytes(self) -> None:
        logits = torch.full((3, 257), -1000.0)
        logits[0, ord("A") + 1] = 0.0
        logits[1, 0] = 0.0
        logits[2, ord("A") + 1] = 0.0
        candidates = decode_ctc_prefix_beam(logits, beam_size=4, top_paths=1)
        self.assertEqual(candidates[0].text, "AA")

    def test_prefix_beam_decodes_utf8_bytes(self) -> None:
        encoded = encode_text("\u03b2")
        logits = torch.full((len(encoded), 257), -1000.0)
        for step, token_id in enumerate(encoded.tolist()):
            logits[step, token_id] = 0.0
        candidates = decode_ctc_prefix_beam(logits, beam_size=4, top_paths=1)
        self.assertEqual(candidates[0].text, "\u03b2")
        self.assertIsInstance(candidates[0].log_score, float)

    def test_prefix_beam_deduplicates_decoded_text(self) -> None:
        logits = torch.full((1, 257), -1000.0)
        logits[0, 0] = math.log(0.10)
        logits[0, 0x80 + 1] = math.log(0.45)
        logits[0, 0x81 + 1] = math.log(0.45)
        candidates = decode_ctc_prefix_beam(logits, beam_size=3)
        replacement = [item for item in candidates if item.text == "\ufffd"]
        self.assertEqual(len(replacement), 1)
        self.assertAlmostEqual(replacement[0].log_score, math.log(0.90), places=5)

    def test_prefix_beam_validates_search_limits(self) -> None:
        logits = torch.zeros((1, 257))
        with self.assertRaisesRegex(ValueError, "beam_size"):
            decode_ctc_prefix_beam(logits, beam_size=0)
        with self.assertRaisesRegex(ValueError, "top_paths"):
            decode_ctc_prefix_beam(logits, beam_size=2, top_paths=3)
        with self.assertRaisesRegex(ValueError, "maximum_bytes"):
            decode_ctc_prefix_beam(logits, maximum_bytes=0)
        with self.assertRaisesRegex(ValueError, "token_beam"):
            decode_ctc_prefix_beam(logits, token_beam=0)

    def test_text_log_score_matches_greedy_normalization(self) -> None:
        logits = torch.full((4, 257), -5.0)
        logits[:, 0] = 2.0
        logits[1, ord("N") + 1] = 8.0
        text, greedy_score = decode_ctc_with_score(logits)
        self.assertEqual(text, "N")
        self.assertAlmostEqual(
            greedy_score,
            ctc_text_log_score(logits, text, normalize_by_bytes=True),
        )

    def test_forward_and_loss(self) -> None:
        model = ChemicalCTCOCR(
            ChemicalCTCConfig(channels=32, hidden_dim=32, layers=1, heads=4)
        )
        logits = model(torch.zeros((2, 3, 48, 192)))
        targets = torch.cat((encode_text("N"), encode_text("OH")))
        loss = ctc_loss(logits, targets, torch.tensor((1, 2)))
        self.assertEqual(logits.shape[:2], (2, 48))
        self.assertTrue(torch.isfinite(loss))

    def test_pretrained_pixel_ctc_requires_loaded_encoder(self) -> None:
        model = PretrainedPixelCTCOCR(
            PretrainedPixelCTCConfig(d_model=32, layers=1, heads=4)
        )
        with self.assertRaisesRegex(RuntimeError, "has not been loaded"):
            model(torch.zeros((1, 3, 48, 192)))

    def test_pretrained_pixel_ctc_preserves_horizontal_steps(self) -> None:
        model = PretrainedPixelCTCOCR(
            PretrainedPixelCTCConfig(d_model=32, layers=1, heads=4)
        )
        model.pixel_encoder_loaded = True
        logits = model(torch.zeros((2, 3, 48, 192)))
        self.assertEqual(logits.shape, (2, 48, 257))
        self.assertTrue(all(not item.requires_grad for item in model.pixel_encoder.parameters()))

    def test_pretrained_pixel_ctc_can_be_explicitly_unfrozen(self) -> None:
        model = PretrainedPixelCTCOCR(
            PretrainedPixelCTCConfig(d_model=32, layers=1, heads=4)
        )
        model.unfreeze_pixel_encoder()
        self.assertFalse(model.pixel_encoder_frozen)
        self.assertTrue(all(item.requires_grad for item in model.pixel_encoder.parameters()))

    def test_conditioned_ctc_accepts_mask_and_anchor_channels(self) -> None:
        model = ConditionedPixelCTCOCR(
            ConditionedPixelCTCConfig(
                condition_channels=2,
                d_model=32,
                layers=1,
                heads=4,
                height_attention=True,
            )
        )
        logits = model(torch.zeros((2, 5, 48, 192)))
        self.assertEqual(logits.shape, (2, 48, 257))
        self.assertTrue(all(not item.requires_grad for item in model.pixel_encoder.parameters()))

    def test_detected_text_crops_use_predicted_extents(self) -> None:
        pixels = torch.ones((3, 100, 100))
        pixels[:, 45:55, 45:55] = -1
        crops, indices = detected_text_crops(
            pixels,
            torch.tensor(((0.5, 0.5), (0.8, 0.8))),
            torch.tensor((1, 0)),
            [{"x": 0.5, "y": 0.5, "half_width": 0.05, "half_height": 0.05}],
            text_role=1,
        )
        self.assertEqual(indices, [0])
        self.assertEqual(crops.shape, (1, 3, 48, 192))
        self.assertLess(float(crops.mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
