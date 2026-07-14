"""Tests for byte-level OCR labels and the internal OCR head."""

from __future__ import annotations

import unittest

import torch

from chemvector.data import ChemVectorCollator
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.ocr import OCRTextCodec, atom_display_text, parse_atom_display_text
from chemvector.pretrain_ocr import set_ocr_trainable


class OCRTextCodecTests(unittest.TestCase):
    def test_roundtrip_accepts_ascii_and_unicode(self) -> None:
        codec = OCRTextCodec()
        for text in ("", "B", "Cl-", "R12", "Ph(OH)", "\u03b1"):
            self.assertEqual(codec.decode(codec.encode(text)), text)

    def test_label_is_not_limited_to_known_elements(self) -> None:
        codec = OCRTextCodec()
        self.assertLess(max(codec.encode("Arbitrary-123")), codec.vocab_size)

    def test_atom_display_text_hides_ordinary_carbon(self) -> None:
        self.assertEqual(
            atom_display_text(
                {
                    "element": "C",
                    "implicit_h_count": 3,
                    "explicit_h_count": 0,
                    "formal_charge": 0,
                    "isotope": 0,
                    "radical_electrons": 0,
                    "no_implicit": False,
                }
            ),
            "",
        )

    def test_atom_display_text_keeps_hydrogen_and_charge(self) -> None:
        self.assertEqual(
            atom_display_text(
                {
                    "element": "N",
                    "implicit_h_count": 2,
                    "explicit_h_count": 0,
                    "formal_charge": 1,
                }
            ),
            "NH2+",
        )

    def test_standard_atom_text_can_be_parsed_for_fusion(self) -> None:
        self.assertEqual(
            parse_atom_display_text("15NH2+."),
            {
                "element": "N",
                "formal_charge": 1,
                "isotope": 15,
                "explicit_h_count": 2,
                "radical_electrons": 1,
            },
        )
        self.assertIsNone(parse_atom_display_text("R1"))


class ChemOCRHeadTests(unittest.TestCase):
    @staticmethod
    def _config() -> ChemVectorConfig:
        return ChemVectorConfig(
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

    def test_forward_selects_only_real_anchors(self) -> None:
        config = self._config()
        model = ChemVectorModel(config)
        pixels = torch.randn(2, 3, 128, 128)
        input_ids = torch.tensor([[1, 3, 2], [1, 4, 2]])
        anchors = torch.tensor(
            [[[0.5, 0.5], [0.25, 0.25]], [[0.5, 0.5], [0.0, 0.0]]]
        )
        anchor_mask = torch.tensor([[True, True], [True, False]])
        ocr_ids = torch.zeros(2, 2, config.ocr_max_length, dtype=torch.long)
        ocr_ids[:, :, 0] = OCRTextCodec.BOS_TOKEN_ID
        ocr_ids[:, :, 1] = ord("A") + OCRTextCodec.BYTE_OFFSET
        ocr_ids[:, :, 2] = OCRTextCodec.EOS_TOKEN_ID

        output = model(
            pixels,
            input_ids,
            torch.ones_like(input_ids),
            ocr_anchor_points=anchors,
            ocr_anchor_mask=anchor_mask,
            ocr_token_ids=ocr_ids,
        )

        self.assertEqual(
            output["ocr_logits"].shape,
            (3, config.ocr_max_length - 1, config.ocr_vocab_size),
        )
        self.assertTrue(torch.isfinite(output["ocr_loss"]))
        output["ocr_loss"].backward()
        self.assertIsNotNone(model.ocr_head.roi_projection.weight.grad)

        with torch.no_grad():
            features = model.vision_encoder.forward_high_resolution(pixels)
            generated = model.ocr_head.generate(
                features,
                anchors,
                anchor_mask,
                pixels.shape[-2:],
            )
        self.assertEqual(generated.size(0), 3)
        self.assertLessEqual(generated.size(1), config.ocr_max_length - 1)

    def test_inference_can_override_the_training_roi_radius(self) -> None:
        model = ChemVectorModel(self._config())
        features = torch.randn(1, model.config.vision_dims[0], 16, 16)
        anchors = torch.tensor([[[0.5, 0.5]]])
        anchor_mask = torch.tensor([[True]])
        default = model.ocr_head.extract_rois(
            features,
            anchors,
            anchor_mask,
            (64, 64),
        )
        narrow = model.ocr_head.extract_rois(
            features,
            anchors,
            anchor_mask,
            (64, 64),
            roi_radius=4.0,
        )
        self.assertEqual(default.shape, narrow.shape)
        self.assertFalse(torch.equal(default, narrow))
        with self.assertRaisesRegex(ValueError, "radius"):
            model.ocr_head.extract_rois(
                features,
                anchors,
                anchor_mask,
                (64, 64),
                roi_radius=0,
            )

    def test_crop_pretraining_freezes_the_shared_encoder_by_default(self) -> None:
        model = ChemVectorModel(self._config())
        trainable = set_ocr_trainable(model)
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for parameter in model.ocr_head.parameters()))
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.vision_encoder.parameters())
        )


class OCRCollatorTests(unittest.TestCase):
    def test_render_atom_centers_become_text_anchors(self) -> None:
        codec = OCRTextCodec()
        collator = ChemVectorCollator(None, ocr_codec=codec)
        sample = {
            "render": {
                "width": 100,
                "height": 80,
                "atoms": [
                    {"atom_id": 0, "image_x": 25, "image_y": 40},
                    {"atom_id": 1, "image_x": 75, "image_y": 20},
                ],
            },
            "chemmol": {
                "atoms": [
                    {
                        "element": "C",
                        "implicit_h_count": 3,
                        "explicit_h_count": 0,
                        "formal_charge": 0,
                        "isotope": 0,
                        "radical_electrons": 0,
                        "no_implicit": False,
                    },
                    {
                        "element": "N",
                        "implicit_h_count": 1,
                        "explicit_h_count": 0,
                        "formal_charge": 0,
                    },
                ]
            },
        }

        batch = collator._ocr_batch([sample])

        self.assertTrue(torch.equal(batch["ocr_anchor_mask"], torch.tensor([[True, True]])))
        self.assertTrue(
            torch.allclose(
                batch["ocr_anchor_points"],
                torch.tensor([[[0.25, 0.5], [0.75, 0.25]]]),
            )
        )
        self.assertEqual(batch["ocr_texts"], [["", "NH"]])
        self.assertEqual(
            codec.decode(batch["ocr_token_ids"][0, 1].tolist()),
            "NH",
        )


if __name__ == "__main__":
    unittest.main()
