"""Tests for byte-level OCR labels and the internal OCR head."""

from __future__ import annotations

from dataclasses import asdict
import unittest

from PIL import Image, ImageDraw
import torch

from chemvector.data import ChemVectorCollator
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.layer_ocr import (
    ChemVisionLayerOCRConfig,
    ROLE_TO_ID,
    assign_predicted_points,
    build_layer_ocr_sample,
    labels_for_predicted_points,
    collate_layer_ocr,
)
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

    def test_all_periodic_table_elements_are_supported(self) -> None:
        self.assertEqual(parse_atom_display_text("Fe")["element"], "Fe")
        self.assertEqual(parse_atom_display_text("Og")["element"], "Og")
        self.assertEqual(parse_atom_display_text("23Na+")["formal_charge"], 1)
        self.assertIsNone(parse_atom_display_text("Ph"))


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
                use_auxiliary_routing=True,
            )
        self.assertEqual(generated.size(0), 3)
        self.assertLessEqual(generated.size(1), config.ocr_max_length - 1)

    def test_pixel_ocr_uses_raw_crops_and_common_label_routing(self) -> None:
        base = self._config()
        config = ChemVectorConfig(
            **{
                **asdict(base),
                "ocr_pixel_enabled": True,
                "ocr_pixel_crop_size": 32,
            }
        )
        model = ChemVectorModel(config)
        pixels = torch.randn(1, 3, 128, 128)
        input_ids = torch.tensor([[1, 3, 2]])
        anchors = torch.tensor([[[0.5, 0.5]]])
        anchor_mask = torch.tensor([[True]])
        codec = OCRTextCodec(max_length=config.ocr_max_length)
        encoded = codec.encode("B")
        ocr_ids = torch.full(
            (1, 1, config.ocr_max_length),
            codec.PAD_TOKEN_ID,
            dtype=torch.long,
        )
        ocr_ids[0, 0, : len(encoded)] = torch.tensor(encoded)

        output = model(
            pixels,
            input_ids,
            torch.ones_like(input_ids),
            ocr_anchor_points=anchors,
            ocr_anchor_mask=anchor_mask,
            ocr_token_ids=ocr_ids,
        )
        self.assertTrue(torch.isfinite(output["ocr_loss"]))
        self.assertIn("ocr_blank_loss", output)
        self.assertIn("ocr_common_loss", output)
        output["ocr_loss"].backward()
        self.assertIsNotNone(model.ocr_head.pixel_encoder[0].weight.grad)

        with torch.no_grad():
            model.ocr_head.blank_classifier.weight.zero_()
            model.ocr_head.blank_classifier.bias.copy_(torch.tensor((10.0, -10.0)))
            model.ocr_head.common_classifier.weight.zero_()
            model.ocr_head.common_classifier.bias.fill_(-10.0)
            model.ocr_head.common_classifier.bias[0] = 10.0
            generated = model.ocr_head.generate(
                pixels,
                anchors,
                anchor_mask,
                pixels.shape[-2:],
                use_auxiliary_routing=True,
            )
        self.assertEqual(codec.decode(generated[0].tolist()), "B")

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

    def test_pixel_multiscale_reuses_the_shared_encoder(self) -> None:
        base = self._config()
        config = ChemVectorConfig(
            **{
                **asdict(base),
                "ocr_pixel_enabled": True,
                "ocr_pixel_multiscale": True,
                "ocr_pixel_scale_radii": (4.0, 8.0, 12.0),
            }
        )
        model = ChemVectorModel(config)
        pixels = torch.randn(1, 3, 64, 64)
        anchors = torch.tensor([[[0.5, 0.5]]])
        anchor_mask = torch.tensor([[True]])

        memory = model.ocr_head.prepare_memory(
            pixels,
            anchors,
            anchor_mask,
            pixels.shape[-2:],
            roi_radius=2.0,
        )

        self.assertEqual(memory.shape, (1, 64, config.ocr_d_model))
        memory.sum().backward()
        self.assertIsNotNone(model.ocr_head.pixel_encoder[0].weight.grad)
        self.assertIsNotNone(model.ocr_head.pixel_scale_score[-1].weight.grad)

    def test_crop_pretraining_freezes_the_shared_encoder_by_default(self) -> None:
        model = ChemVectorModel(self._config())
        trainable = set_ocr_trainable(model)
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for parameter in model.ocr_head.parameters()))
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.vision_encoder.parameters())
        )

    def test_zero_initialized_adapter_preserves_memory(self) -> None:
        base = self._config()
        config = ChemVectorConfig(
            **{
                **asdict(base),
                "ocr_pixel_enabled": True,
                "ocr_residual_adapter": True,
                "ocr_role_head": True,
            }
        )
        model = ChemVectorModel(config)
        memory = torch.randn(3, 64, config.ocr_d_model)

        self.assertTrue(torch.equal(model.ocr_head.adapt_memory(memory), memory))
        self.assertEqual(
            model.ocr_head.role_logits(memory).shape,
            (3, 3),
        )

    def test_adapter_only_training_does_not_open_base_ocr(self) -> None:
        base = self._config()
        config = ChemVectorConfig(
            **{
                **asdict(base),
                "ocr_residual_adapter": True,
                "ocr_role_head": True,
            }
        )
        model = ChemVectorModel(config)

        trainable = set_ocr_trainable(model, adapter_only=True)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }

        self.assertEqual(len(trainable), len(trainable_names))
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                "memory_adapter" in name
                or "role_attention" in name
                or "role_classifier" in name
                for name in trainable_names
            )
        )
        self.assertFalse(model.ocr_head.token_embedding.weight.requires_grad)


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

    def test_layer_dataset_prioritizes_rare_visible_labels(self) -> None:
        atoms = [
            {
                "atom_id": 0,
                "element": "B",
                "center": [10.0, 10.0],
                "bbox": [8.0, 8.0, 12.0, 12.0],
                "visible_text": True,
                "text": "B",
            },
            *[
                {
                    "atom_id": index,
                    "element": "N",
                    "center": [float(index * 10), 20.0],
                    "bbox": [float(index * 10 - 2), 18.0, float(index * 10 + 2), 22.0],
                    "visible_text": True,
                    "text": "N",
                }
                for index in range(1, 8)
            ],
            *[
                {
                    "atom_id": index,
                    "element": "C",
                    "center": [float(index * 10), 30.0],
                    "visible_text": False,
                    "text": "",
                }
                for index in range(8, 16)
            ],
        ]
        sample = {
            "key": "sample",
            "image": torch.zeros(1),
            "layers": {
                "width": 100,
                "height": 100,
                "atoms": atoms,
                "render_parameters": {
                    "font_size": 24,
                    "font_family_id": "font-a",
                },
            },
        }

        result = build_layer_ocr_sample(
            sample,
            seed=1,
            config=ChemVisionLayerOCRConfig(
                maximum_blank_anchors=2,
                maximum_visible_anchors=3,
                maximum_common_visible_anchors=2,
                visible_jitter_copies=1,
                n_jitter_probability=1.0,
            ),
        )

        self.assertEqual(result["texts"].count("B"), 2)
        self.assertEqual(result["texts"].count(""), 2)
        self.assertEqual(len(result["texts"]), 8)
        self.assertTrue(all(category.startswith("large_") for category in result["categories"]))
        self.assertEqual(result["categories"].count("large_visible_jitter"), 3)
        self.assertEqual(result["roles"].count("implicit_c"), 2)
        self.assertEqual(result["roles"].count("atom_text"), 6)

    def test_layer_dataset_can_prioritize_modifier_strings(self) -> None:
        atoms = [
            {
                "atom_id": 0,
                "element": "N",
                "center": [20.0, 20.0],
                "bbox": [16.0, 16.0, 24.0, 24.0],
                "visible_text": True,
                "text": "N",
            },
            {
                "atom_id": 1,
                "element": "N",
                "center": [60.0, 20.0],
                "bbox": [54.0, 16.0, 66.0, 24.0],
                "visible_text": True,
                "text": "NH",
            },
        ]
        sample = {
            "key": "modifier",
            "image": torch.zeros(1),
            "layers": {
                "width": 100,
                "height": 100,
                "atoms": atoms,
                "render_parameters": {"font_size": 16, "font_family_id": "font-a"},
            },
        }
        result = build_layer_ocr_sample(
            sample,
            seed=1,
            config=ChemVisionLayerOCRConfig(
                maximum_blank_anchors=0,
                maximum_visible_anchors=1,
                maximum_common_visible_anchors=0,
                priority_elements=(),
                priority_texts=("NH",),
            ),
        )
        self.assertEqual(result["texts"], ["NH"])

    def test_layer_dataset_can_colorize_visible_nitrogen(self) -> None:
        image = Image.new("RGB", (40, 40), "white")
        ImageDraw.Draw(image).rectangle((15, 15, 24, 24), fill="black")
        sample = {
            "key": "colored-n",
            "image": image,
            "layers": {
                "width": 40,
                "height": 40,
                "atoms": [
                    {
                        "atom_id": 0,
                        "element": "N",
                        "center": [20.0, 20.0],
                        "bbox": [15.0, 15.0, 24.0, 24.0],
                        "visible_text": True,
                        "text": "N",
                    }
                ],
                "render_parameters": {"font_size": 16, "font_family_id": "font-a"},
            },
        }
        result = build_layer_ocr_sample(
            sample,
            seed=1,
            config=ChemVisionLayerOCRConfig(color_probability=1.0),
        )
        red, green, blue = result["image"].getpixel((20, 20))
        self.assertGreater(blue, red)
        self.assertIn("_color", result["categories"][0])

    def test_predicted_peaks_inside_one_glyph_share_its_label(self) -> None:
        atoms = [
            {
                "atom_id": 0,
                "center": [50.0, 50.0],
                "bbox": [40.0, 35.0, 60.0, 65.0],
                "visible_text": True,
                "text": "N",
            },
            {
                "atom_id": 1,
                "center": [90.0, 50.0],
                "bbox": None,
                "visible_text": False,
                "text": "",
            },
        ]

        labels = labels_for_predicted_points(
            [(42.0, 50.0), (50.0, 50.0), (58.0, 50.0), (90.0, 50.0), (5.0, 5.0)],
            atoms,
        )

        self.assertEqual(labels, ["N", "N", "N", "", ""])

    def test_predicted_role_matching_keeps_only_one_peak_per_atom(self) -> None:
        atoms = [
            {
                "atom_id": 0,
                "center": [50.0, 50.0],
                "bbox": [40.0, 35.0, 60.0, 65.0],
                "visible_text": True,
                "text": "N",
            },
            {
                "atom_id": 1,
                "center": [90.0, 50.0],
                "bbox": None,
                "visible_text": False,
                "text": "",
            },
        ]

        labels, roles = assign_predicted_points(
            [(42.0, 50.0), (50.0, 50.0), (58.0, 50.0), (90.0, 50.0), (5.0, 5.0)],
            atoms,
        )

        self.assertEqual(labels, ["", "N", "", "", ""])
        self.assertEqual(
            roles,
            [
                "non_molecule_text",
                "atom_text",
                "non_molecule_text",
                "implicit_c",
                "non_molecule_text",
            ],
        )

    def test_layer_collator_emits_padded_role_targets(self) -> None:
        codec = OCRTextCodec()
        samples = [
            {
                "key": "a",
                "image": torch.zeros(8, 8, 3).numpy().astype("uint8"),
                "anchors": [(0.5, 0.5), (0.25, 0.25)],
                "texts": ["N", ""],
                "categories": ["visible", "blank"],
                "roles": ["atom_text", "implicit_c"],
                "layer_atoms": [],
                "layer_width": 8,
                "layer_height": 8,
            },
            {
                "key": "b",
                "image": torch.zeros(8, 8, 3).numpy().astype("uint8"),
                "anchors": [(0.5, 0.5)],
                "texts": [""],
                "categories": ["blank"],
                "roles": ["non_molecule_text"],
                "layer_atoms": [],
                "layer_width": 8,
                "layer_height": 8,
            },
        ]

        batch = collate_layer_ocr(samples, codec)

        self.assertTrue(
            torch.equal(
                batch["ocr_role_targets"],
                torch.tensor(
                    [
                        [ROLE_TO_ID["atom_text"], ROLE_TO_ID["implicit_c"]],
                        [ROLE_TO_ID["non_molecule_text"], -1],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
