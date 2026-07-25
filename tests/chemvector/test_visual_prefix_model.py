"""Behavioral contract for the image-conditioned visual-prefix decoder.

The implementation is expected at ``chemvector.visual_prefix_model`` with:

* ``VisualPrefixChemVectorModel(ChemVectorConfig)``;
* ``encode_visual_prefix(pixel_values) -> [batch, prefix, d_model]``;
* ``forward(pixel_values, input_ids, attention_mask=None)`` returning at least
  ``loss``, ``logits`` and ``visual_prefix``;
* ``decode_step(token_ids, position, visual_prefix, layer_caches=None)``.

The model module is intentionally imported directly so the contract remains a
hard failure while the implementation is incomplete.
"""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from chemvector.model import ChemVectorConfig
from chemvector.train_visual_svg import visual_only_teacher_forcing_ids
from chemvector.visual_prefix_model import (
    VisualPrefixChemVectorModel,
    VisualPrefixMemory,
)


class VisualPrefixChemVectorModelTests(unittest.TestCase):
    @staticmethod
    def _config() -> ChemVectorConfig:
        return ChemVectorConfig(
            vocab_size=31,
            pad_token_id=0,
            d_model=32,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=64,
            dropout=0.0,
            max_target_length=16,
            vision_dims=(8, 16),
            vision_depths=(1, 1),
        )

    @classmethod
    def _model(cls):
        torch.manual_seed(20260721)
        return VisualPrefixChemVectorModel(cls._config()).eval()

    @staticmethod
    def _images(batch_size: int = 2) -> torch.Tensor:
        image = torch.linspace(-1.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
        first = image.repeat(1, 3, 1, 1)
        second = -first
        images = torch.cat((first, second), dim=0)
        return images[:batch_size]

    def test_logits_change_materially_for_different_images(self) -> None:
        model = self._model()
        input_ids = torch.tensor(
            [[1, 4, 5, 6, 7, 2], [1, 4, 5, 6, 7, 2]],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = model(
                self._images(), input_ids, attention_mask
            )["logits"]

        mean_delta = float((logits[0] - logits[1]).abs().mean())
        self.assertGreater(
            mean_delta,
            1e-4,
            "the decoder is effectively image-invariant despite different visual prefixes",
        )

    def test_target_decoder_is_causal(self) -> None:
        model = self._model()
        image = self._images(batch_size=1)
        original = torch.tensor([[1, 4, 5, 6, 7, 2]], dtype=torch.long)
        changed_future = torch.tensor([[1, 4, 5, 13, 14, 2]], dtype=torch.long)

        with torch.no_grad():
            original_logits = model(image, original)["logits"]
            changed_logits = model(image, changed_future)["logits"]

        first_changed_input = 3
        torch.testing.assert_close(
            original_logits[:, :first_changed_input],
            changed_logits[:, :first_changed_input],
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertGreater(
            float(
                (
                    original_logits[:, first_changed_input:]
                    - changed_logits[:, first_changed_input:]
                )
                .abs()
                .max()
            ),
            1e-6,
            "the causal check is inconclusive because changed target tokens had no effect",
        )

    def test_visual_only_layout_blocks_every_historical_target_token(self) -> None:
        model = self._model()
        tokenizer = VisualSVGTokenizerV01()
        image = self._images(batch_size=1)
        input_ids = torch.tensor([[1, 4, 5, 6, 7, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        teacher_ids = visual_only_teacher_forcing_ids(
            input_ids,
            attention_mask,
            tokenizer,
        )

        with torch.no_grad():
            baseline = model(
                image,
                input_ids,
                attention_mask,
                teacher_forcing_ids=teacher_ids,
                attention_layout="visual-only",
            )["logits"]
            for changed_position in range(teacher_ids.size(1) - 2):
                changed = teacher_ids.clone()
                replacement = (int(changed[0, changed_position]) + 7) % model.config.vocab_size
                if replacement == model.config.pad_token_id:
                    replacement = 1
                changed[0, changed_position] = replacement
                changed_logits = model(
                    image,
                    input_ids,
                    attention_mask,
                    teacher_forcing_ids=changed,
                    attention_layout="visual-only",
                )["logits"]
                torch.testing.assert_close(
                    baseline[:, changed_position + 1 :],
                    changed_logits[:, changed_position + 1 :],
                    atol=1e-6,
                    rtol=1e-6,
                    msg=(
                        "a historical target token changed later logits under the "
                        f"visual-only layout at position {changed_position}"
                    ),
                )

    def test_visual_only_logits_change_for_different_images(self) -> None:
        model = self._model()
        tokenizer = VisualSVGTokenizerV01()
        input_ids = torch.tensor(
            [[1, 4, 5, 6, 7, 2], [1, 4, 5, 6, 7, 2]],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        teacher_ids = visual_only_teacher_forcing_ids(
            input_ids,
            attention_mask,
            tokenizer,
        )

        with torch.no_grad():
            logits = model(
                self._images(),
                input_ids,
                attention_mask,
                teacher_forcing_ids=teacher_ids,
                attention_layout="visual-only",
            )["logits"]

        mean_delta = float((logits[0] - logits[1]).abs().mean())
        self.assertGreater(
            mean_delta,
            1e-4,
            "visual-only logits are effectively invariant to the input image",
        )

    def test_white_image_has_exactly_zero_visual_values(self) -> None:
        model = self._model()
        white = torch.ones((1, 3, 32, 32), dtype=torch.float32)

        with torch.no_grad():
            memory = model.encode_images(white)

        self.assertIsInstance(memory, VisualPrefixMemory)
        torch.testing.assert_close(
            memory.values,
            torch.zeros_like(memory.values),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(int(torch.count_nonzero(memory.values)), 0)

    def test_causal_logits_depend_on_visual_memory_values(self) -> None:
        model = self._model()
        image = self._images(batch_size=1)
        input_ids = torch.tensor([[1, 4, 5, 6, 7, 2]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            memory = model.encode_images(image)
            real_value_logits = model(
                image,
                input_ids,
                attention_mask,
                memory=memory,
            )["logits"]
            zero_value_logits = model(
                image,
                input_ids,
                attention_mask,
                memory=VisualPrefixMemory(
                    prefix=memory.prefix,
                    values=torch.zeros_like(memory.values),
                ),
            )["logits"]

        mean_delta = float((real_value_logits - zero_value_logits).abs().mean())
        self.assertGreater(
            mean_delta,
            1e-4,
            "causal logits do not materially depend on VisualPrefixMemory.values",
        )

    def test_raw_prefix_memory_is_rejected(self) -> None:
        model = self._model()
        image = self._images(batch_size=1)
        input_ids = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)

        with torch.no_grad():
            raw_prefix = model.encode_visual_prefix(image)

        with self.assertRaisesRegex(TypeError, "VisualPrefixMemory"):
            model(image, input_ids, memory=raw_prefix)

    def test_loss_covers_target_tokens_only(self) -> None:
        model = self._model()
        input_ids = torch.tensor(
            [
                [1, 4, 5, 6, 2, 0, 0],
                [1, 8, 9, 2, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = (input_ids != 0).long()

        output = model(self._images(), input_ids, attention_mask)
        logits = output["logits"]
        labels = input_ids[:, 1:]
        valid_targets = attention_mask[:, 1:].bool()
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        expected_loss = token_losses[valid_targets].mean()

        self.assertEqual(logits.shape[:2], labels.shape)
        torch.testing.assert_close(output["loss"], expected_loss, atol=1e-6, rtol=1e-6)

    def test_forward_and_incremental_decode_shapes(self) -> None:
        model = self._model()
        config = model.config
        images = self._images()
        input_ids = torch.tensor(
            [[1, 4, 5, 6, 2], [1, 8, 9, 10, 2]],
            dtype=torch.long,
        )

        with torch.no_grad():
            visual_prefix = model.encode_visual_prefix(images)
            memory = model.encode_images(images)
            output = model(images, input_ids)
            first_logits, first_caches = model.decode_step(
                input_ids[:, 0:1],
                0,
                memory,
                None,
            )
            second_logits, second_caches = model.decode_step(
                input_ids[:, 1:2],
                1,
                memory,
                first_caches,
            )

        self.assertEqual(visual_prefix.ndim, 3)
        self.assertEqual(visual_prefix.shape[0], input_ids.shape[0])
        self.assertGreater(visual_prefix.shape[1], 0)
        self.assertEqual(visual_prefix.shape[2], config.d_model)
        self.assertEqual(output["visual_prefix"].shape, visual_prefix.shape)
        self.assertEqual(
            output["logits"].shape,
            (input_ids.size(0), input_ids.size(1) - 1, config.vocab_size),
        )
        self.assertEqual(first_logits.shape, (input_ids.size(0), 1, config.vocab_size))
        self.assertEqual(second_logits.shape, first_logits.shape)
        self.assertEqual(len(first_caches), config.decoder_layers)
        self.assertEqual(len(second_caches), config.decoder_layers)
        for first, second in zip(first_caches, second_caches):
            self.assertEqual(first.shape[0], input_ids.size(0))
            self.assertEqual(first.shape[-1], config.d_model)
            self.assertEqual(second.shape[0], input_ids.size(0))
            self.assertEqual(second.shape[-1], config.d_model)
            self.assertEqual(second.shape[1], first.shape[1] + 1)

    def test_incremental_logits_match_full_forward(self) -> None:
        model = self._model()
        images = self._images()
        input_ids = torch.tensor(
            [[1, 4, 5, 6, 7, 2], [1, 8, 9, 10, 11, 2]],
            dtype=torch.long,
        )

        with torch.no_grad():
            visual_prefix = model.encode_visual_prefix(images)
            memory = model.encode_images(images)
            full_logits = model(
                images,
                input_ids,
                memory=memory,
            )["logits"]
            caches = None
            step_logits = []
            for position in range(input_ids.size(1) - 1):
                logits, caches = model.decode_step(
                    input_ids[:, position : position + 1],
                    position,
                    memory,
                    caches,
                )
                step_logits.append(logits)
            incremental_logits = torch.cat(step_logits, dim=1)

        max_delta = float((incremental_logits - full_logits).abs().max())
        self.assertLessEqual(
            max_delta,
            1e-5,
            f"full and incremental decoding diverged by {max_delta:.8g}",
        )
        torch.testing.assert_close(
            incremental_logits,
            full_logits,
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
