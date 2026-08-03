"""Tests for deterministic FP32 regional maxima."""

from __future__ import annotations

import unittest

import torch

from chemvector.components.vision.regional_maxima import regional_maxima_mask


class RegionalMaximaTests(unittest.TestCase):
    def test_marks_isolated_maxima_and_supports_leading_dimensions(self) -> None:
        values = torch.tensor(
            [
                [[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]],
                [[-1.0, -2.0, -1.0], [-2.0, -3.0, -2.0]],
            ],
            dtype=torch.float32,
        )

        maxima = regional_maxima_mask(values)

        expected = torch.tensor(
            [
                [[False, False, False], [False, False, True]],
                [[True, False, True], [False, False, False]],
            ]
        )
        torch.testing.assert_close(maxima, expected)

    def test_marks_an_entire_maximum_plateau(self) -> None:
        values = torch.tensor(
            [[0.0, 2.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

        maxima = regional_maxima_mask(values)

        expected = torch.tensor(
            [[False, True, True], [False, True, True], [False, False, False]]
        )
        torch.testing.assert_close(maxima, expected)

    def test_rejects_whole_plateau_when_one_boundary_pixel_is_higher(self) -> None:
        values = torch.tensor(
            [[2.0, 2.0, 3.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            dtype=torch.float32,
        )

        maxima = regional_maxima_mask(values)

        expected = torch.zeros_like(values, dtype=torch.bool)
        expected[0, 2] = True
        torch.testing.assert_close(maxima, expected)

    def test_connectivity_controls_diagonal_flat_zone_propagation(self) -> None:
        values = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=torch.float32,
        )

        four = regional_maxima_mask(values, connectivity=4)
        eight = regional_maxima_mask(values, connectivity=8)

        self.assertTrue(bool(four[0, 0]))
        self.assertFalse(bool(eight[0, 0]))
        self.assertTrue(bool(four[2, 1]))
        self.assertTrue(bool(eight[2, 1]))

    def test_valid_mask_excludes_values_and_invalid_output(self) -> None:
        values = torch.tensor(
            [[1.0, 9.0], [0.0, 0.0]], dtype=torch.float32
        )
        valid = torch.tensor([[True, False], [True, True]])

        maxima = regional_maxima_mask(values, valid_mask=valid)

        expected = torch.tensor([[True, False], [False, False]])
        torch.testing.assert_close(maxima, expected)

    def test_constant_valid_domain_is_one_regional_maximum(self) -> None:
        values = torch.ones(3, 4, dtype=torch.float32)

        maxima = regional_maxima_mask(values)

        self.assertTrue(bool(maxima.all()))
        self.assertEqual(maxima.dtype, torch.bool)
        self.assertEqual(maxima.device, values.device)

    def test_is_deterministic_and_does_not_modify_input(self) -> None:
        values = torch.tensor(
            [[0.0, 1.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float32
        )
        before = values.clone()

        first = regional_maxima_mask(values)
        second = regional_maxima_mask(values)

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(values, before)

    def test_rejects_invalid_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "FP32"):
            regional_maxima_mask(torch.ones(2, 2, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "spatial"):
            regional_maxima_mask(torch.ones(2, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            regional_maxima_mask(
                torch.tensor([[float("nan")]], dtype=torch.float32)
            )
        with self.assertRaisesRegex(ValueError, "4 or 8"):
            regional_maxima_mask(
                torch.ones(2, 2, dtype=torch.float32), connectivity=6
            )
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            regional_maxima_mask(
                torch.ones(2, 2, dtype=torch.float32),
                valid_mask=torch.ones(1, 2, dtype=torch.bool),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_result_matches_cpu(self) -> None:
        values = torch.tensor(
            [[0.0, 2.0, 2.0], [1.0, 2.0, 3.0]], dtype=torch.float32
        )

        cpu = regional_maxima_mask(values)
        cuda = regional_maxima_mask(values.cuda()).cpu()

        torch.testing.assert_close(cuda, cpu)


if __name__ == "__main__":
    unittest.main()
