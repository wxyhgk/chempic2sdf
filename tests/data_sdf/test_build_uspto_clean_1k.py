"""Tests for byte-preserving USPTO clean benchmark preparation."""

from __future__ import annotations

import unittest

from Data.benchmarks.public_ocsr_v1.build_uspto_clean_1k import sdf_bytes


class UsptoCleanPreparationTests(unittest.TestCase):
    def test_sdf_wrapper_does_not_rewrite_molfile(self) -> None:
        molfile = "title\r\nprogram\r\n\r\nM  END\r\n"

        payload = sdf_bytes(molfile)

        self.assertEqual(payload, b"title\r\nprogram\r\n\r\nM  END\n$$$$\n")


if __name__ == "__main__":
    unittest.main()
