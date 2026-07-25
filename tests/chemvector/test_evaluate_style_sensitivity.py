from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from chemvector.evaluate_style_sensitivity import parse_args


class EvaluateStyleSensitivityTest(unittest.TestCase):
    def test_cli_defaults_to_unlocked_validation(self) -> None:
        argv = ["evaluate", "--checkpoint", "verifier.pt", "--output", "metrics.json"]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.split, "validation")
        self.assertEqual(args.samples, 0)
        self.assertIn("validation-000000.tar", args.shard)
        self.assertFalse(hasattr(args, "minimum_margin"))


if __name__ == "__main__":
    unittest.main()
