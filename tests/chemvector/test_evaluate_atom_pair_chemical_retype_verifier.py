from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from chemvector.evaluate_atom_pair_chemical_retype_verifier import parse_args


class EvaluateAtomPairChemicalRetypeVerifierTest(unittest.TestCase):
    def test_cli_exposes_split_but_no_threshold_override(self) -> None:
        argv = [
            "evaluate",
            "--checkpoint",
            "verifier.pt",
            "--split",
            "validation",
            "--output",
            "metrics.json",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.split, "validation")
        self.assertFalse(hasattr(args, "minimum_margin"))


if __name__ == "__main__":
    unittest.main()
