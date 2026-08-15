from __future__ import annotations

import ast
import copy
import json
import unittest
from pathlib import Path

from .verify_confirmation import (
    ConfirmationError,
    PNG_PATH,
    ROOT,
    WITNESS_PATH,
    canonical_bytes,
    verify_frozen_confirmation,
    verify_payload,
)


class IndependentConfirmationTests(unittest.TestCase):
    def test_verifier_does_not_import_exploratory_constructor(self) -> None:
        source_path = ROOT / "verify_confirmation.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(
            "png_ownership_realization_v0_5.ownership_realization" in name
            or "png_ownership_realization_v0_5.build_results" in name
            for name in imported
        ))

    def test_frozen_confirmation_replays_twice_byte_exact(self) -> None:
        first = canonical_bytes(verify_frozen_confirmation())
        second = canonical_bytes(verify_frozen_confirmation())
        self.assertEqual(first, second)
        self.assertEqual(ROOT.joinpath("confirmation.json").read_bytes(), first)

    def test_structural_mutations_are_rejected_without_identity_gate(self) -> None:
        png = PNG_PATH.read_bytes()
        witness = json.loads(WITNESS_PATH.read_bytes())
        mutations = []

        missing_pixel = copy.deepcopy(witness)
        missing_pixel["configuration"]["ownership_sets"]["ownership_universe_runs"][0][2] -= 1
        mutations.append(missing_pixel)

        carrier_rle = copy.deepcopy(witness)
        carrier_rle["configuration"]["carrier_supports"][0]["support_runs"][0][1] += 1
        mutations.append(carrier_rle)

        disconnected = copy.deepcopy(witness)
        disconnected["configuration"]["partition_annotations"][0]["right_region_runs"].pop()
        mutations.append(disconnected)

        double_owned = copy.deepcopy(witness)
        double_owned["configuration"]["partition_annotations"][0]["left_region_runs"][0][2] += 1
        mutations.append(double_owned)

        changed_seed = copy.deepcopy(witness)
        changed_seed["configuration"]["seed_pixels"] = [[20, 19], [20, 22]]
        mutations.append(changed_seed)

        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                with self.assertRaises(ConfirmationError):
                    verify_payload(png, mutation, enforce_frozen_identity=False)

    def test_png_mutation_is_rejected_without_identity_gate(self) -> None:
        png = bytearray(PNG_PATH.read_bytes())
        png[-1] ^= 1
        witness = json.loads(WITNESS_PATH.read_bytes())
        with self.assertRaises(ConfirmationError):
            verify_payload(bytes(png), witness, enforce_frozen_identity=False)


if __name__ == "__main__":
    unittest.main()
