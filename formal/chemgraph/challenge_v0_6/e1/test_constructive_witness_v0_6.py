from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from .produce_constructive_witness_v0_6 import (
        ConstructiveWitnessError,
        build_witness,
        canonical_bytes,
    )
    from .verify_constructive_witness_v0_6 import verify
except ImportError:
    from produce_constructive_witness_v0_6 import (  # type: ignore
        ConstructiveWitnessError,
        build_witness,
        canonical_bytes,
    )
    from verify_constructive_witness_v0_6 import verify  # type: ignore


CONTRACT_SHA256 = "0" * 64


class ConstructiveWitnessTests(unittest.TestCase):
    def test_witness_is_same_tuple_and_different_graph(self) -> None:
        value = build_witness(CONTRACT_SHA256)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(len(value["interpretations"]), 2)
        self.assertNotEqual(
            value["interpretations"][0]["graph_signature"],
            value["interpretations"][1]["graph_signature"],
        )
        for interpretation in value["interpretations"]:
            signature = interpretation["graph_signature"]
            graph = json.loads(signature)
            self.assertEqual(set(graph), {"edges", "nodes"})
            self.assertFalse(signature.endswith("\n"))
        self.assertTrue(value["checks"]["same_tuple_full_gray_bytes_equal"])

    def test_canonical_roundtrip_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.json"
            path.write_bytes(canonical_bytes(build_witness(CONTRACT_SHA256)))
            self.assertEqual(verify(path, CONTRACT_SHA256)["status"], "passed")

    def test_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.json"
            value = build_witness(CONTRACT_SHA256)
            value["status"] = "failed"
            path.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(
                ConstructiveWitnessError, "witness_recomputation_mismatch"
            ):
                verify(path, CONTRACT_SHA256)

    def test_noncanonical_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.json"
            value = build_witness(CONTRACT_SHA256)
            path.write_text(json.dumps(value, indent=2), encoding="ascii")
            with self.assertRaisesRegex(
                ConstructiveWitnessError, "witness_is_not_canonical_json"
            ):
                verify(path, CONTRACT_SHA256)


if __name__ == "__main__":
    unittest.main()
