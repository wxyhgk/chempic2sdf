from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from formal.chemgraph.challenge_v0_5.enumerate_domain_v0_5 import decode_tuple_id
from formal.chemgraph.challenge_v0_6.e2.producer.produce_e2_witnesses_v0_6 import (
    E2ProducerError,
    PARENT_CONTRACT_SHA256,
    V0_6_CONTRACT_SHA256,
    canonical_bytes,
    produce,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
V0_6_CONTRACT = (
    REPOSITORY_ROOT
    / "formal/chemgraph/challenge_v0_6/prereg/p0_contract_v0_6.json"
)
PARENT_CONTRACT = (
    REPOSITORY_ROOT
    / "formal/chemgraph/challenge_v0_5/prereg/p0_contract_v0_5.json"
)


class E2ProducerTests(unittest.TestCase):
    def _produce(
        self, parent: Path, runtime_audit: dict | None = None
    ) -> tuple[dict, dict, Path]:
        output = parent / "delivery"
        manifest, validation = produce(
            output,
            V0_6_CONTRACT,
            V0_6_CONTRACT_SHA256,
            PARENT_CONTRACT,
            PARENT_CONTRACT_SHA256,
            runtime_audit,
        )
        return manifest, validation, output

    def test_all_eight_pairs_and_full_byte_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, validation, output = self._produce(Path(directory))
            self.assertEqual(manifest["pair_count"], 8)
            self.assertEqual(validation["tuple_membership_count"], 16)
            self.assertEqual(manifest["artifact_count"], 64)
            for pair in manifest["pairs"]:
                left, right = pair["left"], pair["right"]
                for side in (left, right):
                    decoded = decode_tuple_id(side["tuple"]["tuple_id"])
                    decoded.pop("schema_version")
                    self.assertEqual(decoded, side["tuple"])
                self.assertNotEqual(
                    left["graph"]["canonical_bytes_ascii"],
                    right["graph"]["canonical_bytes_ascii"],
                )
                for side in (left, right):
                    graph = side["graph"]
                    graph_bytes = graph["canonical_bytes_ascii"].encode("ascii")
                    parsed_graph = json.loads(graph_bytes)
                    self.assertEqual(set(parsed_graph), {"edges", "nodes"})
                    self.assertTrue(all(node[1] == 0 for node in parsed_graph["nodes"]))
                    self.assertEqual(graph["byte_length"], len(graph_bytes))
                    self.assertEqual(graph["sha256"], hashlib.sha256(graph_bytes).hexdigest())
                for field in (
                    "raw_gray", "padded_gray", "canonical_gray", "canonical_png"
                ):
                    for side in (left, right):
                        descriptor = side[field]
                        payload = (output / descriptor["path"]).read_bytes()
                        self.assertEqual(len(payload), descriptor["byte_length"])
                        self.assertEqual(hashlib.sha256(payload).hexdigest(), descriptor["sha256"])
                self.assertNotEqual(
                    (output / left["raw_gray"]["path"]).read_bytes(),
                    (output / right["raw_gray"]["path"]).read_bytes(),
                )
                for field in ("canonical_gray", "canonical_png"):
                    self.assertEqual(
                        (output / left[field]["path"]).read_bytes(),
                        (output / right[field]["path"]).read_bytes(),
                    )

    def test_manifest_and_validation_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, validation, output = self._produce(Path(directory))
            self.assertEqual((output / "manifest.json").read_bytes(), canonical_bytes(manifest))
            self.assertEqual(
                (output / "validation.json").read_bytes(), canonical_bytes(validation)
            )
            self.assertEqual(validation["status"], "passed")
            self.assertFalse(validation["training_authorized"])

    def test_four_distinct_workers_use_frozen_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_audit: dict = {}
            _, validation, _ = self._produce(Path(directory), runtime_audit)
            self.assertEqual(validation["worker_count"], 4)
            self.assertEqual(validation["worker_process_distinct_count"], 4)
            self.assertEqual(len(set(runtime_audit["worker_pids"])), 4)
            self.assertNotIn(os.getpid(), runtime_audit["worker_pids"])
            self.assertEqual(
                runtime_audit["worker_assignments"],
                {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]},
            )
            self.assertEqual(
                validation["worker_assignments"],
                [
                    {"pair_ordinals": [0, 4], "worker_id": 0},
                    {"pair_ordinals": [1, 5], "worker_id": 1},
                    {"pair_ordinals": [2, 6], "worker_id": 2},
                    {"pair_ordinals": [3, 7], "worker_id": 3},
                ],
            )

    def test_replay_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_manifest, first_validation, _ = self._produce(root / "first")
            second_manifest, second_validation, _ = self._produce(root / "second")
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_validation, second_validation)
            self.assertEqual(
                first_manifest["artifact_tree_sha256"],
                second_manifest["artifact_tree_sha256"],
            )

    def test_wrong_contract_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                E2ProducerError, "expected_v0_6_contract_sha256_not_frozen"
            ):
                produce(
                    Path(directory) / "delivery",
                    V0_6_CONTRACT,
                    "0" * 64,
                    PARENT_CONTRACT,
                    PARENT_CONTRACT_SHA256,
                )


if __name__ == "__main__":
    unittest.main()
