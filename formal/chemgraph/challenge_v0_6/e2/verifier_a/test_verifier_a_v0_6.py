from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from formal.chemgraph.challenge_v0_6.e2.verifier_a.independent_e2_core_v0_6 import (
    IndependentVerificationError,
    canonical_json_bytes,
)
from formal.chemgraph.challenge_v0_6.e2.verifier_a.verify_e2_manifest_v0_6 import (
    _artifact_tree_hash,
    _expected_manifest,
    _run_fixed_workers,
    verify_manifest,
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


class VerifierATests(unittest.TestCase):
    def _fixture(self, parent: Path) -> tuple[Path, dict[str, object]]:
        delivery = parent / "delivery"
        manifest, artifacts = _expected_manifest()
        for relative, payload in artifacts.items():
            path = delivery / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        manifest_path = delivery / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        return manifest_path, manifest

    def _verify(self, manifest: Path) -> dict[str, object]:
        return verify_manifest(manifest, V0_6_CONTRACT, PARENT_CONTRACT)

    def test_valid_independent_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._fixture(Path(directory))
            result = self._verify(manifest)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["pair_count"], 8)
            self.assertEqual(result["full_byte_sidecar_count"], 64)
            self.assertEqual(result["translation_preprocess_replay_count"], 80)
            self.assertEqual(result["workers"], 4)
            self.assertEqual(result["worker_start_method"], "spawn")
            worker_results = result["worker_results"]
            self.assertEqual(
                [item["pair_indices"] for item in worker_results],
                [[0, 1], [2, 3], [4, 5], [6, 7]],
            )
            self.assertEqual(
                [item["recomputed_pair_count"] for item in worker_results],
                [2, 2, 2, 2],
            )
            self.assertEqual(result["worker_process_count"], 4)
            self.assertTrue(all("pid" not in item for item in worker_results))
            self.assertFalse(result["training_authorized"])

    def test_fixed_scheduler_uses_four_distinct_child_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._fixture(Path(directory))
            worker_results = _run_fixed_workers(manifest)
            pids = {item["pid"] for item in worker_results}
            self.assertEqual(len(pids), 4)
            self.assertNotIn(os.getpid(), pids)
            self.assertEqual(
                [item["pair_indices"] for item in worker_results],
                [[0, 1], [2, 3], [4, 5], [6, 7]],
            )

    def test_tuple_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, manifest = self._fixture(Path(directory))
            manifest["pairs"][0]["left"]["tuple"]["tuple_id"] += 1  # type: ignore[index]
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                IndependentVerificationError, "worker_pair_recomputation_mismatch"
            ):
                self._verify(manifest_path)

    def test_sidecar_tampering_is_rejected_after_digest_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, manifest = self._fixture(Path(directory))
            delivery = manifest_path.parent
            descriptor = manifest["pairs"][0]["left"]["canonical_gray"]  # type: ignore[index]
            relative = descriptor["path"]
            path = delivery / relative
            changed = bytearray(path.read_bytes())
            changed[len(changed) // 2] ^= 0xFF
            path.write_bytes(changed)
            descriptor["sha256"] = hashlib.sha256(changed).hexdigest()
            payloads = {
                item.relative_to(delivery).as_posix(): item.read_bytes()
                for item in (delivery / "artifacts").rglob("*") if item.is_file()
            }
            manifest["artifact_tree_sha256"] = _artifact_tree_hash(payloads)
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                IndependentVerificationError, "artifact_full_bytes_mismatch"
            ):
                self._verify(manifest_path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, _ = self._fixture(Path(directory))
            manifest_path.write_bytes(b'{"status":"passed","status":"passed"}\n')
            with self.assertRaisesRegex(
                IndependentVerificationError, "duplicate_json_key:status"
            ):
                self._verify(manifest_path)

    def test_extra_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, _ = self._fixture(Path(directory))
            (manifest_path.parent / "artifacts/extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(
                IndependentVerificationError, "artifact_path_set_mismatch"
            ):
                self._verify(manifest_path)

    def test_artifact_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, manifest = self._fixture(Path(directory))
            descriptor = manifest["pairs"][0]["left"]["raw_gray"]  # type: ignore[index]
            path = manifest_path.parent / descriptor["path"]
            path.unlink()
            os.symlink("left.canonical.gray", path)
            with self.assertRaisesRegex(
                IndependentVerificationError, "artifact_symlink"
            ):
                self._verify(manifest_path)

    def test_verifier_sources_do_not_import_frozen_producer_stack(self) -> None:
        forbidden = {
            "formal.chemgraph.challenge_v0_6.e2.producer",
            "formal.chemgraph.challenge_v0_4",
            "formal.chemgraph.challenge_v0_5",
        }
        for filename in (
            "independent_e2_core_v0_6.py",
            "verify_e2_manifest_v0_6.py",
        ):
            source = (Path(__file__).parent / filename).read_text(encoding="ascii")
            tree = ast.parse(source)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            self.assertFalse(
                any(any(name.startswith(prefix) for prefix in forbidden) for name in imported),
                (filename, imported),
            )


if __name__ == "__main__":
    unittest.main()
