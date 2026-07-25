"""Tests for the generic PyTorch list-of-dicts shard cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.sharded_cache import (
    ShardedCacheIntegrityError,
    ShardedCacheManifestError,
    ShardedListDataset,
    load_sharded_cache_manifest,
    write_sharded_cache,
)


def manifest_sha256(files: list[dict]) -> str:
    digest = hashlib.sha256()
    for shard in files:
        digest.update(shard["file"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(shard["count"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(shard["sha256"]))
        digest.update(b"\n")
    return digest.hexdigest()


class ShardedCacheTests(unittest.TestCase):
    def records(self, count: int = 5) -> list[dict]:
        return [{"index": index, "nested": {"value": index * 2}} for index in range(count)]

    def test_write_manifest_and_map_style_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            manifest = write_sharded_cache(self.records(), root, shard_size=2)

            self.assertEqual(manifest.total_count, 5)
            self.assertEqual(manifest.shard_size, 2)
            self.assertEqual([shard.count for shard in manifest.files], [2, 2, 1])
            self.assertEqual(
                [shard.file for shard in manifest.files],
                ["shard-000000.pt", "shard-000001.pt", "shard-000002.pt"],
            )
            self.assertTrue(all(len(shard.sha256) == 64 for shard in manifest.files))
            self.assertEqual(load_sharded_cache_manifest(root), manifest)
            self.assertEqual(
                list(Path(directory).glob(".cache.tmp-*")),
                [],
                "atomic publication must not leave staging directories",
            )

            dataset = ShardedListDataset(root, max_cached_shards=2)
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset[0]["index"], 0)
            self.assertEqual(dataset[-1]["index"], 4)
            self.assertEqual([item["index"] for item in dataset[1:5:2]], [1, 3])
            self.assertEqual([item["index"] for item in dataset[::-1]], [4, 3, 2, 1, 0])
            with self.assertRaises(IndexError):
                _ = dataset[5]

    def test_shards_are_lazy_and_lru_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            write_sharded_cache(self.records(6), root, shard_size=2)
            real_load = torch.load

            with patch("chemvector.sharded_cache.torch.load", wraps=real_load) as mocked_load:
                dataset = ShardedListDataset(root, max_cached_shards=2)
                self.assertEqual(mocked_load.call_count, 0)
                self.assertEqual(dataset[0]["index"], 0)
                self.assertEqual(dataset[1]["index"], 1)
                self.assertEqual(mocked_load.call_count, 1)
                self.assertEqual(dataset.cached_shard_indices, (0,))

                self.assertEqual(dataset[2]["index"], 2)
                self.assertEqual(dataset[4]["index"], 4)
                self.assertEqual(mocked_load.call_count, 3)
                self.assertEqual(dataset.cached_shard_indices, (1, 2))

                self.assertEqual(dataset[0]["index"], 0)
                self.assertEqual(mocked_load.call_count, 4)
                self.assertEqual(dataset.cached_shard_indices, (2, 0))
                _, kwargs = mocked_load.call_args
                self.assertIs(kwargs["weights_only"], False)
                self.assertEqual(kwargs["map_location"], "cpu")

    def test_missing_and_damaged_shards_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing"
            write_sharded_cache(self.records(3), root, shard_size=2)
            (root / "shard-000001.pt").unlink()
            with self.assertRaisesRegex(ShardedCacheIntegrityError, "Missing shard"):
                ShardedListDataset(root)

            damaged_root = Path(directory) / "damaged"
            write_sharded_cache(self.records(3), damaged_root, shard_size=2)
            with (damaged_root / "shard-000000.pt").open("ab") as handle:
                handle.write(b"damage")
            dataset = ShardedListDataset(damaged_root)
            with self.assertRaisesRegex(ShardedCacheIntegrityError, "SHA256 mismatch"):
                _ = dataset[0]

    def test_payload_count_mismatch_is_detected_after_valid_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            write_sharded_cache(self.records(2), root, shard_size=2)
            shard_path = root / "shard-000000.pt"
            torch.save(self.records(1), shard_path)

            payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            payload["files"][0]["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            payload["sha256"] = manifest_sha256(payload["files"])
            (root / "manifest.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            dataset = ShardedListDataset(root)
            with self.assertRaisesRegex(ShardedCacheIntegrityError, "contains 1 records"):
                dataset.validate()

    def test_validate_rechecks_shards_already_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            write_sharded_cache(self.records(2), root, shard_size=2)
            dataset = ShardedListDataset(root)
            self.assertEqual(dataset[0]["index"], 0)

            with (root / "shard-000000.pt").open("ab") as handle:
                handle.write(b"damage-after-first-load")

            with self.assertRaisesRegex(ShardedCacheIntegrityError, "SHA256 mismatch"):
                dataset.validate()

    def test_invalid_manifest_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            write_sharded_cache(self.records(3), root, shard_size=2)
            path = root / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["total_count"] = 4
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ShardedCacheManifestError, "total_count"):
                load_sharded_cache_manifest(root)

    def test_failed_write_never_publishes_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            records = iter([{"valid": True}, "not-a-dictionary"])
            with self.assertRaisesRegex(TypeError, "dictionaries only"):
                write_sharded_cache(records, root, shard_size=1)

            self.assertFalse(root.exists())
            self.assertEqual(list(Path(directory).glob(".cache.tmp-*")), [])

    def test_empty_cache_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            manifest = write_sharded_cache([], root, shard_size=4)
            dataset = ShardedListDataset(root)

            self.assertEqual(manifest.total_count, 0)
            self.assertEqual(manifest.files, ())
            self.assertEqual(len(dataset), 0)
            dataset.validate()
            with self.assertRaises(IndexError):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
