"""Integrity-checked, lazily loaded caches for lists of dictionaries.

The cache is intentionally independent of ChemVector's training pipelines.  A
writer publishes a complete directory atomically, while ``ShardedListDataset``
provides ordinary PyTorch map-style indexing without loading every shard into
memory.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import operator
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any

import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = "chemvector-sharded-list-cache-1"
MANIFEST_FILENAME = "manifest.json"


class ShardedCacheError(RuntimeError):
    """Base error for an unreadable or invalid sharded cache."""


class ShardedCacheManifestError(ShardedCacheError):
    """Raised when a cache manifest is absent or internally inconsistent."""


class ShardedCacheIntegrityError(ShardedCacheError):
    """Raised when a shard is missing, damaged, or has unexpected contents."""


@dataclass(frozen=True, slots=True)
class ShardFile:
    """One file entry from a sharded cache manifest."""

    file: str
    count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShardedCacheManifest:
    """Validated, immutable view of ``manifest.json``."""

    total_count: int
    shard_size: int
    sha256: str
    files: tuple[ShardFile, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": "torch-list-of-dicts",
            "total_count": self.total_count,
            "shard_size": self.shard_size,
            "sha256": self.sha256,
            "files": [
                {
                    "file": shard.file,
                    "count": shard.count,
                    "sha256": shard.sha256,
                }
                for shard in self.files
            ],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(files: Iterable[ShardFile]) -> str:
    """Hash the ordered file metadata and each file's content digest."""

    digest = hashlib.sha256()
    for shard in files:
        digest.update(shard.file.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(shard.count).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(shard.sha256))
        digest.update(b"\n")
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_torch_file(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        torch.save(records, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(path: Path, manifest: ShardedCacheManifest) -> None:
    payload = json.dumps(
        manifest.to_dict(),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_sharded_cache(
    records: Iterable[dict[str, Any]],
    directory: str | Path,
    *,
    shard_size: int,
) -> ShardedCacheManifest:
    """Write dictionaries into a newly published, fixed-size shard directory.

    All output is first written to a temporary sibling directory.  The target
    directory becomes visible only after every shard and the manifest have
    been flushed successfully.  Existing targets are rejected so an old cache
    can never be partially replaced.
    """

    if isinstance(shard_size, bool) or not isinstance(shard_size, int):
        raise TypeError("shard_size must be an integer")
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than zero")

    destination = Path(directory)
    if not destination.name:
        raise ValueError("directory must name a cache directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Cache directory already exists: {destination}")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        files: list[ShardFile] = []
        shard: list[dict[str, Any]] = []
        total_count = 0

        def flush_shard() -> None:
            nonlocal shard
            if not shard:
                return
            filename = f"shard-{len(files):06d}.pt"
            path = staging / filename
            _write_torch_file(path, shard)
            files.append(
                ShardFile(
                    file=filename,
                    count=len(shard),
                    sha256=_sha256_file(path),
                )
            )
            shard = []

        for record in records:
            if not isinstance(record, dict):
                raise TypeError(
                    "Sharded caches accept dictionaries only; "
                    f"record {total_count} has type {type(record).__name__}"
                )
            shard.append(record)
            total_count += 1
            if len(shard) == shard_size:
                flush_shard()
        flush_shard()

        manifest = ShardedCacheManifest(
            total_count=total_count,
            shard_size=shard_size,
            sha256=_manifest_sha256(files),
            files=tuple(files),
        )
        _write_manifest(staging / MANIFEST_FILENAME, manifest)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _require_integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ShardedCacheManifestError(
            f"Manifest field {field!r} must be an integer >= {minimum}"
        )
    return value


def _validate_relative_file(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise ShardedCacheManifestError(
            f"Manifest files[{index}].file must be a non-empty string"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", ".."}:
        raise ShardedCacheManifestError(
            f"Manifest files[{index}].file is not a safe relative path: {value!r}"
        )
    return value


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ShardedCacheManifestError(f"Manifest field {field!r} is not SHA256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ShardedCacheManifestError(
            f"Manifest field {field!r} is not SHA256"
        ) from error
    return value.lower()


def load_sharded_cache_manifest(directory: str | Path) -> ShardedCacheManifest:
    """Read and validate the internal consistency of a cache manifest."""

    root = Path(directory)
    path = root / MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ShardedCacheManifestError(f"Missing cache manifest: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ShardedCacheManifestError(f"Cannot read cache manifest: {path}") from error

    if not isinstance(payload, dict):
        raise ShardedCacheManifestError("Cache manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ShardedCacheManifestError(
            "Unsupported cache schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("format") != "torch-list-of-dicts":
        raise ShardedCacheManifestError(
            f"Unsupported cache format: {payload.get('format')!r}"
        )

    total_count = _require_integer(payload.get("total_count"), "total_count", minimum=0)
    shard_size = _require_integer(payload.get("shard_size"), "shard_size", minimum=1)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ShardedCacheManifestError("Manifest field 'files' must be a list")

    files: list[ShardFile] = []
    seen_names: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            raise ShardedCacheManifestError(f"Manifest files[{index}] must be an object")
        filename = _validate_relative_file(raw_file.get("file"), index)
        if filename in seen_names:
            raise ShardedCacheManifestError(f"Duplicate shard file in manifest: {filename}")
        seen_names.add(filename)
        files.append(
            ShardFile(
                file=filename,
                count=_require_integer(
                    raw_file.get("count"),
                    f"files[{index}].count",
                    minimum=1,
                ),
                sha256=_validate_sha256(
                    raw_file.get("sha256"),
                    f"files[{index}].sha256",
                ),
            )
        )

    if sum(shard.count for shard in files) != total_count:
        raise ShardedCacheManifestError(
            "Manifest total_count does not equal the sum of shard counts"
        )
    expected_file_count = (total_count + shard_size - 1) // shard_size
    if len(files) != expected_file_count:
        raise ShardedCacheManifestError(
            "Manifest file count is inconsistent with total_count and shard_size"
        )
    for index, shard in enumerate(files):
        expected_count = (
            shard_size
            if index < len(files) - 1
            else total_count - shard_size * (len(files) - 1)
        )
        if shard.count != expected_count:
            raise ShardedCacheManifestError(
                f"Manifest shard {shard.file!r} has count {shard.count}; "
                f"expected {expected_count}"
            )

    declared_sha256 = _validate_sha256(payload.get("sha256"), "sha256")
    actual_manifest_sha256 = _manifest_sha256(files)
    if declared_sha256 != actual_manifest_sha256:
        raise ShardedCacheManifestError(
            "Manifest SHA256 does not match its ordered file entries"
        )
    return ShardedCacheManifest(
        total_count=total_count,
        shard_size=shard_size,
        sha256=declared_sha256,
        files=tuple(files),
    )


class ShardedListDataset(Dataset[dict[str, Any]]):
    """Map-style dataset that lazily loads integrity-checked list shards."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_cached_shards: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(max_cached_shards, bool) or not isinstance(max_cached_shards, int):
            raise TypeError("max_cached_shards must be an integer")
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be greater than zero")

        self.directory = Path(directory)
        self.manifest = load_sharded_cache_manifest(self.directory)
        self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._shard_starts: list[int] = []
        offset = 0
        for shard in self.manifest.files:
            path = self.directory / shard.file
            if not path.is_file():
                raise ShardedCacheIntegrityError(f"Missing shard file: {path}")
            self._shard_starts.append(offset)
            offset += shard.count

    @property
    def cached_shard_indices(self) -> tuple[int, ...]:
        """Shard indices currently resident in memory, oldest first."""

        return tuple(self._cache)

    def clear_cache(self) -> None:
        """Drop all deserialized shards from memory."""

        self._cache.clear()

    def __len__(self) -> int:
        return self.manifest.total_count

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached

        record = self.manifest.files[shard_index]
        path = self.directory / record.file
        if not path.is_file():
            raise ShardedCacheIntegrityError(f"Missing shard file: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != record.sha256:
            raise ShardedCacheIntegrityError(
                f"SHA256 mismatch for shard {record.file!r}: "
                f"expected {record.sha256}, got {actual_sha256}"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ShardedCacheIntegrityError(
                f"Cannot deserialize shard {record.file!r}"
            ) from error
        if not isinstance(payload, list):
            raise ShardedCacheIntegrityError(
                f"Shard {record.file!r} must contain a list, "
                f"not {type(payload).__name__}"
            )
        if len(payload) != record.count:
            raise ShardedCacheIntegrityError(
                f"Shard {record.file!r} contains {len(payload)} records; "
                f"manifest declares {record.count}"
            )
        for local_index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ShardedCacheIntegrityError(
                    f"Shard {record.file!r} item {local_index} is not a dictionary"
                )

        self._cache[shard_index] = payload
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return payload

    def _get_one(self, index: int) -> dict[str, Any]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"ShardedListDataset index out of range: {index}")
        shard_index = bisect_right(self._shard_starts, index) - 1
        local_index = index - self._shard_starts[shard_index]
        return self._load_shard(shard_index)[local_index]

    def __getitem__(
        self,
        index: int | slice,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self._get_one(position) for position in range(*index.indices(len(self)))]
        try:
            integer_index = operator.index(index)
        except TypeError as error:
            raise TypeError("Dataset indices must be integers or slices") from error
        return self._get_one(integer_index)

    def validate(self) -> None:
        """Load and verify every shard without retaining more than the LRU limit."""

        self.clear_cache()
        for shard_index in range(len(self.manifest.files)):
            self._load_shard(shard_index)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


__all__ = [
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "ShardFile",
    "ShardedCacheError",
    "ShardedCacheIntegrityError",
    "ShardedCacheManifest",
    "ShardedCacheManifestError",
    "ShardedListDataset",
    "load_sharded_cache_manifest",
    "write_sharded_cache",
]
