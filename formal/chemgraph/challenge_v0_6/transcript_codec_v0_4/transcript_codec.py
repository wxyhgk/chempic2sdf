"""Canonical binary codec for E2 v0.4 relation transcripts.

The codec is deliberately narrower than a generic serialization library. It
accepts only the single frozen v1 domain, R01 through R46, frozen owner and
checker identities, nonempty ordered full-byte inputs, and a nonempty result.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Iterable


DOMAIN = b"chemgraph-e2-v0.4/relation-transcript/v1"
OFFLINE_VERIFIER_SOURCE_SHA256 = (
    "729f42feb9b5d13833e0fe6f9da1c51d70f55933045f34eee601eb720753cc0a"
)
OFFLINE_VERIFIER_TREE_SHA256 = (
    "3c5307d766fc7376659b22f74f354a912b71ff7c9cabcf90df87c79aec44b7e7"
)
MAX_INPUT_COUNT = 1_000_000
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_OWNERS = (
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "outer_deadline_executor+bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "monitor_verified_by_bootstrap_watchdog",
    "monitor_verified_by_bootstrap_watchdog",
    "monitor_verified_by_bootstrap_watchdog",
    "monitor_verified_by_bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor_verified_by_bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "outer_deadline_executor+bootstrap_watchdog",
    "outer_deadline_executor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "outer_deadline_executor+bootstrap_watchdog",
    "outer_deadline_executor+bootstrap_watchdog",
    "outer_deadline_executor+bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor+bootstrap_watchdog",
    "monitor_verified_by_bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
    "bootstrap_watchdog",
)


class TranscriptCodecError(ValueError):
    """Base class for all rejected transcript encodings."""


class UnknownValueError(TranscriptCodecError):
    """A frozen domain, relation, owner, or identity is unknown."""


class OverflowTranscriptError(TranscriptCodecError):
    """A length/count cannot be represented or exceeds the frozen limit."""


class TruncatedTranscriptError(TranscriptCodecError):
    """The input ends before a declared frame is complete."""


class TrailingDataError(TranscriptCodecError):
    """Bytes remain after the unique transcript value."""


class NonCanonicalTranscriptError(TranscriptCodecError):
    """The bytes use an alias or differ from canonical re-encoding."""


def _validate_relation_id(relation_id: int) -> None:
    if isinstance(relation_id, bool) or not isinstance(relation_id, int):
        raise TypeError("relation_id must be an integer")
    if not 1 <= relation_id <= 46:
        raise UnknownValueError(f"unknown relation: {relation_id!r}")


def relation_tag(relation_id: int) -> str:
    _validate_relation_id(relation_id)
    return f"r{relation_id:02d}"


def expected_owner(relation_id: int) -> str:
    _validate_relation_id(relation_id)
    return _OWNERS[relation_id - 1]


def expected_verifier_id(relation_id: int) -> str:
    _validate_relation_id(relation_id)
    version = "v04" if relation_id == 29 else "v03"
    return f"chemgraph_e2_{version}_verify_r{relation_id:02d}"


def expected_theorem_id(relation_id: int) -> str:
    _validate_relation_id(relation_id)
    version = "v04" if relation_id == 29 else "v03"
    return f"chemgraph_e2_{version}_relation_r{relation_id:02d}_sound"


def _u32(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise OverflowTranscriptError(f"u32 overflow: {value}")
    return value.to_bytes(4, "big")


def _u64(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise OverflowTranscriptError(f"u64 overflow: {value}")
    return value.to_bytes(8, "big")


def _frame_u32(value: bytes) -> bytes:
    return _u32(len(value)) + value


def _frame_u64(value: bytes) -> bytes:
    return _u64(len(value)) + value


def _strict_utf8(value: str, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise NonCanonicalTranscriptError(f"{field} must be a nonempty string")
    try:
        return value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise NonCanonicalTranscriptError(f"{field} is not strict UTF-8") from error


def _coerce_nonempty_bytes(value: bytes | bytearray | memoryview, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} must be bytes-like")
    result = bytes(value)
    if not result:
        raise NonCanonicalTranscriptError(f"{field} must be nonempty")
    return result


@dataclass(frozen=True)
class RelationTranscript:
    relation_id: int
    owner: str
    verifier_id: str
    theorem_id: str
    offline_verifier_source_sha256: str
    offline_verifier_tree_sha256: str
    inputs: tuple[bytes, ...]
    result: bytes

    @classmethod
    def canonical(
        cls,
        relation_id: int,
        inputs: Iterable[bytes | bytearray | memoryview],
        result: bytes | bytearray | memoryview,
    ) -> "RelationTranscript":
        _validate_relation_id(relation_id)
        normalized_inputs = tuple(
            _coerce_nonempty_bytes(value, f"inputs[{index}]")
            for index, value in enumerate(inputs)
        )
        if not normalized_inputs:
            raise NonCanonicalTranscriptError("inputs must contain at least one full-byte record")
        if len(normalized_inputs) > MAX_INPUT_COUNT:
            raise OverflowTranscriptError("input count exceeds the frozen limit")
        return cls(
            relation_id=relation_id,
            owner=expected_owner(relation_id),
            verifier_id=expected_verifier_id(relation_id),
            theorem_id=expected_theorem_id(relation_id),
            offline_verifier_source_sha256=OFFLINE_VERIFIER_SOURCE_SHA256,
            offline_verifier_tree_sha256=OFFLINE_VERIFIER_TREE_SHA256,
            inputs=normalized_inputs,
            result=_coerce_nonempty_bytes(result, "result"),
        )


def _validate_canonical_value(value: RelationTranscript) -> None:
    _validate_relation_id(value.relation_id)
    expected = RelationTranscript.canonical(value.relation_id, value.inputs, value.result)
    if value != expected:
        raise NonCanonicalTranscriptError("transcript metadata differs from frozen canonical values")
    for field, digest in (
        ("offline_verifier_source_sha256", value.offline_verifier_source_sha256),
        ("offline_verifier_tree_sha256", value.offline_verifier_tree_sha256),
    ):
        if _LOWER_SHA256.fullmatch(digest) is None:
            raise NonCanonicalTranscriptError(f"{field} is not lowercase SHA256")


def encode_transcript(value: RelationTranscript) -> bytes:
    if not isinstance(value, RelationTranscript):
        raise TypeError("value must be RelationTranscript")
    _validate_canonical_value(value)
    header = (
        DOMAIN,
        _strict_utf8(relation_tag(value.relation_id), "relation_tag"),
        _strict_utf8(value.owner, "owner"),
        _strict_utf8(value.verifier_id, "verifier_id"),
        _strict_utf8(value.theorem_id, "theorem_id"),
        _strict_utf8(value.offline_verifier_source_sha256, "source_sha256"),
        _strict_utf8(value.offline_verifier_tree_sha256, "tree_sha256"),
    )
    encoded_inputs = _u64(len(value.inputs)) + b"".join(_frame_u64(item) for item in value.inputs)
    return b"".join(_frame_u32(item) for item in header) + encoded_inputs + _frame_u64(value.result)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def _take(self, length: int, field: str) -> bytes:
        if length > sys.maxsize:
            raise OverflowTranscriptError(f"{field} length exceeds platform size")
        end = self.offset + length
        if end > len(self.data):
            raise TruncatedTranscriptError(
                f"{field} declares {length} bytes with only {len(self.data) - self.offset} remaining"
            )
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def u32(self, field: str) -> int:
        return int.from_bytes(self._take(4, f"{field}.length"), "big")

    def u64(self, field: str) -> int:
        return int.from_bytes(self._take(8, f"{field}.length"), "big")

    def frame_u32(self, field: str) -> bytes:
        return self._take(self.u32(field), field)

    def frame_u64(self, field: str) -> bytes:
        return self._take(self.u64(field), field)

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise TrailingDataError(f"{len(self.data) - self.offset} trailing bytes")


def _decode_utf8(raw: bytes, field: str) -> str:
    if not raw:
        raise NonCanonicalTranscriptError(f"{field} must be nonempty")
    try:
        value = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise NonCanonicalTranscriptError(f"{field} is not strict UTF-8") from error
    if value.encode("utf-8") != raw:
        raise NonCanonicalTranscriptError(f"{field} has a noncanonical UTF-8 encoding")
    return value


def _decode_relation_tag(tag: str) -> int:
    if re.fullmatch(r"r(?:0[1-9]|[1-3][0-9]|4[0-6])", tag) is not None:
        return int(tag[1:])
    numeric = re.fullmatch(r"[rR]([0-9]+)", tag)
    if numeric is not None:
        relation_id = int(numeric.group(1))
        if 1 <= relation_id <= 46:
            raise NonCanonicalTranscriptError(f"noncanonical relation tag: {tag!r}")
        raise UnknownValueError(f"unknown relation tag: {tag!r}")
    raise UnknownValueError(f"unknown relation tag: {tag!r}")


def decode_transcript(data: bytes | bytearray | memoryview) -> RelationTranscript:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    raw = bytes(data)
    reader = _Reader(raw)
    domain = reader.frame_u32("domain")
    if domain != DOMAIN:
        raise UnknownValueError(f"unknown transcript domain: {domain!r}")
    tag = _decode_utf8(reader.frame_u32("relation_tag"), "relation_tag")
    relation_id = _decode_relation_tag(tag)
    owner = _decode_utf8(reader.frame_u32("owner"), "owner")
    verifier_id = _decode_utf8(reader.frame_u32("verifier_id"), "verifier_id")
    theorem_id = _decode_utf8(reader.frame_u32("theorem_id"), "theorem_id")
    source_sha256 = _decode_utf8(reader.frame_u32("source_sha256"), "source_sha256")
    tree_sha256 = _decode_utf8(reader.frame_u32("tree_sha256"), "tree_sha256")
    if owner != expected_owner(relation_id):
        raise UnknownValueError(f"unexpected owner for R{relation_id:02d}: {owner!r}")
    if verifier_id != expected_verifier_id(relation_id):
        raise UnknownValueError(f"unexpected verifier_id for R{relation_id:02d}")
    if theorem_id != expected_theorem_id(relation_id):
        raise UnknownValueError(f"unexpected theorem_id for R{relation_id:02d}")
    if source_sha256 != OFFLINE_VERIFIER_SOURCE_SHA256:
        if source_sha256.lower() == OFFLINE_VERIFIER_SOURCE_SHA256:
            raise NonCanonicalTranscriptError("source SHA256 must be lowercase")
        raise UnknownValueError("unexpected frozen verifier source SHA256")
    if tree_sha256 != OFFLINE_VERIFIER_TREE_SHA256:
        if tree_sha256.lower() == OFFLINE_VERIFIER_TREE_SHA256:
            raise NonCanonicalTranscriptError("tree SHA256 must be lowercase")
        raise UnknownValueError("unexpected frozen verifier tree SHA256")

    input_count = reader.u64("inputs")
    if input_count == 0:
        raise NonCanonicalTranscriptError("inputs must contain at least one full-byte record")
    if input_count > MAX_INPUT_COUNT:
        raise OverflowTranscriptError(f"input count exceeds {MAX_INPUT_COUNT}")
    inputs = []
    for index in range(input_count):
        item = reader.frame_u64(f"inputs[{index}]")
        if not item:
            raise NonCanonicalTranscriptError(f"inputs[{index}] must be nonempty")
        inputs.append(item)
    result = reader.frame_u64("result")
    if not result:
        raise NonCanonicalTranscriptError("result must be nonempty")
    reader.finish()

    value = RelationTranscript(
        relation_id=relation_id,
        owner=owner,
        verifier_id=verifier_id,
        theorem_id=theorem_id,
        offline_verifier_source_sha256=source_sha256,
        offline_verifier_tree_sha256=tree_sha256,
        inputs=tuple(inputs),
        result=result,
    )
    if encode_transcript(value) != raw:
        raise NonCanonicalTranscriptError("decoded transcript differs from canonical re-encoding")
    return value
