from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from transcript_codec import (
    DOMAIN,
    MAX_INPUT_COUNT,
    NonCanonicalTranscriptError,
    OverflowTranscriptError,
    RelationTranscript,
    TrailingDataError,
    TruncatedTranscriptError,
    UnknownValueError,
    decode_transcript,
    encode_transcript,
    expected_owner,
    expected_theorem_id,
    expected_verifier_id,
)


HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
KNOWN_R01_SHA256 = "29325d264d67a27dd2dcc2c664ce1d350172bfaf6f68448b805719c7653b125d"


def canonical(relation_id: int = 1) -> RelationTranscript:
    return RelationTranscript.canonical(relation_id, (b"first\n", b"second\x00"), b"result\xff")


def header_end(raw: bytes) -> int:
    offset = 0
    for _ in range(7):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        offset += 4 + length
    return offset


def replace_u32_frame(raw: bytes, index: int, value: bytes) -> bytes:
    offset = 0
    for current in range(7):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        end = offset + 4 + length
        if current == index:
            return raw[:offset] + len(value).to_bytes(4, "big") + value + raw[end:]
        offset = end
    raise AssertionError(index)


class TranscriptCodecTests(unittest.TestCase):
    def test_roundtrip_all_relations(self) -> None:
        for relation_id in range(1, 47):
            value = RelationTranscript.canonical(
                relation_id,
                (f"record-{relation_id}".encode(), b"\x00\xff\n"),
                f"result-{relation_id}".encode(),
            )
            with self.subTest(relation_id=relation_id):
                self.assertEqual(value, decode_transcript(encode_transcript(value)))

    def test_metadata_is_frozen(self) -> None:
        for relation_id in range(1, 47):
            value = RelationTranscript.canonical(relation_id, (b"x",), b"y")
            self.assertEqual(expected_owner(relation_id), value.owner)
            self.assertEqual(expected_verifier_id(relation_id), value.verifier_id)
            self.assertEqual(expected_theorem_id(relation_id), value.theorem_id)

    def test_empty_input_list_item_and_result_are_rejected(self) -> None:
        with self.assertRaises(NonCanonicalTranscriptError):
            RelationTranscript.canonical(1, (), b"result")
        with self.assertRaises(NonCanonicalTranscriptError):
            RelationTranscript.canonical(1, (b"",), b"result")
        with self.assertRaises(NonCanonicalTranscriptError):
            RelationTranscript.canonical(1, (b"record",), b"")

        raw = encode_transcript(canonical())
        offset = header_end(raw)
        no_inputs = raw[:offset] + (0).to_bytes(8, "big") + (1).to_bytes(8, "big") + b"x"
        empty_item = (
            raw[:offset]
            + (1).to_bytes(8, "big")
            + (0).to_bytes(8, "big")
            + (1).to_bytes(8, "big")
            + b"x"
        )
        empty_result = (
            raw[:offset]
            + (1).to_bytes(8, "big")
            + (1).to_bytes(8, "big")
            + b"x"
            + (0).to_bytes(8, "big")
        )
        for malformed in (no_inputs, empty_item, empty_result):
            with self.subTest(malformed=malformed), self.assertRaises(NonCanonicalTranscriptError):
                decode_transcript(malformed)

    def test_unknown_domain_and_relation_are_rejected(self) -> None:
        raw = encode_transcript(canonical())
        with self.assertRaises(UnknownValueError):
            decode_transcript(replace_u32_frame(raw, 0, b"unknown-domain"))
        with self.assertRaises(UnknownValueError):
            decode_transcript(replace_u32_frame(raw, 1, b"r47"))

    def test_noncanonical_relation_hash_and_metadata_are_rejected(self) -> None:
        raw = encode_transcript(canonical())
        with self.assertRaises(NonCanonicalTranscriptError):
            decode_transcript(replace_u32_frame(raw, 1, b"R01"))
        source = replace_u32_frame(raw, 5, canonical().offline_verifier_source_sha256.upper().encode())
        with self.assertRaises(NonCanonicalTranscriptError):
            decode_transcript(source)
        with self.assertRaises(UnknownValueError):
            decode_transcript(replace_u32_frame(raw, 2, b"wrong-owner"))

    def test_invalid_utf8_is_rejected(self) -> None:
        raw = encode_transcript(canonical())
        with self.assertRaises(NonCanonicalTranscriptError):
            decode_transcript(replace_u32_frame(raw, 2, b"\xff"))

    def test_overflow_count_and_frame_are_rejected(self) -> None:
        raw = encode_transcript(canonical())
        offset = header_end(raw)
        huge_count = raw[:offset] + (MAX_INPUT_COUNT + 1).to_bytes(8, "big") + raw[offset + 8 :]
        with self.assertRaises(OverflowTranscriptError):
            decode_transcript(huge_count)
        first_length_offset = offset + 8
        huge_frame = (
            raw[:first_length_offset]
            + (2**64 - 1).to_bytes(8, "big")
            + raw[first_length_offset + 8 :]
        )
        with self.assertRaises(OverflowTranscriptError):
            decode_transcript(huge_frame)

    def test_truncation_and_trailing_bytes_are_rejected(self) -> None:
        raw = encode_transcript(canonical())
        for cut in (0, 1, 4, len(raw) // 2, len(raw) - 1):
            with self.subTest(cut=cut), self.assertRaises(TruncatedTranscriptError):
                decode_transcript(raw[:cut])
        with self.assertRaises(TrailingDataError):
            decode_transcript(raw + b"\x00")

    def test_manual_metadata_mutation_cannot_encode(self) -> None:
        value = canonical(29)
        with self.assertRaises(NonCanonicalTranscriptError):
            encode_transcript(replace(value, verifier_id="chemgraph_e2_v03_verify_r29"))

    def test_known_fixture(self) -> None:
        binary = (FIXTURES / "r01_minimal.bin").read_bytes()
        sidecar_raw = (FIXTURES / "r01_minimal.json").read_bytes()
        sidecar = json.loads(sidecar_raw)
        self.assertEqual(
            (json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            sidecar_raw,
        )
        self.assertEqual(KNOWN_R01_SHA256, hashlib.sha256(binary).hexdigest())
        decoded = decode_transcript(binary)
        self.assertEqual(1, decoded.relation_id)
        self.assertEqual(sidecar["input_hex"], [value.hex() for value in decoded.inputs])
        self.assertEqual(sidecar["result_hex"], decoded.result.hex())
        self.assertFalse(sidecar["authority"])
        self.assertFalse(sidecar["authorization_record"])

    def test_all_fixtures_decode_and_match_sidecars(self) -> None:
        binaries = sorted(FIXTURES.glob("*.bin"))
        self.assertGreaterEqual(len(binaries), 5)
        for binary_path in binaries:
            raw = binary_path.read_bytes()
            sidecar = json.loads(binary_path.with_suffix(".json").read_bytes())
            decoded = decode_transcript(raw)
            with self.subTest(fixture=binary_path.name):
                self.assertEqual(sidecar["transcript_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(sidecar["byte_length"], len(raw))
                self.assertEqual(sidecar["relation_id"], f"R{decoded.relation_id:02d}")
                self.assertEqual(sidecar["owner"], decoded.owner)
                self.assertFalse(sidecar["authority"])
                self.assertFalse(sidecar["authorization_record"])


if __name__ == "__main__":
    unittest.main()
