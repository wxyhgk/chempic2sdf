#!/usr/bin/env python3
"""Generate deterministic non-authoritative transcript codec fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transcript_codec import RelationTranscript, encode_transcript


FIXTURES = (
    ("r01_minimal", 1, (b"{}\n", b'{"status":"passed"}\n'), b"\x01contract-bindings"),
    ("r04_binary", 4, (b"ledger\x00record", b"authorization\n"), bytes(range(32))),
    ("r24_utf8", 24, ("路径/marker".encode("utf-8"),), "固定路径结果".encode("utf-8")),
    ("r29_exact_four", 29, (b"authorization", b"reviewer-1", b"reviewer-2", b"registry"), b"exact-four"),
    ("r46_phase", 46, (b'{"phase":"official"}\n',), b"dry=14;official=17"),
)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def main() -> None:
    output = Path(__file__).resolve().parent / "fixtures"
    output.mkdir(exist_ok=True)
    expected_names: set[str] = set()
    for fixture_id, relation_id, inputs, result in FIXTURES:
        transcript = RelationTranscript.canonical(relation_id, inputs, result)
        encoded = encode_transcript(transcript)
        binary_name = f"{fixture_id}.bin"
        sidecar_name = f"{fixture_id}.json"
        expected_names.update((binary_name, sidecar_name))
        (output / binary_name).write_bytes(encoded)
        sidecar = {
            "authority": False,
            "authorization_record": False,
            "byte_length": len(encoded),
            "fixture_id": fixture_id,
            "input_hex": [value.hex() for value in transcript.inputs],
            "offline_verifier_source_sha256": transcript.offline_verifier_source_sha256,
            "offline_verifier_tree_sha256": transcript.offline_verifier_tree_sha256,
            "owner": transcript.owner,
            "relation_id": f"R{relation_id:02d}",
            "result_hex": transcript.result.hex(),
            "schema_version": "chemgraph-e2-relation-transcript-fixture-v0.4",
            "theorem_id": transcript.theorem_id,
            "transcript_sha256": hashlib.sha256(encoded).hexdigest(),
            "verifier_id": transcript.verifier_id,
        }
        (output / sidecar_name).write_bytes(canonical_json(sidecar))
    for path in output.iterdir():
        if path.is_file() and path.name not in expected_names:
            path.unlink()


if __name__ == "__main__":
    main()
