#!/usr/bin/env python3
"""Build deterministic, chemistry-free fixtures for blind carrier validators."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).resolve().parent


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def grayscale_png(decoded: bytes, width: int = 768, height: int = 768) -> bytes:
    if len(decoded) != width * height:
        raise ValueError("decoded grayscale byte length does not match dimensions")
    rows = [
        b"\x00" + decoded[row * width:(row + 1) * width]
        for row in range(height)
    ]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
            + chunk(b"IEND", b""))


def synthetic_png() -> bytes:
    decoded = bytearray([255]) * (768 * 768)
    decoded[384 * 768 + 100:384 * 768 + 501] = bytes([0]) * 401
    return grayscale_png(bytes(decoded))


def synthetic_double_png() -> bytes:
    decoded = bytearray([255]) * (768 * 768)
    for row in (383, 385):
        decoded[row * 768 + 100:row * 768 + 501] = bytes([0]) * 401
    return grayscale_png(bytes(decoded))


def input_record(png: bytes) -> dict:
    return {
        "image": {
            "bytes_base64": base64.b64encode(png).decode("ascii"),
            "format": "png",
            "height": 768,
            "width": 768,
        },
        "schema_version": "chemgraph-blind-carrier-input-v0.1",
        "training_authorized": False,
    }


def prediction_record() -> dict:
    return {
        "bond_hypotheses": [
            {
                "begin_terminal_id": 0,
                "end_terminal_id": 1,
                "id": 0,
                "lane_endpoint_alignment": [{
                    "begin_terminal_id": 0,
                    "end_terminal_id": 1,
                    "lane_id": 0,
                }],
                "lane_ids": [0],
                "order": 1,
            },
            {
                "begin_terminal_id": 1,
                "end_terminal_id": 2,
                "id": 1,
                "lane_endpoint_alignment": [{
                    "begin_terminal_id": 1,
                    "end_terminal_id": 2,
                    "lane_id": 1,
                }],
                "lane_ids": [1],
                "order": 1,
            },
        ],
        "image": {"coordinate_format": "Q4-pixel", "height": 768, "width": 768},
        "interpretations": [{
            "id": 0,
            "ownership": [
                {"owner_id": 0, "owner_kind": "bond_lane", "segment_id": 0},
                {"owner_id": 1, "owner_kind": "bond_lane", "segment_id": 1},
            ],
            "selected_bond_ids": [0, 1],
        }],
        "lane_hypotheses": [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0, "segment_ids": [0]},
            {"begin_terminal_id": 1, "end_terminal_id": 2, "id": 1, "segment_ids": [1]},
        ],
        "schema_version": "chemgraph-blind-carrier-prediction-v0.1",
        "segments": [
            {
                "id": 0,
                "polyline_q4": [[1600, 6144], [4800, 6144]],
                "support_pixel_count": 201,
                "support_runs": [[384, 100, 201]],
                "width_q4": 16,
            },
            {
                "id": 1,
                "polyline_q4": [[4800, 6144], [8000, 6144]],
                "support_pixel_count": 201,
                "support_runs": [[384, 300, 201]],
                "width_q4": 16,
            },
        ],
        "status": "ok",
        "terminals": [
            {"center_q4": [1600, 6144], "id": 0, "support_segment_ids": [0]},
            {"center_q4": [4800, 6144], "id": 1, "support_segment_ids": [0, 1]},
            {"center_q4": [8000, 6144], "id": 2, "support_segment_ids": [1]},
        ],
        "training_authorized": False,
    }


def double_prediction_record() -> dict:
    return {
        "bond_hypotheses": [{
            "begin_terminal_id": 0,
            "end_terminal_id": 1,
            "id": 0,
            "lane_endpoint_alignment": [
                {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 0},
                {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 1},
            ],
            "lane_ids": [0, 1],
            "order": 2,
        }],
        "image": {"coordinate_format": "Q4-pixel", "height": 768, "width": 768},
        "interpretations": [{
            "id": 0,
            "ownership": [
                {"owner_id": 0, "owner_kind": "bond_lane", "segment_id": 0},
                {"owner_id": 1, "owner_kind": "bond_lane", "segment_id": 1},
            ],
            "selected_bond_ids": [0],
        }],
        "lane_hypotheses": [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0, "segment_ids": [0]},
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 1, "segment_ids": [1]},
        ],
        "schema_version": "chemgraph-blind-carrier-prediction-v0.1",
        "segments": [
            {
                "id": 0,
                "polyline_q4": [[1608, 6128], [8008, 6128]],
                "support_pixel_count": 401,
                "support_runs": [[383, 100, 401]],
                "width_q4": 16,
            },
            {
                "id": 1,
                "polyline_q4": [[1608, 6160], [8008, 6160]],
                "support_pixel_count": 401,
                "support_runs": [[385, 100, 401]],
                "width_q4": 16,
            },
        ],
        "status": "ok",
        "terminals": [
            {"center_q4": [1608, 6144], "id": 0, "support_segment_ids": [0, 1]},
            {"center_q4": [8008, 6144], "id": 1, "support_segment_ids": [0, 1]},
        ],
        "training_authorized": False,
    }


def main() -> None:
    png = synthetic_png()
    double_png = synthetic_double_png()
    (ROOT / "synthetic.png").write_bytes(png)
    (ROOT / "synthetic_double.png").write_bytes(double_png)
    (ROOT / "valid_inbox.json").write_bytes(canonical_bytes(input_record(png)))
    (ROOT / "valid_double_inbox.json").write_bytes(canonical_bytes(input_record(double_png)))
    (ROOT / "valid_prediction.json").write_bytes(canonical_bytes(prediction_record()))
    (ROOT / "valid_double_prediction.json").write_bytes(canonical_bytes(double_prediction_record()))
    (ROOT / "valid_reject_prediction.json").write_bytes(canonical_bytes({
        "reason": "no_supported_carrier",
        "schema_version": "chemgraph-blind-carrier-prediction-v0.1",
        "status": "reject",
        "training_authorized": False,
    }))


if __name__ == "__main__":
    main()
