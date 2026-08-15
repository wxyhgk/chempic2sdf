#!/usr/bin/env python3
"""Build deterministic chemistry-free Blind Carrier v0.2 fixtures."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "chemgraph-blind-carrier-prediction-v0.2"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def grayscale_png(rows_with_ink: tuple[int, ...]) -> bytes:
    width = height = 768
    decoded = bytearray([255]) * (width * height)
    for row in rows_with_ink:
        decoded[row * width + 100:row * width + 501] = bytes([0]) * 401
    scanlines = b"".join(
        b"\x00" + decoded[row * width:(row + 1) * width]
        for row in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + chunk(b"IEND", b""))


def input_record(png: bytes) -> dict:
    return {
        "image": {
            "bytes_base64": base64.b64encode(png).decode("ascii"),
            "format": "png", "height": 768, "width": 768,
        },
        "schema_version": "chemgraph-blind-carrier-input-v0.2",
        "training_authorized": False,
    }


def chain_prediction() -> dict:
    return {
        "bond_hypotheses": [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0,
             "lane_endpoint_alignment": [{"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": 0}],
             "lane_ids": [0], "order": 1},
            {"begin_terminal_id": 1, "end_terminal_id": 2, "id": 1,
             "lane_endpoint_alignment": [{"begin_terminal_id": 1, "end_terminal_id": 2, "lane_id": 1}],
             "lane_ids": [1], "order": 1},
        ],
        "image": {"coordinate_format": "Q4-pixel", "height": 768, "width": 768},
        "interpretations": [{"id": 0, "ownership": [
            {"owner_id": 0, "owner_kind": "bond_lane", "segment_id": 0},
            {"owner_id": 1, "owner_kind": "bond_lane", "segment_id": 1},
        ], "selected_bond_ids": [0, 1]}],
        "lane_hypotheses": [
            {"begin_terminal_id": 0, "end_terminal_id": 1, "id": 0, "segment_ids": [0]},
            {"begin_terminal_id": 1, "end_terminal_id": 2, "id": 1, "segment_ids": [1]},
        ],
        "schema_version": SCHEMA_VERSION,
        "segments": [
            {"id": 0, "polyline_q4": [[1600, 6144], [4800, 6144]],
             "support_pixel_count": 201, "support_runs": [[384, 100, 201]], "width_q4": 16},
            {"id": 1, "polyline_q4": [[4800, 6144], [8000, 6144]],
             "support_pixel_count": 201, "support_runs": [[384, 300, 201]], "width_q4": 16},
        ],
        "status": "ok",
        "terminals": [
            {"center_q4": [1600, 6144], "endpoint_alignment": [{"endpoint_index": 0, "segment_id": 0}], "id": 0, "support_segment_ids": [0]},
            {"center_q4": [4800, 6144], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 0}, {"endpoint_index": 0, "segment_id": 1}], "id": 1, "support_segment_ids": [0, 1]},
            {"center_q4": [8000, 6144], "endpoint_alignment": [{"endpoint_index": 1, "segment_id": 1}], "id": 2, "support_segment_ids": [1]},
        ],
        "training_authorized": False,
    }


def parallel_prediction(order: int) -> dict:
    rows = {2: (383, 385), 3: (381, 384, 387)}[order]
    y_values = [16 * row + 8 for row in rows]
    segments = [
        {"id": index, "polyline_q4": [[1608, y], [8008, y]],
         "support_pixel_count": 401, "support_runs": [[row, 100, 401]], "width_q4": 16}
        for index, (row, y) in enumerate(zip(rows, y_values))
    ]
    begin_alignment = [
        {"endpoint_index": 0, "segment_id": index} for index in range(order)
    ]
    end_alignment = [
        {"endpoint_index": 1, "segment_id": index} for index in range(order)
    ]
    lanes = [
        {"begin_terminal_id": 0, "end_terminal_id": 1, "id": index, "segment_ids": [index]}
        for index in range(order)
    ]
    return {
        "bond_hypotheses": [{
            "begin_terminal_id": 0, "end_terminal_id": 1, "id": 0,
            "lane_endpoint_alignment": [
                {"begin_terminal_id": 0, "end_terminal_id": 1, "lane_id": index}
                for index in range(order)
            ],
            "lane_ids": list(range(order)), "order": order,
        }],
        "image": {"coordinate_format": "Q4-pixel", "height": 768, "width": 768},
        "interpretations": [{"id": 0, "ownership": [
            {"owner_id": index, "owner_kind": "bond_lane", "segment_id": index}
            for index in range(order)
        ], "selected_bond_ids": [0]}],
        "lane_hypotheses": lanes,
        "schema_version": SCHEMA_VERSION,
        "segments": segments,
        "status": "ok",
        "terminals": [
            {"center_q4": [1608, 6152], "endpoint_alignment": begin_alignment,
             "id": 0, "support_segment_ids": list(range(order))},
            {"center_q4": [8008, 6152], "endpoint_alignment": end_alignment,
             "id": 1, "support_segment_ids": list(range(order))},
        ],
        "training_authorized": False,
    }


def main() -> None:
    chain_png = grayscale_png((384,))
    double_png = grayscale_png((383, 385))
    triple_png = grayscale_png((381, 384, 387))
    for name, png, prediction in (
        ("", chain_png, chain_prediction()),
        ("_double", double_png, parallel_prediction(2)),
        ("_triple", triple_png, parallel_prediction(3)),
    ):
        (ROOT / f"synthetic{name}.png").write_bytes(png)
        (ROOT / f"valid{name}_inbox.json").write_bytes(canonical_bytes(input_record(png)))
        (ROOT / f"valid{name}_prediction.json").write_bytes(canonical_bytes(prediction))
    (ROOT / "valid_reject_prediction.json").write_bytes(canonical_bytes({
        "reason": "no_supported_carrier", "schema_version": SCHEMA_VERSION,
        "status": "reject", "training_authorized": False,
    }))


if __name__ == "__main__":
    main()
