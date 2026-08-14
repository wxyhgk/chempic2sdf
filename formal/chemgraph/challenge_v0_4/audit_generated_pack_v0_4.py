#!/usr/bin/env python3
"""Independent stdlib audit for a generated ChemGraph v0.4 pack.

This module deliberately does not import the generator or rasterizer.  It
checks the emitted contract from bytes and reconstructs graph signatures from
Interpretation records.  Candidate completeness beyond the emitted
Interpretations is treated as a separate semantic gate.
"""

from __future__ import annotations

import argparse
import binascii
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Iterable
import zlib

try:
    from .validate_schema_bundle import SchemaInstanceError, validate_instance
except ImportError:
    from validate_schema_bundle import SchemaInstanceError, validate_instance  # type: ignore


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
Q = 1 << 16
CANVAS_SIZE = 768
AA_SAMPLES = {"binary_1x": 1, "box_2x2": 2, "box_4x4": 4}
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
EXPECTED_SMOKE_QUOTAS = {**{f"s{i}": 4 for i in range(1, 9)},
                         **{f"c{i}": 4 for i in range(1, 5)}}
EXPECTED_FULL_QUOTAS = {
    **{f"s{i}": 512 for i in range(1, 9)},
    "c1": 512,
    "c2": 384,
    "c3": 256,
    "c4": 384,
}
JSONL_SCHEMAS = {
    "ink_programs.jsonl": "ink_program.schema.json",
    "transforms.jsonl": "transform.schema.json",
    "interpretations.jsonl": "interpretation.schema.json",
    "graph_sets.jsonl": "graph_set.schema.json",
    "observations.jsonl": "observation_gt.schema.json",
    "rejection_log.jsonl": "rejection_log.schema.json",
}


class PackAuditError(ValueError):
    """The pack is malformed enough that the audit cannot continue safely."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackAuditError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def _load_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackAuditError(f"invalid_json:{path}:{error}") from error
    if raw != _canonical_json_bytes(value):
        raise PackAuditError(f"noncanonical_json:{path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PackAuditError(f"unreadable_jsonl:{path}:{error}") from error
    if raw and not raw.endswith(b"\n"):
        raise PackAuditError(f"jsonl_missing_final_newline:{path}")
    records = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PackAuditError(f"invalid_jsonl:{path}:{line_number}:{error}") from error
        if not isinstance(value, dict):
            raise PackAuditError(f"jsonl_record_not_object:{path}:{line_number}")
        if line + b"\n" != _canonical_json_bytes(value):
            raise PackAuditError(f"noncanonical_jsonl:{path}:{line_number}")
        records.append(value)
    return records


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _domain_hash(domain: str, payload: bytes) -> str:
    encoded = domain.encode("ascii")
    framed = len(encoded).to_bytes(4, "big") + encoded + len(payload).to_bytes(8, "big") + payload
    return _sha256(framed)


def _safe_artifact(root: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise PackAuditError(f"unsafe_relative_path:{relative_path}")
    path = root.joinpath(*posix.parts)
    if path.is_symlink() or not path.is_file():
        raise PackAuditError(f"artifact_not_regular_file:{relative_path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise PackAuditError(f"artifact_escapes_root:{relative_path}") from error
    return path


def _check_artifact(root: Path, reference: dict[str, Any]) -> None:
    path = _safe_artifact(root, reference["relative_path"])
    payload = path.read_bytes()
    if len(payload) != reference["byte_length"]:
        raise PackAuditError(f"artifact_length_mismatch:{reference['relative_path']}")
    if _sha256(payload) != reference["sha256"]:
        raise PackAuditError(f"artifact_hash_mismatch:{reference['relative_path']}")


def _decode_canonical_gray_png(payload: bytes) -> bytes:
    if not payload.startswith(PNG_SIGNATURE):
        raise PackAuditError("png_bad_signature")
    cursor = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while cursor < len(payload):
        if cursor + 12 > len(payload):
            raise PackAuditError("png_truncated_chunk")
        length = struct.unpack(">I", payload[cursor:cursor + 4])[0]
        kind = payload[cursor + 4:cursor + 8]
        end = cursor + 12 + length
        if end > len(payload):
            raise PackAuditError("png_truncated_payload")
        chunk_payload = payload[cursor + 8:end - 4]
        expected_crc = struct.unpack(">I", payload[end - 4:end])[0]
        actual_crc = binascii.crc32(chunk_payload, binascii.crc32(kind)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PackAuditError("png_crc_mismatch")
        chunks.append((kind, chunk_payload))
        cursor = end
        if kind == b"IEND":
            break
    if cursor != len(payload):
        raise PackAuditError("png_trailing_bytes")
    if [kind for kind, _ in chunks] != [b"IHDR", b"IDAT", b"IEND"]:
        raise PackAuditError("png_noncanonical_chunks")
    if len(chunks[0][1]) != 13:
        raise PackAuditError("png_bad_ihdr_length")
    header = struct.unpack(">IIBBBBB", chunks[0][1])
    if header != (768, 768, 8, 0, 0, 0, 0):
        raise PackAuditError(f"png_wrong_profile:{header}")
    try:
        scanlines = zlib.decompress(chunks[1][1])
    except zlib.error as error:
        raise PackAuditError(f"png_zlib_error:{error}") from error
    if len(scanlines) != 768 * 769:
        raise PackAuditError("png_wrong_decoded_length")
    gray = bytearray(768 * 768)
    for row in range(768):
        source = row * 769
        if scanlines[source] != 0:
            raise PackAuditError("png_nonzero_filter")
        gray[row * 768:(row + 1) * 768] = scanlines[source + 1:source + 769]
    return bytes(gray)


def _inside_segment(
    sample_x: int,
    sample_y: int,
    start: tuple[int, int],
    end: tuple[int, int],
    width_q16: int,
    cap: str,
) -> bool:
    ax, ay = start
    bx, by = end
    vx, vy = bx - ax, by - ay
    wx, wy = sample_x - ax, sample_y - ay
    length_squared = vx * vx + vy * vy
    if length_squared == 0:
        return False
    radius = width_q16 // 2
    radius_squared = radius * radius
    projection = wx * vx + wy * vy
    cross = wx * vy - wy * vx
    if cap == "butt":
        return (0 <= projection <= length_squared and
                cross * cross <= radius_squared * length_squared)
    if cap == "square":
        extension = radius * math.isqrt(length_squared)
        return (-extension <= projection <= length_squared + extension and
                cross * cross <= radius_squared * length_squared)
    if cap != "round":
        raise PackAuditError(f"unsupported_cap:{cap}")
    if projection < 0:
        return wx * wx + wy * wy <= radius_squared
    if projection > length_squared:
        dx, dy = sample_x - bx, sample_y - by
        return dx * dx + dy * dy <= radius_squared
    return cross * cross <= radius_squared * length_squared


def _pixel_bounds(
    start: tuple[int, int],
    end: tuple[int, int],
    width_q16: int,
) -> tuple[int, int, int, int]:
    padding = (width_q16 + Q - 1) // Q + 2
    return (
        max(0, min(start[0], end[0]) // Q - padding),
        min(CANVAS_SIZE - 1, max(start[0], end[0]) // Q + padding),
        max(0, min(start[1], end[1]) // Q - padding),
        min(CANVAS_SIZE - 1, max(start[1], end[1]) // Q + padding),
    )


def _round_half_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return quotient


def _independent_masks(
    primitives: list[dict[str, Any]],
    style: dict[str, Any],
    *,
    include_round_joins: bool,
) -> dict[int, int]:
    sample_count = AA_SAMPLES.get(style["antialias_profile"])
    if sample_count is None:
        raise PackAuditError("unsupported_antialias_profile")
    if style["join"] != "round":
        raise PackAuditError(f"unsupported_join:{style['join']}")
    width_q16 = style["width_q16"]
    offsets = [((2 * index + 1) * Q) // (2 * sample_count)
               for index in range(sample_count)]
    masks: dict[int, int] = {}
    endpoints: Counter[tuple[int, int]] = Counter()
    for primitive in primitives:
        if len(primitive["points_q16"]) != 2:
            raise PackAuditError("audit_only_supports_two_point_primitives")
        start, end = (tuple(point) for point in primitive["points_q16"])
        endpoints.update((start, end))
        min_x, max_x, min_y, max_y = _pixel_bounds(start, end, width_q16)
        for pixel_y in range(min_y, max_y + 1):
            base_y = pixel_y * Q
            row = pixel_y * CANVAS_SIZE
            for pixel_x in range(min_x, max_x + 1):
                base_x = pixel_x * Q
                bit = 1
                mask = masks.get(row + pixel_x, 0)
                for offset_y in offsets:
                    for offset_x in offsets:
                        if not mask & bit and _inside_segment(
                            base_x + offset_x,
                            base_y + offset_y,
                            start,
                            end,
                            width_q16,
                            style["cap"],
                        ):
                            mask |= bit
                        bit <<= 1
                if mask:
                    masks[row + pixel_x] = mask
    if include_round_joins:
        radius = width_q16 // 2
        padding = (width_q16 + Q - 1) // Q + 2
        for center, count in endpoints.items():
            if count < 2:
                continue
            min_x = max(0, center[0] // Q - padding)
            max_x = min(CANVAS_SIZE - 1, center[0] // Q + padding)
            min_y = max(0, center[1] // Q - padding)
            max_y = min(CANVAS_SIZE - 1, center[1] // Q + padding)
            for pixel_y in range(min_y, max_y + 1):
                base_y = pixel_y * Q
                row = pixel_y * CANVAS_SIZE
                for pixel_x in range(min_x, max_x + 1):
                    base_x = pixel_x * Q
                    bit = 1
                    mask = masks.get(row + pixel_x, 0)
                    for offset_y in offsets:
                        for offset_x in offsets:
                            dx = base_x + offset_x - center[0]
                            dy = base_y + offset_y - center[1]
                            if not mask & bit and dx * dx + dy * dy <= radius * radius:
                                mask |= bit
                            bit <<= 1
                    if mask:
                        masks[row + pixel_x] = mask
    return masks


def _independent_gray_sha256(ink: dict[str, Any]) -> str:
    if ink["background_rgb"] != [255, 255, 255] or len(ink["styles"]) != 1:
        raise PackAuditError("unsupported_ink_profile")
    style = ink["styles"][0]
    if style["rgb"] != [0, 0, 0] or style["alpha"] != 255:
        raise PackAuditError("unsupported_ink_color")
    if any(item["style_id"] != style["style_id"] for item in ink["primitives"]):
        raise PackAuditError("multiple_primitive_styles_not_supported")
    masks = _independent_masks(ink["primitives"], style, include_round_joins=True)
    sample_count = AA_SAMPLES[style["antialias_profile"]]
    total_samples = sample_count * sample_count
    gray = bytearray([255]) * (CANVAS_SIZE * CANVAS_SIZE)
    for position, mask in masks.items():
        gray[position] = 255 - _round_half_even(255 * mask.bit_count(), total_samples)
    return _sha256(bytes(gray))


def _eight_neighbor_adjacent(left: set[int], right: set[int]) -> bool:
    smaller, other = (left, right) if len(left) <= len(right) else (right, left)
    for position in smaller:
        x, y = position % CANVAS_SIZE, position // CANVAS_SIZE
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx or dy) and 0 <= x + dx < CANVAS_SIZE and 0 <= y + dy < CANVAS_SIZE:
                    if (y + dy) * CANVAS_SIZE + x + dx in other:
                        return True
    return False


def _rebuild_support_certificate(ink: dict[str, Any]) -> dict[str, Any]:
    if len(ink["styles"]) != 1:
        raise PackAuditError("support_certificate_requires_one_style")
    style = ink["styles"][0]
    supports = []
    for primitive in ink["primitives"]:
        masks = _independent_masks([primitive], style, include_round_joins=False)
        supports.append(set(masks))
    pairs = []
    for left_index, right_index in itertools.combinations(range(len(supports)), 2):
        intersection_count = len(supports[left_index] & supports[right_index])
        adjacent = False if intersection_count else _eight_neighbor_adjacent(
            supports[left_index], supports[right_index]
        )
        separated = intersection_count == 0 and not adjacent
        pairs.append({
            "eight_neighbor_adjacent": adjacent,
            "foreground_intersection_count": intersection_count,
            "left_primitive_id": ink["primitives"][left_index]["primitive_id"],
            "right_primitive_id": ink["primitives"][right_index]["primitive_id"],
            "separated": separated,
        })
    return {
        "all_separated": all(item["separated"] for item in pairs),
        "pairs": pairs,
        "primitive_foreground_pixel_counts": [
            {"pixel_count": len(supports[index]), "primitive_id": item["primitive_id"]}
            for index, item in enumerate(ink["primitives"])
        ],
        "rule": "pairwise_zero_intersection_and_no_8_neighbor_adjacency",
        "schema_version": "hidden-support-certificate-0.4",
    }


def _component_signature(
    nodes: list[tuple[str, int]],
    edges: list[tuple[int, int, int]],
    component: list[int],
) -> tuple[Any, ...]:
    local_degree = {node: 0 for node in component}
    for left, right, _order in edges:
        if left in local_degree and right in local_degree:
            local_degree[left] += 1
            local_degree[right] += 1
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for node in component:
        groups[(nodes[node][0], nodes[node][1], local_degree[node])].append(node)
    ordered_groups = [groups[key] for key in sorted(groups)]
    best: tuple[Any, ...] | None = None
    for products in itertools.product(*(itertools.permutations(group) for group in ordered_groups)):
        order = tuple(item for group in products for item in group)
        position = {node: index for index, node in enumerate(order)}
        labels = tuple(nodes[node] for node in order)
        adjacency = [[0] * len(order) for _ in order]
        for left, right, bond_order in edges:
            if left in position and right in position:
                a, b = position[left], position[right]
                adjacency[a][b] = adjacency[b][a] = bond_order
        upper = tuple(adjacency[a][b] for a in range(len(order)) for b in range(a + 1, len(order)))
        candidate = (labels, upper)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise PackAuditError("empty_graph_component")
    return best


def _graph_signature(nodes: list[tuple[str, int]], edges: list[tuple[int, int, int]]) -> tuple[Any, ...]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(nodes))}
    for left, right, _order in edges:
        if left == right or left not in adjacency or right not in adjacency:
            raise PackAuditError("invalid_graph_edge")
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(adjacency)
    components = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component = []
        unseen.remove(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(_component_signature(nodes, edges, component))
    return tuple(sorted(components))


def _interpretation_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    atom_index = {atom["atom_id"]: index for index, atom in enumerate(record["atoms"])}
    if len(atom_index) != len(record["atoms"]):
        raise PackAuditError(f"duplicate_atom_id:{record['interpretation_id']}")
    nodes = [(atom["chemical_label"], atom["formal_charge"]) for atom in record["atoms"]]
    edges = []
    for bond in record["bonds"]:
        try:
            left = atom_index[bond["begin_atom_id"]]
            right = atom_index[bond["end_atom_id"]]
        except KeyError as error:
            raise PackAuditError(f"bond_missing_atom:{record['interpretation_id']}") from error
        edges.append((left, right, bond["order"]))
    return _graph_signature(nodes, edges)


def _graph_record_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    node_index = {node["node_id"]: index for index, node in enumerate(record["nodes"])}
    if len(node_index) != len(record["nodes"]):
        raise PackAuditError(f"duplicate_node_id:{record['graph_id']}")
    nodes = [(node["chemical_label"], node["formal_charge"]) for node in record["nodes"]]
    edges = []
    for edge in record["edges"]:
        try:
            left = node_index[edge["begin_node_id"]]
            right = node_index[edge["end_node_id"]]
        except KeyError as error:
            raise PackAuditError(f"edge_missing_node:{record['graph_id']}") from error
        edges.append((left, right, edge["order"]))
    return _graph_signature(nodes, edges)


def _orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _strokes_meet(first: list[list[int]], second: list[list[int]]) -> bool:
    a, b = (tuple(first[0]), tuple(first[1]))
    c, d = (tuple(second[0]), tuple(second[1]))
    if a in (c, d) or b in (c, d):
        return True
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return ((o1 < 0 < o2 or o2 < 0 < o1) and
            (o3 < 0 < o4 or o4 < 0 < o3))


def _parallel_lanes_have_white_gap(ink: dict[str, Any]) -> bool:
    primitives = ink["primitives"]
    if len(primitives) < 2:
        return False
    style = ink["styles"][0]
    required_gap_q16 = style["width_q16"] + 2 * Q
    base_a, base_b = (tuple(point) for point in primitives[0]["points_q16"])
    vx, vy = base_b[0] - base_a[0], base_b[1] - base_a[1]
    length_squared = vx * vx + vy * vy
    for left_index in range(len(primitives)):
        left = tuple(primitives[left_index]["points_q16"][0])
        for right_index in range(left_index + 1, len(primitives)):
            right = tuple(primitives[right_index]["points_q16"][0])
            cross = abs(vx * (right[1] - left[1]) - vy * (right[0] - left[0]))
            if cross * cross <= required_gap_q16 * required_gap_q16 * length_squared:
                return False
    return True


def _image_tree_root(records: Iterable[tuple[str, str]]) -> str:
    payload = bytearray()
    for relative_path, digest in sorted(records):
        encoded = relative_path.encode("utf-8")
        payload.extend(len(encoded).to_bytes(4, "big"))
        payload.extend(encoded)
        payload.extend(bytes.fromhex(digest))
    return _domain_hash("chemgraph-hidden-v0.4/image-tree", bytes(payload))


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_tree_sha256(protocol_root: Path) -> str:
    payload = bytearray()
    for path in sorted(protocol_root.glob("*.py"), key=lambda item: item.name):
        data = path.read_bytes()
        name = path.name.encode("utf-8")
        payload.extend(len(name).to_bytes(4, "big"))
        payload.extend(name)
        payload.extend(len(data).to_bytes(8, "big"))
        payload.extend(data)
    return _domain_hash("chemgraph-hidden-v0.4/generator-source", bytes(payload))


def _protocol_bundle_sha256(protocol_root: Path) -> str:
    selected = []
    for path in protocol_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(protocol_root)
        if (path.suffix == ".py" or path.name == "README.md"
                or path.name.endswith(".schema.json")):
            selected.append((relative.as_posix(), path))
    payload = bytearray()
    for relative, path in sorted(selected):
        data = path.read_bytes()
        name = relative.encode("utf-8")
        payload.extend(len(name).to_bytes(4, "big"))
        payload.extend(name)
        payload.extend(len(data).to_bytes(8, "big"))
        payload.extend(data)
    return _domain_hash("chemgraph-hidden-v0.4/protocol-bundle", bytes(payload))


def audit_pack(
    pack_root: Path,
    *,
    replay_root: Path | None = None,
    other_pack_root: Path | None = None,
    raw_seed_hex: str | None = None,
    protocol_root: Path | None = None,
) -> dict[str, Any]:
    root = pack_root.resolve()
    protocol = (protocol_root or Path(__file__).resolve().parent).resolve()
    failures: list[str] = []

    def check(condition: bool, code: str) -> None:
        if not condition:
            failures.append(code)

    config = _load_json(root / "config.json")
    commitment = _load_json(root / "commitment.json")
    manifest = _load_json(root / "sealed_image_pack" / "manifest.json")
    pack = _load_json(root / "sealed_image_pack" / "pack.json")
    gt_index = _load_json(root / "sealed_gt" / "gt.json")
    schema_targets = (
        (config, "challenge_spec.schema.json"),
        (commitment, "commitment.schema.json"),
        (manifest, "image_pack_manifest.schema.json"),
        (gt_index, "sealed_gt_index.schema.json"),
    )
    for value, schema_name in schema_targets:
        try:
            validate_instance(value, schema_name, SCHEMA_DIR)
        except SchemaInstanceError as error:
            failures.append(f"schema:{schema_name}:{error}")

    records: dict[str, list[dict[str, Any]]] = {}
    for file_name, schema_name in JSONL_SCHEMAS.items():
        rows = _load_jsonl(root / "sealed_gt" / file_name)
        records[file_name] = rows
        for index, row in enumerate(rows):
            try:
                validate_instance(row, schema_name, SCHEMA_DIR)
            except SchemaInstanceError as error:
                failures.append(f"schema:{file_name}:{index}:{error}")
    samples = _load_jsonl(root / "sealed_image_pack" / "samples.jsonl")

    for reference in gt_index["files"].values():
        try:
            _check_artifact(root, reference)
        except PackAuditError as error:
            failures.append(str(error))

    images_by_delivery: dict[str, dict[str, Any]] = {}
    image_tree_records = []
    gray_by_delivery: dict[str, str] = {}
    for image in manifest["images"]:
        delivery_id = image["delivery_id"]
        check(delivery_id not in images_by_delivery, "duplicate_delivery_id")
        images_by_delivery[delivery_id] = image
        try:
            path = _safe_artifact(root / "sealed_image_pack", image["relative_path"])
            payload = path.read_bytes()
            check(len(payload) == image["byte_length"], "png_length_mismatch")
            check(_sha256(payload) == image["png_sha256"], "png_hash_mismatch")
            gray = _decode_canonical_gray_png(payload)
            gray_by_delivery[delivery_id] = _sha256(gray)
            image_tree_records.append((image["relative_path"], image["png_sha256"]))
        except PackAuditError as error:
            failures.append(f"png:{delivery_id}:{error}")

    observations = records["observations.jsonl"]
    inks = {row["ink_program_id"]: row for row in records["ink_programs.jsonl"]}
    transforms = {row["transform_id"]: row for row in records["transforms.jsonl"]}
    interpretations = {row["interpretation_id"]: row for row in records["interpretations.jsonl"]}
    graph_sets = {row["graph_set_id"]: row for row in records["graph_sets.jsonl"]}
    observation_ids = set()
    family_counts = Counter()
    c4_unenumerated_junction_witnesses = 0
    c4_contract_failures = 0
    parallel_lane_separation_failures = 0
    parallel_lane_separation_failures_by_family = Counter()
    support_certificate_count = 0

    for observation in observations:
        observation_id = observation["observation_id"]
        check(observation_id not in observation_ids, "duplicate_observation_id")
        observation_ids.add(observation_id)
        family_counts[observation["family_id"]] += 1
        delivery_id = observation["delivery_id"]
        check(delivery_id in images_by_delivery, "observation_missing_delivery")
        if delivery_id in images_by_delivery:
            image = images_by_delivery[delivery_id]
            check(observation["png_sha256"] == image["png_sha256"], "observation_png_hash_mismatch")
            check(observation["production_gray_sha256"] == gray_by_delivery.get(delivery_id),
                  "observation_gray_hash_mismatch")
        ink = inks.get(observation["ink_program_id"])
        check(ink is not None, "observation_missing_ink")
        check(observation["transform_id"] in transforms, "observation_missing_transform")
        graph_set = graph_sets.get(observation["graph_set_id"])
        check(graph_set is not None, "observation_missing_graph_set")
        selected_interpretations = [interpretations.get(item) for item in observation["interpretation_ids"]]
        check(all(item is not None for item in selected_interpretations), "observation_missing_interpretation")
        if ink is None or graph_set is None or any(item is None for item in selected_interpretations):
            continue
        check(_sha256(_canonical_json_bytes(ink)) == observation["ink_program_sha256"],
              "ink_program_hash_mismatch")
        try:
            check(_independent_gray_sha256(ink) == observation["production_gray_sha256"],
                  "independent_raster_mismatch")
        except PackAuditError as error:
            failures.append(f"independent_raster:{observation_id}:{error}")
        check(all(item["ink_program_id"] == ink["ink_program_id"] for item in selected_interpretations),
              "interpretation_ink_mismatch")
        rebuilt = [_interpretation_signature(item) for item in selected_interpretations]
        emitted = [_graph_record_signature(item) for item in graph_set["graphs"]]
        check(len(set(rebuilt)) == len(graph_set["graphs"]), "graph_set_wrong_cardinality")
        check(set(rebuilt) == set(emitted), "graph_set_reconstruction_mismatch")
        mapping = graph_set["interpretation_graph_map"]
        check({item["interpretation_id"] for item in mapping} == set(observation["interpretation_ids"]),
              "graph_set_mapping_incomplete")
        graph_signature_by_id = {
            graph["graph_id"]: _graph_record_signature(graph) for graph in graph_set["graphs"]
        }
        interpretation_signature_by_id = {
            item["interpretation_id"]: _interpretation_signature(item)
            for item in selected_interpretations
        }
        check(all(
            item["graph_id"] in graph_signature_by_id
            and graph_signature_by_id[item["graph_id"]]
            == interpretation_signature_by_id[item["interpretation_id"]]
            for item in mapping
        ), "graph_set_mapping_wrong")
        expected_kind = "none" if len(set(rebuilt)) == 1 else observation["ambiguity_kind"]
        check(graph_set["ambiguity_kind"] == expected_kind, "ambiguity_kind_mismatch")

        if observation["family_id"] in ("c2", "c3"):
            if not _parallel_lanes_have_white_gap(ink):
                parallel_lane_separation_failures += 1
                parallel_lane_separation_failures_by_family[observation["family_id"]] += 1
        if observation["family_id"] in ("c2", "c3", "c4"):
            support_certificate_count += 1
            certificate = observation.get("support_certificate")
            if certificate is None:
                failures.append("support_certificate_missing")
            else:
                try:
                    check(certificate == _rebuild_support_certificate(ink),
                          "support_certificate_mismatch")
                except PackAuditError as error:
                    failures.append(f"support_certificate:{observation_id}:{error}")
        else:
            check("support_certificate" not in observation, "support_certificate_on_wrong_family")
        if observation["family_id"] == "c4":
            glyph_strokes = [item["points_q16"] for item in ink["primitives"]
                             if item["primitive_type"] == "glyph_template"]
            if any(_strokes_meet(left, right)
                   for left, right in itertools.combinations(glyph_strokes, 2)):
                c4_unenumerated_junction_witnesses += 1
            glyph_primitives = [item for item in ink["primitives"]
                                if item["primitive_type"] == "glyph_template"]
            text_interpretations = [item for item in selected_interpretations
                                    if any(role["role"] == "atom_glyph"
                                           for role in item["primitive_roles"])]
            lane_interpretations = [item for item in selected_interpretations
                                    if all(role["role"] == "bond_lane"
                                           for role in item["primitive_roles"])]
            c4_valid = (
                len(ink["primitives"]) == 2
                and len(glyph_primitives) == 1
                and len(text_interpretations) == 1
                and len(lane_interpretations) == 1
                and len(text_interpretations[0]["atoms"]) == 3
                and len(text_interpretations[0]["bonds"]) == 1
                and len(lane_interpretations[0]["atoms"]) == 4
                and len(lane_interpretations[0]["bonds"]) == 2
                and glyph_primitives[0]["glyph_template_id"]
                in {"glyph-I", "glyph-l", "glyph-1", "glyph-/"}
                and text_interpretations[0]["atoms"][2]["chemical_label"]
                == "TEXT_STROKE"
            )
            if not c4_valid:
                c4_contract_failures += 1

    check(set(images_by_delivery) == {row["delivery_id"] for row in observations},
          "delivery_observation_bijection_failed")
    check(len(inks) == len(observations), "ink_observation_count_mismatch")
    check(len(transforms) == len(observations), "transform_observation_count_mismatch")
    check(len(graph_sets) == len(observations), "graph_set_observation_count_mismatch")

    expected_quotas = EXPECTED_SMOKE_QUOTAS if config["profile"] == "smoke" else EXPECTED_FULL_QUOTAS
    check(config["quotas"] == expected_quotas, "config_quota_mismatch")
    check(dict(family_counts) == expected_quotas, "observation_quota_mismatch")
    expected_observations = sum(expected_quotas.values())
    expected_invocations = 72 if config["profile"] == "smoke" else 7680
    check(len(observations) == expected_observations, "observation_count_mismatch")
    check(len(samples) == expected_invocations, "invocation_count_mismatch")
    check(commitment["unique_observation_count"] == expected_observations,
          "commitment_observation_count_mismatch")
    check(commitment["invocation_count"] == expected_invocations,
          "commitment_invocation_count_mismatch")
    check(pack.get("unique_observation_count") == expected_observations,
          "pack_observation_count_mismatch")
    check(pack.get("invocation_count") == expected_invocations, "pack_invocation_count_mismatch")
    check(pack.get("seed_is_present") is False, "pack_claims_seed_present")

    for sample in samples:
        try:
            delivery_id = Path(sample["image_relpath"]).stem
            image = images_by_delivery[delivery_id]
            check(sample["png_sha256"] == image["png_sha256"], "sample_png_hash_mismatch")
            check(sample["byte_size"] == image["byte_length"], "sample_png_length_mismatch")
        except (KeyError, TypeError):
            failures.append("sample_missing_delivery")

    attempts_by_sample: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    accepted_observation_ids = []
    for attempt in records["rejection_log.jsonl"]:
        attempts_by_sample[(attempt["family_id"], attempt["sample_ordinal"])].append(attempt)
        if attempt["accepted"]:
            accepted_observation_ids.append(attempt["terminal_observation_id"])
    for attempts in attempts_by_sample.values():
        attempts.sort(key=lambda item: item["attempt_ordinal"])
        check([item["attempt_ordinal"] for item in attempts] == list(range(len(attempts))),
              "rejection_attempt_gap")
        check(sum(bool(item["accepted"]) for item in attempts) == 1,
              "rejection_accept_count_wrong")
        check(attempts[-1]["accepted"], "rejection_accept_not_terminal")
        for attempt in attempts:
            if attempt["accepted"]:
                check(not attempt["rejection_reasons"] and attempt["terminal_observation_id"] is not None,
                      "accepted_rejection_row_invalid")
            else:
                check(bool(attempt["rejection_reasons"]) and attempt["terminal_observation_id"] is None,
                      "rejected_rejection_row_invalid")
    check(set(accepted_observation_ids) == observation_ids, "rejection_observation_bijection_failed")

    digest_checks = {
        "config_sha256": root / "config.json",
        "samples_manifest_sha256": root / "sealed_image_pack" / "samples.jsonl",
        "image_manifest_sha256": root / "sealed_image_pack" / "manifest.json",
        "gt_index_sha256": root / "sealed_gt" / "gt.json",
        "rejection_log_sha256": root / "sealed_gt" / "rejection_log.jsonl",
        "pack_metadata_sha256": root / "sealed_image_pack" / "pack.json",
    }
    for field, path in digest_checks.items():
        check(commitment[field] == _sha256(path.read_bytes()), f"commitment_mismatch:{field}")
    check(commitment["image_merkle_like_root"] == _image_tree_root(image_tree_records),
          "commitment_mismatch:image_merkle_like_root")
    check(commitment["generator_source_tree_sha256"] == _source_tree_sha256(protocol),
          "commitment_mismatch:generator_source_tree_sha256")
    check(commitment["protocol_bundle_sha256"] == _protocol_bundle_sha256(protocol),
          "commitment_mismatch:protocol_bundle_sha256")

    public_paths = [
        root / "sealed_image_pack" / "manifest.json",
        root / "sealed_image_pack" / "samples.jsonl",
        root / "sealed_image_pack" / "pack.json",
    ]
    forbidden_public_terms = (b'"family', b'"transform', b'"interpretation', b'"graph_set',
                              b'"ambiguity_kind"')
    for path in public_paths:
        payload = path.read_bytes()
        for term in forbidden_public_terms:
            check(term not in payload, f"public_gt_leak:{path.name}:{term.decode()}")
    if raw_seed_hex is not None:
        seed_bytes = raw_seed_hex.lower().encode("ascii")
        for path in (item for item in root.rglob("*") if item.is_file()):
            check(seed_bytes not in path.read_bytes(), f"raw_seed_leak:{path.relative_to(root)}")

    replay_passed: bool | None = None
    if replay_root is not None:
        replay_passed = _tree_digest(root) == _tree_digest(replay_root.resolve())
        check(replay_passed, "deterministic_replay_mismatch")

    cross_pack_overlap_count: int | None = None
    cross_pack_overlap_passed: bool | None = None
    if other_pack_root is not None:
        other_manifest = _load_json(other_pack_root.resolve() / "sealed_image_pack" / "manifest.json")
        other_hashes = {item["png_sha256"] for item in other_manifest["images"]}
        current_hashes = {item["png_sha256"] for item in manifest["images"]}
        cross_pack_overlap_count = len(current_hashes & other_hashes)
        cross_pack_overlap_passed = (
            manifest["pack_id"] != other_manifest["pack_id"] and cross_pack_overlap_count == 0
        )
        check(cross_pack_overlap_passed, "cross_pack_png_overlap")

    semantic_blockers = []
    if parallel_lane_separation_failures:
        semantic_blockers.append("parallel_lane_white_gap_not_proved")
    if c4_unenumerated_junction_witnesses:
        semantic_blockers.append("c4_graph_candidate_set_not_complete")
    if c4_contract_failures:
        semantic_blockers.append("c4_single_stroke_contract_failed")
    artifact_integrity_passed = not failures
    status = "passed" if artifact_integrity_passed and not semantic_blockers else "no_go"
    return {
        "artifact_integrity_passed": artifact_integrity_passed,
        "c4_unenumerated_junction_witnesses": c4_unenumerated_junction_witnesses,
        "c4_contract_failures": c4_contract_failures,
        "cross_pack_overlap_passed": cross_pack_overlap_passed,
        "cross_pack_png_overlap_count": cross_pack_overlap_count,
        "deterministic_replay_passed": replay_passed,
        "failure_codes": sorted(set(failures)),
        "family_counts": dict(sorted(family_counts.items())),
        "invocation_count": len(samples),
        "observation_count": len(observations),
        "parallel_lane_separation_failures": parallel_lane_separation_failures,
        "parallel_lane_separation_failures_by_family": dict(
            sorted(parallel_lane_separation_failures_by_family.items())
        ),
        "semantic_blockers": semantic_blockers,
        "status": status,
        "support_certificate_count": support_certificate_count,
        "training_authorized": False,
        "validator": "independent-stdlib-pack-auditor-v0.4",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--other-pack-root", type=Path)
    parser.add_argument("--raw-seed-hex")
    parser.add_argument("--protocol-root", type=Path)
    args = parser.parse_args()
    result = audit_pack(
        args.pack_root,
        replay_root=args.replay_root,
        other_pack_root=args.other_pack_root,
        raw_seed_hex=args.raw_seed_hex,
        protocol_root=args.protocol_root,
    )
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
