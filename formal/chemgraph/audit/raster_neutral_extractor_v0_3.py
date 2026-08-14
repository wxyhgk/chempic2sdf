#!/usr/bin/env python3
"""Isolated PNG-bytes to neutral raster geometry extractor.

This program intentionally has no repository imports and performs no file-system
access.  Its complete input is one PNG byte stream on stdin and its complete
output is one canonical JSON document on stdout.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib


SCHEMA_VERSION = "raster-neutral-evidence-v0.3"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_DIMENSION = 256
MAX_DECOMPRESSED_BYTES = 256 * 256 * 4 + 256
ALLOWED_COLOR_TYPES = {0: (1, "gray8"), 2: (3, "rgb8"), 4: (2, "gray-alpha8"), 6: (4, "rgba8")}


class ExtractorError(ValueError):
    """A deterministic input rejection."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _decode_png(data: bytes) -> tuple[int, int, str, list[bytes]]:
    if not data.startswith(PNG_SIGNATURE):
        raise ExtractorError("invalid_png_signature")
    offset = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ExtractorError("truncated_png_chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ExtractorError("truncated_png_chunk")
        payload = data[offset + 8 : offset + 8 + length]
        recorded_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != recorded_crc:
            raise ExtractorError("png_crc_mismatch")
        if chunk_type == b"IHDR":
            if ihdr is not None or length != 13 or idat_parts:
                raise ExtractorError("invalid_ihdr_position")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            if ihdr is None or saw_iend:
                raise ExtractorError("invalid_idat_position")
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            if length != 0 or ihdr is None:
                raise ExtractorError("invalid_iend")
            saw_iend = True
            offset = chunk_end
            break
        elif chunk_type and 65 <= chunk_type[0] <= 90:
            raise ExtractorError("unknown_critical_chunk")
        offset = chunk_end
    if not saw_iend or offset != len(data):
        raise ExtractorError("missing_or_nonterminal_iend")
    if ihdr is None or not idat_parts:
        raise ExtractorError("missing_png_image_data")
    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if width < 1 or height < 1 or width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ExtractorError("unsupported_png_dimensions")
    if bit_depth != 8:
        raise ExtractorError("unsupported_png_bit_depth")
    if color_type not in ALLOWED_COLOR_TYPES:
        raise ExtractorError("unsupported_png_color_type")
    if compression != 0 or filtering != 0 or interlace != 0:
        raise ExtractorError("unsupported_png_encoding")
    channels, color_name = ALLOWED_COLOR_TYPES[color_type]
    row_bytes = width * channels
    expected_size = height * (row_bytes + 1)
    if expected_size > MAX_DECOMPRESSED_BYTES:
        raise ExtractorError("png_decompressed_size_limit")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(b"".join(idat_parts), expected_size + 1)
    raw += decompressor.flush()
    if decompressor.unused_data or len(raw) != expected_size:
        raise ExtractorError("invalid_png_decompressed_size")
    rows: list[bytes] = []
    previous = bytes(row_bytes)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        encoded = raw[cursor + 1 : cursor + 1 + row_bytes]
        cursor += row_bytes + 1
        if filter_type > 4:
            raise ExtractorError("unsupported_png_filter")
        decoded = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = decoded[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            else:
                predictor = _paeth(left, above, upper_left)
            decoded[index] = (value + predictor) & 0xFF
        previous = bytes(decoded)
        rows.append(previous)
    return width, height, color_name, rows


def _foreground(
    width: int, height: int, color_name: str, rows: list[bytes]
) -> set[tuple[int, int]]:
    channels = {"gray8": 1, "rgb8": 3, "gray-alpha8": 2, "rgba8": 4}[color_name]
    foreground: set[tuple[int, int]] = set()
    for y in range(height):
        row = rows[y]
        for x in range(width):
            values = row[x * channels : (x + 1) * channels]
            if color_name == "gray8":
                red = green = blue = values[0]
                alpha = 255
            elif color_name == "rgb8":
                red, green, blue = values
                alpha = 255
            elif color_name == "gray-alpha8":
                red = green = blue = values[0]
                alpha = values[1]
            else:
                red, green, blue, alpha = values
            # Composite over white using integer arithmetic, then BT.601 luma.
            red = (red * alpha + 255 * (255 - alpha) + 127) // 255
            green = (green * alpha + 255 * (255 - alpha) + 127) // 255
            blue = (blue * alpha + 255 * (255 - alpha) + 127) // 255
            luma = (77 * red + 150 * green + 29 * blue + 128) // 256
            if luma < 128:
                foreground.add((x, y))
    return foreground


def _row_runs(pixels: set[tuple[int, int]], height: int) -> list[list[int]]:
    by_row: dict[int, list[int]] = {}
    for x, y in pixels:
        by_row.setdefault(y, []).append(x)
    runs: list[list[int]] = []
    for y in range(height):
        xs = sorted(by_row.get(y, ()))
        if not xs:
            continue
        begin = previous = xs[0]
        for x in xs[1:]:
            if x != previous + 1:
                runs.append([y, begin, previous])
                begin = x
            previous = x
        runs.append([y, begin, previous])
    return runs


NEIGHBOR_OFFSETS = (
    (-1, -1), (0, -1), (1, -1), (1, 0),
    (1, 1), (0, 1), (-1, 1), (-1, 0),
)


def _neighbors(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = point
    return [
        (x + dx, y + dy)
        for dx, dy in NEIGHBOR_OFFSETS
        if (x + dx, y + dy) in pixels
    ]


def _connected_components(pixels: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    unseen = set(pixels)
    components: list[set[tuple[int, int]]] = []
    while unseen:
        start = min(unseen, key=lambda point: (point[1], point[0]))
        stack = [start]
        component: set[tuple[int, int]] = set()
        while stack:
            point = stack.pop()
            if point not in unseen:
                continue
            unseen.remove(point)
            component.add(point)
            stack.extend(_neighbors(point, unseen))
        components.append(component)
    components.sort(key=lambda item: (min(y for _, y in item), min(x for x, _ in item)))
    return components


def _transition_count(neighbors: list[bool]) -> int:
    return sum(
        not neighbors[index] and neighbors[(index + 1) % 8]
        for index in range(8)
    )


def _thin(source: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pixels = set(source)
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            remove: list[tuple[int, int]] = []
            for x, y in sorted(pixels, key=lambda point: (point[1], point[0])):
                ring = [(x + dx, y + dy) in pixels for dx, dy in NEIGHBOR_OFFSETS]
                count = sum(ring)
                if count < 2 or count > 6 or _transition_count(ring) != 1:
                    continue
                north, east, south, west = ring[1], ring[3], ring[5], ring[7]
                if phase == 0:
                    protected = (north and east and south) or (east and south and west)
                else:
                    protected = (north and east and west) or (north and south and west)
                if not protected:
                    remove.append((x, y))
            if remove:
                pixels.difference_update(remove)
                changed = True
    return pixels


def _event_regions(
    skeleton: set[tuple[int, int]], predicate
) -> list[set[tuple[int, int]]]:
    return _connected_components({point for point in skeleton if predicate(len(_neighbors(point, skeleton)))})


def _representative(region: set[tuple[int, int]]) -> list[int]:
    count = len(region)
    sum_x = sum(point[0] for point in region)
    sum_y = sum(point[1] for point in region)
    return [(sum_x + count // 2) // count, (sum_y + count // 2) // count]


def _simplify_polyline(polyline: list[tuple[int, int]]) -> list[list[int]]:
    if len(polyline) <= 2:
        return [[x, y] for x, y in polyline]
    output = [polyline[0]]
    prior_direction: tuple[int, int] | None = None
    for index in range(1, len(polyline)):
        dx = polyline[index][0] - polyline[index - 1][0]
        dy = polyline[index][1] - polyline[index - 1][1]
        direction = (dx, dy)
        if prior_direction is not None and direction != prior_direction:
            output.append(polyline[index - 1])
        prior_direction = direction
    output.append(polyline[-1])
    deduplicated: list[tuple[int, int]] = []
    for point in output:
        if not deduplicated or deduplicated[-1] != point:
            deduplicated.append(point)
    return [[x, y] for x, y in deduplicated]


def _trace_lanes(skeleton: set[tuple[int, int]]) -> list[dict[str, object]]:
    event_pixels = {
        point for point in skeleton if len(_neighbors(point, skeleton)) != 2
    }
    regions = _connected_components(event_pixels)
    region_by_pixel = {
        point: region_id for region_id, region in enumerate(regions) for point in region
    }
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    lanes: list[list[tuple[int, int]]] = []

    def edge(left: tuple[int, int], right: tuple[int, int]):
        return tuple(sorted((left, right)))

    # An event region is one neutral junction-sized pixel cluster. Internal
    # skeleton edges describe the region shape, not additional lanes.
    for region_id, region in enumerate(regions):
        for point in region:
            for neighbor in _neighbors(point, skeleton):
                if region_by_pixel.get(neighbor) == region_id:
                    visited_edges.add(edge(point, neighbor))

    for region_id, region in enumerate(regions):
        for start in sorted(region, key=lambda point: (point[1], point[0])):
            for following in sorted(_neighbors(start, skeleton), key=lambda point: (point[1], point[0])):
                if region_by_pixel.get(following) == region_id or edge(start, following) in visited_edges:
                    continue
                path = [start, following]
                visited_edges.add(edge(start, following))
                previous, current = start, following
                while current not in region_by_pixel:
                    options = [
                        point
                        for point in _neighbors(current, skeleton)
                        if point != previous and edge(current, point) not in visited_edges
                    ]
                    if not options:
                        break
                    following = min(options, key=lambda point: (point[1], point[0]))
                    visited_edges.add(edge(current, following))
                    path.append(following)
                    previous, current = current, following
                lanes.append(path)

    # Components with degree two everywhere are closed loops and have no event region.
    for start in sorted(skeleton, key=lambda point: (point[1], point[0])):
        unvisited = [point for point in _neighbors(start, skeleton) if edge(start, point) not in visited_edges]
        if not unvisited:
            continue
        following = min(unvisited, key=lambda point: (point[1], point[0]))
        path = [start, following]
        visited_edges.add(edge(start, following))
        previous, current = start, following
        while True:
            options = [
                point for point in _neighbors(current, skeleton)
                if point != previous and edge(current, point) not in visited_edges
            ]
            if not options:
                break
            following = min(options, key=lambda point: (point[1], point[0]))
            visited_edges.add(edge(current, following))
            path.append(following)
            previous, current = current, following
        lanes.append(path)

    canonical = []
    for path in lanes:
        forward = tuple(path)
        reverse = tuple(reversed(path))
        selected = min(forward, reverse)
        canonical.append(selected)
    canonical = sorted(set(canonical), key=lambda path: tuple((y, x) for x, y in path))
    return [
        {
            "id": index,
            "polyline": _simplify_polyline(list(path)),
            "support_skeleton_pixels": len(path),
        }
        for index, path in enumerate(canonical)
    ]


def extract(data: bytes) -> dict[str, object]:
    width, height, color_name, rows = _decode_png(data)
    foreground = _foreground(width, height, color_name, rows)
    components = _connected_components(foreground)
    skeleton = _thin(foreground)
    terminal_regions = _event_regions(skeleton, lambda degree: degree <= 1)
    branch_regions = _event_regions(skeleton, lambda degree: degree >= 3)
    component_records = []
    for component_id, component in enumerate(components):
        xs = [point[0] for point in component]
        ys = [point[1] for point in component]
        component_records.append(
            {
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "id": component_id,
                "pixel_count": len(component),
            }
        )
    terminal_events = sorted(
        (_representative(region) for region in terminal_regions), key=lambda point: (point[1], point[0])
    )
    branch_events = sorted(
        (_representative(region) for region in branch_regions), key=lambda point: (point[1], point[0])
    )
    return {
        "components": component_records,
        "foreground": {
            "pixel_count": len(foreground),
            "row_runs": _row_runs(foreground, height),
        },
        "image": {
            "color_type": color_name,
            "height": height,
            "threshold_rule": "alpha_composite_white_bt601_luma_lt_128",
            "width": width,
        },
        "lanes": _trace_lanes(skeleton),
        "schema_version": SCHEMA_VERSION,
        "skeleton": {
            "branch_events": branch_events,
            "intersection_regions": [
                sorted(([x, y] for x, y in region), key=lambda point: (point[1], point[0]))
                for region in branch_regions
            ],
            "pixels": [[x, y] for x, y in sorted(skeleton, key=lambda point: (point[1], point[0]))],
            "terminal_events": terminal_events,
        },
        "status": "ok",
        "text_mask": {"row_runs": [], "status": "not_inferred"},
        "unclassified_components": [],
    }


def main() -> int:
    try:
        evidence = extract(sys.stdin.buffer.read())
    except (ExtractorError, struct.error, zlib.error) as error:
        evidence = {
            "reject_reason": str(error),
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
        }
    sys.stdout.buffer.write(_canonical_bytes(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
