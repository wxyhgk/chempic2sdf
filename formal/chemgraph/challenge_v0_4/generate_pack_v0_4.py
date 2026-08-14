#!/usr/bin/env python3
"""Generate a deterministic generator-first ChemGraph hidden challenge pack.

Only Python's standard library is used.  This module does not import Scene,
legacy generators, graph canonicalizers, or future extractor code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

try:
    from .rasterizer_v0_4 import (
        AA_SAMPLES,
        CANVAS_SIZE,
        Q,
        LineStyle,
        Segment,
        png_chunk_types,
        rasterize_png,
        rasterize_support,
    )
except ImportError:
    from rasterizer_v0_4 import (  # type: ignore
        AA_SAMPLES,
        CANVAS_SIZE,
        Q,
        LineStyle,
        Segment,
        png_chunk_types,
        rasterize_png,
        rasterize_support,
    )


SCHEMA_VERSION = "chemgraph-hidden-generator-v0.4"
CHALLENGE_ID = "chemgraph-hidden-v0-4"
MARGIN_PX = 64
MAX_ATTEMPTS = 32
FULL_QUOTAS = {
    "s1": 512, "s2": 512, "s3": 512, "s4": 512,
    "s5": 512, "s6": 512, "s7": 512, "s8": 512,
    "c1": 512, "c2": 384, "c3": 256, "c4": 384,
}
SMOKE_QUOTAS = {family: 4 for family in FULL_QUOTAS}
COLLISION_FAMILIES = frozenset(("c1", "c2", "c3", "c4"))
ROTATIONS = (
    ((1, 1), (0, 1)),
    ((3, 5), (4, 5)),
    ((5, 13), (12, 13)),
    ((8, 17), (15, 17)),
    ((0, 1), (1, 1)),
    ((-3, 5), (4, 5)),
    ((5, 13), (-12, 13)),
    ((-8, 17), (-15, 17)),
)
SCALES = ((1, 2), (5, 8), (3, 4), (1, 1), (5, 4), (3, 2), (2, 1))
REFLECTIONS = ("none", "x", "y")
CAPS = ("butt", "round", "square")
JOINS = ("round",)
AA_PROFILES = tuple(AA_SAMPLES)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def domain_hash(domain: str, payload: bytes) -> str:
    prefix = domain.encode("ascii")
    framed = len(prefix).to_bytes(4, "big") + prefix + len(payload).to_bytes(8, "big") + payload
    return sha256_bytes(framed)


class CounterPRNG:
    """Domain-separated HMAC-SHA256 counter stream with unbiased randbelow."""

    def __init__(self, seed: bytes, domain: str):
        self._seed = seed
        self._domain = domain.encode("ascii")
        self.counter = 0
        self._buffer = b""

    def _block(self) -> bytes:
        message = (b"chemgraph-hidden-v0.4/prng\0" +
                   len(self._domain).to_bytes(2, "big") + self._domain +
                   self.counter.to_bytes(8, "big"))
        self.counter += 1
        return hmac.new(self._seed, message, hashlib.sha256).digest()

    def bytes(self, count: int) -> bytes:
        while len(self._buffer) < count:
            self._buffer += self._block()
        result, self._buffer = self._buffer[:count], self._buffer[count:]
        return result

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        byte_count = max(1, (upper.bit_length() + 7) // 8)
        ceiling = 1 << (8 * byte_count)
        limit = ceiling - ceiling % upper
        while True:
            value = int.from_bytes(self.bytes(byte_count), "big")
            if value < limit:
                return value % upper

    def randint(self, lower: int, upper: int) -> int:
        if upper < lower:
            raise ValueError("empty integer interval")
        return lower + self.randbelow(upper - lower + 1)


def shuffle_in_place(values: list[Any], rng: CounterPRNG) -> None:
    for index in range(len(values) - 1, 0, -1):
        chosen = rng.randbelow(index + 1)
        values[index], values[chosen] = values[chosen], values[index]


def div_round_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return sign * quotient


@dataclass(frozen=True)
class Transform:
    reflection: str
    cos: tuple[int, int]
    sin: tuple[int, int]
    scale: tuple[int, int]
    tx_q16: int
    ty_q16: int

    def point(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        if self.reflection == "x":
            y = -y
        elif self.reflection == "y":
            x = -x
        elif self.reflection != "none":
            raise ValueError(f"bad reflection: {self.reflection}")
        common = self.cos[1] * self.sin[1]
        rotated_x = div_round_signed(
            x * self.cos[0] * self.sin[1] - y * self.sin[0] * self.cos[1], common
        )
        rotated_y = div_round_signed(
            x * self.sin[0] * self.cos[1] + y * self.cos[0] * self.sin[1], common
        )
        scaled_x = div_round_signed(rotated_x * self.scale[0], self.scale[1])
        scaled_y = div_round_signed(rotated_y * self.scale[0], self.scale[1])
        return scaled_x + self.tx_q16, scaled_y + self.ty_q16


@dataclass(frozen=True)
class Primitive:
    start: tuple[int, int]
    end: tuple[int, int]
    glyph_template_id: str | None = None


@dataclass(frozen=True)
class AtomSpec:
    point: tuple[int, int]
    label: str = "C"


@dataclass(frozen=True)
class BondSpec:
    begin: int
    end: int
    order: int
    primitive_indices: tuple[int, ...]


@dataclass(frozen=True)
class InterpretationSpec:
    atoms: tuple[AtomSpec, ...]
    bonds: tuple[BondSpec, ...]
    primitive_roles: tuple[tuple[str, int | None], ...]
    decode_kind: str


@dataclass(frozen=True)
class Motif:
    primitives: tuple[Primitive, ...]
    interpretations: tuple[InterpretationSpec, ...]
    ambiguity_kind: str


def px(x: int, y: int) -> tuple[int, int]:
    return x * Q, y * Q


def _strict_interpretation(nodes: list[tuple[int, int]], edges: list[tuple[int, int, int, int]]) -> InterpretationSpec:
    atoms = tuple(AtomSpec(point) for point in nodes)
    bonds = tuple(BondSpec(begin, end, order, (primitive,))
                  for begin, end, order, primitive in edges)
    roles = tuple(("bond_lane", edge_index) for edge_index in range(len(edges)))
    return InterpretationSpec(atoms, bonds, roles, "strict_graph")


def build_motif(family: str, rng: CounterPRNG, width_px: int) -> Motif:
    length = rng.randint(120, 220)
    half = length // 2
    if family == "s1":
        nodes = [px(-half, 0), px(half, 0)]
        primitives = (Primitive(nodes[0], nodes[1]),)
        return Motif(primitives, (_strict_interpretation(nodes, [(0, 1, 1, 0)]),), "none")
    if family == "s2":
        nodes = [px(-half, 20), px(0, -20), px(half, 45)]
        primitives = (Primitive(nodes[0], nodes[1]), Primitive(nodes[1], nodes[2]))
        return Motif(primitives, (_strict_interpretation(nodes, [(0, 1, 1, 0), (1, 2, 1, 1)]),), "none")
    if family == "s3":
        nodes = [px(0, 0), px(-half, 15), px(half, -half), px(half, half)]
        primitives = tuple(Primitive(nodes[0], nodes[index]) for index in range(1, 4))
        edges = [(0, index, 1, index - 1) for index in range(1, 4)]
        return Motif(primitives, (_strict_interpretation(nodes, edges),), "none")
    if family == "s4":
        nodes = [px(0, 0), px(-half, -half), px(half, -half + 25),
                 px(half - 20, half), px(-half, half - 35)]
        primitives = tuple(Primitive(nodes[0], nodes[index]) for index in range(1, 5))
        edges = [(0, index, 1, index - 1) for index in range(1, 5)]
        return Motif(primitives, (_strict_interpretation(nodes, edges),), "none")
    if family == "s5":
        if rng.randbelow(2) == 0:
            nodes = [px(-half, half // 2), px(0, -half), px(half, half // 2)]
        else:
            nodes = [px(-half, -half), px(half, -half + 10),
                     px(half - 20, half), px(-half, half - 25)]
        primitives = tuple(Primitive(nodes[index], nodes[(index + 1) % len(nodes)])
                           for index in range(len(nodes)))
        edges = [(index, (index + 1) % len(nodes), 1, index) for index in range(len(nodes))]
        return Motif(primitives, (_strict_interpretation(nodes, edges),), "none")
    if family == "s6":
        nodes = [px(-half, -70), px(0, -25), px(-20, 70), px(half, 25)]
        primitives = (Primitive(nodes[0], nodes[1]), Primitive(nodes[2], nodes[3]))
        return Motif(primitives, (_strict_interpretation(nodes, [(0, 1, 1, 0), (2, 3, 1, 1)]),), "none")
    if family == "s7":
        nodes = [px(-half, 0), px(-half + 60, -15), px(20, 70), px(half, -30)]
        primitives = tuple(Primitive(nodes[index], nodes[index + 1]) for index in range(3))
        edges = [(index, index + 1, 1, index) for index in range(3)]
        return Motif(primitives, (_strict_interpretation(nodes, edges),), "none")
    if family == "s8":
        gap = rng.randint(12, 28)
        nodes = [px(-half, 0), px(-gap, 0), px(gap, -half), px(gap, half)]
        primitives = (Primitive(nodes[0], nodes[1]), Primitive(nodes[2], nodes[3]))
        return Motif(primitives, (_strict_interpretation(nodes, [(0, 1, 1, 0), (2, 3, 1, 1)]),), "none")
    if family == "c1":
        endpoints = [px(-half, 0), px(half, 0), px(0, -half), px(0, half)]
        center = px(0, 0)
        primitives = (Primitive(endpoints[0], endpoints[1]), Primitive(endpoints[2], endpoints[3]))
        disconnected = _strict_interpretation(endpoints, [(0, 1, 1, 0), (2, 3, 1, 1)])
        junction_atoms = tuple(AtomSpec(point) for point in [center] + endpoints)
        junction_bonds = tuple(BondSpec(0, index + 1, 1, (index // 2,)) for index in range(4))
        junction = InterpretationSpec(junction_atoms, junction_bonds,
                                      (("bond_lane", 0), ("bond_lane", 2)), "graph_ambiguity")
        disconnected = InterpretationSpec(disconnected.atoms, disconnected.bonds,
                                           disconnected.primitive_roles, "graph_ambiguity")
        return Motif(primitives, (disconnected, junction), "crossing_junction")
    if family in ("c2", "c3"):
        lane_count = 2 if family == "c2" else 3
        # Geometry can be scaled down to 1/2 while stroke width stays fixed.
        # Reserve >= 3 final pixels beyond the width before the exact raster
        # support gate; the gate remains authoritative for rotated subpixels.
        minimum_base_spacing = 2 * (width_px + 3)
        spacing = rng.randint(minimum_base_spacing, minimum_base_spacing + 10)
        offsets = ([(-spacing // 2), (spacing - spacing // 2)] if lane_count == 2
                   else [-spacing, 0, spacing])
        primitives = tuple(Primitive(px(-half, offset), px(half, offset)) for offset in offsets)
        bundle_atoms = (AtomSpec(px(-half, 0)), AtomSpec(px(half, 0)))
        bundle_bond = (BondSpec(0, 1, lane_count, tuple(range(lane_count))),)
        bundle = InterpretationSpec(bundle_atoms, bundle_bond,
                                    tuple(("bond_lane", 0) for _ in primitives), "graph_ambiguity")
        single_atoms: list[AtomSpec] = []
        single_bonds: list[BondSpec] = []
        for index, offset in enumerate(offsets):
            single_atoms.extend((AtomSpec(px(-half, offset)), AtomSpec(px(half, offset))))
            single_bonds.append(BondSpec(index * 2, index * 2 + 1, 1, (index,)))
        singles = InterpretationSpec(tuple(single_atoms), tuple(single_bonds),
                                     tuple(("bond_lane", index) for index in range(lane_count)),
                                     "graph_ambiguity")
        return Motif(primitives, (bundle, singles), "parallel_multiplicity")
    if family == "c4":
        labels = ("I", "l", "1", "/")
        glyph = labels[rng.randbelow(len(labels))]
        x0, x1 = -half, half
        chemical = Primitive(px(x0, -80), px(x1, -80))
        character_strokes = {
            "I": ((0, -35), (0, 35)),
            "l": ((-4, -35), (-4, 35)),
            "1": ((8, -35), (8, 35)),
            "/": ((-22, 35), (22, -35)),
        }
        start, end = character_strokes[glyph]
        glyph_primitives = (Primitive(px(*start), px(*end), f"glyph-{glyph}"),)
        primitives = (chemical,) + glyph_primitives
        text_atoms = (AtomSpec(chemical.start), AtomSpec(chemical.end),
                      AtomSpec(px(0, 0), "TEXT_STROKE"))
        text_bonds = (BondSpec(0, 1, 1, (0,)),)
        text_roles = (("bond_lane", 0),) + tuple(("atom_glyph", 2) for _ in glyph_primitives)
        text = InterpretationSpec(text_atoms, text_bonds, text_roles, "stroke_role_ambiguity")
        lane_atoms: list[AtomSpec] = []
        lane_bonds: list[BondSpec] = []
        for index, primitive in enumerate(primitives):
            lane_atoms.extend((AtomSpec(primitive.start), AtomSpec(primitive.end)))
            lane_bonds.append(BondSpec(2 * index, 2 * index + 1, 1, (index,)))
        lanes = InterpretationSpec(tuple(lane_atoms), tuple(lane_bonds),
                                   tuple(("bond_lane", index) for index in range(len(primitives))),
                                   "stroke_role_ambiguity")
        return Motif(primitives, (text, lanes), "text_or_lane")
    raise ValueError(f"unknown family: {family}")


def choose_style(global_quartet: int) -> tuple[LineStyle, str]:
    # P1 freezes round join only: 9 widths * 3 caps * 3 AA = 81 buckets.
    # Multiplication by 80 permutes the buckets because gcd(80, 81) = 1.
    bucket_index = (global_quartet * 80) % 81
    width_px = bucket_index % 9 + 1
    cap = CAPS[(bucket_index // 9) % len(CAPS)]
    join = "round"
    aa = AA_PROFILES[(bucket_index // 27) % len(AA_PROFILES)]
    style = LineStyle(width_px * Q, cap, aa, join)
    bucket = f"w{width_px}-{cap}-{join}-{aa.replace('_', '-')}"
    return style, bucket


def choose_transform(seed: bytes, pack_id: str, family: str, quartet: int,
                     variant: int, attempt: int) -> Transform:
    rng = CounterPRNG(seed, f"{pack_id}/{family}/{quartet}/{variant}/{attempt}/transform")
    if variant == 0:
        reflection, rotation, scale = "none", ROTATIONS[0], (1, 1)
    else:
        reflection = REFLECTIONS[(quartet + variant + attempt) % len(REFLECTIONS)]
        rotation = ROTATIONS[(quartet * 3 + variant + attempt) % len(ROTATIONS)]
        scale = SCALES[(quartet * 5 + variant + attempt) % len(SCALES)]
    tx = (CANVAS_SIZE // 2 + rng.randint(-32, 32)) * Q
    ty = (CANVAS_SIZE // 2 + rng.randint(-32, 32)) * Q
    return Transform(reflection, rotation[0], rotation[1], scale, tx, ty)


def transform_record(transform_id: str, transform: Transform) -> dict[str, Any]:
    return {
        "schema_version": "hidden-transform-0.4",
        "transform_id": transform_id,
        "transform_order": "reflection_then_rotation_then_scale_then_translation",
        "reflection": transform.reflection,
        "rotation": {"cos": list(transform.cos), "sin": list(transform.sin)},
        "scale": list(transform.scale),
        "translation_q16": [transform.tx_q16, transform.ty_q16],
    }


def valid_transformed(primitives: tuple[Primitive, ...], transform: Transform,
                      width_q16: int) -> tuple[bool, list[tuple[tuple[int, int], tuple[int, int]]]]:
    transformed = [(transform.point(item.start), transform.point(item.end)) for item in primitives]
    margin = max(MARGIN_PX, 4 * ((width_q16 + Q - 1) // Q)) * Q
    limit = CANVAS_SIZE * Q - margin
    for start, end in transformed:
        if start == end:
            return False, transformed
        for x, y in (start, end):
            if not (margin <= x <= limit and margin <= y <= limit):
                return False, transformed
    return True, transformed


def separated_support_certificate(segments: list[Segment], style: LineStyle) -> tuple[bool, dict[str, Any]]:
    """Prove raster-visible primitive supports are disjoint and non-adjacent."""
    supports = [rasterize_support([segment], style) for segment in segments]
    pair_records = []
    all_separated = True
    for left_index in range(len(supports)):
        for right_index in range(left_index + 1, len(supports)):
            left = supports[left_index]
            right = supports[right_index]
            intersection_count = len(left & right)
            eight_neighbor = False
            smaller, other = (left, right) if len(left) <= len(right) else (right, left)
            if not intersection_count:
                for position in smaller:
                    x, y = position % CANVAS_SIZE, position // CANVAS_SIZE
                    found = False
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < CANVAS_SIZE and 0 <= ny < CANVAS_SIZE and
                                    ny * CANVAS_SIZE + nx in other):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        eight_neighbor = True
                        break
            separated = intersection_count == 0 and not eight_neighbor
            all_separated = all_separated and separated
            pair_records.append({
                "left_primitive_id": segments[left_index].segment_id,
                "right_primitive_id": segments[right_index].segment_id,
                "foreground_intersection_count": intersection_count,
                "eight_neighbor_adjacent": eight_neighbor,
                "separated": separated,
            })
    return all_separated, {
        "schema_version": "hidden-support-certificate-0.4",
        "rule": "pairwise_zero_intersection_and_no_8_neighbor_adjacency",
        "primitive_foreground_pixel_counts": [{
            "primitive_id": segment.segment_id,
            "pixel_count": len(supports[index]),
        } for index, segment in enumerate(segments)],
        "pairs": pair_records,
        "all_separated": all_separated,
    }


def make_interpretation(record_index: int, interpretation_index: int, ink_id: str,
                        spec: InterpretationSpec, transform: Transform) -> dict[str, Any]:
    interpretation_id = f"int-{record_index:08d}-{interpretation_index:02d}"
    atoms = [{
        "atom_id": f"a-{index:04d}",
        "chemical_label": atom.label,
        "formal_charge": 0,
        "position_q16": list(transform.point(atom.point)),
    } for index, atom in enumerate(spec.atoms)]
    bonds = [{
        "bond_id": f"b-{index:04d}",
        "begin_atom_id": f"a-{bond.begin:04d}",
        "end_atom_id": f"a-{bond.end:04d}",
        "order": bond.order,
        "primitive_ids": [f"p-{item:06d}" for item in bond.primitive_indices],
    } for index, bond in enumerate(spec.bonds)]
    roles = [{
        "primitive_id": f"p-{index:06d}",
        "role": role,
        "owner_id": (None if owner is None else
                     (f"a-{owner:04d}" if role == "atom_glyph" else f"b-{owner:04d}")),
    } for index, (role, owner) in enumerate(spec.primitive_roles)]
    return {
        "schema_version": "hidden-interpretation-0.4",
        "interpretation_id": interpretation_id,
        "ink_program_id": ink_id,
        "expected_decode_kind": spec.decode_kind,
        "atoms": atoms,
        "bonds": bonds,
        "primitive_roles": roles,
    }


def graph_set_record(record_index: int, observation_id: str, ambiguity_kind: str,
                     interpretations: list[dict[str, Any]]) -> dict[str, Any]:
    graphs = []
    mapping = []
    for graph_index, interpretation in enumerate(interpretations):
        graph_id = f"g-{graph_index:04d}"
        atoms = interpretation["atoms"]
        bonds = interpretation["bonds"]
        graphs.append({
            "graph_id": graph_id,
            "nodes": [{
                "node_id": f"n-{index:04d}",
                "chemical_label": atom["chemical_label"],
                "formal_charge": atom["formal_charge"],
            } for index, atom in enumerate(atoms)],
            "edges": [{
                "begin_node_id": f"n-{int(bond['begin_atom_id'][2:]):04d}",
                "end_node_id": f"n-{int(bond['end_atom_id'][2:]):04d}",
                "order": bond["order"],
            } for bond in bonds],
        })
        mapping.append({"interpretation_id": interpretation["interpretation_id"],
                        "graph_id": graph_id})
    return {
        "schema_version": "hidden-graph-set-0.4",
        "graph_set_id": f"gs-{record_index:08d}",
        "observation_id": observation_id,
        "ambiguity_kind": ambiguity_kind,
        "graphs": graphs,
        "interpretation_graph_map": mapping,
    }


def source_tree_sha256(root: Path) -> str:
    payload = bytearray()
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        data = path.read_bytes()
        name = path.name.encode("utf-8")
        payload.extend(len(name).to_bytes(4, "big"))
        payload.extend(name)
        payload.extend(len(data).to_bytes(8, "big"))
        payload.extend(data)
    return domain_hash("chemgraph-hidden-v0.4/generator-source", bytes(payload))


def protocol_bundle_sha256(root: Path) -> str:
    """Commit code, README, root schemas, and the independent schema bundle."""
    selected = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (path.suffix == ".py" or path.name == "README.md" or
                path.name.endswith(".schema.json")):
            selected.append((relative.as_posix(), path))
    payload = bytearray()
    for relative, path in sorted(selected):
        data = path.read_bytes()
        name = relative.encode("utf-8")
        payload.extend(len(name).to_bytes(4, "big"))
        payload.extend(name)
        payload.extend(len(data).to_bytes(8, "big"))
        payload.extend(data)
    return domain_hash("chemgraph-hidden-v0.4/protocol-bundle", bytes(payload))


def merkle_like_root(records: list[tuple[str, str]]) -> str:
    payload = bytearray()
    for path, digest in sorted(records):
        encoded = path.encode("utf-8")
        payload.extend(len(encoded).to_bytes(4, "big"))
        payload.extend(encoded)
        payload.extend(bytes.fromhex(digest))
    return domain_hash("chemgraph-hidden-v0.4/image-tree", bytes(payload))


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def artifact(path: Path, root: Path, media_type: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_bytes(data),
        "byte_length": len(data),
        "media_type": media_type,
    }


def generate(seed: bytes, output: Path, pack_id: str, smoke: bool) -> dict[str, Any]:
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    if pack_id not in ("pack-a", "pack-b"):
        raise ValueError("pack_id must be pack-a or pack-b")
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
        output.rmdir()
    stage = output.with_name(f".{output.name}.stage-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    image_dir = stage / "sealed_image_pack" / "images"
    gt_dir = stage / "sealed_gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    quotas = SMOKE_QUOTAS if smoke else FULL_QUOTAS
    config = {
        "schema_version": SCHEMA_VERSION,
        "challenge_id": CHALLENGE_ID,
        "pack_id": pack_id,
        "profile": "smoke" if smoke else "full",
        "canvas": [CANVAS_SIZE, CANVAS_SIZE],
        "q_format": "Q16.16",
        "prng": "hmac-sha256-counter-v1",
        "quotas": quotas,
        "line_width_pixels": list(range(1, 10)),
        "caps": list(CAPS),
        "joins": list(JOINS),
        "antialias_profiles": list(AA_PROFILES),
        "rotations": [[list(cos), list(sin)] for cos, sin in ROTATIONS],
        "reflections": list(REFLECTIONS),
        "scales": [list(item) for item in SCALES],
        "maximum_attempts": MAX_ATTEMPTS,
    }
    config_bytes = canonical_json_bytes(config)
    config_sha = sha256_bytes(config_bytes)
    seed_commitment = domain_hash(
        f"chemgraph-hidden-v0.4/{pack_id}/seed-commitment", seed
    )
    source_sha = source_tree_sha256(Path(__file__).resolve().parent)
    protocol_sha = protocol_bundle_sha256(Path(__file__).resolve().parent)

    observations: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    seen_png: set[str] = set()
    global_quartet = 0
    started = time.perf_counter()

    for family, quota in quotas.items():
        if quota % 4:
            raise ValueError(f"family quota must form quartets: {family}={quota}")
        for quartet in range(quota // 4):
            style, style_bucket = choose_style(global_quartet)
            global_quartet += 1
            base_rng = CounterPRNG(seed, f"{pack_id}/{family}/{quartet}/base")
            motif = build_motif(family, base_rng, style.width_q16 // Q)
            for variant in range(4):
                sample_ordinal = quartet * 4 + variant
                accepted = False
                for attempt in range(MAX_ATTEMPTS):
                    transform = choose_transform(seed, pack_id, family, quartet, variant, attempt)
                    is_valid, transformed = valid_transformed(motif.primitives, transform, style.width_q16)
                    reasons: list[str] = []
                    support_certificate = None
                    if not is_valid:
                        reasons.append("out_of_bounds")
                    if not reasons:
                        segments = [Segment(f"p-{index:06d}", start[0], start[1], end[0], end[1])
                                    for index, (start, end) in enumerate(transformed)]
                        if family in ("c2", "c3", "c4"):
                            support_valid, support_certificate = separated_support_certificate(
                                segments, style
                            )
                            if not support_valid:
                                reasons.append("invalid_exact_collision")
                    if not reasons:
                        png, gray_sha = rasterize_png(segments, style)
                        if png_chunk_types(png) != ["IHDR", "IDAT", "IEND"]:
                            reasons.append("rasterization_failure")
                        png_sha = sha256_bytes(png)
                        if png_sha in seen_png:
                            reasons.append("duplicate_observation")
                    rejection_records.append({
                        "family_id": family,
                        "sample_ordinal": sample_ordinal,
                        "quartet_ordinal": quartet,
                        "variant": variant,
                        "attempt_ordinal": attempt,
                        "stream_counter": 0,
                        "accepted": not reasons,
                        "rejection_reasons": reasons,
                        "terminal_observation_id": None,
                    })
                    if reasons:
                        continue
                    observation_index = len(observations)
                    observation_id = f"obs-{observation_index:08d}"
                    ink_id = f"ink-{observation_index:08d}"
                    transform_id = f"tr-{observation_index:08d}"
                    interpretations = [make_interpretation(
                        observation_index, index, ink_id, spec, transform
                    ) for index, spec in enumerate(motif.interpretations)]
                    primitive_records = [{
                        "primitive_id": f"p-{index:06d}",
                        "primitive_type": ("glyph_template" if primitive.glyph_template_id else "line"),
                        "style_id": "s-0000",
                        "points_q16": [list(transformed[index][0]), list(transformed[index][1])],
                        "closed": False,
                        "glyph_template_id": primitive.glyph_template_id,
                    } for index, primitive in enumerate(motif.primitives)]
                    ink_program = {
                        "schema_version": "hidden-ink-program-0.4",
                        "ink_program_id": ink_id,
                        "transform_id": transform_id,
                        "canvas": {"width": CANVAS_SIZE, "height": CANVAS_SIZE,
                                   "origin": "top_left", "x_axis": "right",
                                   "y_axis": "down", "q_format": "Q16.16"},
                        "background_rgb": [255, 255, 255],
                        "styles": [{
                            "style_id": "s-0000", "width_q16": style.width_q16,
                            "cap": style.cap, "join": style.join,
                            "rgb": [0, 0, 0], "alpha": 255,
                            "antialias_profile": style.antialias_profile,
                        }],
                        "primitives": primitive_records,
                    }
                    graph_set = graph_set_record(
                        observation_index, observation_id, motif.ambiguity_kind, interpretations
                    )
                    rejection_records[-1]["terminal_observation_id"] = observation_id
                    observations.append({
                        "observation_index": observation_index,
                        "observation_id": observation_id,
                        "family": family,
                        "quartet": quartet,
                        "variant": variant,
                        "style_bucket": style_bucket,
                        "png": png,
                        "png_sha256": png_sha,
                        "gray_sha256": gray_sha,
                        "ink_program": ink_program,
                        "ink_program_sha256": sha256_bytes(canonical_json_bytes(ink_program)),
                        "transform": transform_record(transform_id, transform),
                        "interpretations": interpretations,
                        "graph_set": graph_set,
                        "support_certificate": support_certificate,
                    })
                    seen_png.add(png_sha)
                    accepted = True
                    break
                if not accepted:
                    raise RuntimeError(f"exhausted attempts for {family}/{sample_ordinal}")

    delivery_order = list(range(len(observations)))
    shuffle_in_place(delivery_order, CounterPRNG(seed, f"{pack_id}/delivery-order"))
    delivery_for_observation: dict[int, str] = {}
    image_manifest_records: list[dict[str, Any]] = []
    image_hash_records: list[tuple[str, str]] = []
    for delivery_number, observation_index in enumerate(delivery_order):
        observation = observations[observation_index]
        delivery_id = f"img-{delivery_number:08d}"
        delivery_for_observation[observation_index] = delivery_id
        relative_path = f"images/{delivery_id}.png"
        atomic_write(image_dir / f"{delivery_id}.png", observation["png"])
        image_manifest_records.append({
            "delivery_id": delivery_id,
            "relative_path": relative_path,
            "png_sha256": observation["png_sha256"],
            "byte_length": len(observation["png"]),
            "media_type": "image/png",
            "width": CANVAS_SIZE,
            "height": CANVAS_SIZE,
        })
        image_hash_records.append((relative_path, observation["png_sha256"]))

    invocations: list[tuple[int, str]] = [(index, "primary") for index in range(len(observations))]
    invocations.extend((index, "collision-replay") for index, item in enumerate(observations)
                       if item["family"] in COLLISION_FAMILIES)
    strict_indices = [index for index, item in enumerate(observations) if item["family"].startswith("s")]
    replay_count = 8 if smoke else 512
    replay_rng = CounterPRNG(seed, f"{pack_id}/strict-replay-selection")
    shuffle_in_place(strict_indices, replay_rng)
    invocations.extend((index, "strict-replay") for index in strict_indices[:replay_count])
    shuffle_in_place(invocations, CounterPRNG(seed, f"{pack_id}/invocation-order"))
    sample_records = []
    for invocation_number, (observation_index, _kind) in enumerate(invocations):
        observation = observations[observation_index]
        delivery_id = delivery_for_observation[observation_index]
        sample_records.append({
            "schema_version": "chemgraph-hidden-sample-v0.4",
            "invocation_id": f"inv-{invocation_number:08d}",
            "image_relpath": f"images/{delivery_id}.png",
            "png_sha256": observation["png_sha256"],
            "byte_size": len(observation["png"]),
        })

    public_manifest = {
        "schema_version": "hidden-image-pack-manifest-0.4",
        "visibility": "pre_prediction_public",
        "challenge_id": CHALLENGE_ID,
        "pack_id": pack_id,
        "spec_sha256": config_sha,
        "png_profile": {"profile_id": "canonical-gray8-768-v0.4", "bit_depth": 8,
                        "color_type": 0, "interlace": 0,
                        "allowed_chunks": ["IHDR", "IDAT", "IEND"],
                        "trailing_bytes_allowed": False},
        "images": image_manifest_records,
    }
    pack_metadata = {
        "schema_version": SCHEMA_VERSION,
        "visibility": "pre_prediction_public",
        "challenge_id": CHALLENGE_ID,
        "pack_id": pack_id,
        "profile": config["profile"],
        "unique_observation_count": len(observations),
        "invocation_count": len(sample_records),
        "image_merkle_like_root": merkle_like_root(image_hash_records),
        "seed_is_present": False,
    }
    public_root = stage / "sealed_image_pack"
    atomic_write(public_root / "samples.jsonl", jsonl_bytes(sample_records))
    atomic_write(public_root / "manifest.json", canonical_json_bytes(public_manifest))
    atomic_write(public_root / "pack.json", canonical_json_bytes(pack_metadata))

    ink_records = [item["ink_program"] for item in observations]
    transform_records = [item["transform"] for item in observations]
    interpretation_records = [record for item in observations for record in item["interpretations"]]
    graph_records = [item["graph_set"] for item in observations]
    gt_observations = []
    for index, item in enumerate(observations):
        delivery_id = delivery_for_observation[index]
        gt_record = {
            "schema_version": "hidden-observation-gt-0.4",
            "visibility": "post_prediction_reveal",
            "observation_id": item["observation_id"],
            "delivery_id": delivery_id,
            "pack_id": pack_id,
            "family_id": item["family"],
            "quartet_ordinal": item["quartet"],
            "variant": item["variant"],
            "style_bucket_id": item["style_bucket"],
            "png_sha256": item["png_sha256"],
            "production_gray_sha256": item["gray_sha256"],
            "ink_program_id": item["ink_program"]["ink_program_id"],
            "ink_program_sha256": item["ink_program_sha256"],
            "transform_id": item["transform"]["transform_id"],
            "interpretation_ids": [record["interpretation_id"] for record in item["interpretations"]],
            "graph_set_id": item["graph_set"]["graph_set_id"],
            "ambiguity_kind": item["graph_set"]["ambiguity_kind"],
        }
        if item["support_certificate"] is not None:
            gt_record["support_certificate"] = item["support_certificate"]
        gt_observations.append(gt_record)
    atomic_write(gt_dir / "ink_programs.jsonl", jsonl_bytes(ink_records))
    atomic_write(gt_dir / "transforms.jsonl", jsonl_bytes(transform_records))
    atomic_write(gt_dir / "interpretations.jsonl", jsonl_bytes(interpretation_records))
    atomic_write(gt_dir / "graph_sets.jsonl", jsonl_bytes(graph_records))
    atomic_write(gt_dir / "observations.jsonl", jsonl_bytes(gt_observations))
    atomic_write(gt_dir / "rejection_log.jsonl", jsonl_bytes(rejection_records))

    gt_index = {
        "schema_version": "chemgraph-hidden-gt-index-v0.4",
        "visibility": "custodian-only-until-reveal",
        "challenge_id": CHALLENGE_ID,
        "pack_id": pack_id,
        "files": {},
    }
    for name in ("ink_programs.jsonl", "transforms.jsonl", "interpretations.jsonl",
                 "graph_sets.jsonl", "observations.jsonl", "rejection_log.jsonl"):
        gt_index["files"][name] = artifact(gt_dir / name, stage, "application/x-ndjson")
    atomic_write(gt_dir / "gt.json", canonical_json_bytes(gt_index))

    commitment = {
        "schema_version": "chemgraph-hidden-commitment-v0.4",
        "challenge_id": CHALLENGE_ID,
        "pack_id": pack_id,
        "profile": config["profile"],
        "seed_commitment": seed_commitment,
        "seed_included": False,
        "generator_source_tree_sha256": source_sha,
        "protocol_bundle_sha256": protocol_sha,
        "config_sha256": config_sha,
        "samples_manifest_sha256": sha256_bytes((public_root / "samples.jsonl").read_bytes()),
        "image_manifest_sha256": sha256_bytes((public_root / "manifest.json").read_bytes()),
        "image_merkle_like_root": pack_metadata["image_merkle_like_root"],
        "gt_index_sha256": sha256_bytes((gt_dir / "gt.json").read_bytes()),
        "rejection_log_sha256": sha256_bytes((gt_dir / "rejection_log.jsonl").read_bytes()),
        "pack_metadata_sha256": sha256_bytes((public_root / "pack.json").read_bytes()),
        "python_version": platform.python_version(),
        "zlib_version": __import__("zlib").ZLIB_VERSION,
        "unique_observation_count": len(observations),
        "invocation_count": len(sample_records),
    }
    atomic_write(stage / "commitment.json", canonical_json_bytes(commitment))
    atomic_write(stage / "config.json", config_bytes)
    os.replace(stage, output)
    elapsed = time.perf_counter() - started
    return {
        "output": str(output),
        "observations": len(observations),
        "invocations": len(sample_records),
        "elapsed_seconds": elapsed,
        "images_per_second": len(observations) / elapsed if elapsed else 0.0,
        "commitment_sha256": sha256_bytes((output / "commitment.json").read_bytes()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-hex", required=True,
                        help="exactly 64 hexadecimal characters; never written to output")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pack-id", choices=("pack-a", "pack-b"), default="pack-a")
    parser.add_argument("--smoke", action="store_true",
                        help="generate public test-vector quotas, not a real challenge pack")
    args = parser.parse_args(argv)
    if len(args.seed_hex) != 64:
        parser.error("--seed-hex must contain exactly 64 hex characters")
    try:
        args.seed = bytes.fromhex(args.seed_hex)
    except ValueError:
        parser.error("--seed-hex must be hexadecimal")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = generate(args.seed, args.output.resolve(), args.pack_id, args.smoke)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
