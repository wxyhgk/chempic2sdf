#!/usr/bin/env python3
"""Standard-library tests for the hidden generator reference implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    from .generate_pack_v0_4 import COLLISION_FAMILIES, build_motif, generate
    from .rasterizer_v0_4 import (
        Q, LineStyle, Segment, png_chunk_types, rasterize_png, rasterize_support,
        round_half_even,
    )
except ImportError:
    from generate_pack_v0_4 import COLLISION_FAMILIES, build_motif, generate  # type: ignore
    from rasterizer_v0_4 import (  # type: ignore
        Q, LineStyle, Segment, png_chunk_types, rasterize_png, rasterize_support,
        round_half_even,
    )


TEST_SEED = bytes(32)


class FixedC4RNG:
    def __init__(self, label_index: int):
        self.label_index = label_index

    def randint(self, lower: int, upper: int) -> int:
        return (lower + upper) // 2

    def randbelow(self, upper: int) -> int:
        if upper != 4:
            raise AssertionError(f"unexpected C4 choice cardinality: {upper}")
        return self.label_index


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class RasterizerTests(unittest.TestCase):
    def test_round_half_even(self) -> None:
        self.assertEqual(round_half_even(1, 2), 0)
        self.assertEqual(round_half_even(3, 2), 2)
        self.assertEqual(round_half_even(5, 2), 2)
        self.assertEqual(round_half_even(7, 2), 4)

    def test_png_is_canonical_and_deterministic(self) -> None:
        segment = Segment("p-000000", 100 * Q, 200 * Q, 600 * Q, 400 * Q)
        style = LineStyle(3 * Q, "round", "box_2x2")
        first, first_gray = rasterize_png([segment], style)
        second, second_gray = rasterize_png([segment], style)
        self.assertEqual(first, second)
        self.assertEqual(first_gray, second_gray)
        self.assertEqual(png_chunk_types(first), ["IHDR", "IDAT", "IEND"])
        self.assertEqual(struct.unpack(">IIBBBBB", first[16:29]), (768, 768, 8, 0, 0, 0, 0))

    def test_union_is_independent_of_segment_order(self) -> None:
        horizontal = Segment("p-000000", 200 * Q, 384 * Q, 568 * Q, 384 * Q)
        vertical = Segment("p-000001", 384 * Q, 200 * Q, 384 * Q, 568 * Q)
        style = LineStyle(2 * Q, "butt", "binary_1x")
        left, _ = rasterize_png([horizontal, vertical], style)
        right, _ = rasterize_png([vertical, horizontal], style)
        self.assertEqual(left, right)

    def test_round_join_changes_only_shared_endpoint_geometry(self) -> None:
        first = Segment("p-000000", 100 * Q, 100 * Q, 200 * Q, 100 * Q)
        shared = Segment("p-000001", 200 * Q, 100 * Q, 200 * Q, 200 * Q)
        separate = Segment("p-000002", 240 * Q, 100 * Q, 240 * Q, 200 * Q)
        style = LineStyle(10 * Q, "butt", "binary_1x", "round")
        independent_shared = rasterize_support([first], style) | rasterize_support([shared], style)
        joined = rasterize_support([first, shared], style)
        self.assertTrue(joined - independent_shared)
        independent_separate = rasterize_support([first], style) | rasterize_support([separate], style)
        self.assertEqual(rasterize_support([first, separate], style), independent_separate)


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="chemgraph-v04-test-")
        cls.output = Path(cls.temporary.name) / "pack"
        cls.result = generate(TEST_SEED, cls.output, "pack-a", smoke=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_smoke_counts(self) -> None:
        self.assertEqual(self.result["observations"], 48)
        self.assertEqual(self.result["invocations"], 72)
        images = list((self.output / "sealed_image_pack" / "images").glob("*.png"))
        self.assertEqual(len(images), 48)
        samples = read_jsonl(self.output / "sealed_image_pack" / "samples.jsonl")
        self.assertEqual(len(samples), 72)
        config = json.loads((self.output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["line_width_pixels"], list(range(1, 10)))
        self.assertEqual(config["antialias_profiles"],
                         ["binary_1x", "box_2x2", "box_4x4"])

    def test_public_records_do_not_reveal_gt(self) -> None:
        forbidden = ("family", "transform", "interpretation", "graph_set", "seed")
        public_files = [
            self.output / "sealed_image_pack" / "samples.jsonl",
            self.output / "sealed_image_pack" / "manifest.json",
        ]
        for path in public_files:
            content = path.read_text(encoding="utf-8")
            for word in forbidden:
                self.assertNotIn(word, content, f"{word} leaked in {path.name}")
            self.assertNotIn("glyph_template_id", content)
            self.assertNotIn("glyph-", content)

    def test_c4_text_graph_is_template_invariant(self) -> None:
        signatures = []
        template_ids = []
        for label_index in range(4):
            motif = build_motif("c4", FixedC4RNG(label_index), width_px=3)
            text = motif.interpretations[0]
            signatures.append((
                tuple((atom.point, atom.label) for atom in text.atoms),
                tuple((bond.begin, bond.end, bond.order, bond.primitive_indices)
                      for bond in text.bonds),
                text.primitive_roles,
                text.decode_kind,
            ))
            template_ids.append(motif.primitives[1].glyph_template_id)
            self.assertEqual(text.atoms[2].label, "TEXT_STROKE")
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(template_ids, ["glyph-I", "glyph-l", "glyph-1", "glyph-/"])

    def test_seed_is_committed_but_not_stored(self) -> None:
        seed_hex = TEST_SEED.hex()
        for path in (item for item in self.output.rglob("*") if item.is_file()):
            self.assertNotIn(seed_hex, path.read_text(encoding="utf-8", errors="ignore"))
        commitment = json.loads((self.output / "commitment.json").read_text(encoding="utf-8"))
        self.assertFalse(commitment["seed_included"])
        self.assertEqual(len(commitment["seed_commitment"]), 64)

    def test_collision_has_one_ink_and_multiple_interpretations(self) -> None:
        gt = read_jsonl(self.output / "sealed_gt" / "observations.jsonl")
        inks = {record["ink_program_id"]: record for record in
                read_jsonl(self.output / "sealed_gt" / "ink_programs.jsonl")}
        for record in gt:
            if record["family_id"] in COLLISION_FAMILIES:
                self.assertGreaterEqual(len(record["interpretation_ids"]), 2)
                self.assertIn(record["ink_program_id"], inks)
                image_path = (self.output / "sealed_image_pack" / "images" /
                              f"{record['delivery_id']}.png")
                self.assertEqual(hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                 record["png_sha256"])

    def test_c3_and_c4_support_certificates(self) -> None:
        gt = read_jsonl(self.output / "sealed_gt" / "observations.jsonl")
        inks = {record["ink_program_id"]: record for record in
                read_jsonl(self.output / "sealed_gt" / "ink_programs.jsonl")}
        rejection = read_jsonl(self.output / "sealed_gt" / "rejection_log.jsonl")
        c3_support_rejections = [record for record in rejection
                                 if record["family_id"] == "c3" and
                                 "invalid_exact_collision" in record["rejection_reasons"]]
        self.assertEqual(c3_support_rejections, [])
        for record in gt:
            if record["family_id"] not in ("c2", "c3", "c4"):
                continue
            certificate = record["support_certificate"]
            self.assertTrue(certificate["all_separated"])
            self.assertTrue(all(pair["foreground_intersection_count"] == 0 and
                                not pair["eight_neighbor_adjacent"] and pair["separated"]
                                for pair in certificate["pairs"]))
            if record["family_id"] == "c4":
                primitives = inks[record["ink_program_id"]]["primitives"]
                self.assertEqual(len(primitives), 2)
                self.assertEqual(sum(item["glyph_template_id"] is not None
                                     for item in primitives), 1)

    def test_protocol_bundle_is_committed(self) -> None:
        commitment = json.loads((self.output / "commitment.json").read_text(encoding="utf-8"))
        self.assertRegex(commitment["protocol_bundle_sha256"], r"^[0-9a-f]{64}$")

    def test_every_png_has_only_required_chunks(self) -> None:
        for path in (self.output / "sealed_image_pack" / "images").glob("*.png"):
            self.assertEqual(png_chunk_types(path.read_bytes()), ["IHDR", "IDAT", "IEND"])

    def test_nonempty_output_is_not_overwritten(self) -> None:
        with self.assertRaises(FileExistsError):
            generate(TEST_SEED, self.output, "pack-a", smoke=True)

    def test_repeat_tree_is_byte_exact(self) -> None:
        second = Path(self.temporary.name) / "pack-second"
        generate(TEST_SEED, second, "pack-a", smoke=True)
        self.assertEqual(tree_digest(self.output), tree_digest(second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
