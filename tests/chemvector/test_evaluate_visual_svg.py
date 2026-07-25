import unittest
import io
import json
from pathlib import Path
import tarfile

from PIL import Image
import torch

from Data.sdf.visual_svg_sequence_v0_1 import (
    ParsedVisualSVGSequenceV01,
    QuadRecordV01,
    TriangleRecordV01,
    parse_visual_svg_v0_1,
)
from chemvector.evaluate_visual_svg import (
    bond_iou,
    content_free_visual_memory,
    different_molecule_permutation,
    polygon_true_positives,
)
from chemvector.model import ChemVectorConfig
from chemvector.visual_prefix_model import VisualPrefixChemVectorModel


class VisualSVGEvaluationTest(unittest.TestCase):
    def test_polygon_matching_separates_triangles_and_quads(self):
        triangle = TriangleRecordV01(0, 0, 10, 0, 0, 10, 10)
        quad = QuadRecordV01(20, 20, 30, 20, 30, 30, 20, 30, 10)
        predicted = ParsedVisualSVGSequenceV01((), (triangle,), (quad,), ())
        expected = ParsedVisualSVGSequenceV01((), (triangle,), (), ())

        self.assertEqual(polygon_true_positives(predicted, expected, 1.0), 1)

    def test_different_molecule_permutation_is_deterministic_and_valid(self):
        molecule_ids = ["a", "a", "b", "c"]

        first = different_molecule_permutation(molecule_ids)
        second = different_molecule_permutation(molecule_ids)

        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        self.assertTrue(
            all(
                molecule_ids[source] != molecule_ids[target]
                for target, source in enumerate(first)
            )
        )

    def test_different_molecule_permutation_reports_impossible_batch(self):
        self.assertIsNone(different_molecule_permutation(["a", "a", "a", "b"]))

    def test_content_free_memory_removes_image_content(self):
        config = ChemVectorConfig(
            vocab_size=31,
            pad_token_id=0,
            d_model=32,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=64,
            dropout=0.0,
            max_target_length=16,
            vision_dims=(8, 16),
            vision_depths=(1, 1),
        )
        model = VisualPrefixChemVectorModel(
            config,
            visual_grid_size=4,
            ink_subgrid=2,
        ).eval()
        images = torch.stack(
            (
                torch.full((3, 32, 32), -1.0),
                torch.full((3, 32, 32), 1.0),
            )
        )

        with torch.no_grad():
            memory = model.encode_images(images)
            content_free = content_free_visual_memory(model, memory)

        torch.testing.assert_close(content_free.prefix[0], content_free.prefix[1])
        torch.testing.assert_close(
            content_free.values,
            torch.zeros_like(content_free.values),
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreater(
            float((memory.prefix - content_free.prefix).abs().max()),
            1e-5,
        )

    def test_cairo_renderer_matches_capacity_baseline(self):
        shard = Path(
            "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
        )
        if not shard.is_file():
            self.skipTest("capacity_100 has not been generated")
        with tarfile.open(shard, "r") as archive:
            key = next(member.name.split(".", 1)[0] for member in archive if member.name.endswith(".visualsvg.txt"))
            sequence = archive.extractfile(key + ".visualsvg.txt").read().decode("utf-8")
            bond_layer = Image.open(io.BytesIO(archive.extractfile(key + ".bond.png").read())).convert("L")
            bond_layer.load()
            meta = json.load(archive.extractfile(key + ".meta.json"))
        parsed = parse_visual_svg_v0_1(sequence)
        self.assertAlmostEqual(
            bond_iou(parsed, bond_layer),
            float(meta["bond_iou"]),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
