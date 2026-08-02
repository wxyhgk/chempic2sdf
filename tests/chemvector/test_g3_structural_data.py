import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch

from chemvector.g3_structural_data import (
    G3StructuralCollator,
    file_sha256,
    load_g3_sample_manifest,
)


def structural_sample(key: str = "sample_a") -> dict:
    return {
        "key": key,
        "image": Image.new("RGB", (32, 32), "white"),
        "chemmol": {
            "molecule_id": "train:molecule-a",
            "atoms": [
                {"id": 10, "element": "N"},
                {"id": 20, "element": "O"},
            ],
            "bonds": [
                {"id": 7, "begin_atom": 10, "end_atom": 20, "type": "DOUBLE"}
            ],
        },
        "render": {
            "variant": "rdkit",
            "width": 32,
            "height": 32,
            "atoms": [
                {
                    "atom_id": 10,
                    "image_x": 0.0,
                    "image_y": 0.0,
                    "quantized_x": 0,
                    "quantized_y": 0,
                },
                {
                    "atom_id": 20,
                    "image_x": 31.0,
                    "image_y": 15.5,
                    "quantized_x": 1023,
                    "quantized_y": 512,
                },
            ],
            "bonds": [
                {
                    "bond_id": 7,
                    "begin_image_x": 0.0,
                    "begin_image_y": 0.0,
                    "end_image_x": 31.0,
                    "end_image_y": 15.5,
                    "display_type": "SINGLE",
                }
            ],
        },
        "meta": {"split": "train"},
    }


class G3StructuralDataTests(unittest.TestCase):
    def test_collator_exposes_only_structural_targets_and_sidecars(self) -> None:
        collator = G3StructuralCollator(
            maximum_atoms=4,
            maximum_bonds=3,
            image_size=32,
        )

        batch = collator([structural_sample()])

        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 32, 32))
        self.assertEqual(batch["atom_counts"].tolist(), [2])
        self.assertEqual(batch["bond_counts"].tolist(), [1])
        torch.testing.assert_close(
            batch["atom_anchors"][0, :2],
            torch.tensor([[0.0, 0.0], [1.0, 512.0 / 1023.0]]),
        )
        self.assertEqual(batch["bond_atom_refs"][0, 0].tolist(), [0, 1])
        self.assertEqual(int(batch["bond_types"][0, 0]), 1)
        self.assertEqual(int(batch["bond_display"][0, 0]), 0)
        self.assertEqual(batch["keys"], ["sample_a"])
        self.assertEqual(batch["molecule_ids"], ["train:molecule-a"])
        for forbidden in (
            "atom_roles",
            "atom_text_center",
            "atom_text_bbox",
            "atom_elements",
            "bond_direction",
            "stereo",
        ):
            self.assertNotIn(forbidden, batch)

    def test_collator_rejects_capacity_overflow(self) -> None:
        collator = G3StructuralCollator(
            maximum_atoms=1,
            maximum_bonds=3,
            image_size=32,
        )

        with self.assertRaisesRegex(ValueError, "capacity"):
            collator([structural_sample()])

    def test_manifest_loader_binds_hash_counts_variants_and_shard(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "train.tar"
            shard.write_bytes(b"train-only-placeholder")
            records = [
                {
                    "sample_key": "train_a",
                    "molecule_id": "molecule_a",
                    "inchikey": "A",
                    "shard": str(shard),
                    "variant": "original",
                },
                {
                    "sample_key": "train_b",
                    "molecule_id": "molecule_b",
                    "inchikey": "B",
                    "shard": str(shard),
                    "variant": "stress",
                },
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            expected_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

            loaded, audit = load_g3_sample_manifest(
                manifest,
                expected_sha256=expected_sha256,
                expected_samples=2,
                expected_molecules=2,
                expected_variants=frozenset({"original", "stress"}),
            )

            self.assertEqual(
                [record.sample_key for record in loaded], ["train_a", "train_b"]
            )
            self.assertEqual(audit["sha256"], file_sha256(manifest))
            self.assertEqual(audit["sample_count"], 2)
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                load_g3_sample_manifest(
                    manifest,
                    expected_sha256=expected_sha256,
                    expected_samples=2,
                    expected_molecules=2,
                    expected_variants=frozenset({"original", "stress"}),
                )


if __name__ == "__main__":
    unittest.main()
