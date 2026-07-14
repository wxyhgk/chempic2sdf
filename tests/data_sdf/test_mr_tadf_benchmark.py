"""Tests for audited MR-TADF benchmark rendering."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor

from Data.sdf.build_mr_tadf_benchmark import build_benchmark


class MrTadfBenchmarkBuildTests(unittest.TestCase):
    def test_only_approved_sdf_is_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            mol = Chem.MolFromSmiles("B1=CC=CC=N1")
            rdDepictor.Compute2DCoords(mol)
            writer = Chem.SDWriter(str(source / "MR001.sdf"))
            writer.write(mol)
            writer.close()
            (source / "unapproved.sdf").write_text("invalid", encoding="utf-8")
            approved = root / "approved.csv"
            with approved.open("w", newline="", encoding="utf-8") as handle:
                manifest_writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "molecule_id",
                        "sdf_file",
                        "inchikey",
                        "maximum_to_median_bond_ratio",
                    ),
                    lineterminator="\n",
                )
                manifest_writer.writeheader()
                manifest_writer.writerows(
                    [
                        {
                            "molecule_id": "MR001",
                            "sdf_file": "MR001.sdf",
                            "inchikey": Chem.MolToInchiKey(mol),
                            "maximum_to_median_bond_ratio": "2.0",
                        }
                    ]
                )
            output = root / "benchmark"

            summary = build_benchmark(approved, source, output, width=128, height=128)

            self.assertEqual(summary["rendered_samples"], 1)
            self.assertEqual(summary["runtime_rejections"], 0)
            self.assertEqual(summary["difficulty_counts"], {"long_bond": 1})
            self.assertEqual(list((output / "images").glob("*.png"))[0].name, "MR001.png")
            self.assertFalse((output / "images/unapproved.png").exists())
            with Image.open(output / "images/MR001.png") as image:
                self.assertEqual(image.size, (128, 128))
            annotation = json.loads(
                (output / "annotations/MR001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(annotation["molecule_id"], "MR001")
            self.assertEqual(len(annotation["atoms"]), mol.GetNumAtoms())

    def test_output_failure_leaves_no_partial_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            mol = Chem.MolFromSmiles("c1ccccc1")
            rdDepictor.Compute2DCoords(mol)
            writer = Chem.SDWriter(str(source / "MR001.sdf"))
            writer.write(mol)
            writer.close()
            approved = root / "approved.csv"
            with approved.open("w", newline="", encoding="utf-8") as handle:
                manifest_writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "molecule_id",
                        "sdf_file",
                        "inchikey",
                        "maximum_to_median_bond_ratio",
                    ),
                    lineterminator="\n",
                )
                manifest_writer.writeheader()
                manifest_writer.writerow(
                    {
                        "molecule_id": "MR001",
                        "sdf_file": "MR001.sdf",
                        "inchikey": Chem.MolToInchiKey(mol),
                        "maximum_to_median_bond_ratio": "1.0",
                    }
                )
            output = root / "benchmark"

            with mock.patch(
                "Data.sdf.build_mr_tadf_benchmark.shutil.copy2",
                side_effect=OSError("simulated copy failure"),
            ):
                summary = build_benchmark(
                    approved,
                    source,
                    output,
                    width=128,
                    height=128,
                )

            self.assertEqual(summary["rendered_samples"], 0)
            self.assertEqual(summary["runtime_rejections"], 1)
            self.assertEqual(list((output / "images").iterdir()), [])
            self.assertEqual(list((output / "annotations").iterdir()), [])
            self.assertEqual(list((output / "ground_truth").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
