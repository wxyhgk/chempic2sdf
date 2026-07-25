from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from Data.sdf import build_chemvision_layers as layer_builder
from Data.sdf.build_visual_svg_layers_c_smoke import (
    find_train_sdf,
    load_exposed_train_rows,
    select_complete_molecules,
)


def _tar_add_json(archive: tarfile.TarFile, name: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _row(molecule_id: str, variant: str, key: str) -> dict[str, str]:
    return {
        "sample_key": key,
        "molecule_id": molecule_id,
        "split": "train",
        "variant": variant,
        "source_sample_key": f"train_{key}_{variant}",
        "source_shard": "shards/train-000000.tar",
        "output_shard": "shards/train-000000.tar",
        "font_family_id": "dejavu-sans",
        "font_size": "14",
        "font_bucket": "normal",
        "bond_line_width": "2.0",
        "rotation_degrees": "0.0",
    }


class VisualSvgLayersCSmokeBuilderTest(unittest.TestCase):
    def test_select_complete_molecules_keeps_four_variants_together(self) -> None:
        rows = []
        for molecule in ("chembl:A", "chembl:B"):
            for index, variant in enumerate(
                ("original", "rdkit", "coordgen", "stress")
            ):
                rows.append(_row(molecule, variant, f"{molecule[-1]}{index}"))
        rows.append(_row("chembl:INCOMPLETE", "original", "x"))

        selected = select_complete_molecules(rows, 1)

        self.assertEqual(len(selected), 4)
        self.assertEqual({row["variant"] for row in selected}, {
            "original",
            "rdkit",
            "coordgen",
            "stress",
        })
        self.assertEqual(len({row["molecule_id"] for row in selected}), 1)
        self.assertEqual(selected, select_complete_molecules(rows, 1))

    def test_safe_manifest_reads_only_explicit_train_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pilot = Path(directory)
            (pilot / "shards").mkdir()
            (pilot / "summary.json").write_text(
                json.dumps({"sample_split_counts": {"train": 1}}),
                encoding="utf-8",
            )
            layers = {
                "sample_key": "0001",
                "molecule_id": "chembl:CHEMBL1",
                "split": "train",
                "variant": "original",
                "source": {
                    "source_sample_key": "train_000001_original",
                    "source_shard": "shards/train-000000.tar",
                },
                "render_parameters": {
                    "font_family_id": "dejavu-sans",
                    "font_size": 14,
                    "bond_line_width": 2.0,
                    "rotation_degrees": 0.0,
                },
            }
            with tarfile.open(pilot / "shards/train-000000.tar", "w") as archive:
                _tar_add_json(archive, "0001.layers.json", layers)
            (pilot / "shards/validation-000000.tar").write_bytes(
                b"must never be opened"
            )

            rows = load_exposed_train_rows(pilot)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_key"], "0001")
            self.assertEqual(rows[0]["split"], "train")

    def test_train_sdf_lookup_never_scans_other_split_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            train = source / "structures/train-000001"
            train.mkdir(parents=True)
            expected = train / "chembl_CHEMBL1.sdf"
            expected.write_text("train", encoding="ascii")
            other = source / "structures/validation-000001"
            other.mkdir(parents=True)
            (other / expected.name).write_text("other", encoding="ascii")

            self.assertEqual(find_train_sdf(source, "chembl:CHEMBL1"), expected)

    def test_legacy_render_wrapper_discards_only_the_svg_return(self) -> None:
        generated = {".png": b"png"}
        metrics = {"foreground_iou": 1.0}
        with patch.object(
            layer_builder,
            "render_and_layers_with_svg",
            return_value=(generated, metrics, b"<svg/>"),
        ):
            actual = layer_builder.render_and_layers(None, {}, {})

        self.assertEqual(actual, (generated, metrics))


if __name__ == "__main__":
    unittest.main()
