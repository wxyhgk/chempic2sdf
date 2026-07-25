from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image
from rdkit import Chem

from Data.sdf.validate_visual_svg_layers_c import (
    MEMBER_SUFFIXES,
    PASS_STATUS,
    validate_smoke_package,
)
from Data.sdf.visual_svg_sequence_v0_1 import serialize_visual_svg_v0_1
from Data.sdf.visual_svg_v0_2 import serialize_visual_svg_v0_2


VARIANTS = ("original", "rdkit", "coordgen", "stress")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _png(mode: str) -> bytes:
    image = Image.new(mode, (768, 768), 255)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _chemmol_and_sdf(smiles: str = "CC") -> tuple[dict, bytes, str]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    inchikey = Chem.MolToInchiKey(molecule)
    atoms = []
    for atom in molecule.GetAtoms():
        atoms.append(
            {
                "id": atom.GetIdx(),
                "source_atom_index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": str(atom.GetChiralTag()).rsplit(".", 1)[-1],
                "cip_code": None,
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
            }
        )
    bonds = []
    for bond in molecule.GetBonds():
        bonds.append(
            {
                "id": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "type": str(bond.GetBondType()).rsplit(".", 1)[-1],
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "stereo": str(bond.GetStereo()).rsplit(".", 1)[-1],
                "stereo_atoms": list(bond.GetStereoAtoms()),
                "ez": None,
            }
        )
    chemmol = {
        "schema_version": "chemmol-0.1",
        "molecule_id": "chembl:TEST1",
        "structure_ref": "structures/train-000000/chembl_TEST1.sdf",
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "inchikey": inchikey,
        "atoms": atoms,
        "bonds": bonds,
    }
    sdf = (Chem.MolToMolBlock(molecule, forceV3000=True) + "\n$$$$\n").encode(
        "utf-8"
    )
    return chemmol, sdf, inchikey


def _primitives(sample_key: str) -> dict:
    return {
        "schema_version": "visual-primitives-0.1",
        "sample_key": sample_key,
        "molecule_id": "chembl:TEST1",
        "width": 768,
        "height": 768,
        "coordinate_space": "image_pixels",
        "coordinate_quantization": 2047,
        "stroke_width_scale": 10,
        "bond_groups": [
            {
                "bond_id": 0,
                "begin_atom": 0,
                "end_atom": 1,
                "chemical_type": "SINGLE",
                "display_type": "SINGLE",
                "direction": "NONE",
                "primitive_ids": [0],
            }
        ],
        "text_instances": [],
        "primitives": [
            {
                "kind": "line",
                "owner_kind": "bond",
                "owner_id": 0,
                "points": [[100.0, 100.0], [300.0, 100.0]],
                "quantized_points": [[267, 267], [800, 267]],
                "stroke_width": 2.0,
                "stroke_width_q": 20,
                "quantization_adjusted": False,
                "source_z_index": 0,
                "primitive_id": 0,
                "canonical_index": 0,
            }
        ],
    }


class SmokeFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.shard = root / "smoke-train-000000.tar"
        self.samples = root / "samples.csv"
        self.paired = root / "paired_manifest.csv"
        self.chemmol, self.sdf, self.inchikey = _chemmol_and_sdf()
        self.members: dict[str, dict[str, bytes]] = {}
        for index, variant in enumerate(VARIANTS):
            key = f"{index:09d}"
            primitives = _primitives(key)
            payload = {
                ".png": _png("RGB"),
                ".text.png": _png("L"),
                ".bond.png": _png("L"),
                ".semantic.png": _png("L"),
                ".primitives.json": _json_bytes(primitives),
                ".visualsvg.txt": serialize_visual_svg_v0_1(primitives).encode(
                    "utf-8"
                ),
                ".visualsvg-v02.json": serialize_visual_svg_v0_2(primitives).encode(
                    "utf-8"
                ),
                ".layers.json": _json_bytes(
                    {
                        "sample_key": key,
                        "molecule_id": "chembl:TEST1",
                        "split": "train",
                        "variant": variant,
                        "source": {"source_shard": "shards/train-000000.tar"},
                    }
                ),
                ".render.json": _json_bytes(
                    {
                        "molecule_id": "chembl:TEST1",
                        "variant": variant,
                        "width": 768,
                        "height": 768,
                    }
                ),
                ".chemmol.json": _json_bytes(self.chemmol),
                ".sdf": self.sdf,
            }
            self.members[key] = payload
        self.write()

    def write(self, *, reverse_tar_order: bool = False) -> None:
        sample_rows = []
        paired_rows = []
        for index, variant in enumerate(VARIANTS):
            key = f"{index:09d}"
            payload = self.members[key]
            hashes = {
                suffix.removeprefix("."): _sha256(payload[suffix])
                for suffix in MEMBER_SUFFIXES[:-1]
            }
            meta = {
                "schema_version": "visual-svg-layers-c-meta-0.1",
                "sample_key": key,
                "molecule_id": "chembl:TEST1",
                "split": "smoke_train",
                "variant": variant,
                "source": {
                    "dataset": "pilot_train",
                    "source_sample_key": f"train_{key}",
                    "source_shard": "shards/train-000000.tar",
                    "structure_ref": "structures/train-000000/chembl_TEST1.sdf",
                },
                "fresh_audit_eligible": False,
                "semantic_svg_sha256": "0" * 64,
                "sdf_inchikey": self.inchikey,
                "sequence_tokens": 4,
                "sha256": hashes,
            }
            payload[".meta.json"] = _json_bytes(meta)
            sample_rows.append(
                {
                    "sample_key": key,
                    "molecule_id": "chembl:TEST1",
                    "split": "smoke_train",
                    "variant": variant,
                    "sdf_inchikey": self.inchikey,
                }
            )
            paired_rows.append(
                {
                    "sample_key": key,
                    "molecule_id": "chembl:TEST1",
                    "inchikey": self.inchikey,
                    "split": "smoke_train",
                    "variant": variant,
                    "image_sha256": _sha256(payload[".png"]),
                    "sdf_sha256": _sha256(payload[".sdf"]),
                    "primitives_sha256": _sha256(payload[".primitives.json"]),
                    "visualsvg_v01_sha256": _sha256(payload[".visualsvg.txt"]),
                    "visualsvg_v02_sha256": _sha256(
                        payload[".visualsvg-v02.json"]
                    ),
                    "render_sha256": _sha256(payload[".render.json"]),
                    "meta_sha256": _sha256(payload[".meta.json"]),
                }
            )
        self._write_csv(self.samples, sample_rows)
        self._write_csv(self.paired, paired_rows)
        names = [
            key + suffix
            for key in sorted(self.members)
            for suffix in MEMBER_SUFFIXES
        ]
        if reverse_tar_order:
            names[0], names[1] = names[1], names[0]
        with tarfile.open(self.shard, "w") as archive:
            for name in names:
                key, suffix = next(
                    (key, name[len(key) :])
                    for key in self.members
                    if name.startswith(key)
                )
                data = self.members[key][suffix]
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(data))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class VisualSvgLayersCIndependentValidatorTest(unittest.TestCase):
    def test_valid_four_layout_smoke_package_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train")

            report = validate_smoke_package(
                fixture.shard, fixture.samples, fixture.paired
            )

            self.assertEqual(report["status"], PASS_STATUS)
            self.assertFalse(report["fresh_audit_eligible"])
            self.assertEqual(report["samples"], 4)
            self.assertEqual(report["tar_members"], 48)

    def test_forbidden_path_is_rejected_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "validation"
            root.mkdir()
            shard = root / "does-not-exist.tar"

            report = validate_smoke_package(shard, shard, shard)

            self.assertEqual(report["status"], "FAIL")
            self.assertIn("forbidden_non_train_reference", report["errors"][0])

    def test_symlink_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train")
            link = fixture.root / "linked.tar"
            link.symlink_to(fixture.shard)

            report = validate_smoke_package(link, fixture.samples, fixture.paired)

            self.assertEqual(report["status"], "FAIL")
            self.assertIn("symlink_rejected", report["errors"][0])

    def test_tar_member_order_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train")
            fixture.write(reverse_tar_order=True)

            report = validate_smoke_package(
                fixture.shard, fixture.samples, fixture.paired
            )

            self.assertEqual(report["status"], "FAIL")
            self.assertIn("tar_member_order_or_names", report["errors"])

    def test_v02_requires_byte_exact_canonical_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train")
            fixture.members["000000000"][".visualsvg-v02.json"] += b"\n"
            fixture.write()

            report = validate_smoke_package(
                fixture.shard, fixture.samples, fixture.paired
            )

            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(
                any("visualsvg_v02_not_exact_canonical" in error for error in report["errors"])
            )

    def test_sdf_chemmol_mismatch_and_variant_hash_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train")
            _, methane_sdf, _ = _chemmol_and_sdf("C")
            fixture.members["000000003"][".sdf"] = methane_sdf
            fixture.write()

            report = validate_smoke_package(
                fixture.shard, fixture.samples, fixture.paired
            )

            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(
                any("sdf_chemmol_inchikey_mismatch" in error for error in report["errors"])
            )
            self.assertTrue(
                any("sdf_hash_not_shared_across_variants" in error for error in report["errors"])
            )

    def test_paired_manifest_hash_must_match_tar_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train")
            with fixture.paired.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["image_sha256"] = "f" * 64
            fixture._write_csv(fixture.paired, rows)

            report = validate_smoke_package(
                fixture.shard, fixture.samples, fixture.paired
            )

            self.assertEqual(report["status"], "FAIL")
            self.assertIn(
                "paired_manifest_hash:000000000:image_sha256",
                report["errors"],
            )


if __name__ == "__main__":
    unittest.main()
