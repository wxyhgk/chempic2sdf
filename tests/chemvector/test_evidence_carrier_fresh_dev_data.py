from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import rdDepictor

from Data.sdf.visual_svg_sequence_v0_1 import serialize_visual_svg_v0_1
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from Data.sdf.visual_svg_v0_2 import (
    serialize_visual_svg_v0_2,
    visual_primitives_v0_1_to_visual_svg_v0_2,
)

import chemvector.evidence_carrier_fresh_dev_data as fresh_dev_data
from chemvector.evidence_carrier_fresh_dev_data import (
    MEMBER_SUFFIXES,
    VARIANTS,
    FreshEvidenceCarrierDataError,
    iter_fresh_evidence_carrier_samples,
    load_fresh_evidence_carrier_package,
)


DEV_FONT_SHA256 = "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45"
DEV_RENDER_PROFILE = "visual-svg-layers-c-clean-bw-768-v0.1"
PHASE_A_FREEZE_PATH = (
    Path(__file__).resolve().parents[2]
    / "Data/sdf/visual_svg_layers_c_v0_1/phase_a_freeze.json"
)
ISOLATION_HASH_FIELDS = (
    "fresh_dev_molecule_manifest_sha256",
    "fresh_dev_sample_manifest_sha256",
    "fresh_dev_paired_manifest_sha256",
    "exposure_manifest_sha256",
    "molecule_id_exclusion_set_sha256",
    "inchikey_full_exclusion_set_sha256",
    "scaffold_id_exclusion_set_sha256",
    "source_record_id_exclusion_set_sha256",
    "isolation_algorithm_sha256",
    "isolation_config_sha256",
    "scaffold_id_algorithm_sha256",
    "source_record_normalization_algorithm_sha256",
    "style_id_algorithm_sha256",
)
ISOLATION_OVERLAP_FIELDS = (
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
    "style_id",
    "font_sha",
    "render_profile",
)
SELECTION_FIELDS = (
    "selection_index",
    "rank_sha256",
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
    "source",
    "source_id",
    "source_split",
    "source_shard",
    "source_structure",
    "source_sdf_sha256",
    "source_sdf_bytes",
    "source_chemmol_sha256",
    "source_atom_count",
    "source_bond_count",
)
REPLACEMENT_FIELDS = (
    "replacement_index",
    *SELECTION_FIELDS[1:],
    "purpose",
    "rendered",
)
IDENTITY_EVIDENCE_HASH_FIELDS = (
    "exposure_manifest_sha256",
    "molecule_id_exclusion_set_sha256",
    "inchikey_full_exclusion_set_sha256",
    "scaffold_id_exclusion_set_sha256",
    "source_record_id_exclusion_set_sha256",
    "isolation_algorithm_sha256",
    "isolation_config_sha256",
    "scaffold_id_algorithm_sha256",
    "source_record_normalization_algorithm_sha256",
    "style_id_algorithm_sha256",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def png_bytes(mode: str, color: int | tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new(mode, (768, 768), color).save(output, format="PNG")
    return output.getvalue()


def enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def molecule_payload(
    molecule_id: str,
    source_record_id: str,
    smiles: str,
) -> tuple[Chem.Mol, dict[str, object], bytes, str]:
    source_molecule = Chem.MolFromSmiles(smiles)
    if source_molecule is None:
        raise AssertionError(f"Invalid test SMILES: {smiles}")
    rdDepictor.Compute2DCoords(source_molecule)
    sdf = (
        Chem.MolToMolBlock(source_molecule, forceV3000=True) + "\n$$$$\n"
    ).encode("utf-8")
    molecule = list(
        Chem.ForwardSDMolSupplier(
            io.BytesIO(sdf),
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
    )[0]
    if molecule is None:
        raise AssertionError(f"Test SDF roundtrip failed: {smiles}")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
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
                "chiral_tag": enum_name(atom.GetChiralTag()),
                "cip_code": (
                    atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
                ),
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
            }
        )
    bonds = []
    for bond in molecule.GetBonds():
        stereo = enum_name(bond.GetStereo())
        bonds.append(
            {
                "id": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "type": enum_name(bond.GetBondType()),
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "stereo": stereo,
                "stereo_atoms": list(bond.GetStereoAtoms()),
                "ez": (
                    "E"
                    if stereo == "STEREOE"
                    else "Z" if stereo == "STEREOZ" else None
                ),
            }
        )
    structure_basename = source_record_id.split(":", 1)[-1]
    structure_ref = f"structures/train-000000/{structure_basename}.sdf"
    chemmol = {
        "schema_version": "chemmol-0.1",
        "molecule_id": molecule_id,
        "structure_ref": structure_ref,
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        "inchikey": inchikey,
        "atoms": atoms,
        "bonds": bonds,
    }
    return molecule, chemmol, sdf, inchikey


def atom_positions(molecule: Chem.Mol) -> list[tuple[float, float]]:
    conformer = molecule.GetConformer()
    raw = [conformer.GetAtomPosition(index) for index in range(molecule.GetNumAtoms())]
    xs = [point.x for point in raw]
    ys = [point.y for point in raw]
    centre_x = 0.5 * (min(xs) + max(xs))
    centre_y = 0.5 * (min(ys) + max(ys))
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    scale = 360.0 / span
    return [
        (384.0 + (point.x - centre_x) * scale, 384.0 - (point.y - centre_y) * scale)
        for point in raw
    ]


def quantized_point(point: tuple[float, float]) -> list[int]:
    return [
        max(0, min(2047, int(round(coordinate * 2047.0 / 768.0))))
        for coordinate in point
    ]


def primitives_payload(
    sample_key: str,
    molecule_id: str,
    molecule: Chem.Mol,
) -> tuple[dict[str, object], list[tuple[float, float]]]:
    positions = atom_positions(molecule)
    groups: list[dict[str, object]] = []
    primitives: list[dict[str, object]] = []
    for bond in molecule.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        begin = positions[begin_atom]
        end = positions[end_atom]
        dx = end[0] - begin[0]
        dy = end[1] - begin[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        perpendicular = (-dy / length, dx / length)
        bond_type = enum_name(bond.GetBondType())
        offsets = {
            "DOUBLE": (-4.0, 4.0),
            "TRIPLE": (-6.0, 0.0, 6.0),
        }.get(bond_type, (0.0,))
        primitive_ids = []
        for offset in offsets:
            first = (
                begin[0] + perpendicular[0] * offset,
                begin[1] + perpendicular[1] * offset,
            )
            second = (
                end[0] + perpendicular[0] * offset,
                end[1] + perpendicular[1] * offset,
            )
            primitive_id = len(primitives)
            primitive_ids.append(primitive_id)
            primitives.append(
                {
                    "kind": "line",
                    "owner_kind": "bond",
                    "owner_id": bond.GetIdx(),
                    "points": [list(first), list(second)],
                    "quantized_points": [
                        quantized_point(first),
                        quantized_point(second),
                    ],
                    "stroke_width": 2.0,
                    "stroke_width_q": 20,
                    "quantization_adjusted": False,
                    "source_z_index": primitive_id,
                    "primitive_id": primitive_id,
                    "canonical_index": primitive_id,
                }
            )
        groups.append(
            {
                "bond_id": bond.GetIdx(),
                "begin_atom": begin_atom,
                "end_atom": end_atom,
                "chemical_type": bond_type,
                "display_type": bond_type,
                "direction": enum_name(bond.GetBondDir()),
                "primitive_ids": primitive_ids,
            }
        )
    return (
        {
            "schema_version": "visual-primitives-0.1",
            "sample_key": sample_key,
            "molecule_id": molecule_id,
            "width": 768,
            "height": 768,
            "coordinate_space": "image_pixels",
            "coordinate_quantization": 2047,
            "stroke_width_scale": 10,
            "bond_groups": groups,
            "text_instances": [],
            "primitives": primitives,
        },
        positions,
    )


def style_payload(variant: str) -> tuple[dict[str, object], str]:
    style = {
        "schema": "visual-svg-layers-c-style-0.1",
        "canvas": {"height": "768.000000", "width": "768.000000"},
        "font_sha256": DEV_FONT_SHA256,
        "font_family": "STIX",
        "render_profile": DEV_RENDER_PROFILE,
        "layout_variant": variant,
        "font_size_bucket": "normal",
        "bond_line_width_bucket": "normal",
        "atom_label_style": "rdkit-outline-glyphs",
        "bond_style": "rdkit-semantic-paths",
        "color_mode": "black_white",
        "background": "opaque_white",
        "antialias": "cairosvg-default",
    }
    encoded = json.dumps(
        style,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return style, f"style-v0.1:{sha256(encoded)}"


def rendered_layers(primitives: dict[str, object]) -> dict[str, bytes]:
    combined = Image.new("RGB", (768, 768), (255, 255, 255))
    bond = Image.new("L", (768, 768), 255)
    semantic = Image.new("L", (768, 768), 255)
    for image in (combined, bond, semantic):
        drawing = ImageDraw.Draw(image)
        color = (0, 0, 0) if image.mode == "RGB" else 0
        for primitive in primitives["primitives"]:
            if primitive["kind"] == "line":
                drawing.line(
                    [tuple(point) for point in primitive["points"]],
                    fill=color,
                    width=2,
                )
    result = {}
    for suffix, image in (
        (".png", combined),
        (".text.png", Image.new("L", (768, 768), 255)),
        (".bond.png", bond),
        (".semantic.png", semantic),
    ):
        output = io.BytesIO()
        image.save(output, format="PNG")
        result[suffix] = output.getvalue()
    return result


class SmokeFixture:
    fields = (
        "sample_key",
        "molecule_id",
        "inchikey",
        "split",
        "variant",
        "shard",
        "image_sha256",
        "sdf_sha256",
        "primitives_sha256",
        "visualsvg_v01_sha256",
        "visualsvg_v02_sha256",
        "layers_sha256",
        "render_sha256",
        "chemmol_sha256",
        "meta_sha256",
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        (root / "shards").mkdir()
        self.commitment_path = root / "commitment.smoke.json"
        self.manifest_path = root / "paired_manifest.smoke.csv"
        self.validation_path = root / "validation.json"
        self.shard_paths = (
            root / "shards/smoke-train-000000.tar",
            root / "shards/smoke-train-000001.tar",
        )
        self.rows: list[dict[str, str]] = []
        self.members: dict[str, dict[str, bytes]] = {}
        self.shard_by_key: dict[str, int] = {}
        rgb = png_bytes("RGB", (255, 255, 255))
        layer = png_bytes("L", 255)
        # Deliberately keep a non-lexicographic manifest order and alternate
        # shards so the iterator must obey manifest order, not tar order.
        molecule_order = (3, 0, 7, 2, 6, 1, 5, 4)
        variant_order = ("stress", "original", "coordgen", "rdkit")
        for molecule_index in molecule_order:
            molecule_id = f"chembl:fixture-{molecule_index:02d}"
            inchikey = f"FIXTURE-INCHIKEY-{molecule_index:02d}"
            sdf = f"fixture sdf {molecule_index}\n$$$$\n".encode("ascii")
            chemmol = json_bytes(
                {"molecule_id": molecule_id, "inchikey": inchikey}
            )
            for variant_index, variant in enumerate(variant_order):
                sample_key = f"smoke_{molecule_index:02d}_{variant}"
                shard_index = (molecule_index + variant_index) % 2
                shard_relative = (
                    f"shards/smoke-train-{shard_index:06d}.tar"
                )
                members = {
                    ".png": rgb,
                    ".text.png": layer,
                    ".bond.png": layer,
                    ".semantic.png": layer,
                    ".primitives.json": json_bytes(
                        {
                            "sample_key": sample_key,
                            "molecule_id": molecule_id,
                            "primitives": [],
                        }
                    ),
                    ".visualsvg.txt": b"VSVG 0\n",
                    ".visualsvg-v02.json": json_bytes(
                        {"schema_version": "visual-svg-0.2", "bond_groups": []}
                    ),
                    ".layers.json": json_bytes(
                        {
                            "sample_key": sample_key,
                            "molecule_id": molecule_id,
                            "split": "train",
                            "variant": variant,
                            "source": {"dataset": "pilot_train"},
                        }
                    ),
                    ".render.json": json_bytes(
                        {"molecule_id": molecule_id, "variant": variant}
                    ),
                    ".chemmol.json": chemmol,
                    ".sdf": sdf,
                }
                member_hashes = {
                    suffix.removeprefix("."): sha256(members[suffix])
                    for suffix in MEMBER_SUFFIXES[:-1]
                }
                members[".meta.json"] = json_bytes(
                    {
                        "sample_key": sample_key,
                        "molecule_id": molecule_id,
                        "split": "smoke_train",
                        "variant": variant,
                        "fresh_audit_eligible": False,
                        "sdf_inchikey": inchikey,
                        "source": {"dataset": "pilot_train"},
                        "sha256": member_hashes,
                    }
                )
                self.members[sample_key] = members
                self.shard_by_key[sample_key] = shard_index
                self.rows.append(
                    {
                        "sample_key": sample_key,
                        "molecule_id": molecule_id,
                        "inchikey": inchikey,
                        "split": "smoke_train",
                        "variant": variant,
                        "shard": shard_relative,
                        "image_sha256": sha256(members[".png"]),
                        "sdf_sha256": sha256(members[".sdf"]),
                        "primitives_sha256": sha256(
                            members[".primitives.json"]
                        ),
                        "visualsvg_v01_sha256": sha256(
                            members[".visualsvg.txt"]
                        ),
                        "visualsvg_v02_sha256": sha256(
                            members[".visualsvg-v02.json"]
                        ),
                        "layers_sha256": sha256(members[".layers.json"]),
                        "render_sha256": sha256(members[".render.json"]),
                        "chemmol_sha256": sha256(members[".chemmol.json"]),
                        "meta_sha256": sha256(members[".meta.json"]),
                    }
                )
        self.write_manifest()
        self.write_shards()
        self.validation_path.write_bytes(
            json_bytes(
                {
                    "status": "passed",
                    "fresh_audit_eligible": False,
                    "molecules": 8,
                    "samples": 32,
                    "members": 32 * len(MEMBER_SUFFIXES),
                }
            )
        )
        self.commitment = {
            "schema_version": "visual-svg-layers-c-input-commitment-0.1",
            "role": "smoke",
            "fresh_audit_eligible": False,
            "sample_count": 32,
            "molecule_count": 8,
            "variants": list(VARIANTS),
            "members_per_sample": len(MEMBER_SUFFIXES),
            "paired_manifest": {
                "path": "paired_manifest.smoke.csv",
                "sha256": sha256(self.manifest_path.read_bytes()),
            },
            "shards": [],
            "validation": {
                "path": "validation.json",
                "sha256": sha256(self.validation_path.read_bytes()),
                "status": "passed",
            },
        }
        self.refresh_shard_commitments()
        self.write_commitment()

    def write_manifest(self) -> None:
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_shards(self, *, extra_member: bool = False) -> None:
        for shard_index, path in enumerate(self.shard_paths):
            with tarfile.open(path, "w") as archive:
                keys = sorted(
                    key
                    for key, current_shard in self.shard_by_key.items()
                    if current_shard == shard_index
                )
                for key in keys:
                    for suffix in MEMBER_SUFFIXES:
                        payload = self.members[key][suffix]
                        info = tarfile.TarInfo(key + suffix)
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                if extra_member and shard_index == 0:
                    payload = b"extra"
                    info = tarfile.TarInfo("uncommitted.bin")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

    def refresh_shard_commitments(self) -> None:
        self.commitment["shards"] = []
        for shard_index, path in enumerate(self.shard_paths):
            sample_count = sum(
                value == shard_index for value in self.shard_by_key.values()
            )
            self.commitment["shards"].append(
                {
                    "path": f"shards/smoke-train-{shard_index:06d}.tar",
                    "sha256": sha256(path.read_bytes()),
                    "bytes": path.stat().st_size,
                    "samples": sample_count,
                    "members": sample_count * len(MEMBER_SUFFIXES),
                }
            )

    def refresh_manifest_commitment(self) -> None:
        self.commitment["paired_manifest"]["sha256"] = sha256(
            self.manifest_path.read_bytes()
        )

    def write_commitment(self, path: Path | None = None) -> Path:
        path = path or self.commitment_path
        path.write_bytes(json_bytes(self.commitment))
        return path

    def rebuild_after_member_change(self) -> None:
        for row in self.rows:
            key = row["sample_key"]
            members = self.members[key]
            row.update(
                {
                    "image_sha256": sha256(members[".png"]),
                    "sdf_sha256": sha256(members[".sdf"]),
                    "primitives_sha256": sha256(members[".primitives.json"]),
                    "visualsvg_v01_sha256": sha256(members[".visualsvg.txt"]),
                    "visualsvg_v02_sha256": sha256(
                        members[".visualsvg-v02.json"]
                    ),
                    "layers_sha256": sha256(members[".layers.json"]),
                    "render_sha256": sha256(members[".render.json"]),
                    "chemmol_sha256": sha256(members[".chemmol.json"]),
                    "meta_sha256": sha256(members[".meta.json"]),
                }
            )
        self.write_manifest()
        self.write_shards()
        self.refresh_manifest_commitment()
        self.refresh_shard_commitments()
        self.write_commitment()


class DevFixture:
    """Small, chemically valid formal-dev package under a patched capacity."""

    role_counts = (8, 32, True, "fresh_dev")
    paired_fields = (
        *SmokeFixture.fields,
        "scaffold_id",
        "source_record_id",
        "style_id",
        "font_sha",
        "render_profile",
    )
    molecule_fields = (
        "molecule_id",
        "inchikey",
        "scaffold_id",
        "source_record_id",
        "sdf_sha256",
    )
    sample_fields = (
        "sample_key",
        "molecule_id",
        "split",
        "variant",
        "style_id",
        "font_sha",
        "render_profile",
        "image_sha256",
        "sdf_sha256",
    )
    smiles = (
        "CC",
        "CCC",
        "CCCC",
        "C=C",
        "C#C",
        "C1CC1",
        "c1ccc(Cc2ccc([C@H]3C[C@@H](N4CCCC4)C3)cc2)cc1",
        "C/C=C/C",
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True)
        (root / "shards").mkdir()
        (root / "preflight_64").mkdir()
        self.commitment_path = root / "commitment.dev.json"
        self.manifest_path = root / "paired_manifest.dev.csv"
        self.molecule_manifest_path = root / "molecule_manifest.dev.csv"
        self.sample_manifest_path = root / "sample_manifest.dev.csv"
        self.selection_commitment_path = root / "selection.commitment.json"
        self.selection_manifest_path = root / "selection.committed.csv"
        self.replacement_manifest_path = root / "replacement_pool.committed.csv"
        self.identity_isolation_path = root / "identity_isolation_report.json"
        self.selection_provenance_path = root / "selection_provenance.json"
        self.isolation_path = root / "isolation_report.json"
        self.preflight_validation_path = (
            root / "preflight_64/preflight_validation.json"
        )
        self.validation_path = root / "validation.json"
        self.freeze_path = root.parent / "phase_a_freeze.json"
        self.freeze_path.write_bytes(PHASE_A_FREEZE_PATH.read_bytes())
        self.shard_paths = (
            root / "shards/dev-000000.tar",
            root / "shards/dev-000001.tar",
        )
        self.rows: list[dict[str, str]] = []
        self.members: dict[str, dict[str, bytes]] = {}
        self.shard_by_key: dict[str, int] = {}
        self.evidence_hashes = {
            name: sha256(f"fixture:{name}".encode("ascii"))
            for name in IDENTITY_EVIDENCE_HASH_FIELDS
        }

        molecule_order = (3, 0, 7, 2, 6, 1, 5, 4)
        variant_order = ("stress", "original", "coordgen", "rdkit")
        for molecule_index in molecule_order:
            molecule_id = f"chembl:fixture-{molecule_index:02d}"
            source_record_id = molecule_id
            scaffold_id = f"scaffold-{molecule_index:02d}"
            molecule, chemmol, sdf, inchikey = molecule_payload(
                molecule_id,
                source_record_id,
                self.smiles[molecule_index],
            )
            for variant_index, variant in enumerate(variant_order):
                key = f"dev_{molecule_index:02d}_{variant}"
                shard_index = (molecule_index + variant_index) % 2
                primitives, positions = primitives_payload(key, molecule_id, molecule)
                style, style_id = style_payload(variant)
                visual_v01 = serialize_visual_svg_v0_1(primitives).encode("utf-8")
                visual_v02 = serialize_visual_svg_v0_2(primitives).encode("utf-8")
                render_parameters = {
                    "renderer_id": "rdkit-svg-semantic-0.1",
                    "rdkit_version": Chem.rdBase.rdkitVersion,
                    "font_family_id": "STIX-Regular",
                    "font_file_sha256": DEV_FONT_SHA256,
                    "font_size": 14,
                    "bond_line_width": 2.0,
                    "rotation_degrees": 0.0,
                    "layout": variant,
                    "stress_operation": "long_bond" if variant == "stress" else None,
                    "color_mode": "black_white",
                    "render_profile": DEV_RENDER_PROFILE,
                    "font_sha": DEV_FONT_SHA256,
                    "style_id": style_id,
                }
                source = {
                    "dataset": "webdataset_full_v0/train",
                    "split": "train",
                    "source_split": "train",
                    "source_sample_key": source_record_id,
                    "source_record_id": source_record_id,
                    "source_shard": "shards/train-000000.tar",
                    "structure_ref": chemmol["structure_ref"],
                    "source_sdf_sha256": sha256(sdf),
                }
                layer_source = {
                    "dataset": "webdataset_full_v0",
                    "source_sample_key": source_record_id,
                    "source_shard": "shards/train-000000.tar",
                }
                groups = {
                    int(group["bond_id"]): group for group in primitives["bond_groups"]
                }
                layers = {
                    "schema_version": "chemvision-layers-0.1",
                    "sample_key": key,
                    "source": layer_source,
                    "molecule_id": molecule_id,
                    "role": "dev",
                    "split": "fresh_dev",
                    "variant": variant,
                    "scaffold_id": scaffold_id,
                    "source_record_id": source_record_id,
                    "style_id": style_id,
                    "font_sha": DEV_FONT_SHA256,
                    "render_profile": DEV_RENDER_PROFILE,
                    "style": style,
                    "width": 768,
                    "height": 768,
                    "mask_encoding": {"background": 0, "bond": 1, "text": 2, "overlap": 3},
                    "atoms": [
                        {
                            "atom_id": atom["id"],
                            "element": atom["element"],
                            "formal_charge": atom["formal_charge"],
                            "aromatic": atom["aromatic"],
                            "implicit_h_count": atom["implicit_h_count"],
                            "center": list(positions[int(atom["id"])]),
                            "visible_text": False,
                            "text": "",
                            "bbox": None,
                        }
                        for atom in chemmol["atoms"]
                    ],
                    "bonds": [
                        self._layer_bond(bond, groups[int(bond["id"])], positions)
                        for bond in chemmol["bonds"]
                    ],
                    "render_parameters": render_parameters,
                }
                render = {
                    "schema_version": "visual-svg-layers-c-render-0.1",
                    "sample_key": key,
                    "molecule_id": molecule_id,
                    "role": "dev",
                    "split": "fresh_dev",
                    "variant": variant,
                    "scaffold_id": scaffold_id,
                    "source_record_id": source_record_id,
                    "font_sha": DEV_FONT_SHA256,
                    "render_profile": DEV_RENDER_PROFILE,
                    "style": style,
                    "width": 768,
                    "height": 768,
                    "style_id": style_id,
                    "render_parameters": render_parameters,
                }
                members = {
                    **rendered_layers(primitives),
                    ".primitives.json": json_bytes(primitives),
                    ".visualsvg.txt": visual_v01,
                    ".visualsvg-v02.json": visual_v02,
                    ".layers.json": json_bytes(layers),
                    ".render.json": json_bytes(render),
                    ".chemmol.json": json_bytes(chemmol),
                    ".sdf": sdf,
                }
                meta = {
                    "schema_version": "visual-svg-layers-c-meta-0.1",
                    "sample_key": key,
                    "molecule_id": molecule_id,
                    "role": "dev",
                    "split": "fresh_dev",
                    "variant": variant,
                    "source": source,
                    "fresh_audit_eligible": True,
                    "semantic_svg_sha256": sha256(members[".primitives.json"]),
                    "sdf_inchikey": inchikey,
                    "sdf_protocol": "source-sdf-byte-preserving-0.1",
                    "sequence_tokens": len(
                        VisualSVGTokenizerV01().encode(visual_v01.decode("utf-8"))
                    ),
                    "scaffold_id": scaffold_id,
                    "source_record_id": source_record_id,
                    "style_id": style_id,
                    "font_sha": DEV_FONT_SHA256,
                    "render_profile": DEV_RENDER_PROFILE,
                    "style": style,
                    "sha256": {
                        suffix.removeprefix("."): sha256(members[suffix])
                        for suffix in MEMBER_SUFFIXES[:-1]
                    },
                }
                members[".meta.json"] = json_bytes(meta)
                self.members[key] = members
                self.shard_by_key[key] = shard_index
                self.rows.append(
                    {
                        "sample_key": key,
                        "molecule_id": molecule_id,
                        "inchikey": inchikey,
                        "split": "fresh_dev",
                        "variant": variant,
                        "shard": f"shards/dev-{shard_index:06d}.tar",
                        "scaffold_id": scaffold_id,
                        "source_record_id": source_record_id,
                        "style_id": style_id,
                        "font_sha": DEV_FONT_SHA256,
                        "render_profile": DEV_RENDER_PROFILE,
                        **self._member_manifest_hashes(members),
                    }
                )

        self.commitment = {
            "schema_version": "visual-svg-layers-c-input-commitment-0.1",
            "role": "dev",
            "split": "fresh_dev",
            "fresh_audit_eligible": True,
            "molecule_count": self.role_counts[0],
            "sample_count": self.role_counts[1],
            "expected_total_members": self.role_counts[1] * len(MEMBER_SUFFIXES),
            "width": 768,
            "height": 768,
            "max_tokens": 4096,
            "sdf_protocol": "source-sdf-byte-preserving-0.1",
            "render_profile": DEV_RENDER_PROFILE,
            "variants": list(VARIANTS),
            "members_per_sample": len(MEMBER_SUFFIXES),
            "freeze": {
                "path": "../phase_a_freeze.json",
                "sha256": sha256(self.freeze_path.read_bytes()),
            },
            "source": {"dataset_root": "Data/sdf/webdataset_full_v0", "split": "train"},
            "exclusion_sets": {
                "molecule_id_sha256": self.evidence_hashes[
                    "molecule_id_exclusion_set_sha256"
                ],
                "inchikey_full_sha256": self.evidence_hashes[
                    "inchikey_full_exclusion_set_sha256"
                ],
                "scaffold_id_sha256": self.evidence_hashes[
                    "scaffold_id_exclusion_set_sha256"
                ],
                "source_record_id_sha256": self.evidence_hashes[
                    "source_record_id_exclusion_set_sha256"
                ],
            },
            "selection": {
                "path": self.selection_commitment_path.name,
                "sha256": "0" * 64,
            },
            "fonts": [
                {"family": "STIX", "sha256": DEV_FONT_SHA256},
                {
                    "family": "P052",
                    "sha256": (
                        "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e927"
                        "78a78d149524eea9"
                    ),
                },
            ],
            "implementation": {
                "builder_sha256": sha256(b"builder"),
                "validator_sha256": sha256(b"validator"),
                "isolation_auditor_sha256": self.evidence_hashes[
                    "isolation_algorithm_sha256"
                ],
                "style_id_algorithm_sha256": self.evidence_hashes[
                    "style_id_algorithm_sha256"
                ],
                "scaffold_id_algorithm_sha256": self.evidence_hashes[
                    "scaffold_id_algorithm_sha256"
                ],
                "source_record_normalization_algorithm_sha256": (
                    self.evidence_hashes[
                        "source_record_normalization_algorithm_sha256"
                    ]
                ),
            },
            "shards": [],
            "validation": {
                "path": self.validation_path.name,
                "sha256": "0" * 64,
                "status": "passed",
            },
        }
        self.rebuild_after_member_change()

    @staticmethod
    def _layer_bond(
        bond: dict[str, object],
        group: dict[str, object],
        positions: list[tuple[float, float]],
    ) -> dict[str, object]:
        begin = positions[int(bond["begin_atom"])]
        end = positions[int(bond["end_atom"])]
        return {
            "bond_id": bond["id"],
            "begin_atom": bond["begin_atom"],
            "end_atom": bond["end_atom"],
            "begin": list(begin),
            "end": list(end),
            "chemical_type": bond["type"],
            "display_type": group["display_type"],
            "direction": group["direction"],
            "bbox": [
                max(0.0, min(begin[0], end[0]) - 8.0),
                max(0.0, min(begin[1], end[1]) - 8.0),
                min(768.0, max(begin[0], end[0]) + 8.0),
                min(768.0, max(begin[1], end[1]) + 8.0),
            ],
        }

    @staticmethod
    def _member_manifest_hashes(members: dict[str, bytes]) -> dict[str, str]:
        return {
            "image_sha256": sha256(members[".png"]),
            "sdf_sha256": sha256(members[".sdf"]),
            "primitives_sha256": sha256(members[".primitives.json"]),
            "visualsvg_v01_sha256": sha256(members[".visualsvg.txt"]),
            "visualsvg_v02_sha256": sha256(members[".visualsvg-v02.json"]),
            "layers_sha256": sha256(members[".layers.json"]),
            "render_sha256": sha256(members[".render.json"]),
            "chemmol_sha256": sha256(members[".chemmol.json"]),
            "meta_sha256": sha256(members[".meta.json"]),
        }

    @staticmethod
    def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _selection_rank(row: dict[str, str]) -> str:
        payload = "\0".join(
            (
                "visual-svg-layers-c-v0.1-phase-a-20260722",
                row["molecule_id"],
                row["inchikey_full"],
                row["scaffold_id"],
                row["source_record_id"],
            )
        ).encode("utf-8")
        return sha256(payload)

    @staticmethod
    def _source_bytes_commitment_sha256(rows: list[dict[str, str]]) -> str:
        payload = [
            {
                "molecule_id": row["molecule_id"],
                "source_sdf_sha256": row["source_sdf_sha256"],
                "source_sdf_bytes": row["source_sdf_bytes"],
                "source_chemmol_sha256": row["source_chemmol_sha256"],
                "source_atom_count": row["source_atom_count"],
                "source_bond_count": row["source_bond_count"],
            }
            for row in rows
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded)

    def _selection_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for paired in self.rows:
            molecule_id = paired["molecule_id"]
            if molecule_id in seen:
                continue
            seen.add(molecule_id)
            members = self.members[paired["sample_key"]]
            chemmol = json.loads(members[".chemmol.json"])
            source_record_id = paired["source_record_id"]
            row = {
                "selection_index": "",
                "rank_sha256": "",
                "molecule_id": molecule_id,
                "inchikey_full": paired["inchikey"],
                "scaffold_id": paired["scaffold_id"],
                "source_record_id": source_record_id,
                "source": "chembl",
                "source_id": source_record_id.split(":", 1)[1],
                "source_split": "train",
                "source_shard": "shards/train-000000.tar",
                "source_structure": str(chemmol["structure_ref"]),
                "source_sdf_sha256": sha256(members[".sdf"]),
                "source_sdf_bytes": str(len(members[".sdf"])),
                "source_chemmol_sha256": sha256(members[".chemmol.json"]),
                "source_atom_count": str(len(chemmol["atoms"])),
                "source_bond_count": str(len(chemmol["bonds"])),
            }
            row["rank_sha256"] = self._selection_rank(row)
            rows.append(row)
        rows.sort(key=lambda row: row["rank_sha256"])
        for index, row in enumerate(rows):
            row["selection_index"] = str(index)
        return rows

    def _replacement_rows(self) -> list[dict[str, str]]:
        rows = []
        selected_rank_ceiling = self._selection_rows()[-1]["rank_sha256"]
        candidate_index = 0
        while len(rows) < 500:
            source_id = f"replacement-{candidate_index:06d}"
            row = {
                "replacement_index": "",
                "rank_sha256": "",
                "molecule_id": f"chembl:{source_id}",
                "inchikey_full": f"REPLACEMENT{candidate_index:016d}"[:27],
                "scaffold_id": f"scaffold-replacement-{candidate_index:06d}",
                "source_record_id": f"chembl:{source_id}",
                "source": "chembl",
                "source_id": source_id,
                "source_split": "train",
                "source_shard": "shards/train-000001.tar",
                "source_structure": f"structures/train-000001/{source_id}.sdf",
                "source_sdf_sha256": sha256(
                    f"sdf:{candidate_index}".encode("ascii")
                ),
                "source_sdf_bytes": str(100 + candidate_index),
                "source_chemmol_sha256": sha256(
                    f"chemmol:{candidate_index}".encode("ascii")
                ),
                "source_atom_count": "2",
                "source_bond_count": "1",
                "purpose": "fresh_dev_pre_render_replacement",
                "rendered": "false",
            }
            row["rank_sha256"] = self._selection_rank(row)
            if row["rank_sha256"] > selected_rank_ceiling:
                rows.append(row)
            candidate_index += 1
        rows.sort(key=lambda row: row["rank_sha256"])
        for index, row in enumerate(rows):
            row["replacement_index"] = str(index)
        return rows

    def _write_selection_provenance(
        self,
        selection_rows: list[dict[str, str]],
    ) -> None:
        selection_config = {
            "ordering": "sha256_rank_v0.1",
            "fresh_count": self.role_counts[0],
            "replacement_count": 500,
            "selection_scan_limit": 20_000,
            "exclusion_match_policy": "reject_candidate_on_any_key_match",
            "identity_fields": [
                "molecule_id",
                "inchikey_full",
                "scaffold_id",
                "source_record_id",
            ],
        }
        exclusion_set_sha256 = {
            name: self.evidence_hashes[f"{name}_exclusion_set_sha256"]
            for name in (
                "molecule_id",
                "inchikey_full",
                "scaffold_id",
                "source_record_id",
            )
        }
        self.selection_provenance_path.write_bytes(
            json_bytes(
                {
                    "schema_version": "visual-svg-layers-c-fresh-dev-selection-0.1",
                    "status": "selected_pending_independent_identity_audit",
                    "freeze_id": "visual-svg-layers-c-v0.1-phase-a-20260722",
                    "freeze_sha256": sha256(self.freeze_path.read_bytes()),
                    "source_split": "train",
                    "fresh_dev_molecules": len(selection_rows),
                    "fresh_dev_samples_planned": self.role_counts[1],
                    "fresh_replacement_molecules": 500,
                    "fresh_replacement_rendered": 0,
                    "structural_rejected": 0,
                    "selection_manifest_sha256": sha256(
                        self.selection_manifest_path.read_bytes()
                    ),
                    "selection_source_bytes_commitment_sha256": (
                        self._source_bytes_commitment_sha256(selection_rows)
                    ),
                    "replacement_manifest_sha256": sha256(
                        self.replacement_manifest_path.read_bytes()
                    ),
                    "selection_config": selection_config,
                    "selection_config_sha256": sha256(
                        json.dumps(
                            selection_config,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ),
                    "exclusion_set_sha256": exclusion_set_sha256,
                    "sealed_rows_materialized": False,
                    "reserve_rows_materialized": False,
                }
            )
        )

    def identity_isolation_report(
        self,
        selection_rows: list[dict[str, str]],
    ) -> dict[str, object]:
        config = {
            "freeze_id": "visual-svg-layers-c-v0.1-phase-a-20260722",
            "fresh_count": self.role_counts[0],
            "replacement_count": 500,
            "selection_order": "sha256_rank_v0.1",
            "source_split": "train",
            "identity_fields": [
                "molecule_id",
                "inchikey_full",
                "scaffold_id",
                "source_record_id",
            ],
            "match_policy": "reject_candidate_on_any_key_match",
            "structural_rejections_allowed_for_pass": False,
        }
        self.evidence_hashes["isolation_config_sha256"] = sha256(
            json.dumps(
                config,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        zero_overlap = {
            "molecule_id_overlap": 0,
            "inchikey_full_overlap": 0,
            "scaffold_id_overlap": 0,
            "source_record_id_overlap": 0,
        }
        return {
            "schema_version": (
                "visual-svg-layers-c-fresh-dev-identity-isolation-0.1"
            ),
            "status": "passed",
            "producer": "independent_identity_auditor",
            "freeze_id": "visual-svg-layers-c-v0.1-phase-a-20260722",
            "fresh_dev_selection_manifest_sha256": sha256(
                self.selection_manifest_path.read_bytes()
            ),
            "fresh_dev_source_bytes_commitment_sha256": (
                self._source_bytes_commitment_sha256(selection_rows)
            ),
            "fresh_dev_replacement_manifest_sha256": sha256(
                self.replacement_manifest_path.read_bytes()
            ),
            "selection_provenance_sha256": sha256(
                self.selection_provenance_path.read_bytes()
            ),
            "config": config,
            **self.evidence_hashes,
            "comparisons": {
                "frozen_b_p0_train_exposure": dict(zero_overlap),
                "smoke_train_32": dict(zero_overlap),
            },
            "fresh_replacement_overlap": dict(zero_overlap),
            "fresh_dev_molecules": self.role_counts[0],
            "fresh_replacement_molecules": 500,
            "structure_prechecked_molecules": 2_500,
            "structural_rejections": 0,
            "old_validation_internal_or_sealed_rows_read": False,
            "sealed_rows_materialized": False,
            "reserve_rows_materialized": False,
        }

    def write_selection_commitment(
        self,
        selection_rows: list[dict[str, str]],
    ) -> None:
        payload = {
            "schema_version": (
                "visual-svg-layers-c-fresh-dev-selection-commitment-0.1"
            ),
            "status": "committed",
            "freeze_id": "visual-svg-layers-c-v0.1-phase-a-20260722",
            "role": "dev",
            "split": "fresh_dev",
            "molecule_count": self.role_counts[0],
            "sample_count": self.role_counts[1],
            "selection_ordering": "sha256_rank_v0.1",
            "freeze": {
                "path": "../phase_a_freeze.json",
                "sha256": sha256(self.freeze_path.read_bytes()),
            },
            "selection_manifest": {
                "path": self.selection_manifest_path.name,
                "sha256": sha256(self.selection_manifest_path.read_bytes()),
            },
            "source_bytes_commitment": {
                "protocol": "source-sdf-byte-preserving-0.1",
                "sha256": self._source_bytes_commitment_sha256(selection_rows),
                "molecule_count": self.role_counts[0],
            },
            "fresh_replacement_manifest": {
                "path": self.replacement_manifest_path.name,
                "sha256": sha256(self.replacement_manifest_path.read_bytes()),
                "molecule_count": 500,
                "rendered": False,
            },
            "identity_isolation_report": {
                "path": self.identity_isolation_path.name,
                "sha256": sha256(self.identity_isolation_path.read_bytes()),
                "status": "passed",
            },
            "selection_provenance": {
                "path": self.selection_provenance_path.name,
                "sha256": sha256(self.selection_provenance_path.read_bytes()),
            },
            "preflight": {
                "selection_indices": list(range(16)),
                "molecule_count": 16,
                "sample_count": 64,
                "counts_toward_final_fresh_dev": True,
                "run_model": False,
            },
            "sealed_rows_materialized": False,
            "reserve_rows_materialized": False,
        }
        self.selection_commitment_path.write_bytes(json_bytes(payload))
        self.commitment["selection"] = {
            "path": self.selection_commitment_path.name,
            "sha256": sha256(self.selection_commitment_path.read_bytes()),
        }

    def write_preflight_validation_report(self) -> None:
        self.preflight_validation_path.write_bytes(
            json_bytes(
                {
                    "schema_version": (
                        "visual-svg-layers-c-fresh-dev-preflight-validation-0.1"
                    ),
                    "status": "PREFLIGHT-PASS",
                    "role": "dev",
                    "split": "fresh_dev",
                    "training_steps": 0,
                    "parameter_updates": 0,
                    "model_run": False,
                    "threshold_or_checkpoint_selection": False,
                    "sealed_rows_read": False,
                    "molecules": 16,
                    "samples": 64,
                    "checked_samples": 64,
                    "selection_commitment_sha256": sha256(
                        self.selection_commitment_path.read_bytes()
                    ),
                    "errors": [],
                }
            )
        )

    def write_selection_artifacts(self) -> None:
        selection_rows = self._selection_rows()
        self._write_csv(
            self.selection_manifest_path,
            SELECTION_FIELDS,
            selection_rows,
        )
        self._write_csv(
            self.replacement_manifest_path,
            REPLACEMENT_FIELDS,
            self._replacement_rows(),
        )
        self._write_selection_provenance(selection_rows)
        self.identity_isolation_path.write_bytes(
            json_bytes(self.identity_isolation_report(selection_rows))
        )
        self.write_selection_commitment(selection_rows)
        self.write_preflight_validation_report()

    def rebind_existing_selection_artifacts(
        self,
        *,
        rewrite_identity_report: bool,
    ) -> None:
        with self.selection_manifest_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            selection_rows = list(csv.DictReader(handle))
        self._write_selection_provenance(selection_rows)
        if rewrite_identity_report:
            self.identity_isolation_path.write_bytes(
                json_bytes(self.identity_isolation_report(selection_rows))
            )
        self.write_selection_commitment(selection_rows)
        self.write_preflight_validation_report()
        self.refresh_dynamic_references()
        self.write_validation_report()
        self.write_commitment()

    def write_manifests(self) -> None:
        self._write_csv(self.manifest_path, self.paired_fields, self.rows)
        by_molecule = {}
        for row in self.rows:
            by_molecule.setdefault(row["molecule_id"], row)
        selection_rows = self._selection_rows()
        molecule_rows = [
            {
                field: by_molecule[selected["molecule_id"]][field]
                for field in self.molecule_fields
            }
            for selected in selection_rows
        ]
        sample_rows = [
            {field: row[field] for field in self.sample_fields} for row in self.rows
        ]
        self._write_csv(
            self.molecule_manifest_path,
            self.molecule_fields,
            molecule_rows,
        )
        self._write_csv(self.sample_manifest_path, self.sample_fields, sample_rows)

    def write_shards(self) -> None:
        for shard_index, path in enumerate(self.shard_paths):
            with tarfile.open(path, "w") as archive:
                keys = sorted(
                    key
                    for key, current_shard in self.shard_by_key.items()
                    if current_shard == shard_index
                )
                for key in keys:
                    for suffix in MEMBER_SUFFIXES:
                        payload = self.members[key][suffix]
                        info = tarfile.TarInfo(key + suffix)
                        info.size = len(payload)
                        info.mtime = 0
                        info.mode = 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, io.BytesIO(payload))

    def refresh_shard_commitments(self) -> None:
        self.commitment["shards"] = []
        for shard_index, path in enumerate(self.shard_paths):
            sample_count = sum(
                value == shard_index for value in self.shard_by_key.values()
            )
            self.commitment["shards"].append(
                {
                    "path": f"shards/dev-{shard_index:06d}.tar",
                    "sha256": sha256(path.read_bytes()),
                    "bytes": path.stat().st_size,
                    "samples": sample_count,
                    "members": sample_count * len(MEMBER_SUFFIXES),
                }
            )

    def isolation_report(self) -> dict[str, object]:
        hashes = {
            name: sha256(f"isolation:{name}".encode("ascii"))
            for name in ISOLATION_HASH_FIELDS
        }
        hashes.update(
            {
                "fresh_dev_paired_manifest_sha256": sha256(self.manifest_path.read_bytes()),
                "fresh_dev_molecule_manifest_sha256": sha256(
                    self.molecule_manifest_path.read_bytes()
                ),
                "fresh_dev_sample_manifest_sha256": sha256(
                    self.sample_manifest_path.read_bytes()
                ),
            }
        )
        hashes.update(self.evidence_hashes)
        zero_overlap = {name: 0 for name in ISOLATION_OVERLAP_FIELDS}
        return {
            "schema_version": "visual-svg-layers-c-isolation-report-0.1",
            "producer": "independent_isolation_auditor",
            "role": "fresh_dev",
            "status": "passed",
            "source_split": "train",
            "overlap_counts": {
                "frozen_b_p0_train_exposure": dict(zero_overlap),
                "smoke_train_32": dict(zero_overlap),
            },
            "hashes": hashes,
        }

    def write_isolation_report(self, payload: dict[str, object] | None = None) -> None:
        report = payload or self.isolation_report()
        self.isolation_path.write_bytes(json_bytes(report))
        self.commitment["isolation_report"] = {
            "path": self.isolation_path.name,
            "sha256": sha256(self.isolation_path.read_bytes()),
            "status": "passed",
        }
        hashes = report["hashes"]
        self.commitment["exclusion_sets"] = {
            "molecule_id_sha256": hashes["molecule_id_exclusion_set_sha256"],
            "inchikey_full_sha256": hashes["inchikey_full_exclusion_set_sha256"],
            "scaffold_id_sha256": hashes["scaffold_id_exclusion_set_sha256"],
            "source_record_id_sha256": hashes[
                "source_record_id_exclusion_set_sha256"
            ],
        }
        self.commitment["implementation"].update(
            {
                "isolation_auditor_sha256": hashes["isolation_algorithm_sha256"],
                "style_id_algorithm_sha256": hashes["style_id_algorithm_sha256"],
                "scaffold_id_algorithm_sha256": hashes[
                    "scaffold_id_algorithm_sha256"
                ],
                "source_record_normalization_algorithm_sha256": hashes[
                    "source_record_normalization_algorithm_sha256"
                ],
            }
        )

    def refresh_dynamic_references(self) -> None:
        self.commitment["selection"] = {
            "path": self.selection_commitment_path.name,
            "sha256": sha256(self.selection_commitment_path.read_bytes()),
        }
        self.commitment["molecule_manifest"] = {
            "path": self.molecule_manifest_path.name,
            "sha256": sha256(self.molecule_manifest_path.read_bytes()),
        }
        self.commitment["sample_manifest"] = {
            "path": self.sample_manifest_path.name,
            "sha256": sha256(self.sample_manifest_path.read_bytes()),
        }
        self.commitment["paired_manifest"] = {
            "path": self.manifest_path.name,
            "sha256": sha256(self.manifest_path.read_bytes()),
        }
        self.commitment["preflight_validation"] = {
            "path": "preflight_64/preflight_validation.json",
            "sha256": sha256(self.preflight_validation_path.read_bytes()),
            "status": "PREFLIGHT-PASS",
        }

    def write_validation_report(self) -> None:
        report = {
            "schema_version": "visual-svg-layers-c-fresh-dev-validation-0.1",
            "status": "passed",
            "fresh_audit_eligible": True,
            "molecules": self.role_counts[0],
            "samples": self.role_counts[1],
            "members": self.role_counts[1] * len(MEMBER_SUFFIXES),
            "checked_samples": self.role_counts[1],
            "checked_members": self.role_counts[1] * len(MEMBER_SUFFIXES),
            "structure_validated_samples": self.role_counts[1],
            "errors": [],
            "artifact_sha256": {
                "paired_manifest_sha256": sha256(self.manifest_path.read_bytes()),
                "molecule_manifest_sha256": sha256(
                    self.molecule_manifest_path.read_bytes()
                ),
                "sample_manifest_sha256": sha256(
                    self.sample_manifest_path.read_bytes()
                ),
                "preflight_validation_sha256": sha256(
                    self.preflight_validation_path.read_bytes()
                ),
                "isolation_report_sha256": sha256(
                    self.isolation_path.read_bytes()
                ),
                "selection_commitment_sha256": sha256(
                    self.selection_commitment_path.read_bytes()
                ),
                "selection_manifest_sha256": sha256(
                    self.selection_manifest_path.read_bytes()
                ),
                "replacement_manifest_sha256": sha256(
                    self.replacement_manifest_path.read_bytes()
                ),
                "identity_isolation_report_sha256": sha256(
                    self.identity_isolation_path.read_bytes()
                ),
                "selection_provenance_sha256": sha256(
                    self.selection_provenance_path.read_bytes()
                ),
            },
            "implementation_sha256": dict(self.commitment["implementation"]),
            "font_inventory": list(self.commitment["fonts"]),
            "shards": [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "samples": row["samples"],
                    "members": row["members"],
                }
                for row in self.commitment["shards"]
            ],
        }
        self.validation_path.write_bytes(json_bytes(report))
        self.commitment["validation"] = {
            "path": self.validation_path.name,
            "sha256": sha256(self.validation_path.read_bytes()),
            "status": "passed",
        }

    def rebuild_after_member_change(self) -> None:
        for row in self.rows:
            key = row["sample_key"]
            members = self.members[key]
            meta = json.loads(members[".meta.json"])
            meta["sha256"] = {
                suffix.removeprefix("."): sha256(members[suffix])
                for suffix in MEMBER_SUFFIXES[:-1]
            }
            members[".meta.json"] = json_bytes(meta)
            row.update(self._member_manifest_hashes(members))
        self.write_manifests()
        self.write_shards()
        if not self.selection_commitment_path.exists():
            self.write_selection_artifacts()
        self.write_preflight_validation_report()
        self.refresh_dynamic_references()
        self.write_isolation_report()
        self.refresh_shard_commitments()
        self.write_validation_report()
        self.write_commitment()

    def write_commitment(self) -> Path:
        self.commitment_path.write_bytes(json_bytes(self.commitment))
        return self.commitment_path

    def expected_commitment_sha256(self) -> str:
        return sha256(self.commitment_path.read_bytes())


class FreshEvidenceCarrierPackageTest(unittest.TestCase):
    def load_dev(
        self,
        fixture: DevFixture,
        *,
        expected_sha256: str | None = None,
    ):
        with patch.dict(
            fresh_dev_data.ROLE_COUNTS,
            {"dev": DevFixture.role_counts},
        ), patch.object(
            fresh_dev_data,
            "_validate_phase_a_freeze",
        ):
            return load_fresh_evidence_carrier_package(
                fixture.commitment_path,
                expected_role="dev",
                expected_commitment_sha256=(
                    expected_sha256 or fixture.expected_commitment_sha256()
                ),
            )

    def test_load_and_iterate_exact_manifest_order_without_directory_scans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            expected_commitment_sha = sha256(fixture.commitment_path.read_bytes())
            with (
                patch.object(Path, "glob", side_effect=AssertionError("glob forbidden")),
                patch.object(Path, "rglob", side_effect=AssertionError("rglob forbidden")),
                patch.object(Path, "iterdir", side_effect=AssertionError("iterdir forbidden")),
            ):
                package = load_fresh_evidence_carrier_package(
                    fixture.commitment_path,
                    expected_role="smoke",
                    expected_commitment_sha256=expected_commitment_sha,
                )
                samples = list(package.iter_samples())

            self.assertEqual(package.sample_count, 32)
            self.assertEqual(package.molecule_count, 8)
            self.assertFalse(package.fresh_audit_eligible)
            self.assertEqual(
                [sample["key"] for sample in samples],
                [row["sample_key"] for row in fixture.rows],
            )
            first = samples[0]
            self.assertEqual(first["image"].mode, "RGB")
            self.assertEqual(first["bond_layer"].mode, "L")
            self.assertEqual(first["identity"].sample_key, first["key"])
            self.assertEqual(first["identity"].split, "smoke_train")
            self.assertEqual(len(first["member_sha256"]), 12)
            self.assertRegex(first["input_fingerprint"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                len({sample["input_fingerprint"] for sample in samples}),
                len(samples),
            )
            convenience_keys = [
                sample["key"]
                for sample in iter_fresh_evidence_carrier_samples(
                    fixture.commitment_path,
                    expected_role="smoke",
                )
            ]
            self.assertEqual(convenience_keys, [row["sample_key"] for row in fixture.rows])

    def test_rejects_commitment_manifest_shard_and_validation_hash_mismatches(self) -> None:
        mutators = (
            (
                "commitment",
                lambda fixture: load_fresh_evidence_carrier_package(
                    fixture.commitment_path,
                    expected_commitment_sha256="0" * 64,
                ),
                "Commitment SHA256 mismatch",
            ),
            (
                "manifest",
                lambda fixture: (
                    fixture.manifest_path.write_bytes(
                        fixture.manifest_path.read_bytes() + b"\n"
                    ),
                    load_fresh_evidence_carrier_package(fixture.commitment_path),
                )[-1],
                "manifest SHA256 mismatch",
            ),
            (
                "shard",
                lambda fixture: (
                    fixture.commitment["shards"][0].__setitem__("sha256", "0" * 64),
                    fixture.write_commitment(),
                    load_fresh_evidence_carrier_package(fixture.commitment_path),
                )[-1],
                "Shard SHA256 mismatch",
            ),
            (
                "validation",
                lambda fixture: (
                    fixture.validation_path.write_bytes(
                        fixture.validation_path.read_bytes() + b"\n"
                    ),
                    load_fresh_evidence_carrier_package(fixture.commitment_path),
                )[-1],
                "Validation report SHA256 mismatch",
            ),
        )
        for name, mutate_and_load, message in mutators:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = SmokeFixture(Path(directory) / "smoke_train_32")
                with self.assertRaisesRegex(FreshEvidenceCarrierDataError, message):
                    mutate_and_load(fixture)

    def test_rejects_path_escape_symlink_and_validation_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            fixture.commitment["paired_manifest"]["path"] = "../outside.csv"
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "Path escape"):
                load_fresh_evidence_carrier_package(fixture.commitment_path)

        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            target = fixture.root / "manifest-target.csv"
            fixture.manifest_path.replace(target)
            fixture.manifest_path.symlink_to(target.name)
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "Symlink rejected"):
                load_fresh_evidence_carrier_package(fixture.commitment_path)

        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            old = fixture.shard_paths[0]
            forbidden = old.with_name("validation-smoke-000000.tar")
            old.replace(forbidden)
            fixture.commitment["shards"][0]["path"] = (
                "shards/validation-smoke-000000.tar"
            )
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "Validation shard"):
                load_fresh_evidence_carrier_package(fixture.commitment_path)

    def test_rejects_uncommitted_shard_bad_layout_closure_and_wrong_role_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            fixture.rows[0]["shard"] = "shards/smoke-uncommitted.tar"
            fixture.write_manifest()
            fixture.refresh_manifest_commitment()
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "uncommitted shard"):
                load_fresh_evidence_carrier_package(fixture.commitment_path)

        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            fixture.rows[0]["variant"] = fixture.rows[1]["variant"]
            fixture.write_manifest()
            fixture.refresh_manifest_commitment()
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "four layouts"):
                load_fresh_evidence_carrier_package(fixture.commitment_path)

        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            fixture.commitment["role"] = "dev"
            fixture.commitment["fresh_audit_eligible"] = True
            dev_commitment = fixture.write_commitment(
                fixture.root / "commitment.dev.json"
            )
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "8000"):
                load_fresh_evidence_carrier_package(
                    dev_commitment,
                    expected_role="dev",
                )

    def test_rejects_extra_tar_member_before_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            fixture.write_shards(extra_member=True)
            fixture.refresh_shard_commitments()
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "exactly 12 members"):
                load_fresh_evidence_carrier_package(fixture.commitment_path)

    def test_iteration_rejects_manifest_member_hash_and_meta_internal_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            fixture.rows[0]["image_sha256"] = "0" * 64
            fixture.write_manifest()
            fixture.refresh_manifest_commitment()
            fixture.write_commitment()
            package = load_fresh_evidence_carrier_package(fixture.commitment_path)
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "Manifest hash mismatch"):
                next(package.iter_samples())

        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            key = fixture.rows[0]["sample_key"]
            meta = json.loads(fixture.members[key][".meta.json"])
            meta["sha256"]["png"] = "0" * 64
            fixture.members[key][".meta.json"] = json_bytes(meta)
            fixture.rebuild_after_member_change()
            package = load_fresh_evidence_carrier_package(fixture.commitment_path)
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "Meta hash mismatch"):
                next(package.iter_samples())

    def test_iteration_rejects_forbidden_sidecar_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SmokeFixture(Path(directory) / "smoke_train_32")
            key = fixture.rows[0]["sample_key"]
            meta = json.loads(fixture.members[key][".meta.json"])
            meta["source"] = {"dataset": "sealed/internal_test"}
            fixture.members[key][".meta.json"] = json_bytes(meta)
            fixture.rebuild_after_member_change()
            package = load_fresh_evidence_carrier_package(fixture.commitment_path)
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "Forbidden"):
                next(package.iter_samples())

    def test_loads_legal_dev_package_with_verified_zero_overlap_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")

            package = self.load_dev(fixture)
            samples = list(package.iter_samples())

            self.assertTrue(package.fresh_audit_eligible)
            self.assertTrue(package.independence_verified)
            self.assertEqual(package.isolation_path, fixture.isolation_path.resolve())
            self.assertEqual(
                package.isolation_sha256,
                sha256(fixture.isolation_path.read_bytes()),
            )
            self.assertEqual(len(samples), DevFixture.role_counts[1])
            self.assertTrue(
                all(sample["identity"].split == "fresh_dev" for sample in samples)
            )

    def test_dev_layers_source_uses_dataset_root_not_split_qualified_meta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            key = "dev_03_stress"
            layers = json.loads(fixture.members[key][".layers.json"])
            layers["source"]["dataset"] = "webdataset_full_v0/train"
            fixture.members[key][".layers.json"] = json_bytes(layers)
            fixture.rebuild_after_member_change()

            package = self.load_dev(fixture)
            with self.assertRaisesRegex(
                FreshEvidenceCarrierDataError,
                "layers source dataset root mismatch",
            ):
                list(package.iter_samples())

    def test_dev_rejects_chemmol_bond_charge_stereo_and_sdf_mismatches(self) -> None:
        mutations = (
            (
                "bond_type",
                "dev_03_stress",
                lambda payload: payload["bonds"][0].__setitem__("type", "SINGLE"),
            ),
            (
                "formal_charge",
                "dev_03_stress",
                lambda payload: payload["atoms"][0].__setitem__("formal_charge", 1),
            ),
            (
                "stereo",
                "dev_07_stress",
                lambda payload: next(
                    bond
                    for bond in payload["bonds"]
                    if bond["stereo"] != "STEREONONE"
                ).update({"stereo": "STEREONONE", "stereo_atoms": [], "ez": None}),
            ),
        )
        for name, key, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = DevFixture(Path(directory) / "fresh_dev")
                chemmol = json.loads(fixture.members[key][".chemmol.json"])
                mutate(chemmol)
                fixture.members[key][".chemmol.json"] = json_bytes(chemmol)
                fixture.rebuild_after_member_change()

                package = self.load_dev(fixture)
                with self.assertRaisesRegex(
                    FreshEvidenceCarrierDataError,
                    "source SDF/ChemMol byte commitment|ChemMol/SDF|ChemMol reconstructed",
                ):
                    list(package.iter_samples())

        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            replacement = fixture.members["dev_00_stress"][".sdf"]
            replacement_hash = sha256(replacement)
            for variant in VARIANTS:
                key = f"dev_03_{variant}"
                fixture.members[key][".sdf"] = replacement
                meta = json.loads(fixture.members[key][".meta.json"])
                meta["source"]["source_sdf_sha256"] = replacement_hash
                fixture.members[key][".meta.json"] = json_bytes(meta)
            fixture.rebuild_after_member_change()

            with self.assertRaisesRegex(
                FreshEvidenceCarrierDataError,
                "molecule manifest differs from committed selection",
            ):
                self.load_dev(fixture)

    def test_dev_rejects_visualsvg_primitives_mismatch_after_hash_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            key = "dev_03_stress"
            primitives = json.loads(fixture.members[key][".primitives.json"])
            primitives["primitives"][0]["quantized_points"][0][0] += 1
            fixture.members[key][".primitives.json"] = json_bytes(primitives)
            fixture.rebuild_after_member_change()

            package = self.load_dev(fixture)
            with self.assertRaisesRegex(
                FreshEvidenceCarrierDataError,
                "Visual SVG v0.[12] differs from primitives",
            ):
                list(package.iter_samples())

    def test_dev_rejects_layers_and_render_identity_drift(self) -> None:
        mutations = (
            ("layers_role", ".layers.json", "role", "train"),
            ("render_source", ".render.json", "source_record_id", "other-source"),
        )
        for name, suffix, field, value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = DevFixture(Path(directory) / "fresh_dev")
                key = "dev_03_stress"
                payload = json.loads(fixture.members[key][suffix])
                payload[field] = value
                fixture.members[key][suffix] = json_bytes(payload)
                fixture.rebuild_after_member_change()

                package = self.load_dev(fixture)
                with self.assertRaisesRegex(
                    FreshEvidenceCarrierDataError,
                    "Fresh-dev (layers|render).*(role|source_record_id) mismatch",
                ):
                    list(package.iter_samples())

    def test_dev_rejects_fake_passed_preflight_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            report = json.loads(fixture.preflight_validation_path.read_bytes())
            report["model_run"] = True
            fixture.preflight_validation_path.write_bytes(json_bytes(report))
            fixture.commitment["preflight_validation"]["sha256"] = sha256(
                fixture.preflight_validation_path.read_bytes()
            )
            fixture.write_commitment()

            with self.assertRaisesRegex(
                FreshEvidenceCarrierDataError,
                "Preflight validation field 'model_run' mismatch",
            ):
                self.load_dev(fixture)

    def test_dev_rejects_unbound_molecule_and_sample_manifests(self) -> None:
        mutations = (
            ("molecule", "scaffold_id", "wrong-scaffold"),
            ("sample", "style_id", "style-v0.1:" + "f" * 64),
        )
        for manifest, field, value in mutations:
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as directory:
                fixture = DevFixture(Path(directory) / "fresh_dev")
                path = (
                    fixture.molecule_manifest_path
                    if manifest == "molecule"
                    else fixture.sample_manifest_path
                )
                fields = (
                    fixture.molecule_fields
                    if manifest == "molecule"
                    else fixture.sample_fields
                )
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0][field] = value
                fixture._write_csv(path, fields, rows)
                fixture.refresh_dynamic_references()
                fixture.write_isolation_report()
                fixture.refresh_shard_commitments()
                fixture.write_validation_report()
                fixture.write_commitment()

                with self.assertRaisesRegex(
                    FreshEvidenceCarrierDataError,
                    "Fresh-dev (molecule|sample) manifest differs",
                ):
                    self.load_dev(fixture)

    def test_dev_rejects_missing_nonpassed_or_overlapping_isolation_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            fixture.commitment.pop("isolation_report")
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "isolation"):
                self.load_dev(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            fixture.isolation_path.unlink()
            with self.assertRaisesRegex(
                FreshEvidenceCarrierDataError,
                "Missing isolation report",
            ):
                self.load_dev(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            report = fixture.isolation_report()
            report["status"] = "failed"
            fixture.write_isolation_report(report)
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "status|passed"):
                self.load_dev(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            report = fixture.isolation_report()
            report["overlap_counts"]["frozen_b_p0_train_exposure"][
                "scaffold_id"
            ] = 1
            fixture.write_isolation_report(report)
            fixture.write_commitment()
            with self.assertRaisesRegex(FreshEvidenceCarrierDataError, "overlap"):
                self.load_dev(fixture)

    def test_dev_rejects_unregistered_commitment_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = DevFixture(Path(directory) / "fresh_dev")
            with self.assertRaisesRegex(
                FreshEvidenceCarrierDataError,
                "Commitment SHA256 mismatch",
            ):
                self.load_dev(fixture, expected_sha256="0" * 64)

    def test_rejects_forbidden_commitment_parent_paths(self) -> None:
        for marker in (
            "validation",
            "validation500",
            "old_validation500",
            "internal_test",
            "internaltest",
            "sealed",
        ):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                fixture = SmokeFixture(
                    Path(directory) / f"synthetic_{marker}" / "smoke_train_32"
                )
                with self.assertRaises(FreshEvidenceCarrierDataError):
                    load_fresh_evidence_carrier_package(
                        fixture.commitment_path,
                        expected_role="smoke",
                        expected_commitment_sha256=sha256(
                            fixture.commitment_path.read_bytes()
                        ),
                    )

    def test_iteration_rejects_forbidden_source_splits_in_sidecars(self) -> None:
        for split in (
            "validation",
            "validation500",
            "old_validation500",
            "internal_test",
            "internaltest",
            "sealed",
        ):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                fixture = SmokeFixture(Path(directory) / "smoke_train_32")
                key = fixture.rows[0]["sample_key"]
                layers = json.loads(fixture.members[key][".layers.json"])
                layers["source"] = {
                    "dataset": "webdataset_full_v0",
                    "split": split,
                }
                fixture.members[key][".layers.json"] = json_bytes(layers)
                meta = json.loads(fixture.members[key][".meta.json"])
                meta["source"] = {
                    "dataset": "webdataset_full_v0",
                    "split": split,
                }
                meta["sha256"] = {
                    suffix.removeprefix("."): sha256(fixture.members[key][suffix])
                    for suffix in MEMBER_SUFFIXES[:-1]
                }
                fixture.members[key][".meta.json"] = json_bytes(meta)
                fixture.rebuild_after_member_change()

                package = load_fresh_evidence_carrier_package(
                    fixture.commitment_path,
                    expected_role="smoke",
                )
                with self.assertRaises(FreshEvidenceCarrierDataError):
                    next(package.iter_samples())


if __name__ == "__main__":
    unittest.main()
