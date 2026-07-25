#!/usr/bin/env python3
"""Independently validate a train-only Visual SVG Layers C smoke shard."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
from typing import Any, Mapping, Sequence

import fastjsonschema
from PIL import Image
from rdkit import Chem

try:
    from .visual_primitives_v0_1 import (
        parsed_visual_svg_from_primitives,
        validate_visual_primitive_references,
    )
    from .visual_svg_sequence_v0_1 import parse_visual_svg_v0_1
    from .visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
        serialize_parsed_visual_svg_v0_2,
        visual_primitives_v0_1_to_visual_svg_v0_2,
    )
except ImportError:
    from visual_primitives_v0_1 import (
        parsed_visual_svg_from_primitives,
        validate_visual_primitive_references,
    )
    from visual_svg_sequence_v0_1 import parse_visual_svg_v0_1
    from visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
        serialize_parsed_visual_svg_v0_2,
        visual_primitives_v0_1_to_visual_svg_v0_2,
    )


SCHEMA_VERSION = "visual-svg-layers-c-independent-validation-0.1"
PASS_STATUS = "SMOKE-PASS"
FAIL_STATUS = "FAIL"
VARIANTS = ("original", "rdkit", "coordgen", "stress")
MEMBER_SUFFIXES = (
    ".png",
    ".text.png",
    ".bond.png",
    ".semantic.png",
    ".primitives.json",
    ".visualsvg.txt",
    ".visualsvg-v02.json",
    ".layers.json",
    ".render.json",
    ".chemmol.json",
    ".sdf",
    ".meta.json",
)
HASHED_SUFFIXES = MEMBER_SUFFIXES[:-1]
FORBIDDEN_PATH_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "reserve",
    "mr_tadf",
    "mr-tadf",
)
SAMPLE_FIELDS = {
    "sample_key",
    "molecule_id",
    "split",
    "variant",
    "sdf_inchikey",
}
PAIRED_FIELDS = {
    "sample_key",
    "molecule_id",
    "inchikey",
    "split",
    "variant",
    "image_sha256",
    "sdf_sha256",
    "primitives_sha256",
    "visualsvg_v01_sha256",
    "visualsvg_v02_sha256",
    "render_sha256",
    "meta_sha256",
}


class IndependentValidationError(ValueError):
    """Raised when the independent smoke contract is violated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_marker(value: object) -> str | None:
    lowered = str(value).lower()
    return next(
        (marker for marker in FORBIDDEN_PATH_MARKERS if marker in lowered),
        None,
    )


def _assert_train_only_text(value: object, *, context: str) -> None:
    marker = _forbidden_marker(value)
    if marker is not None:
        raise IndependentValidationError(
            f"forbidden_non_train_reference:{context}:{marker}"
        )


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _assert_no_symlink_components(path: Path, *, must_exist: bool) -> Path:
    absolute = _absolute_without_resolving(path)
    _assert_train_only_text(absolute, context="input_path")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if must_exist:
                raise IndependentValidationError(f"missing_input:{absolute}")
            break
        if stat.S_ISLNK(mode):
            raise IndependentValidationError(f"symlink_rejected:{current}")
    if must_exist:
        try:
            mode = absolute.stat().st_mode
        except FileNotFoundError as error:
            raise IndependentValidationError(f"missing_input:{absolute}") from error
        if not stat.S_ISREG(mode):
            raise IndependentValidationError(f"input_is_not_regular_file:{absolute}")
    return absolute


def _validate_output_path(path: Path, inputs: Sequence[Path]) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if current.is_symlink():
            raise IndependentValidationError(f"symlink_rejected:{current}")
    if absolute in inputs:
        raise IndependentValidationError("output_must_not_replace_an_input")
    return absolute


def _read_csv(path: Path, required: set[str], *, label: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise IndependentValidationError(
                f"{label}_missing_fields:{','.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise IndependentValidationError(f"{label}_is_empty")
    for row_index, row in enumerate(rows, 2):
        for field, value in row.items():
            _assert_train_only_text(
                value,
                context=f"{label}:{row_index}:{field}",
            )
    return rows


def _compile_schema(filename: str):
    schema_path = Path(__file__).resolve().parent / filename
    return fastjsonschema.compile(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )


def _image_contract(data: bytes, *, suffix: str) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG":
            raise IndependentValidationError(f"image_format:{suffix}:{image.format}")
        if image.size != (768, 768):
            raise IndependentValidationError(
                f"image_size:{suffix}:{image.size[0]}x{image.size[1]}"
            )
        expected_mode = "RGB" if suffix == ".png" else "L"
        if image.mode != expected_mode:
            raise IndependentValidationError(
                f"image_mode:{suffix}:{image.mode}:expected={expected_mode}"
            )


def _parse_single_sdf(data: bytes, *, sample_key: str) -> tuple[Chem.Mol, str]:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise IndependentValidationError(f"sdf_single_sanitized_molecule:{sample_key}")
    molecule = molecules[0]
    Chem.SanitizeMol(molecule)
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey:
        raise IndependentValidationError(f"sdf_missing_inchikey:{sample_key}")
    return molecule, inchikey


def _mol_from_chemmol(data: Mapping[str, Any]) -> Chem.Mol:
    atoms = data.get("atoms")
    bonds = data.get("bonds")
    if not isinstance(atoms, list) or not isinstance(bonds, list):
        raise IndependentValidationError("chemmol_atoms_or_bonds_not_arrays")
    if [atom.get("id") for atom in atoms] != list(range(len(atoms))):
        raise IndependentValidationError("chemmol_atom_ids_not_contiguous")
    if [bond.get("id") for bond in bonds] != list(range(len(bonds))):
        raise IndependentValidationError("chemmol_bond_ids_not_contiguous")

    editable = Chem.RWMol()
    for atom_data in atoms:
        atom = Chem.Atom(int(atom_data["atomic_number"]))
        atom.SetFormalCharge(int(atom_data["formal_charge"]))
        atom.SetIsotope(int(atom_data["isotope"]))
        atom.SetIsAromatic(bool(atom_data["aromatic"]))
        try:
            atom.SetChiralTag(getattr(Chem.ChiralType, atom_data["chiral_tag"]))
        except AttributeError as error:
            raise IndependentValidationError(
                f"chemmol_chiral_tag:{atom_data['chiral_tag']}"
            ) from error
        atom.SetNumExplicitHs(int(atom_data["explicit_h_count"]))
        atom.SetNoImplicit(bool(atom_data["no_implicit"]))
        atom.SetNumRadicalElectrons(int(atom_data["radical_electrons"]))
        editable.AddAtom(atom)
    for bond_data in bonds:
        begin = int(bond_data["begin_atom"])
        end = int(bond_data["end_atom"])
        if not 0 <= begin < len(atoms) or not 0 <= end < len(atoms) or begin == end:
            raise IndependentValidationError("chemmol_bond_endpoint")
        try:
            bond_type = getattr(Chem.BondType, bond_data["type"])
        except AttributeError as error:
            raise IndependentValidationError(
                f"chemmol_bond_type:{bond_data['type']}"
            ) from error
        editable.AddBond(begin, end, bond_type)
        bond = editable.GetBondBetweenAtoms(begin, end)
        bond.SetIsAromatic(bool(bond_data["aromatic"]))
        bond.SetIsConjugated(bool(bond_data["conjugated"]))
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    for bond_data in bonds:
        bond = molecule.GetBondWithIdx(int(bond_data["id"]))
        stereo_atoms = bond_data["stereo_atoms"]
        if len(stereo_atoms) == 2:
            bond.SetStereoAtoms(int(stereo_atoms[0]), int(stereo_atoms[1]))
        try:
            bond.SetStereo(getattr(Chem.BondStereo, bond_data["stereo"]))
        except AttributeError as error:
            raise IndependentValidationError(
                f"chemmol_bond_stereo:{bond_data['stereo']}"
            ) from error
    Chem.AssignStereochemistry(molecule, cleanIt=False, force=True)
    return molecule


def _split_member_name(name: str) -> tuple[str, str]:
    if not name or "/" in name or "\\" in name:
        raise IndependentValidationError(f"tar_member_path:{name}")
    for suffix in sorted(MEMBER_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            key = name[: -len(suffix)]
            if not key:
                break
            return key, suffix
    raise IndependentValidationError(f"unexpected_tar_member:{name}")


def _validate_tar_metadata(member: tarfile.TarInfo) -> None:
    if not member.isfile():
        raise IndependentValidationError(f"tar_member_not_regular:{member.name}")
    if (
        member.mtime != 0
        or member.mode != 0o644
        or member.uid != 0
        or member.gid != 0
        or member.uname not in (None, "")
        or member.gname not in (None, "")
        or member.linkname
        or member.pax_headers
    ):
        raise IndependentValidationError(
            f"nondeterministic_tar_metadata:{member.name}"
        )


def _load_json(data: bytes, *, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentValidationError(f"invalid_json:{label}:{error}") from error


def _validate_identity(
    payload: Mapping[str, Any],
    row: Mapping[str, str],
    *,
    label: str,
    require_sample_key: bool,
) -> None:
    if require_sample_key and payload.get("sample_key") != row["sample_key"]:
        raise IndependentValidationError(f"{label}_sample_key")
    if payload.get("molecule_id") != row["molecule_id"]:
        raise IndependentValidationError(f"{label}_molecule_id")
    if "variant" in payload and payload.get("variant") != row["variant"]:
        raise IndependentValidationError(f"{label}_variant")


def _validate_one_sample(
    members: Mapping[str, bytes],
    row: Mapping[str, str],
    primitive_validator,
    chemmol_validator,
) -> dict[str, Any]:
    key = row["sample_key"]
    for suffix in (".png", ".text.png", ".bond.png", ".semantic.png"):
        _image_contract(members[suffix], suffix=suffix)

    primitives = _load_json(members[".primitives.json"], label=f"{key}.primitives")
    primitive_validator(primitives)
    validate_visual_primitive_references(primitives)
    _validate_identity(primitives, row, label="primitives", require_sample_key=True)

    try:
        visual_v01 = members[".visualsvg.txt"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise IndependentValidationError("visualsvg_v01_utf8") from error
    parsed_v01 = parse_visual_svg_v0_1(visual_v01)
    if parsed_v01 != parsed_visual_svg_from_primitives(primitives):
        raise IndependentValidationError("visualsvg_v01_primitives_mismatch")

    try:
        visual_v02 = members[".visualsvg-v02.json"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise IndependentValidationError("visualsvg_v02_utf8") from error
    parsed_v02 = parse_visual_svg_v0_2(visual_v02)
    if serialize_parsed_visual_svg_v0_2(parsed_v02) != visual_v02:
        raise IndependentValidationError("visualsvg_v02_not_exact_canonical")
    if parsed_v02 != visual_primitives_v0_1_to_visual_svg_v0_2(primitives):
        raise IndependentValidationError("visualsvg_v02_primitives_mismatch")
    if flatten_visual_svg_v0_2_to_v0_1(parsed_v02) != parsed_v01:
        raise IndependentValidationError("visualsvg_v02_flatten_mismatch")

    layers = _load_json(members[".layers.json"], label=f"{key}.layers")
    render = _load_json(members[".render.json"], label=f"{key}.render")
    chemmol = _load_json(members[".chemmol.json"], label=f"{key}.chemmol")
    meta = _load_json(members[".meta.json"], label=f"{key}.meta")
    chemmol_validator(chemmol)
    for payload, label, require_key in (
        (layers, "layers", True),
        (render, "render", False),
        (chemmol, "chemmol", False),
        (meta, "meta", True),
    ):
        _validate_identity(payload, row, label=label, require_sample_key=require_key)
    if layers.get("split") not in {"train", "smoke_train"}:
        raise IndependentValidationError(f"layers_non_train_split:{layers.get('split')}")
    if meta.get("schema_version") != "visual-svg-layers-c-meta-0.1":
        raise IndependentValidationError("meta_schema_version")
    if meta.get("split") != row["split"]:
        raise IndependentValidationError("meta_split")
    if meta.get("fresh_audit_eligible") is not False:
        raise IndependentValidationError("smoke_marked_fresh_audit_eligible")
    for payload, label in ((layers, "layers"), (meta, "meta"), (chemmol, "chemmol")):
        for name, value in payload.items():
            if name in {"source", "structure_ref", "split"}:
                _assert_train_only_text(value, context=f"{key}:{label}:{name}")
    meta_source = meta.get("source")
    if not isinstance(meta_source, dict):
        raise IndependentValidationError("meta_source")
    if not str(meta_source.get("source_sample_key", "")).startswith("train_"):
        raise IndependentValidationError("meta_source_sample_key_not_train")
    if not str(meta_source.get("source_shard", "")).startswith("shards/train-"):
        raise IndependentValidationError("meta_source_shard_not_train")
    if not str(meta_source.get("structure_ref", "")).startswith("structures/train-"):
        raise IndependentValidationError("meta_structure_ref_not_train")

    expected_hash_names = {suffix.removeprefix(".") for suffix in HASHED_SUFFIXES}
    hashes = meta.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_names:
        raise IndependentValidationError("meta_hash_fields")
    actual_hashes = {
        suffix.removeprefix("."): sha256_bytes(members[suffix])
        for suffix in HASHED_SUFFIXES
    }
    if hashes != actual_hashes:
        mismatches = sorted(
            name for name in expected_hash_names if hashes.get(name) != actual_hashes[name]
        )
        raise IndependentValidationError(f"meta_hash_mismatch:{','.join(mismatches)}")

    sdf_molecule, sdf_inchikey = _parse_single_sdf(members[".sdf"], sample_key=key)
    chemmol_molecule = _mol_from_chemmol(chemmol)
    chemmol_inchikey = Chem.MolToInchiKey(chemmol_molecule)
    if not chemmol_inchikey:
        raise IndependentValidationError("chemmol_missing_inchikey")
    declared_inchikey = str(chemmol.get("inchikey", ""))
    if not (
        sdf_inchikey
        == chemmol_inchikey
        == declared_inchikey
        == row["sdf_inchikey"]
        == meta.get("sdf_inchikey")
    ):
        raise IndependentValidationError(
            "sdf_chemmol_inchikey_mismatch:"
            f"sdf={sdf_inchikey}:chemmol={chemmol_inchikey}:"
            f"declared={declared_inchikey}:manifest={row['sdf_inchikey']}"
        )
    if (
        sdf_molecule.GetNumAtoms() != chemmol_molecule.GetNumAtoms()
        or sdf_molecule.GetNumBonds() != chemmol_molecule.GetNumBonds()
    ):
        raise IndependentValidationError("sdf_chemmol_size_mismatch")

    return {
        "sdf_sha256": sha256_bytes(members[".sdf"]),
        "chemmol_sha256": sha256_bytes(members[".chemmol.json"]),
        "inchikey": sdf_inchikey,
        "member_sha256": {
            suffix: sha256_bytes(members[suffix]) for suffix in MEMBER_SUFFIXES
        },
    }


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "fresh_audit_eligible": False,
        "molecules": 0,
        "samples": 0,
        "tar_members": 0,
        "checked_samples": 0,
        "errors": [],
    }


def validate_smoke_package(
    shard_path: str | Path,
    samples_manifest_path: str | Path,
    paired_manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate only the three explicitly supplied train-only input files."""

    report = _base_report()
    errors: list[str] = report["errors"]
    try:
        shard = _assert_no_symlink_components(Path(shard_path), must_exist=True)
        samples_path = _assert_no_symlink_components(
            Path(samples_manifest_path), must_exist=True
        )
        paired_path = _assert_no_symlink_components(
            Path(paired_manifest_path), must_exist=True
        )
        if len({shard, samples_path, paired_path}) != 3:
            raise IndependentValidationError("input_paths_must_be_distinct")
        sample_rows = _read_csv(samples_path, SAMPLE_FIELDS, label="samples_manifest")
        paired_rows = _read_csv(paired_path, PAIRED_FIELDS, label="paired_manifest")
    except Exception as error:
        errors.append(f"input_contract:{type(error).__name__}:{error}")
        return report

    sample_keys = [row["sample_key"] for row in sample_rows]
    if sample_keys != sorted(sample_keys):
        errors.append("samples_manifest_order")
    if len(sample_keys) != len(set(sample_keys)):
        errors.append("samples_manifest_duplicate_key")
    samples_by_key = {row["sample_key"]: row for row in sample_rows}
    for row in sample_rows:
        if not row["sample_key"] or "/" in row["sample_key"] or "\\" in row["sample_key"]:
            errors.append(f"unsafe_sample_key:{row['sample_key']}")
        if row["split"] != "smoke_train":
            errors.append(f"samples_manifest_non_train_split:{row['sample_key']}")
        if row["variant"] not in VARIANTS:
            errors.append(f"samples_manifest_variant:{row['sample_key']}")

    paired_keys = [row["sample_key"] for row in paired_rows]
    if paired_keys != sorted(paired_keys):
        errors.append("paired_manifest_order")
    if len(paired_keys) != len(set(paired_keys)):
        errors.append("paired_manifest_duplicate_key")
    paired_by_key = {row["sample_key"]: row for row in paired_rows}
    if set(paired_by_key) != set(samples_by_key):
        errors.append("paired_manifest_sample_set")

    expected_names = [
        key + suffix for key in sorted(samples_by_key) for suffix in MEMBER_SUFFIXES
    ]
    sample_results: dict[str, dict[str, Any]] = {}
    raw_sdf_hashes: dict[str, str] = {}
    try:
        primitive_validator = _compile_schema(
            "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json"
        )
        chemmol_validator = _compile_schema("schemas/chemmol-0.1.schema.json")
        with tarfile.open(shard, "r") as archive:
            tar_members = archive.getmembers()
            report["tar_members"] = len(tar_members)
            names = [member.name for member in tar_members]
            if names != expected_names:
                errors.append("tar_member_order_or_names")
            if len(names) != len(set(names)):
                errors.append("duplicate_tar_member")
            member_by_name: dict[str, tarfile.TarInfo] = {}
            for member in tar_members:
                try:
                    _split_member_name(member.name)
                    _validate_tar_metadata(member)
                except Exception as error:
                    errors.append(
                        f"tar_contract:{type(error).__name__}:{error}"
                    )
                member_by_name[member.name] = member

            for key in sorted(samples_by_key):
                row = samples_by_key[key]
                members: dict[str, bytes] = {}
                missing = False
                for suffix in MEMBER_SUFFIXES:
                    name = key + suffix
                    member = member_by_name.get(name)
                    if member is None:
                        errors.append(f"missing_member:{name}")
                        missing = True
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        errors.append(f"unreadable_member:{name}")
                        missing = True
                        continue
                    members[suffix] = extracted.read()
                if missing:
                    continue
                raw_sdf_hashes[key] = sha256_bytes(members[".sdf"])
                try:
                    sample_results[key] = _validate_one_sample(
                        members,
                        row,
                        primitive_validator,
                        chemmol_validator,
                    )
                    report["checked_samples"] += 1
                except Exception as error:
                    errors.append(
                        f"sample:{key}:{type(error).__name__}:{error}"
                    )
    except Exception as error:
        errors.append(f"tar_read:{type(error).__name__}:{error}")

    variants_by_molecule: dict[str, list[str]] = defaultdict(list)
    sdf_hashes_by_molecule: dict[str, set[str]] = defaultdict(set)
    inchikeys_by_molecule: dict[str, set[str]] = defaultdict(set)
    for key, row in samples_by_key.items():
        variants_by_molecule[row["molecule_id"]].append(row["variant"])
        if key in raw_sdf_hashes:
            sdf_hashes_by_molecule[row["molecule_id"]].add(raw_sdf_hashes[key])
        result = sample_results.get(key)
        if result is not None:
            inchikeys_by_molecule[row["molecule_id"]].add(result["inchikey"])
    for molecule_id, variants in variants_by_molecule.items():
        if len(variants) != 4 or set(variants) != set(VARIANTS):
            errors.append(
                f"variant_closure:{molecule_id}:{','.join(sorted(variants))}"
            )
        if len(sdf_hashes_by_molecule[molecule_id]) != 1:
            errors.append(f"sdf_hash_not_shared_across_variants:{molecule_id}")
        if len(inchikeys_by_molecule[molecule_id]) != 1:
            errors.append(f"inchikey_not_shared_across_variants:{molecule_id}")

    paired_hash_fields = {
        "image_sha256": ".png",
        "sdf_sha256": ".sdf",
        "primitives_sha256": ".primitives.json",
        "visualsvg_v01_sha256": ".visualsvg.txt",
        "visualsvg_v02_sha256": ".visualsvg-v02.json",
        "render_sha256": ".render.json",
        "meta_sha256": ".meta.json",
    }
    for key, sample_row in samples_by_key.items():
        paired = paired_by_key.get(key)
        result = sample_results.get(key)
        if paired is None:
            continue
        for field in ("molecule_id", "split", "variant"):
            if paired[field] != sample_row[field]:
                errors.append(f"paired_manifest_{field}:{key}")
        if paired["inchikey"] != sample_row["sdf_inchikey"]:
            errors.append(f"paired_manifest_inchikey:{key}")
        if result is None:
            continue
        if paired["inchikey"] != result["inchikey"]:
            errors.append(f"paired_manifest_actual_inchikey:{key}")
        for field, suffix in paired_hash_fields.items():
            if paired[field] != result["member_sha256"][suffix]:
                errors.append(f"paired_manifest_hash:{key}:{field}")

    report.update(
        {
            "molecules": len(variants_by_molecule),
            "samples": len(sample_rows),
            "variant_closure_passed": all(
                len(values) == 4 and set(values) == set(VARIANTS)
                for values in variants_by_molecule.values()
            ),
            "sdf_hash_shared_across_variants": all(
                len(values) == 1 for values in sdf_hashes_by_molecule.values()
            )
            and len(sample_results) == len(sample_rows),
            "paired_manifest_rows": len(paired_rows),
            "shard_sha256": file_sha256(shard),
            "samples_manifest_sha256": file_sha256(samples_path),
            "paired_manifest_sha256": file_sha256(paired_path),
        }
    )
    if not errors:
        report["status"] = PASS_STATUS
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--samples-manifest", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [Path(args.shard), Path(args.samples_manifest), Path(args.paired_manifest)]
    try:
        output = _validate_output_path(
            Path(args.output),
            [_absolute_without_resolving(path) for path in inputs],
        )
        report = validate_smoke_package(*inputs)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception as error:
        report = _base_report()
        report["errors"].append(
            f"output_contract:{type(error).__name__}:{error}"
        )
        print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)
        return 1
    print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)
    return 0 if report["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FAIL_STATUS",
    "MEMBER_SUFFIXES",
    "PASS_STATUS",
    "build_parser",
    "main",
    "validate_smoke_package",
]
