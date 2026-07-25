#!/usr/bin/env python3
"""Independently validate the committed fresh-dev 16/64 preflight package.

Only explicitly supplied fresh-dev and full-v0 train files are opened.  The
validator does not import the preflight builder and does not run any model.
"""

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
import tarfile
from typing import Any, Mapping, Sequence

import fastjsonschema
from PIL import Image
from rdkit import Chem

try:
    from .build_webdataset_sample import mol_from_chemmol
    from .visual_primitives_v0_1 import (
        parsed_visual_svg_from_primitives,
        validate_visual_primitive_references,
    )
    from .visual_svg_sequence_v0_1 import parse_visual_svg_v0_1
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
    from .visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
        serialize_parsed_visual_svg_v0_2,
        visual_primitives_v0_1_to_visual_svg_v0_2,
    )
except ImportError:
    from build_webdataset_sample import mol_from_chemmol
    from visual_primitives_v0_1 import (
        parsed_visual_svg_from_primitives,
        validate_visual_primitive_references,
    )
    from visual_svg_sequence_v0_1 import parse_visual_svg_v0_1
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
    from visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
        serialize_parsed_visual_svg_v0_2,
        visual_primitives_v0_1_to_visual_svg_v0_2,
    )


SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-preflight-validation-0.1"
PASS_STATUS = "PREFLIGHT-PASS"
FAIL_STATUS = "FAIL"
FREEZE_ID = "visual-svg-layers-c-v0.1-phase-a-20260722"
FREEZE_SHA256 = "6c87e2e63b42c6b9f39c33d829fc8883d456b7c5f82cd15b2672963a7f560b9c"
SOURCE_VALIDATION_SHA256 = "8e37f9c2037e01d8f417b882e1405fa625a8a5f81f283f647d5788f972f6aa78"
TRAIN_MANIFEST_INVENTORY_SHA256 = "cee36b8e8737c3de1826d868955dcc27dcf656d63d8a498f08425708d14acbdb"
SELECTION_SCHEMA_VERSION = (
    "visual-svg-layers-c-fresh-dev-selection-commitment-0.1"
)
IDENTITY_AUDIT_SCHEMA_VERSION = (
    "visual-svg-layers-c-fresh-dev-identity-isolation-0.1"
)
SPLIT = "fresh_dev"
ROLE = "dev"
VARIANTS = ("original", "rdkit", "coordgen", "stress")
MOLECULE_COUNT = 16
SAMPLE_COUNT = 64
COMMITTED_MOLECULE_COUNT = 2000
COMMITTED_SAMPLE_COUNT = 8000
MAX_TOKENS = 4096
RENDER_PROFILE = "visual-svg-layers-c-clean-bw-768-v0.1"
SDF_PROTOCOL = "source-sdf-byte-preserving-0.1"
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
FORBIDDEN_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "reserve",
    "mr_tadf",
    "mr-tadf",
)
SELECTION_FIELDS = {
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
}
MOLECULE_FIELDS = SELECTION_FIELDS | {
    "role",
    "split",
    "variants",
    "sdf_sha256",
    "sdf_bytes",
    "atom_count",
    "bond_count",
}
SAMPLE_FIELDS = {
    "sample_key",
    "selection_index",
    "molecule_id",
    "role",
    "split",
    "variant",
    "scaffold_id",
    "source_record_id",
    "source_structure",
    "source_sdf_sha256",
    "style_id",
    "font_sha",
    "render_profile",
    "shard",
    "sdf_inchikey",
    "sequence_tokens",
}
PAIRED_FIELDS = {
    "sample_key",
    "molecule_id",
    "inchikey",
    "scaffold_id",
    "source_record_id",
    "role",
    "split",
    "variant",
    "shard",
    "style_id",
    "font_sha",
    "render_profile",
    "image_sha256",
    "sdf_sha256",
    "primitives_sha256",
    "visualsvg_v01_sha256",
    "visualsvg_v02_sha256",
    "layers_sha256",
    "render_sha256",
    "chemmol_sha256",
    "meta_sha256",
}
FONT_ALLOWLIST = {
    "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45": {
        "file": "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf",
        "family": "STIX",
    },
    "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e92778a78d149524eea9": {
        "file": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
        "family": "P052",
    },
}


class PreflightValidationError(ValueError):
    """Raised for a protocol violation in the preflight package."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _forbidden_marker(value: object) -> str | None:
    lowered = str(value).lower()
    return next((item for item in FORBIDDEN_MARKERS if item in lowered), None)


def _assert_allowed_text(value: object, *, context: str) -> None:
    marker = _forbidden_marker(value)
    if marker is not None:
        raise PreflightValidationError(
            f"forbidden_reference:{context}:{marker}"
        )


def _regular_file(path: Path, *, allow_aggregate_validation: bool = False) -> Path:
    absolute = Path(os.path.abspath(path))
    if not allow_aggregate_validation:
        _assert_allowed_text(absolute, context="input_path")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise PreflightValidationError(f"missing_input:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise PreflightValidationError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise PreflightValidationError(f"input_not_regular:{absolute}")
    return absolute


def _safe_relative(base: Path, relative: object, *, context: str) -> Path:
    text = str(relative)
    _assert_allowed_text(text, context=context)
    value = Path(text)
    if value.is_absolute() or ".." in value.parts:
        raise PreflightValidationError(f"unsafe_relative_path:{context}:{text}")
    result = _regular_file(base / value)
    if os.path.commonpath((str(base), str(result))) != str(base):
        raise PreflightValidationError(f"path_escapes_base:{context}:{text}")
    return result


def _safe_reference(
    base: Path,
    relative: object,
    *,
    context: str,
    containment_root: Path,
) -> Path:
    """Resolve a declared parent reference inside one version directory."""

    text = str(relative)
    _assert_allowed_text(text, context=context)
    value = Path(text)
    if value.is_absolute():
        raise PreflightValidationError(f"absolute_reference:{context}:{text}")
    containment = Path(os.path.abspath(containment_root))
    candidate = Path(os.path.abspath(base / value))
    if os.path.commonpath((str(containment), str(candidate))) != str(containment):
        raise PreflightValidationError(f"reference_escapes_version:{context}:{text}")
    return _regular_file(candidate)


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightValidationError(f"invalid_json:{path}") from error
    if not isinstance(value, dict):
        raise PreflightValidationError(f"json_root_not_object:{path}")
    return value


def _validate_frozen_source(source: Path) -> None:
    expected = (Path(__file__).resolve().parent / "webdataset_full_v0").resolve()
    if source != expected:
        raise PreflightValidationError(
            f"source_root_not_frozen:{source}:{expected}"
        )
    validation_path = _regular_file(
        source / "validation.json", allow_aggregate_validation=True
    )
    if file_sha256(validation_path) != SOURCE_VALIDATION_SHA256:
        raise PreflightValidationError("source_validation_sha256")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(validation, dict):
        raise PreflightValidationError("source_validation_root")
    if validation.get("status") != "passed" or validation.get("scaffold_overlap") != 0:
        raise PreflightValidationError("source_validation_contract")
    paths = sorted(
        [
            *(source / "manifests/shards").glob("train-*.molecules.csv"),
            *(source / "manifests/shards").glob("train-*.samples.csv"),
        ]
    )
    if len(paths) != 260:
        raise PreflightValidationError("train_manifest_inventory_count")
    digest = hashlib.sha256()
    for path in paths:
        path = _regular_file(path)
        relative = path.relative_to(source).as_posix()
        digest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
    if digest.hexdigest() != TRAIN_MANIFEST_INVENTORY_SHA256:
        raise PreflightValidationError("train_manifest_inventory_sha256")


def _json_member(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightValidationError(f"invalid_json:{label}") from error
    if not isinstance(value, dict):
        raise PreflightValidationError(f"json_root_not_object:{label}")
    return value


def _read_csv(
    path: Path,
    required: set[str],
    *,
    label: str,
) -> list[dict[str, str]]:
    with _regular_file(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise PreflightValidationError(
                f"{label}_missing_fields:{','.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise PreflightValidationError(f"{label}_empty")
    for row_number, row in enumerate(rows, 2):
        for field in required:
            if not str(row.get(field, "")).strip():
                raise PreflightValidationError(
                    f"{label}_empty_field:{row_number}:{field}"
                )
        for field, value in row.items():
            _assert_allowed_text(value, context=f"{label}:{row_number}:{field}")
    return rows


def _selection_rank(row: Mapping[str, str]) -> str:
    payload = "\0".join(
        (
            FREEZE_ID,
            row["molecule_id"],
            row["inchikey_full"],
            row["scaffold_id"],
            row["source_record_id"],
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_bytes_commitment_sha256(rows: Sequence[Mapping[str, str]]) -> str:
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
    return sha256_bytes(_canonical_json_bytes(payload))


def _validate_freeze(freeze: Mapping[str, Any]) -> None:
    if (
        freeze.get("freeze_id") != FREEZE_ID
        or freeze.get("phase") != "A"
        or freeze.get("status")
        != "frozen_fresh_dev_preflight_authorized"
        or freeze.get("training_steps") != 0
        or freeze.get("parameter_updates") != 0
    ):
        raise PreflightValidationError("freeze_static_contract")
    split = (freeze.get("splits") or {}).get(SPLIT) or {}
    preflight = freeze.get("preflight") or {}
    if (
        split.get("authorized") is not True
        or split.get("role") != ROLE
        or int(split.get("molecule_count", -1)) != COMMITTED_MOLECULE_COUNT
        or int(split.get("sample_count", -1)) != COMMITTED_SAMPLE_COUNT
        or preflight.get("run_model") is not False
        or int(preflight.get("molecule_count", -1)) != MOLECULE_COUNT
        or int(preflight.get("sample_count", -1)) != SAMPLE_COUNT
        or preflight.get("selection")
        != "first_16_committed_fresh_dev_molecules"
    ):
        raise PreflightValidationError("freeze_preflight_contract")
    if (freeze.get("sdf") or {}).get("protocol") != SDF_PROTOCOL:
        raise PreflightValidationError("freeze_sdf_protocol")
    if (freeze.get("render") or {}).get("render_profile") != RENDER_PROFILE:
        raise PreflightValidationError("freeze_render_profile")
    if int((freeze.get("sequence") or {}).get("max_tokens", -1)) != MAX_TOKENS:
        raise PreflightValidationError("freeze_max_tokens")


def _load_selection(
    commitment_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    commitment = _json_file(commitment_path)
    if (
        commitment.get("schema_version") != SELECTION_SCHEMA_VERSION
        or commitment.get("status") != "committed"
        or commitment.get("freeze_id") != FREEZE_ID
        or commitment.get("role") != ROLE
        or commitment.get("split") != SPLIT
        or int(commitment.get("molecule_count", -1)) != COMMITTED_MOLECULE_COUNT
        or int(commitment.get("sample_count", -1)) != COMMITTED_SAMPLE_COUNT
        or commitment.get("selection_ordering") != "sha256_rank_v0.1"
    ):
        raise PreflightValidationError("selection_commitment_contract")
    if (commitment.get("freeze") or {}).get("sha256") != file_sha256(freeze_path):
        raise PreflightValidationError("selection_freeze_sha256")
    if file_sha256(freeze_path) != FREEZE_SHA256:
        raise PreflightValidationError("phase_a_freeze_sha256")
    manifest_ref = commitment.get("selection_manifest") or {}
    manifest = _safe_relative(
        commitment_path.parent,
        manifest_ref.get("path", ""),
        context="selection_manifest",
    )
    manifest_sha = file_sha256(manifest)
    if manifest_ref.get("sha256") != manifest_sha:
        raise PreflightValidationError("selection_manifest_sha256")
    rows = _read_csv(manifest, SELECTION_FIELDS, label="selection_manifest")
    if len(rows) != COMMITTED_MOLECULE_COUNT:
        raise PreflightValidationError("selection_manifest_count")
    if [int(row["selection_index"]) for row in rows] != list(
        range(COMMITTED_MOLECULE_COUNT)
    ):
        raise PreflightValidationError("selection_index_sequence")
    if [row["rank_sha256"] for row in rows] != sorted(
        row["rank_sha256"] for row in rows
    ):
        raise PreflightValidationError("selection_rank_order")
    for row in rows:
        if (
            row["rank_sha256"] != _selection_rank(row)
            or row["source_split"] != "train"
            or row["source_record_id"] != f"{row['source']}:{row['source_id']}"
            or not row["source_shard"].startswith("shards/train-")
            or not row["source_structure"].startswith("structures/train-")
        ):
            raise PreflightValidationError(
                f"selection_row_contract:{row['selection_index']}"
            )
    for field in (
        "molecule_id",
        "inchikey_full",
        "source_record_id",
    ):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise PreflightValidationError(f"selection_duplicate:{field}")
    source_commitment = commitment.get("source_bytes_commitment") or {}
    if (
        source_commitment.get("protocol") != SDF_PROTOCOL
        or int(source_commitment.get("molecule_count", -1))
        != COMMITTED_MOLECULE_COUNT
        or source_commitment.get("sha256")
        != _source_bytes_commitment_sha256(rows)
    ):
        raise PreflightValidationError("source_bytes_commitment")

    audit_ref = commitment.get("identity_isolation_report") or {}
    audit_path = _safe_relative(
        commitment_path.parent,
        audit_ref.get("path", ""),
        context="identity_isolation_report",
    )
    if audit_ref.get("sha256") != file_sha256(audit_path):
        raise PreflightValidationError("identity_audit_sha256")
    audit = _json_file(audit_path)
    if (
        audit.get("schema_version") != IDENTITY_AUDIT_SCHEMA_VERSION
        or audit.get("status") != "passed"
        or audit.get("fresh_dev_selection_manifest_sha256") != manifest_sha
    ):
        raise PreflightValidationError("identity_audit_contract")
    comparisons = audit.get("comparisons")
    if not isinstance(comparisons, dict):
        raise PreflightValidationError("identity_audit_comparisons")
    for exposure in ("frozen_b_p0_train_exposure", "smoke_train_32"):
        result = comparisons.get(exposure)
        if not isinstance(result, dict):
            raise PreflightValidationError(f"identity_audit_missing:{exposure}")
        for name in (
            "molecule_id_overlap",
            "inchikey_full_overlap",
            "scaffold_id_overlap",
            "source_record_id_overlap",
        ):
            if result.get(name) != 0:
                raise PreflightValidationError(
                    f"identity_audit_nonzero:{exposure}:{name}"
                )
    for name in (
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
    ):
        value = audit.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise PreflightValidationError(f"identity_audit_hash:{name}")
    return rows, commitment


def _image_contract(data: bytes, *, suffix: str) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.size != (768, 768):
            raise PreflightValidationError(f"image_contract:{suffix}")
        expected_mode = "RGB" if suffix == ".png" else "L"
        if image.mode != expected_mode:
            raise PreflightValidationError(
                f"image_mode:{suffix}:{image.mode}:{expected_mode}"
            )
        if suffix == ".semantic.png":
            extrema = image.getextrema()
            if not isinstance(extrema, tuple) or extrema[0] < 0 or extrema[1] > 3:
                raise PreflightValidationError("semantic_pixel_range")


def _split_member_name(name: str) -> tuple[str, str]:
    if not name or "/" in name or "\\" in name:
        raise PreflightValidationError(f"tar_member_path:{name}")
    for suffix in sorted(MEMBER_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix) and name[: -len(suffix)]:
            return name[: -len(suffix)], suffix
    raise PreflightValidationError(f"unexpected_tar_member:{name}")


def _tar_metadata(member: tarfile.TarInfo) -> None:
    if not member.isfile():
        raise PreflightValidationError(f"tar_member_not_regular:{member.name}")
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
        raise PreflightValidationError(
            f"nondeterministic_tar_metadata:{member.name}"
        )


def _parse_sdf(data: bytes, *, sample_key: str) -> tuple[Chem.Mol, str]:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data), sanitize=True, removeHs=False, strictParsing=True
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise PreflightValidationError(f"sdf_not_single_sanitized:{sample_key}")
    molecule = molecules[0]
    Chem.SanitizeMol(molecule)
    key = Chem.MolToInchiKey(molecule)
    if not key:
        raise PreflightValidationError(f"sdf_missing_inchikey:{sample_key}")
    return molecule, key


def _enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def _chemmol_sdf_contract(
    chemmol: Mapping[str, Any],
    sdf: Chem.Mol,
    *,
    expected_inchikey: str,
) -> None:
    atoms = chemmol.get("atoms")
    bonds = chemmol.get("bonds")
    if not isinstance(atoms, list) or not isinstance(bonds, list):
        raise PreflightValidationError("chemmol_arrays")
    if len(atoms) != sdf.GetNumAtoms() or len(bonds) != sdf.GetNumBonds():
        raise PreflightValidationError("chemmol_sdf_count")
    if chemmol.get("inchikey") != expected_inchikey:
        raise PreflightValidationError("chemmol_inchikey")
    reconstructed = mol_from_chemmol(dict(chemmol))
    Chem.SanitizeMol(reconstructed)
    if Chem.MolToInchiKey(reconstructed) != expected_inchikey:
        raise PreflightValidationError("chemmol_reconstructed_inchikey")
    Chem.AssignStereochemistry(sdf, cleanIt=False, force=True)
    if [int(atom["id"]) for atom in atoms] != list(range(len(atoms))):
        raise PreflightValidationError("chemmol_atom_ids")
    if [int(bond["id"]) for bond in bonds] != list(range(len(bonds))):
        raise PreflightValidationError("chemmol_bond_ids")
    source_atom_indices: set[int] = set()
    for atom_data in atoms:
        atom_id = int(atom_data["id"])
        source_index = int(atom_data["source_atom_index"])
        if source_index in source_atom_indices:
            raise PreflightValidationError("chemmol_source_atom_duplicate")
        source_atom_indices.add(source_index)
        if atom_id < 0 or atom_id >= sdf.GetNumAtoms():
            raise PreflightValidationError("chemmol_atom_id_range")
        atom = sdf.GetAtomWithIdx(atom_id)
        actual = (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetIsAromatic(),
            _enum_name(atom.GetChiralTag()),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
            atom.GetNoImplicit(),
            atom.GetNumRadicalElectrons(),
            atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
        )
        expected = (
            int(atom_data["atomic_number"]),
            int(atom_data["formal_charge"]),
            int(atom_data["isotope"]),
            bool(atom_data["aromatic"]),
            str(atom_data["chiral_tag"]),
            int(atom_data["explicit_h_count"]),
            int(atom_data["implicit_h_count"]),
            bool(atom_data["no_implicit"]),
            int(atom_data["radical_electrons"]),
            atom_data.get("cip_code"),
        )
        if actual != expected:
            raise PreflightValidationError(f"chemmol_atom:{atom_id}")
    if source_atom_indices != set(range(len(atoms))):
        raise PreflightValidationError("chemmol_source_atom_permutation")
    for bond_data in bonds:
        begin = int(bond_data["begin_atom"])
        end = int(bond_data["end_atom"])
        bond = sdf.GetBondBetweenAtoms(begin, end)
        if bond is None:
            raise PreflightValidationError("chemmol_bond_endpoint")
        stereo_name = _enum_name(bond.GetStereo())
        actual_ez = (
            "E"
            if stereo_name == "STEREOE"
            else "Z"
            if stereo_name == "STEREOZ"
            else None
        )
        actual = (
            _enum_name(bond.GetBondType()),
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
            stereo_name,
            tuple(int(value) for value in bond.GetStereoAtoms()),
            actual_ez,
        )
        expected = (
            str(bond_data["type"]),
            bool(bond_data["aromatic"]),
            bool(bond_data["conjugated"]),
            str(bond_data["stereo"]),
            tuple(int(value) for value in bond_data["stereo_atoms"]),
            bond_data.get("ez"),
        )
        if actual != expected:
            raise PreflightValidationError(f"chemmol_bond:{bond_data['id']}")


def _style_id(descriptor: Mapping[str, Any]) -> str:
    expected_fields = {
        "schema",
        "canvas",
        "font_sha256",
        "font_family",
        "render_profile",
        "layout_variant",
        "font_size_bucket",
        "bond_line_width_bucket",
        "atom_label_style",
        "bond_style",
        "color_mode",
        "background",
        "antialias",
    }
    if set(descriptor) != expected_fields:
        raise PreflightValidationError("style_descriptor_fields")
    forbidden = {"molecule_id", "sample_key", "split", "random_seed"}
    if forbidden & set(descriptor):
        raise PreflightValidationError("style_descriptor_forbidden_field")
    if descriptor.get("render_profile") != RENDER_PROFILE:
        raise PreflightValidationError("style_render_profile")
    if descriptor.get("color_mode") != "black_white":
        raise PreflightValidationError("style_color_mode")
    if descriptor.get("background") != "opaque_white":
        raise PreflightValidationError("style_background")
    font_sha = str(descriptor.get("font_sha256", ""))
    font = FONT_ALLOWLIST.get(font_sha)
    if font is None or descriptor.get("font_family") != font["family"]:
        raise PreflightValidationError("style_font_allowlist")
    return "style-v0.1:" + sha256_bytes(_canonical_json_bytes(descriptor))


def _validate_bond_groups(
    primitives: Mapping[str, Any],
    parsed_v02: Any,
    *,
    sdf_bonds: int,
) -> tuple[int, int]:
    source_groups = primitives.get("bond_groups")
    if not isinstance(source_groups, list) or len(source_groups) != sdf_bonds:
        raise PreflightValidationError("bond_group_count")
    if len(parsed_v02.bond_groups) != len(source_groups):
        raise PreflightValidationError("v02_bond_group_count")
    doubles = triples = 0
    for source_group, group in zip(source_groups, parsed_v02.bond_groups):
        if int(source_group["begin_atom"]) == int(source_group["end_atom"]):
            raise PreflightValidationError("bond_group_endpoint_identity")
        if group.kind == "parallel_2":
            doubles += 1
            if len(group.members) != 2:
                raise PreflightValidationError("double_member_count")
        elif group.kind == "parallel_3":
            triples += 1
            if len(group.members) != 3:
                raise PreflightValidationError("triple_member_count")
        elif group.kind in {"single_line", "solid_triangle", "solid_quad"}:
            if len(group.members) != 1:
                raise PreflightValidationError("single_member_count")
    return doubles, triples


def _sidecar_contract(
    payload: Mapping[str, Any],
    row: Mapping[str, str],
    *,
    label: str,
) -> Mapping[str, Any]:
    for field, expected in (
        ("molecule_id", row["molecule_id"]),
        ("role", ROLE),
        ("split", SPLIT),
        ("variant", row["variant"]),
        ("scaffold_id", row["scaffold_id"]),
        ("source_record_id", row["source_record_id"]),
        ("style_id", row["style_id"]),
        ("font_sha", row["font_sha"]),
        ("render_profile", RENDER_PROFILE),
    ):
        if field == "variant" and field not in payload:
            continue
        if payload.get(field) != expected:
            raise PreflightValidationError(f"{label}_{field}")
    descriptor = payload.get("style")
    if not isinstance(descriptor, dict):
        raise PreflightValidationError(f"{label}_style")
    if _style_id(descriptor) != row["style_id"]:
        raise PreflightValidationError(f"{label}_style_id")
    return descriptor


def _validate_sample(
    members: Mapping[str, bytes],
    row: Mapping[str, str],
    paired: Mapping[str, str],
    molecule_row: Mapping[str, str],
    source: Path,
    primitive_validator,
    chemmol_validator,
    tokenizer: VisualSVGTokenizerV01,
) -> dict[str, Any]:
    key = row["sample_key"]
    for suffix in (".png", ".text.png", ".bond.png", ".semantic.png"):
        _image_contract(members[suffix], suffix=suffix)
    primitives = _json_member(members[".primitives.json"], label=f"{key}.primitives")
    primitive_validator(primitives)
    validate_visual_primitive_references(primitives)
    if (
        primitives.get("sample_key") != key
        or primitives.get("molecule_id") != row["molecule_id"]
    ):
        raise PreflightValidationError("primitives_identity")

    try:
        visual_v01 = members[".visualsvg.txt"].decode("utf-8")
        visual_v02 = members[".visualsvg-v02.json"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise PreflightValidationError("visualsvg_utf8") from error
    parsed_v01 = parse_visual_svg_v0_1(visual_v01)
    if parsed_v01 != parsed_visual_svg_from_primitives(primitives):
        raise PreflightValidationError("v01_primitives")
    tokens = tokenizer.encode(visual_v01)
    if tokenizer.decode(tokens) != visual_v01 or len(tokens) > MAX_TOKENS:
        raise PreflightValidationError("v01_token_contract")
    if int(row["sequence_tokens"]) != len(tokens):
        raise PreflightValidationError("manifest_sequence_tokens")
    parsed_v02 = parse_visual_svg_v0_2(visual_v02)
    if serialize_parsed_visual_svg_v0_2(parsed_v02) != visual_v02:
        raise PreflightValidationError("v02_canonical")
    if parsed_v02 != visual_primitives_v0_1_to_visual_svg_v0_2(primitives):
        raise PreflightValidationError("v02_primitives")
    if flatten_visual_svg_v0_2_to_v0_1(parsed_v02) != parsed_v01:
        raise PreflightValidationError("v02_flatten")

    layers = _json_member(members[".layers.json"], label=f"{key}.layers")
    render = _json_member(members[".render.json"], label=f"{key}.render")
    chemmol = _json_member(members[".chemmol.json"], label=f"{key}.chemmol")
    meta = _json_member(members[".meta.json"], label=f"{key}.meta")
    chemmol_validator(chemmol)
    for payload, label in ((layers, "layers"), (render, "render"), (meta, "meta")):
        descriptor = _sidecar_contract(payload, row, label=label)
        if descriptor["font_sha256"] != row["font_sha"]:
            raise PreflightValidationError(f"{label}_font_sha")
    if layers.get("sample_key") != key or meta.get("sample_key") != key:
        raise PreflightValidationError("sidecar_sample_key")
    if meta.get("fresh_audit_eligible") is not True:
        raise PreflightValidationError("fresh_audit_eligible")
    if meta.get("sdf_protocol") != SDF_PROTOCOL:
        raise PreflightValidationError("meta_sdf_protocol")
    meta_source = meta.get("source")
    if not isinstance(meta_source, dict):
        raise PreflightValidationError("meta_source")
    if (
        meta_source.get("source_split") != "train"
        or meta_source.get("source_record_id") != row["source_record_id"]
        or not str(meta_source.get("source_sample_key", "")).startswith("train_")
        or not str(meta_source.get("source_shard", "")).startswith("shards/train-")
        or meta_source.get("structure_ref") != row["source_structure"]
        or meta_source.get("source_sdf_sha256") != row["source_sdf_sha256"]
    ):
        raise PreflightValidationError("meta_source_contract")

    expected_meta_hashes = {
        suffix.removeprefix("."): sha256_bytes(members[suffix])
        for suffix in HASHED_SUFFIXES
    }
    if meta.get("sha256") != expected_meta_hashes:
        raise PreflightValidationError("meta_member_hashes")
    hash_fields = {
        "image_sha256": ".png",
        "sdf_sha256": ".sdf",
        "primitives_sha256": ".primitives.json",
        "visualsvg_v01_sha256": ".visualsvg.txt",
        "visualsvg_v02_sha256": ".visualsvg-v02.json",
        "layers_sha256": ".layers.json",
        "render_sha256": ".render.json",
        "chemmol_sha256": ".chemmol.json",
        "meta_sha256": ".meta.json",
    }
    for field, suffix in hash_fields.items():
        if paired[field] != sha256_bytes(members[suffix]):
            raise PreflightValidationError(f"paired_hash:{field}")

    sdf, inchikey = _parse_sdf(members[".sdf"], sample_key=key)
    if not (
        inchikey
        == row["sdf_inchikey"]
        == paired["inchikey"]
        == molecule_row["inchikey_full"]
        == meta.get("sdf_inchikey")
    ):
        raise PreflightValidationError("inchikey_chain")
    _chemmol_sdf_contract(chemmol, sdf, expected_inchikey=inchikey)
    if (
        sdf.GetNumAtoms() != int(molecule_row["source_atom_count"])
        or sdf.GetNumBonds() != int(molecule_row["source_bond_count"])
    ):
        raise PreflightValidationError("committed_source_atom_bond_count")
    source_sdf_path = _safe_relative(
        source,
        row["source_structure"],
        context="source_structure",
    )
    source_sdf = source_sdf_path.read_bytes()
    if source_sdf != members[".sdf"]:
        raise PreflightValidationError("source_sdf_bytes")
    if sha256_bytes(source_sdf) != row["source_sdf_sha256"]:
        raise PreflightValidationError("source_sdf_sha256")
    if sha256_bytes(source_sdf) != molecule_row["source_sdf_sha256"]:
        raise PreflightValidationError("committed_source_sdf_sha256")
    if len(source_sdf) != int(molecule_row["source_sdf_bytes"]):
        raise PreflightValidationError("committed_source_sdf_bytes")
    if (
        sha256_bytes(members[".chemmol.json"])
        != molecule_row["source_chemmol_sha256"]
    ):
        raise PreflightValidationError("committed_source_chemmol_sha256")
    doubles, triples = _validate_bond_groups(
        primitives,
        parsed_v02,
        sdf_bonds=sdf.GetNumBonds(),
    )
    return {
        "sdf_sha256": sha256_bytes(members[".sdf"]),
        "chemmol_sha256": sha256_bytes(members[".chemmol.json"]),
        "style_id": row["style_id"],
        "font_sha": row["font_sha"],
        "tokens": len(tokens),
        "double_groups": doubles,
        "triple_groups": triples,
    }


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "freeze_id": FREEZE_ID,
        "role": ROLE,
        "split": SPLIT,
        "training_steps": 0,
        "parameter_updates": 0,
        "model_run": False,
        "threshold_or_checkpoint_selection": False,
        "sealed_rows_read": False,
        "molecules": 0,
        "samples": 0,
        "tar_members": 0,
        "checked_samples": 0,
        "errors": [],
    }


def validate_preflight(
    preflight_root: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Validate one explicitly supplied fresh-dev preflight directory."""

    report = _base_report()
    errors: list[str] = report["errors"]
    try:
        root = Path(os.path.abspath(preflight_root))
        source = Path(os.path.abspath(source_root))
        _assert_allowed_text(root, context="preflight_root")
        _assert_allowed_text(source, context="source_root")
        _validate_frozen_source(source)
        input_commitment_path = _regular_file(root / "preflight_input_commitment.json")
        input_commitment = _json_file(input_commitment_path)
        if (
            input_commitment.get("schema_version")
            != "visual-svg-layers-c-fresh-dev-preflight-input-0.1"
            or input_commitment.get("status")
            != "built_pending_independent_validation"
            or input_commitment.get("role") != ROLE
            or input_commitment.get("split") != SPLIT
            or int(input_commitment.get("committed_molecule_count", -1))
            != COMMITTED_MOLECULE_COUNT
            or int(input_commitment.get("committed_sample_count", -1))
            != COMMITTED_SAMPLE_COUNT
            or int(input_commitment.get("preflight_molecule_count", -1))
            != MOLECULE_COUNT
            or int(input_commitment.get("preflight_sample_count", -1))
            != SAMPLE_COUNT
            or input_commitment.get("selection_indices")
            != list(range(MOLECULE_COUNT))
        ):
            raise PreflightValidationError("input_commitment_contract")

        freeze_ref = input_commitment.get("freeze") or {}
        version_root = root.parent.parent
        freeze_path = _safe_reference(
            root,
            freeze_ref.get("path", ""),
            context="freeze",
            containment_root=version_root,
        )
        if freeze_ref.get("sha256") != file_sha256(freeze_path):
            raise PreflightValidationError("freeze_sha256")
        if freeze_ref.get("sha256") != FREEZE_SHA256:
            raise PreflightValidationError("phase_a_freeze_sha256")
        freeze = _json_file(freeze_path)
        _validate_freeze(freeze)
        selection_ref = input_commitment.get("selection_commitment") or {}
        selection_path = _safe_reference(
            root,
            selection_ref.get("path", ""),
            context="selection_commitment",
            containment_root=version_root,
        )
        if selection_ref.get("sha256") != file_sha256(selection_path):
            raise PreflightValidationError("selection_commitment_sha256")
        selection, _ = _load_selection(selection_path, freeze_path)

        manifests = input_commitment.get("manifests") or {}
        molecule_ref = manifests.get("molecules") or {}
        sample_ref = manifests.get("samples") or {}
        paired_ref = manifests.get("paired") or {}
        molecule_path = _safe_relative(
            root, molecule_ref.get("path", ""), context="molecules_manifest"
        )
        sample_path = _safe_relative(
            root, sample_ref.get("path", ""), context="samples_manifest"
        )
        paired_path = _safe_relative(
            root, paired_ref.get("path", ""), context="paired_manifest"
        )
        for path, ref, label in (
            (molecule_path, molecule_ref, "molecules"),
            (sample_path, sample_ref, "samples"),
            (paired_path, paired_ref, "paired"),
        ):
            if ref.get("sha256") != file_sha256(path):
                raise PreflightValidationError(f"{label}_manifest_sha256")
        molecule_rows = _read_csv(
            molecule_path, MOLECULE_FIELDS, label="molecules_manifest"
        )
        sample_rows = _read_csv(
            sample_path, SAMPLE_FIELDS, label="samples_manifest"
        )
        paired_rows = _read_csv(
            paired_path, PAIRED_FIELDS, label="paired_manifest"
        )
        shard_ref = input_commitment.get("shard") or {}
        shard = _safe_relative(root, shard_ref.get("path", ""), context="shard")
        if shard_ref.get("sha256") != file_sha256(shard):
            raise PreflightValidationError("shard_sha256")
    except Exception as error:
        errors.append(f"input_contract:{type(error).__name__}:{error}")
        return report

    if len(molecule_rows) != MOLECULE_COUNT:
        errors.append(f"molecule_count:{len(molecule_rows)}")
    if len(sample_rows) != SAMPLE_COUNT:
        errors.append(f"sample_count:{len(sample_rows)}")
    if len(paired_rows) != SAMPLE_COUNT:
        errors.append(f"paired_count:{len(paired_rows)}")
    report["molecules"] = len(molecule_rows)
    report["samples"] = len(sample_rows)
    first_committed = selection[:MOLECULE_COUNT]
    committed_by_index = {
        row["selection_index"]: row for row in first_committed
    }
    molecule_by_id: dict[str, dict[str, str]] = {}
    for row in molecule_rows:
        committed = committed_by_index.get(row["selection_index"])
        if committed is None or any(
            row[field] != committed[field] for field in SELECTION_FIELDS
        ):
            errors.append(f"molecule_not_first_16_commitment:{row['molecule_id']}")
        if row["role"] != ROLE or row["split"] != SPLIT or row["variants"] != "4":
            errors.append(f"molecule_role_split_variant:{row['molecule_id']}")
        if row["molecule_id"] in molecule_by_id:
            errors.append(f"duplicate_molecule:{row['molecule_id']}")
        molecule_by_id[row["molecule_id"]] = row
    if set(molecule_by_id) != {row["molecule_id"] for row in first_committed}:
        errors.append("molecule_identity_set")

    sample_keys = [row["sample_key"] for row in sample_rows]
    paired_keys = [row["sample_key"] for row in paired_rows]
    if sample_keys != sorted(sample_keys) or len(sample_keys) != len(set(sample_keys)):
        errors.append("sample_key_order_or_duplicate")
    if paired_keys != sorted(paired_keys) or len(paired_keys) != len(set(paired_keys)):
        errors.append("paired_key_order_or_duplicate")
    samples_by_key = {row["sample_key"]: row for row in sample_rows}
    paired_by_key = {row["sample_key"]: row for row in paired_rows}
    if set(samples_by_key) != set(paired_by_key):
        errors.append("paired_sample_set")
    for row in sample_rows:
        if (
            row["role"] != ROLE
            or row["split"] != SPLIT
            or row["variant"] not in VARIANTS
            or row["render_profile"] != RENDER_PROFILE
            or row["font_sha"] not in FONT_ALLOWLIST
            or row["shard"] != "shards/fresh-dev-preflight-000000.tar"
            or row["molecule_id"] not in molecule_by_id
        ):
            errors.append(f"sample_manifest_contract:{row['sample_key']}")
        molecule_row = molecule_by_id.get(row["molecule_id"])
        if molecule_row is not None and (
            row["scaffold_id"] != molecule_row["scaffold_id"]
            or row["source_record_id"] != molecule_row["source_record_id"]
            or row["source_structure"] != molecule_row["source_structure"]
        ):
            errors.append(f"sample_identity_contract:{row['sample_key']}")
    for key, paired in paired_by_key.items():
        sample = samples_by_key.get(key)
        if sample is None:
            continue
        for field in (
            "molecule_id",
            "scaffold_id",
            "source_record_id",
            "role",
            "split",
            "variant",
            "shard",
            "style_id",
            "font_sha",
            "render_profile",
        ):
            if paired[field] != sample[field]:
                errors.append(f"paired_identity:{key}:{field}")

    expected_names = [
        key + suffix for key in sorted(samples_by_key) for suffix in MEMBER_SUFFIXES
    ]
    primitive_validator = fastjsonschema.compile(
        json.loads(
            (
                Path(__file__).resolve().parent
                / "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json"
            ).read_text(encoding="utf-8")
        )
    )
    chemmol_validator = fastjsonschema.compile(
        json.loads(
            (
                Path(__file__).resolve().parent / "schemas/chemmol-0.1.schema.json"
            ).read_text(encoding="utf-8")
        )
    )
    tokenizer = VisualSVGTokenizerV01()
    results: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(shard, "r") as archive:
            members = archive.getmembers()
            report["tar_members"] = len(members)
            names = [member.name for member in members]
            if names != expected_names:
                errors.append("tar_member_order_or_names")
            if len(names) != len(set(names)):
                errors.append("duplicate_tar_member")
            by_name = {}
            for member in members:
                try:
                    _split_member_name(member.name)
                    _tar_metadata(member)
                except Exception as error:
                    errors.append(f"tar_contract:{type(error).__name__}:{error}")
                by_name[member.name] = member
            for key in sorted(samples_by_key):
                payload: dict[str, bytes] = {}
                for suffix in MEMBER_SUFFIXES:
                    member = by_name.get(key + suffix)
                    if member is None:
                        errors.append(f"missing_member:{key}{suffix}")
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        errors.append(f"unreadable_member:{key}{suffix}")
                        continue
                    payload[suffix] = extracted.read()
                if len(payload) != len(MEMBER_SUFFIXES):
                    continue
                try:
                    results[key] = _validate_sample(
                        payload,
                        samples_by_key[key],
                        paired_by_key[key],
                        molecule_by_id[samples_by_key[key]["molecule_id"]],
                        source,
                        primitive_validator,
                        chemmol_validator,
                        tokenizer,
                    )
                    report["checked_samples"] += 1
                except Exception as error:
                    errors.append(f"sample:{key}:{type(error).__name__}:{error}")
    except Exception as error:
        errors.append(f"tar_read:{type(error).__name__}:{error}")

    variants_by_molecule: dict[str, set[str]] = defaultdict(set)
    sdf_hashes_by_molecule: dict[str, set[str]] = defaultdict(set)
    chemmol_hashes_by_molecule: dict[str, set[str]] = defaultdict(set)
    for key, row in samples_by_key.items():
        variants_by_molecule[row["molecule_id"]].add(row["variant"])
        result = results.get(key)
        if result is not None:
            sdf_hashes_by_molecule[row["molecule_id"]].add(result["sdf_sha256"])
            chemmol_hashes_by_molecule[row["molecule_id"]].add(
                result["chemmol_sha256"]
            )
    for molecule_id in molecule_by_id:
        if variants_by_molecule[molecule_id] != set(VARIANTS):
            errors.append(f"variant_closure:{molecule_id}")
        if len(sdf_hashes_by_molecule[molecule_id]) != 1:
            errors.append(f"sdf_variant_bytes:{molecule_id}")
        if len(chemmol_hashes_by_molecule[molecule_id]) != 1:
            errors.append(f"chemmol_variant_bytes:{molecule_id}")

    for font_sha, font in FONT_ALLOWLIST.items():
        try:
            if file_sha256(_regular_file(Path(font["file"]))) != font_sha:
                errors.append(f"font_file_sha256:{font_sha}")
        except Exception as error:
            errors.append(f"font_file:{type(error).__name__}:{error}")

    report.update(
        {
            "status": PASS_STATUS if not errors else FAIL_STATUS,
            "members_per_sample": len(MEMBER_SUFFIXES),
            "expected_tar_members": SAMPLE_COUNT * len(MEMBER_SUFFIXES),
            "source_sdf_byte_preservation_rate": (
                report["checked_samples"] / SAMPLE_COUNT
            ),
            "variant_closure_molecules": sum(
                values == set(VARIANTS) for values in variants_by_molecule.values()
            ),
            "max_sequence_tokens": max(
                (value["tokens"] for value in results.values()), default=0
            ),
            "double_bond_groups_checked": sum(
                value["double_groups"] for value in results.values()
            ),
            "triple_bond_groups_checked": sum(
                value["triple_groups"] for value in results.values()
            ),
            "style_ids": len({value["style_id"] for value in results.values()}),
            "font_hashes": sorted(
                {value["font_sha"] for value in results.values()}
            ),
            "sdf_protocol": SDF_PROTOCOL,
            "render_profile": RENDER_PROFILE,
            "selection_commitment_sha256": file_sha256(selection_path),
            "shard_sha256": file_sha256(shard),
            "validator_sha256": file_sha256(Path(__file__).resolve()),
        }
    )
    return report


def _validate_output_path(path: Path, inputs: Sequence[Path]) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        if current.is_symlink():
            raise PreflightValidationError(f"symlink_rejected:{current}")
    if absolute in inputs:
        raise PreflightValidationError("output_replaces_input")
    return absolute


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    version_root = root / "visual_svg_layers_c_v0_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-root",
        default=str(version_root / "fresh_dev/preflight_64"),
    )
    parser.add_argument(
        "--source",
        default=str(root / "webdataset_full_v0"),
    )
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.preflight_root).expanduser().resolve(strict=False)
    output = (
        Path(args.output).expanduser().resolve(strict=False)
        if args.output
        else root / "preflight_validation.json"
    )
    report = validate_preflight(root, args.source)
    output = _validate_output_path(output, (root / "preflight_input_commitment.json",))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(report) + b"\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if report["status"] != PASS_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "FAIL_STATUS",
    "PASS_STATUS",
    "PreflightValidationError",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "validate_preflight",
]
