#!/usr/bin/env python3
"""Build the committed 16-molecule/64-image fresh-dev preflight package.

The builder is intentionally fail-closed.  It consumes the first 16 identities
from an already committed 2,000-molecule fresh-dev selection; it never selects
identities itself and never reads validation, internal-test, sealed, reserve, or
MR-TADF rows.  No model is imported or executed.
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
import random
import shutil
import stat
import tarfile
from typing import Any, Mapping, Sequence

import fastjsonschema
from rdkit import Chem, rdBase

try:
    from . import build_chemvision_layers as layer_builder
    from .build_visual_svg_capacity_100 import json_bytes, tar_add_bytes
    from .build_webdataset_sample import mol_from_chemmol
    from .visual_svg_layers_c import (
        HASHED_SUFFIXES,
        MEMBER_SUFFIXES,
        assemble_c_sample,
    )
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
except ImportError:
    import build_chemvision_layers as layer_builder
    from build_visual_svg_capacity_100 import json_bytes, tar_add_bytes
    from build_webdataset_sample import mol_from_chemmol
    from visual_svg_layers_c import (
        HASHED_SUFFIXES,
        MEMBER_SUFFIXES,
        assemble_c_sample,
    )
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-preflight-build-0.1"
SELECTION_SCHEMA_VERSION = (
    "visual-svg-layers-c-fresh-dev-selection-commitment-0.1"
)
IDENTITY_AUDIT_SCHEMA_VERSION = (
    "visual-svg-layers-c-fresh-dev-identity-isolation-0.1"
)
FREEZE_ID = "visual-svg-layers-c-v0.1-phase-a-20260722"
FREEZE_SHA256 = "6c87e2e63b42c6b9f39c33d829fc8883d456b7c5f82cd15b2672963a7f560b9c"
SOURCE_VALIDATION_SHA256 = "8e37f9c2037e01d8f417b882e1405fa625a8a5f81f283f647d5788f972f6aa78"
TRAIN_MANIFEST_INVENTORY_SHA256 = "cee36b8e8737c3de1826d868955dcc27dcf656d63d8a498f08425708d14acbdb"
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
FORBIDDEN_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "reserve",
    "mr_tadf",
    "mr-tadf",
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
PAIRED_FIELDS = (
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
)
FONT_SPECS = (
    {
        "font_family_id": "stix-regular",
        "font_family": "STIX",
        "file": "STIX-Regular.otf",
        "source_file": "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf",
        "sha256": "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45",
    },
    {
        "font_family_id": "p052-roman",
        "font_family": "P052",
        "file": "P052-Roman.otf",
        "source_file": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
        "sha256": "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e92778a78d149524eea9",
    },
)
FONT_BUCKETS = (
    ("small", 11),
    ("normal", 16),
    ("large", 24),
    ("extreme", 32),
)
LINE_WIDTH_BUCKETS = (
    ("thin", 1.4),
    ("normal", 2.0),
    ("thick", 2.6),
)


class PreflightBuildError(ValueError):
    """Raised when a committed fresh-dev precondition is not satisfied."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _forbidden_marker(value: object) -> str | None:
    lowered = str(value).lower()
    return next((item for item in FORBIDDEN_MARKERS if item in lowered), None)


def _assert_allowed_text(value: object, *, context: str) -> None:
    marker = _forbidden_marker(value)
    if marker is not None:
        raise PreflightBuildError(f"forbidden_reference:{context}:{marker}")


def _regular_file_without_symlinks(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    _assert_allowed_text(absolute, context="input_path")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise PreflightBuildError(f"missing_input:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise PreflightBuildError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise PreflightBuildError(f"input_not_regular_file:{absolute}")
    return absolute


def _safe_relative_file(base: Path, relative: object, *, context: str) -> Path:
    text = str(relative)
    _assert_allowed_text(text, context=context)
    value = Path(text)
    if value.is_absolute() or ".." in value.parts:
        raise PreflightBuildError(f"unsafe_relative_path:{context}:{text}")
    candidate = _regular_file_without_symlinks(base / value)
    if os.path.commonpath((str(base), str(candidate))) != str(base):
        raise PreflightBuildError(f"path_escapes_base:{context}:{text}")
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_regular_file_without_symlinks(path).read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightBuildError(f"invalid_json:{path}") from error
    if not isinstance(value, dict):
        raise PreflightBuildError(f"json_root_not_object:{path}")
    return value


def _validate_frozen_source(source: Path) -> dict[str, Any]:
    expected = (Path(__file__).resolve().parent / "webdataset_full_v0").resolve()
    if source != expected:
        raise PreflightBuildError(f"source_root_not_frozen:{source}:{expected}")
    validation_path = source / "validation.json"
    if validation_path.is_symlink() or not validation_path.is_file():
        raise PreflightBuildError("source_validation_missing_or_symlink")
    if file_sha256(validation_path) != SOURCE_VALIDATION_SHA256:
        raise PreflightBuildError("source_validation_sha256_mismatch")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("scaffold_overlap") != 0:
        raise PreflightBuildError("source_validation_contract")
    paths = sorted(
        [
            *(source / "manifests/shards").glob("train-*.molecules.csv"),
            *(source / "manifests/shards").glob("train-*.samples.csv"),
        ]
    )
    if len(paths) != 260:
        raise PreflightBuildError(f"train_manifest_inventory_count:{len(paths)}")
    digest = hashlib.sha256()
    for path in paths:
        path = _regular_file_without_symlinks(path)
        relative = path.relative_to(source).as_posix()
        digest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
    if digest.hexdigest() != TRAIN_MANIFEST_INVENTORY_SHA256:
        raise PreflightBuildError("train_manifest_inventory_sha256_mismatch")
    return validation


def _require_static_freeze(freeze: Mapping[str, Any]) -> None:
    expected = {
        "freeze_id": FREEZE_ID,
        "phase": "A",
        "training_steps": 0,
        "parameter_updates": 0,
    }
    for field, value in expected.items():
        if freeze.get(field) != value:
            raise PreflightBuildError(
                f"freeze_static_mismatch:{field}:{freeze.get(field)!r}"
            )
    if freeze.get("status") != "frozen_fresh_dev_preflight_authorized":
        raise PreflightBuildError("fresh_dev_preflight_not_authorized")
    split = (freeze.get("splits") or {}).get("fresh_dev") or {}
    preflight = freeze.get("preflight") or {}
    if (
        split.get("authorized") is not True
        or split.get("role") != ROLE
        or split.get("split") != SPLIT
        or int(split.get("molecule_count", -1)) != COMMITTED_MOLECULE_COUNT
        or int(split.get("sample_count", -1)) != COMMITTED_SAMPLE_COUNT
        or preflight.get("run_model") is not False
        or int(preflight.get("molecule_count", -1)) != MOLECULE_COUNT
        or int(preflight.get("sample_count", -1)) != SAMPLE_COUNT
        or preflight.get("selection") != "first_16_committed_fresh_dev_molecules"
    ):
        raise PreflightBuildError("freeze_fresh_dev_contract_mismatch")
    if (freeze.get("sdf") or {}).get("protocol") != SDF_PROTOCOL:
        raise PreflightBuildError("freeze_sdf_protocol_mismatch")
    if (freeze.get("render") or {}).get("render_profile") != RENDER_PROFILE:
        raise PreflightBuildError("freeze_render_profile_mismatch")
    if int((freeze.get("sequence") or {}).get("max_tokens", -1)) != MAX_TOKENS:
        raise PreflightBuildError("freeze_max_tokens_mismatch")
    if tuple((freeze.get("sample_contract") or {}).get("members") or ()) != tuple(
        "sample" + suffix for suffix in MEMBER_SUFFIXES
    ):
        raise PreflightBuildError("freeze_member_contract_mismatch")


def selection_rank(row: Mapping[str, str]) -> str:
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


def source_bytes_commitment_sha256(rows: Sequence[Mapping[str, str]]) -> str:
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
    return sha256_bytes(canonical_json_bytes(payload))


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    with _regular_file_without_symlinks(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        if missing:
            raise PreflightBuildError(
                f"manifest_missing_fields:{path.name}:{','.join(missing)}"
            )
        rows = list(reader)
    for row_number, row in enumerate(rows, 2):
        for field in required:
            if not str(row.get(field, "")).strip():
                raise PreflightBuildError(
                    f"manifest_empty_field:{path.name}:{row_number}:{field}"
                )
        for field, value in row.items():
            _assert_allowed_text(value, context=f"{path.name}:{row_number}:{field}")
    return rows


def _validate_identity_audit(
    audit_path: Path,
    *,
    expected_sha256: str,
    selection_manifest_sha256: str,
) -> dict[str, Any]:
    if file_sha256(audit_path) != expected_sha256:
        raise PreflightBuildError("identity_audit_sha256_mismatch")
    audit = _load_json(audit_path)
    if (
        audit.get("schema_version") != IDENTITY_AUDIT_SCHEMA_VERSION
        or audit.get("status") != "passed"
        or audit.get("fresh_dev_selection_manifest_sha256")
        != selection_manifest_sha256
    ):
        raise PreflightBuildError("identity_audit_contract_mismatch")
    comparisons = audit.get("comparisons")
    if not isinstance(comparisons, dict):
        raise PreflightBuildError("identity_audit_missing_comparisons")
    for exposure in ("frozen_b_p0_train_exposure", "smoke_train_32"):
        result = comparisons.get(exposure)
        if not isinstance(result, dict):
            raise PreflightBuildError(f"identity_audit_missing:{exposure}")
        for field in (
            "molecule_id_overlap",
            "inchikey_full_overlap",
            "scaffold_id_overlap",
            "source_record_id_overlap",
        ):
            if result.get(field) != 0:
                raise PreflightBuildError(
                    f"identity_audit_nonzero:{exposure}:{field}:{result.get(field)!r}"
                )
    required_hashes = (
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
    for field in required_hashes:
        value = audit.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise PreflightBuildError(f"identity_audit_hash_missing:{field}")
    return audit


def load_committed_selection(
    commitment_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Load and verify a complete 2,000-identity selection commitment."""

    freeze_path = _regular_file_without_symlinks(freeze_path)
    if file_sha256(freeze_path) != FREEZE_SHA256:
        raise PreflightBuildError("phase_a_freeze_sha256_mismatch")
    freeze = _load_json(freeze_path)
    _require_static_freeze(freeze)
    commitment_path = _regular_file_without_symlinks(commitment_path)
    commitment = _load_json(commitment_path)
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
        raise PreflightBuildError("selection_commitment_contract_mismatch")
    freeze_ref = commitment.get("freeze") or {}
    if freeze_ref.get("sha256") != file_sha256(freeze_path):
        raise PreflightBuildError("selection_commitment_freeze_sha256_mismatch")

    manifest_ref = commitment.get("selection_manifest") or {}
    manifest_path = _safe_relative_file(
        commitment_path.parent,
        manifest_ref.get("path", ""),
        context="selection_manifest",
    )
    manifest_sha = file_sha256(manifest_path)
    if manifest_ref.get("sha256") != manifest_sha:
        raise PreflightBuildError("selection_manifest_sha256_mismatch")
    rows = _read_csv(manifest_path, SELECTION_FIELDS)
    if len(rows) != COMMITTED_MOLECULE_COUNT:
        raise PreflightBuildError(
            f"selection_manifest_count:{len(rows)}:{COMMITTED_MOLECULE_COUNT}"
        )
    if [int(row["selection_index"]) for row in rows] != list(
        range(COMMITTED_MOLECULE_COUNT)
    ):
        raise PreflightBuildError("selection_index_not_contiguous")
    if any(row["source_split"] != "train" for row in rows):
        raise PreflightBuildError("selection_source_split_not_train")
    for row in rows:
        if row["rank_sha256"] != selection_rank(row):
            raise PreflightBuildError(
                f"selection_rank_mismatch:{row['selection_index']}"
            )
        if row["source_record_id"] != f"{row['source']}:{row['source_id']}":
            raise PreflightBuildError(
                f"source_record_id_not_canonical:{row['selection_index']}"
            )
        if not row["source_shard"].startswith("shards/train-"):
            raise PreflightBuildError("selection_source_shard_not_train")
        if not row["source_structure"].startswith("structures/train-"):
            raise PreflightBuildError("selection_source_structure_not_train")
    if [row["rank_sha256"] for row in rows] != sorted(
        row["rank_sha256"] for row in rows
    ):
        raise PreflightBuildError("selection_rows_not_rank_sorted")
    for field in (
        "molecule_id",
        "inchikey_full",
        "source_record_id",
    ):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise PreflightBuildError(f"selection_duplicate_identity:{field}")
    source_commitment = commitment.get("source_bytes_commitment") or {}
    if (
        source_commitment.get("protocol") != SDF_PROTOCOL
        or int(source_commitment.get("molecule_count", -1))
        != COMMITTED_MOLECULE_COUNT
        or source_commitment.get("sha256")
        != source_bytes_commitment_sha256(rows)
    ):
        raise PreflightBuildError("source_bytes_commitment_mismatch")

    audit_ref = commitment.get("identity_isolation_report") or {}
    audit_path = _safe_relative_file(
        commitment_path.parent,
        audit_ref.get("path", ""),
        context="identity_isolation_report",
    )
    audit = _validate_identity_audit(
        audit_path,
        expected_sha256=str(audit_ref.get("sha256", "")),
        selection_manifest_sha256=manifest_sha,
    )
    return rows, commitment, audit


def _source_manifest_path(source: Path, source_shard: str, kind: str) -> Path:
    if not source_shard.startswith("shards/train-") or not source_shard.endswith(
        ".tar"
    ):
        raise PreflightBuildError(f"invalid_train_shard:{source_shard}")
    stem = Path(source_shard).stem
    return _regular_file_without_symlinks(
        source / "manifests" / "shards" / f"{stem}.{kind}.csv"
    )


def _verify_source_molecule_rows(
    source: Path,
    selected: Sequence[dict[str, str]],
) -> None:
    wanted_by_manifest: dict[Path, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in selected:
        manifest = _source_manifest_path(source, row["source_shard"], "molecules")
        wanted_by_manifest[manifest][row["molecule_id"]] = row
    for manifest, wanted in wanted_by_manifest.items():
        found: dict[str, dict[str, str]] = {}
        with manifest.open(newline="", encoding="utf-8") as handle:
            for source_row in csv.DictReader(handle):
                molecule_id = source_row.get("molecule_id", "")
                if molecule_id in wanted:
                    found[molecule_id] = source_row
        if set(found) != set(wanted):
            missing = sorted(set(wanted) - set(found))
            raise PreflightBuildError(f"source_molecule_rows_missing:{missing[:1]}")
        for molecule_id, committed in wanted.items():
            source_row = found[molecule_id]
            expected = {
                "source": committed["source"],
                "source_id": committed["source_id"],
                "split": "train",
                "shard": committed["source_shard"],
                "scaffold": committed["scaffold_id"],
                "inchikey": committed["inchikey_full"],
                "structure_ref": committed["source_structure"],
            }
            for field, value in expected.items():
                if source_row.get(field) != value:
                    raise PreflightBuildError(
                        f"source_molecule_mismatch:{molecule_id}:{field}"
                    )


def _load_source_sample_rows(
    source: Path,
    selected: Sequence[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    wanted_by_manifest: dict[Path, set[str]] = defaultdict(set)
    for row in selected:
        wanted_by_manifest[
            _source_manifest_path(source, row["source_shard"], "samples")
        ].add(row["molecule_id"])
    result: dict[tuple[str, str], dict[str, str]] = {}
    for manifest, wanted in wanted_by_manifest.items():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("molecule_id") not in wanted:
                    continue
                if row.get("split") != "train" or not row.get("shard", "").startswith(
                    "shards/train-"
                ):
                    raise PreflightBuildError("source_sample_row_not_train")
                key = (row["molecule_id"], row["variant"])
                if key in result:
                    raise PreflightBuildError(f"duplicate_source_sample:{key}")
                result[key] = row
    expected = {
        (row["molecule_id"], variant)
        for row in selected
        for variant in VARIANTS
    }
    if set(result) != expected:
        raise PreflightBuildError("source_sample_variant_closure_mismatch")
    return result


def _read_tar_members(
    source: Path,
    sample_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[tuple[str, str], tuple[bytes, dict[str, Any]]]:
    wanted_by_shard: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for identity, row in sample_rows.items():
        shard = row["shard"]
        _assert_allowed_text(shard, context="source_sample_shard")
        if not shard.startswith("shards/train-"):
            raise PreflightBuildError("source_sample_shard_not_train")
        key = row["sample_key"]
        wanted_by_shard[shard][f"{key}.chemmol.json"] = identity
        wanted_by_shard[shard][f"{key}.render.json"] = identity
    raw: dict[tuple[str, str], dict[str, bytes]] = defaultdict(dict)
    for shard_name, wanted in sorted(wanted_by_shard.items()):
        shard = _regular_file_without_symlinks(source / shard_name)
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                identity = wanted.get(member.name)
                if identity is None:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise PreflightBuildError(f"source_member_unreadable:{member.name}")
                suffix = member.name.rsplit(".", 2)[-2] + ".json"
                raw[identity][suffix] = extracted.read()
    assets = {}
    for identity in sample_rows:
        payload = raw.get(identity, {})
        if set(payload) != {"chemmol.json", "render.json"}:
            raise PreflightBuildError(f"source_members_missing:{identity}")
        try:
            source_render = json.loads(payload["render.json"])
        except json.JSONDecodeError as error:
            raise PreflightBuildError(f"source_render_invalid:{identity}") from error
        assets[identity] = (payload["chemmol.json"], source_render)
    return assets


def _parse_single_sdf(data: bytes, *, molecule_id: str) -> Chem.Mol:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise PreflightBuildError(f"source_sdf_not_single_sanitized:{molecule_id}")
    molecule = molecules[0]
    Chem.SanitizeMol(molecule)
    return molecule


def _enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def _validate_chemmol_against_sdf(
    chemmol_bytes: bytes,
    sdf_bytes: bytes,
    *,
    identity: Mapping[str, str],
) -> dict[str, Any]:
    try:
        chemmol = json.loads(chemmol_bytes)
    except json.JSONDecodeError as error:
        raise PreflightBuildError(
            f"source_chemmol_invalid:{identity['molecule_id']}"
        ) from error
    if chemmol.get("molecule_id") != identity["molecule_id"]:
        raise PreflightBuildError("source_chemmol_molecule_id_mismatch")
    sdf_mol = _parse_single_sdf(sdf_bytes, molecule_id=identity["molecule_id"])
    Chem.AssignStereochemistry(sdf_mol, cleanIt=False, force=True)
    sdf_key = Chem.MolToInchiKey(sdf_mol)
    reconstructed = mol_from_chemmol(chemmol)
    Chem.SanitizeMol(reconstructed)
    if (
        sdf_key != identity["inchikey_full"]
        or chemmol.get("inchikey") != identity["inchikey_full"]
        or Chem.MolToInchiKey(reconstructed) != identity["inchikey_full"]
    ):
        raise PreflightBuildError("source_inchikey_mismatch")
    atoms = chemmol.get("atoms") or []
    bonds = chemmol.get("bonds") or []
    if len(atoms) != sdf_mol.GetNumAtoms() or len(bonds) != sdf_mol.GetNumBonds():
        raise PreflightBuildError("source_chemmol_sdf_count_mismatch")
    if (
        len(atoms) != int(identity["source_atom_count"])
        or len(bonds) != int(identity["source_bond_count"])
    ):
        raise PreflightBuildError("committed_source_atom_bond_count_mismatch")
    if [int(atom["id"]) for atom in atoms] != list(range(len(atoms))):
        raise PreflightBuildError("source_atom_ids_not_contiguous")
    if [int(bond["id"]) for bond in bonds] != list(range(len(bonds))):
        raise PreflightBuildError("source_bond_ids_not_contiguous")
    source_atom_indices: set[int] = set()
    for atom_data in atoms:
        atom_id = int(atom_data["id"])
        source_index = int(atom_data["source_atom_index"])
        if source_index in source_atom_indices:
            raise PreflightBuildError("source_atom_index_duplicate")
        source_atom_indices.add(source_index)
        if atom_id < 0 or atom_id >= sdf_mol.GetNumAtoms():
            raise PreflightBuildError("source_atom_id_out_of_range")
        atom = sdf_mol.GetAtomWithIdx(atom_id)
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
        cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
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
            cip_code,
        )
        if actual != expected:
            raise PreflightBuildError(
                f"source_atom_chemistry_mismatch:{identity['molecule_id']}:{atom_id}"
            )
    if source_atom_indices != set(range(len(atoms))):
        raise PreflightBuildError("source_atom_index_not_full_permutation")
    for bond_data in bonds:
        begin = int(bond_data["begin_atom"])
        end = int(bond_data["end_atom"])
        bond = sdf_mol.GetBondBetweenAtoms(begin, end)
        if bond is None:
            raise PreflightBuildError("source_bond_endpoint_mismatch")
        expected = (
            str(bond_data["type"]),
            bool(bond_data["aromatic"]),
            bool(bond_data["conjugated"]),
            str(bond_data["stereo"]),
            tuple(int(value) for value in bond_data["stereo_atoms"]),
            bond_data.get("ez"),
        )
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
        if actual != expected:
            raise PreflightBuildError(
                f"source_bond_chemistry_mismatch:{identity['molecule_id']}:{bond_data['id']}"
            )
    return chemmol


def _load_source_sdf(source: Path, identity: Mapping[str, str]) -> tuple[bytes, str]:
    relative = identity["source_structure"]
    if not relative.startswith("structures/train-"):
        raise PreflightBuildError("source_structure_not_train")
    path = _safe_relative_file(source, relative, context="source_structure")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != identity["source_sdf_sha256"]:
        raise PreflightBuildError("committed_source_sdf_sha256_mismatch")
    if len(data) != int(identity["source_sdf_bytes"]):
        raise PreflightBuildError("committed_source_sdf_bytes_mismatch")
    return data, digest


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(item) for item in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _font_specs_checked(freeze: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    frozen_fonts = {
        (item["file"], item["family"], item["sha256"])
        for item in (freeze.get("fonts") or {}).get("fresh_dev", ())
    }
    expected = {
        (item["file"], item["font_family"], item["sha256"])
        for item in FONT_SPECS
    }
    if frozen_fonts != expected:
        raise PreflightBuildError("freeze_fresh_dev_font_allowlist_mismatch")
    result = []
    for item in FONT_SPECS:
        path = _regular_file_without_symlinks(Path(item["source_file"]))
        if file_sha256(path) != item["sha256"]:
            raise PreflightBuildError(f"font_sha256_mismatch:{item['file']}")
        result.append(dict(item))
    return tuple(result)


def _style_descriptor(
    *,
    font: Mapping[str, str],
    variant: str,
    font_bucket: str,
    line_width_bucket: str,
) -> dict[str, Any]:
    return {
        "schema": "visual-svg-layers-c-style-0.1",
        "canvas": {"height": "768.000000", "width": "768.000000"},
        "font_sha256": font["sha256"],
        "font_family": font["font_family"],
        "render_profile": RENDER_PROFILE,
        "layout_variant": variant,
        "font_size_bucket": font_bucket,
        "bond_line_width_bucket": line_width_bucket,
        "atom_label_style": "rdkit-outline-glyphs",
        "bond_style": "rdkit-semantic-paths",
        "color_mode": "black_white",
        "background": "opaque_white",
        "antialias": "cairosvg-default",
    }


def style_id(descriptor: Mapping[str, Any]) -> str:
    return "style-v0.1:" + sha256_bytes(canonical_json_bytes(descriptor))


def _render_task(
    identity: Mapping[str, str],
    source_row: Mapping[str, str],
    *,
    variant: str,
    font: Mapping[str, str],
    selection_index: int,
) -> tuple[layer_builder.RenderTask, dict[str, Any]]:
    bucket_name, font_size = FONT_BUCKETS[
        _stable_int(FREEZE_ID, "font-bucket", identity["molecule_id"], variant)
        % len(FONT_BUCKETS)
    ]
    width_bucket, bond_width = LINE_WIDTH_BUCKETS[
        _stable_int(FREEZE_ID, "line-width", identity["molecule_id"], variant)
        % len(LINE_WIDTH_BUCKETS)
    ]
    rotation_rng = random.Random(
        _stable_int(FREEZE_ID, "rotation", identity["molecule_id"], variant)
    )
    descriptor = _style_descriptor(
        font=font,
        variant=variant,
        font_bucket=bucket_name,
        line_width_bucket=width_bucket,
    )
    task = layer_builder.RenderTask(
        sample_key=f"fresh_dev_{selection_index:06d}_{variant}",
        molecule_id=identity["molecule_id"],
        split=SPLIT,
        variant=variant,
        source_sample_key=source_row["sample_key"],
        source_shard=source_row["shard"],
        output_shard="shards/fresh-dev-preflight-000000.tar",
        font_family_id=font["font_family_id"],
        font_source_file=font["source_file"],
        font_file_sha256=font["sha256"],
        font_size=font_size,
        font_bucket=bucket_name,
        bond_line_width=bond_width,
        rotation_degrees=round(rotation_rng.uniform(-12.0, 12.0), 6),
        repeat_check=False,
    )
    return task, {"descriptor": descriptor, "style_id": style_id(descriptor)}


def _enrich_members(
    members: dict[str, bytes],
    *,
    identity: Mapping[str, str],
    style: Mapping[str, Any],
    source_sdf_sha256: str,
) -> None:
    descriptor = dict(style["descriptor"])
    common = {
        "role": ROLE,
        "split": SPLIT,
        "scaffold_id": identity["scaffold_id"],
        "source_record_id": identity["source_record_id"],
        "style_id": style["style_id"],
        "font_sha": descriptor["font_sha256"],
        "render_profile": RENDER_PROFILE,
    }
    layers = json.loads(members[".layers.json"])
    layers.update(common)
    layers["style"] = descriptor
    layers["render_parameters"].update(
        {
            "style_id": style["style_id"],
            "font_sha": descriptor["font_sha256"],
            "render_profile": RENDER_PROFILE,
        }
    )
    members[".layers.json"] = canonical_json_bytes(layers, newline=True)

    render = json.loads(members[".render.json"])
    render.update(common)
    render["style"] = descriptor
    render["render_parameters"].update(
        {
            "style_id": style["style_id"],
            "font_sha": descriptor["font_sha256"],
            "render_profile": RENDER_PROFILE,
        }
    )
    members[".render.json"] = canonical_json_bytes(render, newline=True)

    meta = json.loads(members[".meta.json"])
    meta.update(common)
    meta["source"].update(
        {
            "source_record_id": identity["source_record_id"],
            "source_split": "train",
            "source_sdf_sha256": source_sdf_sha256,
        }
    )
    meta["style"] = descriptor
    meta["sdf_protocol"] = SDF_PROTOCOL
    meta["sha256"] = {
        suffix.removeprefix("."): sha256_bytes(members[suffix])
        for suffix in HASHED_SUFFIXES
    }
    members[".meta.json"] = canonical_json_bytes(meta, newline=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_shard(
    path: Path,
    samples: Sequence[tuple[str, Mapping[str, bytes], Mapping[str, Any]]],
) -> None:
    temporary = path.with_suffix(".tmp")
    with tarfile.open(temporary, "w") as archive:
        for key, members, _ in sorted(samples, key=lambda item: item[0]):
            for suffix in MEMBER_SUFFIXES:
                tar_add_bytes(archive, key + suffix, members[suffix])
    temporary.replace(path)


def _clean_output(output: Path, *, overwrite: bool) -> None:
    generated = (
        output / "shards",
        output / "molecules.csv",
        output / "samples.csv",
        output / "paired.csv",
        output / "candidate_rejected.csv",
        output / "summary.json",
        output / "preflight_input_commitment.json",
        output / "preflight_validation.json",
    )
    existing = [path for path in generated if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Preflight output exists; pass --overwrite: {existing[0]}"
        )
    if overwrite:
        for path in generated:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    freeze_path = Path(args.freeze).expanduser().resolve()
    selection_commitment_path = Path(args.selection_commitment).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    _assert_allowed_text(source, context="source_root")
    _assert_allowed_text(selection_commitment_path, context="selection_commitment")
    _assert_allowed_text(output, context="output_root")
    _validate_frozen_source(source)

    selection, selection_commitment, identity_audit = load_committed_selection(
        selection_commitment_path,
        freeze_path,
    )
    freeze = _load_json(freeze_path)
    fonts = _font_specs_checked(freeze)
    selected = selection[:MOLECULE_COUNT]
    _verify_source_molecule_rows(source, selected)
    source_sample_rows = _load_source_sample_rows(source, selected)
    source_assets = _read_tar_members(source, source_sample_rows)
    _clean_output(output, overwrite=args.overwrite)

    schema_path = (
        Path(__file__).resolve().parent
        / "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json"
    )
    primitive_validator = fastjsonschema.compile(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    tokenizer = VisualSVGTokenizerV01()
    samples: list[tuple[str, dict[str, bytes], dict[str, Any]]] = []
    molecule_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for identity in selected:
        selection_index = int(identity["selection_index"])
        try:
            sdf_bytes, sdf_sha = _load_source_sdf(source, identity)
            molecule_samples = []
            validated_chemmol_sha: set[str] = set()
            for variant in VARIANTS:
                source_row = source_sample_rows[(identity["molecule_id"], variant)]
                chemmol_bytes, source_render = source_assets[
                    (identity["molecule_id"], variant)
                ]
                chemmol_sha = sha256_bytes(chemmol_bytes)
                if chemmol_sha != identity["source_chemmol_sha256"]:
                    raise PreflightBuildError(
                        f"committed_chemmol_sha256_mismatch:{identity['molecule_id']}:{variant}"
                    )
                _validate_chemmol_against_sdf(
                    chemmol_bytes,
                    sdf_bytes,
                    identity=identity,
                )
                validated_chemmol_sha.add(chemmol_sha)
                font = fonts[
                    _stable_int(
                        FREEZE_ID,
                        "font",
                        identity["molecule_id"],
                        variant,
                    )
                    % len(fonts)
                ]
                task, style = _render_task(
                    identity,
                    source_row,
                    variant=variant,
                    font=font,
                    selection_index=selection_index,
                )
                members, diagnostics = assemble_c_sample(
                    task,
                    chemmol_bytes,
                    sdf_bytes,
                    source_render,
                    primitive_validator=primitive_validator,
                    tokenizer=tokenizer,
                    output_split=SPLIT,
                    source_dataset="webdataset_full_v0/train",
                    structure_ref=identity["source_structure"],
                    fresh_audit_eligible=True,
                    max_tokens=MAX_TOKENS,
                )
                _enrich_members(
                    members,
                    identity=identity,
                    style=style,
                    source_sdf_sha256=sdf_sha,
                )
                diagnostics.update(
                    {
                        "selection_index": selection_index,
                        "role": ROLE,
                        "scaffold_id": identity["scaffold_id"],
                        "source_record_id": identity["source_record_id"],
                        "source_structure": identity["source_structure"],
                        "source_sdf_sha256": sdf_sha,
                        "style_id": style["style_id"],
                        "font_sha": style["descriptor"]["font_sha256"],
                        "render_profile": RENDER_PROFILE,
                        "shard": "shards/fresh-dev-preflight-000000.tar",
                    }
                )
                molecule_samples.append((task.sample_key, members, diagnostics))
            if len(validated_chemmol_sha) != 1:
                raise PreflightBuildError(
                    f"source_chemmol_bytes_differ_across_variants:{identity['molecule_id']}"
                )
            if validated_chemmol_sha != {identity["source_chemmol_sha256"]}:
                raise PreflightBuildError(
                    f"committed_chemmol_sha256_mismatch:{identity['molecule_id']}"
                )
            samples.extend(molecule_samples)
            sdf_mol = _parse_single_sdf(
                sdf_bytes,
                molecule_id=identity["molecule_id"],
            )
            molecule_rows.append(
                {
                    **{field: identity[field] for field in SELECTION_FIELDS},
                    "role": ROLE,
                    "split": SPLIT,
                    "variants": len(VARIANTS),
                    "sdf_sha256": sdf_sha,
                    "sdf_bytes": len(sdf_bytes),
                    "atom_count": sdf_mol.GetNumAtoms(),
                    "bond_count": sdf_mol.GetNumBonds(),
                }
            )
        except Exception as error:
            rejected.append(
                {
                    "selection_index": identity["selection_index"],
                    "molecule_id": identity["molecule_id"],
                    "stage": "pre_render_contract",
                    "error_code": type(error).__name__,
                    "error_message": str(error),
                }
            )
            break

    if rejected:
        _write_csv(
            output / "candidate_rejected.csv",
            rejected,
            (
                "selection_index",
                "molecule_id",
                "stage",
                "error_code",
                "error_message",
            ),
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "review_required",
            "reason": "committed_preflight_identity_rejected",
            "persistent_png_or_tar_written": False,
            "rejected": rejected,
        }
        (output / "summary.json").write_bytes(json_bytes(report))
        raise RuntimeError(
            f"Committed preflight identity rejected: {rejected[0]['molecule_id']}"
        )
    if len(molecule_rows) != MOLECULE_COUNT or len(samples) != SAMPLE_COUNT:
        raise PreflightBuildError("preflight_count_mismatch_before_write")

    (output / "shards").mkdir(parents=True, exist_ok=True)
    shard = output / "shards/fresh-dev-preflight-000000.tar"
    _write_shard(shard, samples)
    sample_rows = [item[2] for item in sorted(samples, key=lambda item: item[0])]
    _write_csv(
        output / "molecules.csv",
        molecule_rows,
        (*SELECTION_FIELDS, "role", "split", "variants", "sdf_sha256", "sdf_bytes", "atom_count", "bond_count"),
    )
    _write_csv(
        output / "samples.csv",
        sample_rows,
        (
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
            "primitive_count",
            "visual_svg_v02_group_count",
        ),
    )
    paired_rows = []
    for key, members, row in sorted(samples, key=lambda item: item[0]):
        paired_rows.append(
            {
                "sample_key": key,
                "molecule_id": row["molecule_id"],
                "inchikey": row["sdf_inchikey"],
                "scaffold_id": row["scaffold_id"],
                "source_record_id": row["source_record_id"],
                "role": ROLE,
                "split": SPLIT,
                "variant": row["variant"],
                "shard": row["shard"],
                "style_id": row["style_id"],
                "font_sha": row["font_sha"],
                "render_profile": RENDER_PROFILE,
                "image_sha256": sha256_bytes(members[".png"]),
                "sdf_sha256": sha256_bytes(members[".sdf"]),
                "primitives_sha256": sha256_bytes(members[".primitives.json"]),
                "visualsvg_v01_sha256": sha256_bytes(members[".visualsvg.txt"]),
                "visualsvg_v02_sha256": sha256_bytes(members[".visualsvg-v02.json"]),
                "layers_sha256": sha256_bytes(members[".layers.json"]),
                "render_sha256": sha256_bytes(members[".render.json"]),
                "chemmol_sha256": sha256_bytes(members[".chemmol.json"]),
                "meta_sha256": sha256_bytes(members[".meta.json"]),
            }
        )
    _write_csv(output / "paired.csv", paired_rows, PAIRED_FIELDS)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "built_pending_independent_validation",
        "freeze_id": FREEZE_ID,
        "role": ROLE,
        "split": SPLIT,
        "training_steps": 0,
        "parameter_updates": 0,
        "model_run": False,
        "threshold_or_checkpoint_selection": False,
        "sealed_rows_read": False,
        "selection_commitment_sha256": file_sha256(selection_commitment_path),
        "identity_isolation_report_sha256": file_sha256(
            _safe_relative_file(
                selection_commitment_path.parent,
                selection_commitment["identity_isolation_report"]["path"],
                context="identity_isolation_report",
            )
        ),
        "identity_isolation_status": identity_audit["status"],
        "molecules": len(molecule_rows),
        "samples": len(samples),
        "members_per_sample": len(MEMBER_SUFFIXES),
        "tar_members": len(samples) * len(MEMBER_SUFFIXES),
        "width": 768,
        "height": 768,
        "max_tokens": MAX_TOKENS,
        "sdf_protocol": SDF_PROTOCOL,
        "render_profile": RENDER_PROFILE,
        "rdkit_version": rdBase.rdkitVersion,
        "shard_sha256": file_sha256(shard),
        "molecules_manifest_sha256": file_sha256(output / "molecules.csv"),
        "samples_manifest_sha256": file_sha256(output / "samples.csv"),
        "paired_manifest_sha256": file_sha256(output / "paired.csv"),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "assembler_sha256": file_sha256(Path(assemble_c_sample.__code__.co_filename)),
        "layer_builder_sha256": file_sha256(Path(layer_builder.__file__)),
        "primitive_schema_sha256": file_sha256(schema_path),
    }
    (output / "summary.json").write_bytes(json_bytes(summary))
    input_commitment = {
        "schema_version": "visual-svg-layers-c-fresh-dev-preflight-input-0.1",
        "status": "built_pending_independent_validation",
        "freeze": {
            "path": os.path.relpath(freeze_path, output),
            "sha256": file_sha256(freeze_path),
        },
        "selection_commitment": {
            "path": os.path.relpath(selection_commitment_path, output),
            "sha256": file_sha256(selection_commitment_path),
        },
        "role": ROLE,
        "split": SPLIT,
        "committed_molecule_count": COMMITTED_MOLECULE_COUNT,
        "committed_sample_count": COMMITTED_SAMPLE_COUNT,
        "preflight_molecule_count": MOLECULE_COUNT,
        "preflight_sample_count": SAMPLE_COUNT,
        "selection_indices": list(range(MOLECULE_COUNT)),
        "members_per_sample": len(MEMBER_SUFFIXES),
        "shard": {
            "path": "shards/fresh-dev-preflight-000000.tar",
            "sha256": summary["shard_sha256"],
            "samples": SAMPLE_COUNT,
            "members": SAMPLE_COUNT * len(MEMBER_SUFFIXES),
        },
        "manifests": {
            "molecules": {"path": "molecules.csv", "sha256": summary["molecules_manifest_sha256"]},
            "samples": {"path": "samples.csv", "sha256": summary["samples_manifest_sha256"]},
            "paired": {"path": "paired.csv", "sha256": summary["paired_manifest_sha256"]},
        },
        "independent_validation": {
            "path": "preflight_validation.json",
            "required_status": "PREFLIGHT-PASS",
            "present": False,
        },
    }
    (output / "preflight_input_commitment.json").write_bytes(
        json_bytes(input_commitment)
    )
    return {"output": str(output), **summary}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    version_root = root / "visual_svg_layers_c_v0_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(root / "webdataset_full_v0"),
    )
    parser.add_argument(
        "--freeze",
        default=str(version_root / "phase_a_freeze.json"),
    )
    parser.add_argument(
        "--selection-commitment",
        default=str(version_root / "fresh_dev/selection.commitment.json"),
    )
    parser.add_argument(
        "--output",
        default=str(version_root / "fresh_dev/preflight_64"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = build(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "COMMITTED_MOLECULE_COUNT",
    "COMMITTED_SAMPLE_COUNT",
    "FORBIDDEN_MARKERS",
    "FREEZE_ID",
    "FREEZE_SHA256",
    "IDENTITY_AUDIT_SCHEMA_VERSION",
    "MOLECULE_COUNT",
    "PAIRED_FIELDS",
    "PreflightBuildError",
    "ROLE",
    "SAMPLE_COUNT",
    "SCHEMA_VERSION",
    "SELECTION_FIELDS",
    "SELECTION_SCHEMA_VERSION",
    "SPLIT",
    "VARIANTS",
    "build",
    "build_parser",
    "canonical_json_bytes",
    "load_committed_selection",
    "main",
    "selection_rank",
    "source_bytes_commitment_sha256",
    "style_id",
]
