"""Committed, split-isolated inputs for fresh Evidence Carrier evaluation.

The loader intentionally treats the commitment as the complete authority for
which files may be opened.  It never scans the dataset directory and rejects
path traversal, symlinks, sealed data, and validation/internal-test shards.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Iterator, Mapping, Sequence

import fastjsonschema
from PIL import Image
from rdkit import Chem

from Data.sdf.build_webdataset_sample import mol_from_chemmol
from Data.sdf.visual_primitives_v0_1 import (
    validate_visual_primitive_references,
)
from Data.sdf.visual_svg_sequence_v0_1 import serialize_visual_svg_v0_1
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from Data.sdf.visual_svg_v0_2 import (
    visual_primitives_v0_1_to_visual_svg_v0_2,
    visual_svg_v0_2_to_dict,
)


COMMITMENT_SCHEMA_VERSION = "visual-svg-layers-c-input-commitment-0.1"
INPUT_FINGERPRINT_SCHEMA_VERSION = "evidence-carrier-fresh-input-0.1"
ISOLATION_REPORT_SCHEMA_VERSION = "visual-svg-layers-c-isolation-report-0.1"
PHASE_A_FREEZE_SCHEMA_VERSION = "visual-svg-layers-c-phase-freeze-0.1"
PHASE_A_FREEZE_SHA256 = (
    "6c87e2e63b42c6b9f39c33d829fc8883d456b7c5f82cd15b2672963a7f560b9c"
)
PHASE_A_FREEZE_ID = "visual-svg-layers-c-v0.1-phase-a-20260722"
SELECTION_COMMITMENT_SCHEMA_VERSION = (
    "visual-svg-layers-c-fresh-dev-selection-commitment-0.1"
)
IDENTITY_ISOLATION_SCHEMA_VERSION = (
    "visual-svg-layers-c-fresh-dev-identity-isolation-0.1"
)
DEV_RENDER_PROFILE = "visual-svg-layers-c-clean-bw-768-v0.1"
DEV_SDF_PROTOCOL = "source-sdf-byte-preserving-0.1"
CHEMMOL_SCHEMA_SHA256 = (
    "9500d6b24abbbf3b0fbfdaa3aece2928d721a862a886fbfe1756a477a8b61a81"
)
DEV_FONT_SHA256 = frozenset(
    (
        "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45",
        "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e92778a78d149524eea9",
    )
)
DEV_FONT_FAMILIES = {
    "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45": "STIX",
    "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e92778a78d149524eea9": "P052",
}
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
ALLOWED_ROLES = frozenset(("smoke", "dev"))
ROLE_COUNTS = {
    "smoke": (8, 32, False, "smoke_train"),
    "dev": (2_000, 8_000, True, "fresh_dev"),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STYLE_ID_RE = re.compile(r"^style-v0\.1:([0-9a-f]{64})$")
_SAFE_SAMPLE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_SOURCE_SHARD_RE = re.compile(r"^shards/train-([0-9]{6})\.tar$")
_SOURCE_STRUCTURE_RE = re.compile(
    r"^structures/train-([0-9]{6})/[A-Za-z0-9][A-Za-z0-9_.-]*\.sdf$"
)
_FORBIDDEN_REFERENCE_MARKERS = frozenset(
    (
        "validation",
        "old_validation",
        "internal_test",
        "sealed",
        "reserve",
        "mr_tadf",
        "mrtadf",
    )
)
_MANIFEST_HASH_FIELDS = {
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
_MANIFEST_REQUIRED_FIELDS = frozenset(
    (
        "sample_key",
        "molecule_id",
        "inchikey",
        "split",
        "variant",
        "shard",
        *_MANIFEST_HASH_FIELDS,
    )
)
_DEV_MANIFEST_REQUIRED_FIELDS = frozenset(
    (
        "scaffold_id",
        "source_record_id",
        "style_id",
        "font_sha",
        "render_profile",
    )
)
_DEV_STRUCTURE_OVERLAP_FIELDS = (
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
)
_DEV_IDENTITY_OVERLAP_FIELDS = tuple(
    f"{name}_overlap" for name in _DEV_STRUCTURE_OVERLAP_FIELDS
)
_DEV_STYLE_OVERLAP_FIELDS = ("style_id", "font_sha", "render_profile")
_DEV_ISOLATION_OVERLAP_FIELDS = (
    *_DEV_STRUCTURE_OVERLAP_FIELDS,
    *_DEV_STYLE_OVERLAP_FIELDS,
)
_DEV_ISOLATION_EXPOSURE_SETS = (
    "frozen_b_p0_train_exposure",
    "smoke_train_32",
)
_DEV_ISOLATION_HASH_FIELDS = (
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
_DEV_IMPLEMENTATION_HASH_FIELDS = (
    "builder_sha256",
    "validator_sha256",
    "isolation_auditor_sha256",
    "style_id_algorithm_sha256",
    "scaffold_id_algorithm_sha256",
    "source_record_normalization_algorithm_sha256",
)
_STYLE_FIELDS = frozenset(
    (
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
    )
)
_DEV_MOLECULE_MANIFEST_FIELDS = frozenset(
    (
        "molecule_id",
        "inchikey",
        "scaffold_id",
        "source_record_id",
        "sdf_sha256",
    )
)
_DEV_SAMPLE_MANIFEST_FIELDS = frozenset(
    (
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
)
_DEV_SELECTION_MANIFEST_FIELDS = frozenset(
    (
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
)
_DEV_REPLACEMENT_MANIFEST_FIELDS = frozenset(
    (
        *(_DEV_SELECTION_MANIFEST_FIELDS - {"selection_index"}),
        "replacement_index",
        "purpose",
        "rendered",
    )
)


class FreshEvidenceCarrierDataError(ValueError):
    """Raised when a committed fresh-input package violates its contract."""


@dataclass(frozen=True, slots=True)
class FreshEvidenceCarrierIdentity:
    sample_key: str
    molecule_id: str
    inchikey: str
    variant: str
    split: str
    shard: str
    scaffold_id: str | None = None
    source_record_id: str | None = None
    style_id: str | None = None
    font_sha: str | None = None
    render_profile: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            name: value
            for name, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class FreshEvidenceCarrierShard:
    relative_path: str
    path: Path
    sha256: str
    byte_count: int
    sample_count: int
    member_count: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class FreshEvidenceCarrierArtifact:
    label: str
    relative_path: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    index: int
    identity: FreshEvidenceCarrierIdentity
    hashes: tuple[tuple[str, str], ...]

    @property
    def sample_key(self) -> str:
        return self.identity.sample_key

    @property
    def shard(self) -> str:
        return self.identity.shard


@dataclass(frozen=True, slots=True)
class _SelectionIdentity:
    index: int
    rank_sha256: str
    molecule_id: str
    inchikey: str
    scaffold_id: str
    source_record_id: str
    source: str
    source_id: str
    source_split: str
    source_shard: str
    source_structure: str
    source_sdf_sha256: str
    source_sdf_bytes: int
    source_chemmol_sha256: str
    source_atom_count: int
    source_bond_count: int


@dataclass(frozen=True, slots=True)
class FreshEvidenceCarrierPackage:
    role: str
    fresh_audit_eligible: bool
    root: Path
    commitment_path: Path
    commitment_sha256: str
    manifest_path: Path
    manifest_sha256: str
    validation_path: Path
    validation_sha256: str
    freeze_path: Path | None
    freeze_sha256: str | None
    isolation_path: Path | None
    isolation_sha256: str | None
    independence_verified: bool
    supporting_artifacts: tuple[FreshEvidenceCarrierArtifact, ...]
    shards: tuple[FreshEvidenceCarrierShard, ...]
    identities: tuple[FreshEvidenceCarrierIdentity, ...]
    input_fingerprint: str
    _rows: tuple[_ManifestRow, ...]
    _selections: tuple[_SelectionIdentity, ...]

    @property
    def sample_count(self) -> int:
        return len(self.identities)

    @property
    def molecule_count(self) -> int:
        return len({identity.molecule_id for identity in self.identities})

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        """Yield fully verified samples in exact manifest row order."""

        yield from _iter_package_samples(self)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _selection_rank(row: Mapping[str, str]) -> str:
    payload = "\0".join(
        (
            PHASE_A_FREEZE_ID,
            row["molecule_id"],
            row["inchikey_full"],
            row["scaffold_id"],
            row["source_record_id"],
        )
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _selection_source_bytes_sha256(rows: Sequence[Mapping[str, str]]) -> str:
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
    return _sha256_bytes(_canonical_json(payload))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreshEvidenceCarrierDataError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8")
        payload = json.loads(decoded, object_pairs_hook=_json_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshEvidenceCarrierDataError(f"Invalid JSON in {label}: {error}") from error
    if not isinstance(payload, dict):
        raise FreshEvidenceCarrierDataError(f"{label} must contain a JSON object")
    return payload


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FreshEvidenceCarrierDataError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _strict_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FreshEvidenceCarrierDataError(
            f"{label} must be a lowercase SHA256 digest"
        )
    return value


def _strict_decimal(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, str) or _DECIMAL_RE.fullmatch(value) is None:
        raise FreshEvidenceCarrierDataError(
            f"{label} must be a canonical non-negative decimal string"
        )
    parsed = int(value)
    if parsed < minimum:
        raise FreshEvidenceCarrierDataError(f"{label} must be >= {minimum}")
    return parsed


def _strict_style_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _STYLE_ID_RE.fullmatch(value) is None:
        raise FreshEvidenceCarrierDataError(
            f"{label} must use style-v0.1:<lowercase SHA256>"
        )
    return value


def _normalized_reference(value: Any) -> str:
    lowered = str(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def _reject_forbidden_reference(value: Any, *, label: str) -> None:
    normalized = _normalized_reference(value)
    compact = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    marker = next(
        (
            item
            for item in sorted(_FORBIDDEN_REFERENCE_MARKERS)
            if (
                item in normalized.split("_")
                or re.sub(r"[^a-z0-9]+", "", item) in compact
            )
        ),
        None,
    )
    if marker is not None:
        raise FreshEvidenceCarrierDataError(
            f"Forbidden {marker!r} reference in {label}"
        )


def _reject_forbidden_json_references(value: Any, *, label: str) -> None:
    if isinstance(value, str):
        _reject_forbidden_reference(value, label=label)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_forbidden_json_references(child, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_forbidden_json_references(child, label=f"{label}[{index}]")


def _absolute_without_resolving(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(os.path.abspath(path))


def _assert_regular_without_symlinks(path: Path, *, label: str) -> tuple[Path, os.stat_result]:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError as error:
            raise FreshEvidenceCarrierDataError(f"Missing {label}: {absolute}") from error
        if stat.S_ISLNK(current_stat.st_mode):
            raise FreshEvidenceCarrierDataError(f"Symlink rejected for {label}: {current}")
    final_stat = absolute.stat()
    if not stat.S_ISREG(final_stat.st_mode):
        raise FreshEvidenceCarrierDataError(f"{label} is not a regular file: {absolute}")
    return absolute, final_stat


def _safe_committed_path(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[str, Path, os.stat_result]:
    if not isinstance(value, str) or not value:
        raise FreshEvidenceCarrierDataError(f"{label} path must be a nonempty string")
    if "\\" in value or "\x00" in value:
        raise FreshEvidenceCarrierDataError(f"Unsafe {label} path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise FreshEvidenceCarrierDataError(f"Path escape rejected for {label}: {value!r}")
    _reject_forbidden_reference(value, label=f"{label} path")
    absolute, path_stat = _assert_regular_without_symlinks(
        root.joinpath(*pure.parts),
        label=label,
    )
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise FreshEvidenceCarrierDataError(
            f"Path escape rejected for {label}: {value!r}"
        ) from error
    return pure.as_posix(), absolute, path_stat


def _safe_validation_report_path(
    root: Path,
    value: Any,
) -> tuple[str, Path, os.stat_result]:
    """Resolve a report path while allowing the literal filename validation.json."""

    if not isinstance(value, str) or not value:
        raise FreshEvidenceCarrierDataError("Validation path must be nonempty")
    if "\\" in value or "\x00" in value:
        raise FreshEvidenceCarrierDataError("Unsafe validation report path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise FreshEvidenceCarrierDataError(
            "Validation report path escapes package root"
        )
    if pure.name != "validation.json":
        raise FreshEvidenceCarrierDataError(
            "Validation report filename must be validation.json"
        )
    for part in pure.parts[:-1]:
        _reject_forbidden_reference(part, label="validation report parent path")
    absolute, path_stat = _assert_regular_without_symlinks(
        root.joinpath(*pure.parts),
        label="validation report",
    )
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise FreshEvidenceCarrierDataError(
            "Validation report path escapes package root"
        ) from error
    return pure.as_posix(), absolute, path_stat


def _safe_preflight_report_path(
    root: Path,
    value: Any,
) -> tuple[str, Path, os.stat_result]:
    if not isinstance(value, str) or not value:
        raise FreshEvidenceCarrierDataError("Preflight validation path must be nonempty")
    if "\\" in value or "\x00" in value:
        raise FreshEvidenceCarrierDataError("Unsafe preflight validation path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise FreshEvidenceCarrierDataError(
            "Preflight validation path escapes package root"
        )
    if pure.name != "preflight_validation.json":
        raise FreshEvidenceCarrierDataError(
            "Preflight report filename must be preflight_validation.json"
        )
    for part in pure.parts[:-1]:
        _reject_forbidden_reference(part, label="preflight report parent path")
    absolute, path_stat = _assert_regular_without_symlinks(
        root.joinpath(*pure.parts),
        label="preflight validation report",
    )
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise FreshEvidenceCarrierDataError(
            "Preflight validation path escapes package root"
        ) from error
    return pure.as_posix(), absolute, path_stat


def _verified_artifact_reference(
    root: Path,
    reference: Any,
    *,
    label: str,
    preflight_report: bool = False,
) -> FreshEvidenceCarrierArtifact:
    if not isinstance(reference, Mapping):
        raise FreshEvidenceCarrierDataError(f"Commitment {label} must be an object")
    if preflight_report:
        relative, path, _ = _safe_preflight_report_path(
            root,
            reference.get("path"),
        )
    else:
        relative, path, _ = _safe_committed_path(
            root,
            reference.get("path"),
            label=label,
        )
    expected = _strict_sha256(
        reference.get("sha256"),
        label=f"commitment.{label}.sha256",
    )
    if _sha256_file(path) != expected:
        raise FreshEvidenceCarrierDataError(f"{label} SHA256 mismatch")
    return FreshEvidenceCarrierArtifact(
        label=label,
        relative_path=relative,
        path=path,
        sha256=expected,
    )


def _style_id_from_payload(payload: Mapping[str, Any]) -> str:
    if set(payload) != _STYLE_FIELDS:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev canonical style payload has incorrect fields"
        )
    if payload.get("render_profile") != DEV_RENDER_PROFILE:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev canonical style render_profile mismatch"
        )
    if payload.get("font_sha256") not in DEV_FONT_SHA256:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev canonical style font is outside the allowlist"
        )
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "style-v0.1:" + _sha256_bytes(encoded)


def _require_role_path(relative_path: str, *, role: str, label: str) -> None:
    expected = "smoke" if role == "smoke" else "dev"
    if expected not in _normalized_reference(relative_path).split("_"):
        raise FreshEvidenceCarrierDataError(
            f"{label} is not split-specific for role {role!r}: {relative_path}"
        )


def _load_manifest(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if len(fields) != len(set(fields)):
                raise FreshEvidenceCarrierDataError("Manifest has duplicate columns")
            missing = sorted(_MANIFEST_REQUIRED_FIELDS - set(fields))
            if missing:
                raise FreshEvidenceCarrierDataError(
                    "Manifest is missing fields: " + ", ".join(missing)
                )
            rows = list(reader)
    except UnicodeDecodeError as error:
        raise FreshEvidenceCarrierDataError("Manifest is not valid UTF-8") from error
    if not rows:
        raise FreshEvidenceCarrierDataError("Manifest cannot be empty")
    for row_index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise FreshEvidenceCarrierDataError(
                f"Manifest row {row_index} has an invalid column count"
            )
    return rows, fields


def _validate_manifest_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    role: str,
    declared_shards: frozenset[str],
) -> tuple[_ManifestRow, ...]:
    expected_molecules, expected_samples, _, expected_split = ROLE_COUNTS[role]
    if len(rows) != expected_samples:
        raise FreshEvidenceCarrierDataError(
            f"Role {role!r} requires exactly {expected_samples} manifest rows"
        )
    parsed: list[_ManifestRow] = []
    seen_keys: set[str] = set()
    variants_by_molecule: dict[str, list[str]] = defaultdict(list)
    inchikeys_by_molecule: dict[str, set[str]] = defaultdict(set)
    sdf_hashes_by_molecule: dict[str, set[str]] = defaultdict(set)
    scaffold_ids_by_molecule: dict[str, set[str]] = defaultdict(set)
    source_records_by_molecule: dict[str, set[str]] = defaultdict(set)
    molecule_by_inchikey: dict[str, str] = {}
    molecule_by_source_record: dict[str, str] = {}
    for index, row in enumerate(rows):
        row_number = index + 2
        if role == "dev":
            missing_dev = sorted(_DEV_MANIFEST_REQUIRED_FIELDS - set(row))
            if missing_dev:
                raise FreshEvidenceCarrierDataError(
                    "Fresh-dev manifest is missing fields: "
                    + ", ".join(missing_dev)
                )
        for field in _MANIFEST_REQUIRED_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not value:
                raise FreshEvidenceCarrierDataError(
                    f"Manifest row {row_number} field {field!r} is empty"
                )
            _reject_forbidden_reference(value, label=f"manifest row {row_number}.{field}")
        sample_key = row["sample_key"]
        if _SAFE_SAMPLE_KEY_RE.fullmatch(sample_key) is None:
            raise FreshEvidenceCarrierDataError(
                f"Unsafe sample key in manifest row {row_number}: {sample_key!r}"
            )
        if sample_key in seen_keys:
            raise FreshEvidenceCarrierDataError(f"Duplicate sample key: {sample_key}")
        seen_keys.add(sample_key)
        if row["split"] != expected_split:
            raise FreshEvidenceCarrierDataError(
                f"Manifest row {row_number} has split={row['split']!r}; "
                f"expected {expected_split!r}"
            )
        if "validation" in _normalized_reference(row["split"]):
            raise FreshEvidenceCarrierDataError("Validation split is forbidden")
        variant = row["variant"]
        if variant not in VARIANTS:
            raise FreshEvidenceCarrierDataError(
                f"Manifest row {row_number} has unsupported variant {variant!r}"
            )
        shard = row["shard"]
        if shard not in declared_shards:
            raise FreshEvidenceCarrierDataError(
                f"Manifest row {row_number} references an uncommitted shard: {shard}"
            )
        if "validation" in _normalized_reference(shard):
            raise FreshEvidenceCarrierDataError(
                f"Validation shard is forbidden: {shard}"
            )
        hashes = []
        for field, suffix in _MANIFEST_HASH_FIELDS.items():
            hashes.append(
                (suffix, _strict_sha256(row[field], label=f"manifest {sample_key}.{field}"))
            )
        identity = FreshEvidenceCarrierIdentity(
            sample_key=sample_key,
            molecule_id=row["molecule_id"],
            inchikey=row["inchikey"],
            variant=variant,
            split=row["split"],
            shard=shard,
            scaffold_id=row.get("scaffold_id") or None,
            source_record_id=row.get("source_record_id") or None,
            style_id=row.get("style_id") or None,
            font_sha=row.get("font_sha") or None,
            render_profile=row.get("render_profile") or None,
        )
        if role == "dev":
            for field in _DEV_MANIFEST_REQUIRED_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise FreshEvidenceCarrierDataError(
                        f"Fresh-dev manifest row {row_number} field {field!r} is empty"
                    )
                _reject_forbidden_reference(
                    value,
                    label=f"fresh-dev manifest row {row_number}.{field}",
                )
            _strict_sha256(
                row["font_sha"],
                label=f"fresh-dev manifest row {row_number}.font_sha",
            )
            _strict_style_id(
                row["style_id"],
                label=f"fresh-dev manifest row {row_number}.style_id",
            )
            if row["font_sha"] not in DEV_FONT_SHA256:
                raise FreshEvidenceCarrierDataError(
                    f"Fresh-dev manifest row {row_number} font is outside the allowlist"
                )
            if row["render_profile"] != DEV_RENDER_PROFILE:
                raise FreshEvidenceCarrierDataError(
                    f"Fresh-dev manifest row {row_number} render_profile mismatch"
                )
        parsed.append(_ManifestRow(index=index, identity=identity, hashes=tuple(hashes)))
        variants_by_molecule[identity.molecule_id].append(variant)
        inchikeys_by_molecule[identity.molecule_id].add(identity.inchikey)
        sdf_hashes_by_molecule[identity.molecule_id].add(
            dict(hashes)[".sdf"]
        )
        if identity.scaffold_id is not None:
            scaffold_ids_by_molecule[identity.molecule_id].add(identity.scaffold_id)
        if identity.source_record_id is not None:
            source_records_by_molecule[identity.molecule_id].add(
                identity.source_record_id
            )
        if role == "dev":
            prior_molecule = molecule_by_inchikey.setdefault(
                identity.inchikey,
                identity.molecule_id,
            )
            if prior_molecule != identity.molecule_id:
                raise FreshEvidenceCarrierDataError(
                    "Fresh-dev InChIKey maps to more than one molecule_id"
                )
            assert identity.source_record_id is not None
            prior_source = molecule_by_source_record.setdefault(
                identity.source_record_id,
                identity.molecule_id,
            )
            if prior_source != identity.molecule_id:
                raise FreshEvidenceCarrierDataError(
                    "Fresh-dev source_record_id maps to more than one molecule_id"
                )

    if len(variants_by_molecule) != expected_molecules:
        raise FreshEvidenceCarrierDataError(
            f"Role {role!r} requires exactly {expected_molecules} molecules"
        )
    for molecule_id, variants in variants_by_molecule.items():
        if len(variants) != len(VARIANTS) or set(variants) != set(VARIANTS):
            raise FreshEvidenceCarrierDataError(
                f"Molecule {molecule_id!r} does not contain exactly four layouts"
            )
        if len(inchikeys_by_molecule[molecule_id]) != 1:
            raise FreshEvidenceCarrierDataError(
                f"Molecule {molecule_id!r} has inconsistent InChIKeys"
            )
        if len(sdf_hashes_by_molecule[molecule_id]) != 1:
            raise FreshEvidenceCarrierDataError(
                f"Molecule {molecule_id!r} does not share one SDF across layouts"
            )
        if role == "dev" and len(scaffold_ids_by_molecule[molecule_id]) != 1:
            raise FreshEvidenceCarrierDataError(
                f"Molecule {molecule_id!r} has inconsistent scaffold_id"
            )
        if role == "dev" and len(source_records_by_molecule[molecule_id]) != 1:
            raise FreshEvidenceCarrierDataError(
                f"Molecule {molecule_id!r} has inconsistent source_record_id"
            )
    return tuple(parsed)


def _read_projection_manifest(
    path: Path,
    *,
    required_fields: frozenset[str],
    label: str,
    exact_fields: bool = False,
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if len(fields) != len(set(fields)):
                raise FreshEvidenceCarrierDataError(
                    f"{label} has duplicate columns"
                )
            missing = sorted(required_fields - set(fields))
            if missing:
                raise FreshEvidenceCarrierDataError(
                    f"{label} is missing fields: {', '.join(missing)}"
                )
            if exact_fields and set(fields) != set(required_fields):
                raise FreshEvidenceCarrierDataError(
                    f"{label} fields differ from the frozen protocol"
                )
            rows = list(reader)
    except UnicodeDecodeError as error:
        raise FreshEvidenceCarrierDataError(f"{label} is not valid UTF-8") from error
    for row_number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise FreshEvidenceCarrierDataError(
                f"{label} row {row_number} has an invalid column count"
            )
        for name in required_fields:
            value = row[name]
            if not value:
                raise FreshEvidenceCarrierDataError(
                    f"{label} row {row_number} field {name!r} is empty"
                )
            _reject_forbidden_reference(
                value,
                label=f"{label} row {row_number}.{name}",
            )
    return rows


def _validate_selection_identity(
    row: Mapping[str, str],
    *,
    expected_index: int,
    index_field: str,
    label: str,
) -> _SelectionIdentity:
    index = _strict_decimal(
        row.get(index_field),
        label=f"{label}.{index_field}",
    )
    if index != expected_index:
        raise FreshEvidenceCarrierDataError(
            f"{label} {index_field} is not contiguous"
        )
    rank_sha256 = _strict_sha256(
        row.get("rank_sha256"),
        label=f"{label}.rank_sha256",
    )
    if rank_sha256 != _selection_rank(row):
        raise FreshEvidenceCarrierDataError(f"{label} rank_sha256 mismatch")
    source = row["source"]
    source_id = row["source_id"]
    if _SAFE_SAMPLE_KEY_RE.fullmatch(source) is None or (
        _SAFE_SAMPLE_KEY_RE.fullmatch(source_id) is None
    ):
        raise FreshEvidenceCarrierDataError(f"{label} source identity is unsafe")
    source_record_id = f"{source}:{source_id}"
    if (
        row["source_record_id"] != source_record_id
        or row["molecule_id"] != source_record_id
    ):
        raise FreshEvidenceCarrierDataError(
            f"{label} source_record_id or molecule_id mismatch"
        )
    if row["source_split"] != "train":
        raise FreshEvidenceCarrierDataError(f"{label} source_split must be train")
    shard_match = _SOURCE_SHARD_RE.fullmatch(row["source_shard"])
    structure_match = _SOURCE_STRUCTURE_RE.fullmatch(row["source_structure"])
    if (
        shard_match is None
        or structure_match is None
        or shard_match.group(1) != structure_match.group(1)
    ):
        raise FreshEvidenceCarrierDataError(
            f"{label} source shard/structure reference mismatch"
        )
    source_sdf_sha256 = _strict_sha256(
        row["source_sdf_sha256"],
        label=f"{label}.source_sdf_sha256",
    )
    source_chemmol_sha256 = _strict_sha256(
        row["source_chemmol_sha256"],
        label=f"{label}.source_chemmol_sha256",
    )
    return _SelectionIdentity(
        index=index,
        rank_sha256=rank_sha256,
        molecule_id=row["molecule_id"],
        inchikey=row["inchikey_full"],
        scaffold_id=row["scaffold_id"],
        source_record_id=row["source_record_id"],
        source=source,
        source_id=source_id,
        source_split=row["source_split"],
        source_shard=row["source_shard"],
        source_structure=row["source_structure"],
        source_sdf_sha256=source_sdf_sha256,
        source_sdf_bytes=_strict_decimal(
            row["source_sdf_bytes"],
            label=f"{label}.source_sdf_bytes",
            minimum=1,
        ),
        source_chemmol_sha256=source_chemmol_sha256,
        source_atom_count=_strict_decimal(
            row["source_atom_count"],
            label=f"{label}.source_atom_count",
            minimum=1,
        ),
        source_bond_count=_strict_decimal(
            row["source_bond_count"],
            label=f"{label}.source_bond_count",
        ),
    )


def _validate_selection_identity_sets(
    selection: Sequence[_SelectionIdentity],
    replacement: Sequence[_SelectionIdentity],
) -> None:
    for name in ("molecule_id", "inchikey", "source_record_id"):
        selected_values = [getattr(item, name) for item in selection]
        if len(selected_values) != len(set(selected_values)):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev selection contains duplicate {name}"
            )
    for name in ("molecule_id", "inchikey", "scaffold_id", "source_record_id"):
        selected_values = {getattr(item, name) for item in selection}
        replacement_values = [getattr(item, name) for item in replacement]
        if len(replacement_values) != len(set(replacement_values)):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev replacement contains duplicate {name}"
            )
        if selected_values.intersection(replacement_values):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev selection/replacement overlap on {name}"
            )
    selected_order = [(item.rank_sha256, item.molecule_id) for item in selection]
    replacement_order = [(item.rank_sha256, item.molecule_id) for item in replacement]
    if selected_order != sorted(selected_order):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev selection is not ordered by frozen rank"
        )
    if replacement_order != sorted(replacement_order):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev replacement is not ordered by frozen rank"
        )
    if selected_order and replacement_order and replacement_order[0] <= selected_order[-1]:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev replacement does not follow the selected rank prefix"
        )


def _validate_dev_projection_manifests(
    rows: Sequence[_ManifestRow],
    *,
    molecule_manifest_path: Path,
    sample_manifest_path: Path,
    selection_rows: Sequence[_SelectionIdentity],
) -> None:
    molecule_rows = _read_projection_manifest(
        molecule_manifest_path,
        required_fields=_DEV_MOLECULE_MANIFEST_FIELDS,
        label="Fresh-dev molecule manifest",
    )
    sample_rows = _read_projection_manifest(
        sample_manifest_path,
        required_fields=_DEV_SAMPLE_MANIFEST_FIELDS,
        label="Fresh-dev sample manifest",
    )
    expected_molecules, expected_samples, _, _ = ROLE_COUNTS["dev"]
    if len(molecule_rows) != expected_molecules:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev molecule manifest count mismatch"
        )
    if len(sample_rows) != expected_samples:
        raise FreshEvidenceCarrierDataError("Fresh-dev sample manifest count mismatch")
    expected_by_molecule: dict[str, dict[str, str]] = {}
    for row in rows:
        identity = row.identity
        hashes = dict(row.hashes)
        expected = {
            "molecule_id": identity.molecule_id,
            "inchikey": identity.inchikey,
            "scaffold_id": str(identity.scaffold_id),
            "source_record_id": str(identity.source_record_id),
            "sdf_sha256": hashes[".sdf"],
        }
        prior = expected_by_molecule.setdefault(identity.molecule_id, expected)
        if prior != expected:
            raise FreshEvidenceCarrierDataError(
                "Paired manifest molecule projection is inconsistent"
            )
    actual_by_molecule = {row["molecule_id"]: row for row in molecule_rows}
    if len(actual_by_molecule) != len(molecule_rows):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev molecule manifest contains duplicate molecule_id"
        )
    for molecule_id, expected in expected_by_molecule.items():
        actual = actual_by_molecule.get(molecule_id)
        if actual is None or any(actual[name] != value for name, value in expected.items()):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev molecule manifest differs for {molecule_id}"
            )
    if len(selection_rows) != len(molecule_rows):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev selection and molecule manifest counts differ"
        )
    for selected, molecule in zip(selection_rows, molecule_rows):
        expected = {
            "molecule_id": selected.molecule_id,
            "inchikey": selected.inchikey,
            "scaffold_id": selected.scaffold_id,
            "source_record_id": selected.source_record_id,
            "sdf_sha256": selected.source_sdf_sha256,
        }
        if any(molecule[name] != value for name, value in expected.items()):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev molecule manifest differs from committed selection"
            )
    if len(sample_rows) != len(rows):
        raise FreshEvidenceCarrierDataError("Fresh-dev sample projection count mismatch")
    for projected, paired in zip(sample_rows, rows):
        identity = paired.identity
        hashes = dict(paired.hashes)
        expected = {
            "sample_key": identity.sample_key,
            "molecule_id": identity.molecule_id,
            "split": identity.split,
            "variant": identity.variant,
            "style_id": str(identity.style_id),
            "font_sha": str(identity.font_sha),
            "render_profile": str(identity.render_profile),
            "image_sha256": hashes[".png"],
            "sdf_sha256": hashes[".sdf"],
        }
        if any(projected[name] != value for name, value in expected.items()):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev sample manifest differs for {identity.sample_key}"
            )


def _validate_tar_members(
    shard: FreshEvidenceCarrierShard,
    rows: Sequence[_ManifestRow],
) -> None:
    expected_names = {
        row.sample_key + suffix for row in rows for suffix in MEMBER_SUFFIXES
    }
    try:
        with tarfile.open(shard.path, "r:*") as archive:
            members = archive.getmembers()
    except (tarfile.TarError, OSError) as error:
        raise FreshEvidenceCarrierDataError(
            f"Cannot read committed shard {shard.relative_path}: {error}"
        ) from error
    actual_names: list[str] = []
    for member in members:
        if not member.isfile():
            raise FreshEvidenceCarrierDataError(
                f"Shard member is not a regular file: {member.name}"
            )
        if "/" in member.name or "\\" in member.name:
            raise FreshEvidenceCarrierDataError(
                f"Shard member contains a path: {member.name}"
            )
        actual_names.append(member.name)
    if len(actual_names) != len(set(actual_names)):
        raise FreshEvidenceCarrierDataError(
            f"Shard {shard.relative_path} contains duplicate member names"
        )
    if len(actual_names) != shard.member_count or set(actual_names) != expected_names:
        missing = sorted(expected_names - set(actual_names))[:3]
        extra = sorted(set(actual_names) - expected_names)[:3]
        raise FreshEvidenceCarrierDataError(
            f"Shard {shard.relative_path} does not contain exactly 12 members per sample; "
            f"missing={missing}, extra={extra}"
        )


def _validate_validation_report(
    payload: Mapping[str, Any],
    *,
    role: str,
    commitment_status: str,
) -> None:
    expected_molecules, expected_samples, eligible, _ = ROLE_COUNTS[role]
    if role == "dev" and payload.get("schema_version") != (
        "visual-svg-layers-c-fresh-dev-validation-0.1"
    ):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev validation report schema_version mismatch"
        )
    status = payload.get("status")
    if status != commitment_status or str(status).lower() not in {"passed", "smoke-pass"}:
        raise FreshEvidenceCarrierDataError(
            "Validation report and commitment must both declare a passed status"
        )
    if payload.get("fresh_audit_eligible") is not eligible:
        raise FreshEvidenceCarrierDataError(
            "Validation fresh_audit_eligible disagrees with the role contract"
        )
    if _strict_int(payload.get("molecules"), label="validation.molecules") != expected_molecules:
        raise FreshEvidenceCarrierDataError("Validation molecule count is incorrect")
    if _strict_int(payload.get("samples"), label="validation.samples") != expected_samples:
        raise FreshEvidenceCarrierDataError("Validation sample count is incorrect")
    declared_members = payload.get("members", payload.get("tar_members"))
    if _strict_int(
        declared_members,
        label="validation.members",
    ) != expected_samples * len(MEMBER_SUFFIXES):
        raise FreshEvidenceCarrierDataError("Validation member count is incorrect")
    if role == "dev":
        if _strict_int(
            payload.get("checked_samples"),
            label="validation.checked_samples",
        ) != expected_samples:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev validation checked_samples is incorrect"
            )
        if _strict_int(
            payload.get("checked_members"),
            label="validation.checked_members",
        ) != expected_samples * len(MEMBER_SUFFIXES):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev validation checked_members is incorrect"
            )
        if _strict_int(
            payload.get("structure_validated_samples"),
            label="validation.structure_validated_samples",
        ) != expected_samples:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev validation structure count is incorrect"
            )
        if payload.get("errors") != []:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev validation report contains errors"
            )


def _validate_dev_validation_bindings(
    payload: Mapping[str, Any],
    *,
    commitment: Mapping[str, Any],
    manifest_sha256: str,
    isolation_sha256: str,
    supporting_artifacts: Sequence[FreshEvidenceCarrierArtifact],
    shards: Sequence[FreshEvidenceCarrierShard],
) -> None:
    artifacts = {artifact.label: artifact for artifact in supporting_artifacts}
    expected_hashes = {
        "paired_manifest_sha256": manifest_sha256,
        "molecule_manifest_sha256": artifacts["molecule manifest"].sha256,
        "sample_manifest_sha256": artifacts["sample manifest"].sha256,
        "preflight_validation_sha256": artifacts["preflight validation"].sha256,
        "isolation_report_sha256": isolation_sha256,
        "selection_commitment_sha256": artifacts["selection commitment"].sha256,
        "selection_manifest_sha256": artifacts["selection manifest"].sha256,
        "replacement_manifest_sha256": artifacts["replacement manifest"].sha256,
        "identity_isolation_report_sha256": artifacts[
            "identity isolation report"
        ].sha256,
        "selection_provenance_sha256": artifacts["selection provenance"].sha256,
    }
    actual_hashes = payload.get("artifact_sha256")
    if not isinstance(actual_hashes, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev validation report lacks artifact_sha256"
        )
    if set(actual_hashes) != set(expected_hashes):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev validation artifact binding fields differ from the protocol"
        )
    for name, expected in expected_hashes.items():
        if actual_hashes.get(name) != expected:
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev validation artifact binding mismatch: {name}"
            )
    implementation = commitment.get("implementation")
    if not isinstance(implementation, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev commitment implementation is invalid"
        )
    validation_implementation = payload.get("implementation_sha256")
    if validation_implementation != dict(implementation):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev validation implementation bindings differ from commitment"
        )
    expected_fonts = [
        {"family": family, "sha256": digest}
        for digest, family in DEV_FONT_FAMILIES.items()
    ]
    if commitment.get("fonts") != expected_fonts:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev commitment font inventory is not canonical"
        )
    if payload.get("font_inventory") != expected_fonts:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev validation font inventory differs from commitment"
        )
    shard_rows = payload.get("shards")
    expected_shards = [
        {
            "path": shard.relative_path,
            "sha256": shard.sha256,
            "samples": shard.sample_count,
            "members": shard.member_count,
        }
        for shard in shards
    ]
    if shard_rows != expected_shards:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev validation shard bindings differ from commitment"
        )


def _validate_preflight_report(
    payload: Mapping[str, Any],
    *,
    commitment_status: Any,
    selection_commitment_sha256: str,
) -> None:
    expected = {
        "schema_version": "visual-svg-layers-c-fresh-dev-preflight-validation-0.1",
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
    }
    if commitment_status != "PREFLIGHT-PASS":
        raise FreshEvidenceCarrierDataError(
            "Commitment preflight validation status is not PREFLIGHT-PASS"
        )
    for name, value in expected.items():
        if payload.get(name) != value:
            raise FreshEvidenceCarrierDataError(
                f"Preflight validation field {name!r} mismatch"
            )
    if payload.get("errors") != []:
        raise FreshEvidenceCarrierDataError("Preflight validation contains errors")
    if payload.get("selection_commitment_sha256") != selection_commitment_sha256:
        raise FreshEvidenceCarrierDataError(
            "Preflight validation selection commitment mismatch"
        )


def _validate_selection_commitment(
    root: Path,
    reference: Any,
    *,
    freeze_sha256: str,
) -> tuple[
    list[FreshEvidenceCarrierArtifact],
    list[_SelectionIdentity],
    Mapping[str, Any],
]:
    selection_artifact = _verified_artifact_reference(
        root,
        reference,
        label="selection commitment",
    )
    if selection_artifact.relative_path != "selection.commitment.json":
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev selection commitment path mismatch"
        )
    payload = _load_json_bytes(
        selection_artifact.path.read_bytes(),
        label="selection commitment",
    )
    expected = {
        "schema_version": SELECTION_COMMITMENT_SCHEMA_VERSION,
        "status": "committed",
        "freeze_id": PHASE_A_FREEZE_ID,
        "role": "dev",
        "split": "fresh_dev",
        "molecule_count": ROLE_COUNTS["dev"][0],
        "sample_count": ROLE_COUNTS["dev"][1],
        "selection_ordering": "sha256_rank_v0.1",
        "sealed_rows_materialized": False,
        "reserve_rows_materialized": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise FreshEvidenceCarrierDataError(
                f"Selection commitment field {name!r} mismatch"
            )
    freeze = payload.get("freeze")
    if not isinstance(freeze, Mapping) or (
        freeze.get("path") != "../phase_a_freeze.json"
        or freeze.get("sha256") != freeze_sha256
    ):
        raise FreshEvidenceCarrierDataError(
            "Selection commitment freeze binding mismatch"
        )
    preflight = payload.get("preflight")
    if not isinstance(preflight, Mapping) or (
        preflight.get("selection_indices") != list(range(16))
        or preflight.get("molecule_count") != 16
        or preflight.get("sample_count") != 64
        or preflight.get("counts_toward_final_fresh_dev") is not True
        or preflight.get("run_model") is not False
    ):
        raise FreshEvidenceCarrierDataError(
            "Selection commitment preflight contract mismatch"
        )
    source_bytes = payload.get("source_bytes_commitment")
    if not isinstance(source_bytes, Mapping) or (
        source_bytes.get("protocol") != DEV_SDF_PROTOCOL
        or source_bytes.get("molecule_count") != ROLE_COUNTS["dev"][0]
    ):
        raise FreshEvidenceCarrierDataError(
            "Selection source-byte commitment mismatch"
        )
    source_bytes_sha256 = _strict_sha256(
        source_bytes.get("sha256"),
        label="selection.source_bytes_commitment.sha256",
    )
    artifacts = [selection_artifact]
    references = (
        ("selection manifest", payload.get("selection_manifest")),
        ("replacement manifest", payload.get("fresh_replacement_manifest")),
        ("identity isolation report", payload.get("identity_isolation_report")),
        ("selection provenance", payload.get("selection_provenance")),
    )
    by_label: dict[str, FreshEvidenceCarrierArtifact] = {}
    for label, artifact_reference in references:
        artifact = _verified_artifact_reference(
            root,
            artifact_reference,
            label=label,
        )
        artifacts.append(artifact)
        by_label[label] = artifact
    replacement_reference = payload.get("fresh_replacement_manifest")
    if not isinstance(replacement_reference, Mapping) or (
        replacement_reference.get("molecule_count") != 500
        or replacement_reference.get("rendered") is not False
    ):
        raise FreshEvidenceCarrierDataError(
            "Selection replacement-pool commitment mismatch"
        )
    identity_reference = payload.get("identity_isolation_report")
    if not isinstance(identity_reference, Mapping) or identity_reference.get("status") != "passed":
        raise FreshEvidenceCarrierDataError(
            "Selection identity isolation status mismatch"
        )
    selection_rows = _read_projection_manifest(
        by_label["selection manifest"].path,
        required_fields=_DEV_SELECTION_MANIFEST_FIELDS,
        label="Fresh-dev selection manifest",
        exact_fields=True,
    )
    if len(selection_rows) != ROLE_COUNTS["dev"][0]:
        raise FreshEvidenceCarrierDataError("Fresh-dev selection manifest count mismatch")
    if _selection_source_bytes_sha256(selection_rows) != source_bytes_sha256:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev selection source-byte commitment mismatch"
        )
    selection = [
        _validate_selection_identity(
            row,
            expected_index=index,
            index_field="selection_index",
            label=f"Fresh-dev selection row {index + 2}",
        )
        for index, row in enumerate(selection_rows)
    ]
    replacement_rows = _read_projection_manifest(
        by_label["replacement manifest"].path,
        required_fields=_DEV_REPLACEMENT_MANIFEST_FIELDS,
        label="Fresh-dev replacement manifest",
        exact_fields=True,
    )
    if len(replacement_rows) != 500:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev replacement manifest count mismatch"
        )
    replacement = []
    for index, row in enumerate(replacement_rows):
        if (
            row["purpose"] != "fresh_dev_pre_render_replacement"
            or row["rendered"] != "false"
        ):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev replacement purpose/rendered contract mismatch"
            )
        replacement.append(
            _validate_selection_identity(
                row,
                expected_index=index,
                index_field="replacement_index",
                label=f"Fresh-dev replacement row {index + 2}",
            )
        )
    _validate_selection_identity_sets(selection, replacement)
    provenance = _load_json_bytes(
        by_label["selection provenance"].path.read_bytes(),
        label="selection provenance",
    )
    expected_provenance = {
        "schema_version": "visual-svg-layers-c-fresh-dev-selection-0.1",
        "status": "selected_pending_independent_identity_audit",
        "freeze_id": PHASE_A_FREEZE_ID,
        "freeze_sha256": freeze_sha256,
        "source_split": "train",
        "fresh_dev_molecules": ROLE_COUNTS["dev"][0],
        "fresh_dev_samples_planned": ROLE_COUNTS["dev"][1],
        "fresh_replacement_molecules": 500,
        "fresh_replacement_rendered": 0,
        "structural_rejected": 0,
        "sealed_rows_materialized": False,
        "reserve_rows_materialized": False,
        "selection_manifest_sha256": by_label["selection manifest"].sha256,
        "selection_source_bytes_commitment_sha256": source_bytes_sha256,
        "replacement_manifest_sha256": by_label["replacement manifest"].sha256,
    }
    for name, expected_value in expected_provenance.items():
        if provenance.get(name) != expected_value:
            raise FreshEvidenceCarrierDataError(
                f"Selection provenance field {name!r} mismatch"
            )
    selection_config = provenance.get("selection_config")
    expected_selection_config = {
        "ordering": "sha256_rank_v0.1",
        "fresh_count": ROLE_COUNTS["dev"][0],
        "replacement_count": 500,
        "selection_scan_limit": 20_000,
        "exclusion_match_policy": "reject_candidate_on_any_key_match",
        "identity_fields": list(_DEV_STRUCTURE_OVERLAP_FIELDS),
    }
    if selection_config != expected_selection_config or provenance.get(
        "selection_config_sha256"
    ) != _sha256_bytes(_canonical_json(expected_selection_config)):
        raise FreshEvidenceCarrierDataError(
            "Selection provenance configuration mismatch"
        )
    identity_report = _load_json_bytes(
        by_label["identity isolation report"].path.read_bytes(),
        label="identity isolation report",
    )
    if (
        identity_report.get("schema_version") != IDENTITY_ISOLATION_SCHEMA_VERSION
        or identity_report.get("status") != "passed"
        or identity_report.get("producer") != "independent_identity_auditor"
        or identity_report.get("freeze_id") != PHASE_A_FREEZE_ID
        or identity_report.get("fresh_dev_selection_manifest_sha256")
        != by_label["selection manifest"].sha256
        or identity_report.get("fresh_dev_source_bytes_commitment_sha256")
        != source_bytes_sha256
        or identity_report.get("fresh_dev_replacement_manifest_sha256")
        != by_label["replacement manifest"].sha256
        or identity_report.get("selection_provenance_sha256")
        != by_label["selection provenance"].sha256
        or identity_report.get("fresh_dev_molecules") != ROLE_COUNTS["dev"][0]
        or identity_report.get("fresh_replacement_molecules") != 500
        or identity_report.get("structure_prechecked_molecules") != 2_500
        or identity_report.get("structural_rejections") != 0
        or identity_report.get("old_validation_internal_or_sealed_rows_read") is not False
        or identity_report.get("sealed_rows_materialized") is not False
        or identity_report.get("reserve_rows_materialized") is not False
    ):
        raise FreshEvidenceCarrierDataError(
            "Identity isolation report contract mismatch"
        )
    fresh_replacement_overlap = identity_report.get("fresh_replacement_overlap")
    if not isinstance(fresh_replacement_overlap, Mapping) or set(
        fresh_replacement_overlap
    ) != set(_DEV_IDENTITY_OVERLAP_FIELDS):
        raise FreshEvidenceCarrierDataError(
            "Identity isolation fresh/replacement fields differ"
        )
    if set(fresh_replacement_overlap.values()) != {0}:
        raise FreshEvidenceCarrierDataError(
            "Identity isolation reports fresh/replacement overlap"
        )
    comparisons = identity_report.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Identity isolation report lacks comparisons"
        )
    for exposure in _DEV_ISOLATION_EXPOSURE_SETS:
        values = comparisons.get(exposure)
        if not isinstance(values, Mapping) or set(values) != set(
            _DEV_IDENTITY_OVERLAP_FIELDS
        ):
            raise FreshEvidenceCarrierDataError(
                f"Identity isolation comparison fields differ for {exposure}"
            )
        if set(values.values()) != {0}:
            raise FreshEvidenceCarrierDataError(
                f"Identity isolation overlap is nonzero for {exposure}"
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
        _strict_sha256(
            identity_report.get(name),
            label=f"identity_isolation.{name}",
        )
    identity_config = identity_report.get("config")
    if not isinstance(identity_config, Mapping) or _sha256_bytes(
        _canonical_json(identity_config)
    ) != identity_report.get("isolation_config_sha256"):
        raise FreshEvidenceCarrierDataError(
            "Identity isolation configuration hash mismatch"
        )
    provenance_exclusions = provenance.get("exclusion_set_sha256")
    if not isinstance(provenance_exclusions, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Selection provenance lacks exclusion-set commitments"
        )
    for name in _DEV_STRUCTURE_OVERLAP_FIELDS:
        expected_hash = identity_report.get(f"{name}_exclusion_set_sha256")
        if provenance_exclusions.get(name) != expected_hash:
            raise FreshEvidenceCarrierDataError(
                f"Selection provenance exclusion hash mismatch: {name}"
            )
    return artifacts, selection, identity_report


def _validate_phase_a_freeze(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PHASE_A_FREEZE_SCHEMA_VERSION:
        raise FreshEvidenceCarrierDataError(
            "Phase A freeze has an unsupported schema_version"
        )
    if payload.get("freeze_id") != PHASE_A_FREEZE_ID:
        raise FreshEvidenceCarrierDataError("Phase A freeze_id mismatch")
    if payload.get("phase") != "A" or payload.get("training_steps") != 0:
        raise FreshEvidenceCarrierDataError("Phase A freeze execution contract mismatch")
    if payload.get("parameter_updates") != 0:
        raise FreshEvidenceCarrierDataError("Phase A freeze permits no parameter updates")
    split = payload.get("splits", {}).get("fresh_dev", {})
    if not isinstance(split, Mapping) or (
        split.get("role") != "dev"
        or split.get("split") != "fresh_dev"
        or split.get("molecule_count") != ROLE_COUNTS["dev"][0]
        or split.get("sample_count") != ROLE_COUNTS["dev"][1]
        or split.get("authorized") is not True
    ):
        raise FreshEvidenceCarrierDataError("Phase A fresh_dev authorization mismatch")


def _validate_dev_isolation_report(
    payload: Mapping[str, Any],
    *,
    identity_report: Mapping[str, Any],
    molecule_manifest_sha256: str,
    sample_manifest_sha256: str,
    paired_manifest_sha256: str,
) -> None:
    if payload.get("schema_version") != ISOLATION_REPORT_SCHEMA_VERSION:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report has an unsupported schema_version"
        )
    if payload.get("status") != "passed" or payload.get("role") != "fresh_dev":
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report must declare role=fresh_dev and status=passed"
        )
    if payload.get("source_split") != "train":
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation source_split must be train"
        )
    overlap = payload.get("overlap_counts")
    if not isinstance(overlap, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report lacks overlap_counts"
        )
    if payload.get("producer") != "independent_isolation_auditor":
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report producer is not independent_isolation_auditor"
        )
    for group in _DEV_ISOLATION_EXPOSURE_SETS:
        values = overlap.get(group)
        if not isinstance(values, Mapping):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev isolation report lacks overlap_counts.{group}"
            )
        if set(values) != set(_DEV_ISOLATION_OVERLAP_FIELDS):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev isolation overlap fields differ for {group}"
            )
        for field in _DEV_ISOLATION_OVERLAP_FIELDS:
            count = _strict_int(
                values.get(field),
                label=f"isolation.overlap_counts.{group}.{field}",
            )
            if count != 0:
                raise FreshEvidenceCarrierDataError(
                    f"Fresh-dev isolation overlap is nonzero: {group}.{field}={count}"
                )
    hashes = payload.get("hashes")
    if not isinstance(hashes, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report lacks hashes"
        )
    if set(hashes) != set(_DEV_ISOLATION_HASH_FIELDS):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report hash fields differ from the freeze"
        )
    for name in _DEV_ISOLATION_HASH_FIELDS:
        _strict_sha256(
            hashes.get(name),
            label=f"isolation.hashes.{name}",
        )
    if hashes["fresh_dev_paired_manifest_sha256"] != paired_manifest_sha256:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report is not bound to the paired manifest"
        )
    if hashes["fresh_dev_molecule_manifest_sha256"] != molecule_manifest_sha256:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report is not bound to the molecule manifest"
        )
    if hashes["fresh_dev_sample_manifest_sha256"] != sample_manifest_sha256:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev isolation report is not bound to the sample manifest"
        )
    for name in (
        "exposure_manifest_sha256",
        "molecule_id_exclusion_set_sha256",
        "inchikey_full_exclusion_set_sha256",
        "scaffold_id_exclusion_set_sha256",
        "source_record_id_exclusion_set_sha256",
        "scaffold_id_algorithm_sha256",
        "source_record_normalization_algorithm_sha256",
        "style_id_algorithm_sha256",
    ):
        if hashes[name] != identity_report.get(name):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev isolation report differs from identity audit: {name}"
            )


def _validate_dev_commitment_evidence(
    commitment: Mapping[str, Any],
    isolation_report: Mapping[str, Any],
) -> None:
    hashes = isolation_report["hashes"]
    assert isinstance(hashes, Mapping)
    expected_exclusions = {
        "molecule_id_sha256": hashes["molecule_id_exclusion_set_sha256"],
        "inchikey_full_sha256": hashes["inchikey_full_exclusion_set_sha256"],
        "scaffold_id_sha256": hashes["scaffold_id_exclusion_set_sha256"],
        "source_record_id_sha256": hashes[
            "source_record_id_exclusion_set_sha256"
        ],
    }
    if commitment.get("exclusion_sets") != expected_exclusions:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev commitment exclusion-set hashes mismatch"
        )
    fonts = commitment.get("fonts")
    if not isinstance(fonts, list) or not fonts:
        raise FreshEvidenceCarrierDataError("Fresh-dev commitment fonts are invalid")
    seen_fonts: set[str] = set()
    for item in fonts:
        if not isinstance(item, Mapping):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment font entry is invalid"
            )
        digest = _strict_sha256(item.get("sha256"), label="commitment.font.sha256")
        if set(item) != {"family", "sha256"}:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment font entry fields are invalid"
            )
        if DEV_FONT_FAMILIES.get(digest) != item.get("family") or digest in seen_fonts:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment font allowlist mismatch"
            )
        seen_fonts.add(digest)
    if seen_fonts != set(DEV_FONT_FAMILIES):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev commitment font inventory is incomplete"
        )
    implementation = commitment.get("implementation")
    expected_implementation = {
        "isolation_auditor_sha256": hashes["isolation_algorithm_sha256"],
        "style_id_algorithm_sha256": hashes["style_id_algorithm_sha256"],
        "scaffold_id_algorithm_sha256": hashes["scaffold_id_algorithm_sha256"],
        "source_record_normalization_algorithm_sha256": hashes[
            "source_record_normalization_algorithm_sha256"
        ],
    }
    if not isinstance(implementation, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev commitment implementation is invalid"
        )
    if set(implementation) != set(_DEV_IMPLEMENTATION_HASH_FIELDS):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev commitment implementation fields differ from the protocol"
        )
    for name in _DEV_IMPLEMENTATION_HASH_FIELDS:
        _strict_sha256(
            implementation.get(name),
            label=f"commitment.implementation.{name}",
        )
    for name, expected in expected_implementation.items():
        if implementation.get(name) != expected:
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev implementation binding mismatch: {name}"
            )


def load_fresh_evidence_carrier_package(
    commitment_path: str | Path,
    *,
    expected_role: str | None = None,
    expected_commitment_sha256: str | None = None,
) -> FreshEvidenceCarrierPackage:
    """Load and fully validate a smoke or fresh-dev input commitment.

    Only the commitment itself and the exact manifest, validation report, and
    shards named by it are opened.  The directory is never enumerated.
    """

    commitment, _ = _assert_regular_without_symlinks(
        Path(commitment_path),
        label="commitment",
    )
    _reject_forbidden_reference(
        str(commitment.parent),
        label="resolved commitment parent",
    )
    commitment_bytes = commitment.read_bytes()
    commitment_sha256 = _sha256_bytes(commitment_bytes)
    if expected_commitment_sha256 is not None:
        expected_digest = _strict_sha256(
            expected_commitment_sha256,
            label="expected_commitment_sha256",
        )
        if commitment_sha256 != expected_digest:
            raise FreshEvidenceCarrierDataError("Commitment SHA256 mismatch")
    payload = _load_json_bytes(commitment_bytes, label="commitment")
    if payload.get("schema_version") != COMMITMENT_SCHEMA_VERSION:
        raise FreshEvidenceCarrierDataError("Unsupported commitment schema_version")
    role = payload.get("role")
    if role not in ALLOWED_ROLES:
        raise FreshEvidenceCarrierDataError("Commitment role must be smoke or dev")
    expected_molecules, expected_samples, eligible, _ = ROLE_COUNTS[role]
    if _strict_int(
        payload.get("sample_count"),
        label="commitment.sample_count",
    ) != expected_samples:
        raise FreshEvidenceCarrierDataError(
            f"Commitment sample_count is incorrect; role {role!r} requires {expected_samples}"
        )
    if _strict_int(
        payload.get("molecule_count"),
        label="commitment.molecule_count",
    ) != expected_molecules:
        raise FreshEvidenceCarrierDataError(
            f"Commitment molecule_count is incorrect; role {role!r} requires {expected_molecules}"
        )
    if expected_role is not None:
        if expected_role not in ALLOWED_ROLES:
            raise FreshEvidenceCarrierDataError("expected_role must be smoke or dev")
        if role != expected_role:
            raise FreshEvidenceCarrierDataError(
                f"Commitment role {role!r} differs from expected role {expected_role!r}"
            )
    _require_role_path(commitment.name, role=role, label="commitment")
    root = commitment.parent
    if role == "dev":
        if commitment.name != "commitment.dev.json" or root.name != "fresh_dev":
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment must be fresh_dev/commitment.dev.json"
            )
        required_static = {
            "split": "fresh_dev",
            "expected_total_members": expected_samples * len(MEMBER_SUFFIXES),
            "width": 768,
            "height": 768,
            "max_tokens": 4_096,
            "sdf_protocol": DEV_SDF_PROTOCOL,
            "render_profile": DEV_RENDER_PROFILE,
        }
        for name, expected in required_static.items():
            if payload.get(name) != expected:
                raise FreshEvidenceCarrierDataError(
                    f"Fresh-dev commitment static field {name!r} mismatch"
                )
        required_dynamic = {
            "freeze",
            "source",
            "exclusion_sets",
            "isolation_report",
            "selection",
            "fonts",
            "implementation",
            "molecule_manifest",
            "sample_manifest",
            "paired_manifest",
            "shards",
            "preflight_validation",
            "validation",
        }
        missing_dynamic = sorted(required_dynamic - set(payload))
        if missing_dynamic:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment lacks dynamic fields: "
                + ", ".join(missing_dynamic)
            )
        for name in (
            "independent_generalization",
            "independence_verified",
            "scaffold_disjoint",
            "style_disjoint",
        ):
            if payload.get(name) not in (None, False):
                raise FreshEvidenceCarrierDataError(
                    f"Fresh-dev commitment cannot self-assert {name}"
                )
        source = payload.get("source")
        if not isinstance(source, Mapping) or source.get("split") != "train":
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment source.split must be train"
            )
        _reject_forbidden_json_references(source, label="commitment.source")
        for name in (
            "exclusion_sets",
            "selection",
            "fonts",
            "implementation",
            "molecule_manifest",
            "sample_manifest",
        ):
            _reject_forbidden_json_references(
                payload[name],
                label=f"commitment.{name}",
            )
    if payload.get("fresh_audit_eligible") is not eligible:
        raise FreshEvidenceCarrierDataError(
            "Commitment fresh_audit_eligible disagrees with its role"
        )
    if tuple(payload.get("variants", ())) != VARIANTS:
        raise FreshEvidenceCarrierDataError("Commitment variants are not the four frozen layouts")
    if _strict_int(
        payload.get("members_per_sample"),
        label="commitment.members_per_sample",
    ) != len(MEMBER_SUFFIXES):
        raise FreshEvidenceCarrierDataError("Commitment requires exactly 12 members per sample")

    freeze_path: Path | None = None
    freeze_sha256: str | None = None
    isolation_path: Path | None = None
    isolation_sha256: str | None = None
    isolation_payload: Mapping[str, Any] | None = None
    supporting_artifacts: list[FreshEvidenceCarrierArtifact] = []
    selection_rows: list[_SelectionIdentity] = []
    identity_isolation_payload: Mapping[str, Any] | None = None
    selection_commitment_sha256: str | None = None
    if role == "dev":
        freeze_ref = payload["freeze"]
        if not isinstance(freeze_ref, Mapping):
            raise FreshEvidenceCarrierDataError("Fresh-dev commitment freeze must be an object")
        if freeze_ref.get("path") != "../phase_a_freeze.json":
            raise FreshEvidenceCarrierDataError("Fresh-dev freeze reference path mismatch")
        freeze_path, _ = _assert_regular_without_symlinks(
            root.parent / "phase_a_freeze.json",
            label="Phase A freeze",
        )
        freeze_sha256 = _strict_sha256(
            freeze_ref.get("sha256"),
            label="commitment.freeze.sha256",
        )
        if freeze_sha256 != PHASE_A_FREEZE_SHA256:
            raise FreshEvidenceCarrierDataError("Phase A freeze SHA256 is not registered")
        freeze_bytes = freeze_path.read_bytes()
        if _sha256_bytes(freeze_bytes) != freeze_sha256:
            raise FreshEvidenceCarrierDataError("Phase A freeze SHA256 mismatch")
        _validate_phase_a_freeze(
            _load_json_bytes(freeze_bytes, label="Phase A freeze")
        )
        (
            selection_artifacts,
            selection_rows,
            identity_isolation_payload,
        ) = _validate_selection_commitment(
            root,
            payload["selection"],
            freeze_sha256=freeze_sha256,
        )
        supporting_artifacts.extend(selection_artifacts)
        selection_commitment_sha256 = selection_artifacts[0].sha256

    manifest_ref = payload.get("paired_manifest")
    if not isinstance(manifest_ref, dict):
        raise FreshEvidenceCarrierDataError("Commitment lacks paired_manifest")
    manifest_relative, manifest_path, _ = _safe_committed_path(
        root,
        manifest_ref.get("path"),
        label="paired manifest",
    )
    _require_role_path(manifest_relative, role=role, label="paired manifest")
    manifest_sha256 = _strict_sha256(
        manifest_ref.get("sha256"),
        label="commitment.paired_manifest.sha256",
    )
    if _sha256_file(manifest_path) != manifest_sha256:
        raise FreshEvidenceCarrierDataError("Paired manifest SHA256 mismatch")

    if role == "dev":
        molecule_manifest_artifact = _verified_artifact_reference(
            root,
            payload["molecule_manifest"],
            label="molecule manifest",
        )
        sample_manifest_artifact = _verified_artifact_reference(
            root,
            payload["sample_manifest"],
            label="sample manifest",
        )
        preflight_artifact = _verified_artifact_reference(
            root,
            payload["preflight_validation"],
            label="preflight validation",
            preflight_report=True,
        )
        supporting_artifacts.extend(
            (
                molecule_manifest_artifact,
                sample_manifest_artifact,
                preflight_artifact,
            )
        )
        preflight_reference = payload["preflight_validation"]
        assert isinstance(preflight_reference, Mapping)
        assert selection_commitment_sha256 is not None
        _validate_preflight_report(
            _load_json_bytes(
                preflight_artifact.path.read_bytes(),
                label="preflight validation report",
            ),
            commitment_status=preflight_reference.get("status"),
            selection_commitment_sha256=selection_commitment_sha256,
        )
        isolation_ref = payload["isolation_report"]
        if not isinstance(isolation_ref, Mapping):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment isolation_report must be an object"
            )
        if isolation_ref.get("path") != "isolation_report.json":
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev isolation report path mismatch"
            )
        _, isolation_path, _ = _safe_committed_path(
            root,
            isolation_ref.get("path"),
            label="isolation report",
        )
        isolation_sha256 = _strict_sha256(
            isolation_ref.get("sha256"),
            label="commitment.isolation_report.sha256",
        )
        isolation_bytes = isolation_path.read_bytes()
        if _sha256_bytes(isolation_bytes) != isolation_sha256:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev isolation report SHA256 mismatch"
            )
        if isolation_ref.get("status") != "passed":
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev commitment isolation status is not passed"
            )
        isolation_payload = _load_json_bytes(
            isolation_bytes,
            label="fresh-dev isolation report",
        )
        _reject_forbidden_json_references(
            isolation_payload,
            label="fresh-dev isolation report",
        )
        assert identity_isolation_payload is not None
        _validate_dev_isolation_report(
            isolation_payload,
            identity_report=identity_isolation_payload,
            molecule_manifest_sha256=molecule_manifest_artifact.sha256,
            sample_manifest_sha256=sample_manifest_artifact.sha256,
            paired_manifest_sha256=manifest_sha256,
        )
        _validate_dev_commitment_evidence(payload, isolation_payload)

    validation_ref = payload.get("validation")
    if not isinstance(validation_ref, dict):
        raise FreshEvidenceCarrierDataError("Commitment lacks validation")
    _, validation_path, _ = _safe_validation_report_path(
        root,
        validation_ref.get("path"),
    )
    validation_sha256 = _strict_sha256(
        validation_ref.get("sha256"),
        label="commitment.validation.sha256",
    )
    validation_bytes = validation_path.read_bytes()
    if _sha256_bytes(validation_bytes) != validation_sha256:
        raise FreshEvidenceCarrierDataError("Validation report SHA256 mismatch")
    validation_payload = _load_json_bytes(validation_bytes, label="validation report")
    commitment_status = validation_ref.get("status")
    if not isinstance(commitment_status, str):
        raise FreshEvidenceCarrierDataError("Commitment validation status must be a string")
    _validate_validation_report(
        validation_payload,
        role=role,
        commitment_status=commitment_status,
    )

    shard_refs = payload.get("shards")
    if not isinstance(shard_refs, list) or not shard_refs:
        raise FreshEvidenceCarrierDataError("Commitment shards must be a nonempty list")
    shards: list[FreshEvidenceCarrierShard] = []
    seen_shard_paths: set[str] = set()
    for shard_index, shard_ref in enumerate(shard_refs):
        if not isinstance(shard_ref, dict):
            raise FreshEvidenceCarrierDataError(
                f"Commitment shard {shard_index} must be an object"
            )
        raw_shard_path = shard_ref.get("path")
        if "validation" in _normalized_reference(raw_shard_path):
            raise FreshEvidenceCarrierDataError(
                f"Validation shard is forbidden: {raw_shard_path}"
            )
        relative, shard_path, shard_stat = _safe_committed_path(
            root,
            raw_shard_path,
            label=f"shard {shard_index}",
        )
        _require_role_path(relative, role=role, label=f"shard {shard_index}")
        if "validation" in _normalized_reference(relative):
            raise FreshEvidenceCarrierDataError(f"Validation shard is forbidden: {relative}")
        if relative in seen_shard_paths:
            raise FreshEvidenceCarrierDataError(f"Duplicate committed shard: {relative}")
        seen_shard_paths.add(relative)
        declared_bytes = _strict_int(
            shard_ref.get("bytes"), label=f"shards[{shard_index}].bytes", minimum=1
        )
        if shard_stat.st_size != declared_bytes:
            raise FreshEvidenceCarrierDataError(f"Shard byte count mismatch: {relative}")
        declared_sha256 = _strict_sha256(
            shard_ref.get("sha256"),
            label=f"shards[{shard_index}].sha256",
        )
        if _sha256_file(shard_path) != declared_sha256:
            raise FreshEvidenceCarrierDataError(f"Shard SHA256 mismatch: {relative}")
        shard_samples = _strict_int(
            shard_ref.get("samples"),
            label=f"shards[{shard_index}].samples",
            minimum=1,
        )
        shard_members = _strict_int(
            shard_ref.get("members"),
            label=f"shards[{shard_index}].members",
            minimum=1,
        )
        if shard_members != shard_samples * len(MEMBER_SUFFIXES):
            raise FreshEvidenceCarrierDataError(
                f"Shard {relative} does not declare 12 members per sample"
            )
        shards.append(
            FreshEvidenceCarrierShard(
                relative_path=relative,
                path=shard_path,
                sha256=declared_sha256,
                byte_count=declared_bytes,
                sample_count=shard_samples,
                member_count=shard_members,
                device=shard_stat.st_dev,
                inode=shard_stat.st_ino,
            )
        )
    if sum(shard.sample_count for shard in shards) != expected_samples:
        raise FreshEvidenceCarrierDataError("Committed shard sample counts do not add up")
    if sum(shard.member_count for shard in shards) != expected_samples * len(MEMBER_SUFFIXES):
        raise FreshEvidenceCarrierDataError("Committed shard member counts do not add up")
    if role == "dev":
        assert isolation_sha256 is not None
        _validate_dev_validation_bindings(
            validation_payload,
            commitment=payload,
            manifest_sha256=manifest_sha256,
            isolation_sha256=isolation_sha256,
            supporting_artifacts=supporting_artifacts,
            shards=shards,
        )

    manifest_rows, _ = _load_manifest(manifest_path)
    rows = _validate_manifest_rows(
        manifest_rows,
        role=role,
        declared_shards=frozenset(seen_shard_paths),
    )
    if role == "dev":
        artifact_by_label = {
            artifact.label: artifact for artifact in supporting_artifacts
        }
        _validate_dev_projection_manifests(
            rows,
            molecule_manifest_path=artifact_by_label["molecule manifest"].path,
            sample_manifest_path=artifact_by_label["sample manifest"].path,
            selection_rows=selection_rows,
        )
    rows_by_shard: dict[str, list[_ManifestRow]] = defaultdict(list)
    for row in rows:
        rows_by_shard[row.shard].append(row)
    for shard in shards:
        shard_rows = rows_by_shard.get(shard.relative_path, [])
        if len(shard_rows) != shard.sample_count:
            raise FreshEvidenceCarrierDataError(
                f"Manifest sample count disagrees for shard {shard.relative_path}"
            )
        _validate_tar_members(shard, shard_rows)

    all_input_paths = [manifest_path, validation_path, *(shard.path for shard in shards)]
    if freeze_path is not None:
        all_input_paths.append(freeze_path)
    if isolation_path is not None:
        all_input_paths.append(isolation_path)
    all_input_paths.extend(artifact.path for artifact in supporting_artifacts)
    all_inputs = set(all_input_paths)
    if commitment in all_inputs or len(all_inputs) != len(all_input_paths):
        raise FreshEvidenceCarrierDataError("Committed input paths must be distinct")
    fingerprint_payload = {
        "schema_version": INPUT_FINGERPRINT_SCHEMA_VERSION,
        "role": role,
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
        "validation_sha256": validation_sha256,
        "shards": [
            {"path": shard.relative_path, "sha256": shard.sha256}
            for shard in shards
        ],
        "sample_keys": [row.sample_key for row in rows],
    }
    if freeze_sha256 is not None:
        fingerprint_payload["freeze_sha256"] = freeze_sha256
    if isolation_sha256 is not None:
        fingerprint_payload["isolation_sha256"] = isolation_sha256
    if supporting_artifacts:
        fingerprint_payload["supporting_artifacts"] = [
            {
                "label": artifact.label,
                "path": artifact.relative_path,
                "sha256": artifact.sha256,
            }
            for artifact in supporting_artifacts
        ]
    package_fingerprint = _sha256_bytes(_canonical_json(fingerprint_payload))
    return FreshEvidenceCarrierPackage(
        role=role,
        fresh_audit_eligible=eligible,
        root=root,
        commitment_path=commitment,
        commitment_sha256=commitment_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        validation_path=validation_path,
        validation_sha256=validation_sha256,
        freeze_path=freeze_path,
        freeze_sha256=freeze_sha256,
        isolation_path=isolation_path,
        isolation_sha256=isolation_sha256,
        independence_verified=(role == "dev" and isolation_payload is not None),
        supporting_artifacts=tuple(supporting_artifacts),
        shards=tuple(shards),
        identities=tuple(row.identity for row in rows),
        input_fingerprint=package_fingerprint,
        _rows=rows,
        _selections=tuple(selection_rows),
    )


def _assert_unchanged_shard(shard: FreshEvidenceCarrierShard) -> None:
    _, current = _assert_regular_without_symlinks(shard.path, label="committed shard")
    if (
        current.st_dev != shard.device
        or current.st_ino != shard.inode
        or current.st_size != shard.byte_count
    ):
        raise FreshEvidenceCarrierDataError(
            f"Committed shard changed after validation: {shard.relative_path}"
        )


def _extract_member(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise FreshEvidenceCarrierDataError(f"Missing tar member: {name}") from error
    if not member.isfile():
        raise FreshEvidenceCarrierDataError(f"Tar member is not regular: {name}")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FreshEvidenceCarrierDataError(f"Cannot read tar member: {name}")
    return extracted.read()


def _decode_png(value: bytes, *, label: str, expected_mode: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(value)) as opened:
            if opened.format != "PNG":
                raise FreshEvidenceCarrierDataError(f"{label} is not a PNG image")
            opened.load()
            if opened.size != (768, 768):
                raise FreshEvidenceCarrierDataError(
                    f"{label} must be exactly 768x768, got {opened.size}"
                )
            if opened.mode != expected_mode:
                raise FreshEvidenceCarrierDataError(
                    f"{label} must use mode {expected_mode}, got {opened.mode}"
                )
            return opened.copy()
    except (OSError, SyntaxError) as error:
        raise FreshEvidenceCarrierDataError(f"Invalid PNG in {label}: {error}") from error


def _validate_sidecar_identity(
    payload: Mapping[str, Any],
    identity: FreshEvidenceCarrierIdentity,
    *,
    label: str,
    require_sample_key: bool,
    require_variant: bool,
) -> None:
    if require_sample_key and payload.get("sample_key") != identity.sample_key:
        raise FreshEvidenceCarrierDataError(f"{label} sample_key mismatch")
    if payload.get("molecule_id") != identity.molecule_id:
        raise FreshEvidenceCarrierDataError(f"{label} molecule_id mismatch")
    if require_variant and payload.get("variant") != identity.variant:
        raise FreshEvidenceCarrierDataError(f"{label} variant mismatch")


def _sample_fingerprint(
    package: FreshEvidenceCarrierPackage,
    row: _ManifestRow,
    member_sha256: Mapping[str, str],
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "schema_version": INPUT_FINGERPRINT_SCHEMA_VERSION,
                "package_input_fingerprint": package.input_fingerprint,
                "manifest_row_index": row.index,
                "identity": row.identity.as_dict(),
                "member_sha256": dict(sorted(member_sha256.items())),
            }
        )
    )


@lru_cache(maxsize=1)
def _chemmol_validator() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "Data/sdf/schemas/chemmol-0.1.schema.json"
    )
    if _sha256_file(path) != CHEMMOL_SCHEMA_SHA256:
        raise FreshEvidenceCarrierDataError("ChemMol schema SHA256 mismatch")
    payload = _load_json_bytes(path.read_bytes(), label="ChemMol schema")
    return fastjsonschema.compile(payload)


@lru_cache(maxsize=1)
def _visual_svg_tokenizer() -> VisualSVGTokenizerV01:
    return VisualSVGTokenizerV01()


def _enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def _parse_single_sdf(value: bytes, *, sample_key: str) -> tuple[Chem.Mol, str]:
    try:
        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(value),
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
        molecules = list(supplier)
        if len(molecules) != 1 or molecules[0] is None:
            raise ValueError("not exactly one sanitized record")
        molecule = molecules[0]
        Chem.SanitizeMol(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        inchikey = Chem.MolToInchiKey(molecule)
    except Exception as error:
        raise FreshEvidenceCarrierDataError(
            f"Invalid single-record SDF for {sample_key}: {error}"
        ) from error
    if not inchikey:
        raise FreshEvidenceCarrierDataError(
            f"SDF lacks an InChIKey for {sample_key}"
        )
    return molecule, inchikey


def _validate_chemmol_sdf_contract(
    chemmol: Mapping[str, Any],
    sdf: Chem.Mol,
    *,
    identity: FreshEvidenceCarrierIdentity,
    inchikey: str,
) -> None:
    try:
        _chemmol_validator()(dict(chemmol))
    except Exception as error:
        raise FreshEvidenceCarrierDataError(
            f"ChemMol schema validation failed for {identity.sample_key}: {error}"
        ) from error
    if (
        chemmol.get("molecule_id") != identity.molecule_id
        or chemmol.get("inchikey") != identity.inchikey
        or inchikey != identity.inchikey
    ):
        raise FreshEvidenceCarrierDataError("ChemMol/SDF identity mismatch")
    atoms = chemmol.get("atoms")
    bonds = chemmol.get("bonds")
    if not isinstance(atoms, list) or not isinstance(bonds, list):
        raise FreshEvidenceCarrierDataError("ChemMol atom/bond arrays are invalid")
    if len(atoms) != sdf.GetNumAtoms() or len(bonds) != sdf.GetNumBonds():
        raise FreshEvidenceCarrierDataError("ChemMol/SDF atom or bond count mismatch")
    atom_ids = [int(item["id"]) for item in atoms]
    source_indices = [int(item["source_atom_index"]) for item in atoms]
    expected_atom_ids = list(range(sdf.GetNumAtoms()))
    if atom_ids != expected_atom_ids or sorted(source_indices) != expected_atom_ids:
        raise FreshEvidenceCarrierDataError(
            "ChemMol atom IDs or source_atom_index permutation is invalid"
        )
    for atom_data in atoms:
        atom_id = int(atom_data["id"])
        atom = sdf.GetAtomWithIdx(atom_id)
        actual = {
            "element": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "isotope": atom.GetIsotope(),
            "aromatic": atom.GetIsAromatic(),
            "chiral_tag": _enum_name(atom.GetChiralTag()),
            "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
            "explicit_h_count": atom.GetNumExplicitHs(),
            "implicit_h_count": atom.GetNumImplicitHs(),
            "no_implicit": atom.GetNoImplicit(),
            "radical_electrons": atom.GetNumRadicalElectrons(),
        }
        expected = {name: atom_data[name] for name in actual}
        if actual != expected:
            raise FreshEvidenceCarrierDataError(
                f"ChemMol/SDF atom mismatch at atom {atom_id}"
            )
    bond_ids = [int(item["id"]) for item in bonds]
    if bond_ids != list(range(sdf.GetNumBonds())):
        raise FreshEvidenceCarrierDataError("ChemMol bond IDs are not canonical")
    for bond_data in bonds:
        bond_id = int(bond_data["id"])
        bond = sdf.GetBondWithIdx(bond_id)
        stereo = _enum_name(bond.GetStereo())
        actual = {
            "begin_atom": bond.GetBeginAtomIdx(),
            "end_atom": bond.GetEndAtomIdx(),
            "type": _enum_name(bond.GetBondType()),
            "aromatic": bond.GetIsAromatic(),
            "conjugated": bond.GetIsConjugated(),
            "stereo": stereo,
            "stereo_atoms": list(bond.GetStereoAtoms()),
            "ez": "E" if stereo == "STEREOE" else "Z" if stereo == "STEREOZ" else None,
        }
        expected = {name: bond_data[name] for name in actual}
        if actual != expected:
            raise FreshEvidenceCarrierDataError(
                f"ChemMol/SDF bond mismatch at bond {bond_id}"
            )
    sdf_smiles = Chem.MolToSmiles(
        sdf,
        canonical=True,
        isomericSmiles=True,
    )
    if chemmol.get("canonical_isomeric_smiles") != sdf_smiles:
        raise FreshEvidenceCarrierDataError(
            "ChemMol canonical SMILES/source SDF mismatch"
        )
    try:
        rebuilt = mol_from_chemmol(dict(chemmol))
        rebuilt_key = Chem.MolToInchiKey(rebuilt)
    except Exception as error:
        raise FreshEvidenceCarrierDataError(
            f"ChemMol reconstruction failed: {error}"
        ) from error
    if rebuilt_key != identity.inchikey:
        raise FreshEvidenceCarrierDataError("ChemMol reconstructed InChIKey mismatch")


def _validate_visual_contract(
    primitives: Mapping[str, Any],
    visualsvg: str,
    visualsvg_v02: Mapping[str, Any],
    *,
    identity: FreshEvidenceCarrierIdentity,
    meta: Mapping[str, Any],
    sdf: Chem.Mol,
) -> None:
    try:
        validate_visual_primitive_references(dict(primitives))
        expected_v01 = serialize_visual_svg_v0_1(dict(primitives))
        expected_v02 = visual_svg_v0_2_to_dict(
            visual_primitives_v0_1_to_visual_svg_v0_2(dict(primitives))
        )
        token_count = len(_visual_svg_tokenizer().encode(visualsvg))
    except Exception as error:
        raise FreshEvidenceCarrierDataError(
            f"Visual SVG validation failed for {identity.sample_key}: {error}"
        ) from error
    if primitives.get("sample_key") != identity.sample_key:
        raise FreshEvidenceCarrierDataError("Primitives sample_key mismatch")
    if primitives.get("molecule_id") != identity.molecule_id:
        raise FreshEvidenceCarrierDataError("Primitives molecule_id mismatch")
    if primitives.get("width") != 768 or primitives.get("height") != 768:
        raise FreshEvidenceCarrierDataError("Primitives canvas mismatch")
    if expected_v01 != visualsvg:
        raise FreshEvidenceCarrierDataError("Visual SVG v0.1 differs from primitives")
    if expected_v02 != dict(visualsvg_v02):
        raise FreshEvidenceCarrierDataError("Visual SVG v0.2 differs from primitives")
    if token_count > 4_096 or meta.get("sequence_tokens") != token_count:
        raise FreshEvidenceCarrierDataError("Visual SVG token count mismatch")
    groups = primitives.get("bond_groups")
    if not isinstance(groups, list) or len(groups) != sdf.GetNumBonds():
        raise FreshEvidenceCarrierDataError("Primitive bond group count mismatch")
    for group in groups:
        bond_id = int(group["bond_id"])
        if not 0 <= bond_id < sdf.GetNumBonds():
            raise FreshEvidenceCarrierDataError("Primitive bond group ID out of range")
        bond = sdf.GetBondWithIdx(bond_id)
        endpoints = {int(group["begin_atom"]), int(group["end_atom"])}
        if endpoints != {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}:
            raise FreshEvidenceCarrierDataError(
                f"Primitive bond endpoints mismatch at bond {bond_id}"
            )
        if group.get("chemical_type") != _enum_name(bond.GetBondType()):
            raise FreshEvidenceCarrierDataError(
                f"Primitive bond type mismatch at bond {bond_id}"
            )


def _validate_graph_sidecar(
    payload: Mapping[str, Any],
    *,
    label: str,
    sdf: Chem.Mol,
) -> None:
    atoms = payload.get("atoms")
    bonds = payload.get("bonds")
    if not isinstance(atoms, list) or len(atoms) != sdf.GetNumAtoms():
        raise FreshEvidenceCarrierDataError(f"{label} atom count mismatch")
    if not isinstance(bonds, list) or len(bonds) != sdf.GetNumBonds():
        raise FreshEvidenceCarrierDataError(f"{label} bond count mismatch")
    if sorted(int(item["atom_id"]) for item in atoms) != list(
        range(sdf.GetNumAtoms())
    ):
        raise FreshEvidenceCarrierDataError(f"{label} atom IDs mismatch")
    for atom_data in atoms:
        atom_id = int(atom_data["atom_id"])
        atom = sdf.GetAtomWithIdx(atom_id)
        expected = {
            "element": atom.GetSymbol(),
            "formal_charge": atom.GetFormalCharge(),
            "aromatic": atom.GetIsAromatic(),
            "implicit_h_count": atom.GetNumImplicitHs(),
        }
        if any(atom_data.get(name) != value for name, value in expected.items()):
            raise FreshEvidenceCarrierDataError(
                f"{label} atom chemistry mismatch at atom {atom_id}"
            )
    if sorted(int(item["bond_id"]) for item in bonds) != list(
        range(sdf.GetNumBonds())
    ):
        raise FreshEvidenceCarrierDataError(f"{label} bond IDs mismatch")
    for bond_data in bonds:
        bond_id = int(bond_data["bond_id"])
        bond = sdf.GetBondWithIdx(bond_id)
        endpoints = {int(bond_data["begin_atom"]), int(bond_data["end_atom"])}
        if endpoints != {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}:
            raise FreshEvidenceCarrierDataError(
                f"{label} bond endpoints mismatch at bond {bond_id}"
            )
        if bond_data.get("chemical_type") != _enum_name(bond.GetBondType()):
            raise FreshEvidenceCarrierDataError(
                f"{label} bond type mismatch at bond {bond_id}"
            )


def _validate_dev_sidecars(
    identity: FreshEvidenceCarrierIdentity,
    *,
    selection: _SelectionIdentity,
    layers: Mapping[str, Any],
    meta: Mapping[str, Any],
    render: Mapping[str, Any],
) -> None:
    source = meta.get("source")
    if not isinstance(source, Mapping):
        raise FreshEvidenceCarrierDataError("Fresh-dev meta lacks source metadata")
    declared_source_splits = {
        value
        for value in (source.get("split"), source.get("source_split"))
        if value is not None
    }
    if declared_source_splits != {"train"}:
        raise FreshEvidenceCarrierDataError("Fresh-dev meta source split must be train")
    if source.get("source_record_id") != identity.source_record_id:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev meta source_record_id mismatch"
        )
    expected_source = {
        "dataset": "webdataset_full_v0/train",
        "source_record_id": selection.source_record_id,
        "source_shard": selection.source_shard,
        "source_split": selection.source_split,
        "structure_ref": selection.source_structure,
        "source_sdf_sha256": selection.source_sdf_sha256,
    }
    for name, expected_value in expected_source.items():
        if source.get(name) != expected_value:
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev meta source commitment mismatch: {name}"
            )
    expected = {
        "role": "dev",
        "split": "fresh_dev",
        "scaffold_id": identity.scaffold_id,
        "source_record_id": identity.source_record_id,
        "style_id": identity.style_id,
        "font_sha": identity.font_sha,
        "render_profile": identity.render_profile,
    }
    for payload, label in ((layers, "layers"), (render, "render"), (meta, "meta")):
        for name, value in expected.items():
            if payload.get(name) != value:
                raise FreshEvidenceCarrierDataError(
                    f"Fresh-dev {label} {name} mismatch"
                )
    style = meta.get("style")
    if not isinstance(style, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev meta lacks canonical style payload"
        )
    computed_style_id = _style_id_from_payload(style)
    if computed_style_id != identity.style_id:
        raise FreshEvidenceCarrierDataError("Fresh-dev style_id hash mismatch")
    if style.get("layout_variant") != identity.variant:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev canonical style layout_variant mismatch"
        )
    if layers.get("style") != style or render.get("style") != style:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev sidecar canonical style payloads differ"
        )
    layers_source = layers.get("source")
    if not isinstance(layers_source, Mapping):
        raise FreshEvidenceCarrierDataError("Fresh-dev layers lacks source metadata")
    if layers_source.get("dataset") != "webdataset_full_v0":
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev layers source dataset root mismatch"
        )
    for name in ("source_sample_key", "source_shard"):
        if layers_source.get(name) != source.get(name):
            raise FreshEvidenceCarrierDataError(
                f"Fresh-dev layers/meta source {name} mismatch"
            )
    parameters = render.get("render_parameters")
    if not isinstance(parameters, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev render lacks render_parameters"
        )
    declared_font_hashes = {
        value
        for value in (
            parameters.get("font_file_sha256"),
            parameters.get("font_sha"),
        )
        if value is not None
    }
    if declared_font_hashes != {identity.font_sha}:
        raise FreshEvidenceCarrierDataError("Fresh-dev render font hash mismatch")
    if parameters.get("render_profile") != identity.render_profile:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev render profile mismatch"
        )
    if parameters.get("style_id") != identity.style_id:
        raise FreshEvidenceCarrierDataError("Fresh-dev render style_id mismatch")
    if render.get("style_id") != identity.style_id:
        raise FreshEvidenceCarrierDataError("Fresh-dev render style_id mismatch")
    layer_parameters = layers.get("render_parameters")
    if not isinstance(layer_parameters, Mapping):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev layers lacks render_parameters"
        )
    if (
        layer_parameters.get("style_id") != identity.style_id
        or layer_parameters.get("font_sha") != identity.font_sha
        or layer_parameters.get("render_profile") != identity.render_profile
    ):
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev layers render parameter identity mismatch"
        )


def _decode_sample(
    package: FreshEvidenceCarrierPackage,
    row: _ManifestRow,
    members: Mapping[str, bytes],
    *,
    selection: _SelectionIdentity | None,
) -> dict[str, Any]:
    actual_hashes = {
        suffix: _sha256_bytes(members[suffix]) for suffix in MEMBER_SUFFIXES
    }
    for suffix, expected in row.hashes:
        if actual_hashes[suffix] != expected:
            raise FreshEvidenceCarrierDataError(
                f"Manifest hash mismatch for {row.sample_key}{suffix}"
            )

    identity = row.identity
    primitives = _load_json_bytes(
        members[".primitives.json"], label=f"{identity.sample_key}.primitives.json"
    )
    visualsvg_v02 = _load_json_bytes(
        members[".visualsvg-v02.json"], label=f"{identity.sample_key}.visualsvg-v02.json"
    )
    layers = _load_json_bytes(
        members[".layers.json"], label=f"{identity.sample_key}.layers.json"
    )
    render = _load_json_bytes(
        members[".render.json"], label=f"{identity.sample_key}.render.json"
    )
    chemmol = _load_json_bytes(
        members[".chemmol.json"], label=f"{identity.sample_key}.chemmol.json"
    )
    meta = _load_json_bytes(
        members[".meta.json"], label=f"{identity.sample_key}.meta.json"
    )
    for payload, label in (
        (primitives, "primitives"),
        (visualsvg_v02, "visualsvg_v02"),
        (layers, "layers"),
        (render, "render"),
        (chemmol, "chemmol"),
        (meta, "meta"),
    ):
        _reject_forbidden_json_references(
            payload,
            label=f"{identity.sample_key}.{label}",
        )
    _validate_sidecar_identity(
        primitives,
        identity,
        label="primitives",
        require_sample_key=True,
        require_variant=False,
    )
    _validate_sidecar_identity(
        layers,
        identity,
        label="layers",
        require_sample_key=True,
        require_variant=True,
    )
    _validate_sidecar_identity(
        render,
        identity,
        label="render",
        require_sample_key=False,
        require_variant=True,
    )
    _validate_sidecar_identity(
        chemmol,
        identity,
        label="chemmol",
        require_sample_key=False,
        require_variant=False,
    )
    _validate_sidecar_identity(
        meta,
        identity,
        label="meta",
        require_sample_key=True,
        require_variant=True,
    )
    if meta.get("split") != identity.split:
        raise FreshEvidenceCarrierDataError("meta split mismatch")
    if meta.get("fresh_audit_eligible") is not package.fresh_audit_eligible:
        raise FreshEvidenceCarrierDataError("meta fresh_audit_eligible mismatch")
    if meta.get("sdf_inchikey") != identity.inchikey:
        raise FreshEvidenceCarrierDataError("meta InChIKey mismatch")
    if chemmol.get("inchikey") != identity.inchikey:
        raise FreshEvidenceCarrierDataError("ChemMol InChIKey mismatch")
    expected_layer_splits = (
        {"train", "smoke_train"} if package.role == "smoke" else {"fresh_dev"}
    )
    if layers.get("split") not in expected_layer_splits:
        raise FreshEvidenceCarrierDataError("layers split mismatch")
    if package.role == "dev":
        if selection is None or selection.molecule_id != identity.molecule_id:
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev sample lacks its committed selection identity"
            )
        _validate_dev_sidecars(
            identity,
            selection=selection,
            layers=layers,
            meta=meta,
            render=render,
        )

    meta_hashes = meta.get("sha256")
    expected_meta_names = {suffix.removeprefix(".") for suffix in HASHED_SUFFIXES}
    if not isinstance(meta_hashes, dict) or set(meta_hashes) != expected_meta_names:
        raise FreshEvidenceCarrierDataError("meta sha256 fields are incomplete")
    for suffix in HASHED_SUFFIXES:
        name = suffix.removeprefix(".")
        expected = _strict_sha256(
            meta_hashes.get(name),
            label=f"meta.sha256.{name}",
        )
        if actual_hashes[suffix] != expected:
            raise FreshEvidenceCarrierDataError(
                f"Meta hash mismatch for {identity.sample_key}{suffix}"
            )
    try:
        visualsvg = members[".visualsvg.txt"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise FreshEvidenceCarrierDataError("Visual SVG v0.1 is not UTF-8") from error
    if package.role == "dev":
        assert selection is not None
        if (
            len(members[".sdf"]) != selection.source_sdf_bytes
            or actual_hashes[".sdf"] != selection.source_sdf_sha256
            or actual_hashes[".chemmol.json"]
            != selection.source_chemmol_sha256
        ):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev source SDF/ChemMol byte commitment mismatch"
            )
        atoms = chemmol.get("atoms")
        bonds = chemmol.get("bonds")
        if (
            not isinstance(atoms, list)
            or not isinstance(bonds, list)
            or len(atoms) != selection.source_atom_count
            or len(bonds) != selection.source_bond_count
        ):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev source atom/bond count commitment mismatch"
            )
        sdf, sdf_inchikey = _parse_single_sdf(
            members[".sdf"],
            sample_key=identity.sample_key,
        )
        _validate_chemmol_sdf_contract(
            chemmol,
            sdf,
            identity=identity,
            inchikey=sdf_inchikey,
        )
        _validate_visual_contract(
            primitives,
            visualsvg,
            visualsvg_v02,
            identity=identity,
            meta=meta,
            sdf=sdf,
        )
        _validate_graph_sidecar(layers, label="layers", sdf=sdf)
        source = meta["source"]
        if (
            meta.get("sdf_protocol") != DEV_SDF_PROTOCOL
            or source.get("source_sdf_sha256") != actual_hashes[".sdf"]
        ):
            raise FreshEvidenceCarrierDataError(
                "Fresh-dev SDF provenance mismatch"
            )

    image = _decode_png(
        members[".png"],
        label=f"{identity.sample_key}.png",
        expected_mode="RGB",
    )
    text_layer = _decode_png(
        members[".text.png"],
        label=f"{identity.sample_key}.text.png",
        expected_mode="L",
    )
    bond_layer = _decode_png(
        members[".bond.png"],
        label=f"{identity.sample_key}.bond.png",
        expected_mode="L",
    )
    semantic_layer = _decode_png(
        members[".semantic.png"],
        label=f"{identity.sample_key}.semantic.png",
        expected_mode="L",
    )
    fingerprint = _sample_fingerprint(package, row, actual_hashes)
    return {
        "key": identity.sample_key,
        "image": image,
        "text_layer": text_layer,
        "bond_layer": bond_layer,
        "semantic_layer": semantic_layer,
        "primitives": primitives,
        "visualsvg": visualsvg,
        "visualsvg_v02": visualsvg_v02,
        "layers": layers,
        "render": render,
        "chemmol": chemmol,
        "meta": meta,
        "sdf_bytes": members[".sdf"],
        "identity": identity,
        "member_sha256": actual_hashes,
        "input_fingerprint": fingerprint,
    }


def _iter_package_samples(
    package: FreshEvidenceCarrierPackage,
) -> Iterator[dict[str, Any]]:
    selections = {item.molecule_id: item for item in package._selections}
    if package.role == "dev" and len(selections) != package.molecule_count:
        raise FreshEvidenceCarrierDataError(
            "Fresh-dev selection identity inventory changed after loading"
        )
    with ExitStack() as stack:
        archives: dict[str, tarfile.TarFile] = {}
        for shard in package.shards:
            _assert_unchanged_shard(shard)
            archives[shard.relative_path] = stack.enter_context(
                tarfile.open(shard.path, "r:*")
            )
        for row in package._rows:
            archive = archives[row.shard]
            members = {
                suffix: _extract_member(archive, row.sample_key + suffix)
                for suffix in MEMBER_SUFFIXES
            }
            yield _decode_sample(
                package,
                row,
                members,
                selection=selections.get(row.identity.molecule_id),
            )


def iter_fresh_evidence_carrier_samples(
    commitment_path: str | Path,
    *,
    expected_role: str | None = None,
    expected_commitment_sha256: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Convenience iterator over one committed smoke or fresh-dev package."""

    package = load_fresh_evidence_carrier_package(
        commitment_path,
        expected_role=expected_role,
        expected_commitment_sha256=expected_commitment_sha256,
    )
    yield from package.iter_samples()


__all__ = [
    "ALLOWED_ROLES",
    "COMMITMENT_SCHEMA_VERSION",
    "FreshEvidenceCarrierDataError",
    "FreshEvidenceCarrierIdentity",
    "FreshEvidenceCarrierPackage",
    "FreshEvidenceCarrierShard",
    "HASHED_SUFFIXES",
    "INPUT_FINGERPRINT_SCHEMA_VERSION",
    "MEMBER_SUFFIXES",
    "VARIANTS",
    "iter_fresh_evidence_carrier_samples",
    "load_fresh_evidence_carrier_package",
]
