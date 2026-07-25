#!/usr/bin/env python3
"""Independently validate the full fresh-dev WebDataset package.

The validator does not import the full builder.  It validates every declared
tar member and chemistry/Visual-SVG sidecar, binds all five frozen selection
artifacts, the independent isolation report, implementation hashes, and the
complete STIX/P052 font inventory.  ``--allow-partial`` exercises the same
sample contract for a deliberately small builder output but can never produce
a final passed report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
from typing import Any, Mapping, Sequence

import fastjsonschema

try:
    from . import validate_visual_svg_layers_c_fresh_dev_preflight as pv
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
except ImportError:
    import validate_visual_svg_layers_c_fresh_dev_preflight as pv
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-validation-0.1"
PARTIAL_SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-partial-validation-0.1"
BUILD_SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-build-0.1"
PARTIAL_BUILD_SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-partial-build-0.1"
MOLECULE_COUNT = 2_000
SAMPLE_COUNT = 8_000
MEMBER_COUNT = SAMPLE_COUNT * len(pv.MEMBER_SUFFIXES)
MOLECULE_MANIFEST = "molecule_manifest.dev.csv"
SAMPLE_MANIFEST = "sample_manifest.dev.csv"
PAIRED_MANIFEST = "paired_manifest.dev.csv"
BUILD_MANIFEST = "build.dev.json"
ISOLATION_REPORT = "isolation_report.json"
FONT_INVENTORY = [
    {
        "family": "STIX",
        "sha256": "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45",
    },
    {
        "family": "P052",
        "sha256": "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e92778a78d149524eea9",
    },
]
IMPLEMENTATION_FIELDS = {
    "builder_sha256",
    "validator_sha256",
    "isolation_auditor_sha256",
    "style_id_algorithm_sha256",
    "scaffold_id_algorithm_sha256",
    "source_record_normalization_algorithm_sha256",
}
ISOLATION_HASH_FIELDS = {
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
}
OVERLAP_FIELDS = {
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
    "style_id",
    "font_sha",
    "render_profile",
}
SAMPLE_PROJECTION_FIELDS = {
    "sample_key",
    "molecule_id",
    "split",
    "variant",
    "style_id",
    "font_sha",
    "render_profile",
    "image_sha256",
    "sdf_sha256",
}
SAMPLE_DIAGNOSTIC_FIELDS = {"group_diagnostics_mode"}
PREFLIGHT_MOLECULE_COUNT = 16
FULL_RASTER_AUDIT_STRIDE = 100
GROUP_STRUCTURAL_PROTOCOL = "visual-svg-group-structural-gate-0.1"
MOLECULE_PROJECTION_FIELDS = {
    "molecule_id",
    "inchikey",
    "scaffold_id",
    "source_record_id",
    "sdf_sha256",
}


class FreshDevValidationError(ValueError):
    """Raised when a full-package validation precondition is invalid."""


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _regular_file(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise FreshDevValidationError(f"missing_file:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise FreshDevValidationError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise FreshDevValidationError(f"not_regular_file:{absolute}")
    return absolute


def _read_json(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshDevValidationError(f"invalid_json:{path}") from error
    if not isinstance(value, dict):
        raise FreshDevValidationError(f"json_root_not_object:{path}")
    return value


def _safe_relative(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise FreshDevValidationError(f"unsafe_relative_path:{label}:{relative}")
    candidate = _regular_file(root / relative)
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise FreshDevValidationError(f"path_escapes_root:{label}:{relative}")
    return candidate


def _read_csv(
    path: Path,
    required: set[str],
    *,
    label: str,
) -> list[dict[str, str]]:
    with _regular_file(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise FreshDevValidationError(
                f"{label}_missing_fields:{sorted(missing)}"
            )
        rows = list(reader)
    for index, row in enumerate(rows, 2):
        if None in row or any(row.get(field, "") == "" for field in required):
            raise FreshDevValidationError(f"{label}_invalid_row:{index}")
    return rows


def _selection_artifacts(
    fresh_root: Path,
    selection_commitment: Mapping[str, Any],
    selection_commitment_path: Path,
) -> dict[str, str]:
    result = {
        "selection_commitment_sha256": _file_sha256(selection_commitment_path)
    }
    references = (
        ("selection_manifest_sha256", "selection_manifest"),
        ("replacement_manifest_sha256", "fresh_replacement_manifest"),
        ("identity_isolation_report_sha256", "identity_isolation_report"),
        ("selection_provenance_sha256", "selection_provenance"),
    )
    for output_name, reference_name in references:
        reference = selection_commitment.get(reference_name)
        if not isinstance(reference, Mapping):
            raise FreshDevValidationError(
                f"selection_reference_missing:{reference_name}"
            )
        path = _safe_relative(
            fresh_root,
            reference.get("path", ""),
            label=reference_name,
        )
        digest = _file_sha256(path)
        if reference.get("sha256") != digest:
            raise FreshDevValidationError(
                f"selection_reference_sha256:{reference_name}"
            )
        result[output_name] = digest
    return result


def _preflight_members(
    preflight_root: Path,
    *,
    selection_sha256: str,
    selection_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, dict[str, bytes]], str]:
    report_path = _regular_file(preflight_root / "preflight_validation.json")
    report = _read_json(report_path)
    expected = {
        "schema_version": "visual-svg-layers-c-fresh-dev-preflight-validation-0.1",
        "status": "PREFLIGHT-PASS",
        "role": "dev",
        "split": "fresh_dev",
        "molecules": 16,
        "samples": 64,
        "checked_samples": 64,
        "training_steps": 0,
        "parameter_updates": 0,
        "model_run": False,
        "threshold_or_checkpoint_selection": False,
        "sealed_rows_read": False,
        "selection_commitment_sha256": selection_sha256,
        "errors": [],
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise FreshDevValidationError(f"preflight_report:{name}")
    commitment = _read_json(preflight_root / "preflight_input_commitment.json")
    molecule_rows = _read_csv(
        preflight_root / "molecules.csv",
        set(pv.MOLECULE_FIELDS),
        label="preflight_molecules",
    )
    if len(molecule_rows) != 16 or [
        int(row["selection_index"]) for row in molecule_rows
    ] != list(range(16)):
        raise FreshDevValidationError("preflight_molecule_selection_indices")
    for expected, actual in zip(selection_rows[:16], molecule_rows):
        if any(actual[field] != expected[field] for field in pv.SELECTION_FIELDS):
            raise FreshDevValidationError("preflight_molecule_selection_projection")
    shard_ref = commitment.get("shard")
    if not isinstance(shard_ref, Mapping):
        raise FreshDevValidationError("preflight_shard_reference")
    shard = _safe_relative(
        preflight_root,
        shard_ref.get("path", ""),
        label="preflight_shard",
    )
    digest = _file_sha256(shard)
    if shard_ref.get("sha256") != digest or report.get("shard_sha256") != digest:
        raise FreshDevValidationError("preflight_shard_sha256")
    result: dict[str, dict[str, bytes]] = defaultdict(dict)
    suffixes = sorted(pv.MEMBER_SUFFIXES, key=len, reverse=True)
    with tarfile.open(shard, "r") as archive:
        for member in archive:
            suffix = next(
                (value for value in suffixes if member.name.endswith(value)),
                None,
            )
            if suffix is None:
                raise FreshDevValidationError(
                    f"preflight_member_suffix:{member.name}"
                )
            key = member.name[: -len(suffix)]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FreshDevValidationError(
                    f"preflight_member_unreadable:{member.name}"
                )
            result[key][suffix] = extracted.read()
    expected_keys = {
        f"fresh_dev_{index:06d}_{variant}"
        for index in range(16)
        for variant in pv.VARIANTS
    }
    if set(result) != expected_keys or any(
        set(members) != set(pv.MEMBER_SUFFIXES) for members in result.values()
    ):
        raise FreshDevValidationError("preflight_member_closure")
    return dict(result), _file_sha256(report_path)


def _validate_isolation(
    fresh_root: Path,
    *,
    molecule_sha256: str,
    sample_sha256: str,
    paired_sha256: str,
    identity_report: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    path = _regular_file(fresh_root / ISOLATION_REPORT)
    report = _read_json(path)
    if (
        report.get("schema_version")
        != "visual-svg-layers-c-isolation-report-0.1"
        or report.get("producer") != "independent_isolation_auditor"
        or report.get("role") != "fresh_dev"
        or report.get("status") != "passed"
        or report.get("source_split") != "train"
    ):
        raise FreshDevValidationError("isolation_report_contract")
    overlap = report.get("overlap_counts")
    if not isinstance(overlap, Mapping):
        raise FreshDevValidationError("isolation_overlap_missing")
    for exposure in ("frozen_b_p0_train_exposure", "smoke_train_32"):
        values = overlap.get(exposure)
        if not isinstance(values, Mapping) or set(values) != OVERLAP_FIELDS:
            raise FreshDevValidationError(f"isolation_overlap_fields:{exposure}")
        if any(type(value) is not int or value != 0 for value in values.values()):
            raise FreshDevValidationError(f"isolation_overlap_nonzero:{exposure}")
    hashes = report.get("hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != ISOLATION_HASH_FIELDS:
        raise FreshDevValidationError("isolation_hash_fields")
    expected_manifests = {
        "fresh_dev_molecule_manifest_sha256": molecule_sha256,
        "fresh_dev_sample_manifest_sha256": sample_sha256,
        "fresh_dev_paired_manifest_sha256": paired_sha256,
    }
    for name, value in expected_manifests.items():
        if hashes.get(name) != value:
            raise FreshDevValidationError(f"isolation_manifest_binding:{name}")
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
        if hashes.get(name) != identity_report.get(name):
            raise FreshDevValidationError(f"isolation_identity_binding:{name}")
    config = report.get("config")
    if not isinstance(config, Mapping) or hashlib.sha256(
        _canonical_json(dict(config))
    ).hexdigest() != hashes.get("isolation_config_sha256"):
        raise FreshDevValidationError("isolation_config_sha256")
    auditor = (
        Path(__file__).resolve().parent
        / "audit_visual_svg_layers_c_fresh_dev_isolation.py"
    )
    if hashes.get("isolation_algorithm_sha256") != _file_sha256(auditor):
        raise FreshDevValidationError("isolation_algorithm_sha256")
    return report, _file_sha256(path)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    fresh_root = Path(args.fresh_root).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    freeze_path = Path(args.freeze).expanduser().resolve()
    preflight_root = Path(args.preflight_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    build_path = _regular_file(fresh_root / BUILD_MANIFEST)
    build = _read_json(build_path)
    full_build = build.get("schema_version") == BUILD_SCHEMA_VERSION
    if not full_build and build.get("schema_version") != PARTIAL_BUILD_SCHEMA_VERSION:
        raise FreshDevValidationError("build_schema_version")
    if not full_build and not args.allow_partial:
        raise FreshDevValidationError("partial_build_requires_allow_partial")
    expected_molecules = MOLECULE_COUNT if full_build else int(build.get("molecules", 0))
    expected_samples = expected_molecules * len(pv.VARIANTS)
    if (
        build.get("role") != "dev"
        or build.get("split") != "fresh_dev"
        or build.get("molecules") != expected_molecules
        or build.get("samples") != expected_samples
        or build.get("members") != expected_samples * len(pv.MEMBER_SUFFIXES)
        or build.get("training_steps") != 0
        or build.get("parameter_updates") != 0
        or build.get("model_run") is not False
        or build.get("errors") != []
    ):
        raise FreshDevValidationError("build_static_contract")
    pv._validate_frozen_source(source)
    selection_path = _regular_file(fresh_root / "selection.commitment.json")
    selection_rows, selection_commitment = pv._load_selection(
        selection_path,
        freeze_path,
    )
    selection_sha256 = _file_sha256(selection_path)
    selection_hashes = _selection_artifacts(
        fresh_root,
        selection_commitment,
        selection_path,
    )
    if build.get("selection_artifact_sha256") != selection_hashes:
        raise FreshDevValidationError("build_selection_artifact_bindings")
    canonical_preflight, preflight_sha256 = _preflight_members(
        preflight_root,
        selection_sha256=selection_sha256,
        selection_rows=selection_rows,
    )

    manifests = build.get("manifests")
    if not isinstance(manifests, Mapping):
        raise FreshDevValidationError("build_manifests_missing")
    manifest_specs = {
        "molecule": (
            MOLECULE_MANIFEST,
            pv.MOLECULE_FIELDS | MOLECULE_PROJECTION_FIELDS,
        ),
        "sample": (
            SAMPLE_MANIFEST,
            pv.SAMPLE_FIELDS | SAMPLE_PROJECTION_FIELDS | SAMPLE_DIAGNOSTIC_FIELDS,
        ),
        "paired": (PAIRED_MANIFEST, pv.PAIRED_FIELDS),
    }
    manifest_paths = {}
    manifest_hashes = {}
    manifest_rows = {}
    for name, (expected_name, required) in manifest_specs.items():
        reference = manifests.get(name)
        if not isinstance(reference, Mapping) or reference.get("path") != expected_name:
            raise FreshDevValidationError(f"build_manifest_reference:{name}")
        path = _regular_file(fresh_root / expected_name)
        digest = _file_sha256(path)
        if reference.get("sha256") != digest:
            raise FreshDevValidationError(f"build_manifest_sha256:{name}")
        rows = _read_csv(path, set(required), label=f"{name}_manifest")
        manifest_paths[name] = path
        manifest_hashes[name] = digest
        manifest_rows[name] = rows
    molecule_rows = manifest_rows["molecule"]
    sample_rows = manifest_rows["sample"]
    paired_rows = manifest_rows["paired"]
    if (
        len(molecule_rows) != expected_molecules
        or len(sample_rows) != expected_samples
        or len(paired_rows) != expected_samples
    ):
        raise FreshDevValidationError("manifest_count_contract")
    selected = selection_rows[:expected_molecules]
    for selected_row, molecule in zip(selected, molecule_rows):
        if any(
            molecule[field] != selected_row[field]
            for field in pv.SELECTION_FIELDS
        ):
            raise FreshDevValidationError("molecule_selection_projection")
        if molecule["inchikey"] != selected_row["inchikey_full"]:
            raise FreshDevValidationError("molecule_inchikey_projection")
    if [row["sample_key"] for row in sample_rows] != sorted(
        row["sample_key"] for row in sample_rows
    ) or [row["sample_key"] for row in sample_rows] != [
        row["sample_key"] for row in paired_rows
    ]:
        raise FreshDevValidationError("sample_manifest_order")
    sample_by_key = {row["sample_key"]: row for row in sample_rows}
    paired_by_key = {row["sample_key"]: row for row in paired_rows}
    molecule_by_id = {row["molecule_id"]: row for row in molecule_rows}
    if (
        len(sample_by_key) != expected_samples
        or len(paired_by_key) != expected_samples
        or len(molecule_by_id) != expected_molecules
    ):
        raise FreshDevValidationError("manifest_duplicate_identity")
    for sample, paired in zip(sample_rows, paired_rows):
        for field in SAMPLE_PROJECTION_FIELDS:
            if sample[field] != paired[field]:
                raise FreshDevValidationError(
                    f"sample_paired_projection:{sample['sample_key']}:{field}"
                )

    expected_full_raster_molecules = sum(
        index < PREFLIGHT_MOLECULE_COUNT
        or index % FULL_RASTER_AUDIT_STRIDE == 0
        for index in range(expected_molecules)
    )
    expected_group_diagnostics = {
        "full_raster_rule": (
            f"selection_index<{PREFLIGHT_MOLECULE_COUNT} or "
            f"selection_index%{FULL_RASTER_AUDIT_STRIDE}==0"
        ),
        "full_raster_molecules": expected_full_raster_molecules,
        "full_raster_samples": expected_full_raster_molecules * len(pv.VARIANTS),
        "structural_protocol": GROUP_STRUCTURAL_PROTOCOL,
        "structural_molecules": expected_molecules - expected_full_raster_molecules,
        "structural_samples": (
            expected_molecules - expected_full_raster_molecules
        )
        * len(pv.VARIANTS),
    }
    actual_group_diagnostic_counts = {"full_raster": 0, "structural": 0}
    for row in sample_rows:
        selection_index = int(row["selection_index"])
        expected_mode = (
            "full_raster"
            if selection_index < PREFLIGHT_MOLECULE_COUNT
            or selection_index % FULL_RASTER_AUDIT_STRIDE == 0
            else "structural"
        )
        actual_mode = row["group_diagnostics_mode"]
        if actual_mode != expected_mode:
            raise FreshDevValidationError(
                "sample_group_diagnostics_schedule:"
                f"{row['sample_key']}:{actual_mode}:{expected_mode}"
            )
        actual_group_diagnostic_counts[actual_mode] += 1
    if (
        actual_group_diagnostic_counts["full_raster"]
        != expected_group_diagnostics["full_raster_samples"]
        or actual_group_diagnostic_counts["structural"]
        != expected_group_diagnostics["structural_samples"]
        or build.get("group_diagnostics") != expected_group_diagnostics
    ):
        raise FreshDevValidationError("group_diagnostics_contract")

    identity_report = _read_json(fresh_root / "identity_isolation_report.json")
    isolation_report = None
    isolation_sha256 = None
    if full_build:
        isolation_report, isolation_sha256 = _validate_isolation(
            fresh_root,
            molecule_sha256=manifest_hashes["molecule"],
            sample_sha256=manifest_hashes["sample"],
            paired_sha256=manifest_hashes["paired"],
            identity_report=identity_report,
        )

    shard_refs = build.get("shards")
    if not isinstance(shard_refs, list) or not shard_refs:
        raise FreshDevValidationError("build_shards_missing")
    declared_shards = {}
    for reference in shard_refs:
        if not isinstance(reference, Mapping):
            raise FreshDevValidationError("build_shard_reference")
        relative = str(reference.get("path", ""))
        if not relative.startswith("shards/dev-") or not relative.endswith(".tar"):
            raise FreshDevValidationError(f"build_shard_path:{relative}")
        path = _safe_relative(fresh_root, relative, label="dev_shard")
        digest = _file_sha256(path)
        rows = [row for row in paired_rows if row["shard"] == relative]
        if (
            reference.get("sha256") != digest
            or reference.get("bytes") != path.stat().st_size
            or reference.get("samples") != len(rows)
            or reference.get("members") != len(rows) * len(pv.MEMBER_SUFFIXES)
        ):
            raise FreshDevValidationError(f"build_shard_binding:{relative}")
        declared_shards[relative] = (path, reference, rows)
    if sum(len(value[2]) for value in declared_shards.values()) != expected_samples:
        raise FreshDevValidationError("shard_sample_sum")

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
                Path(__file__).resolve().parent
                / "schemas/chemmol-0.1.schema.json"
            ).read_text(encoding="utf-8")
        )
    )
    tokenizer = VisualSVGTokenizerV01()
    errors: list[str] = []
    checked_samples = 0
    checked_members = 0
    preflight_members_equal = 0
    results = {}
    for relative in sorted(declared_shards):
        shard, reference, shard_rows = declared_shards[relative]
        expected_names = [
            row["sample_key"] + suffix
            for row in sorted(shard_rows, key=lambda item: item["sample_key"])
            for suffix in pv.MEMBER_SUFFIXES
        ]
        try:
            with tarfile.open(shard, "r") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                if names != expected_names or len(names) != len(set(names)):
                    raise FreshDevValidationError(
                        f"tar_member_order_or_names:{relative}"
                    )
                by_name = {}
                for member in members:
                    pv._tar_metadata(member)
                    by_name[member.name] = member
                for row in sorted(shard_rows, key=lambda item: item["sample_key"]):
                    key = row["sample_key"]
                    payload = {}
                    for suffix in pv.MEMBER_SUFFIXES:
                        extracted = archive.extractfile(by_name[key + suffix])
                        if extracted is None:
                            raise FreshDevValidationError(
                                f"unreadable_member:{key}{suffix}"
                            )
                        payload[suffix] = extracted.read()
                        checked_members += 1
                    canonical = canonical_preflight.get(key)
                    if canonical is not None:
                        for suffix in pv.MEMBER_SUFFIXES:
                            if payload[suffix] != canonical[suffix]:
                                raise FreshDevValidationError(
                                    f"preflight_byte_mismatch:{key}{suffix}"
                                )
                            preflight_members_equal += 1
                    results[key] = pv._validate_sample(
                        payload,
                        sample_by_key[key],
                        paired_by_key[key],
                        molecule_by_id[row["molecule_id"]],
                        source,
                        primitive_validator,
                        chemmol_validator,
                        tokenizer,
                    )
                    checked_samples += 1
        except Exception as error:
            errors.append(f"shard:{relative}:{type(error).__name__}:{error}")
            break

    expected_preflight_members = min(expected_molecules, 16) * 4 * len(
        pv.MEMBER_SUFFIXES
    )
    if preflight_members_equal != expected_preflight_members:
        errors.append(
            f"preflight_member_count:{preflight_members_equal}:{expected_preflight_members}"
        )
    if checked_samples != expected_samples:
        errors.append(f"checked_samples:{checked_samples}:{expected_samples}")
    if checked_members != expected_samples * len(pv.MEMBER_SUFFIXES):
        errors.append(
            f"checked_members:{checked_members}:{expected_samples * len(pv.MEMBER_SUFFIXES)}"
        )

    for font_sha, descriptor in pv.FONT_ALLOWLIST.items():
        try:
            if _file_sha256(_regular_file(Path(descriptor["file"]))) != font_sha:
                errors.append(f"font_file_sha256:{font_sha}")
        except Exception as error:
            errors.append(f"font_file:{type(error).__name__}:{error}")
    if build.get("font_inventory") != FONT_INVENTORY:
        errors.append("build_font_inventory")
    builder_path = (
        Path(__file__).resolve().parent
        / "build_visual_svg_layers_c_fresh_dev.py"
    )
    build_implementation = build.get("implementation_sha256")
    if not isinstance(build_implementation, Mapping) or (
        build_implementation.get("builder_sha256") != _file_sha256(builder_path)
    ):
        errors.append("builder_implementation_sha256")

    implementation = {
        "builder_sha256": (
            build_implementation.get("builder_sha256", "0" * 64)
            if isinstance(build_implementation, Mapping)
            else "0" * 64
        ),
        "validator_sha256": _file_sha256(Path(__file__).resolve()),
        "isolation_auditor_sha256": (
            isolation_report["hashes"]["isolation_algorithm_sha256"]
            if isolation_report is not None
            else "0" * 64
        ),
        "style_id_algorithm_sha256": identity_report[
            "style_id_algorithm_sha256"
        ],
        "scaffold_id_algorithm_sha256": identity_report[
            "scaffold_id_algorithm_sha256"
        ],
        "source_record_normalization_algorithm_sha256": identity_report[
            "source_record_normalization_algorithm_sha256"
        ],
    }
    if set(implementation) != IMPLEMENTATION_FIELDS:
        errors.append("implementation_field_closure")
    artifact_sha256 = {
        "paired_manifest_sha256": manifest_hashes["paired"],
        "molecule_manifest_sha256": manifest_hashes["molecule"],
        "sample_manifest_sha256": manifest_hashes["sample"],
        "preflight_validation_sha256": preflight_sha256,
        "isolation_report_sha256": isolation_sha256 or "0" * 64,
        **selection_hashes,
    }
    full_pass = full_build and not errors
    report = {
        "schema_version": SCHEMA_VERSION if full_build else PARTIAL_SCHEMA_VERSION,
        "status": "passed" if full_pass else "partial_pass" if not errors else "failed",
        "finalizable": full_pass,
        "fresh_audit_eligible": full_build,
        "molecules": expected_molecules,
        "samples": expected_samples,
        "members": expected_samples * len(pv.MEMBER_SUFFIXES),
        "checked_samples": checked_samples,
        "checked_members": checked_members,
        "structure_validated_samples": len(results),
        "preflight_byte_identity_members": preflight_members_equal,
        "group_diagnostics": expected_group_diagnostics,
        "artifact_sha256": artifact_sha256,
        "implementation_sha256": implementation,
        "font_inventory": FONT_INVENTORY,
        "shards": [
            {
                "path": reference["path"],
                "sha256": reference["sha256"],
                "samples": reference["samples"],
                "members": reference["members"],
            }
            for reference in shard_refs
        ],
        "max_sequence_tokens": max(
            (int(value["tokens"]) for value in results.values()),
            default=0,
        ),
        "double_bond_groups_checked": sum(
            int(value["double_groups"]) for value in results.values()
        ),
        "triple_bond_groups_checked": sum(
            int(value["triple_groups"]) for value in results.values()
        ),
        "training_steps": 0,
        "parameter_updates": 0,
        "model_run": False,
        "sealed_rows_read": False,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Validation report exists; pass --overwrite: {output}")
    output.write_bytes(_canonical_json(report, newline=True))
    return {"output": str(output), **report}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    version_root = root / "visual_svg_layers_c_v0_1"
    fresh_root = version_root / "fresh_dev"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", default=str(fresh_root))
    parser.add_argument("--source", default=str(root / "webdataset_full_v0"))
    parser.add_argument("--freeze", default=str(version_root / "phase_a_freeze.json"))
    parser.add_argument(
        "--preflight-root",
        default=str(fresh_root / "preflight_64"),
    )
    parser.add_argument("--output", default=str(fresh_root / "validation.json"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = validate(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "FONT_INVENTORY",
    "IMPLEMENTATION_FIELDS",
    "SCHEMA_VERSION",
    "FreshDevValidationError",
    "build_parser",
    "main",
    "validate",
]
