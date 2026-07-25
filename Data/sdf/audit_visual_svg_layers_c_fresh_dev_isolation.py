#!/usr/bin/env python3
"""Independently audit final fresh-dev identity and rendering isolation.

The auditor is separate from the full builder.  It binds the final three
manifests, rechecks their identity projection against the frozen selection,
and measures identity/style/font/render-profile overlap against the two
already exposed training sets.  It does not inspect any held-out split and
does not run a model.
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


SCHEMA_VERSION = "visual-svg-layers-c-isolation-report-0.1"
FREEZE_ID = "visual-svg-layers-c-v0.1-phase-a-20260722"
MOLECULE_COUNT = 2_000
SAMPLE_COUNT = 8_000
IDENTITY_FIELDS = (
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
)
STYLE_FIELDS = ("style_id", "font_sha", "render_profile")
OVERLAP_FIELDS = (*IDENTITY_FIELDS, *STYLE_FIELDS)
FORBIDDEN_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "reserve",
    "mr_tadf",
    "mr-tadf",
    "mrtadf",
)
PILOT_PIPELINE_PROFILE = "chemvision-layers-pilot-train-v0.1"
SMOKE_PIPELINE_PROFILE = "visual-svg-layers-c-smoke-train-v0.1"
PILOT_TRAIN_SHARD_SHA256 = {
    "train-000000.tar": "6e5a21d14cd0d1521de44f67eaea111e4c0272d7b7652cb1e88c13f82443b042",
    "train-000001.tar": "9ca92e15d260d4372135177990e1f9fdeca771d8e78cbcd7ba2c735128035aea",
    "train-000002.tar": "5d9eba5abe8277368b68ec9bdd6fc3c4abeaaf339acd0874e00a815d95311aa7",
}
SMOKE_COMMITMENT_SHA256 = (
    "e8c0bf1fbd792b60836ab879ee7a8b2dbee230f0a6a426bf710fee1e0ad6825a"
)
SMOKE_TRAIN_SHARD_SHA256 = {
    "smoke-train-000000.tar": (
        "ed2461b635cb68787da3195b38f6001e7a0990d0376b83615da0e4b571e30cbc"
    ),
}
HASH_FIELDS = (
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
MOLECULE_REQUIRED = {
    "molecule_id",
    "inchikey",
    "scaffold_id",
    "source_record_id",
    "sdf_sha256",
}
SAMPLE_REQUIRED = {
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
PAIRED_REQUIRED = SAMPLE_REQUIRED | {"inchikey", "shard"}
PAIRED_REQUIRED.add("layers_sha256")
SELECTION_REQUIRED = {
    "selection_index",
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
}


class FreshDevIsolationError(ValueError):
    """Raised when final fresh-dev isolation cannot be established."""


def _assert_allowed_path(path: Path, *, label: str) -> None:
    normalized = str(path).lower().replace("\\", "/")
    marker = next((value for value in FORBIDDEN_MARKERS if value in normalized), None)
    if marker is not None:
        raise FreshDevIsolationError(f"forbidden_path:{label}:{marker}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
            raise FreshDevIsolationError(f"missing_file:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise FreshDevIsolationError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise FreshDevIsolationError(f"not_regular_file:{absolute}")
    return absolute


def _read_json(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshDevIsolationError(f"invalid_json:{path}") from error
    if not isinstance(value, dict):
        raise FreshDevIsolationError(f"json_root_not_object:{path}")
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
        if not required.issubset(fields):
            raise FreshDevIsolationError(
                f"{label}_missing_fields:{sorted(required - fields)}"
            )
        rows = list(reader)
    for index, row in enumerate(rows, 2):
        if None in row or any(row.get(field, "") == "" for field in required):
            raise FreshDevIsolationError(f"{label}_invalid_row:{index}")
    return rows


def _identity_sets(rows: Sequence[Mapping[str, str]]) -> dict[str, set[str]]:
    return {field: {row[field] for row in rows} for field in IDENTITY_FIELDS}


def _fixed_decimal(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreshDevIsolationError(f"style_numeric_field:{label}")
    number = float(value)
    if not number > 0.0:
        raise FreshDevIsolationError(f"style_numeric_range:{label}")
    return f"{number:.6f}"


def _style_signature_from_layers(
    layers: Mapping[str, Any],
    *,
    pipeline_profile: str,
    expected_split: str,
    context: str,
) -> tuple[str, str, str]:
    """Build one comparable signature; missing legacy fields fail closed."""

    if layers.get("split") != expected_split:
        raise FreshDevIsolationError(f"style_split:{context}")
    parameters = layers.get("render_parameters")
    if not isinstance(parameters, Mapping):
        raise FreshDevIsolationError(f"style_render_parameters:{context}")
    variant = layers.get("variant")
    layout = parameters.get("layout")
    if not isinstance(variant, str) or not variant or layout != variant:
        raise FreshDevIsolationError(f"style_layout:{context}")
    font_sha = parameters.get("font_file_sha256")
    if not isinstance(font_sha, str) or len(font_sha) != 64:
        raise FreshDevIsolationError(f"style_font_sha:{context}")
    color_mode = parameters.get("color_mode")
    if not isinstance(color_mode, str) or not color_mode:
        raise FreshDevIsolationError(f"style_color_mode:{context}")
    if not isinstance(pipeline_profile, str) or not pipeline_profile:
        raise FreshDevIsolationError(f"style_pipeline_profile:{context}")
    descriptor = {
        "schema": "render-style-signature-0.1",
        "canvas": {
            "width": _fixed_decimal(layers.get("width"), label=f"{context}.width"),
            "height": _fixed_decimal(
                layers.get("height"), label=f"{context}.height"
            ),
        },
        "layout": layout,
        "font_sha256": font_sha,
        "font_size": _fixed_decimal(
            parameters.get("font_size"), label=f"{context}.font_size"
        ),
        "bond_line_width": _fixed_decimal(
            parameters.get("bond_line_width"),
            label=f"{context}.bond_line_width",
        ),
        "color_mode": color_mode,
    }
    signature = "render-style-v0.1:" + _sha256_bytes(
        _canonical_json(descriptor)
    )
    return signature, font_sha, pipeline_profile


def _fresh_style_sets(
    fresh_root: Path,
    paired_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    wanted_by_shard: dict[str, dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in paired_rows:
        shard = row["shard"]
        if not shard.startswith("shards/dev-") or not shard.endswith(".tar"):
            raise FreshDevIsolationError("fresh_style_shard_path")
        wanted_by_shard[shard][row["sample_key"] + ".layers.json"] = row
    style_sets = {field: set() for field in STYLE_FIELDS}
    render_signatures: set[str] = set()
    checked = 0
    shard_records = []
    for relative, wanted in sorted(wanted_by_shard.items()):
        shard = _regular_file(fresh_root / relative)
        found = set()
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                row = wanted.get(member.name)
                if row is None:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FreshDevIsolationError("fresh_layer_unreadable")
                data = extracted.read()
                if _sha256_bytes(data) != row["layers_sha256"]:
                    raise FreshDevIsolationError("fresh_layer_sha256")
                layers = json.loads(data)
                if not isinstance(layers, dict) or (
                    layers.get("sample_key") != row["sample_key"]
                    or layers.get("molecule_id") != row["molecule_id"]
                    or layers.get("style_id") != row["style_id"]
                    or layers.get("font_sha") != row["font_sha"]
                    or layers.get("render_profile") != row["render_profile"]
                ):
                    raise FreshDevIsolationError("fresh_layer_identity")
                signature, font_sha, profile = _style_signature_from_layers(
                    layers,
                    pipeline_profile=row["render_profile"],
                    expected_split="fresh_dev",
                    context=row["sample_key"],
                )
                if font_sha != row["font_sha"]:
                    raise FreshDevIsolationError("fresh_layer_font_binding")
                style_sets["style_id"].add(row["style_id"])
                style_sets["font_sha"].add(font_sha)
                style_sets["render_profile"].add(profile)
                render_signatures.add(signature)
                found.add(member.name)
                checked += 1
        if found != set(wanted):
            raise FreshDevIsolationError(f"fresh_layer_inventory:{relative}")
        shard_records.append(
            {
                "path": relative,
                "sha256": _file_sha256(shard),
                "layers_checked": len(found),
            }
        )
    if checked != SAMPLE_COUNT:
        raise FreshDevIsolationError("fresh_layer_count")
    return style_sets, {
        "layers_checked": checked,
        "shards": shard_records,
        "style_signature_protocol": "render-style-signature-0.1",
        "render_style_signature_count": len(render_signatures),
        "render_style_signature_set_sha256": _sha256_bytes(
            "\n".join(sorted(render_signatures)).encode("utf-8")
        ),
        "render_style_signature_role": "inventory_only",
    }


def _overlap_counts(
    fresh_identity: Mapping[str, set[str]],
    identity_counts: Mapping[str, int],
    fresh_style: Mapping[str, set[str]],
    exposure_style: Mapping[str, set[str]],
    *,
    legacy_style_id_missing: bool,
) -> dict[str, int]:
    result = {field: int(identity_counts[field]) for field in IDENTITY_FIELDS}
    font_overlap = len(fresh_style["font_sha"] & exposure_style["font_sha"])
    if legacy_style_id_missing:
        if exposure_style["style_id"]:
            raise FreshDevIsolationError("legacy_style_id_set_must_be_empty")
        if font_overlap != 0:
            raise FreshDevIsolationError(
                "style_id_zero_inference_requires_font_sha_disjoint"
            )
        style_id_overlap = 0
    else:
        style_id_overlap = len(
            fresh_style["style_id"] & exposure_style["style_id"]
        )
    result.update(
        {
            "style_id": style_id_overlap,
            "font_sha": font_overlap,
            "render_profile": len(
                fresh_style["render_profile"]
                & exposure_style["render_profile"]
            ),
        }
    )
    if set(result) != set(OVERLAP_FIELDS):
        raise FreshDevIsolationError("overlap_field_closure")
    del fresh_identity
    return result


def _pilot_style_sets(pilot: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    shards = sorted((pilot / "shards").glob("train-*.tar"))
    if [path.name for path in shards] != sorted(PILOT_TRAIN_SHARD_SHA256):
        raise FreshDevIsolationError("pilot_train_shard_inventory")
    style_sets = {field: set() for field in STYLE_FIELDS}
    render_signatures: set[str] = set()
    checked = 0
    shard_records = []
    for shard in shards:
        shard = _regular_file(shard)
        digest = _file_sha256(shard)
        if digest != PILOT_TRAIN_SHARD_SHA256[shard.name]:
            raise FreshDevIsolationError(f"pilot_train_shard_sha256:{shard.name}")
        shard_layers = 0
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.name.endswith(".layers.json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FreshDevIsolationError("pilot_layer_unreadable")
                layers = json.load(extracted)
                if not isinstance(layers, dict) or layers.get("split") != "train":
                    raise FreshDevIsolationError("pilot_layer_split")
                source = layers.get("source")
                if not isinstance(source, Mapping) or not str(
                    source.get("source_shard", "")
                ).startswith("shards/train-"):
                    raise FreshDevIsolationError("pilot_layer_source_split")
                signature, font_sha, profile = _style_signature_from_layers(
                    layers,
                    pipeline_profile=PILOT_PIPELINE_PROFILE,
                    expected_split="train",
                    context=member.name,
                )
                style_sets["font_sha"].add(font_sha)
                style_sets["render_profile"].add(profile)
                render_signatures.add(signature)
                checked += 1
                shard_layers += 1
        shard_records.append(
            {
                "path": f"shards/{shard.name}",
                "sha256": digest,
                "layers_checked": shard_layers,
            }
        )
    if checked != 9_000:
        raise FreshDevIsolationError("pilot_train_layer_count")
    return style_sets, {
        "train_layers_checked": checked,
        "train_shards": shard_records,
        "legacy_schema": "chemvision-layers-v0.1",
        "pipeline_profile": PILOT_PIPELINE_PROFILE,
        "style_signature_protocol": "render-style-signature-0.1",
        "render_style_signature_count": len(render_signatures),
        "render_style_signature_set_sha256": _sha256_bytes(
            "\n".join(sorted(render_signatures)).encode("utf-8")
        ),
        "render_style_signature_role": "inventory_only",
        "missing_required_field_policy": "fail_closed",
    }


def _smoke_style_sets(smoke: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    commitment_path = _regular_file(smoke / "commitment.smoke.json")
    if _file_sha256(commitment_path) != SMOKE_COMMITMENT_SHA256:
        raise FreshDevIsolationError("smoke_commitment_sha256")
    commitment = _read_json(commitment_path)
    if (
        commitment.get("role") != "smoke"
        or commitment.get("molecule_count") != 8
        or commitment.get("sample_count") != 32
    ):
        raise FreshDevIsolationError("smoke_commitment_contract")
    shard_refs = commitment.get("shards")
    if not isinstance(shard_refs, list) or len(shard_refs) != 1:
        raise FreshDevIsolationError("smoke_shard_inventory")
    style_sets = {field: set() for field in STYLE_FIELDS}
    render_signatures: set[str] = set()
    checked = 0
    shard_records = []
    for reference in shard_refs:
        if not isinstance(reference, Mapping):
            raise FreshDevIsolationError("smoke_shard_reference")
        relative = Path(str(reference.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise FreshDevIsolationError("smoke_shard_path")
        if (
            relative.as_posix() != "shards/smoke-train-000000.tar"
            or reference.get("samples") != 32
            or reference.get("members") != 32 * 12
        ):
            raise FreshDevIsolationError("smoke_shard_contract")
        shard = _regular_file(smoke / relative)
        digest = _file_sha256(shard)
        if (
            reference.get("sha256") != digest
            or digest != SMOKE_TRAIN_SHARD_SHA256[shard.name]
        ):
            raise FreshDevIsolationError("smoke_shard_sha256")
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.name.endswith(".layers.json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FreshDevIsolationError("smoke_layer_unreadable")
                layers = json.load(extracted)
                if not isinstance(layers, dict):
                    raise FreshDevIsolationError("smoke_layer_root")
                signature, font_sha, profile = _style_signature_from_layers(
                    layers,
                    pipeline_profile=SMOKE_PIPELINE_PROFILE,
                    expected_split=str(layers.get("split")),
                    context=member.name,
                )
                if layers.get("split") not in {"train", "smoke_train"}:
                    raise FreshDevIsolationError("smoke_layer_split")
                style_sets["font_sha"].add(font_sha)
                style_sets["render_profile"].add(profile)
                render_signatures.add(signature)
                checked += 1
        shard_records.append(
            {
                "path": relative.as_posix(),
                "sha256": digest,
            }
        )
    if checked != 32:
        raise FreshDevIsolationError("smoke_layer_count")
    return style_sets, {
        "commitment_sha256": _file_sha256(commitment_path),
        "layers_checked": checked,
        "shards": shard_records,
        "legacy_schema": "visual-svg-layers-c-smoke-v0.1",
        "pipeline_profile": SMOKE_PIPELINE_PROFILE,
        "style_signature_protocol": "render-style-signature-0.1",
        "render_style_signature_count": len(render_signatures),
        "render_style_signature_set_sha256": _sha256_bytes(
            "\n".join(sorted(render_signatures)).encode("utf-8")
        ),
        "render_style_signature_role": "inventory_only",
        "missing_required_field_policy": "fail_closed",
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    fresh_root = Path(args.fresh_root).expanduser().resolve()
    pilot = Path(args.pilot).expanduser().resolve()
    smoke = Path(args.smoke).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    for path, label in (
        (fresh_root, "fresh_root"),
        (pilot, "pilot"),
        (smoke, "smoke"),
        (output, "output"),
    ):
        _assert_allowed_path(path, label=label)
    molecule_path = _regular_file(fresh_root / "molecule_manifest.dev.csv")
    sample_path = _regular_file(fresh_root / "sample_manifest.dev.csv")
    paired_path = _regular_file(fresh_root / "paired_manifest.dev.csv")
    selection_path = _regular_file(fresh_root / "selection.committed.csv")
    identity_report_path = _regular_file(
        fresh_root / "identity_isolation_report.json"
    )

    molecule_rows = _read_csv(
        molecule_path,
        MOLECULE_REQUIRED,
        label="fresh_molecules",
    )
    sample_rows = _read_csv(sample_path, SAMPLE_REQUIRED, label="fresh_samples")
    paired_rows = _read_csv(paired_path, PAIRED_REQUIRED, label="fresh_paired")
    selection_rows = _read_csv(
        selection_path,
        SELECTION_REQUIRED,
        label="fresh_selection",
    )
    if (
        len(molecule_rows) != MOLECULE_COUNT
        or len(selection_rows) != MOLECULE_COUNT
        or len(sample_rows) != SAMPLE_COUNT
        or len(paired_rows) != SAMPLE_COUNT
    ):
        raise FreshDevIsolationError("fresh_count_contract")
    if [int(row["selection_index"]) for row in selection_rows] != list(
        range(MOLECULE_COUNT)
    ):
        raise FreshDevIsolationError("selection_index_contract")
    for selected, molecule in zip(selection_rows, molecule_rows):
        expected = {
            "molecule_id": selected["molecule_id"],
            "inchikey": selected["inchikey_full"],
            "scaffold_id": selected["scaffold_id"],
            "source_record_id": selected["source_record_id"],
        }
        if any(molecule[field] != value for field, value in expected.items()):
            raise FreshDevIsolationError("molecule_selection_projection")
    if [row["sample_key"] for row in sample_rows] != [
        row["sample_key"] for row in paired_rows
    ]:
        raise FreshDevIsolationError("sample_paired_order")
    for sample, paired in zip(sample_rows, paired_rows):
        for field in SAMPLE_REQUIRED:
            if sample[field] != paired[field]:
                raise FreshDevIsolationError(
                    f"sample_paired_projection:{sample['sample_key']}:{field}"
                )
    if any(row["split"] != "fresh_dev" for row in paired_rows):
        raise FreshDevIsolationError("fresh_split_contract")

    identity_report = _read_json(identity_report_path)
    if (
        identity_report.get("schema_version")
        != "visual-svg-layers-c-fresh-dev-identity-isolation-0.1"
        or identity_report.get("status") != "passed"
        or identity_report.get("producer") != "independent_identity_auditor"
        or identity_report.get("fresh_dev_selection_manifest_sha256")
        != _file_sha256(selection_path)
    ):
        raise FreshDevIsolationError("identity_report_contract")
    comparisons = identity_report.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise FreshDevIsolationError("identity_report_comparisons")

    fresh_identity_rows = [
        {
            "molecule_id": row["molecule_id"],
            "inchikey_full": row["inchikey"],
            "scaffold_id": row["scaffold_id"],
            "source_record_id": row["source_record_id"],
        }
        for row in molecule_rows
    ]
    fresh_identity = _identity_sets(fresh_identity_rows)
    fresh_style, fresh_style_inventory = _fresh_style_sets(
        fresh_root,
        paired_rows,
    )
    pilot_style, pilot_inventory = _pilot_style_sets(pilot)
    smoke_style, smoke_inventory = _smoke_style_sets(smoke)

    overlap_counts = {}
    for exposure, style_sets in (
        ("frozen_b_p0_train_exposure", pilot_style),
        ("smoke_train_32", smoke_style),
    ):
        identity_values = comparisons.get(exposure)
        if not isinstance(identity_values, Mapping):
            raise FreshDevIsolationError(f"identity_comparison_missing:{exposure}")
        translated = {}
        for field in IDENTITY_FIELDS:
            value = identity_values.get(f"{field}_overlap")
            if type(value) is not int or value != 0:
                raise FreshDevIsolationError(
                    f"identity_overlap_nonzero:{exposure}:{field}"
                )
            translated[field] = value
        overlap_counts[exposure] = _overlap_counts(
            fresh_identity,
            translated,
            fresh_style,
            style_sets,
            legacy_style_id_missing=True,
        )
    if any(
        count
        for values in overlap_counts.values()
        for count in values.values()
    ):
        raise FreshDevIsolationError("fresh_exposure_overlap")

    config = {
        "freeze_id": FREEZE_ID,
        "role": "fresh_dev",
        "source_split": "train",
        "molecule_count": MOLECULE_COUNT,
        "sample_count": SAMPLE_COUNT,
        "identity_fields": list(IDENTITY_FIELDS),
        "style_fields": list(STYLE_FIELDS),
        "identity_evidence": "frozen_independent_identity_report",
        "style_id_overlap_semantics": "canonical-style-v0.1",
        "style_id_zero_implied_by_font_sha_disjoint": True,
        "style_id_inference_basis": (
            "canonical style-v0.1 descriptor includes font_sha256"
        ),
        "style_evidence": {
            "frozen_b_p0_train_exposure": (
                "three fixed-SHA train tar layers sidecars"
            ),
            "smoke_train_32": "all committed layers sidecars",
            "fresh_dev": "all paired-manifest-bound layers sidecars",
        },
        "missing_required_style_field_policy": "fail_closed",
        "render_style_signature_role": "inventory_only_not_overlap_field",
    }
    hashes = {
        "fresh_dev_molecule_manifest_sha256": _file_sha256(molecule_path),
        "fresh_dev_sample_manifest_sha256": _file_sha256(sample_path),
        "fresh_dev_paired_manifest_sha256": _file_sha256(paired_path),
        "exposure_manifest_sha256": identity_report["exposure_manifest_sha256"],
        "molecule_id_exclusion_set_sha256": identity_report[
            "molecule_id_exclusion_set_sha256"
        ],
        "inchikey_full_exclusion_set_sha256": identity_report[
            "inchikey_full_exclusion_set_sha256"
        ],
        "scaffold_id_exclusion_set_sha256": identity_report[
            "scaffold_id_exclusion_set_sha256"
        ],
        "source_record_id_exclusion_set_sha256": identity_report[
            "source_record_id_exclusion_set_sha256"
        ],
        "isolation_algorithm_sha256": _file_sha256(Path(__file__).resolve()),
        "isolation_config_sha256": _sha256_bytes(_canonical_json(config)),
        "scaffold_id_algorithm_sha256": identity_report[
            "scaffold_id_algorithm_sha256"
        ],
        "source_record_normalization_algorithm_sha256": identity_report[
            "source_record_normalization_algorithm_sha256"
        ],
        "style_id_algorithm_sha256": identity_report[
            "style_id_algorithm_sha256"
        ],
    }
    if set(hashes) != set(HASH_FIELDS):
        raise FreshDevIsolationError("isolation_hash_field_closure")
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer": "independent_isolation_auditor",
        "role": "fresh_dev",
        "status": "passed",
        "source_split": "train",
        "overlap_counts": overlap_counts,
        "hashes": hashes,
        "identity_isolation_report_sha256": _file_sha256(identity_report_path),
        "selection_manifest_sha256": _file_sha256(selection_path),
        "config": config,
        "style_exposure_inventory": {
            "frozen_b_p0_train_exposure": pilot_inventory,
            "smoke_train_32": smoke_inventory,
        },
        "fresh_style_source_inventory": fresh_style_inventory,
        "fresh_style_inventory": {
            field: sorted(values) for field, values in fresh_style.items()
        },
        "old_heldout_or_sealed_rows_read": False,
        "model_run": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Isolation report exists; pass --overwrite: {output}")
    output.write_bytes(_canonical_json(report, newline=True))
    return {"output": str(output), **report}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    version_root = root / "visual_svg_layers_c_v0_1"
    fresh_root = version_root / "fresh_dev"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", default=str(fresh_root))
    parser.add_argument(
        "--pilot",
        default=str(root / "chemvision_layers_v0_1/pilot_10k"),
    )
    parser.add_argument(
        "--smoke",
        default=str(version_root / "smoke_train_32"),
    )
    parser.add_argument("--output", default=str(fresh_root / "isolation_report.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = audit(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "HASH_FIELDS",
    "OVERLAP_FIELDS",
    "SCHEMA_VERSION",
    "FreshDevIsolationError",
    "audit",
    "build_parser",
    "main",
]
