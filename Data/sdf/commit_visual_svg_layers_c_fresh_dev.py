#!/usr/bin/env python3
"""Freeze a validated full fresh-dev package as ``commitment.dev.json``.

The command accepts only the complete 2,000/8,000 build, a passed independent
isolation report, and a passed full validation report.  It recomputes every
referenced hash, writes the commitment canonically, and finally asks the
production loader to validate the resulting package shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "visual-svg-layers-c-input-commitment-0.1"
BUILD_SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-build-0.1"
VALIDATION_SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-validation-0.1"
ISOLATION_SCHEMA_VERSION = "visual-svg-layers-c-isolation-report-0.1"
FREEZE_SHA256 = "6c87e2e63b42c6b9f39c33d829fc8883d456b7c5f82cd15b2672963a7f560b9c"
MOLECULE_COUNT = 2_000
SAMPLE_COUNT = 8_000
MEMBERS_PER_SAMPLE = 12
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
ARTIFACT_FIELDS = {
    "paired_manifest_sha256",
    "molecule_manifest_sha256",
    "sample_manifest_sha256",
    "preflight_validation_sha256",
    "isolation_report_sha256",
    "selection_commitment_sha256",
    "selection_manifest_sha256",
    "replacement_manifest_sha256",
    "identity_isolation_report_sha256",
    "selection_provenance_sha256",
}


class FreshDevCommitmentError(ValueError):
    """Raised when final fresh-dev artifacts are not ready to freeze."""


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
            raise FreshDevCommitmentError(f"missing_file:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise FreshDevCommitmentError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise FreshDevCommitmentError(f"not_regular_file:{absolute}")
    return absolute


def _read_json(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshDevCommitmentError(f"invalid_json:{path}") from error
    if not isinstance(value, dict):
        raise FreshDevCommitmentError(f"json_root_not_object:{path}")
    return value


def _safe_relative(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise FreshDevCommitmentError(f"unsafe_relative_path:{label}:{relative}")
    path = _regular_file(root / relative)
    if os.path.commonpath((str(root), str(path))) != str(root):
        raise FreshDevCommitmentError(f"path_escapes_root:{label}:{relative}")
    return path


def _selection_artifacts(
    fresh_root: Path,
    selection_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, str]:
    result = {"selection_commitment_sha256": _file_sha256(selection_path)}
    for output_name, reference_name in (
        ("selection_manifest_sha256", "selection_manifest"),
        ("replacement_manifest_sha256", "fresh_replacement_manifest"),
        ("identity_isolation_report_sha256", "identity_isolation_report"),
        ("selection_provenance_sha256", "selection_provenance"),
    ):
        reference = selection.get(reference_name)
        if not isinstance(reference, Mapping):
            raise FreshDevCommitmentError(
                f"selection_reference_missing:{reference_name}"
            )
        path = _safe_relative(
            fresh_root,
            reference.get("path", ""),
            label=reference_name,
        )
        digest = _file_sha256(path)
        if reference.get("sha256") != digest:
            raise FreshDevCommitmentError(
                f"selection_reference_sha256:{reference_name}"
            )
        result[output_name] = digest
    return result


def _hardlink_regular(source: Path, destination: Path) -> None:
    source = _regular_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _validate_candidate_with_production_loader(
    candidate_temp: Path,
    *,
    fresh_root: Path,
    payload: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Validate the exact temp-file inode in an equivalent fresh_dev tree."""

    stage = Path(
        tempfile.mkdtemp(
            prefix=".commitment-candidate-",
            dir=fresh_root.parent,
        )
    )
    candidate_root = stage / "fresh_dev"
    candidate_root.mkdir()
    try:
        _hardlink_regular(
            fresh_root.parent / "phase_a_freeze.json",
            stage / "phase_a_freeze.json",
        )
        relative_inputs = {
            str(payload["isolation_report"]["path"]),
            str(payload["selection"]["path"]),
            str(payload["molecule_manifest"]["path"]),
            str(payload["sample_manifest"]["path"]),
            str(payload["paired_manifest"]["path"]),
            str(payload["preflight_validation"]["path"]),
            str(payload["validation"]["path"]),
            *(str(reference["path"]) for reference in payload["shards"]),
            str(selection["selection_manifest"]["path"]),
            str(selection["fresh_replacement_manifest"]["path"]),
            str(selection["identity_isolation_report"]["path"]),
            str(selection["selection_provenance"]["path"]),
        }
        for relative in sorted(relative_inputs):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise FreshDevCommitmentError(
                    f"candidate_input_path:{relative}"
                )
            _hardlink_regular(fresh_root / path, candidate_root / path)
        os.link(candidate_temp, candidate_root / "commitment.dev.json")

        from chemvector.evidence_carrier_fresh_dev_data import (
            load_fresh_evidence_carrier_package,
        )

        candidate_commitment = candidate_root / "commitment.dev.json"
        package = load_fresh_evidence_carrier_package(
            candidate_commitment,
            expected_role="dev",
            expected_commitment_sha256=_file_sha256(candidate_temp),
        )
        if package.sample_count != SAMPLE_COUNT or package.molecule_count != MOLECULE_COUNT:
            raise FreshDevCommitmentError("production_loader_count")
    finally:
        shutil.rmtree(stage)


def _write_fsynced_temp(output: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".commitment.dev.",
        suffix=".tmp",
        dir=output.parent,
    )
    path = Path(name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_install_with_restore(candidate: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists():
        descriptor, backup_name = tempfile.mkstemp(
            prefix=".commitment.dev.backup.",
            suffix=".tmp",
            dir=output.parent,
        )
        os.close(descriptor)
        backup = Path(backup_name)
        backup.unlink()
        os.link(_regular_file(output), backup)
    try:
        os.replace(candidate, output)
        try:
            _fsync_directory(output.parent)
        except Exception:
            if backup is None:
                output.unlink(missing_ok=True)
            else:
                os.replace(backup, output)
                backup = None
            _fsync_directory(output.parent)
            raise
        if backup is not None:
            backup.unlink()
            backup = None
    finally:
        if backup is not None:
            backup.unlink(missing_ok=True)


def commit(args: argparse.Namespace) -> dict[str, Any]:
    fresh_root = Path(args.fresh_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    if fresh_root.name != "fresh_dev" or output != fresh_root / "commitment.dev.json":
        raise FreshDevCommitmentError(
            "commitment_path_must_be_fresh_dev_commitment_dev_json"
        )
    build_path = _regular_file(fresh_root / "build.dev.json")
    validation_path = _regular_file(fresh_root / "validation.json")
    isolation_path = _regular_file(fresh_root / "isolation_report.json")
    selection_path = _regular_file(fresh_root / "selection.commitment.json")
    freeze_path = _regular_file(fresh_root.parent / "phase_a_freeze.json")
    build = _read_json(build_path)
    validation = _read_json(validation_path)
    isolation = _read_json(isolation_path)
    selection = _read_json(selection_path)

    if (
        build.get("schema_version") != BUILD_SCHEMA_VERSION
        or build.get("status") != "built_pending_independent_validation"
        or build.get("finalizable") is not True
        or build.get("molecules") != MOLECULE_COUNT
        or build.get("samples") != SAMPLE_COUNT
        or build.get("members") != SAMPLE_COUNT * MEMBERS_PER_SAMPLE
        or build.get("errors") != []
    ):
        raise FreshDevCommitmentError("full_build_contract")
    if (
        validation.get("schema_version") != VALIDATION_SCHEMA_VERSION
        or validation.get("status") != "passed"
        or validation.get("finalizable") is not True
        or validation.get("fresh_audit_eligible") is not True
        or validation.get("molecules") != MOLECULE_COUNT
        or validation.get("samples") != SAMPLE_COUNT
        or validation.get("members") != SAMPLE_COUNT * MEMBERS_PER_SAMPLE
        or validation.get("checked_samples") != SAMPLE_COUNT
        or validation.get("checked_members") != SAMPLE_COUNT * MEMBERS_PER_SAMPLE
        or validation.get("structure_validated_samples") != SAMPLE_COUNT
        or validation.get("errors") != []
    ):
        raise FreshDevCommitmentError("full_validation_contract")
    if (
        isolation.get("schema_version") != ISOLATION_SCHEMA_VERSION
        or isolation.get("status") != "passed"
        or isolation.get("producer") != "independent_isolation_auditor"
        or isolation.get("role") != "fresh_dev"
    ):
        raise FreshDevCommitmentError("isolation_report_contract")
    if _file_sha256(freeze_path) != FREEZE_SHA256:
        raise FreshDevCommitmentError("phase_a_freeze_sha256")

    artifact_sha256 = validation.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != ARTIFACT_FIELDS:
        raise FreshDevCommitmentError("validation_artifact_fields")
    selection_hashes = _selection_artifacts(fresh_root, selection_path, selection)
    for name, digest in selection_hashes.items():
        if artifact_sha256.get(name) != digest:
            raise FreshDevCommitmentError(f"validation_selection_binding:{name}")

    manifests = build.get("manifests")
    if not isinstance(manifests, Mapping):
        raise FreshDevCommitmentError("build_manifests")
    manifest_bindings = {}
    for build_name, output_name, expected_path, artifact_name in (
        ("molecule", "molecule_manifest", "molecule_manifest.dev.csv", "molecule_manifest_sha256"),
        ("sample", "sample_manifest", "sample_manifest.dev.csv", "sample_manifest_sha256"),
        ("paired", "paired_manifest", "paired_manifest.dev.csv", "paired_manifest_sha256"),
    ):
        reference = manifests.get(build_name)
        if not isinstance(reference, Mapping) or reference.get("path") != expected_path:
            raise FreshDevCommitmentError(f"build_manifest_reference:{build_name}")
        path = _regular_file(fresh_root / expected_path)
        digest = _file_sha256(path)
        if reference.get("sha256") != digest or artifact_sha256.get(artifact_name) != digest:
            raise FreshDevCommitmentError(f"manifest_binding:{build_name}")
        manifest_bindings[output_name] = {"path": expected_path, "sha256": digest}

    isolation_sha256 = _file_sha256(isolation_path)
    preflight_path = _regular_file(fresh_root / "preflight_64/preflight_validation.json")
    preflight_sha256 = _file_sha256(preflight_path)
    if (
        artifact_sha256.get("isolation_report_sha256") != isolation_sha256
        or artifact_sha256.get("preflight_validation_sha256") != preflight_sha256
    ):
        raise FreshDevCommitmentError("validation_supporting_artifact_binding")

    implementation = validation.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or set(implementation) != IMPLEMENTATION_FIELDS:
        raise FreshDevCommitmentError("validation_implementation_fields")
    for name, value in implementation.items():
        if not isinstance(value, str) or len(value) != 64:
            raise FreshDevCommitmentError(f"validation_implementation_hash:{name}")
    if validation.get("font_inventory") != FONT_INVENTORY or build.get(
        "font_inventory"
    ) != FONT_INVENTORY:
        raise FreshDevCommitmentError("font_inventory_binding")

    isolation_hashes = isolation.get("hashes")
    if not isinstance(isolation_hashes, Mapping) or set(
        isolation_hashes
    ) != ISOLATION_HASH_FIELDS:
        raise FreshDevCommitmentError("isolation_hash_fields")
    expected_implementation = {
        "isolation_auditor_sha256": isolation_hashes[
            "isolation_algorithm_sha256"
        ],
        "style_id_algorithm_sha256": isolation_hashes[
            "style_id_algorithm_sha256"
        ],
        "scaffold_id_algorithm_sha256": isolation_hashes[
            "scaffold_id_algorithm_sha256"
        ],
        "source_record_normalization_algorithm_sha256": isolation_hashes[
            "source_record_normalization_algorithm_sha256"
        ],
    }
    for name, value in expected_implementation.items():
        if implementation.get(name) != value:
            raise FreshDevCommitmentError(f"implementation_binding:{name}")
    implementation_paths = {
        "builder_sha256": (
            Path(__file__).resolve().parent
            / "build_visual_svg_layers_c_fresh_dev.py"
        ),
        "validator_sha256": (
            Path(__file__).resolve().parent
            / "validate_visual_svg_layers_c_fresh_dev.py"
        ),
        "isolation_auditor_sha256": (
            Path(__file__).resolve().parent
            / "audit_visual_svg_layers_c_fresh_dev_isolation.py"
        ),
    }
    for name, path in implementation_paths.items():
        if implementation.get(name) != _file_sha256(path):
            raise FreshDevCommitmentError(f"implementation_file_binding:{name}")

    shard_refs = build.get("shards")
    validation_shards = validation.get("shards")
    if not isinstance(shard_refs, list) or not shard_refs:
        raise FreshDevCommitmentError("build_shards")
    expected_validation_shards = []
    committed_shards = []
    for reference in shard_refs:
        if not isinstance(reference, Mapping):
            raise FreshDevCommitmentError("build_shard_reference")
        path = _safe_relative(
            fresh_root,
            reference.get("path", ""),
            label="dev_shard",
        )
        digest = _file_sha256(path)
        if (
            reference.get("sha256") != digest
            or reference.get("bytes") != path.stat().st_size
            or reference.get("members")
            != int(reference.get("samples", -1)) * MEMBERS_PER_SAMPLE
        ):
            raise FreshDevCommitmentError(f"shard_binding:{reference.get('path')}")
        expected_validation_shards.append(
            {
                "path": reference["path"],
                "sha256": digest,
                "samples": reference["samples"],
                "members": reference["members"],
            }
        )
        committed_shards.append(dict(reference))
    if validation_shards != expected_validation_shards or sum(
        int(row["samples"]) for row in committed_shards
    ) != SAMPLE_COUNT:
        raise FreshDevCommitmentError("validation_shard_binding")

    exclusion_sets = {
        "molecule_id_sha256": isolation_hashes[
            "molecule_id_exclusion_set_sha256"
        ],
        "inchikey_full_sha256": isolation_hashes[
            "inchikey_full_exclusion_set_sha256"
        ],
        "scaffold_id_sha256": isolation_hashes[
            "scaffold_id_exclusion_set_sha256"
        ],
        "source_record_id_sha256": isolation_hashes[
            "source_record_id_exclusion_set_sha256"
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": "dev",
        "split": "fresh_dev",
        "fresh_audit_eligible": True,
        "molecule_count": MOLECULE_COUNT,
        "sample_count": SAMPLE_COUNT,
        "expected_total_members": SAMPLE_COUNT * MEMBERS_PER_SAMPLE,
        "width": 768,
        "height": 768,
        "max_tokens": 4_096,
        "sdf_protocol": "source-sdf-byte-preserving-0.1",
        "render_profile": "visual-svg-layers-c-clean-bw-768-v0.1",
        "variants": ["original", "rdkit", "coordgen", "stress"],
        "members_per_sample": MEMBERS_PER_SAMPLE,
        "freeze": {
            "path": "../phase_a_freeze.json",
            "sha256": FREEZE_SHA256,
        },
        "source": {
            "dataset_root": "Data/sdf/webdataset_full_v0",
            "split": "train",
        },
        "exclusion_sets": exclusion_sets,
        "isolation_report": {
            "path": "isolation_report.json",
            "sha256": isolation_sha256,
            "status": "passed",
        },
        "selection": {
            "path": "selection.commitment.json",
            "sha256": _file_sha256(selection_path),
        },
        "fonts": FONT_INVENTORY,
        "implementation": dict(implementation),
        **manifest_bindings,
        "shards": committed_shards,
        "preflight_validation": {
            "path": "preflight_64/preflight_validation.json",
            "sha256": preflight_sha256,
            "status": "PREFLIGHT-PASS",
        },
        "validation": {
            "path": "validation.json",
            "sha256": _file_sha256(validation_path),
            "status": "passed",
        },
    }
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Commitment exists; pass --overwrite: {output}")
    candidate_bytes = _canonical_json(payload, newline=True)
    candidate_temp = _write_fsynced_temp(output, candidate_bytes)
    try:
        _validate_candidate_with_production_loader(
            candidate_temp,
            fresh_root=fresh_root,
            payload=payload,
            selection=selection,
        )
        _atomic_install_with_restore(candidate_temp, output)
    except Exception:
        candidate_temp.unlink(missing_ok=True)
        raise
    return {
        "schema_version": "visual-svg-layers-c-fresh-dev-commit-run-0.1",
        "status": "committed",
        "output": str(output),
        "sha256": _file_sha256(output),
        "molecules": MOLECULE_COUNT,
        "samples": SAMPLE_COUNT,
        "members": SAMPLE_COUNT * MEMBERS_PER_SAMPLE,
        "production_loader_checked": True,
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    fresh_root = root / "visual_svg_layers_c_v0_1/fresh_dev"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", default=str(fresh_root))
    parser.add_argument("--output", default=str(fresh_root / "commitment.dev.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = commit(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "FreshDevCommitmentError",
    "build_parser",
    "commit",
    "main",
]
