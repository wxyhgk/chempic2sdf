#!/usr/bin/env python3
"""Select the committed Phase-A fresh-dev identities from full-v0 train only.

This program selects fresh_dev and its fresh replacement pool.  It deliberately
does not select, materialize, or inspect sealed-audit or reserve identities.
An independent auditor must validate these outputs before creating
``selection.commitment.json``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
from typing import Any, Mapping, Sequence

from rdkit import Chem

try:
    from .build_visual_svg_layers_c_fresh_dev_preflight import (
        FREEZE_ID,
        SELECTION_FIELDS,
        _validate_chemmol_against_sdf,
        canonical_json_bytes,
        file_sha256,
        selection_rank,
        sha256_bytes,
    )
    from .build_webdataset_sample import mol_from_chemmol, scaffold_smiles
except ImportError:
    from build_visual_svg_layers_c_fresh_dev_preflight import (
        FREEZE_ID,
        SELECTION_FIELDS,
        _validate_chemmol_against_sdf,
        canonical_json_bytes,
        file_sha256,
        selection_rank,
        sha256_bytes,
    )
    from build_webdataset_sample import mol_from_chemmol, scaffold_smiles


SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-selection-0.1"
FREEZE_SHA256 = "6c87e2e63b42c6b9f39c33d829fc8883d456b7c5f82cd15b2672963a7f560b9c"
PILOT_SUMMARY_SHA256 = "0ca09a9759bfd902e265555e07592b71f87e95a464fc5ceaa21e4257031ef8f7"
PILOT_TRAIN_SHARD_SHA256 = {
    "train-000000.tar": "6e5a21d14cd0d1521de44f67eaea111e4c0272d7b7652cb1e88c13f82443b042",
    "train-000001.tar": "9ca92e15d260d4372135177990e1f9fdeca771d8e78cbcd7ba2c735128035aea",
    "train-000002.tar": "5d9eba5abe8277368b68ec9bdd6fc3c4abeaaf339acd0874e00a815d95311aa7",
}
SMOKE_MOLECULES_SHA256 = "3dddac1ff9e5babb7d98dbca835648ee229f1574c4328cbd525d33a9a3ecae60"
SMOKE_COMMITMENT_SHA256 = "e8c0bf1fbd792b60836ab879ee7a8b2dbee230f0a6a426bf710fee1e0ad6825a"
SOURCE_VALIDATION_SHA256 = "8e37f9c2037e01d8f417b882e1405fa625a8a5f81f283f647d5788f972f6aa78"
TRAIN_MANIFEST_INVENTORY_SHA256 = "cee36b8e8737c3de1826d868955dcc27dcf656d63d8a498f08425708d14acbdb"
FRESH_COUNT = 2000
REPLACEMENT_COUNT = 500
FORBIDDEN_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "reserve",
    "mr_tadf",
    "mr-tadf",
)
IDENTITY_FIELDS = (
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
)
REPLACEMENT_FIELDS = (
    "replacement_index",
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
    "purpose",
    "rendered",
)


class FreshDevSelectionError(ValueError):
    """Raised when train-only selection cannot satisfy the frozen contract."""


def _forbidden_marker(value: object) -> str | None:
    lowered = str(value).lower()
    return next((item for item in FORBIDDEN_MARKERS if item in lowered), None)


def _assert_allowed(value: object, *, context: str) -> None:
    marker = _forbidden_marker(value)
    if marker is not None:
        raise FreshDevSelectionError(f"forbidden_reference:{context}:{marker}")


def _regular_file(path: Path, *, allow_aggregate_validation: bool = False) -> Path:
    absolute = Path(os.path.abspath(path))
    if not allow_aggregate_validation:
        _assert_allowed(absolute, context="input_path")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise FreshDevSelectionError(f"missing_input:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise FreshDevSelectionError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise FreshDevSelectionError(f"input_not_regular:{absolute}")
    return absolute


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _set_sha256(values: Sequence[str] | set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


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
    return sha256_bytes(canonical_json_bytes(payload))


def _load_freeze(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    if file_sha256(path) != FREEZE_SHA256:
        raise FreshDevSelectionError("phase_a_freeze_sha256_mismatch")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if (
        freeze.get("freeze_id") != FREEZE_ID
        or freeze.get("status") != "frozen_fresh_dev_preflight_authorized"
        or freeze.get("phase") != "A"
        or freeze.get("training_steps") != 0
        or freeze.get("parameter_updates") != 0
    ):
        raise FreshDevSelectionError("freeze_contract_mismatch")
    source = freeze.get("source") or {}
    if (
        source.get("required_source_split") != "train"
        or source.get("read_old_validation_or_internal_rows") is not False
    ):
        raise FreshDevSelectionError("freeze_source_contract_mismatch")
    selection = freeze.get("selection") or {}
    if (
        selection.get("ordering") != "sha256_rank_v0.1"
        or selection.get("materialize_sealed_rows_before_fresh_dev_gate") is not False
    ):
        raise FreshDevSelectionError("freeze_selection_contract_mismatch")
    return freeze


def _validate_frozen_source(source: Path) -> tuple[Path, dict[str, Any]]:
    expected = (Path(__file__).resolve().parent / "webdataset_full_v0").resolve()
    if source != expected:
        raise FreshDevSelectionError(f"source_root_not_frozen:{source}:{expected}")
    aggregate_path = _regular_file(
        source / "validation.json", allow_aggregate_validation=True
    )
    if file_sha256(aggregate_path) != SOURCE_VALIDATION_SHA256:
        raise FreshDevSelectionError("source_validation_sha256_mismatch")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("status") != "passed" or aggregate.get("scaffold_overlap") != 0:
        raise FreshDevSelectionError("source_aggregate_validation_not_passed")
    paths = sorted(
        [
            *(source / "manifests/shards").glob("train-*.molecules.csv"),
            *(source / "manifests/shards").glob("train-*.samples.csv"),
        ]
    )
    if len(paths) != 260:
        raise FreshDevSelectionError(f"train_manifest_inventory_count:{len(paths)}")
    digest = hashlib.sha256()
    for path in paths:
        path = _regular_file(path)
        relative = path.relative_to(source).as_posix()
        digest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
    if digest.hexdigest() != TRAIN_MANIFEST_INVENTORY_SHA256:
        raise FreshDevSelectionError("train_manifest_inventory_sha256_mismatch")
    return aggregate_path, aggregate


def _source_record(row: Mapping[str, str]) -> str:
    return f"{row['source']}:{row['source_id']}"


def _candidate_from_source(row: Mapping[str, str]) -> dict[str, str]:
    if row["molecule_id"] != _source_record(row):
        raise FreshDevSelectionError("source_record_id_not_molecule_id")
    candidate = {
        "selection_index": "",
        "rank_sha256": "",
        "molecule_id": row["molecule_id"],
        "inchikey_full": row["inchikey"],
        "scaffold_id": row["scaffold"],
        "source_record_id": _source_record(row),
        "source": row["source"],
        "source_id": row["source_id"],
        "source_split": row["split"],
        "source_shard": row["shard"],
        "source_structure": row["structure_ref"],
    }
    candidate["rank_sha256"] = selection_rank(candidate)
    return candidate


def load_train_candidates(source: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read only explicitly named train molecule manifests."""

    manifest_root = source / "manifests/shards"
    manifests = sorted(manifest_root.glob("train-*.molecules.csv"))
    if not manifests:
        raise FreshDevSelectionError("no_train_molecule_manifests")
    candidates: list[dict[str, str]] = []
    manifest_records: list[dict[str, str]] = []
    for manifest in manifests:
        manifest = _regular_file(manifest)
        stem = manifest.name.removesuffix(".molecules.csv")
        expected_shard = f"shards/{stem}.tar"
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "molecule_id",
                "source",
                "source_id",
                "split",
                "shard",
                "scaffold",
                "inchikey",
                "structure_ref",
            }
            if not required.issubset(set(reader.fieldnames or ())):
                raise FreshDevSelectionError(f"train_manifest_fields:{manifest.name}")
            row_count = 0
            for row in reader:
                row_count += 1
                if (
                    row["split"] != "train"
                    or row["shard"] != expected_shard
                    or not row["structure_ref"].startswith(f"structures/{stem}/")
                    or row["molecule_id"] != _source_record(row)
                ):
                    raise FreshDevSelectionError(
                        f"train_manifest_row_contract:{manifest.name}:{row_count + 1}"
                    )
                for field, value in row.items():
                    _assert_allowed(
                        value,
                        context=f"{manifest.name}:{row_count + 1}:{field}",
                    )
                candidates.append(_candidate_from_source(row))
        manifest_records.append(
            {
                "path": manifest.relative_to(source).as_posix(),
                "sha256": file_sha256(manifest),
                "rows": str(row_count),
            }
        )
    for field in ("molecule_id", "source_record_id"):
        values = [row[field] for row in candidates]
        if len(values) != len(set(values)):
            raise FreshDevSelectionError(f"duplicate_train_identity:{field}")
    return candidates, manifest_records


def _identity_from_chemmol(chemmol: Mapping[str, Any]) -> dict[str, str]:
    molecule_id = str(chemmol["molecule_id"])
    if ":" not in molecule_id:
        raise FreshDevSelectionError(f"exposure_molecule_id:{molecule_id}")
    source, source_id = molecule_id.split(":", 1)
    molecule = mol_from_chemmol(dict(chemmol))
    Chem.SanitizeMol(molecule)
    canonical_smiles = str(chemmol["canonical_isomeric_smiles"])
    return {
        "selection_index": "",
        "rank_sha256": "",
        "molecule_id": molecule_id,
        "inchikey_full": str(chemmol["inchikey"]),
        "scaffold_id": scaffold_smiles(molecule, canonical_smiles),
        "source_record_id": molecule_id,
        "source": source,
        "source_id": source_id,
        "source_split": "train",
        "source_shard": "exposure-only",
        "source_structure": str(chemmol.get("structure_ref", "")),
    }


def _load_exposure_rows_from_shards(
    shard_paths: Sequence[Path],
    *,
    relative_to: Path,
    expected_layer_splits: set[str],
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    identities: dict[str, dict[str, str]] = {}
    records = []
    if not shard_paths:
        raise FreshDevSelectionError("exposure_shards_missing")
    for shard in shard_paths:
        shard = _regular_file(shard)
        if expected_sha256 is not None:
            expected = expected_sha256.get(shard.name)
            if expected is None or file_sha256(shard) != expected:
                raise FreshDevSelectionError(
                    f"exposure_shard_inventory_or_sha256:{shard.name}"
                )
        layer_splits: dict[str, str] = {}
        chemmol_rows = 0
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if member.name.endswith(".layers.json"):
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise FreshDevSelectionError(
                            f"exposure_layer_unreadable:{member.name}"
                        )
                    layers = json.load(extracted)
                    split = str(layers.get("split", ""))
                    if split not in expected_layer_splits:
                        raise FreshDevSelectionError(
                            f"exposure_layer_split:{split}"
                        )
                    source = layers.get("source") or {}
                    if not str(source.get("source_shard", "")).startswith(
                        "shards/train-"
                    ):
                        raise FreshDevSelectionError("exposure_source_not_train")
                    key = member.name.removesuffix(".layers.json")
                    layer_splits[key] = split
                elif member.name.endswith(".chemmol.json"):
                    key = member.name.removesuffix(".chemmol.json")
                    if key not in layer_splits:
                        raise FreshDevSelectionError(
                            f"exposure_chemmol_precedes_layer:{key}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise FreshDevSelectionError(
                            f"exposure_chemmol_unreadable:{member.name}"
                        )
                    chemmol = json.load(extracted)
                    identity = _identity_from_chemmol(chemmol)
                    previous = identities.get(identity["molecule_id"])
                    if previous is not None and any(
                        previous[field] != identity[field]
                        for field in IDENTITY_FIELDS
                    ):
                        raise FreshDevSelectionError(
                            f"exposure_identity_drift:{identity['molecule_id']}"
                        )
                    identities[identity["molecule_id"]] = identity
                    chemmol_rows += 1
        records.append(
            {
                "path": shard.relative_to(relative_to).as_posix(),
                "sha256": file_sha256(shard),
                "chemmol_rows": str(chemmol_rows),
            }
        )
    if not identities:
        raise FreshDevSelectionError("exposure_identity_empty")
    return list(identities.values()), records


def _load_smoke_exposure_rows(
    smoke: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = _regular_file(smoke / "molecules.csv")
    if file_sha256(manifest) != SMOKE_MOLECULES_SHA256:
        raise FreshDevSelectionError("smoke_molecules_sha256_mismatch")
    commitment_path = _regular_file(smoke / "commitment.smoke.json")
    if file_sha256(commitment_path) != SMOKE_COMMITMENT_SHA256:
        raise FreshDevSelectionError("smoke_commitment_sha256_mismatch")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if (
        commitment.get("role") != "smoke"
        or commitment.get("fresh_audit_eligible") is not False
        or commitment.get("molecule_count") != 8
        or commitment.get("sample_count") != 32
        or commitment.get("members_per_sample") != 12
        or len(commitment.get("shards") or ()) != 1
    ):
        raise FreshDevSelectionError("smoke_commitment_contract")
    paired_ref = commitment.get("paired_manifest") or {}
    paired = _regular_file(smoke / str(paired_ref.get("path", "")))
    if paired_ref.get("sha256") != file_sha256(paired):
        raise FreshDevSelectionError("smoke_paired_sha256_mismatch")
    shard_ref = commitment["shards"][0]
    committed_shard = _regular_file(smoke / str(shard_ref.get("path", "")))
    if (
        shard_ref.get("sha256") != file_sha256(committed_shard)
        or shard_ref.get("samples") != 32
        or shard_ref.get("members") != 384
    ):
        raise FreshDevSelectionError("smoke_shard_commitment_mismatch")
    manifest_ids = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "smoke_train":
                raise FreshDevSelectionError("smoke_manifest_split")
            manifest_ids.add(row["molecule_id"])
    if not manifest_ids:
        raise FreshDevSelectionError("smoke_exposure_empty")
    if len(manifest_ids) != 8:
        raise FreshDevSelectionError("smoke_molecule_count")
    rows, shard_records = _load_exposure_rows_from_shards(
        sorted((smoke / "shards").glob("smoke-train-*.tar")),
        relative_to=smoke,
        expected_layer_splits={"train", "smoke_train"},
    )
    if {row["molecule_id"] for row in rows} != manifest_ids:
        raise FreshDevSelectionError("smoke_manifest_shard_identity_mismatch")
    if sum(int(item["chemmol_rows"]) for item in shard_records) != 32:
        raise FreshDevSelectionError("smoke_sample_inventory")
    return rows, {
        "path": manifest.relative_to(smoke).as_posix(),
        "sha256": file_sha256(manifest),
        "rows": str(len(manifest_ids)),
        "shards": shard_records,
    }


def _identity_sets(rows: Sequence[Mapping[str, str]]) -> dict[str, set[str]]:
    return {field: {row[field] for row in rows} for field in IDENTITY_FIELDS}


def _overlaps_any(row: Mapping[str, str], sets: Mapping[str, set[str]]) -> bool:
    return any(row[field] in sets[field] for field in IDENTITY_FIELDS)


def _sample_manifest_path(source: Path, shard: str) -> Path:
    if not shard.startswith("shards/train-") or not shard.endswith(".tar"):
        raise FreshDevSelectionError(f"invalid_train_shard:{shard}")
    stem = Path(shard).stem
    return _regular_file(
        source / "manifests/shards" / f"{stem}.samples.csv"
    )


def load_original_sample_rows(
    source: Path,
    candidates: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    wanted_by_manifest: dict[Path, set[str]] = defaultdict(set)
    for row in candidates:
        wanted_by_manifest[_sample_manifest_path(source, row["source_shard"])].add(
            row["molecule_id"]
        )
    result = {}
    for manifest, wanted in wanted_by_manifest.items():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("molecule_id") not in wanted or row.get("variant") != "original":
                    continue
                if row.get("split") != "train" or not row.get("shard", "").startswith(
                    "shards/train-"
                ):
                    raise FreshDevSelectionError("original_sample_not_train")
                if row["molecule_id"] in result:
                    raise FreshDevSelectionError("duplicate_original_sample")
                result[row["molecule_id"]] = row
    missing = sorted({row["molecule_id"] for row in candidates} - set(result))
    if missing:
        raise FreshDevSelectionError(f"original_sample_missing:{missing[0]}")
    return result


class TrainChemmolReader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.archives: dict[str, tarfile.TarFile] = {}

    def read(self, sample_row: Mapping[str, str]) -> bytes:
        shard_name = sample_row["shard"]
        archive = self.archives.get(shard_name)
        if archive is None:
            archive = tarfile.open(_regular_file(self.source / shard_name), "r")
            self.archives[shard_name] = archive
        name = sample_row["sample_key"] + ".chemmol.json"
        try:
            member = archive.getmember(name)
        except KeyError as error:
            raise FreshDevSelectionError(f"source_chemmol_missing:{name}") from error
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FreshDevSelectionError(f"source_chemmol_unreadable:{name}")
        return extracted.read()

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()


def _candidate_precheck(
    source: Path,
    row: Mapping[str, str],
    original_sample: Mapping[str, str],
    reader: TrainChemmolReader,
) -> dict[str, Any]:
    structure = row["source_structure"]
    if not structure.startswith("structures/train-"):
        raise FreshDevSelectionError("candidate_structure_not_train")
    sdf_path = _regular_file(source / structure)
    sdf_bytes = sdf_path.read_bytes()
    chemmol_bytes = reader.read(original_sample)
    try:
        source_chemmol = json.loads(chemmol_bytes)
    except json.JSONDecodeError as error:
        raise FreshDevSelectionError("candidate_chemmol_json") from error
    check_identity = dict(row)
    check_identity["source_atom_count"] = str(len(source_chemmol.get("atoms", ())))
    check_identity["source_bond_count"] = str(len(source_chemmol.get("bonds", ())))
    chemmol = _validate_chemmol_against_sdf(
        chemmol_bytes,
        sdf_bytes,
        identity=check_identity,
    )
    return {
        "sdf_sha256": sha256_bytes(sdf_bytes),
        "sdf_bytes": len(sdf_bytes),
        "chemmol_sha256": sha256_bytes(chemmol_bytes),
        "atoms": len(chemmol["atoms"]),
        "bonds": len(chemmol["bonds"]),
    }


def _output_paths(output: Path) -> tuple[Path, ...]:
    return (
        output / "selection.committed.csv",
        output / "replacement_pool.committed.csv",
        output / "candidate_rejected.csv",
        output / "selection_provenance.json",
        output / "identity_isolation_report.json",
        output / "selection.commitment.json",
        output / "preflight_64",
    )


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    existing = [path for path in _output_paths(output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Fresh-dev selection output exists: {existing[0]}")
    if overwrite:
        for path in _output_paths(output):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)


def select(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    pilot = Path(args.pilot).expanduser().resolve()
    smoke = Path(args.smoke).expanduser().resolve()
    freeze_path = Path(args.freeze).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    for value, context in (
        (source, "source"),
        (pilot, "pilot"),
        (smoke, "smoke"),
        (output, "output"),
    ):
        _assert_allowed(value, context=context)
    freeze = _load_freeze(freeze_path)
    aggregate_validation, aggregate = _validate_frozen_source(source)

    candidates, source_manifest_records = load_train_candidates(source)
    by_molecule = {row["molecule_id"]: row for row in candidates}
    pilot_summary_path = _regular_file(pilot / "summary.json")
    if file_sha256(pilot_summary_path) != PILOT_SUMMARY_SHA256:
        raise FreshDevSelectionError("pilot_summary_sha256_mismatch")
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    if (
        (pilot_summary.get("molecule_split_counts") or {}).get("train") != 2250
        or (pilot_summary.get("sample_split_counts") or {}).get("train") != 9000
    ):
        raise FreshDevSelectionError("pilot_summary_train_inventory")
    pilot_shards = sorted((pilot / "shards").glob("train-*.tar"))
    if [path.name for path in pilot_shards] != sorted(PILOT_TRAIN_SHARD_SHA256):
        raise FreshDevSelectionError("pilot_train_shard_inventory")
    pilot_rows, pilot_records = _load_exposure_rows_from_shards(
        pilot_shards,
        relative_to=pilot,
        expected_layer_splits={"train"},
        expected_sha256=PILOT_TRAIN_SHARD_SHA256,
    )
    if len(pilot_rows) != 2250 or sum(
        int(item["chemmol_rows"]) for item in pilot_records
    ) != 9000:
        raise FreshDevSelectionError("pilot_train_exposure_inventory")
    smoke_rows, smoke_record = _load_smoke_exposure_rows(smoke)
    normalized_pilot_rows = []
    normalized_smoke_rows = []
    for exposure, destination in (
        *((row, normalized_pilot_rows) for row in pilot_rows),
        *((row, normalized_smoke_rows) for row in smoke_rows),
    ):
        source_row = by_molecule.get(exposure["molecule_id"])
        if source_row is None:
            destination.append(exposure)
            continue
        for field in ("molecule_id", "inchikey_full", "source_record_id"):
            if source_row[field] != exposure[field]:
                raise FreshDevSelectionError(
                    f"exposure_source_identity_mismatch:{exposure['molecule_id']}:{field}"
                )
        destination.append(source_row)
    pilot_rows = normalized_pilot_rows
    smoke_rows = normalized_smoke_rows
    exposure_sets = _identity_sets([*pilot_rows, *smoke_rows])
    eligible_with_aliases = [
        row for row in candidates if not _overlaps_any(row, exposure_sets)
    ]
    eligible_with_aliases.sort(
        key=lambda row: (row["rank_sha256"], row["molecule_id"])
    )
    eligible = []
    seen_inchikeys = set()
    for row in eligible_with_aliases:
        if row["inchikey_full"] in seen_inchikeys:
            continue
        seen_inchikeys.add(row["inchikey_full"])
        eligible.append(row)
    selection_scan_limit = min(len(eligible), 20000)
    selection_scan = eligible[:selection_scan_limit]
    original_rows = load_original_sample_rows(source, selection_scan)

    fresh: list[dict[str, str]] = []
    replacement: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    fresh_sets = {field: set() for field in IDENTITY_FIELDS}
    reader = TrainChemmolReader(source)
    try:
        for row in selection_scan:
            if len(fresh) >= FRESH_COUNT and len(replacement) >= REPLACEMENT_COUNT:
                break
            if len(fresh) >= FRESH_COUNT and _overlaps_any(row, fresh_sets):
                continue
            try:
                precheck = _candidate_precheck(
                    source,
                    row,
                    original_rows[row["molecule_id"]],
                    reader,
                )
            except Exception as error:
                rejected.append(
                    {
                        "rank_sha256": row["rank_sha256"],
                        "molecule_id": row["molecule_id"],
                        "stage": "structure_precheck",
                        "error_code": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                continue
            selected_row = dict(row)
            selected_row.update(
                {
                    "source_sdf_sha256": str(precheck["sdf_sha256"]),
                    "source_sdf_bytes": str(precheck["sdf_bytes"]),
                    "source_chemmol_sha256": str(precheck["chemmol_sha256"]),
                    "source_atom_count": str(precheck["atoms"]),
                    "source_bond_count": str(precheck["bonds"]),
                }
            )
            if len(fresh) < FRESH_COUNT:
                selected_row["selection_index"] = str(len(fresh))
                fresh.append(selected_row)
                for field in IDENTITY_FIELDS:
                    fresh_sets[field].add(row[field])
            else:
                selected_row["replacement_index"] = str(len(replacement))
                selected_row["purpose"] = "fresh_dev_pre_render_replacement"
                selected_row["rendered"] = "false"
                replacement.append(selected_row)
                for field in IDENTITY_FIELDS:
                    fresh_sets[field].add(row[field])
    finally:
        reader.close()
    if len(fresh) != FRESH_COUNT or len(replacement) != REPLACEMENT_COUNT:
        raise FreshDevSelectionError(
            f"selection_capacity:{len(fresh)}:{len(replacement)}"
        )

    _prepare_output(output, overwrite=args.overwrite)
    _write_csv(output / "selection.committed.csv", fresh, SELECTION_FIELDS)
    _write_csv(
        output / "replacement_pool.committed.csv",
        replacement,
        REPLACEMENT_FIELDS,
    )
    _write_csv(
        output / "candidate_rejected.csv",
        rejected,
        ("rank_sha256", "molecule_id", "stage", "error_code", "error_message"),
    )

    selector_path = Path(__file__).resolve()
    scaffold_algorithm = Path(__file__).resolve().parent / "build_webdataset_sample.py"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "status": "selected_pending_independent_identity_audit",
        "freeze_id": FREEZE_ID,
        "freeze_sha256": file_sha256(freeze_path),
        "source_dataset": str(source),
        "source_split": "train",
        "source_train_manifest_count": len(source_manifest_records),
        "source_train_manifest_inventory_count": 260,
        "source_train_manifest_inventory_sha256": TRAIN_MANIFEST_INVENTORY_SHA256,
        "source_train_manifests": source_manifest_records,
        "source_aggregate_validation": {
            "path": str(aggregate_validation),
            "sha256": file_sha256(aggregate_validation),
            "status": aggregate["status"],
            "scaffold_overlap": aggregate["scaffold_overlap"],
        },
        "candidate_count": len(candidates),
        "eligible_before_inchikey_alias_dedup": len(eligible_with_aliases),
        "eligible_after_exposure_exclusion": len(eligible),
        "inchikey_aliases_deduplicated": (
            len(eligible_with_aliases) - len(eligible)
        ),
        "selection_scan_limit": selection_scan_limit,
        "fresh_dev_molecules": len(fresh),
        "fresh_dev_samples_planned": len(fresh) * 4,
        "fresh_replacement_molecules": len(replacement),
        "fresh_replacement_rendered": 0,
        "structural_rejected": len(rejected),
        "sealed_rows_materialized": False,
        "reserve_rows_materialized": False,
        "pilot_train_exposure": {
            "molecules": len(pilot_rows),
            "shards": pilot_records,
        },
        "smoke_train_32_exposure": smoke_record,
        "exclusion_set_sha256": {
            field: _set_sha256(exposure_sets[field]) for field in IDENTITY_FIELDS
        },
        "selection_manifest_sha256": file_sha256(
            output / "selection.committed.csv"
        ),
        "selection_source_bytes_commitment_sha256": (
            _source_bytes_commitment_sha256(fresh)
        ),
        "replacement_manifest_sha256": file_sha256(
            output / "replacement_pool.committed.csv"
        ),
        "candidate_rejected_sha256": file_sha256(
            output / "candidate_rejected.csv"
        ),
        "selector_path": str(selector_path),
        "selector_sha256": file_sha256(selector_path),
        "scaffold_algorithm_path": str(scaffold_algorithm),
        "scaffold_algorithm_sha256": file_sha256(scaffold_algorithm),
        "source_record_normalization": "source + ':' + source_id",
        "source_record_normalization_sha256": sha256_bytes(
            b"source-record-v0.1:source + ':' + source_id"
        ),
        "selection_config": {
            "ordering": "sha256_rank_v0.1",
            "fresh_count": FRESH_COUNT,
            "replacement_count": REPLACEMENT_COUNT,
            "selection_scan_limit": selection_scan_limit,
            "exclusion_match_policy": "reject_candidate_on_any_key_match",
            "identity_fields": list(IDENTITY_FIELDS),
        },
    }
    provenance["selection_config_sha256"] = sha256_bytes(
        canonical_json_bytes(provenance["selection_config"])
    )
    (output / "selection_provenance.json").write_bytes(
        canonical_json_bytes(provenance, newline=True)
    )
    return {"output": str(output), **provenance}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    version_root = root / "visual_svg_layers_c_v0_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(root / "webdataset_full_v0"))
    parser.add_argument(
        "--pilot", default=str(root / "chemvision_layers_v0_1/pilot_10k")
    )
    parser.add_argument(
        "--smoke", default=str(version_root / "smoke_train_32")
    )
    parser.add_argument(
        "--freeze", default=str(version_root / "phase_a_freeze.json")
    )
    parser.add_argument("--output", default=str(version_root / "fresh_dev"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = select(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "FreshDevSelectionError",
    "IDENTITY_FIELDS",
    "REPLACEMENT_COUNT",
    "REPLACEMENT_FIELDS",
    "SCHEMA_VERSION",
    "build_parser",
    "load_train_candidates",
    "main",
    "select",
]
