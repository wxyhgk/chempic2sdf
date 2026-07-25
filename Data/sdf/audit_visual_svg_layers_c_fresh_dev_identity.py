#!/usr/bin/env python3
"""Independently audit and commit the Phase-A fresh-dev identity selection.

The auditor reconstructs candidates and exposed identities from explicit train
artifacts.  It does not import the selector, read forbidden split rows, select
sealed/reserve identities, render images, or run a model.
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
from rdkit import Chem

try:
    from .build_visual_svg_layers_c_fresh_dev_preflight import (
        IDENTITY_AUDIT_SCHEMA_VERSION,
        SELECTION_FIELDS,
        SELECTION_SCHEMA_VERSION,
    )
    from .build_webdataset_sample import mol_from_chemmol, scaffold_smiles
except ImportError:
    from build_visual_svg_layers_c_fresh_dev_preflight import (
        IDENTITY_AUDIT_SCHEMA_VERSION,
        SELECTION_FIELDS,
        SELECTION_SCHEMA_VERSION,
    )
    from build_webdataset_sample import mol_from_chemmol, scaffold_smiles


SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-identity-audit-run-0.1"
FREEZE_ID = "visual-svg-layers-c-v0.1-phase-a-20260722"
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
IDENTITY_FIELDS = (
    "molecule_id",
    "inchikey_full",
    "scaffold_id",
    "source_record_id",
)
REPLACEMENT_FIELDS = {
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
}
FORBIDDEN_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "reserve",
    "mr_tadf",
    "mr-tadf",
)


class IdentityAuditError(ValueError):
    """Raised when the selected identities fail independent isolation audit."""


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


def _forbidden_marker(value: object) -> str | None:
    lowered = str(value).lower()
    return next((item for item in FORBIDDEN_MARKERS if item in lowered), None)


def _assert_allowed(value: object, *, context: str) -> None:
    marker = _forbidden_marker(value)
    if marker is not None:
        raise IdentityAuditError(f"forbidden_reference:{context}:{marker}")


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
            raise IdentityAuditError(f"missing_input:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise IdentityAuditError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise IdentityAuditError(f"input_not_regular:{absolute}")
    return absolute


def _read_csv(path: Path, required: set[str], *, label: str) -> list[dict[str, str]]:
    with _regular_file(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise IdentityAuditError(f"{label}_missing_fields:{','.join(missing)}")
        rows = list(reader)
    for row_number, row in enumerate(rows, 2):
        for field in required:
            if not str(row.get(field, "")).strip():
                raise IdentityAuditError(f"{label}_empty:{row_number}:{field}")
        for field, value in row.items():
            _assert_allowed(value, context=f"{label}:{row_number}:{field}")
    return rows


def _validate_frozen_source(source: Path) -> tuple[Path, dict[str, Any]]:
    expected = (Path(__file__).resolve().parent / "webdataset_full_v0").resolve()
    if source != expected:
        raise IdentityAuditError(f"source_root_not_frozen:{source}:{expected}")
    aggregate_path = _regular_file(
        source / "validation.json", allow_aggregate_validation=True
    )
    if _file_sha256(aggregate_path) != SOURCE_VALIDATION_SHA256:
        raise IdentityAuditError("source_validation_sha256")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("status") != "passed" or aggregate.get("scaffold_overlap") != 0:
        raise IdentityAuditError("source_aggregate_validation")
    paths = sorted(
        [
            *(source / "manifests/shards").glob("train-*.molecules.csv"),
            *(source / "manifests/shards").glob("train-*.samples.csv"),
        ]
    )
    if len(paths) != 260:
        raise IdentityAuditError("train_manifest_inventory_count")
    digest = hashlib.sha256()
    for path in paths:
        path = _regular_file(path)
        relative = path.relative_to(source).as_posix()
        digest.update(f"{_file_sha256(path)}  {relative}\n".encode("utf-8"))
    if digest.hexdigest() != TRAIN_MANIFEST_INVENTORY_SHA256:
        raise IdentityAuditError("train_manifest_inventory_sha256")
    return aggregate_path, aggregate


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
    return _sha256_bytes(_canonical_json(payload))


def _identity_sets(rows: Sequence[Mapping[str, str]]) -> dict[str, set[str]]:
    return {field: {row[field] for row in rows} for field in IDENTITY_FIELDS}


def _overlap_counts(
    left: Mapping[str, set[str]], right: Mapping[str, set[str]]
) -> dict[str, int]:
    return {
        "molecule_id_overlap": len(left["molecule_id"] & right["molecule_id"]),
        "inchikey_full_overlap": len(
            left["inchikey_full"] & right["inchikey_full"]
        ),
        "scaffold_id_overlap": len(left["scaffold_id"] & right["scaffold_id"]),
        "source_record_id_overlap": len(
            left["source_record_id"] & right["source_record_id"]
        ),
    }


def _overlaps_any(row: Mapping[str, str], sets: Mapping[str, set[str]]) -> bool:
    return any(row[field] in sets[field] for field in IDENTITY_FIELDS)


def _load_train_candidates(source: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manifests = sorted((source / "manifests/shards").glob("train-*.molecules.csv"))
    if not manifests:
        raise IdentityAuditError("train_manifests_missing")
    candidates = []
    records = []
    for manifest in manifests:
        manifest = _regular_file(manifest)
        stem = manifest.name.removesuffix(".molecules.csv")
        row_count = 0
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
                raise IdentityAuditError(f"source_manifest_fields:{manifest.name}")
            for source_row in reader:
                row_count += 1
                if (
                    source_row["split"] != "train"
                    or source_row["shard"] != f"shards/{stem}.tar"
                    or not source_row["structure_ref"].startswith(
                        f"structures/{stem}/"
                    )
                    or source_row["molecule_id"]
                    != f"{source_row['source']}:{source_row['source_id']}"
                ):
                    raise IdentityAuditError(
                        f"source_manifest_row:{manifest.name}:{row_count + 1}"
                    )
                row = {
                    "selection_index": "",
                    "rank_sha256": "",
                    "molecule_id": source_row["molecule_id"],
                    "inchikey_full": source_row["inchikey"],
                    "scaffold_id": source_row["scaffold"],
                    "source_record_id": (
                        f"{source_row['source']}:{source_row['source_id']}"
                    ),
                    "source": source_row["source"],
                    "source_id": source_row["source_id"],
                    "source_split": "train",
                    "source_shard": source_row["shard"],
                    "source_structure": source_row["structure_ref"],
                }
                row["rank_sha256"] = _selection_rank(row)
                candidates.append(row)
        records.append(
            {
                "path": manifest.relative_to(source).as_posix(),
                "sha256": _file_sha256(manifest),
                "rows": row_count,
            }
        )
    for field in ("molecule_id", "source_record_id"):
        values = [row[field] for row in candidates]
        if len(values) != len(set(values)):
            raise IdentityAuditError(f"source_duplicate:{field}")
    return candidates, records


def _identity_from_chemmol(chemmol: Mapping[str, Any]) -> dict[str, str]:
    molecule_id = str(chemmol["molecule_id"])
    if ":" not in molecule_id:
        raise IdentityAuditError(f"exposure_molecule_id:{molecule_id}")
    source, source_id = molecule_id.split(":", 1)
    molecule = mol_from_chemmol(dict(chemmol))
    Chem.SanitizeMol(molecule)
    return {
        "selection_index": "",
        "rank_sha256": "",
        "molecule_id": molecule_id,
        "inchikey_full": str(chemmol["inchikey"]),
        "scaffold_id": scaffold_smiles(
            molecule, str(chemmol["canonical_isomeric_smiles"])
        ),
        "source_record_id": molecule_id,
        "source": source,
        "source_id": source_id,
        "source_split": "train",
        "source_shard": "exposure-only",
        "source_structure": str(chemmol.get("structure_ref", "")),
    }


def _exposure_rows_from_shards(
    shards: Sequence[Path],
    *,
    relative_to: Path,
    expected_splits: set[str],
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    identities: dict[str, dict[str, str]] = {}
    records = []
    if not shards:
        raise IdentityAuditError("exposure_shards_missing")
    for shard in shards:
        shard = _regular_file(shard)
        if expected_sha256 is not None:
            expected = expected_sha256.get(shard.name)
            if expected is None or _file_sha256(shard) != expected:
                raise IdentityAuditError(
                    f"exposure_shard_inventory_or_sha256:{shard.name}"
                )
        layer_splits: dict[str, str] = {}
        count = 0
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if member.name.endswith(".layers.json"):
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise IdentityAuditError(f"exposure_layer:{member.name}")
                    layers = json.load(extracted)
                    split = str(layers.get("split", ""))
                    if split not in expected_splits:
                        raise IdentityAuditError(f"exposure_layer_split:{split}")
                    source = layers.get("source") or {}
                    if not str(source.get("source_shard", "")).startswith(
                        "shards/train-"
                    ):
                        raise IdentityAuditError("exposure_source_split")
                    layer_splits[member.name.removesuffix(".layers.json")] = split
                elif member.name.endswith(".chemmol.json"):
                    key = member.name.removesuffix(".chemmol.json")
                    if key not in layer_splits:
                        raise IdentityAuditError("exposure_chemmol_before_layer")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise IdentityAuditError(f"exposure_chemmol:{member.name}")
                    identity = _identity_from_chemmol(json.load(extracted))
                    previous = identities.get(identity["molecule_id"])
                    if previous is not None and any(
                        previous[field] != identity[field]
                        for field in IDENTITY_FIELDS
                    ):
                        raise IdentityAuditError("exposure_identity_drift")
                    identities[identity["molecule_id"]] = identity
                    count += 1
        records.append(
            {
                "path": shard.relative_to(relative_to).as_posix(),
                "sha256": _file_sha256(shard),
                "chemmol_rows": count,
            }
        )
    if not identities:
        raise IdentityAuditError("pilot_exposure_empty")
    return list(identities.values()), records


def _smoke_rows(smoke: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = _regular_file(smoke / "molecules.csv")
    if _file_sha256(manifest) != SMOKE_MOLECULES_SHA256:
        raise IdentityAuditError("smoke_molecules_sha256")
    commitment_path = _regular_file(smoke / "commitment.smoke.json")
    if _file_sha256(commitment_path) != SMOKE_COMMITMENT_SHA256:
        raise IdentityAuditError("smoke_commitment_sha256")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if (
        commitment.get("role") != "smoke"
        or commitment.get("fresh_audit_eligible") is not False
        or commitment.get("molecule_count") != 8
        or commitment.get("sample_count") != 32
        or commitment.get("members_per_sample") != 12
        or len(commitment.get("shards") or ()) != 1
    ):
        raise IdentityAuditError("smoke_commitment_contract")
    paired_ref = commitment.get("paired_manifest") or {}
    paired = _regular_file(smoke / str(paired_ref.get("path", "")))
    if paired_ref.get("sha256") != _file_sha256(paired):
        raise IdentityAuditError("smoke_paired_sha256")
    shard_ref = commitment["shards"][0]
    committed_shard = _regular_file(smoke / str(shard_ref.get("path", "")))
    if (
        shard_ref.get("sha256") != _file_sha256(committed_shard)
        or shard_ref.get("samples") != 32
        or shard_ref.get("members") != 384
    ):
        raise IdentityAuditError("smoke_shard_commitment")
    rows = _read_csv(
        manifest, {"molecule_id", "split"}, label="smoke_molecules"
    )
    if len(rows) != 8 or any(row["split"] != "smoke_train" for row in rows):
        raise IdentityAuditError("smoke_split")
    exposure_rows, shard_records = _exposure_rows_from_shards(
        sorted((smoke / "shards").glob("smoke-train-*.tar")),
        relative_to=smoke,
        expected_splits={"train", "smoke_train"},
    )
    if {row["molecule_id"] for row in exposure_rows} != {
        row["molecule_id"] for row in rows
    }:
        raise IdentityAuditError("smoke_manifest_shard_identity")
    if sum(int(record["chemmol_rows"]) for record in shard_records) != 32:
        raise IdentityAuditError("smoke_sample_inventory")
    return exposure_rows, {
        "path": manifest.relative_to(smoke).as_posix(),
        "sha256": _file_sha256(manifest),
        "rows": len(rows),
        "shards": shard_records,
    }


def _original_sample_rows(
    source: Path, selected: Sequence[Mapping[str, str]]
) -> dict[str, dict[str, str]]:
    wanted_by_manifest: dict[Path, set[str]] = defaultdict(set)
    for row in selected:
        stem = Path(row["source_shard"]).stem
        manifest = _regular_file(
            source / "manifests/shards" / f"{stem}.samples.csv"
        )
        wanted_by_manifest[manifest].add(row["molecule_id"])
    result = {}
    for manifest, wanted in wanted_by_manifest.items():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("molecule_id") not in wanted or row.get("variant") != "original":
                    continue
                if row.get("split") != "train" or not row.get("shard", "").startswith(
                    "shards/train-"
                ):
                    raise IdentityAuditError("source_sample_not_train")
                result[row["molecule_id"]] = row
    if set(result) != {row["molecule_id"] for row in selected}:
        raise IdentityAuditError("source_original_sample_set")
    return result


class _ChemmolReader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.archives: dict[str, tarfile.TarFile] = {}

    def read(self, row: Mapping[str, str]) -> bytes:
        shard_name = row["shard"]
        archive = self.archives.get(shard_name)
        if archive is None:
            archive = tarfile.open(_regular_file(self.source / shard_name), "r")
            self.archives[shard_name] = archive
        name = row["sample_key"] + ".chemmol.json"
        try:
            member = archive.getmember(name)
        except KeyError as error:
            raise IdentityAuditError(f"chemmol_missing:{name}") from error
        extracted = archive.extractfile(member)
        if extracted is None:
            raise IdentityAuditError(f"chemmol_unreadable:{name}")
        return extracted.read()

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()


def _enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def _validate_structure(
    source: Path,
    identity: Mapping[str, str],
    sample_row: Mapping[str, str],
    reader: _ChemmolReader,
    chemmol_validator,
) -> None:
    structure = identity["source_structure"]
    if not structure.startswith("structures/train-"):
        raise IdentityAuditError("structure_not_train")
    sdf_bytes = _regular_file(source / structure).read_bytes()
    if _sha256_bytes(sdf_bytes) != identity["source_sdf_sha256"]:
        raise IdentityAuditError("committed_source_sdf_sha256")
    if len(sdf_bytes) != int(identity["source_sdf_bytes"]):
        raise IdentityAuditError("committed_source_sdf_bytes")
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(sdf_bytes), sanitize=True, removeHs=False, strictParsing=True
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise IdentityAuditError("sdf_single_record")
    sdf = molecules[0]
    Chem.SanitizeMol(sdf)
    Chem.AssignStereochemistry(sdf, cleanIt=False, force=True)
    if Chem.MolToInchiKey(sdf) != identity["inchikey_full"]:
        raise IdentityAuditError("sdf_inchikey")
    chemmol_bytes = reader.read(sample_row)
    if _sha256_bytes(chemmol_bytes) != identity["source_chemmol_sha256"]:
        raise IdentityAuditError("committed_source_chemmol_sha256")
    chemmol = json.loads(chemmol_bytes)
    chemmol_validator(chemmol)
    if (
        chemmol.get("molecule_id") != identity["molecule_id"]
        or chemmol.get("inchikey") != identity["inchikey_full"]
    ):
        raise IdentityAuditError("chemmol_identity")
    reconstructed = mol_from_chemmol(chemmol)
    Chem.SanitizeMol(reconstructed)
    if Chem.MolToInchiKey(reconstructed) != identity["inchikey_full"]:
        raise IdentityAuditError("chemmol_reconstructed_inchikey")
    atoms = chemmol["atoms"]
    bonds = chemmol["bonds"]
    if len(atoms) != sdf.GetNumAtoms() or len(bonds) != sdf.GetNumBonds():
        raise IdentityAuditError("chemmol_sdf_count")
    if (
        len(atoms) != int(identity["source_atom_count"])
        or len(bonds) != int(identity["source_bond_count"])
    ):
        raise IdentityAuditError("committed_atom_bond_count")
    if [int(atom["id"]) for atom in atoms] != list(range(len(atoms))):
        raise IdentityAuditError("atom_ids_not_contiguous")
    if [int(bond["id"]) for bond in bonds] != list(range(len(bonds))):
        raise IdentityAuditError("bond_ids_not_contiguous")
    source_atom_indices = set()
    for atom_data in atoms:
        atom_id = int(atom_data["id"])
        source_index = int(atom_data["source_atom_index"])
        if source_index in source_atom_indices:
            raise IdentityAuditError("source_atom_index_duplicate")
        source_atom_indices.add(source_index)
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
            raise IdentityAuditError(f"atom_chemistry:{atom_id}")
    if source_atom_indices != set(range(len(atoms))):
        raise IdentityAuditError("source_atom_index_not_full_permutation")
    for bond_data in bonds:
        bond = sdf.GetBondBetweenAtoms(
            int(bond_data["begin_atom"]), int(bond_data["end_atom"])
        )
        if bond is None:
            raise IdentityAuditError("bond_endpoint")
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
            raise IdentityAuditError(f"bond_chemistry:{bond_data['id']}")


def _same_identity_rows(
    actual: Sequence[Mapping[str, str]], expected: Sequence[Mapping[str, str]]
) -> bool:
    if len(actual) != len(expected):
        return False
    fields = (
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
    )
    return all(
        all(left[field] == right[field] for field in fields)
        for left, right in zip(actual, expected)
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    pilot = Path(args.pilot).expanduser().resolve()
    smoke = Path(args.smoke).expanduser().resolve()
    freeze_path = Path(args.freeze).expanduser().resolve()
    fresh_root = Path(args.fresh_root).expanduser().resolve()
    for value, context in (
        (source, "source"),
        (pilot, "pilot"),
        (smoke, "smoke"),
        (fresh_root, "fresh_root"),
    ):
        _assert_allowed(value, context=context)
    freeze_path = _regular_file(freeze_path)
    if _file_sha256(freeze_path) != FREEZE_SHA256:
        raise IdentityAuditError("phase_a_freeze_sha256_mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("freeze_id") != FREEZE_ID
        or freeze.get("status") != "frozen_fresh_dev_preflight_authorized"
        or freeze.get("training_steps") != 0
        or freeze.get("parameter_updates") != 0
    ):
        raise IdentityAuditError("freeze_contract")
    aggregate_path, aggregate = _validate_frozen_source(source)

    selection_path = _regular_file(fresh_root / "selection.committed.csv")
    replacement_path = _regular_file(
        fresh_root / "replacement_pool.committed.csv"
    )
    provenance_path = _regular_file(fresh_root / "selection_provenance.json")
    rejected_path = _regular_file(fresh_root / "candidate_rejected.csv")
    selection = _read_csv(
        selection_path, set(SELECTION_FIELDS), label="fresh_selection"
    )
    replacement = _read_csv(
        replacement_path, REPLACEMENT_FIELDS, label="fresh_replacement"
    )
    with rejected_path.open(newline="", encoding="utf-8") as handle:
        rejected = list(csv.DictReader(handle))
    if rejected:
        raise IdentityAuditError(
            "selector_has_structural_rejections_independent_replay_required"
        )
    if len(selection) != FRESH_COUNT or len(replacement) != REPLACEMENT_COUNT:
        raise IdentityAuditError("selected_capacity")
    if [int(row["selection_index"]) for row in selection] != list(range(FRESH_COUNT)):
        raise IdentityAuditError("selection_index")
    if [int(row["replacement_index"]) for row in replacement] != list(
        range(REPLACEMENT_COUNT)
    ):
        raise IdentityAuditError("replacement_index")
    if any(
        row["purpose"] != "fresh_dev_pre_render_replacement"
        or row["rendered"] != "false"
        for row in replacement
    ):
        raise IdentityAuditError("replacement_purpose_or_rendered")
    if any(row["rank_sha256"] != _selection_rank(row) for row in selection):
        raise IdentityAuditError("selection_rank")
    if any(row["rank_sha256"] != _selection_rank(row) for row in replacement):
        raise IdentityAuditError("replacement_rank")

    candidates, source_manifest_records = _load_train_candidates(source)
    by_molecule = {row["molecule_id"]: row for row in candidates}
    pilot_summary_path = _regular_file(pilot / "summary.json")
    if _file_sha256(pilot_summary_path) != PILOT_SUMMARY_SHA256:
        raise IdentityAuditError("pilot_summary_sha256")
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    if (
        (pilot_summary.get("molecule_split_counts") or {}).get("train") != 2250
        or (pilot_summary.get("sample_split_counts") or {}).get("train") != 9000
    ):
        raise IdentityAuditError("pilot_summary_inventory")
    pilot_shards = sorted((pilot / "shards").glob("train-*.tar"))
    if [path.name for path in pilot_shards] != sorted(PILOT_TRAIN_SHARD_SHA256):
        raise IdentityAuditError("pilot_train_shard_inventory")
    pilot_rows, pilot_records = _exposure_rows_from_shards(
        pilot_shards,
        relative_to=pilot,
        expected_splits={"train"},
        expected_sha256=PILOT_TRAIN_SHARD_SHA256,
    )
    if len(pilot_rows) != 2250 or sum(
        int(record["chemmol_rows"]) for record in pilot_records
    ) != 9000:
        raise IdentityAuditError("pilot_train_exposure_inventory")
    smoke_rows, smoke_record = _smoke_rows(smoke)
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
                raise IdentityAuditError(
                    f"exposure_source_identity_mismatch:{exposure['molecule_id']}:{field}"
                )
        destination.append(source_row)
    pilot_rows = normalized_pilot_rows
    smoke_rows = normalized_smoke_rows
    pilot_sets = _identity_sets(pilot_rows)
    smoke_sets = _identity_sets(smoke_rows)
    exposure_sets = {
        field: pilot_sets[field] | smoke_sets[field] for field in IDENTITY_FIELDS
    }
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
    expected_fresh = eligible[:FRESH_COUNT]
    if not _same_identity_rows(selection, expected_fresh):
        raise IdentityAuditError("fresh_selection_not_deterministic_prefix")
    fresh_sets = _identity_sets(selection)
    allocated_sets = {field: set(values) for field, values in fresh_sets.items()}
    expected_replacement = []
    for row in eligible[FRESH_COUNT:]:
        if _overlaps_any(row, allocated_sets):
            continue
        expected_replacement.append(row)
        for field in IDENTITY_FIELDS:
            allocated_sets[field].add(row[field])
        if len(expected_replacement) == REPLACEMENT_COUNT:
            break
    if not _same_identity_rows(replacement, expected_replacement):
        raise IdentityAuditError("replacement_not_deterministic_prefix")

    selection_sets = _identity_sets(selection)
    replacement_sets = _identity_sets(replacement)
    comparisons = {
        "frozen_b_p0_train_exposure": _overlap_counts(
            selection_sets, pilot_sets
        ),
        "smoke_train_32": _overlap_counts(selection_sets, smoke_sets),
    }
    if any(value for result in comparisons.values() for value in result.values()):
        raise IdentityAuditError("fresh_exposure_overlap")
    allocation_overlap = _overlap_counts(selection_sets, replacement_sets)
    if any(allocation_overlap.values()):
        raise IdentityAuditError("fresh_replacement_overlap")
    for field in IDENTITY_FIELDS:
        values = [row[field] for row in replacement]
        if len(values) != len(set(values)):
            raise IdentityAuditError(f"replacement_internal_duplicate:{field}")

    original_rows = _original_sample_rows(source, [*selection, *replacement])
    chemmol_schema = json.loads(
        (Path(__file__).resolve().parent / "schemas/chemmol-0.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    chemmol_validator = fastjsonschema.compile(chemmol_schema)
    reader = _ChemmolReader(source)
    checked = 0
    try:
        for row in [*selection, *replacement]:
            _validate_structure(
                source,
                row,
                original_rows[row["molecule_id"]],
                reader,
                chemmol_validator,
            )
            checked += 1
    finally:
        reader.close()

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("status")
        != "selected_pending_independent_identity_audit"
        or provenance.get("selection_manifest_sha256")
        != _file_sha256(selection_path)
        or provenance.get("selection_source_bytes_commitment_sha256")
        != _source_bytes_commitment_sha256(selection)
        or provenance.get("replacement_manifest_sha256")
        != _file_sha256(replacement_path)
        or provenance.get("source_train_manifest_inventory_sha256")
        != TRAIN_MANIFEST_INVENTORY_SHA256
        or provenance.get("sealed_rows_materialized") is not False
        or provenance.get("reserve_rows_materialized") is not False
    ):
        raise IdentityAuditError("selection_provenance_contract")

    union_exposure_hashes = {
        field: _set_sha256(exposure_sets[field]) for field in IDENTITY_FIELDS
    }
    auditor_path = Path(__file__).resolve()
    scaffold_algorithm = Path(__file__).resolve().parent / "build_webdataset_sample.py"
    style_algorithm = (
        Path(__file__).resolve().parent
        / "build_visual_svg_layers_c_fresh_dev_preflight.py"
    )
    config = {
        "freeze_id": FREEZE_ID,
        "source_split": "train",
        "fresh_count": FRESH_COUNT,
        "replacement_count": REPLACEMENT_COUNT,
        "selection_order": "sha256_rank_v0.1",
        "identity_fields": list(IDENTITY_FIELDS),
        "match_policy": "reject_candidate_on_any_key_match",
        "structural_rejections_allowed_for_pass": False,
    }
    exposure_manifest_hash = hashlib.sha256()
    for record in [*source_manifest_records, *pilot_records, smoke_record]:
        exposure_manifest_hash.update(str(record["sha256"]).encode("ascii"))
    report = {
        "schema_version": IDENTITY_AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "producer": "independent_identity_auditor",
        "freeze_id": FREEZE_ID,
        "fresh_dev_selection_manifest_sha256": _file_sha256(selection_path),
        "fresh_dev_source_bytes_commitment_sha256": (
            _source_bytes_commitment_sha256(selection)
        ),
        "fresh_dev_replacement_manifest_sha256": _file_sha256(replacement_path),
        "selection_provenance_sha256": _file_sha256(provenance_path),
        "source_aggregate_validation_sha256": _file_sha256(aggregate_path),
        "source_train_manifest_count": len(source_manifest_records),
        "source_train_manifest_inventory_count": 260,
        "source_train_manifest_inventory_sha256": TRAIN_MANIFEST_INVENTORY_SHA256,
        "exposure_manifest_sha256": exposure_manifest_hash.hexdigest(),
        "molecule_id_exclusion_set_sha256": union_exposure_hashes["molecule_id"],
        "inchikey_full_exclusion_set_sha256": union_exposure_hashes[
            "inchikey_full"
        ],
        "scaffold_id_exclusion_set_sha256": union_exposure_hashes["scaffold_id"],
        "source_record_id_exclusion_set_sha256": union_exposure_hashes[
            "source_record_id"
        ],
        "isolation_algorithm_sha256": _file_sha256(auditor_path),
        "isolation_config_sha256": _sha256_bytes(_canonical_json(config)),
        "scaffold_id_algorithm_sha256": _file_sha256(scaffold_algorithm),
        "source_record_normalization_algorithm_sha256": _sha256_bytes(
            b"source-record-v0.1:source + ':' + source_id"
        ),
        "style_id_algorithm_sha256": _file_sha256(style_algorithm),
        "comparisons": comparisons,
        "fresh_replacement_overlap": allocation_overlap,
        "fresh_dev_molecules": len(selection),
        "fresh_replacement_molecules": len(replacement),
        "structure_prechecked_molecules": checked,
        "structural_rejections": 0,
        "old_validation_internal_or_sealed_rows_read": False,
        "sealed_rows_materialized": False,
        "reserve_rows_materialized": False,
        "source_train_manifests": source_manifest_records,
        "pilot_train_exposure_shards": pilot_records,
        "smoke_train_32_exposure_manifest": smoke_record,
        "config": config,
    }
    report_path = fresh_root / "identity_isolation_report.json"
    report_path.write_bytes(_canonical_json(report, newline=True))

    commitment = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "committed",
        "freeze_id": FREEZE_ID,
        "role": "dev",
        "split": "fresh_dev",
        "molecule_count": FRESH_COUNT,
        "sample_count": FRESH_COUNT * 4,
        "selection_ordering": "sha256_rank_v0.1",
        "freeze": {
            "path": "../phase_a_freeze.json",
            "sha256": _file_sha256(freeze_path),
        },
        "selection_manifest": {
            "path": "selection.committed.csv",
            "sha256": _file_sha256(selection_path),
        },
        "source_bytes_commitment": {
            "protocol": "source-sdf-byte-preserving-0.1",
            "sha256": _source_bytes_commitment_sha256(selection),
            "molecule_count": FRESH_COUNT,
        },
        "fresh_replacement_manifest": {
            "path": "replacement_pool.committed.csv",
            "sha256": _file_sha256(replacement_path),
            "molecule_count": REPLACEMENT_COUNT,
            "rendered": False,
        },
        "identity_isolation_report": {
            "path": "identity_isolation_report.json",
            "sha256": _file_sha256(report_path),
            "status": "passed",
        },
        "selection_provenance": {
            "path": "selection_provenance.json",
            "sha256": _file_sha256(provenance_path),
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
    commitment_path = fresh_root / "selection.commitment.json"
    commitment_path.write_bytes(_canonical_json(commitment, newline=True))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "IDENTITY-AUDIT-PASS",
        "fresh_root": str(fresh_root),
        "selection_commitment": str(commitment_path),
        "selection_commitment_sha256": _file_sha256(commitment_path),
        "identity_isolation_report": str(report_path),
        "identity_isolation_report_sha256": _file_sha256(report_path),
        "fresh_dev_molecules": len(selection),
        "fresh_replacement_molecules": len(replacement),
        "structure_prechecked_molecules": checked,
        "comparisons": comparisons,
        "fresh_replacement_overlap": allocation_overlap,
        "sealed_rows_materialized": False,
        "reserve_rows_materialized": False,
    }


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
    parser.add_argument("--fresh-root", default=str(version_root / "fresh_dev"))
    return parser


def main() -> None:
    result = audit(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "IdentityAuditError",
    "SCHEMA_VERSION",
    "audit",
    "build_parser",
    "main",
]
