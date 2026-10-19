#!/usr/bin/env python3
"""Independently audit and commit the frozen ChemTraverse v0.1 manifests.

The auditor is intentionally limited to the public style-train and fresh-dev
WebDataset shards. It never discovers or opens sealed/route-confirm paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
AB_DIR = ROOT / "chemtraverse_v0_1_ab_50k"
CAPACITY_DIR = ROOT / "chemtraverse_v0_1_capacity"
FRESH_DIR = ROOT / "chemtraverse_v0_1_fresh_dev"
TRAIN_SHARD_DIR = ROOT / "chemvector_joint_v1/style_train_100k/shards"
FRESH_SHARD_DIR = ROOT / "chemvector_joint_v1/fresh_eval_12k/shards"

VARIANTS = frozenset({"original", "rdkit", "coordgen", "stress"})
REQUIRED_MEMBERS = (
    ".png",
    ".chemmol.json",
    ".render.json",
    ".layers.json",
)
CAPACITY_SPLITS = (
    "overfit_32",
    "train_5k",
    "dev_500",
    "dev_route_select_100",
    "dev_route_confirm_400",
)


class AuditError(RuntimeError):
    """Raised when a frozen delivery does not satisfy its contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise AuditError(f"Blank JSONL line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"Expected object: {path}:{line_number}")
            records.append(value)
    return records, file_sha256(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def validate_shard_path(raw_path: str, shard_dir: Path, pattern: str) -> Path:
    path = Path(raw_path)
    require(path.is_absolute(), f"Shard path is not absolute: {raw_path}")
    resolved = path.resolve(strict=True)
    require(resolved.parent == shard_dir.resolve(), f"Shard outside allowlist: {raw_path}")
    require(re.fullmatch(pattern, resolved.name) is not None, f"Bad shard name: {raw_path}")
    require(resolved.is_file() and not resolved.is_symlink(), f"Bad shard file: {raw_path}")
    return resolved


def validate_summary(
    directory: Path,
    records: list[dict[str, Any]],
    records_hash: str,
    rejections: list[dict[str, Any]],
    rejections_hash: str,
) -> dict[str, Any]:
    summary = load_json(directory / "summary.json")
    counts = summary.get("counts", {})
    require(counts.get("accepted") == len(records), "summary accepted count mismatch")
    require(counts.get("rejected") == len(rejections), "summary rejected count mismatch")
    require(
        counts.get("examined") == len(records) + len(rejections),
        "summary examined count mismatch",
    )
    require(
        summary.get("unique_molecules") == len({row["molecule_id"] for row in records}),
        "summary unique molecule count mismatch",
    )
    require(summary.get("samples_sha256") == records_hash, "summary samples hash mismatch")
    require(
        summary.get("source_rejections_sha256") == rejections_hash,
        "summary source rejection hash mismatch",
    )
    return summary


def validate_token_audit(directory: Path, manifest_count: int) -> dict[str, Any]:
    path = directory / "token_length_audit.json"
    audit = load_json(path)
    counts = audit.get("counts", {})
    tokenizer = audit.get("tokenizer", {})
    require(audit.get("status") == "passed", f"Token audit failed: {path}")
    require(counts.get("manifest") == manifest_count, "token manifest count mismatch")
    require(counts.get("audited") == manifest_count, "token audited count mismatch")
    require(counts.get("over_limit") == 0, "token audit has over-limit samples")
    require(not audit.get("over_limit"), "token over-limit details are not empty")
    require(tokenizer.get("max_length") == 3072, "unexpected token context limit")
    require(audit.get("lengths", {}).get("maximum", 3073) <= 3072, "token maximum exceeds limit")
    return audit


def member_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise AuditError(f"Unreadable tar member: {name}")
    value = json.load(handle)
    if not isinstance(value, dict):
        raise AuditError(f"Tar member is not a JSON object: {name}")
    return value


def validate_variant_groups(records: list[dict[str, Any]], label: str) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["molecule_id"]].append(row)
    complete = 0
    incomplete = 0
    for molecule_id, group in grouped.items():
        variants = [row["variant"] for row in group]
        require(len(variants) == len(set(variants)), f"Duplicate variant for {label}:{molecule_id}")
        require(set(variants) <= VARIANTS, f"Unknown variant for {label}:{molecule_id}")
        require(len(group) <= 4, f"More than four variants for {label}:{molecule_id}")
        if len(group) == 4:
            require(set(variants) == VARIANTS, f"Incomplete four-variant group: {label}:{molecule_id}")
            complete += 1
        else:
            incomplete += 1
    return {"complete_molecules": complete, "incomplete_molecules": incomplete}


def validate_manifest_tar_labels(
    records: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    shard_dir: Path,
    shard_pattern: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_shard: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[str] = set()
    accepted_keys: set[tuple[Path, str]] = set()
    identity_by_molecule: dict[str, str] = {}
    counts: Counter[str] = Counter()

    for row in records:
        require(
            set(row) == {"sample_key", "shard", "molecule_id", "inchikey", "variant"},
            f"Unexpected manifest fields: {row.get('sample_key')}",
        )
        key = row["sample_key"]
        require(isinstance(key, str) and key, "Invalid sample key")
        require(key not in seen_keys, f"Duplicate sample key: {key}")
        seen_keys.add(key)
        require(row["variant"] in VARIANTS, f"Unknown variant: {key}")
        require(key.endswith("_" + row["variant"]), f"Key/variant mismatch: {key}")
        shard = validate_shard_path(row["shard"], shard_dir, shard_pattern)
        by_shard[shard].append(row)
        accepted_keys.add((shard, key))
        prior_key = identity_by_molecule.setdefault(row["molecule_id"], row["inchikey"])
        require(prior_key == row["inchikey"], f"Molecule maps to multiple InChIKeys: {row['molecule_id']}")

    rejection_keys: set[tuple[Path, str]] = set()
    for row in rejections:
        require("sample_key" in row and "shard" in row, "Malformed source rejection")
        shard = validate_shard_path(row["shard"], shard_dir, shard_pattern)
        item = (shard, row["sample_key"])
        require(item not in rejection_keys, f"Duplicate source rejection: {row['sample_key']}")
        require(item not in accepted_keys, f"Accepted/rejected overlap: {row['sample_key']}")
        rejection_keys.add(item)

    shard_artifacts: list[dict[str, Any]] = []
    for shard in sorted(by_shard):
        with tarfile.open(shard, "r:") as archive:
            members = {member.name: member for member in archive.getmembers()}
            for row in by_shard[shard]:
                key = row["sample_key"]
                for suffix in REQUIRED_MEMBERS:
                    name = key + suffix
                    require(name in members and members[name].isfile(), f"Missing tar member: {name}")
                    require(members[name].size > 0, f"Empty tar member: {name}")
                chemmol = member_json(archive, key + ".chemmol.json")
                render = member_json(archive, key + ".render.json")
                member_json(archive, key + ".layers.json")
                require(chemmol.get("molecule_id") == row["molecule_id"], f"ChemMol molecule mismatch: {key}")
                require(chemmol.get("inchikey") == row["inchikey"], f"ChemMol InChIKey mismatch: {key}")
                require(render.get("molecule_id") == row["molecule_id"], f"Render molecule mismatch: {key}")
                require(render.get("variant") == row["variant"], f"Render variant mismatch: {key}")
                counts["samples_checked"] += 1
                counts["members_checked"] += len(REQUIRED_MEMBERS)
        shard_artifacts.append(
            {
                "path": relative_to_root(shard),
                "bytes": shard.stat().st_size,
                "sha256": file_sha256(shard),
                "manifest_samples": len(by_shard[shard]),
            }
        )
    return shard_artifacts, counts


def validate_capacity(ab_records: list[dict[str, Any]], ab_hash: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary = load_json(CAPACITY_DIR / "summary.json")
    require(summary.get("source_manifest_sha256") == ab_hash, "capacity source hash mismatch")
    source_by_key = {row["sample_key"]: row for row in ab_records}
    require(len(source_by_key) == len(ab_records), "AB source has duplicate sample keys")
    split_records: dict[str, list[dict[str, Any]]] = {}
    molecule_sets: dict[str, set[str]] = {}
    for name in CAPACITY_SPLITS:
        path = CAPACITY_DIR / f"{name}.jsonl"
        records, digest = load_jsonl(path)
        declared = summary.get("splits", {}).get(name, {})
        require(declared.get("path") == path.name, f"capacity path mismatch: {name}")
        require(declared.get("sha256") == digest, f"capacity hash mismatch: {name}")
        require(declared.get("samples") == len(records), f"capacity sample count mismatch: {name}")
        require(len({row["sample_key"] for row in records}) == len(records), f"capacity duplicate key: {name}")
        for row in records:
            require(source_by_key.get(row["sample_key"]) == row, f"capacity row not byte-semantically in AB: {name}")
        groups = validate_variant_groups(records, name)
        require(groups["incomplete_molecules"] == 0, f"capacity has incomplete molecules: {name}")
        molecules = {row["molecule_id"] for row in records}
        require(declared.get("molecules") == len(molecules), f"capacity molecule count mismatch: {name}")
        split_records[name] = records
        molecule_sets[name] = molecules
    require(molecule_sets["overfit_32"] <= molecule_sets["train_5k"], "overfit is not a train subset")
    require(not molecule_sets["train_5k"] & molecule_sets["dev_500"], "capacity train/dev overlap")
    require(
        not molecule_sets["dev_route_select_100"] & molecule_sets["dev_route_confirm_400"],
        "capacity route split overlap",
    )
    require(
        molecule_sets["dev_route_select_100"] | molecule_sets["dev_route_confirm_400"]
        == molecule_sets["dev_500"],
        "capacity route splits do not partition dev",
    )
    return summary, split_records


def artifact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def make_commitment(
    dataset: str,
    directory: Path,
    artifacts: list[Path],
    validation_path: Path,
    extra: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    body = {
        "schema_version": "chemtraverse-data-commitment-0.1",
        "dataset": dataset,
        "status": "committed",
        "artifacts": [artifact(path) for path in artifacts],
        "validation": artifact(validation_path),
        "auditor": artifact(Path(__file__).resolve()),
        **extra,
    }
    digest = hashlib.sha256(json_bytes(body)).hexdigest()
    commitment = {"commitment_sha256": digest, **body}
    write_json(directory / "commitment.json", commitment)
    # Recompute the defined self-hash from the installed artifact.
    installed = load_json(directory / "commitment.json")
    installed_digest = installed.pop("commitment_sha256")
    require(hashlib.sha256(json_bytes(installed)).hexdigest() == installed_digest == digest, "commitment self-hash mismatch")
    return commitment, digest


def main() -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    ab_records, ab_hash = load_jsonl(AB_DIR / "samples.jsonl")
    ab_rejections, ab_rejections_hash = load_jsonl(AB_DIR / "source_rejections.jsonl")
    fresh_records, fresh_hash = load_jsonl(FRESH_DIR / "samples.jsonl")
    fresh_rejections, fresh_rejections_hash = load_jsonl(FRESH_DIR / "source_rejections.jsonl")

    ab_summary = validate_summary(AB_DIR, ab_records, ab_hash, ab_rejections, ab_rejections_hash)
    fresh_summary = validate_summary(
        FRESH_DIR, fresh_records, fresh_hash, fresh_rejections, fresh_rejections_hash
    )
    ab_tokens = validate_token_audit(AB_DIR, len(ab_records))
    fresh_tokens = validate_token_audit(FRESH_DIR, len(fresh_records))
    ab_groups = validate_variant_groups(ab_records, "ab_50k")
    fresh_groups = validate_variant_groups(fresh_records, "fresh_dev")

    ab_shards, ab_tar_counts = validate_manifest_tar_labels(
        ab_records, ab_rejections, TRAIN_SHARD_DIR, r"train-\d{6}\.tar"
    )
    fresh_shards, fresh_tar_counts = validate_manifest_tar_labels(
        fresh_records, fresh_rejections, FRESH_SHARD_DIR, r"fresh_dev-\d{6}\.tar"
    )

    ab_molecules = {row["molecule_id"] for row in ab_records}
    fresh_molecules = {row["molecule_id"] for row in fresh_records}
    ab_inchikeys = {row["inchikey"] for row in ab_records}
    fresh_inchikeys = {row["inchikey"] for row in fresh_records}
    molecule_overlap = ab_molecules & fresh_molecules
    inchikey_overlap = ab_inchikeys & fresh_inchikeys
    require(not molecule_overlap, "AB and fresh-dev molecule IDs overlap")
    require(not inchikey_overlap, "AB and fresh-dev InChIKeys overlap")

    capacity_summary, capacity_splits = validate_capacity(ab_records, ab_hash)
    isolation = {
        "ab_fresh_molecule_id_overlap": 0,
        "ab_fresh_inchikey_overlap": 0,
        "ab_molecule_id_set_sha256": set_sha256(ab_molecules),
        "ab_inchikey_set_sha256": set_sha256(ab_inchikeys),
        "fresh_molecule_id_set_sha256": set_sha256(fresh_molecules),
        "fresh_inchikey_set_sha256": set_sha256(fresh_inchikeys),
    }

    ab_validation = {
        "schema_version": "chemtraverse-data-validation-0.1",
        "created_at": created_at,
        "dataset": "chemtraverse_v0_1_ab_50k",
        "status": "passed",
        "sealed_rows_read": False,
        "samples": len(ab_records),
        "source_rejections": len(ab_rejections),
        "molecules": len(ab_molecules),
        "variant_groups": ab_groups,
        "tar_audit": dict(ab_tar_counts),
        "token_maximum": ab_tokens["lengths"]["maximum"],
        "token_over_limit": 0,
        "summary_counts": ab_summary["counts"],
        "isolation": isolation,
        "source_shards": ab_shards,
        "errors": [],
    }
    fresh_validation = {
        "schema_version": "chemtraverse-data-validation-0.1",
        "created_at": created_at,
        "dataset": "chemtraverse_v0_1_fresh_dev",
        "status": "passed",
        "sealed_rows_read": False,
        "samples": len(fresh_records),
        "source_rejections": len(fresh_rejections),
        "molecules": len(fresh_molecules),
        "variant_groups": fresh_groups,
        "tar_audit": dict(fresh_tar_counts),
        "token_maximum": fresh_tokens["lengths"]["maximum"],
        "token_over_limit": 0,
        "summary_counts": fresh_summary["counts"],
        "isolation": isolation,
        "source_shards": fresh_shards,
        "errors": [],
    }
    capacity_validation = {
        "schema_version": "chemtraverse-capacity-validation-0.1",
        "created_at": created_at,
        "dataset": "chemtraverse_v0_1_capacity",
        "status": "passed",
        "sealed_rows_read": False,
        "source_manifest_sha256": ab_hash,
        "complete_source_molecules": capacity_summary["complete_source_molecules"],
        "skipped_incomplete_molecules": capacity_summary["skipped_incomplete_molecules"],
        "relations": {
            "overfit_subset_of_train": True,
            "train_dev_molecule_overlap": 0,
            "route_select_confirm_molecule_overlap": 0,
            "route_splits_partition_dev": True,
        },
        "splits": {
            name: {
                "samples": len(rows),
                "molecules": len({row["molecule_id"] for row in rows}),
                "molecule_id_set_sha256": set_sha256(row["molecule_id"] for row in rows),
                "inchikey_set_sha256": set_sha256(row["inchikey"] for row in rows),
            }
            for name, rows in capacity_splits.items()
        },
        "errors": [],
    }

    ab_validation_path = AB_DIR / "validation.json"
    fresh_validation_path = FRESH_DIR / "validation.json"
    capacity_validation_path = CAPACITY_DIR / "validation.json"
    write_json(ab_validation_path, ab_validation)
    write_json(fresh_validation_path, fresh_validation)
    write_json(capacity_validation_path, capacity_validation)

    _, ab_commitment = make_commitment(
        "chemtraverse_v0_1_ab_50k",
        AB_DIR,
        [AB_DIR / name for name in ("samples.jsonl", "source_rejections.jsonl", "summary.json", "token_length_audit.json")],
        ab_validation_path,
        {"source_shards": ab_shards, "identity": isolation},
    )
    _, fresh_commitment = make_commitment(
        "chemtraverse_v0_1_fresh_dev",
        FRESH_DIR,
        [FRESH_DIR / name for name in ("samples.jsonl", "source_rejections.jsonl", "summary.json", "token_length_audit.json")],
        fresh_validation_path,
        {"source_shards": fresh_shards, "identity": isolation},
    )
    _, capacity_commitment = make_commitment(
        "chemtraverse_v0_1_capacity",
        CAPACITY_DIR,
        [CAPACITY_DIR / "summary.json"]
        + [CAPACITY_DIR / f"{name}.jsonl" for name in CAPACITY_SPLITS],
        capacity_validation_path,
        {"source_manifest_sha256": ab_hash},
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "sealed_rows_read": False,
                "commitments": {
                    "ab_50k": ab_commitment,
                    "capacity": capacity_commitment,
                    "fresh_dev": fresh_commitment,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
