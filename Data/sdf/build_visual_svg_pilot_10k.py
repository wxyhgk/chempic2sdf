#!/usr/bin/env python3
"""Build the molecule-isolated Visual SVG pilot dataset deterministically."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from typing import Any

import fastjsonschema

try:
    from .build_visual_svg_capacity_100 import (
        MEMBER_SUFFIXES,
        PilotArchives,
        json_bytes,
        prepare_sample,
        sha256_bytes,
        stable_int,
        tar_add_bytes,
    )
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
except ImportError:
    from build_visual_svg_capacity_100 import (
        MEMBER_SUFFIXES,
        PilotArchives,
        json_bytes,
        prepare_sample,
        sha256_bytes,
        stable_int,
        tar_add_bytes,
    )
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


SPLIT_ORDER = ("train", "validation", "internal_test")
GROUP_AUDIT_PER_SPLIT = 50
SAMPLE_FIELDS = (
    "sample_key",
    "molecule_id",
    "split",
    "variant",
    "source_sample_key",
    "source_shard",
    "source_output_shard",
    "output_shard",
    "group_alignment_audited",
    "features",
    "line_count",
    "triangle_count",
    "quad_count",
    "text_box_count",
    "primitive_count",
    "quantization_adjusted_lines",
    "sequence_tokens",
    "bond_iou",
    "group_iou_min",
    "group_iou_mean",
    "group_iou_min_nonadjusted",
    "polygon_group_iou_min",
    "group_alignment_f1_min_nonadjusted",
    "polygon_alignment_f1_min",
    "png_sha256",
    "primitives_sha256",
    "sequence_sha256",
)
REJECTED_FIELDS = (
    "sample_key",
    "molecule_id",
    "split",
    "variant",
    "error_type",
    "error",
)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_source_rows(pilot: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    rows: dict[str, dict[str, str]] = {}
    rejected: list[dict[str, str]] = []
    with (pilot / "samples.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_key",
            "molecule_id",
            "split",
            "variant",
            "source_sample_key",
            "source_shard",
            "output_shard",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Source samples.csv is missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            key = row["sample_key"]
            if not key:
                rejected.append(
                    {
                        "sample_key": "",
                        "molecule_id": row.get("molecule_id", ""),
                        "split": row.get("split", ""),
                        "variant": row.get("variant", ""),
                        "error_type": "ManifestError",
                        "error": f"Empty sample_key at row {row_number}",
                    }
                )
                continue
            if key in rows:
                rejected.append(
                    {
                        "sample_key": key,
                        "molecule_id": row.get("molecule_id", ""),
                        "split": row.get("split", ""),
                        "variant": row.get("variant", ""),
                        "error_type": "ManifestError",
                        "error": f"Duplicate sample_key at row {row_number}",
                    }
                )
                continue
            if row["split"] not in SPLIT_ORDER:
                rejected.append(
                    {
                        "sample_key": key,
                        "molecule_id": row.get("molecule_id", ""),
                        "split": row.get("split", ""),
                        "variant": row.get("variant", ""),
                        "error_type": "ManifestError",
                        "error": f"Unsupported split: {row['split']}",
                    }
                )
                continue
            rows[key] = row
    return rows, rejected


def selected_keys_by_split(
    rows: dict[str, dict[str, str]],
    limit_per_split: int | None,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for split in SPLIT_ORDER:
        keys = sorted(key for key, row in rows.items() if row["split"] == split)
        if limit_per_split is not None:
            keys = keys[:limit_per_split]
        if not keys:
            raise RuntimeError(f"No source samples selected for split: {split}")
        selected[split] = keys
    return selected


def group_audit_keys_by_split(
    selected: dict[str, list[str]],
    samples_per_split: int,
) -> dict[str, list[str]]:
    if samples_per_split < 0:
        raise ValueError("Group audit sample count cannot be negative")
    result = {}
    for split in SPLIT_ORDER:
        ranked = sorted(
            selected[split],
            key=lambda key: (stable_int(20260721, "group-audit", split, key), key),
        )
        result[split] = sorted(ranked[:samples_per_split])
    return result


def group_audit_key_sha256(keys_by_split: dict[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for split in SPLIT_ORDER:
        for key in keys_by_split[split]:
            digest.update(f"{split}\t{key}\n".encode("ascii"))
    return digest.hexdigest()


def patch_sample_identity(
    members: dict[str, bytes],
    result: dict[str, Any],
    row: dict[str, str],
    output_shard: str,
    *,
    group_alignment_audited: bool,
) -> None:
    meta = json.loads(members[".meta.json"])
    meta.update(
        {
            "schema_version": "visual-svg-pilot-meta-0.1",
            "sample_key": row["sample_key"],
            "molecule_id": row["molecule_id"],
            "split": row["split"],
            "variant": row["variant"],
            "source_sample_key": row["source_sample_key"],
            "source_shard": row["source_shard"],
            "source_output_shard": row["output_shard"],
            "output_shard": output_shard,
            "group_alignment_audited": group_alignment_audited,
        }
    )
    members[".meta.json"] = json_bytes(meta)
    result.update(
        {
            "sample_key": row["sample_key"],
            "molecule_id": row["molecule_id"],
            "split": row["split"],
            "variant": row["variant"],
            "source_sample_key": row["source_sample_key"],
            "source_shard": row["source_shard"],
            "source_output_shard": row["output_shard"],
            "output_shard": output_shard,
            "group_alignment_audited": group_alignment_audited,
        }
    )


def write_split_shard(
    path: Path,
    samples: list[tuple[str, dict[str, bytes], dict[str, Any]]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with tarfile.open(temporary, "w") as archive:
        for key, members, _ in sorted(samples, key=lambda item: item[0]):
            for suffix in MEMBER_SUFFIXES:
                tar_add_bytes(archive, key + suffix, members[suffix])
    temporary.replace(path)


def audit_split_shard(
    path: Path,
    split: str,
    samples: list[tuple[str, dict[str, bytes], dict[str, Any]]],
) -> dict[str, Any]:
    expected_names = [
        key + suffix
        for key, _, _ in sorted(samples, key=lambda item: item[0])
        for suffix in MEMBER_SUFFIXES
    ]
    expected_by_key = {key: members for key, members, _ in samples}
    checked_members = 0
    checked_meta_hashes = 0
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != expected_names:
            raise RuntimeError(f"Tar member order or manifest mismatch for {split}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate tar member names for {split}")
        for member in members:
            if (
                member.mtime != 0
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
            ):
                raise RuntimeError(f"Non-deterministic tar metadata: {member.name}")
        for key, _, result in sorted(samples, key=lambda item: item[0]):
            for suffix in MEMBER_SUFFIXES:
                extracted = archive.extractfile(key + suffix)
                if extracted is None:
                    raise RuntimeError(f"Cannot read {key}{suffix}")
                actual = extracted.read()
                if actual != expected_by_key[key][suffix]:
                    raise RuntimeError(f"Post-write content mismatch: {key}{suffix}")
                checked_members += 1
            meta_file = archive.extractfile(key + ".meta.json")
            if meta_file is None:
                raise RuntimeError(f"Cannot read metadata for {key}")
            meta = json.load(meta_file)
            for name in ("sample_key", "molecule_id", "split", "variant", "output_shard"):
                if str(meta[name]) != str(result[name]):
                    raise RuntimeError(f"Metadata identity mismatch for {key}: {name}")
            if meta["split"] != split:
                raise RuntimeError(f"Split identity mismatch for {key}")
            for name, expected_hash in meta["sha256"].items():
                extracted = archive.extractfile(key + "." + name)
                if extracted is None or sha256_bytes(extracted.read()) != expected_hash:
                    raise RuntimeError(f"Post-write hash mismatch for {key}.{name}")
                checked_meta_hashes += 1
    return {
        "path": f"shards/{path.name}",
        "samples": len(samples),
        "members": len(expected_names),
        "checked_members": checked_members,
        "checked_meta_hashes": checked_meta_hashes,
        "bytes": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def dataset_content_sha256(
    samples: list[tuple[str, dict[str, bytes], dict[str, Any]]],
) -> str:
    digest = hashlib.sha256()
    for key, members, _ in sorted(samples, key=lambda item: item[0]):
        digest.update(key.encode("ascii"))
        digest.update(b"\0")
        for suffix in MEMBER_SUFFIXES:
            digest.update(suffix.encode("ascii"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(members[suffix]).digest())
    return digest.hexdigest()


def write_failure(
    output: Path,
    selected_counts: dict[str, int],
    accepted: int,
    rejected: list[dict[str, str]],
) -> None:
    write_csv(output / "rejected.csv", REJECTED_FIELDS, rejected)
    summary = {
        "schema_version": "visual-svg-pilot-summary-0.1",
        "status": "failed",
        "selected_split_counts": selected_counts,
        "accepted": accepted,
        "rejected": len(rejected),
    }
    validation = {
        "schema_version": "visual-svg-pilot-validation-0.1",
        "status": "failed",
        "errors": rejected,
    }
    (output / "summary.json").write_bytes(json_bytes(summary))
    (output / "validation.json").write_bytes(json_bytes(validation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot", default="Data/sdf/chemvision_layers_v0_1/pilot_10k"
    )
    parser.add_argument(
        "--output", default="Data/sdf/visual_svg_v0_1/pilot_10k"
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument(
        "--group-audit-per-split", type=int, default=GROUP_AUDIT_PER_SPLIT
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        raise ValueError("--limit-per-split must be positive")
    if args.group_audit_per_split < 0:
        raise ValueError("--group-audit-per-split must be nonnegative")

    pilot = Path(args.pilot).resolve()
    output = Path(args.output).resolve()
    source_validation = json.loads((pilot / "validation.json").read_text(encoding="utf-8"))
    if source_validation.get("status") != "passed":
        raise RuntimeError("Source pilot validation status is not passed")
    for name in ("scaffold_overlap", "font_family_overlap"):
        if int(source_validation.get(name, -1)) != 0:
            raise RuntimeError(f"Source pilot {name} is not zero")
    if output.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "shards").mkdir(parents=True)

    rows, rejected = load_source_rows(pilot)
    selected = selected_keys_by_split(rows, args.limit_per_split)
    audit_keys = group_audit_keys_by_split(selected, args.group_audit_per_split)
    audit_key_sets = {split: set(keys) for split, keys in audit_keys.items()}
    selected_counts = {split: len(selected[split]) for split in SPLIT_ORDER}
    molecule_splits: dict[str, set[str]] = {}
    for split in SPLIT_ORDER:
        for key in selected[split]:
            molecule_splits.setdefault(rows[key]["molecule_id"], set()).add(split)
    molecule_overlap = {
        molecule_id: sorted(splits)
        for molecule_id, splits in molecule_splits.items()
        if len(splits) > 1
    }
    if molecule_overlap:
        rejected.extend(
            {
                "sample_key": "",
                "molecule_id": molecule_id,
                "split": ";".join(splits),
                "variant": "",
                "error_type": "SplitLeakageError",
                "error": "Molecule appears in multiple output splits",
            }
            for molecule_id, splits in sorted(molecule_overlap.items())
        )
    if rejected:
        write_failure(output, selected_counts, 0, rejected)
        raise RuntimeError(f"Source manifest validation failed with {len(rejected)} errors")

    archives = PilotArchives(pilot, rows)
    tokenizer = VisualSVGTokenizerV01()
    schema_path = Path(__file__).parent / "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json"
    validator = fastjsonschema.compile(json.loads(schema_path.read_text(encoding="utf-8")))
    prepared_by_split: dict[str, list[tuple[str, dict[str, bytes], dict[str, Any]]]] = {
        split: [] for split in SPLIT_ORDER
    }
    prepared_total = 0
    started = time.perf_counter()
    try:
        for split in SPLIT_ORDER:
            output_shard = f"shards/{split}-000000.tar"
            for key in selected[split]:
                row = rows[key]
                group_alignment_audited = key in audit_key_sets[split]
                try:
                    members, result, _ = prepare_sample(
                        key,
                        row,
                        archives,
                        tokenizer,
                        validator,
                        args.max_tokens,
                        validate_group_alignment=group_alignment_audited,
                    )
                    patch_sample_identity(
                        members,
                        result,
                        row,
                        output_shard,
                        group_alignment_audited=group_alignment_audited,
                    )
                    prepared_by_split[split].append((key, members, result))
                    prepared_total += 1
                    if prepared_total % 250 == 0:
                        print(
                            f"prepared={prepared_total}/{sum(selected_counts.values())} "
                            f"split={split} elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
                except Exception as error:
                    rejected.append(
                        {
                            "sample_key": key,
                            "molecule_id": row["molecule_id"],
                            "split": split,
                            "variant": row["variant"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    finally:
        archives.close()

    accepted = sum(len(items) for items in prepared_by_split.values())
    if rejected:
        write_failure(output, selected_counts, accepted, rejected)
        raise RuntimeError(f"Visual SVG pilot rejected {len(rejected)} samples")

    shard_audits = {}
    all_samples: list[tuple[str, dict[str, bytes], dict[str, Any]]] = []
    try:
        for split in SPLIT_ORDER:
            samples = prepared_by_split[split]
            shard_path = output / f"shards/{split}-000000.tar"
            write_split_shard(shard_path, samples)
            shard_audits[split] = audit_split_shard(shard_path, split, samples)
            all_samples.extend(samples)
    except Exception as error:
        post_write_error = {
            "sample_key": "",
            "molecule_id": "",
            "split": "",
            "variant": "",
            "error_type": type(error).__name__,
            "error": f"Post-write audit failed: {error}",
        }
        write_failure(output, selected_counts, accepted, [post_write_error])
        raise

    sample_rows = [result for _, _, result in sorted(all_samples, key=lambda item: item[0])]
    write_csv(output / "samples.csv", SAMPLE_FIELDS, sample_rows)
    write_csv(output / "rejected.csv", REJECTED_FIELDS, [])
    tokenizer.save(output / "tokenizer.json")
    split_counts = Counter(row["split"] for row in sample_rows)
    variant_counts = Counter(row["variant"] for row in sample_rows)
    primitive_totals = {
        name: sum(int(row[f"{name}_count"]) for row in sample_rows)
        for name in ("line", "triangle", "quad", "text_box")
    }
    token_counts = [int(row["sequence_tokens"]) for row in sample_rows]
    bond_ious = [float(row["bond_iou"]) for row in sample_rows]
    audited_rows = [row for row in sample_rows if row["group_alignment_audited"]]
    audited_alignment = [
        float(row["group_alignment_f1_min_nonadjusted"]) for row in audited_rows
    ]
    audited_polygon_alignment = [
        float(row["polygon_alignment_f1_min"]) for row in audited_rows
    ]
    summary = {
        "schema_version": "visual-svg-pilot-summary-0.1",
        "status": "passed",
        "limited": args.limit_per_split is not None,
        "limit_per_split": args.limit_per_split,
        "samples": len(sample_rows),
        "molecules": len({row["molecule_id"] for row in sample_rows}),
        "split_counts": dict(sorted(split_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "primitive_counts": primitive_totals,
        "sequence_tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
        },
        "bond_iou": {
            "min": min(bond_ious),
            "mean": sum(bond_ious) / len(bond_ious),
        },
        "group_alignment_audit": {
            "selection_rule": "lowest stable SHA256 rank per split",
            "requested_per_split": args.group_audit_per_split,
            "split_counts": {
                split: len(audit_keys[split]) for split in SPLIT_ORDER
            },
            "sample_count": len(audited_rows),
            "sample_key_sha256": group_audit_key_sha256(audit_keys),
            "minimum_nonadjusted_f1": min(audited_alignment, default=None),
            "minimum_polygon_f1": min(audited_polygon_alignment, default=None),
        },
        "tar_members": sum(item["members"] for item in shard_audits.values()),
        "shards": shard_audits,
        "content_sha256": dataset_content_sha256(all_samples),
        "source_samples_csv_sha256": sha256_bytes((pilot / "samples.csv").read_bytes()),
        "rejected": 0,
    }
    validation = {
        "schema_version": "visual-svg-pilot-validation-0.1",
        "status": "passed",
        "checks": {
            "source_validation_status": source_validation["status"],
            "source_scaffold_overlap": source_validation["scaffold_overlap"],
            "source_font_family_overlap": source_validation["font_family_overlap"],
            "selected_molecule_cross_split_overlap": len(molecule_overlap),
            "selected_split_counts": selected_counts,
            "unique_sample_keys": len({row["sample_key"] for row in sample_rows}),
            "unique_molecules": summary["molecules"],
            "post_write_members": summary["tar_members"],
            "post_write_content_hashes": sum(
                item["checked_members"] for item in shard_audits.values()
            ),
            "post_write_meta_hashes": sum(
                item["checked_meta_hashes"] for item in shard_audits.values()
            ),
            "minimum_bond_iou": summary["bond_iou"]["min"],
            "group_alignment_audit": summary["group_alignment_audit"],
            "content_sha256": summary["content_sha256"],
        },
        "errors": [],
    }
    (output / "summary.json").write_bytes(json_bytes(summary))
    (output / "validation.json").write_bytes(json_bytes(validation))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
