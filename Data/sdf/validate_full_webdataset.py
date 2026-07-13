#!/usr/bin/env python3
"""Validate the completed ChemMol WebDataset package end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

from chemmol_webdataset import ChemMolWebDataset


EXPECTED_VARIANTS = {"original", "rdkit", "coordgen", "stress"}
EXPECTED_SUFFIXES = {
    ".png",
    ".chemmol.json",
    ".render.json",
    ".meta.json",
    ".chemmolseq.txt",
}


def key_digest(keys: set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_member_name(name: str) -> tuple[str, str]:
    for suffix in sorted(EXPECTED_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    raise ValueError(f"Unexpected tar member: {name}")


def validate_image(data: bytes, label: str) -> None:
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"{label}: expected RGB, got {image.mode}")
        if image.size != (768, 768):
            raise ValueError(f"{label}: expected 768x768, got {image.size}")
        if all(low == high == 255 for low, high in image.getextrema()):
            raise ValueError(f"{label}: image is blank white")


def validate_shard(task: tuple[str, int, str]) -> dict:
    shard_path_text, expected_samples, expected_digest = task
    shard_path = Path(shard_path_text)
    suffixes_by_key: dict[str, set[str]] = defaultdict(set)
    first_png = None
    last_png = None
    member_count = 0

    with tarfile.open(shard_path, "r") as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError(f"{shard_path.name}: non-file member {member.name}")
            member_count += 1
            key, suffix = split_member_name(member.name)
            if suffix in suffixes_by_key[key]:
                raise ValueError(f"{shard_path.name}: duplicate {key}{suffix}")
            suffixes_by_key[key].add(suffix)
            if suffix == ".png":
                if first_png is None:
                    first_png = member
                last_png = member

        if len(suffixes_by_key) != expected_samples:
            raise ValueError(
                f"{shard_path.name}: {len(suffixes_by_key)} samples, "
                f"expected {expected_samples}"
            )
        if member_count != expected_samples * len(EXPECTED_SUFFIXES):
            raise ValueError(
                f"{shard_path.name}: {member_count} members, "
                f"expected {expected_samples * len(EXPECTED_SUFFIXES)}"
            )
        incomplete = [
            key for key, suffixes in suffixes_by_key.items()
            if suffixes != EXPECTED_SUFFIXES
        ]
        if incomplete:
            raise ValueError(f"{shard_path.name}: incomplete sample {incomplete[0]}")
        if key_digest(set(suffixes_by_key)) != expected_digest:
            raise ValueError(f"{shard_path.name}: sample keys differ from samples.csv")
        if first_png is None or last_png is None:
            raise ValueError(f"{shard_path.name}: no PNG members")
        for member, position in ((first_png, "first"), (last_png, "last")):
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"{shard_path.name}: cannot read {member.name}")
            validate_image(extracted.read(), f"{shard_path.name}:{position}")

    loaded = next(iter(ChemMolWebDataset(str(shard_path))))
    if loaded["image"].size != (768, 768):
        raise ValueError(f"{shard_path.name}: loader returned wrong image size")

    return {
        "shard": shard_path.name,
        "members": member_count,
        "samples": len(suffixes_by_key),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="webdataset_full_v0")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    molecules = read_csv(root / "molecules.csv")
    samples = read_csv(root / "samples.csv")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

    molecule_ids = [row["molecule_id"] for row in molecules]
    sample_keys = [row["sample_key"] for row in samples]
    if len(molecules) != 180_000 or len(set(molecule_ids)) != 180_000:
        raise ValueError("Expected exactly 180,000 unique molecule IDs")
    if len(samples) != 720_000 or len(set(sample_keys)) != 720_000:
        raise ValueError("Expected exactly 720,000 unique sample keys")

    molecule_by_id = {row["molecule_id"]: row for row in molecules}
    variants_by_molecule: dict[str, set[str]] = defaultdict(set)
    sample_keys_by_shard: dict[str, set[str]] = defaultdict(set)
    variant_counts = Counter()
    color_counts = Counter()
    for row in samples:
        molecule = molecule_by_id.get(row["molecule_id"])
        if molecule is None:
            raise ValueError(f"Unknown molecule in samples.csv: {row['molecule_id']}")
        if row["split"] != molecule["split"]:
            raise ValueError(f"Split mismatch for {row['sample_key']}")
        if row["shard"] != molecule["shard"]:
            raise ValueError(f"Shard mismatch for {row['sample_key']}")
        variants_by_molecule[row["molecule_id"]].add(row["variant"])
        sample_keys_by_shard[row["shard"]].add(row["sample_key"])
        variant_counts[row["variant"]] += 1
        color_counts[row["color_mode"]] += 1

    bad_variants = [
        molecule_id
        for molecule_id, variants in variants_by_molecule.items()
        if variants != EXPECTED_VARIANTS
    ]
    if bad_variants or len(variants_by_molecule) != len(molecules):
        raise ValueError(f"Incomplete variants for molecule: {bad_variants[:1]}")

    source_counts = Counter(row["source"] for row in molecules)
    split_counts = Counter(row["split"] for row in molecules)
    if source_counts != Counter({"chembl": 150_000, "coconut": 30_000}):
        raise ValueError(f"Unexpected source counts: {source_counts}")
    expected_split_counts = {"train": 162_008, "validation": 8_996, "internal_test": 8_996}
    if split_counts != Counter(expected_split_counts):
        raise ValueError(f"Unexpected split counts: {split_counts}")

    indices_by_split: dict[str, set[int]] = defaultdict(set)
    scaffolds_by_split: dict[str, set[str]] = defaultdict(set)
    for row in molecules:
        indices_by_split[row["split"]].add(int(row["split_index"]))
        if row["scaffold"]:
            scaffolds_by_split[row["split"]].add(row["scaffold"])
    for split, count in expected_split_counts.items():
        if indices_by_split[split] != set(range(count)):
            raise ValueError(f"Non-contiguous split_index values in {split}")
    splits = sorted(scaffolds_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = scaffolds_by_split[left] & scaffolds_by_split[right]
            if overlap:
                raise ValueError(f"Scaffold overlap between {left} and {right}: {next(iter(overlap))}")

    structure_refs = {row["structure_ref"] for row in molecules}
    actual_structures = {
        path.relative_to(root).as_posix()
        for path in (root / "structures").rglob("*.sdf")
    }
    if actual_structures != structure_refs:
        raise ValueError("Structure files differ from molecules.csv references")

    actual_shards = {
        path.relative_to(root).as_posix()
        for path in (root / "shards").glob("*.tar")
    }
    if actual_shards != set(sample_keys_by_shard):
        raise ValueError("Tar shards differ from samples.csv references")
    if len(actual_shards) != 149:
        raise ValueError(f"Expected 149 shards, got {len(actual_shards)}")

    forbidden = []
    for path in root.rglob("*"):
        if path.name in {".venv", "__pycache__", ".ipynb_checkpoints"} or path.suffix == ".pyc":
            forbidden.append(path)
    if forbidden:
        raise ValueError(f"Forbidden package content: {forbidden[0]}")

    tasks = []
    for shard in sorted(actual_shards):
        keys = sample_keys_by_shard[shard]
        tasks.append((str(root / shard), len(keys), key_digest(keys)))

    tar_samples = 0
    tar_members = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(validate_shard, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            tar_samples += result["samples"]
            tar_members += result["members"]
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"Validated shards: {completed}/{len(tasks)}", flush=True)

    if tar_samples != len(samples) or tar_members != len(samples) * 5:
        raise ValueError("Archive totals do not match manifest totals")
    if summary["samples"] != len(samples) or summary["valid_molecules"] != len(molecules):
        raise ValueError("summary.json totals do not match CSV manifests")

    result = {
        "status": "passed",
        "molecules": len(molecules),
        "samples": len(samples),
        "structures": len(actual_structures),
        "shards": len(actual_shards),
        "tar_members": tar_members,
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "variant_counts": dict(variant_counts),
        "color_mode_counts": dict(color_counts),
        "scaffold_overlap": 0,
        "forbidden_files": 0,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
