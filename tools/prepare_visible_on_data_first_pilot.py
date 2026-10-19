#!/usr/bin/env python3
"""Prepare an identity-disjoint visible O/N data-first pilot manifest.

This script only reads the already rendered ChemVision tar shards and writes a
small JSON manifest.  It does not copy images, read a sealed test split, train,
or expose target coordinates to a model.  Molecule groups are kept intact so
all style views of one molecule stay on one side of the pilot split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "Data/sdf/chemvision_layers_v0_2/font_pilot_10k"
sys.path.insert(0, str(ROOT))
from chemvector.core.data import ChemVectorIterableDataset


def _stable_key(seed: int, molecule_id: str) -> str:
    return hashlib.sha256(f"{seed}:{molecule_id}".encode()).hexdigest()


def _font_bucket(size: int) -> str:
    if size <= 12:
        return "small"
    if size <= 19:
        return "normal"
    if size <= 28:
        return "large"
    return "extreme"


def _rotation_bucket(value: float) -> str:
    if value < -4.0:
        return "neg"
    if value > 4.0:
        return "pos"
    return "near_zero"


def _read_layers(tar: tarfile.TarFile, sample_key: str) -> dict[str, Any]:
    member = tar.getmember(f"{sample_key}.layers.json")
    stream = tar.extractfile(member)
    if stream is None:
        raise RuntimeError(f"missing layers payload for {sample_key}")
    return json.load(stream)


def _representable_keys(data: Path) -> set[str]:
    """Return keys accepted by the production v0.4 target serializer."""
    shards = sorted(str(path) for path in (data / "shards").glob("train-*.tar"))
    if not shards:
        raise FileNotFoundError(f"no train shards under {data / 'shards'}")
    return {
        sample["key"]
        for sample in ChemVectorIterableDataset(
            shards,
            sequence_version="0.4",
            load_raster_layers=False,
        )
    }


def _group_rows(data: Path, representable: set[str]) -> dict[str, list[dict[str, Any]]]:
    csv_path = data / "samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # The source dataset has molecule-disjoint train/validation/test
            # partitions.  The pilot deliberately carves both sides from the
            # public train partition, so no sealed/internal test is consumed.
            if row["split"] != "train":
                continue
            if row["sample_key"] in representable:
                groups[row["molecule_id"]].append(dict(row))
    return groups


def _annotate_groups(data: Path, groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_shard: dict[str, tarfile.TarFile] = {}
    out: list[dict[str, Any]] = []
    try:
        for molecule_id, rows in groups.items():
            # Keep only complete four-view groups.  A partial group would make
            # style coverage look better than the actual image exposure.
            if len(rows) != 4:
                continue
            visible = Counter()
            degrees = Counter()
            implicit_c = 0
            for row in rows:
                # Payloads live in the output shard; source_shard is only
                # provenance for the upstream webdataset and is not present
                # under this rendered dataset.
                shard = row["output_shard"]
                if shard not in by_shard:
                    by_shard[shard] = tarfile.open(data / shard, "r")
                layers = _read_layers(by_shard[shard], row["sample_key"])
                atom_by_id = {int(a["atom_id"]): a for a in layers.get("atoms", [])}
                degree = Counter()
                for bond in layers.get("bonds", []):
                    degree[int(bond["begin_atom"])] += 1
                    degree[int(bond["end_atom"])] += 1
                for atom in layers.get("atoms", []):
                    element = str(atom.get("element", ""))
                    if element in {"O", "N"} and bool(atom.get("visible_text")):
                        visible[element] += 1
                        degrees[f"{element}_degree_{min(degree[int(atom['atom_id'])], 3)}"] += 1
                    if element == "C" and not bool(atom.get("visible_text")):
                        implicit_c += 1
            styles = Counter()
            fonts = Counter()
            sizes = Counter()
            rotations = Counter()
            for row in rows:
                styles[str(row["variant"])] += 1
                fonts[str(row["font_family_id"])] += 1
                sizes[_font_bucket(int(row["font_size"]))] += 1
                rotations[_rotation_bucket(float(row["rotation_degrees"]))] += 1
            out.append({
                "molecule_id": molecule_id,
                "rows": sorted(rows, key=lambda x: x["sample_key"]),
                "visible_on": dict(visible),
                "visible_on_degree": dict(degrees),
                "implicit_c": implicit_c,
                "variants": dict(styles),
                "fonts": dict(fonts),
                "font_buckets": dict(sizes),
                "rotation_buckets": dict(rotations),
            })
    finally:
        for tar in by_shard.values():
            tar.close()
    return out


def _choose(groups: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if len(groups) < count:
        raise RuntimeError(f"only {len(groups)} complete molecules available; need {count}")
    remaining = sorted(groups, key=lambda g: _stable_key(seed, g["molecule_id"]))
    selected: list[dict[str, Any]] = []
    seen_fonts: set[str] = set()
    seen_sizes: set[str] = set()
    seen_rotations: set[str] = set()
    # The first pass intentionally rewards missing style strata.  Afterwards
    # stable hash order makes the manifest reproducible without optimization.
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda i: (
                len(set(remaining[i]["fonts"]) - seen_fonts)
                + len(set(remaining[i]["font_buckets"]) - seen_sizes)
                + len(set(remaining[i]["rotation_buckets"]) - seen_rotations),
                int(remaining[i]["visible_on"].get("O", 0) + remaining[i]["visible_on"].get("N", 0)),
                _stable_key(seed, remaining[i]["molecule_id"]),
            ),
        )
        group = remaining.pop(best_index)
        selected.append(group)
        seen_fonts.update(group["fonts"])
        seen_sizes.update(group["font_buckets"])
        seen_rotations.update(group["rotation_buckets"])
    return selected


def _totals(groups: list[dict[str, Any]]) -> dict[str, Any]:
    visible = Counter()
    degree = Counter()
    fonts = Counter()
    sizes = Counter()
    rotations = Counter()
    variants = Counter()
    rows = 0
    for group in groups:
        rows += len(group["rows"])
        visible.update(group["visible_on"])
        degree.update(group["visible_on_degree"])
        fonts.update(group["fonts"])
        sizes.update(group["font_buckets"])
        rotations.update(group["rotation_buckets"])
        variants.update(group["variants"])
    return {
        "molecules": len(groups), "images": rows,
        "visible_on": dict(visible), "visible_on_degree": dict(degree),
        "implicit_c": sum(g["implicit_c"] for g in groups),
        "fonts": dict(fonts), "font_buckets": dict(sizes),
        "rotation_buckets": dict(rotations), "variants": dict(variants),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pilot", choices=("1k", "5k"), default="1k")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = {"1k": (200, 50), "5k": (1000, 250)}
    train_molecules, holdout_molecules = sizes[args.pilot]
    groups = _annotate_groups(args.data, _group_rows(args.data, _representable_keys(args.data)))
    selected = _choose(groups, train_molecules + holdout_molecules, args.seed)
    train = selected[:train_molecules]
    holdout = selected[train_molecules:]
    train_ids = {g["molecule_id"] for g in train}
    holdout_ids = {g["molecule_id"] for g in holdout}
    if train_ids & holdout_ids:
        raise RuntimeError("molecule identity overlap in pilot split")
    train_summary = _totals(train)
    holdout_summary = _totals(holdout)
    minimum_visible = 400 if args.pilot == "1k" else 2000
    minimum_holdout_visible = 150 if args.pilot == "1k" else 750
    if any(train_summary["visible_on"].get(e, 0) < minimum_visible for e in ("O", "N")):
        raise RuntimeError("train visible O/N quota is not met")
    if any(holdout_summary["visible_on"].get(e, 0) < minimum_holdout_visible for e in ("O", "N")):
        raise RuntimeError("holdout visible O/N quota is not met")
    for summary, label in ((train_summary, "train"), (holdout_summary, "holdout")):
        if len(summary["fonts"]) < 4 or set(summary["font_buckets"]) != {"small", "normal", "large", "extreme"}:
            raise RuntimeError(f"{label} style coverage is incomplete")
        if set(summary["rotation_buckets"]) != {"neg", "near_zero", "pos"}:
            raise RuntimeError(f"{label} rotation coverage is incomplete")
    result = {
        "schema_version": "chemvector-visible-on-data-first-pilot-0.1",
        "pilot": args.pilot, "seed": args.seed, "read_only": True,
        "training_performed": False, "sealed_test_read": False,
        "source_dataset": str(args.data),
        "identity_disjoint": True,
        "train": {"summary": train_summary, "molecules": train},
        "holdout": {"summary": holdout_summary, "molecules": holdout},
        "gates": {
            "train_images": 800 if args.pilot == "1k" else 4000,
            "holdout_images": 200 if args.pilot == "1k" else 1000,
            "visible_O_and_N_each_min": minimum_visible,
            "holdout_visible_O_and_N_each_min": minimum_holdout_visible,
            "implicit_c_replay_required": True,
            "identity_overlap_must_equal": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # The existing WebDataset reader accepts an include_keys set.  Emit only
    # sample keys for downstream read-only audits/training preparation; the
    # manifest remains the source of style and target-count summaries.
    for name, selected_groups in (("train", train), ("holdout", holdout)):
        keys = sorted(row["sample_key"] for group in selected_groups for row in group["rows"])
        (args.output.parent / f"{args.output.stem}-{name}-keys.txt").write_text(
            "\n".join(keys) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output": str(args.output), "train": result["train"]["summary"], "holdout": result["holdout"]["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
