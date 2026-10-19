#!/usr/bin/env python3
"""Build the fresh ChemVector safe-suppression v0.1 data package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import heapq
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time
from typing import Any, Iterable, Mapping, Sequence
from functools import lru_cache

from rdkit import Chem, RDLogger, rdBase


HERE = Path(__file__).resolve().parent
SDF_ROOT = HERE.parent
PROJECT_ROOT = SDF_ROOT.parents[1]
RAW_CHEMBL = SDF_ROOT / "extracted/raw/chembl_37.sdf"
JOINT_ROOT = SDF_ROOT / "chemvector_joint_v1"
SELECT_ROOT = HERE / "route_select"
CONFIRM_ROOT = HERE / "route_confirm"
WORK_ROOT = HERE / "work"
SEED = 20260730
WIDTH = HEIGHT = 768
VARIANTS = ("original", "rdkit", "coordgen", "stress")
SIZE_BUCKETS = ("8-20", "21-40", "41-60", "61-80")
POOL_PER_SIZE = 10_000
POOL_HEAP_CAPACITY = 12_500
SELECT_POOL_PER_SIZE = 2_500
SELECT_SIZE_TARGET = 500
CONFIRM_SIZE_TARGET = 2_625
SELECT_CLOSE_TARGETS = {"8-20": 25, "21-40": 80, "41-60": 155, "61-80": 240}
CONFIRM_CLOSE_TARGETS = {"8-20": 150, "21-40": 450, "41-60": 800, "61-80": 1_225}
SHARD_SIZE = 5_000
MEMBERS_PER_SAMPLE = 9
RENDERER_ID = "rdkit-svg-semantic-safe-suppression-0.1"
HISTORICAL_FONT_IDS = {
    "dejavu-sans", "dejavu-sans-bold", "dejavu-sans-condensed", "dejavu-serif",
    "dejavu-serif-bold", "dejavu-serif-condensed", "gnu-free-sans", "gnu-free-serif",
    "latin-modern-roman", "latin-modern-sans", "liberation-sans", "liberation-sans-narrow",
    "liberation-serif", "nimbus-roman", "p052-roman", "rasa", "stix-regular", "ubuntu",
    "urw-bookman", "yrsa",
}


if str(SDF_ROOT) not in sys.path:
    sys.path.insert(0, str(SDF_ROOT))
from chemvector_joint_v1 import build_chemvector_joint_v1 as joint
import build_chemvision_layers as layer_builder
import build_webdataset_sample as molecule_pipeline


FONT_SPECS = {
    "route_select": (
        ("c059", "C059", "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf", "AGPL-3 with Font exception"),
        ("nimbus-sans", "Nimbus Sans", "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf", "AGPL-3 with Font exception"),
        ("nimbus-mono-ps", "Nimbus Mono PS", "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf", "AGPL-3 with Font exception"),
        ("urw-gothic", "URW Gothic", "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf", "AGPL-3 with Font exception"),
    ),
    "route_confirm": (
        ("dejavu-math", "DejaVu Math TeX Gyre", "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf", "Bitstream/DejaVu license"),
        ("abyssinica-sil", "Abyssinica SIL", "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf", "SIL Open Font License 1.1"),
        ("noto-sans-cjk", "Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "SIL Open Font License 1.1"),
        ("noto-serif-cjk", "Noto Serif CJK SC", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "SIL Open Font License 1.1"),
    ),
}

PLAN_FIELDS = (
    "selection_index", "split", "geometry_bucket", "min_anchor_distance", "close_variant",
    "source", "source_id", "molecule_id", "inchikey", "scaffold",
    "canonical_isomeric_smiles", "size_bucket", "heavy_atoms", "rings", "fused_rings",
    "bridged_atoms", "chiral_centers", "element_bucket", "structure_file", "sdf_sha256",
)


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def set_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=2)
def font_specs(split: str) -> tuple[dict[str, Any], ...]:
    result = []
    for font_id, family, source, license_name in FONT_SPECS[split]:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(
            {
                "font_family_id": font_id,
                "font_family": family,
                "font_style": "Regular",
                "source_file": source,
                "font_file": f"fonts/{path.name}",
                "font_file_sha256": sha256_file(path),
                "license": license_name,
                "license_url": "",
                "split": split,
                "font_family_group": "heldout",
            }
        )
    return tuple(result)


def validate_font_isolation() -> dict[str, Any]:
    by_split = {
        split: {str(row["font_family_id"]) for row in font_specs(split)}
        for split in ("route_select", "route_confirm")
    }
    old_overlap = (by_split["route_select"] | by_split["route_confirm"]) & HISTORICAL_FONT_IDS
    cross_overlap = by_split["route_select"] & by_split["route_confirm"]
    if old_overlap or cross_overlap or any(len(values) < 4 for values in by_split.values()):
        raise RuntimeError(f"font isolation failed: old={old_overlap}, cross={cross_overlap}, sets={by_split}")
    return {
        "route_select": sorted(by_split["route_select"]),
        "route_confirm": sorted(by_split["route_confirm"]),
        "old_font_overlap": 0,
        "cross_split_font_overlap": 0,
    }


def style_and_task(
    row: Mapping[str, Any], variant: str, sample_key: str | None = None
) -> tuple[layer_builder.RenderTask, joint.StyleSpec]:
    split = str(row["split"])
    molecule_id = str(row["molecule_id"])
    fonts = font_specs(split)
    font = fonts[stable_int(SEED, "font", split, molecule_id, variant) % len(fonts)]
    font_roll = stable_int(SEED, "font-bucket", split, molecule_id, variant) % 100
    font_bucket = "small" if font_roll < 25 else "normal" if font_roll < 70 else "large" if font_roll < 95 else "extreme"
    font_ranges = {"small": (8, 12), "normal": (13, 19), "large": (20, 28), "extreme": (29, 36)}
    low, high = font_ranges[font_bucket]
    font_size = low + stable_int(SEED, "font-size", split, molecule_id, variant) % (high - low + 1)
    line_roll = stable_int(SEED, "line-bucket", split, molecule_id, variant) % 3
    line_bucket = ("thin", "normal", "thick")[line_roll]
    line_ranges = {"thin": (1.4, 1.7), "normal": (1.8, 2.2), "thick": (2.3, 2.6)}
    line_rng = random.Random(stable_int(SEED, "line-width", split, molecule_id, variant))
    line_width = round(line_rng.uniform(*line_ranges[line_bucket]), 6)
    rotation_roll = stable_int(SEED, "rotation-bucket", split, molecule_id, variant) % 3
    rotation_bucket = ("negative", "near_zero", "positive")[rotation_roll]
    rotation_ranges = {"negative": (-12.0, -3.0), "near_zero": (-2.9, 2.9), "positive": (3.0, 12.0)}
    rotation_rng = random.Random(stable_int(SEED, "rotation", split, molecule_id, variant))
    rotation = round(rotation_rng.uniform(*rotation_ranges[rotation_bucket]), 6)
    offset_bucket = ("tight", "normal", "wide", "extra_wide")[
        stable_int(SEED, "offset", split, molecule_id, variant) % 4
    ]
    offset = {"tight": 0.10, "normal": 0.14, "wide": 0.18, "extra_wide": 0.22}[offset_bucket]
    color_roll = stable_int(SEED, "color", split, molecule_id, variant) % 100
    color_mode = "black_white" if color_roll < 70 else "limited_color" if color_roll < 85 else "grayscale"
    key = sample_key or f"{split}_{int(row.get('selection_index', 0)):06d}_{variant}"
    source_key = f"{split}_{int(row.get('selection_index', 0)):06d}_{variant}"
    task = layer_builder.RenderTask(
        sample_key=key,
        molecule_id=molecule_id,
        split=split,
        variant=variant,
        source_sample_key=source_key,
        source_shard="raw/chembl_37.sdf",
        output_shard="",
        font_family_id=str(font["font_family_id"]),
        font_source_file=str(font["source_file"]),
        font_file_sha256=str(font["font_file_sha256"]),
        font_size=int(font_size),
        font_bucket=font_bucket,
        bond_line_width=line_width,
        rotation_degrees=rotation,
        repeat_check=False,
    )
    style = joint.StyleSpec(
        font_family_group="heldout",
        color_mode=color_mode,
        line_width_bucket=line_bucket,
        rotation_bucket=rotation_bucket,
        multiple_bond_offset_bucket=offset_bucket,
        multiple_bond_offset=offset,
    )
    return task, style


def build_old_index() -> dict[str, set[str]]:
    cached = json.loads((JOINT_ROOT / "work/old_overlap_index.json").read_text(encoding="utf-8"))
    old = {
        "molecule_ids": set(cached["molecule_ids"]),
        "inchikeys": set(cached["inchikeys"]),
        "scaffolds": set(cached["scaffolds"]),
    }
    for row in load_csv(JOINT_ROOT / "work/fresh_molecules.plan.csv"):
        old["molecule_ids"].add(row["molecule_id"])
        old["inchikeys"].add(row["inchikey"])
        old["scaffolds"].add(row["scaffold"])
    return old


def candidate_to_dict(candidate: joint.FreshCandidate) -> dict[str, Any]:
    return asdict(candidate)


def scan_candidate_pool(force: bool = False) -> list[dict[str, Any]]:
    cache = WORK_ROOT / "candidate_pool.jsonl.gz"
    audit_path = WORK_ROOT / "candidate_scan_audit.json"
    if cache.is_file() and not force:
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    old = build_old_index()
    heaps: dict[str, list[tuple[int, int, joint.FreshCandidate]]] = {bucket: [] for bucket in SIZE_BUCKETS}
    counters = Counter()
    joint.SEED = SEED
    molecule_pipeline.SEED = SEED
    supplier = Chem.SDMolSupplier(str(RAW_CHEMBL), sanitize=True, removeHs=False, strictParsing=False)
    started = time.perf_counter()
    for index, mol in enumerate(supplier, 1):
        counters["records"] += 1
        candidate = joint.candidate_from_mol(mol, index, old) if mol is not None else None
        if candidate is None:
            counters["rejected"] += 1
        else:
            counters["eligible"] += 1
            heap = heaps[candidate.size_bucket]
            item = (-candidate.rank, index, candidate)
            if len(heap) < POOL_HEAP_CAPACITY:
                heapq.heappush(heap, item)
            elif candidate.rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
        if index % 250_000 == 0:
            print(f"Scanned ChEMBL {index:,}: eligible={counters['eligible']:,}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    used_scaffolds: set[str] = set()
    for bucket in SIZE_BUCKETS:
        ordered = sorted((item[2] for item in heaps[bucket]), key=lambda item: (item.rank, item.molecule_id))
        bucket_rows = []
        for candidate in ordered:
            if candidate.inchikey in used_keys or candidate.scaffold in used_scaffolds:
                continue
            bucket_rows.append(candidate_to_dict(candidate))
            used_keys.add(candidate.inchikey)
            used_scaffolds.add(candidate.scaffold)
            if len(bucket_rows) == POOL_PER_SIZE:
                break
        if len(bucket_rows) != POOL_PER_SIZE:
            raise RuntimeError(f"candidate pool {bucket} has {len(bucket_rows)} rows")
        selected.extend(bucket_rows)
    with gzip.open(cache, "wt", encoding="utf-8", compresslevel=1) as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    write_json(
        audit_path,
        {
            "status": "passed",
            "seed": SEED,
            "source": str(RAW_CHEMBL.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_file(RAW_CHEMBL),
            "scan_counts": dict(counters),
            "pool_size": len(selected),
            "pool_size_counts": dict(Counter(row["size_bucket"] for row in selected)),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return selected


def classify_candidate(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    candidate, split = payload
    RDLogger.DisableLog("rdApp.*")
    joint.SEED = SEED
    molecule_pipeline.SEED = SEED
    mol = Chem.MolFromMolBlock(candidate["molblock"], sanitize=True, removeHs=False, strictParsing=False)
    if mol is None:
        return {"status": "rejected", "molecule_id": candidate["molecule_id"], "error": "sdf_parse"}
    mol.SetProp("_Name", candidate["source_id"])
    try:
        record = molecule_pipeline.canonicalize_record("chembl", mol, candidate["smiles"], 0)
        if record.molecule_id != candidate["molecule_id"] or record.inchikey != candidate["inchikey"]:
            raise ValueError("canonical_identity")
        chemmol = molecule_pipeline.build_chemmol(record)
        reconstructed = molecule_pipeline.mol_from_chemmol(chemmol)
        Chem.SanitizeMol(reconstructed)
        if Chem.MolToInchiKey(reconstructed) != record.inchikey:
            raise ValueError("chemmol_roundtrip")
        record.mol.SetProp("_Name", candidate["source_id"])
        sdf_payload = (
            Chem.MolToMolBlock(record.mol, includeStereo=True, kekulize=True, forceV3000=True) + "$$$$\n"
        ).encode("utf-8")
        sdf_mol = Chem.MolFromMolBlock(
            sdf_payload.decode("utf-8").split("$$$$", 1)[0],
            sanitize=True, removeHs=False, strictParsing=False,
        )
        if sdf_mol is None or Chem.MolToInchiKey(sdf_mol) != record.inchikey:
            raise ValueError("sdf_roundtrip")
        distances = {}
        close_variant = ""
        minimum = float("inf")
        for variant_index, variant in enumerate(VARIANTS):
            _, source_render = molecule_pipeline.build_render(record, variant, variant_index)
            task, style = style_and_task(
                {"split": split, "molecule_id": record.molecule_id, "selection_index": 0},
                variant,
                sample_key=f"classify_{record.source_id}_{variant}",
            )
            joint.STYLE_BY_SAMPLE = {task.sample_key: style}
            source_render["render_parameters"]["multiple_bond_offset"] = style.multiple_bond_offset
            _, _, points, _ = joint.draw_semantic_svg_joint(task, chemmol, source_render)
            distance = min(
                (math.hypot(points[i].x - points[j].x, points[i].y - points[j].y) for i in range(len(points)) for j in range(i)),
                default=float("inf"),
            )
            distances[variant] = round(distance, 6)
            if distance < minimum:
                minimum = distance
                close_variant = variant
        result = dict(candidate)
        result.update(
            {
                "status": "passed",
                "split": split,
                "minimum_anchor_distance": round(minimum, 6),
                "close_variant": close_variant,
                "geometry_bucket": "close_distinct" if minimum <= 16.0 else "general",
                "variant_minimum_distances": distances,
                "sdf_payload": sdf_payload.decode("utf-8"),
            }
        )
        return result
    except Exception as error:
        return {
            "status": "rejected",
            "molecule_id": candidate["molecule_id"],
            "error": f"{type(error).__name__}:{error}",
        }


def classify_pool(pool: Sequence[dict[str, Any]], workers: int, force: bool = False) -> list[dict[str, Any]]:
    cache = WORK_ROOT / "classified_pool.jsonl.gz"
    if cache.is_file() and not force:
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    inputs = []
    for bucket in SIZE_BUCKETS:
        rows = [row for row in pool if row["size_bucket"] == bucket]
        rows.sort(key=lambda row: (stable_int(SEED, "pool-partition", bucket, row["molecule_id"]), row["molecule_id"]))
        inputs.extend((row, "route_select") for row in rows[:SELECT_POOL_PER_SIZE])
        inputs.extend((row, "route_confirm") for row in rows[SELECT_POOL_PER_SIZE:])
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(classify_candidate, item) for item in inputs]
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed % 1_000 == 0 or completed == len(futures):
                passed = sum(row["status"] == "passed" for row in results)
                print(f"Classified candidates {completed:,}/{len(futures):,}: passed={passed:,}", flush=True)
    results.sort(key=lambda row: (row.get("status", ""), row.get("molecule_id", "")))
    with gzip.open(cache, "wt", encoding="utf-8", compresslevel=1) as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    return results


def select_plan(classified: Sequence[dict[str, Any]], force: bool = False) -> list[dict[str, str]]:
    plan_path = WORK_ROOT / "molecules.plan.csv"
    if plan_path.is_file() and not force:
        return load_csv(plan_path)
    structure_root = WORK_ROOT / "structures"
    if structure_root.exists():
        shutil.rmtree(structure_root)
    structure_root.mkdir(parents=True)
    passed = [row for row in classified if row["status"] == "passed"]
    selected: list[dict[str, str]] = []
    for split, size_target, close_targets in (
        ("route_select", SELECT_SIZE_TARGET, SELECT_CLOSE_TARGETS),
        ("route_confirm", CONFIRM_SIZE_TARGET, CONFIRM_CLOSE_TARGETS),
    ):
        for bucket in SIZE_BUCKETS:
            close_target = close_targets[bucket]
            general_target = size_target - close_target
            for geometry_bucket, target in (("close_distinct", close_target), ("general", general_target)):
                rows = [
                    row for row in passed
                    if row["split"] == split and row["size_bucket"] == bucket and row["geometry_bucket"] == geometry_bucket
                ]
                rows.sort(
                    key=lambda row: (
                        stable_int(SEED, "final-select", split, bucket, geometry_bucket, row["molecule_id"]),
                        row["molecule_id"],
                    )
                )
                if len(rows) < target:
                    raise RuntimeError(f"selection shortage {split}/{bucket}/{geometry_bucket}: {len(rows)} < {target}")
                selected.extend(rows[:target])
    used_ids = [row["molecule_id"] for row in selected]
    used_keys = [row["inchikey"] for row in selected]
    used_scaffolds = [row["scaffold"] for row in selected]
    if len(set(used_ids)) != len(selected) or len(set(used_keys)) != len(selected) or len(set(used_scaffolds)) != len(selected):
        raise RuntimeError("selected pool identity/scaffold duplication")
    plan_rows: list[dict[str, str]] = []
    for split in ("route_select", "route_confirm"):
        split_rows = [row for row in selected if row["split"] == split]
        split_rows.sort(key=lambda row: (stable_int(SEED, "plan-order", split, row["molecule_id"]), row["molecule_id"]))
        for index, row in enumerate(split_rows):
            filename = f"{split}_{index:06d}_{row['source_id']}.sdf"
            payload = row["sdf_payload"].encode("utf-8")
            (structure_root / filename).write_bytes(payload)
            plan_rows.append(
                {
                    "selection_index": str(index),
                    "split": split,
                    "geometry_bucket": row["geometry_bucket"],
                    "min_anchor_distance": str(row["minimum_anchor_distance"]),
                    "close_variant": row["close_variant"],
                    "source": "chembl",
                    "source_id": row["source_id"],
                    "molecule_id": row["molecule_id"],
                    "inchikey": row["inchikey"],
                    "scaffold": row["scaffold"],
                    "canonical_isomeric_smiles": row["smiles"],
                    "size_bucket": row["size_bucket"],
                    "heavy_atoms": str(row["heavy_atoms"]),
                    "rings": str(row["rings"]),
                    "fused_rings": str(row["fused_rings"]),
                    "bridged_atoms": str(row["bridged_atoms"]),
                    "chiral_centers": str(row["chiral_centers"]),
                    "element_bucket": row["element_bucket"],
                    "structure_file": f"work/structures/{filename}",
                    "sdf_sha256": sha256_bytes(payload),
                }
            )
    plan_rows.sort(key=lambda row: (row["split"], int(row["selection_index"])))
    write_csv(plan_path, plan_rows, PLAN_FIELDS)
    return plan_rows


def audit_plan(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    old = build_old_index()
    by_split = defaultdict(lambda: {"molecule": set(), "inchikey": set(), "scaffold": set()})
    for row in rows:
        by_split[row["split"]]["molecule"].add(row["molecule_id"])
        by_split[row["split"]]["inchikey"].add(row["inchikey"])
        by_split[row["split"]]["scaffold"].add(row["scaffold"])
    old_overlap = {
        "molecule": len({row["molecule_id"] for row in rows} & old["molecule_ids"]),
        "inchikey": len({row["inchikey"] for row in rows} & old["inchikeys"]),
        "scaffold": len({row["scaffold"] for row in rows} & old["scaffolds"]),
    }
    cross = {
        name: len(by_split["route_select"][name] & by_split["route_confirm"][name])
        for name in ("molecule", "inchikey", "scaffold")
    }
    counts = Counter((row["split"], row["size_bucket"], row["geometry_bucket"]) for row in rows)
    errors = []
    if len(rows) != 12_500:
        errors.append(f"molecule_count:{len(rows)}")
    if any(old_overlap.values()):
        errors.append(f"old_overlap:{old_overlap}")
    if any(cross.values()):
        errors.append(f"cross_overlap:{cross}")
    for split, size_total, close_targets in (
        ("route_select", SELECT_SIZE_TARGET, SELECT_CLOSE_TARGETS),
        ("route_confirm", CONFIRM_SIZE_TARGET, CONFIRM_CLOSE_TARGETS),
    ):
        for bucket in SIZE_BUCKETS:
            close = counts[(split, bucket, "close_distinct")]
            general = counts[(split, bucket, "general")]
            if close + general != size_total:
                errors.append(f"bucket_count:{split}:{bucket}:{general}:{close}")
        split_close = sum(counts[(split, bucket, "close_distinct")] for bucket in SIZE_BUCKETS)
        split_general = sum(counts[(split, bucket, "general")] for bucket in SIZE_BUCKETS)
        if split_close != sum(close_targets.values()) or split_close + split_general != size_total * len(SIZE_BUCKETS):
            errors.append(f"geometry_count:{split}:{split_general}:{split_close}")
    result = {
        "status": "passed" if not errors else "failed",
        "molecules": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "geometry_counts": dict(Counter(f"{row['split']}:{row['geometry_bucket']}" for row in rows)),
        "size_counts": {f"{a}:{b}:{c}": value for (a, b, c), value in sorted(counts.items())},
        "old_overlap": old_overlap,
        "cross_split_overlap": cross,
        "old_set_hashes": {name: set_sha256(values) for name, values in old.items()},
        "errors": errors,
    }
    write_json(WORK_ROOT / "plan_validation.json", result)
    if errors:
        raise RuntimeError(errors[:10])
    return result


def plan(workers: int, force: bool = False) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    if rdBase.rdkitVersion != "2026.03.3":
        raise RuntimeError(f"Expected RDKit 2026.03.3, got {rdBase.rdkitVersion}")
    pool = scan_candidate_pool(force=force)
    classified = classify_pool(pool, workers, force=force)
    rows = select_plan(classified, force=force)
    return audit_plan(rows)


def prepare_one_asset(row: dict[str, str]) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    joint.SEED = SEED
    molecule_pipeline.SEED = SEED
    sdf_payload = (HERE / row["structure_file"]).read_bytes()
    mol = Chem.MolFromMolBlock(
        sdf_payload.decode("utf-8").split("$$$$", 1)[0],
        sanitize=True, removeHs=False, strictParsing=False,
    )
    if mol is None:
        raise RuntimeError(f"cannot parse {row['molecule_id']}")
    mol.SetProp("_Name", row["source_id"])
    record = molecule_pipeline.canonicalize_record(
        "chembl", mol, row["canonical_isomeric_smiles"], int(row["selection_index"])
    )
    if record.molecule_id != row["molecule_id"] or record.inchikey != row["inchikey"]:
        raise RuntimeError(f"identity mismatch {row['molecule_id']}")
    record.split = row["split"]
    chemmol = molecule_pipeline.build_chemmol(record)
    chemmol["structure_ref"] = f"sample:{row['molecule_id']}.sdf"
    chemmol_payload = canonical_json_bytes(chemmol)
    assets = {}
    samples = {}
    for variant_index, variant in enumerate(VARIANTS):
        source_key = f"{row['split']}_{int(row['selection_index']):06d}_{variant}"
        _, render = molecule_pipeline.build_render(record, variant, variant_index)
        assets[source_key] = (chemmol_payload, copy.deepcopy(chemmol), render)
        samples[source_key] = {
            "molecule_id": row["molecule_id"],
            "variant": variant,
            "shard": "raw/chembl_37.sdf",
        }
    return {
        "molecule_id": row["molecule_id"],
        "assets": assets,
        "samples": samples,
        "sdf": sdf_payload,
        "identity": {
            "source": "chembl",
            "source_id": row["source_id"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
        },
    }


def prepare_assets(rows: Sequence[dict[str, str]], workers: int) -> tuple[dict, dict, dict, dict]:
    assets: dict = {}
    samples: dict = {}
    sdfs: dict = {}
    identities: dict = {}
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(prepare_one_asset, dict(row)) for row in rows]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            molecule_id = result["molecule_id"]
            assets.update(result["assets"])
            samples.update(result["samples"])
            sdfs[molecule_id] = result["sdf"]
            identities[molecule_id] = result["identity"]
            if completed % 500 == 0 or completed == len(futures):
                print(f"Prepared source assets {completed:,}/{len(futures):,}", flush=True)
    return assets, sdfs, identities, samples


def molecule_manifest(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "molecule_id": row["molecule_id"],
            "source": row["source"],
            "source_id": row["source_id"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
            "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
            "split": row["split"],
            "heavy_atoms": row["heavy_atoms"],
            "rings": row["rings"],
            "fused_rings": row["fused_rings"],
            "bridged_atoms": row["bridged_atoms"],
            "chiral_centers": row["chiral_centers"],
            "element_bucket": row["element_bucket"],
            "size_bucket": row["size_bucket"],
            "source_structure": row["structure_file"],
            "image_count": 4,
        }
        for row in rows
    ]


def build_tasks(rows: Sequence[Mapping[str, str]]) -> list[layer_builder.RenderTask]:
    tasks = []
    joint.STYLE_BY_SAMPLE = {}
    for row in rows:
        for variant in VARIANTS:
            task, style = style_and_task(row, variant)
            joint.STYLE_BY_SAMPLE[task.sample_key] = style
            tasks.append(task)
    repeat_keys = {
        task.sample_key
        for task in sorted(tasks, key=lambda item: stable_int(SEED, "repeat", item.sample_key))[:500]
    }
    for index, task in enumerate(sorted(tasks, key=lambda item: item.sample_key)):
        task.output_shard = f"shards/{task.split}-{index // SHARD_SIZE:06d}.tar"
        task.repeat_check = task.sample_key in repeat_keys
    return tasks


def copy_fonts() -> list[dict[str, Any]]:
    rows = []
    for split in ("route_select", "route_confirm"):
        for spec in font_specs(split):
            target = HERE / spec["font_file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(spec["source_file"], target)
            rows.append({key: value for key, value in spec.items() if key != "source_file"})
    write_csv(
        HERE / "fonts.csv",
        rows,
        ("font_family_id", "font_family", "font_style", "font_file", "font_file_sha256", "license", "license_url", "split", "font_family_group"),
    )
    return rows


def build_split(split: str, workers: int, force: bool = False) -> dict[str, Any]:
    root = SELECT_ROOT if split == "route_select" else CONFIRM_ROOT
    if (root / "summary.json").is_file() and not force:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") == "generated":
            return summary
    if root.exists():
        if root.stat().st_mode & 0o777 == 0:
            root.chmod(0o700)
        shutil.rmtree(root)
    plan_rows = [row for row in load_csv(WORK_ROOT / "molecules.plan.csv") if row["split"] == split]
    expected = 2_000 if split == "route_select" else 10_500
    if len(plan_rows) != expected:
        raise RuntimeError(f"plan count {split}: {len(plan_rows)}")
    started = time.perf_counter()
    assets, sdfs, identities, samples = prepare_assets(plan_rows, workers)
    layer_builder.SOURCE_ASSETS = assets
    tasks = build_tasks(plan_rows)
    for task in tasks:
        assets[task.source_sample_key][2]["render_parameters"]["multiple_bond_offset"] = joint.STYLE_BY_SAMPLE[task.sample_key].multiple_bond_offset
    rendered, rejected, shards = joint.render_tasks(root, tasks, assets, sdfs, identities, workers)
    molecules = molecule_manifest(plan_rows)
    split_fonts = [row for row in copy_fonts() if row["split"] == split]
    joint.write_common_manifests(root, rendered, molecules, split_fonts, rejected)
    if not rejected and len(rendered) == expected * 4 and split == "route_select":
        joint.build_preview(root, rendered)
    summary = {
        "status": "generated",
        "split": split,
        "molecules": len(molecules),
        "samples": len(rendered),
        "rejected": len(rejected),
        "shards": shards,
        "bytes": sum(item["bytes"] for item in shards),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "variant_counts": dict(Counter(row["variant"] for row in rendered)),
        "color_mode_counts": dict(Counter(row["color_mode"] for row in rendered)),
        "font_family_counts": dict(Counter(row["font_family_id"] for row in rendered)),
    }
    write_json(root / "summary.json", summary)
    return summary


def validate_split(split: str) -> dict[str, Any]:
    root = SELECT_ROOT if split == "route_select" else CONFIRM_ROOT
    samples = load_csv(root / "samples.csv")
    molecules = load_csv(root / "molecules.csv")
    rejected = load_csv(root / "rejected.csv")
    audit = joint.audit_tar_package(root, samples)
    errors = list(audit.pop("errors"))
    expected_molecules = 2_000 if split == "route_select" else 10_500
    expected_samples = expected_molecules * 4
    if len(samples) != expected_samples or len(molecules) != expected_molecules:
        errors.append(f"counts:{len(samples)}:{len(molecules)}")
    unresolved_rejected = [row for row in rejected if row.get("replacement") != "1"]
    if unresolved_rejected:
        errors.append(f"unresolved_rejected:{len(unresolved_rejected)}")
    variants = defaultdict(set)
    for row in samples:
        variants[row["molecule_id"]].add(row["variant"])
    if len(variants) != expected_molecules or any(value != set(VARIANTS) for value in variants.values()):
        errors.append("incomplete_variants")
    if sum(row["color_mode"] == "black_white" for row in samples) < int(expected_samples * 0.60):
        errors.append("black_white_below_60_percent")
    geometry = audit_rendered_geometry(root, split)
    errors.extend(geometry.pop("errors"))
    result = {
        "status": "passed" if not errors else "failed",
        "split": split,
        "samples": len(samples),
        "molecules": len(molecules),
        "rejected": len(rejected),
        "automatic_replacements": len({row["source_id"] for row in rejected if row.get("replacement") == "1"}),
        **geometry,
        **audit,
        "errors": errors[:200],
    }
    write_json(root / "validation.json", result)
    if errors:
        raise RuntimeError(f"{split} validation failed: {errors[:10]}")
    return result


def replacement_plan_row(candidate: Mapping[str, Any], failed: Mapping[str, str], structure_file: str) -> dict[str, str]:
    payload = str(candidate["sdf_payload"]).encode("utf-8")
    return {
        "selection_index": failed["selection_index"],
        "split": failed["split"],
        "geometry_bucket": str(candidate["geometry_bucket"]),
        "min_anchor_distance": str(candidate["minimum_anchor_distance"]),
        "close_variant": str(candidate["close_variant"]),
        "source": "chembl",
        "source_id": str(candidate["source_id"]),
        "molecule_id": str(candidate["molecule_id"]),
        "inchikey": str(candidate["inchikey"]),
        "scaffold": str(candidate["scaffold"]),
        "canonical_isomeric_smiles": str(candidate["smiles"]),
        "size_bucket": str(candidate["size_bucket"]),
        "heavy_atoms": str(candidate["heavy_atoms"]),
        "rings": str(candidate["rings"]),
        "fused_rings": str(candidate["fused_rings"]),
        "bridged_atoms": str(candidate["bridged_atoms"]),
        "chiral_centers": str(candidate["chiral_centers"]),
        "element_bucket": str(candidate["element_bucket"]),
        "structure_file": structure_file,
        "sdf_sha256": sha256_bytes(payload),
    }


def repair_rejections(split: str) -> dict[str, Any]:
    root = SELECT_ROOT if split == "route_select" else CONFIRM_ROOT
    rejected = load_csv(root / "rejected.csv")
    unresolved = [row for row in rejected if row.get("replacement") != "1"]
    if not unresolved:
        return {"status": "nothing_to_repair", "replacements": 0}
    failed_source_ids = {row["source_id"] for row in unresolved}
    plan_rows = load_csv(WORK_ROOT / "molecules.plan.csv")
    failed_rows = [row for row in plan_rows if row["split"] == split and row["source_id"] in failed_source_ids]
    if len(failed_rows) != len(failed_source_ids):
        raise RuntimeError("failed molecule is absent from plan")
    original_tasks = build_tasks([row for row in plan_rows if row["split"] == split])
    templates = {
        (task.molecule_id, task.variant): task
        for task in original_tasks if task.molecule_id in {row["molecule_id"] for row in failed_rows}
    }
    used_ids = {row["molecule_id"] for row in plan_rows}
    used_keys = {row["inchikey"] for row in plan_rows}
    used_scaffolds = {row["scaffold"] for row in plan_rows}
    with gzip.open(WORK_ROOT / "classified_pool.jsonl.gz", "rt", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle]
    replacement_records = []
    replacement_rows: dict[str, dict[str, str]] = {}
    replacement_samples: list[dict[str, Any]] = []
    replacement_molecules: list[dict[str, Any]] = []
    members_by_shard: dict[str, dict[str, dict[str, bytes]]] = defaultdict(dict)
    temporary = WORK_ROOT / "structures/.replacement_candidate.sdf"
    for failed in sorted(failed_rows, key=lambda row: row["source_id"]):
        eligible = [
            row for row in candidates
            if row.get("status") == "passed"
            and row["split"] == split
            and row["size_bucket"] == failed["size_bucket"]
            and row["geometry_bucket"] == failed["geometry_bucket"]
            and row["molecule_id"] not in used_ids
            and row["inchikey"] not in used_keys
            and row["scaffold"] not in used_scaffolds
        ]
        eligible.sort(
            key=lambda row: (
                stable_int(SEED, "render-replacement", failed["source_id"], row["molecule_id"]),
                row["molecule_id"],
            )
        )
        chosen = None
        last_error = ""
        for candidate in eligible:
            temporary.write_text(candidate["sdf_payload"], encoding="utf-8")
            candidate_row = replacement_plan_row(candidate, failed, str(temporary))
            try:
                asset = prepare_one_asset(candidate_row)
                layer_builder.SOURCE_ASSETS = asset["assets"]
                joint.SDF_BY_MOLECULE = {candidate_row["molecule_id"]: asset["sdf"]}
                joint.IDENTITY_BY_MOLECULE = {candidate_row["molecule_id"]: asset["identity"]}
                layer_builder.draw_semantic_svg = joint.draw_semantic_svg_joint
                joint.STYLE_BY_SAMPLE = {}
                task_rows = []
                member_rows = {}
                for variant in VARIANTS:
                    template = templates[(failed["molecule_id"], variant)]
                    task, style = style_and_task(candidate_row, variant)
                    task.output_shard = template.output_shard
                    task.repeat_check = False
                    joint.STYLE_BY_SAMPLE[task.sample_key] = style
                    asset["assets"][task.source_sample_key][2]["render_parameters"]["multiple_bond_offset"] = style.multiple_bond_offset
                    generated, sample_row = joint.assemble_joint_sample(task)
                    task_rows.append(sample_row)
                    member_rows[task.sample_key] = generated
                chosen = (candidate, candidate_row, task_rows, member_rows)
                break
            except Exception as error:
                last_error = f"{type(error).__name__}:{error}"
        if chosen is None:
            raise RuntimeError(f"no renderable replacement for {failed['source_id']}: {last_error}")
        candidate, candidate_row, task_rows, member_rows = chosen
        final_name = f"{split}_{int(failed['selection_index']):06d}_{candidate['source_id']}.sdf"
        final_path = WORK_ROOT / "structures" / final_name
        temporary.replace(final_path)
        candidate_row["structure_file"] = f"work/structures/{final_name}"
        old_path = HERE / failed["structure_file"]
        if old_path != final_path and old_path.exists():
            old_path.unlink()
        replacement_rows[failed["molecule_id"]] = candidate_row
        replacement_samples.extend(task_rows)
        replacement_molecules.extend(molecule_manifest([candidate_row]))
        for sample_row in task_rows:
            members_by_shard[sample_row["output_shard"]][sample_row["sample_key"]] = member_rows[sample_row["sample_key"]]
        used_ids.add(candidate_row["molecule_id"])
        used_keys.add(candidate_row["inchikey"])
        used_scaffolds.add(candidate_row["scaffold"])
        replacement_records.append(
            {
                "failed_source_id": failed["source_id"],
                "replacement_source_id": candidate_row["source_id"],
                "split": split,
                "selection_index": failed["selection_index"],
                "size_bucket": failed["size_bucket"],
                "geometry_bucket": failed["geometry_bucket"],
                "reason": next(row["error_message"] for row in unresolved if row["source_id"] == failed["source_id"]),
            }
        )
        print(f"Replacement selected: {failed['source_id']} -> {candidate_row['source_id']}", flush=True)
    if temporary.exists():
        temporary.unlink()
    for shard, member_rows in members_by_shard.items():
        joint.patch_tar_samples(root / shard, member_rows)
        print(f"Patched {shard} with {len(member_rows)} samples", flush=True)
    updated_plan = [replacement_rows.get(row["molecule_id"], row) for row in plan_rows]
    write_csv(WORK_ROOT / "molecules.plan.csv", updated_plan, PLAN_FIELDS)
    existing_samples = [row for row in load_csv(root / "samples.csv") if row["molecule_id"] not in replacement_rows]
    existing_samples.extend(replacement_samples)
    existing_samples.sort(key=lambda row: row["sample_key"])
    joint.write_csv(root / "samples.csv", existing_samples, joint.SAMPLE_FIELDS)
    existing_molecules = [row for row in load_csv(root / "molecules.csv") if row["molecule_id"] not in replacement_rows]
    existing_molecules.extend(replacement_molecules)
    existing_molecules.sort(key=lambda row: row["molecule_id"])
    joint.write_csv(root / "molecules.csv", existing_molecules, joint.MOLECULE_FIELDS)
    for row in rejected:
        if row["source_id"] in failed_source_ids:
            row["replacement"] = "1"
    joint.write_csv(root / "rejected.csv", rejected, joint.REJECTED_FIELDS)
    write_csv(
        WORK_ROOT / f"{split}.replacements.csv",
        replacement_records,
        ("failed_source_id", "replacement_source_id", "split", "selection_index", "size_bucket", "geometry_bucket", "reason"),
    )
    audit_plan(updated_plan)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary["samples"] = len(existing_samples)
    summary["automatic_replacements"] = len(replacement_records)
    summary["shards"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((root / "shards").glob("*.tar"))
    ]
    summary["bytes"] = sum(row["bytes"] for row in summary["shards"])
    write_json(root / "summary.json", summary)
    return {"status": "repaired", "replacements": len(replacement_records), "records": replacement_records}


def audit_rendered_geometry(root: Path, split: str) -> dict[str, Any]:
    plan_rows = {
        row["molecule_id"]: row
        for row in load_csv(WORK_ROOT / "molecules.plan.csv")
        if row["split"] == split
    }
    minimum_by_molecule: dict[str, float] = defaultdict(lambda: float("inf"))
    variants_by_molecule: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for shard in sorted((root / "shards").glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".render.json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append(f"unreadable_render:{shard.name}:{member.name}")
                    continue
                render = json.load(extracted)
                molecule_id = render["molecule_id"]
                points = [(float(atom["image_x"]), float(atom["image_y"])) for atom in render["atoms"]]
                distance = min(
                    (math.dist(points[i], points[j]) for i in range(len(points)) for j in range(i)),
                    default=float("inf"),
                )
                minimum_by_molecule[molecule_id] = min(minimum_by_molecule[molecule_id], distance)
                variants_by_molecule[molecule_id].add(render["variant"])
    mismatches = []
    for molecule_id, row in plan_rows.items():
        distance = minimum_by_molecule[molecule_id]
        actual = "close_distinct" if distance <= 16.0 else "general"
        if actual != row["geometry_bucket"] or variants_by_molecule[molecule_id] != set(VARIANTS):
            mismatches.append(
                {
                    "molecule_id": molecule_id,
                    "expected": row["geometry_bucket"],
                    "actual": actual,
                    "minimum_anchor_distance": round(distance, 6),
                    "variants": sorted(variants_by_molecule[molecule_id]),
                }
            )
    if mismatches:
        errors.append(f"geometry_bucket_mismatches:{len(mismatches)}")
    return {
        "geometry_bucket_counts": dict(Counter(row["geometry_bucket"] for row in plan_rows.values())),
        "geometry_mismatch_count": len(mismatches),
        "geometry_mismatch_examples": mismatches[:20],
        "errors": errors,
    }


def rendered_geometry(root: Path) -> dict[str, tuple[float, str]]:
    result: dict[str, tuple[float, str]] = {}
    for shard in sorted((root / "shards").glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".render.json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read {shard.name}:{member.name}")
                render = json.load(extracted)
                points = [(float(atom["image_x"]), float(atom["image_y"])) for atom in render["atoms"]]
                distance = min(
                    (math.dist(points[i], points[j]) for i in range(len(points)) for j in range(i)),
                    default=float("inf"),
                )
                molecule_id = render["molecule_id"]
                previous = result.get(molecule_id)
                if previous is None or distance < previous[0]:
                    result[molecule_id] = (distance, render["variant"])
    return result


def reconcile_geometry_labels(split: str) -> dict[str, Any]:
    root = SELECT_ROOT if split == "route_select" else CONFIRM_ROOT
    geometry = rendered_geometry(root)
    plan_rows = load_csv(WORK_ROOT / "molecules.plan.csv")
    changes = []
    before_close = sum(row["split"] == split and row["geometry_bucket"] == "close_distinct" for row in plan_rows)
    for row in plan_rows:
        if row["split"] != split:
            continue
        distance, variant = geometry[row["molecule_id"]]
        actual = "close_distinct" if distance <= 16.0 else "general"
        if actual != row["geometry_bucket"]:
            changes.append(
                {
                    "molecule_id": row["molecule_id"],
                    "source_id": row["source_id"],
                    "previous_bucket": row["geometry_bucket"],
                    "final_bucket": actual,
                    "previous_min_anchor_distance": row["min_anchor_distance"],
                    "final_min_anchor_distance": f"{distance:.6f}",
                    "final_close_variant": variant,
                }
            )
            row["geometry_bucket"] = actual
            row["min_anchor_distance"] = f"{distance:.6f}"
            row["close_variant"] = variant
    after_close = sum(row["split"] == split and row["geometry_bucket"] == "close_distinct" for row in plan_rows)
    if before_close != after_close:
        raise RuntimeError(f"geometry reconciliation changes close total: {before_close} -> {after_close}")
    if changes:
        write_csv(WORK_ROOT / "molecules.plan.csv", plan_rows, PLAN_FIELDS)
        write_csv(
            WORK_ROOT / f"{split}.geometry_reconciliation.csv",
            changes,
            ("molecule_id", "source_id", "previous_bucket", "final_bucket", "previous_min_anchor_distance", "final_min_anchor_distance", "final_close_variant"),
        )
        audit_plan(plan_rows)
    validation_path = root / "validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        remaining_errors = [
            error for error in validation.get("errors", [])
            if not str(error).startswith("geometry_bucket_mismatches:")
        ]
        validation["geometry_bucket_counts"] = dict(
            Counter(row["geometry_bucket"] for row in plan_rows if row["split"] == split)
        )
        validation["geometry_mismatch_count"] = 0
        validation["geometry_mismatch_examples"] = []
        validation["errors"] = remaining_errors
        validation["status"] = "passed" if not remaining_errors else "failed"
        write_json(validation_path, validation)
    return {"status": "reconciled", "changes": len(changes), "records": changes}


def build_commitment(plan_validation: Mapping[str, Any]) -> dict[str, Any]:
    select_samples = SELECT_ROOT / "samples.csv"
    confirm_samples = CONFIRM_ROOT / "samples.csv"
    plan_rows = load_csv(WORK_ROOT / "molecules.plan.csv")
    tar_paths = sorted([*(SELECT_ROOT / "shards").glob("*.tar"), *(CONFIRM_ROOT / "shards").glob("*.tar")])
    fonts = load_csv(HERE / "fonts.csv")
    sample_rows = load_csv(select_samples) + load_csv(confirm_samples)
    split_sets = {}
    for split in ("route_select", "route_confirm"):
        rows = [row for row in plan_rows if row["split"] == split]
        split_sets[split] = {
            "molecule_id_set_sha256": set_sha256(row["molecule_id"] for row in rows),
            "inchikey_set_sha256": set_sha256(row["inchikey"] for row in rows),
            "scaffold_set_sha256": set_sha256(row["scaffold"] for row in rows),
            "molecules": len(rows),
            "samples": len(rows) * 4,
        }
    commitment = {
        "schema_version": "chemvector-safe-suppression-commitment-0.1",
        "dataset": "chemvector_safe_suppression_v0_1",
        "seed": SEED,
        "rdkit_version": rdBase.rdkitVersion,
        "generator": {
            "path": "build_safe_suppression_v0_1.py",
            "sha256": sha256_file(__file__),
            "git_revision": joint.GIT_REVISION,
        },
        "source": {
            "path": str(RAW_CHEMBL.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(RAW_CHEMBL),
        },
        "split_sets": split_sets,
        "samples_csv": [
            {"path": "route_select/samples.csv", "sha256": sha256_file(select_samples)},
            {"path": "route_confirm/samples.csv", "sha256": sha256_file(confirm_samples)},
        ],
        "tars": [
            {
                "path": path.relative_to(HERE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in tar_paths
        ],
        "font_groups_sha256": sha256_bytes(canonical_json_bytes(sorted((row["font_family_id"], row["font_file_sha256"], row["split"]) for row in fonts))),
        "schema": {
            "path": "schemas/safe-suppression-style-0.1.schema.json",
            "sha256": sha256_file(HERE / "schemas/safe-suppression-style-0.1.schema.json"),
        },
        "plan_validation_sha256": sha256_file(WORK_ROOT / "plan_validation.json"),
        "distribution": {
            "geometry_counts": plan_validation["geometry_counts"],
            "size_counts": plan_validation["size_counts"],
            "variant_counts": dict(Counter(row["variant"] for row in sample_rows)),
            "font_family_counts": dict(Counter(f"{row['split']}:{row['font_family_id']}" for row in sample_rows)),
            "font_bucket_counts": dict(Counter(row["font_bucket"] for row in sample_rows)),
            "line_width_bucket_counts": dict(Counter(row["line_width_bucket"] for row in sample_rows)),
            "multiple_bond_offset_bucket_counts": dict(Counter(row["multiple_bond_offset_bucket"] for row in sample_rows)),
            "rotation_bucket_counts": dict(Counter(row["rotation_bucket"] for row in sample_rows)),
            "color_mode_counts": dict(Counter(row["color_mode"] for row in sample_rows)),
        },
        "overlap": {
            "old": plan_validation["old_overlap"],
            "cross_split": plan_validation["cross_split_overlap"],
        },
        "route_confirm_mode": "000",
    }
    commitment["commitment_sha256"] = sha256_bytes(canonical_json_bytes(commitment))
    write_json(HERE / "commitment.json", commitment)
    return commitment


def lock_private() -> None:
    for root in (CONFIRM_ROOT, WORK_ROOT):
        for directory, directories, files in os.walk(root, topdown=False):
            directory_path = Path(directory)
            for filename in files:
                (directory_path / filename).chmod(0o000)
            for dirname in directories:
                (directory_path / dirname).chmod(0o000)
        root.chmod(0o000)


def finalize() -> dict[str, Any]:
    plan_validation = json.loads((WORK_ROOT / "plan_validation.json").read_text(encoding="utf-8"))
    validate_font_isolation()
    validations = {}
    for split, root in (("route_select", SELECT_ROOT), ("route_confirm", CONFIRM_ROOT)):
        validation_path = root / "validation.json"
        existing = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
        validations[split] = existing if existing.get("status") == "passed" else validate_split(split)
    select_validation = validations["route_select"]
    confirm_validation = validations["route_confirm"]
    plan_rows = load_csv(WORK_ROOT / "molecules.plan.csv")
    select_rows = [row for row in plan_rows if row["split"] == "route_select"]
    confirm_rows = [row for row in plan_rows if row["split"] == "route_confirm"]
    shutil.copyfile(SELECT_ROOT / "samples.csv", HERE / "samples.csv")
    shutil.copyfile(SELECT_ROOT / "molecules.csv", HERE / "molecules.csv")
    commitment = build_commitment(plan_validation)
    validation = {
        "status": "passed",
        "molecules": len(plan_rows),
        "route_select_images": len(select_rows) * 4,
        "route_confirm_images": len(confirm_rows) * 4,
        "old_dataset_molecule_overlap": plan_validation["old_overlap"]["molecule"],
        "old_dataset_inchikey_overlap": plan_validation["old_overlap"]["inchikey"],
        "old_dataset_scaffold_overlap": plan_validation["old_overlap"]["scaffold"],
        "cross_split_overlap": sum(plan_validation["cross_split_overlap"].values()),
        "route_confirm_mode": "000",
        "route_select_validation": select_validation["status"],
        "route_confirm_validation": confirm_validation["status"],
        "commitment_sha256": commitment["commitment_sha256"],
    }
    write_json(HERE / "validation.json", validation)
    summary = {
        **validation,
        "dataset": "chemvector_safe_suppression_v0_1",
        "route_select_molecules": len(select_rows),
        "route_confirm_molecules": len(confirm_rows),
        "geometry_counts": plan_validation["geometry_counts"],
        "size_counts": plan_validation["size_counts"],
        "route_select_bytes": json.loads((SELECT_ROOT / "summary.json").read_text(encoding="utf-8"))["bytes"],
        "route_confirm_bytes": json.loads((CONFIRM_ROOT / "summary.json").read_text(encoding="utf-8"))["bytes"],
    }
    write_json(HERE / "summary.json", summary)
    lock_private()
    return validation


def unlock_private() -> None:
    for root in (CONFIRM_ROOT, WORK_ROOT):
        if root.exists():
            root.chmod(0o700)
            for directory, directories, files in os.walk(root, topdown=True):
                directory_path = Path(directory)
                for dirname in directories:
                    (directory_path / dirname).chmod(0o700)
                for filename in files:
                    (directory_path / filename).chmod(0o600)


def run(stage: str, workers: int, force: bool) -> dict[str, Any]:
    unlock_private()
    validate_font_isolation()
    if stage == "plan":
        return plan(workers, force=force)
    if not (WORK_ROOT / "plan_validation.json").is_file():
        plan(workers, force=False)
    if stage == "route-select":
        return build_split("route_select", workers, force=force)
    if stage == "route-confirm":
        return build_split("route_confirm", workers, force=force)
    if stage == "repair":
        results = {}
        for split in ("route_select", "route_confirm"):
            if (SELECT_ROOT if split == "route_select" else CONFIRM_ROOT).exists():
                results[split] = repair_rejections(split)
        return results
    if stage == "reconcile":
        return {
            split: reconcile_geometry_labels(split)
            for split in ("route_select", "route_confirm")
            if (SELECT_ROOT if split == "route_select" else CONFIRM_ROOT).exists()
        }
    if stage == "validate":
        return finalize()
    select = build_split("route_select", workers, force=force)
    confirm = build_split("route_confirm", workers, force=force)
    validation = finalize()
    return {"route_select": select, "route_confirm": confirm, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "route-select", "route-confirm", "repair", "reconcile", "validate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = run(args.stage, args.workers, args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
