#!/usr/bin/env python3
"""Build the full v0 scaffold split and parallel WebDataset shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fastjsonschema
from rdkit import Chem, RDLogger, rdBase

import build_webdataset_sample as pipeline


OUTPUT = Path("webdataset_full_v0")
INPUTS = (
    ("chembl", Path("extracted/prepared/train/chembl_150k.sdf")),
    ("coconut", Path("extracted/prepared/train/coconut_complex_30k.sdf")),
)
MOLECULES_PER_SHARD = 1250
DEFAULT_WORKERS = 32


@dataclass(slots=True)
class PlanItem:
    source: str
    source_id: str
    molecule_id: str
    smiles: str
    inchikey: str
    scaffold: str
    stratum: str
    source_atom_indices: list[int]
    molecule: bytes
    split: str = ""
    split_index: int = -1
    shard_index: int = -1


CHEMMOL_VALIDATOR = None
RENDER_VALIDATOR = None


def worker_init() -> None:
    global CHEMMOL_VALIDATOR, RENDER_VALIDATOR
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    CHEMMOL_VALIDATOR = fastjsonschema.compile(
        json.loads(Path("schemas/chemmol-0.1.schema.json").read_text())
    )
    RENDER_VALIDATOR = fastjsonschema.compile(
        json.loads(Path("schemas/render-0.1.schema.json").read_text())
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_plan(replan: bool = False) -> dict:
    plan_summary_path = OUTPUT / "plan_summary.json"
    input_dir = OUTPUT / "work" / "input_shards"
    if plan_summary_path.exists() and not replan:
        return json.loads(plan_summary_path.read_text())

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    input_dir.mkdir(parents=True)
    shutil.copytree(
        "schemas",
        OUTPUT / "schemas",
        ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__", "*.pyc"),
    )
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    started = time.perf_counter()
    items: list[PlanItem] = []
    rejected = []
    seen_smiles = set()

    for source, path in INPUTS:
        supplier = Chem.SDMolSupplier(
            str(path), sanitize=True, removeHs=False, strictParsing=False
        )
        for source_index, mol in enumerate(supplier, 1):
            if mol is None:
                rejected.append(
                    {"source": source, "source_index": source_index, "error": "parse_or_sanitize"}
                )
                continue
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if smiles in seen_smiles:
                    rejected.append(
                        {"source": source, "source_index": source_index, "error": "duplicate_structure"}
                    )
                    continue
                record = pipeline.canonicalize_record(source, mol, smiles, source_index)
            except Exception as error:
                rejected.append(
                    {
                        "source": source,
                        "source_index": source_index,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            seen_smiles.add(smiles)
            items.append(
                PlanItem(
                    source=record.source,
                    source_id=record.source_id,
                    molecule_id=record.molecule_id,
                    smiles=record.smiles,
                    inchikey=record.inchikey,
                    scaffold=record.scaffold,
                    stratum=record.stratum,
                    source_atom_indices=record.source_atom_indices,
                    molecule=record.mol.ToBinary(Chem.PropertyPickleOptions.AllProps),
                )
            )
            if len(items) % 10_000 == 0:
                print(f"plan: canonicalized {len(items):,} molecules", flush=True)

    split_counts = pipeline.assign_scaffold_splits(items)
    by_split: dict[str, list[PlanItem]] = defaultdict(list)
    for item in items:
        by_split[item.split].append(item)
    for split, split_items in by_split.items():
        split_items.sort(
            key=lambda item: pipeline.stable_int(pipeline.SEED, "full-plan", item.molecule_id)
        )
        for split_index, item in enumerate(split_items):
            item.split_index = split_index
            item.shard_index = split_index // MOLECULES_PER_SHARD

    writers: dict[tuple[str, int], Chem.SDWriter] = {}
    plan_rows = []
    try:
        for count, item in enumerate(items, 1):
            key = (item.split, item.shard_index)
            if key not in writers:
                path = input_dir / f"{item.split}-{item.shard_index:06d}.sdf"
                writers[key] = Chem.SDWriter(str(path))
            mol = Chem.Mol(item.molecule)
            mol.SetProp("plan_source", item.source)
            mol.SetProp("plan_source_id", item.source_id)
            mol.SetProp("plan_molecule_id", item.molecule_id)
            mol.SetProp("plan_smiles", item.smiles)
            mol.SetProp("plan_inchikey", item.inchikey)
            mol.SetProp("plan_scaffold", item.scaffold)
            mol.SetProp("plan_stratum", item.stratum)
            mol.SetProp("plan_split", item.split)
            mol.SetIntProp("plan_split_index", item.split_index)
            mol.SetIntProp("plan_shard_index", item.shard_index)
            mol.SetProp("plan_source_atom_indices", json.dumps(item.source_atom_indices))
            writers[key].write(mol)
            plan_rows.append(
                {
                    "molecule_id": item.molecule_id,
                    "source": item.source,
                    "source_id": item.source_id,
                    "split": item.split,
                    "split_index": item.split_index,
                    "shard_index": item.shard_index,
                    "scaffold": item.scaffold,
                    "stratum": item.stratum,
                    "canonical_isomeric_smiles": item.smiles,
                    "inchikey": item.inchikey,
                }
            )
            if count % 20_000 == 0:
                print(f"plan: wrote {count:,}/{len(items):,} molecules to input shards", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    write_csv(OUTPUT / "plan_molecules.csv", plan_rows)
    write_csv(OUTPUT / "plan_rejected.csv", rejected)
    summary = {
        "planned_molecules": len(items),
        "planned_samples": len(items) * len(pipeline.VARIANTS),
        "source_counts": dict(Counter(item.source for item in items)),
        "split_counts": split_counts,
        "input_shards": len(writers),
        "molecules_per_shard": MOLECULES_PER_SHARD,
        "rejected_during_plan": len(rejected),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "rdkit_version": rdBase.rdkitVersion,
        "seed": pipeline.SEED,
    }
    plan_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def record_from_planned_mol(mol: Chem.Mol) -> pipeline.MoleculeRecord:
    return pipeline.MoleculeRecord(
        source=mol.GetProp("plan_source"),
        source_id=mol.GetProp("plan_source_id"),
        molecule_id=mol.GetProp("plan_molecule_id"),
        mol=mol,
        source_atom_indices=json.loads(mol.GetProp("plan_source_atom_indices")),
        smiles=mol.GetProp("plan_smiles"),
        inchikey=mol.GetProp("plan_inchikey"),
        scaffold=mol.GetProp("plan_scaffold"),
        stratum=mol.GetProp("plan_stratum"),
        split=mol.GetProp("plan_split"),
    )


def render_shard(input_path_string: str) -> dict:
    input_path = Path(input_path_string)
    stem = input_path.stem
    complete_path = OUTPUT / "manifests" / "shards" / f"{stem}.complete.json"
    if complete_path.exists():
        result = json.loads(complete_path.read_text())
        result["resumed"] = True
        return result

    started = time.perf_counter()
    tar_path = OUTPUT / "shards" / f"{stem}.tar"
    temp_tar = tar_path.with_suffix(".tar.tmp")
    structure_dir = OUTPUT / "structures" / stem
    shard_manifest_dir = OUTPUT / "manifests" / "shards"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    shard_manifest_dir.mkdir(parents=True, exist_ok=True)
    if temp_tar.exists():
        temp_tar.unlink()
    if structure_dir.exists():
        shutil.rmtree(structure_dir)
    structure_dir.mkdir(parents=True)

    molecule_rows = []
    sample_rows = []
    rejected_rows = []
    maximum_delta = 0.0
    supplier = Chem.SDMolSupplier(
        str(input_path), sanitize=True, removeHs=False, strictParsing=False
    )
    with tarfile.open(temp_tar, "w") as archive:
        for local_index, mol in enumerate(supplier):
            if mol is None:
                rejected_rows.append(
                    {"input_shard": stem, "local_index": local_index, "error": "parse_or_sanitize"}
                )
                continue
            record = record_from_planned_mol(mol)
            split_index = mol.GetIntProp("plan_split_index")
            structure_name = (
                f"{pipeline.clean_identifier(record.source)}_"
                f"{pipeline.clean_identifier(record.source_id)}.sdf"
            )
            structure_ref = f"structures/{stem}/{structure_name}"
            try:
                chemmol = pipeline.build_chemmol(record)
                chemmol["structure_ref"] = structure_ref
                CHEMMOL_VALIDATOR(chemmol)
                pipeline.validate_chemical_roundtrip(chemmol)
                rendered = []
                molecule_max_delta = 0.0
                for variant in pipeline.VARIANTS:
                    png, render = pipeline.build_render(record, variant, split_index)
                    RENDER_VALIDATOR(render)
                    delta = pipeline.validate_geometry_roundtrip(chemmol, render)
                    molecule_max_delta = max(molecule_max_delta, delta)
                    sequence = pipeline.serialize_chemmolseq(chemmol, render)
                    rendered.append((variant, png, render, sequence))
            except Exception as error:
                rejected_rows.append(
                    {
                        "input_shard": stem,
                        "local_index": local_index,
                        "molecule_id": record.molecule_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            structure_path = structure_dir / structure_name
            writer = Chem.SDWriter(str(structure_path))
            writer.write(record.mol)
            writer.close()
            molecule_rows.append(
                {
                    "molecule_id": record.molecule_id,
                    "source": record.source,
                    "source_id": record.source_id,
                    "split": record.split,
                    "split_index": split_index,
                    "shard": f"shards/{stem}.tar",
                    "scaffold": record.scaffold,
                    "stratum": record.stratum,
                    "canonical_isomeric_smiles": record.smiles,
                    "inchikey": record.inchikey,
                    "structure_ref": structure_ref,
                }
            )
            for variant, png, render, sequence in rendered:
                sample_key = f"{record.split}_{split_index:06d}_{variant}"
                meta = {
                    "schema_version": "sample-meta-0.1",
                    "sample_key": sample_key,
                    "molecule_id": record.molecule_id,
                    "source": record.source,
                    "source_id": record.source_id,
                    "split": record.split,
                    "variant": variant,
                    "structure_ref": structure_ref,
                    "rdkit_version": rdBase.rdkitVersion,
                    "seed": pipeline.SEED,
                    "chemmolseq_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                }
                pipeline.tar_add_bytes(archive, f"{sample_key}.png", png)
                pipeline.tar_add_bytes(archive, f"{sample_key}.chemmol.json", pipeline.json_bytes(chemmol))
                pipeline.tar_add_bytes(archive, f"{sample_key}.render.json", pipeline.json_bytes(render))
                pipeline.tar_add_bytes(archive, f"{sample_key}.meta.json", pipeline.json_bytes(meta))
                pipeline.tar_add_bytes(archive, f"{sample_key}.chemmolseq.txt", sequence.encode())
                sample_rows.append(
                    {
                        "sample_key": sample_key,
                        "molecule_id": record.molecule_id,
                        "split": record.split,
                        "variant": variant,
                        "shard": f"shards/{stem}.tar",
                        "color_mode": render["render_parameters"]["color_mode"],
                        "layout": render["render_parameters"]["layout"],
                        "stress_operation": render["render_parameters"]["stress_operation"] or "",
                    }
                )
            maximum_delta = max(maximum_delta, molecule_max_delta)

    os.replace(temp_tar, tar_path)
    write_csv(shard_manifest_dir / f"{stem}.molecules.csv", molecule_rows)
    write_csv(shard_manifest_dir / f"{stem}.samples.csv", sample_rows)
    write_csv(shard_manifest_dir / f"{stem}.rejected.csv", rejected_rows)
    result = {
        "input_shard": stem,
        "molecules": len(molecule_rows),
        "samples": len(sample_rows),
        "rejected": len(rejected_rows),
        "bytes": tar_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_geometry_delta_pixels": maximum_delta,
        "resumed": False,
    }
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def combine_manifests(results: list[dict], plan_summary: dict, elapsed: float) -> dict:
    manifest_dir = OUTPUT / "manifests" / "shards"
    combined = {}
    for kind in ("molecules", "samples", "rejected"):
        rows = []
        for path in sorted(manifest_dir.glob(f"*.{kind}.csv")):
            with path.open(encoding="utf-8", newline="") as stream:
                rows.extend(csv.DictReader(stream))
        write_csv(OUTPUT / f"{kind}.csv", rows)
        combined[kind] = rows

    samples = combined["samples"]
    molecules = combined["molecules"]
    summary = {
        "planned_molecules": plan_summary["planned_molecules"],
        "valid_molecules": len(molecules),
        "samples": len(samples),
        "rejected_during_render": len(combined["rejected"]),
        "source_counts": dict(Counter(row["source"] for row in molecules)),
        "split_counts": dict(Counter(row["split"] for row in molecules)),
        "variant_counts": dict(Counter(row["variant"] for row in samples)),
        "color_mode_counts": dict(Counter(row["color_mode"] for row in samples)),
        "stress_operation_counts": dict(
            Counter(row["stress_operation"] for row in samples if row["stress_operation"])
        ),
        "shards": len(results),
        "bytes": sum(result["bytes"] for result in results),
        "elapsed_seconds_this_run": round(elapsed, 3),
        "max_geometry_delta_pixels": max(
            (result["max_geometry_delta_pixels"] for result in results), default=0.0
        ),
        "rdkit_version": rdBase.rdkitVersion,
        "seed": pipeline.SEED,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--replan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_summary = build_plan(replan=args.replan)
    input_paths = sorted((OUTPUT / "work" / "input_shards").glob("*.sdf"))
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as executor:
        futures = {executor.submit(render_shard, str(path)): path for path in input_paths}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"render: {completed}/{len(input_paths)} shards complete; "
                f"molecules={sum(item['molecules'] for item in results):,}; "
                f"rejected={sum(item['rejected'] for item in results):,}",
                flush=True,
            )
    summary = combine_manifests(results, plan_summary, time.perf_counter() - started)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
