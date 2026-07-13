#!/usr/bin/env python3
"""Benchmark strict ChemMol rendering with independent parallel tar writers."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fastjsonschema
from rdkit import Chem, RDLogger, rdBase

import build_webdataset_sample as pipeline


WORKERS = 32
INPUT = Path("webdataset_sample_1k")
OUTPUT = Path("parallel_render_benchmark")


def load_record(row: dict) -> pipeline.MoleculeRecord:
    path = INPUT / row["structure_ref"]
    mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False, strictParsing=False)
    if mol is None:
        raise ValueError(f"Cannot load {path}")
    return pipeline.MoleculeRecord(
        source=row["source"],
        source_id=row["source_id"],
        molecule_id=row["molecule_id"],
        mol=mol,
        source_atom_indices=list(range(mol.GetNumAtoms())),
        smiles=row["canonical_isomeric_smiles"],
        inchikey=row["inchikey"],
        scaffold=row["scaffold"],
        stratum=row["stratum"],
        split=row["split"],
    )


def render_chunk(worker_id: int, rows: list[dict]) -> dict:
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    started = time.perf_counter()
    chemmol_schema = json.loads(Path("schemas/chemmol-0.1.schema.json").read_text())
    render_schema = json.loads(Path("schemas/render-0.1.schema.json").read_text())
    chemmol_validator = fastjsonschema.compile(chemmol_schema)
    render_validator = fastjsonschema.compile(render_schema)
    output = OUTPUT / f"benchmark-{worker_id:02d}.tar"
    samples = 0
    maximum_delta = 0.0

    with tarfile.open(output, "w") as archive:
        for molecule_index, row in enumerate(rows):
            record = load_record(row)
            chemmol = pipeline.build_chemmol(record)
            chemmol["structure_ref"] = row["structure_ref"]
            chemmol_validator(chemmol)
            pipeline.validate_chemical_roundtrip(chemmol)
            for variant in pipeline.VARIANTS:
                png, render = pipeline.build_render(record, variant, molecule_index)
                render_validator(render)
                delta = pipeline.validate_geometry_roundtrip(chemmol, render)
                maximum_delta = max(maximum_delta, delta)
                sequence = pipeline.serialize_chemmolseq(chemmol, render)
                key = f"worker{worker_id:02d}_{molecule_index:06d}_{variant}"
                meta = {
                    "schema_version": "sample-meta-0.1",
                    "sample_key": key,
                    "molecule_id": record.molecule_id,
                    "source": record.source,
                    "source_id": record.source_id,
                    "split": record.split,
                    "variant": variant,
                    "structure_ref": row["structure_ref"],
                    "rdkit_version": rdBase.rdkitVersion,
                    "seed": pipeline.SEED,
                    "chemmolseq_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                }
                pipeline.tar_add_bytes(archive, f"{key}.png", png)
                pipeline.tar_add_bytes(archive, f"{key}.chemmol.json", pipeline.json_bytes(chemmol))
                pipeline.tar_add_bytes(archive, f"{key}.render.json", pipeline.json_bytes(render))
                pipeline.tar_add_bytes(archive, f"{key}.meta.json", pipeline.json_bytes(meta))
                pipeline.tar_add_bytes(archive, f"{key}.chemmolseq.txt", sequence.encode())
                samples += 1
    return {
        "worker_id": worker_id,
        "molecules": len(rows),
        "samples": samples,
        "bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "max_geometry_delta_pixels": maximum_delta,
    }


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    rows = list(csv.DictReader((INPUT / "molecules.csv").open()))
    chunks = [[] for _ in range(WORKERS)]
    for index, row in enumerate(rows):
        chunks[index % WORKERS].append(row)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(render_chunk, worker_id, chunk) for worker_id, chunk in enumerate(chunks)]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    samples = sum(result["samples"] for result in results)
    summary = {
        "workers": WORKERS,
        "molecules": sum(result["molecules"] for result in results),
        "samples": samples,
        "bytes": sum(result["bytes"] for result in results),
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(samples / elapsed, 2),
        "molecules_per_second": round(len(rows) / elapsed, 2),
        "slowest_worker_seconds": round(max(result["elapsed_seconds"] for result in results), 3),
        "max_geometry_delta_pixels": max(result["max_geometry_delta_pixels"] for result in results),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
