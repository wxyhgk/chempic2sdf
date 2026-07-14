#!/usr/bin/env python3
"""Benchmark parallel ChemVector inference on rendered MR-TADF structures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import torch

from Data.sdf.chemsvg_sequence_v4 import mol_from_chemsvg_v4
from chemvector.evaluate import load_model
from chemvector.image_io import image_tensor
from chemvector.parallel_v6 import generate_parallel_chemsvg_v6, strip_autoregressive_modules
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image
from chemvector.sdf_io import write_sdf_v4


def failure_category(error: Exception) -> str:
    message = str(error).lower()
    if "not connected" in message or "component" in message:
        return "disconnected"
    if "valence" in message:
        return "valence"
    if "kekul" in message:
        return "kekulize"
    return "other"


def summarize_records(records: list[dict]) -> dict:
    samples = len(records)
    difficulty_groups: dict[str, list[dict]] = defaultdict(list)
    failure_counts = Counter()
    for record in records:
        difficulty_groups[record["difficulty"]].append(record)
        if record.get("failure_category"):
            failure_counts[record["failure_category"]] += 1

    def rates(group: list[dict]) -> dict[str, float | int]:
        count = len(group)
        return {
            "samples": count,
            "sdf_valid_rate": sum(bool(row["sdf_valid"]) for row in group)
            / max(count, 1),
            "formula_match_rate": sum(bool(row["formula_match"]) for row in group)
            / max(count, 1),
            "nonstereo_structure_match_rate": sum(
                bool(row["nonstereo_structure_match"]) for row in group
            )
            / max(count, 1),
            "chemical_match_rate": sum(bool(row["chemical_match"]) for row in group)
            / max(count, 1),
        }

    summary = rates(records)
    summary.update(
        {
            "average_seconds": sum(float(row["seconds"]) for row in records)
            / max(samples, 1),
            "atom_count_match_rate": sum(
                bool(row["atom_count_match"]) for row in records
            )
            / max(samples, 1),
            "bond_count_match_rate": sum(
                bool(row["bond_count_match"]) for row in records
            )
            / max(samples, 1),
            "failure_categories": dict(sorted(failure_counts.items())),
            "difficulty_rates": {
                name: rates(group) for name, group in sorted(difficulty_groups.items())
            },
        }
    )
    return summary


def canonical_smiles(mol: Chem.Mol, *, isomeric: bool) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_benchmark_config(config_path: Path, config: dict) -> None:
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            keys = sorted(set(existing) | set(config))
            mismatches = {
                key: (existing.get(key), config.get(key))
                for key in keys
                if existing.get(key) != config.get(key)
            }
            raise ValueError(
                "Existing benchmark output uses a different configuration: "
                f"{mismatches}"
            )
        return
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/chemvector-v5-vertex-10k/last.pt",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="Data/benchmarks/mr_tadf_v1/manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluations/mr-tadf-v1-v071",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument("--maximum-vertices", type=int, default=128)
    parser.add_argument("--save-sdf", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    if args.cpu_threads < 1 or args.maximum_vertices < 1:
        raise ValueError("CPU threads and maximum vertices must be positive")
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    benchmark_manifest = Path(args.benchmark_manifest)
    benchmark_root = benchmark_manifest.parent
    with benchmark_manifest.open(newline="", encoding="utf-8") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    if args.samples is not None:
        benchmark_rows = benchmark_rows[: args.samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records = []
    if records_path.is_file():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {record["molecule_id"] for record in records}
    checkpoint_path = Path(args.checkpoint)
    config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "benchmark_manifest": str(benchmark_manifest),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest),
        "device": args.device,
        "vertex_threshold": args.vertex_threshold,
        "edge_threshold": args.edge_threshold,
        "ocr_roi_radius": args.ocr_roi_radius,
        "maximum_vertices": args.maximum_vertices,
    }
    config_path = output_dir / "benchmark_config.json"
    ensure_benchmark_config(config_path, config)

    model, _ = load_model(checkpoint_path, device)
    strip_autoregressive_modules(model)
    sdf_dir = output_dir / "sdf"
    if args.save_sdf:
        sdf_dir.mkdir(parents=True, exist_ok=True)

    for row in benchmark_rows:
        molecule_id = row["molecule_id"]
        if molecule_id in completed:
            continue
        image_path = benchmark_root / row["image_file"]
        image = Image.open(image_path).convert("RGB")
        pixels = image_tensor(image).unsqueeze(0).to(device)
        expected = Chem.MolFromMolFile(
            str(benchmark_root / row["sdf_file"]),
            removeHs=False,
        )
        if expected is None:
            raise ValueError(f"Cannot read benchmark ground truth: {row['sdf_file']}")
        expected_nonstereo = canonical_smiles(expected, isomeric=False)
        record = {
            "molecule_id": molecule_id,
            "difficulty": row["difficulty"],
            "maximum_to_median_bond_ratio": float(
                row["maximum_to_median_bond_ratio"]
            ),
            "expected_atoms": int(row["atoms"]),
            "expected_bonds": int(row["bonds"]),
            "predicted_atoms": 0,
            "predicted_bonds": 0,
            "sdf_valid": False,
            "formula_match": False,
            "atom_count_match": False,
            "bond_count_match": False,
            "nonstereo_structure_match": False,
            "chemical_match": False,
        }
        started = time.perf_counter()
        try:
            generated = generate_parallel_chemsvg_v6(
                model,
                pixels,
                vertex_threshold=args.vertex_threshold,
                maximum_vertices=args.maximum_vertices,
                edge_threshold=args.edge_threshold,
                ocr_roi_radius=args.ocr_roi_radius,
            )
            repaired = repair_chemsvg_v4_from_image(
                generated.sequence,
                image,
                config=InkGraphRepairConfig(aromatic_vote_source="vertices"),
            )
            predicted = mol_from_chemsvg_v4(repaired.sequence)
            actual_smiles = canonical_smiles(predicted, isomeric=True)
            actual_nonstereo = canonical_smiles(predicted, isomeric=False)
            actual_inchikey = Chem.MolToInchiKey(predicted)
            record.update(
                {
                    "sdf_valid": True,
                    "predicted_atoms": predicted.GetNumAtoms(),
                    "predicted_bonds": predicted.GetNumBonds(),
                    "actual_formula": rdMolDescriptors.CalcMolFormula(predicted),
                    "actual_smiles": actual_smiles,
                    "actual_inchikey": actual_inchikey,
                    "formula_match": rdMolDescriptors.CalcMolFormula(predicted)
                    == row["formula"],
                    "atom_count_match": predicted.GetNumAtoms() == int(row["atoms"]),
                    "bond_count_match": predicted.GetNumBonds() == int(row["bonds"]),
                    "nonstereo_structure_match": actual_nonstereo
                    == expected_nonstereo,
                    "chemical_match": actual_inchikey == row["inchikey"],
                }
            )
            if args.save_sdf:
                write_sdf_v4(repaired.sequence, sdf_dir / f"{molecule_id}.sdf")
        except Exception as error:
            record["error"] = str(error)
            record["failure_category"] = failure_category(error)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        completed.add(molecule_id)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(records) % 25 == 0:
            summary = summarize_records(records)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    summary = summarize_records(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
