#!/usr/bin/env python3
"""在公开图片/Molfile 数据集上评测 ChemVector 的 SDF 图恢复能力。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import torch

from Data.sdf.chemsvg_sequence_v4 import mol_from_chemsvg_v4
from chemvector.benchmark_mr_tadf import (
    ensure_benchmark_config,
    failure_category,
    molecular_similarity,
    sha256_file,
    write_invalid_prediction_sdf,
    write_prediction_sdf,
)
from chemvector.evaluate import load_model
from chemvector.image_io import image_tensor, prepare_input_image
from chemvector.parallel_v6 import generate_parallel_chemsvg_v6, strip_autoregressive_modules
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image
from chemvector.structure_gate import (
    expert_route_probability,
    load_packed_structure_expert,
    load_structure_gate,
)


def _atom_properties(atom: Chem.Atom) -> tuple[int, int, int, int, bool]:
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
    )


def sdf_graph_match(
    expected: Chem.Mol,
    predicted: Chem.Mol,
    *,
    use_chirality: bool,
) -> bool:
    """比较 SDF 表达的带属性分子图，不借助 SMILES 字符串。"""

    if (
        expected.GetNumAtoms() != predicted.GetNumAtoms()
        or expected.GetNumBonds() != predicted.GetNumBonds()
        or len(Chem.GetMolFrags(expected)) != len(Chem.GetMolFrags(predicted))
    ):
        return False
    expected_copy = Chem.Mol(expected)
    predicted_copy = Chem.Mol(predicted)
    if not use_chirality:
        Chem.RemoveStereochemistry(expected_copy)
        Chem.RemoveStereochemistry(predicted_copy)

    match = predicted_copy.GetSubstructMatch(
        expected_copy,
        useChirality=use_chirality,
    )
    if len(match) != expected_copy.GetNumAtoms():
        return False
    for expected_index, predicted_index in enumerate(match):
        if _atom_properties(expected_copy.GetAtomWithIdx(expected_index)) != _atom_properties(
            predicted_copy.GetAtomWithIdx(predicted_index)
        ):
            return False
    for expected_bond in expected_copy.GetBonds():
        predicted_bond = predicted_copy.GetBondBetweenAtoms(
            match[expected_bond.GetBeginAtomIdx()],
            match[expected_bond.GetEndAtomIdx()],
        )
        if predicted_bond is None:
            return False
        if (
            predicted_bond.GetBondType() != expected_bond.GetBondType()
            or predicted_bond.GetIsAromatic() != expected_bond.GetIsAromatic()
        ):
            return False
        if use_chirality and predicted_bond.GetStereo() != expected_bond.GetStereo():
            return False
    return True


def summarize_records(records: list[dict]) -> dict:
    samples = len(records)
    divisor = max(samples, 1)
    valid = [row for row in records if row["sdf_valid"]]
    failures = Counter(
        row["failure_category"]
        for row in records
        if row.get("failure_category")
    )
    return {
        "samples": samples,
        "sdf_valid_count": len(valid),
        "chemical_match_count": sum(bool(row["chemical_match"]) for row in records),
        "sdf_valid_rate": sum(bool(row["sdf_valid"]) for row in records) / divisor,
        "formula_match_rate": sum(bool(row["formula_match"]) for row in records)
        / divisor,
        "atom_count_match_rate": sum(
            bool(row["atom_count_match"]) for row in records
        )
        / divisor,
        "bond_count_match_rate": sum(
            bool(row["bond_count_match"]) for row in records
        )
        / divisor,
        "nonstereo_structure_match_rate": sum(
            bool(row["nonstereo_structure_match"]) for row in records
        )
        / divisor,
        "chemical_match_rate": sum(bool(row["chemical_match"]) for row in records)
        / divisor,
        "average_fingerprint_similarity": sum(
            float(row.get("fingerprint_similarity", 0.0)) for row in records
        )
        / divisor,
        "average_fingerprint_similarity_valid_only": sum(
            float(row.get("fingerprint_similarity", 0.0)) for row in valid
        )
        / max(len(valid), 1),
        "average_seconds": sum(float(row["seconds"]) for row in records) / divisor,
        "failure_categories": dict(sorted(failures.items())),
    }


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_summary(path: Path, records: list[dict]) -> dict:
    summary = summarize_records(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/chemvector-v076-vertex-pilot-head-lr1e4-9k/best.pt",
    )
    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--gate-threshold", type=float, default=0.9)
    parser.add_argument(
        "--benchmark-manifest",
        default="Data/benchmarks/public_ocsr_v1/prepared/uspto_clean_1k/manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluations/uspto-clean-1k-v076",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--content-size", type=int, default=548)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument("--maximum-vertices", type=int, default=128)
    parser.add_argument(
        "--repair-candidate-mode",
        choices=("all", "model", "model_connect"),
        default="all",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples 必须大于 0")
    if args.cpu_threads < 1 or args.maximum_vertices < 1:
        raise ValueError("cpu-threads 和 maximum-vertices 必须大于 0")
    if not 1 <= args.content_size <= args.image_size:
        raise ValueError("content-size 必须位于 1 到 image-size 之间")
    if not 0.0 <= args.gate_threshold <= 1.0:
        raise ValueError("gate-threshold 必须位于 0 到 1 之间")
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    manifest_path = Path(args.benchmark_manifest)
    benchmark_root = manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ready"]
    if args.samples is not None:
        rows = rows[: args.samples]

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
    completed = {row["molecule_id"] for row in records}
    checkpoint_path = Path(args.checkpoint)
    config = {
        "benchmark_type": "public_ocsr_sdf_graph_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "device": args.device,
        "image_size": args.image_size,
        "content_size": args.content_size,
        "vertex_threshold": args.vertex_threshold,
        "edge_threshold": args.edge_threshold,
        "ocr_roi_radius": args.ocr_roi_radius,
        "maximum_vertices": args.maximum_vertices,
        "repair_candidate_mode": args.repair_candidate_mode,
        "primary_comparison": "SDF attributed graph isomorphism with stereochemistry",
        "gate_checkpoint": args.gate_checkpoint,
        "gate_threshold": args.gate_threshold if args.gate_checkpoint else None,
    }
    ensure_benchmark_config(output_dir / "benchmark_config.json", config)

    model, _ = load_model(checkpoint_path, device)
    strip_autoregressive_modules(model)
    gate = None
    expert_model = None
    if args.gate_checkpoint:
        gate, gate_payload = load_structure_gate(args.gate_checkpoint, device)
        if Path(gate_payload["base_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Gate base checkpoint does not match --checkpoint")
        if "expert_structure_tensors" in gate_payload:
            expert_model = load_packed_structure_expert(model, gate_payload, device)
        else:
            expert_model, _ = load_model(
                Path(gate_payload["expert_checkpoint"]), device
            )
            strip_autoregressive_modules(expert_model)
    raw_dir = output_dir / "raw_chemsvg"
    repaired_dir = output_dir / "repaired_chemsvg"
    sdf_dir = output_dir / "sdf"
    prepared_dir = output_dir / "prepared_inputs"
    for directory in (raw_dir, repaired_dir, sdf_dir, prepared_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for row in rows:
        molecule_id = row["molecule_id"]
        if molecule_id in completed:
            continue
        source_image = Image.open(benchmark_root / row["image_file"])
        image = prepare_input_image(
            source_image,
            image_size=args.image_size,
            content_size=args.content_size,
        )
        image.save(prepared_dir / f"{molecule_id}.png")
        pixels = image_tensor(image).unsqueeze(0).to(device)
        expected = Chem.MolFromMolFile(
            str(benchmark_root / row["sdf_file"]),
            sanitize=True,
            removeHs=False,
            strictParsing=False,
        )
        if expected is None:
            raise ValueError(f"无法读取 benchmark 真值：{row['sdf_file']}")

        record = {
            "molecule_id": molecule_id,
            "dataset": row.get("dataset"),
            "subset": row.get("subset"),
            "expected_atoms": expected.GetNumAtoms(),
            "expected_bonds": expected.GetNumBonds(),
            "predicted_atoms": 0,
            "predicted_bonds": 0,
            "sdf_valid": False,
            "formula_match": False,
            "atom_count_match": False,
            "bond_count_match": False,
            "nonstereo_structure_match": False,
            "chemical_match": False,
        }
        inference_model = model
        if gate is not None:
            route_probability = expert_route_probability(model, gate, pixels)
            record["expert_route_probability"] = route_probability
            record["selected_expert"] = route_probability >= args.gate_threshold
            if record["selected_expert"]:
                inference_model = expert_model
        diagnostic_sequence = None
        pending_sdf = None
        started = time.perf_counter()
        try:
            generated = generate_parallel_chemsvg_v6(
                inference_model,
                pixels,
                vertex_threshold=args.vertex_threshold,
                maximum_vertices=args.maximum_vertices,
                edge_threshold=args.edge_threshold,
                ocr_roi_radius=args.ocr_roi_radius,
            )
            diagnostic_sequence = generated.sequence
            (raw_dir / f"{molecule_id}.chemsvg.txt").write_text(
                generated.sequence, encoding="utf-8"
            )
            record["raw_vertices"] = len(generated.vertices)
            record["raw_strokes"] = len(generated.strokes)
            repaired = repair_chemsvg_v4_from_image(
                generated.sequence,
                image,
                pair_predictions=generated.pair_predictions,
                config=InkGraphRepairConfig(
                    aromatic_vote_source="vertices",
                    candidate_pair_mode=args.repair_candidate_mode,
                ),
                validate_chemistry=False,
            )
            diagnostic_sequence = repaired.sequence
            (repaired_dir / f"{molecule_id}.chemsvg.txt").write_text(
                repaired.sequence, encoding="utf-8"
            )
            record["repaired_vertices"] = repaired.vertices
            record["repaired_strokes"] = repaired.strokes
            record["model_bridge_strokes"] = repaired.model_bridge_strokes
            predicted = mol_from_chemsvg_v4(repaired.sequence)
            pending_sdf = sdf_dir / f"pending_{molecule_id}.sdf"
            delivered = write_prediction_sdf(predicted, pending_sdf, sanitize=True)
            actual_formula = rdMolDescriptors.CalcMolFormula(delivered)
            expected_formula = rdMolDescriptors.CalcMolFormula(expected)
            record.update(
                {
                    "sdf_valid": True,
                    "predicted_atoms": delivered.GetNumAtoms(),
                    "predicted_bonds": delivered.GetNumBonds(),
                    "actual_formula": actual_formula,
                    "fingerprint_similarity": molecular_similarity(expected, delivered),
                    "formula_match": actual_formula == expected_formula,
                    "atom_count_match": delivered.GetNumAtoms()
                    == expected.GetNumAtoms(),
                    "bond_count_match": delivered.GetNumBonds()
                    == expected.GetNumBonds(),
                    "nonstereo_structure_match": sdf_graph_match(
                        expected, delivered, use_chirality=False
                    ),
                    "chemical_match": sdf_graph_match(
                        expected, delivered, use_chirality=True
                    ),
                }
            )
            prefix = "correct" if record["chemical_match"] else "wrong"
            final_sdf = sdf_dir / f"{prefix}_{molecule_id}.sdf"
            pending_sdf.replace(final_sdf)
            record["prediction_sdf_file"] = str(final_sdf.relative_to(output_dir))
        except Exception as error:
            if pending_sdf is not None:
                pending_sdf.unlink(missing_ok=True)
            record["error"] = str(error)
            record["failure_category"] = failure_category(error)
            if diagnostic_sequence is not None:
                invalid_sdf = sdf_dir / f"invalid_{molecule_id}.sdf"
                try:
                    atoms, bonds = write_invalid_prediction_sdf(
                        diagnostic_sequence, invalid_sdf
                    )
                    record["predicted_atoms"] = atoms
                    record["predicted_bonds"] = bonds
                    record["prediction_sdf_file"] = str(
                        invalid_sdf.relative_to(output_dir)
                    )
                except Exception as diagnostic_error:
                    record["diagnostic_sdf_error"] = str(diagnostic_error)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        completed.add(molecule_id)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(records) % 25 == 0:
            print(
                json.dumps(
                    write_summary(output_dir / "summary.json", records),
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_records(records_path, records)
    summary = write_summary(output_dir / "summary.json", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
