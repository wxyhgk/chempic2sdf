#!/usr/bin/env python3
"""End-to-end scaffold validation for detected ChemSVG v0.5 inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from rdkit import Chem, RDLogger
import torch

from Data.sdf.chemsvg_sequence_v4 import mol_from_chemsvg_v4, parse_chemsvg_v4
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from chemvector.evaluate import load_model, normalized_smiles
from chemvector.evaluation_metrics import compare_graphs, draw_overlay, summarize
from chemvector.generation_v4 import generate_detected_chemsvg_v4
from chemvector.image_io import tensor_to_image
from chemvector.parallel_v6 import (
    generate_parallel_chemsvg_v6,
    strip_autoregressive_modules,
)
from chemvector.repair_v4 import (
    InkGraphRepairConfig,
    repair_chemsvg_v4_from_image,
)
from chemvector.train import expand_shards, make_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--inference-mode",
        choices=("autoregressive", "parallel"),
        default="autoregressive",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--stroke-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument("--vertex-tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--failure-overlays", type=int, default=20)
    parser.add_argument("--summary-every", type=int, default=10)
    parser.add_argument("--reject-keys-file")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.inference_mode == "autoregressive" and args.device != "cuda":
        raise ValueError("Autoregressive evaluation currently requires CUDA")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if args.summary_every < 0:
        raise ValueError("summary-every must be nonnegative")
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.cpu_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records = []
    completed_keys = set()
    if records_path.is_file():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed_keys = {record["key"] for record in records}

    device = torch.device(args.device)
    model, tokenizer = load_model(Path(args.checkpoint), device)
    if not isinstance(tokenizer, ChemSVGTokenizerV4):
        raise ValueError("Detected evaluation requires ChemSVG v0.4")
    if args.inference_mode == "parallel":
        strip_autoregressive_modules(model)
    reject_keys = set()
    if args.reject_keys_file:
        reject_keys = {
            line.strip()
            for line in Path(args.reject_keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    loader = make_loader(
        expand_shards(args.shards),
        tokenizer,
        model.config,
        batch_size=1,
        image_size=args.image_size,
        workers=0,
        shuffle=False,
        reject_keys=reject_keys,
    )
    overlays = len(list((output_dir / "failures").glob("*.png")))
    for batch in loader:
        if len(records) >= args.samples:
            break
        key = batch["keys"][0]
        if key in completed_keys:
            continue
        expected_sequence = batch["target_sequences"][0]
        expected = parse_chemsvg_v4(expected_sequence)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        image = tensor_to_image(batch["pixel_values"][0])
        started = time.perf_counter()
        predicted = None
        record = {
            "key": key,
            "generation_completed": False,
            "repair_success": False,
            "sdf_valid": False,
            "chemical_match": False,
            "exact_topology": False,
            "expected_vertices": len(expected.vertices),
            "predicted_vertices": 0,
            "matched_vertices": 0,
            "correct_elements": 0,
            "expected_edges": len(expected.strokes),
            "predicted_edges": 0,
            "correct_edges": 0,
            "correct_bond_types": 0,
        }
        try:
            if args.inference_mode == "parallel":
                generated = generate_parallel_chemsvg_v6(
                    model,
                    pixels,
                    vertex_threshold=args.vertex_threshold,
                    edge_threshold=args.stroke_threshold,
                    ocr_roi_radius=args.ocr_roi_radius,
                )
                generated_sequence = generated.sequence
                record["generation_completed"] = True
                record["generation_chemically_valid"] = False
                record["generation_error"] = None
                repair_config = InkGraphRepairConfig(
                    aromatic_vote_source="vertices"
                )
            else:
                generated = generate_detected_chemsvg_v4(
                    model,
                    pixels,
                    tokenizer,
                    vertex_threshold=args.vertex_threshold,
                    stroke_threshold=args.stroke_threshold,
                    ocr_roi_radius=args.ocr_roi_radius,
                )
                record["generation_completed"] = generated.completed
                record["generation_chemically_valid"] = generated.chemically_valid
                record["generation_error"] = generated.error
                if generated.sequence is None:
                    raise ValueError(
                        generated.error or "generation returned no sequence"
                    )
                generated_sequence = generated.sequence
                repair_config = None
            repaired = repair_chemsvg_v4_from_image(
                generated_sequence,
                image,
                config=repair_config,
            )
            predicted = parse_chemsvg_v4(repaired.sequence)
            record["repair_success"] = True
            comparison = compare_graphs(
                expected,
                predicted,
                tolerance_pixels=args.vertex_tolerance_pixels,
                image_size=args.image_size,
            )
            record.update(asdict(comparison))
            actual_mol = mol_from_chemsvg_v4(repaired.sequence)
            record["sdf_valid"] = True
            actual_smiles = normalized_smiles(actual_mol)
            actual_inchikey = Chem.MolToInchiKey(actual_mol)
            expected_mol = mol_from_chemsvg_v4(expected_sequence)
            record["chemical_match"] = (
                actual_smiles == normalized_smiles(expected_mol)
                and actual_inchikey == Chem.MolToInchiKey(expected_mol)
            )
            record["actual_smiles"] = actual_smiles
            record["actual_inchikey"] = actual_inchikey
        except Exception as error:
            record["error"] = str(error)
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        completed_keys.add(key)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not record["chemical_match"] and overlays < args.failure_overlays:
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            draw_overlay(image, expected, predicted).save(failure_dir / f"{key}.png")
            (failure_dir / f"{key}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            overlays += 1
        if args.summary_every and (
            len(records) % args.summary_every == 0 or len(records) == args.samples
        ):
            summary = summarize(records)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    summary = summarize(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
