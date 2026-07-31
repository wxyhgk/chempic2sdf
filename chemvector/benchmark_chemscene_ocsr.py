#!/usr/bin/env python3
"""Benchmark ChemScene inference on a standard image/SDF manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import time

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import torch

from Data.sdf.chemscene_v0_1 import chemsvg_from_chemscene_v0_1
from chemvector.benchmark_mr_tadf import (
    ensure_benchmark_config,
    failure_category,
    molecular_similarity,
    sha256_file,
    write_invalid_prediction_sdf,
    write_prediction_sdf,
)
from chemvector.benchmark_public_ocsr import sdf_graph_match, summarize_records
from chemvector.benchmark_object_decomposition import evaluate_benchmark_directory
from chemvector.image_io import image_tensor, prepare_input_image
from chemvector.infer_chemscene_hybrid import (
    ChemSceneHybridPipeline,
    compile_scene_with_aromatic_fallback,
)


def write_summary(path: Path, records: list[dict]) -> dict:
    summary = summarize_records(records)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("difficulty"):
            groups[str(record["difficulty"])].append(record)
    if groups:
        summary["difficulty_rates"] = {
            name: summarize_records(rows) for name, rows in sorted(groups.items())
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return summary


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def deduplicate_records(records: list[dict]) -> tuple[list[dict], int]:
    unique = []
    seen = set()
    for record in records:
        molecule_id = str(record["molecule_id"])
        if molecule_id in seen:
            continue
        seen.add(molecule_id)
        unique.append(record)
    return unique, len(records) - len(unique)


def acquire_output_lock(output_dir: Path):
    lock_path = output_dir / ".benchmark.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"Another benchmark process is already writing to {output_dir}"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def checkpoint_component(path: str | None) -> dict | None:
    if path is None:
        return None
    checkpoint = Path(path)
    return {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}


def write_empty_diagnostic_sdf(path: Path, error: str) -> None:
    """Persist an explicit empty prediction when the selector abstains."""

    block = "\n".join(
        (
            "ChemVector empty diagnostic prediction",
            "  ChemVector",
            "",
            "  0  0  0  0  0  0  0  0  0  0999 V2000",
            "M  END",
            ">  <CHEMVECTOR_STATUS>",
            "invalid_empty_prediction",
            "",
            ">  <CHEMVECTOR_ERROR>",
            error,
            "",
            "$$$$",
            "",
        )
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(block, encoding="utf-8")
    temporary.replace(path)


def ensure_error_record_sdf(record: dict, output_dir: Path, sdf_dir: Path) -> bool:
    prediction_file = record.get("prediction_sdf_file")
    if prediction_file and (output_dir / prediction_file).is_file():
        return False
    if not record.get("error"):
        return False
    invalid_path = sdf_dir / f"invalid_{record['molecule_id']}.sdf"
    write_empty_diagnostic_sdf(invalid_path, str(record["error"]))
    record["prediction_sdf_file"] = str(invalid_path.relative_to(output_dir))
    record["predicted_atoms"] = 0
    record["predicted_bonds"] = 0
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--model-version")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--content-size", type=int, default=548)
    parser.add_argument("--image-is-prepared", action="store_true")
    parser.add_argument(
        "--selector-checkpoint",
        default="outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
    )
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument("--expert-bond-checkpoint")
    parser.add_argument("--style-gate-checkpoint")
    parser.add_argument("--style-gate-threshold", type=float, default=0.90)
    parser.add_argument("--bond-utility-router-checkpoint")
    parser.add_argument(
        "--bond-utility-break-budget", choices=("0", "1", "2", "5", "10"), default="0"
    )
    parser.add_argument("--text-interior-bond-veto", action="store_true")
    parser.add_argument(
        "--graph-corrector-checkpoint",
        default="outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
    )
    parser.add_argument(
        "--ctc-ocr-checkpoint",
        default="outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
    )
    parser.add_argument(
        "--text-bbox-checkpoint",
        default="outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt",
    )
    parser.add_argument("--ctc-anchor-fallback-margin", type=float, default=0.25)
    parser.add_argument("--bond-set-selector-checkpoint")
    parser.add_argument("--endpoint-grouper-checkpoint")
    parser.add_argument("--line-band-checkpoint")
    parser.add_argument("--semantic-bond-relation-checkpoint")
    parser.add_argument("--semantic-bond-existence-checkpoint")
    parser.add_argument("--semantic-bond-existence-threshold", type=float, default=0.60)
    parser.add_argument("--semantic-bond-existence-chunk-size", type=int, default=256)
    parser.add_argument(
        "--semantic-bond-relation-mode",
        choices=("chemical_all", "aromatic_ring_only"),
        default="chemical_all",
    )
    parser.add_argument("--semantic-aromatic-mean-threshold", type=float, default=0.72)
    parser.add_argument(
        "--semantic-aromatic-minimum-threshold", type=float, default=0.25
    )
    parser.add_argument(
        "--junction-projection-score-source",
        choices=("detector", "selector"),
        default="detector",
    )
    parser.add_argument(
        "--junction-bond-type-source",
        choices=("detector", "line_band"),
        default="detector",
    )
    parser.add_argument(
        "--junction-source",
        choices=(
            "bond_endpoints",
            "junction_aware_bonds",
            "semantic_relations",
            "junction_semantic_relations",
        ),
        default="bond_endpoints",
    )
    parser.add_argument(
        "--graph-correction-mode",
        choices=("none", "type_only", "full", "invalid_only"),
        default="type_only",
    )
    parser.add_argument("--bond-threshold", type=float, default=0.25)
    parser.add_argument("--endpoint-cluster-radius-pixels", type=float, default=12.0)
    parser.add_argument("--text-attach-radius-pixels", type=float, default=24.0)
    parser.add_argument(
        "--atom-decode-mode",
        choices=("greedy", "structured_invalid_only"),
        default="greedy",
    )
    args = parser.parse_args()

    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    RDLogger.DisableLog("rdApp.*")
    device = torch.device(args.device)
    manifest_path = Path(args.benchmark_manifest)
    benchmark_root = manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if not row.get("status") or row.get("status") == "ready"
        ]
    if args.samples is not None:
        rows = rows[: args.samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_lock = acquire_output_lock(output_dir)
    config = {
        "benchmark_type": "chemscene_ocsr_sdf_graph_v1",
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "checkpoint": args.bond_set_selector_checkpoint or args.bond_checkpoint,
        "checkpoint_sha256": sha256_file(
            Path(args.bond_set_selector_checkpoint or args.bond_checkpoint)
        ),
        "model_version": args.model_version
        or (
            "v0.8.26"
            if (
                args.junction_source == "junction_aware_bonds"
                and args.junction_projection_score_source == "selector"
            )
            else "v0.8.23"
            if args.atom_decode_mode == "structured_invalid_only"
            else "v0.8.22"
            if args.junction_source == "junction_aware_bonds"
            else "v0.8.16"
        ),
        "protocol": (
            "top-128 Bond selector -> visual endpoint grouping -> OCR label attachment"
            + (
                " -> CTC top-k fixed-topology atom decode"
                if args.atom_decode_mode == "structured_invalid_only"
                else ""
            )
            + " -> ChemScene"
            if args.junction_source == "junction_aware_bonds"
            else "semantic Bond endpoints -> Junction + OCR label attachment + ChemScene"
        ),
        "selector_checkpoint": args.selector_checkpoint,
        "graph_corrector_checkpoint": args.graph_corrector_checkpoint,
        "ctc_ocr_checkpoint": args.ctc_ocr_checkpoint,
        "text_bbox_checkpoint": args.text_bbox_checkpoint,
        "ctc_anchor_fallback_margin": args.ctc_anchor_fallback_margin,
        "bond_set_selector_checkpoint": args.bond_set_selector_checkpoint,
        "expert_bond_checkpoint": args.expert_bond_checkpoint,
        "style_gate_checkpoint": args.style_gate_checkpoint,
        "style_gate_threshold": args.style_gate_threshold,
        "bond_utility_router_checkpoint": args.bond_utility_router_checkpoint,
        "bond_utility_break_budget": args.bond_utility_break_budget,
        "text_interior_bond_veto": args.text_interior_bond_veto,
        "endpoint_grouper_checkpoint": args.endpoint_grouper_checkpoint,
        "line_band_checkpoint": args.line_band_checkpoint,
        "semantic_bond_relation_checkpoint": args.semantic_bond_relation_checkpoint,
        "semantic_bond_relation_mode": args.semantic_bond_relation_mode,
        "semantic_bond_existence_checkpoint": args.semantic_bond_existence_checkpoint,
        "semantic_bond_existence_threshold": args.semantic_bond_existence_threshold,
        "semantic_bond_existence_chunk_size": args.semantic_bond_existence_chunk_size,
        "semantic_aromatic_mean_threshold": args.semantic_aromatic_mean_threshold,
        "semantic_aromatic_minimum_threshold": args.semantic_aromatic_minimum_threshold,
        "junction_projection_score_source": args.junction_projection_score_source,
        "junction_bond_type_source": args.junction_bond_type_source,
        "component_checkpoints": {
            "selector": checkpoint_component(args.selector_checkpoint),
            "bond_detector": checkpoint_component(args.bond_checkpoint),
            "bond_style_expert": checkpoint_component(args.expert_bond_checkpoint),
            "bond_style_gate": checkpoint_component(args.style_gate_checkpoint),
            "bond_utility_router": checkpoint_component(
                args.bond_utility_router_checkpoint
            ),
            "graph_corrector": checkpoint_component(args.graph_corrector_checkpoint),
            "ctc_ocr": checkpoint_component(args.ctc_ocr_checkpoint),
            "text_bbox": checkpoint_component(args.text_bbox_checkpoint),
            "bond_set_selector": checkpoint_component(
                args.bond_set_selector_checkpoint
            ),
            "endpoint_grouper": checkpoint_component(
                args.endpoint_grouper_checkpoint
            ),
            "line_band": checkpoint_component(args.line_band_checkpoint),
            "semantic_bond_relation": checkpoint_component(
                args.semantic_bond_relation_checkpoint
            ),
            "semantic_bond_existence": checkpoint_component(
                args.semantic_bond_existence_checkpoint
            ),
        },
        "junction_source": args.junction_source,
        "graph_correction_mode": args.graph_correction_mode,
        "device": args.device,
        "image_size": args.image_size,
        "content_size": args.content_size,
        "image_is_prepared": args.image_is_prepared,
        "bond_threshold": args.bond_threshold,
        "endpoint_cluster_radius_pixels": args.endpoint_cluster_radius_pixels,
        "text_attach_radius_pixels": args.text_attach_radius_pixels,
        "atom_decode_mode": args.atom_decode_mode,
        "uses_oracle_junctions": False,
        "uses_oracle_ocr": False,
        "uses_mr_tadf_for_tuning": False,
    }
    ensure_benchmark_config(output_dir / "benchmark_config.json", config)
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint),
        Path(args.bond_checkpoint),
        device,
        None,
        Path(args.graph_corrector_checkpoint),
        None,
        Path(args.ctc_ocr_checkpoint),
        Path(args.text_bbox_checkpoint),
        args.ctc_anchor_fallback_margin,
        Path(args.bond_set_selector_checkpoint)
        if args.bond_set_selector_checkpoint
        else None,
        Path(args.endpoint_grouper_checkpoint)
        if args.endpoint_grouper_checkpoint
        else None,
        Path(args.line_band_checkpoint) if args.line_band_checkpoint else None,
        args.junction_projection_score_source,
        args.junction_bond_type_source,
        Path(args.semantic_bond_relation_checkpoint)
        if args.semantic_bond_relation_checkpoint
        else None,
        args.semantic_bond_relation_mode,
        args.semantic_aromatic_mean_threshold,
        args.semantic_aromatic_minimum_threshold,
        Path(args.semantic_bond_existence_checkpoint)
        if args.semantic_bond_existence_checkpoint
        else None,
        args.semantic_bond_existence_threshold,
        args.semantic_bond_existence_chunk_size,
        expert_bond_checkpoint=(
            Path(args.expert_bond_checkpoint)
            if args.expert_bond_checkpoint
            else None
        ),
        style_gate_checkpoint=(
            Path(args.style_gate_checkpoint) if args.style_gate_checkpoint else None
        ),
        style_gate_threshold=args.style_gate_threshold,
        bond_utility_router_checkpoint=(
            Path(args.bond_utility_router_checkpoint)
            if args.bond_utility_router_checkpoint
            else None
        ),
        bond_utility_break_budget=args.bond_utility_break_budget,
        text_interior_bond_veto=args.text_interior_bond_veto,
    )

    records_path = output_dir / "records.jsonl"
    loaded_records = (
        [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
        if records_path.is_file()
        else []
    )
    records, duplicate_count = deduplicate_records(loaded_records)
    raw_dir = output_dir / "raw_chemsvg"
    repaired_dir = output_dir / "repaired_chemsvg"
    sdf_dir = output_dir / "sdf"
    prepared_dir = output_dir / "prepared_inputs"
    for directory in (raw_dir, repaired_dir, sdf_dir, prepared_dir):
        directory.mkdir(parents=True, exist_ok=True)
    records_changed = duplicate_count > 0
    rerun_ids = set()
    for record in records:
        prediction_file = record.get("prediction_sdf_file")
        if prediction_file and (output_dir / prediction_file).is_file():
            continue
        if not record.get("error"):
            rerun_ids.add(record["molecule_id"])
            records_changed = True
            continue
        records_changed = (
            ensure_error_record_sdf(record, output_dir, sdf_dir) or records_changed
        )
    if rerun_ids:
        records = [
            record for record in records if record["molecule_id"] not in rerun_ids
        ]
    if records_changed:
        write_records(records_path, records)
    completed = {record["molecule_id"] for record in records}

    for row in rows:
        molecule_id = row["molecule_id"]
        if molecule_id in completed:
            continue
        source = Image.open(benchmark_root / row["image_file"]).convert("RGB")
        image = (
            source
            if args.image_is_prepared
            else prepare_input_image(
                source,
                image_size=args.image_size,
                content_size=args.content_size,
            )
        )
        if image.size != (args.image_size, args.image_size):
            image = image.resize(
                (args.image_size, args.image_size), Image.Resampling.BICUBIC
            )
        image.save(prepared_dir / f"{molecule_id}.png")
        expected = Chem.MolFromMolFile(
            str(benchmark_root / row["sdf_file"]),
            sanitize=True,
            removeHs=False,
            strictParsing=False,
        )
        if expected is None:
            raise ValueError(f"Cannot read benchmark ground truth: {row['sdf_file']}")
        record = {
            "molecule_id": molecule_id,
            "dataset": row.get("dataset"),
            "subset": row.get("subset"),
            "difficulty": row.get("difficulty"),
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
        scene = None
        pending_sdf = None
        started = time.perf_counter()
        try:
            pixels = image_tensor(image).unsqueeze(0).to(device)
            scene, diagnostics = pipeline.predict(
                pixels,
                molecule_id,
                bond_threshold=args.bond_threshold,
                graph_correction_mode=args.graph_correction_mode,
                junction_source=args.junction_source,
                endpoint_cluster_radius_pixels=args.endpoint_cluster_radius_pixels,
                text_attach_radius_pixels=args.text_attach_radius_pixels,
                atom_decode_mode=args.atom_decode_mode,
            )
            raw_sequence = chemsvg_from_chemscene_v0_1(scene)
            (raw_dir / f"{molecule_id}.chemsvg.txt").write_text(
                raw_sequence, encoding="utf-8"
            )
            sdf, compiled_scene, aromatic_fallback = compile_scene_with_aromatic_fallback(scene)
            repaired_sequence = chemsvg_from_chemscene_v0_1(compiled_scene)
            (repaired_dir / f"{molecule_id}.chemsvg.txt").write_text(
                repaired_sequence, encoding="utf-8"
            )
            predicted = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=False)
            if predicted is None:
                raise ValueError("Cannot read compiled ChemScene SDF")
            pending_sdf = sdf_dir / f"pending_{molecule_id}.sdf"
            delivered = write_prediction_sdf(predicted, pending_sdf, sanitize=True)
            expected_formula = rdMolDescriptors.CalcMolFormula(expected)
            actual_formula = rdMolDescriptors.CalcMolFormula(delivered)
            record.update(
                {
                    "sdf_valid": True,
                    "predicted_atoms": delivered.GetNumAtoms(),
                    "predicted_bonds": delivered.GetNumBonds(),
                    "actual_formula": actual_formula,
                    "formula_match": actual_formula == expected_formula,
                    "atom_count_match": delivered.GetNumAtoms() == expected.GetNumAtoms(),
                    "bond_count_match": delivered.GetNumBonds() == expected.GetNumBonds(),
                    "nonstereo_structure_match": sdf_graph_match(expected, delivered, use_chirality=False),
                    "chemical_match": sdf_graph_match(expected, delivered, use_chirality=True),
                    "fingerprint_similarity": molecular_similarity(expected, delivered),
                    "aromatic_fallback": aromatic_fallback,
                    "diagnostics": diagnostics,
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
            if scene is not None:
                try:
                    sequence = chemsvg_from_chemscene_v0_1(scene)
                    invalid_path = sdf_dir / f"invalid_{molecule_id}.sdf"
                    atoms, bonds = write_invalid_prediction_sdf(sequence, invalid_path)
                    record["predicted_atoms"] = atoms
                    record["predicted_bonds"] = bonds
                    record["prediction_sdf_file"] = str(invalid_path.relative_to(output_dir))
                except Exception as diagnostic_error:
                    record["diagnostic_sdf_error"] = str(diagnostic_error)
            ensure_error_record_sdf(record, output_dir, sdf_dir)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        completed.add(molecule_id)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(records) % 25 == 0 or len(records) == len(rows):
            summary = write_summary(output_dir / "summary.json", records)
            print(
                f"evaluated={len(records)}/{len(rows)} valid={summary['sdf_valid_rate']:.4f} "
                f"exact={summary['chemical_match_rate']:.4f}",
                flush=True,
            )
    write_summary(output_dir / "summary.json", records)
    decomposition = evaluate_benchmark_directory(output_dir, manifest_path)
    print(
        f"decomposition skeleton={decomposition['skeleton_exact_rate']:.4f} "
        f"atom={decomposition['atom_typed_skeleton_exact_rate']:.4f} "
        f"bond={decomposition['bond_typed_skeleton_exact_rate']:.4f}",
        flush=True,
    )
    output_lock.close()


if __name__ == "__main__":
    main()
