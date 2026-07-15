#!/usr/bin/env python3
"""Measure MR-TADF upper bounds with truth elements and/or truth vertices."""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemmol_sequence_v3 import canonical_chiral_tag
from Data.sdf.chemsvg_sequence_v4 import (
    ParsedChemSVGSequenceV4,
    VertexRecordV4,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
    serialize_parsed_chemsvg_v4,
)
from chemvector.benchmark_mr_tadf import (
    ensure_benchmark_config,
    failure_category,
    molecular_similarity,
    nonstereo_inchikey,
    sha256_file,
    summarize_records,
    write_invalid_prediction_sdf,
    write_prediction_sdf,
)
from chemvector.evaluate import load_model
from chemvector.image_io import image_tensor
from chemvector.ocr import OCRTextCodec, parse_atom_display_text
from chemvector.parallel_v6 import (
    build_parallel_sequence,
    score_vertex_pairs_parallel,
    strip_autoregressive_modules,
)
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image


ORACLE_MODES = (
    "oracle-elements",
    "oracle-vertices",
    "oracle-vertices-elements",
)


def nearest_unused_coordinate(
    x: int,
    y: int,
    used: set[tuple[int, int]],
) -> tuple[int, int]:
    if (x, y) not in used:
        return x, y
    for radius in range(1, 9):
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = (
                    max(0, min(1023, x + dx)),
                    max(0, min(1023, y + dy)),
                )
                candidates.append((dx * dx + dy * dy, candidate))
        for _, candidate in sorted(candidates):
            if candidate not in used:
                return candidate
    raise ValueError("Cannot separate colliding truth vertices")


def truth_vertex_records(
    mol: Chem.Mol,
    annotation: dict,
) -> tuple[tuple[VertexRecordV4, ...], int]:
    width = int(annotation["width"])
    height = int(annotation["height"])
    positions = {int(row["atom_id"]): row for row in annotation["atoms"]}
    vertices = []
    used = set()
    adjusted = 0
    for atom_id in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_id)
        position = positions[atom_id]
        x = max(0, min(1023, round(float(position["image_x"]) / width * 1023)))
        y = max(0, min(1023, round(float(position["image_y"]) / height * 1023)))
        coordinate = nearest_unused_coordinate(x, y, used)
        adjusted += int(coordinate != (x, y))
        x, y = coordinate
        used.add(coordinate)
        vertices.append(
            VertexRecordV4(
                x=x,
                y=y,
                element=atom.GetSymbol(),
                formal_charge=atom.GetFormalCharge(),
                isotope=atom.GetIsotope(),
                aromatic=atom.GetIsAromatic(),
                explicit_h_count=atom.GetNumExplicitHs(),
                no_implicit=atom.GetNoImplicit(),
                radical_electrons=atom.GetNumRadicalElectrons(),
                canonical_chiral_tag=canonical_chiral_tag(atom),
            )
        )
    return tuple(vertices), adjusted


def apply_truth_ocr_properties(
    sequence: str,
    truth_vertices: tuple[VertexRecordV4, ...],
    *,
    maximum_distance: float = 35.0,
) -> tuple[str, int]:
    parsed = parse_chemsvg_v4(sequence)
    predicted_xy = np.asarray([(item.x, item.y) for item in parsed.vertices])
    truth_xy = np.asarray([(item.x, item.y) for item in truth_vertices])
    distances = np.linalg.norm(predicted_xy[:, None, :] - truth_xy[None, :, :], axis=2)
    predicted_indices, truth_indices = linear_sum_assignment(distances)
    assignments = {
        int(predicted): int(truth)
        for predicted, truth in zip(predicted_indices, truth_indices)
        if distances[predicted, truth] <= maximum_distance
    }
    vertices = tuple(
        replace(
            vertex,
            element=truth_vertices[assignments[index]].element,
            formal_charge=truth_vertices[assignments[index]].formal_charge,
            isotope=truth_vertices[assignments[index]].isotope,
            explicit_h_count=truth_vertices[assignments[index]].explicit_h_count,
            no_implicit=truth_vertices[assignments[index]].no_implicit,
            radical_electrons=truth_vertices[assignments[index]].radical_electrons,
        )
        if index in assignments
        else vertex
        for index, vertex in enumerate(parsed.vertices)
    )
    return (
        serialize_parsed_chemsvg_v4(
            ParsedChemSVGSequenceV4(vertices, parsed.strokes)
        ),
        len(assignments),
    )


@torch.inference_mode()
def model_vertices_at_truth_positions(
    model,
    pixels: torch.Tensor,
    stroke_features: torch.Tensor,
    truth_vertices: tuple[VertexRecordV4, ...],
    *,
    ocr_roi_radius: float,
) -> tuple[VertexRecordV4, ...]:
    points = torch.tensor(
        [[(vertex.x / 1023.0, vertex.y / 1023.0) for vertex in truth_vertices]],
        device=pixels.device,
        dtype=torch.float32,
    )
    mask = torch.ones(
        (1, len(truth_vertices)),
        device=pixels.device,
        dtype=torch.bool,
    )
    ocr_features = model.ocr_source(pixels, stroke_features)
    ocr_ids = model.ocr_head.generate(
        ocr_features,
        points,
        mask,
        pixels.shape[-2:],
        roi_radius=ocr_roi_radius,
    )
    codec = OCRTextCodec(max_length=model.config.ocr_max_length)
    labels = [codec.decode(row.tolist()) for row in ocr_ids.cpu()]
    vertex_output = model.vertex_head(stroke_features)
    aromatic = vertex_output["vertex_aromatic_logits"].sigmoid()[0, 0]
    height, width = aromatic.shape
    records = []
    for truth, label in zip(truth_vertices, labels):
        parsed = parse_atom_display_text(label)
        if parsed is None or parsed["element"] == "C":
            parsed = {
                "element": "C",
                "formal_charge": 0,
                "isotope": 0,
                "explicit_h_count": 0,
                "radical_electrons": 0,
            }
        cell_x = min(width - 1, int(truth.x / 1023.0 * width))
        cell_y = min(height - 1, int(truth.y / 1023.0 * height))
        explicit_h_count = int(parsed["explicit_h_count"])
        records.append(
            VertexRecordV4(
                x=truth.x,
                y=truth.y,
                element=str(parsed["element"]),
                formal_charge=int(parsed["formal_charge"]),
                isotope=int(parsed["isotope"]),
                aromatic=bool(aromatic[cell_y, cell_x] >= 0.5),
                explicit_h_count=explicit_h_count,
                no_implicit=bool(explicit_h_count),
                radical_electrons=int(parsed["radical_electrons"]),
                canonical_chiral_tag="CHI_UNSPECIFIED",
            )
        )
    return tuple(records)


@torch.inference_mode()
def sequence_from_vertices(
    model,
    stroke_features: torch.Tensor,
    vertices: tuple[VertexRecordV4, ...],
    pixel_size: tuple[int, int],
    *,
    edge_threshold: float,
) -> str:
    pairs, probabilities = score_vertex_pairs_parallel(
        model,
        stroke_features,
        vertices,
        pixel_size,
    )
    sequence, _ = build_parallel_sequence(
        vertices,
        pairs,
        probabilities,
        edge_threshold=edge_threshold,
    )
    return sequence


def initial_record(row: dict, mode: str) -> dict:
    return {
        "molecule_id": row["molecule_id"],
        "oracle_mode": mode,
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


def evaluate_sequence(
    *,
    sequence: str,
    image: Image.Image,
    expected: Chem.Mol,
    row: dict,
    mode: str,
    output_dir: Path,
    extra: dict | None = None,
) -> dict:
    record = initial_record(row, mode)
    if extra:
        record.update(extra)
    raw_dir = output_dir / "raw_chemsvg"
    repaired_dir = output_dir / "repaired_chemsvg"
    sdf_dir = output_dir / "sdf"
    for directory in (raw_dir, repaired_dir, sdf_dir):
        directory.mkdir(parents=True, exist_ok=True)
    molecule_id = row["molecule_id"]
    (raw_dir / f"{molecule_id}.chemsvg.txt").write_text(
        sequence,
        encoding="utf-8",
    )
    diagnostic_sequence = sequence
    pending_sdf = sdf_dir / f"pending_{molecule_id}.sdf"
    started = time.perf_counter()
    try:
        repaired = repair_chemsvg_v4_from_image(
            sequence,
            image,
            config=InkGraphRepairConfig(aromatic_vote_source="vertices"),
            validate_chemistry=False,
        )
        diagnostic_sequence = repaired.sequence
        (repaired_dir / f"{molecule_id}.chemsvg.txt").write_text(
            repaired.sequence,
            encoding="utf-8",
        )
        predicted = mol_from_chemsvg_v4(repaired.sequence)
        delivered = write_prediction_sdf(predicted, pending_sdf, sanitize=True)
        actual_inchikey = Chem.MolToInchiKey(delivered)
        record.update(
            {
                "sdf_valid": True,
                "predicted_atoms": delivered.GetNumAtoms(),
                "predicted_bonds": delivered.GetNumBonds(),
                "actual_formula": rdMolDescriptors.CalcMolFormula(delivered),
                "actual_inchikey": actual_inchikey,
                "fingerprint_similarity": molecular_similarity(expected, delivered),
                "formula_match": rdMolDescriptors.CalcMolFormula(delivered)
                == row["formula"],
                "atom_count_match": delivered.GetNumAtoms() == int(row["atoms"]),
                "bond_count_match": delivered.GetNumBonds() == int(row["bonds"]),
                "nonstereo_structure_match": nonstereo_inchikey(delivered)
                == nonstereo_inchikey(expected),
                "chemical_match": actual_inchikey == row["inchikey"],
                "raw_vertices": len(parse_chemsvg_v4(sequence).vertices),
                "raw_strokes": len(parse_chemsvg_v4(sequence).strokes),
                "repaired_vertices": repaired.vertices,
                "repaired_strokes": repaired.strokes,
            }
        )
        prefix = "correct" if record["chemical_match"] else "wrong"
        final_sdf = sdf_dir / f"{prefix}_{molecule_id}.sdf"
        pending_sdf.replace(final_sdf)
        record["prediction_sdf_file"] = str(final_sdf.relative_to(output_dir))
    except Exception as error:
        pending_sdf.unlink(missing_ok=True)
        record["error"] = str(error)
        record["failure_category"] = failure_category(error)
        invalid_sdf = sdf_dir / f"invalid_{molecule_id}.sdf"
        try:
            atom_count, bond_count = write_invalid_prediction_sdf(
                diagnostic_sequence,
                invalid_sdf,
            )
            record["predicted_atoms"] = atom_count
            record["predicted_bonds"] = bond_count
            record["prediction_sdf_file"] = str(
                invalid_sdf.relative_to(output_dir)
            )
        except Exception as diagnostic_error:
            record["diagnostic_sdf_error"] = str(diagnostic_error)
    record["seconds"] = time.perf_counter() - started
    return record


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, records: list[dict]) -> None:
    path.write_text(
        json.dumps(summarize_records(records), ensure_ascii=False, indent=2) + "\n",
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
        "--baseline-dir",
        default="outputs/evaluations/mr-tadf-v1-v071-diagnostics-v2",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evaluations",
    )
    parser.add_argument("--run-prefix", default="mr-tadf-v1-v071")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    manifest_path = Path(args.benchmark_manifest)
    benchmark_root = manifest_path.parent
    baseline_dir = Path(args.baseline_dir)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.samples is not None:
        rows = rows[: args.samples]

    output_dirs = {
        mode: Path(args.output_root) / f"{args.run_prefix}-{mode}"
        for mode in ORACLE_MODES
    }
    mode_records = {}
    completed = {}
    for mode, output_dir in output_dirs.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "benchmark_manifest": str(manifest_path),
            "benchmark_manifest_sha256": sha256_file(manifest_path),
            "baseline_dir": str(baseline_dir),
            "oracle_mode": mode,
            "device": args.device,
            "edge_threshold": args.edge_threshold,
            "ocr_roi_radius": args.ocr_roi_radius,
        }
        ensure_benchmark_config(output_dir / "benchmark_config.json", config)
        records = load_records(output_dir / "records.jsonl")
        mode_records[mode] = records
        completed[mode] = {record["molecule_id"] for record in records}

    model, _ = load_model(checkpoint, device)
    strip_autoregressive_modules(model)
    for row_index, row in enumerate(rows, start=1):
        molecule_id = row["molecule_id"]
        pending_modes = [
            mode for mode in ORACLE_MODES if molecule_id not in completed[mode]
        ]
        if not pending_modes:
            continue
        image = Image.open(benchmark_root / row["image_file"]).convert("RGB")
        expected = Chem.MolFromMolFile(
            str(benchmark_root / row["sdf_file"]),
            removeHs=False,
        )
        if expected is None:
            raise ValueError(f"Cannot read benchmark truth: {row['sdf_file']}")
        annotation = json.loads(
            (benchmark_root / row["annotation_file"]).read_text(encoding="utf-8")
        )
        truth_vertices, quantization_adjustments = truth_vertex_records(
            expected,
            annotation,
        )
        sequences = {}
        extras = {}
        if "oracle-elements" in pending_modes:
            baseline_sequence = (
                baseline_dir / "raw_chemsvg" / f"{molecule_id}.chemsvg.txt"
            ).read_text(encoding="utf-8")
            sequence, matched = apply_truth_ocr_properties(
                baseline_sequence,
                truth_vertices,
            )
            sequences["oracle-elements"] = sequence
            extras["oracle-elements"] = {
                "oracle_matched_vertices": matched,
                "oracle_quantization_adjustments": quantization_adjustments,
            }

        truth_vertex_modes = {
            mode
            for mode in pending_modes
            if mode in {"oracle-vertices", "oracle-vertices-elements"}
        }
        if truth_vertex_modes:
            pixels = image_tensor(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                stroke_features = model.vision_encoder.forward_stroke_resolution(pixels)
                model_truth_vertices = model_vertices_at_truth_positions(
                    model,
                    pixels,
                    stroke_features,
                    truth_vertices,
                    ocr_roi_radius=args.ocr_roi_radius,
                )
                model_sequence = sequence_from_vertices(
                    model,
                    stroke_features,
                    model_truth_vertices,
                    pixels.shape[-2:],
                    edge_threshold=args.edge_threshold,
                )
            sequences["oracle-vertices"] = model_sequence
            extras["oracle-vertices"] = {
                "oracle_quantization_adjustments": quantization_adjustments,
            }
            parsed_model = parse_chemsvg_v4(model_sequence)
            oracle_vertex_records = tuple(
                replace(
                    model_vertex,
                    element=truth_vertex.element,
                    formal_charge=truth_vertex.formal_charge,
                    isotope=truth_vertex.isotope,
                    explicit_h_count=truth_vertex.explicit_h_count,
                    no_implicit=truth_vertex.no_implicit,
                    radical_electrons=truth_vertex.radical_electrons,
                )
                for model_vertex, truth_vertex in zip(
                    model_truth_vertices,
                    truth_vertices,
                )
            )
            sequences["oracle-vertices-elements"] = serialize_parsed_chemsvg_v4(
                ParsedChemSVGSequenceV4(
                    oracle_vertex_records,
                    parsed_model.strokes,
                )
            )
            extras["oracle-vertices-elements"] = {
                "oracle_quantization_adjustments": quantization_adjustments,
            }

        for mode in pending_modes:
            record = evaluate_sequence(
                sequence=sequences[mode],
                image=image,
                expected=expected,
                row=row,
                mode=mode,
                output_dir=output_dirs[mode],
                extra=extras.get(mode),
            )
            mode_records[mode].append(record)
            completed[mode].add(molecule_id)
            append_record(output_dirs[mode] / "records.jsonl", record)
            if len(mode_records[mode]) % 25 == 0:
                write_summary(output_dirs[mode] / "summary.json", mode_records[mode])
        if row_index % 25 == 0:
            print(
                json.dumps(
                    {
                        mode: summarize_records(mode_records[mode])
                        for mode in ORACLE_MODES
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final = {}
    for mode in ORACLE_MODES:
        write_summary(output_dirs[mode] / "summary.json", mode_records[mode])
        final[mode] = summarize_records(mode_records[mode])
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
