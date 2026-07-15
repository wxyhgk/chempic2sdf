#!/usr/bin/env python3
"""Measure the repair upper bound provided by a ground-truth bond layer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
from typing import Iterator

import numpy as np
from PIL import Image
from rdkit import Chem, RDLogger
import torch

from Data.sdf.chemsvg_sequence_v4 import (
    ChemSVGCoordinateCollisionError,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
    serialize_chemsvg_v4,
)
from chemvector.evaluate import load_model
from chemvector.evaluation_metrics import compare_graphs, draw_overlay
from chemvector.image_io import image_tensor
from chemvector.parallel_v6 import (
    generate_parallel_chemsvg_v6,
    strip_autoregressive_modules,
)
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image


REQUIRED_SUFFIXES = (
    ".png",
    ".bond.png",
    ".layers.json",
    ".chemmol.json",
    ".render.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bond_alpha_to_ink_image(bond_layer: Image.Image) -> Image.Image:
    """Convert a white-on-black alpha layer into black ink on white."""

    alpha = np.asarray(bond_layer.convert("L"), dtype=np.uint8)
    return Image.fromarray(255 - alpha, mode="L").convert("RGB")


def endpoint_boxes_for_sequence(
    sequence: str,
    layers: dict,
    image_size: tuple[int, int],
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Associate predicted vertices inside a ground-truth text instance with its box."""

    width, height = image_size
    visible = [atom for atom in layers["atoms"] if atom["visible_text"]]
    boxes = {}
    for vertex in parse_chemsvg_v4(sequence).vertices:
        point = np.array(
            (
                vertex.x / 1023.0 * (width - 1),
                vertex.y / 1023.0 * (height - 1),
            )
        )
        candidates = []
        for atom in visible:
            x0, y0, x1, y1 = atom["bbox"]
            expanded = (x0 - 3.0, y0 - 3.0, x1 + 3.0, y1 + 3.0)
            if expanded[0] <= point[0] <= expanded[2] and expanded[1] <= point[1] <= expanded[3]:
                center = np.asarray(atom["center"], dtype=np.float64)
                candidates.append((float(np.linalg.norm(point - center)), expanded))
        if candidates:
            boxes[(vertex.x, vertex.y)] = min(candidates, key=lambda item: item[0])[1]
    return boxes


def load_manifest_rows(
    dataset_root: Path,
    splits: set[str],
    samples: int | None,
) -> list[dict[str, str]]:
    with (dataset_root / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in splits]
    rows.sort(key=lambda row: (row["split"], row["sample_key"]))
    if samples is not None:
        rows = rows[:samples]
    return rows


def iter_samples(
    dataset_root: Path,
    rows: list[dict[str, str]],
) -> Iterator[tuple[dict[str, str], dict[str, bytes]]]:
    rows_by_shard: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_shard.setdefault(row["output_shard"], []).append(row)
    for shard_name, shard_rows in sorted(rows_by_shard.items()):
        row_by_key = {row["sample_key"]: row for row in shard_rows}
        wanted = {
            key + suffix: (key, suffix)
            for key in row_by_key
            for suffix in REQUIRED_SUFFIXES
        }
        payloads: dict[str, dict[str, bytes]] = {}
        with tarfile.open(dataset_root / shard_name, "r") as archive:
            for member in archive:
                target = wanted.get(member.name)
                if target is None:
                    continue
                key, suffix = target
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read tar member: {member.name}")
                sample_payloads = payloads.setdefault(key, {})
                sample_payloads[suffix] = extracted.read()
                if len(sample_payloads) == len(REQUIRED_SUFFIXES):
                    yield row_by_key[key], sample_payloads
                    del payloads[key]
        if payloads:
            missing = next(iter(payloads))
            raise ValueError(f"Incomplete sample in {shard_name}: {missing}")


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def evaluate_repair(
    sequence: str,
    repair_image: Image.Image,
    expected_sequence: str,
    endpoint_boxes: dict[tuple[int, int], tuple[float, float, float, float]] | None = None,
) -> dict:
    expected = parse_chemsvg_v4(expected_sequence)
    result = {
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
        "false_edges": 0,
        "missed_edges": len(expected.strokes),
    }
    predicted = None
    try:
        repaired = repair_chemsvg_v4_from_image(
            sequence,
            repair_image,
            config=InkGraphRepairConfig(aromatic_vote_source="vertices"),
            endpoint_boxes=endpoint_boxes,
            validate_chemistry=False,
        )
        predicted = parse_chemsvg_v4(repaired.sequence)
        comparison = compare_graphs(expected, predicted)
        result.update(asdict(comparison))
        result["repair_success"] = True
        result["false_edges"] = max(
            0,
            comparison.predicted_edges - comparison.correct_edges,
        )
        result["missed_edges"] = max(
            0,
            comparison.expected_edges - comparison.correct_edges,
        )
        try:
            actual_mol = mol_from_chemsvg_v4(repaired.sequence)
            expected_mol = mol_from_chemsvg_v4(expected_sequence)
            result["sdf_valid"] = True
            result["chemical_match"] = (
                Chem.MolToInchiKey(actual_mol) == Chem.MolToInchiKey(expected_mol)
            )
        except Exception as chemistry_error:
            result["chemistry_error"] = str(chemistry_error)
    except Exception as error:
        result["error"] = str(error)
    return result | {"predicted": predicted}


def summarize_mode(records: list[dict], mode: str) -> dict:
    values = [record[mode] for record in records]
    expected_edges = sum(value["expected_edges"] for value in values)
    correct_edges = sum(value["correct_edges"] for value in values)
    predicted_edges = sum(value["predicted_edges"] for value in values)
    return {
        "samples": len(values),
        "repair_success_rate": _safe_ratio(
            sum(value["repair_success"] for value in values), len(values)
        ),
        "sdf_valid_rate": _safe_ratio(
            sum(value["sdf_valid"] for value in values), len(values)
        ),
        "chemical_match_rate": _safe_ratio(
            sum(value["chemical_match"] for value in values), len(values)
        ),
        "exact_topology_rate": _safe_ratio(
            sum(value["exact_topology"] for value in values), len(values)
        ),
        "edge_precision": _safe_ratio(correct_edges, predicted_edges),
        "edge_recall": _safe_ratio(correct_edges, expected_edges),
        "false_edges": sum(value["false_edges"] for value in values),
        "missed_edges": sum(value["missed_edges"] for value in values),
    }


def _group_summary(records: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record[field]), []).append(record)
    return {
        key: {
            "baseline": summarize_mode(group, "baseline"),
            "oracle_bond": summarize_mode(group, "oracle_bond"),
            "oracle_bbox_bond": summarize_mode(group, "oracle_bbox_bond"),
        }
        for key, group in sorted(groups.items())
    }


def summarize_records(records: list[dict], *, excluded_samples: int = 0) -> dict:
    baseline = summarize_mode(records, "baseline")
    oracle = summarize_mode(records, "oracle_bond")
    oracle_bbox = summarize_mode(records, "oracle_bbox_bond")
    baseline_false = baseline["false_edges"]
    false_edge_reduction = (
        (baseline_false - oracle["false_edges"]) / baseline_false
        if baseline_false
        else 0.0
    )
    baseline_wrong_oracle_correct = sum(
        not record["baseline"]["exact_topology"]
        and record["oracle_bond"]["exact_topology"]
        for record in records
    )
    baseline_correct_oracle_wrong = sum(
        record["baseline"]["exact_topology"]
        and not record["oracle_bond"]["exact_topology"]
        for record in records
    )
    large_text_records = [
        record
        for record in records
        if record["font_bucket"] in {"large", "extreme"}
        and record["visible_text_count"] > 0
    ]
    return {
        "schema_version": "oracle-semantic-layer-0.1",
        "samples": len(records),
        "excluded_samples": excluded_samples,
        "baseline": baseline,
        "oracle_bond": oracle,
        "oracle_bbox_bond": oracle_bbox,
        "delta": {
            "false_edge_reduction": false_edge_reduction,
            "edge_recall": oracle["edge_recall"] - baseline["edge_recall"],
            "exact_topology_rate": (
                oracle["exact_topology_rate"] - baseline["exact_topology_rate"]
            ),
            "chemical_match_rate": (
                oracle["chemical_match_rate"] - baseline["chemical_match_rate"]
            ),
            "baseline_wrong_oracle_correct": baseline_wrong_oracle_correct,
            "baseline_correct_oracle_wrong": baseline_correct_oracle_wrong,
        },
        "bbox_delta": {
            "false_edge_reduction": (
                (baseline_false - oracle_bbox["false_edges"]) / baseline_false
                if baseline_false
                else 0.0
            ),
            "edge_recall": oracle_bbox["edge_recall"] - baseline["edge_recall"],
            "exact_topology_rate": (
                oracle_bbox["exact_topology_rate"] - baseline["exact_topology_rate"]
            ),
            "chemical_match_rate": (
                oracle_bbox["chemical_match_rate"] - baseline["chemical_match_rate"]
            ),
            "baseline_wrong_oracle_correct": sum(
                not record["baseline"]["exact_topology"]
                and record["oracle_bbox_bond"]["exact_topology"]
                for record in records
            ),
            "baseline_correct_oracle_wrong": sum(
                record["baseline"]["exact_topology"]
                and not record["oracle_bbox_bond"]["exact_topology"]
                for record in records
            ),
        },
        "large_text": {
            "samples": len(large_text_records),
            "baseline": summarize_mode(large_text_records, "baseline"),
            "oracle_bond": summarize_mode(large_text_records, "oracle_bond"),
            "oracle_bbox_bond": summarize_mode(
                large_text_records, "oracle_bbox_bond"
            ),
        },
        "by_font_bucket": _group_summary(records, "font_bucket"),
        "by_font_family": _group_summary(records, "font_family_id"),
        "by_variant": _group_summary(records, "variant"),
        "by_split": _group_summary(records, "split"),
    }


def serializable_record(record: dict) -> dict:
    clean = dict(record)
    for mode in ("baseline", "oracle_bond", "oracle_bbox_bond"):
        clean[mode] = {
            key: value
            for key, value in clean[mode].items()
            if key != "predicted"
        }
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/chemvector-v073-pixel-ocr-original-refine-1k/last.pt",
    )
    parser.add_argument(
        "--dataset-root",
        default="Data/sdf/chemvision_layers_v0_1/pilot_10k",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluations/chemvision-pilot-oracle-bond-v073",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "internal_test"),
        default=("validation", "internal_test"),
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument("--failure-previews", type=int, default=40)
    args = parser.parse_args()
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.set_num_threads(args.cpu_threads)
    RDLogger.DisableLog("rdApp.*")

    checkpoint = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    excluded_path = output_dir / "excluded.jsonl"
    rows = load_manifest_rows(dataset_root, set(args.splits), args.samples)
    completed = set()
    records = []
    if records_path.is_file():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed = {record["sample_key"] for record in records}
    excluded = []
    if excluded_path.is_file():
        excluded = [
            json.loads(line)
            for line in excluded_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed.update(record["sample_key"] for record in excluded)
    config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_root": str(dataset_root),
        "splits": list(args.splits),
        "samples": args.samples,
        "device": args.device,
        "vertex_threshold": args.vertex_threshold,
        "edge_threshold": args.edge_threshold,
        "ocr_roi_radius": args.ocr_roi_radius,
    }
    config_path = output_dir / "benchmark_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Existing output directory uses a different configuration")
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    device = torch.device(args.device)
    model, _ = load_model(checkpoint, device)
    strip_autoregressive_modules(model)
    preview_count = len(list((output_dir / "previews").glob("*.png")))
    pending_rows = [row for row in rows if row["sample_key"] not in completed]
    for index, (row, payloads) in enumerate(
        iter_samples(dataset_root, pending_rows),
        start=len(records) + 1,
    ):
        combined = Image.open(io.BytesIO(payloads[".png"])).convert("RGB")
        bond_layer = Image.open(io.BytesIO(payloads[".bond.png"])).convert("L")
        bond_ink = bond_alpha_to_ink_image(bond_layer)
        chemmol = json.loads(payloads[".chemmol.json"])
        render = json.loads(payloads[".render.json"])
        layers = json.loads(payloads[".layers.json"])
        try:
            expected_sequence = serialize_chemsvg_v4(chemmol, render)
        except ChemSVGCoordinateCollisionError as error:
            exclusion = {
                "sample_key": row["sample_key"],
                "molecule_id": row["molecule_id"],
                "split": row["split"],
                "variant": row["variant"],
                "reason": "chemsvg_coordinate_collision",
                "error": str(error),
            }
            excluded.append(exclusion)
            completed.add(row["sample_key"])
            with excluded_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(exclusion, ensure_ascii=False) + "\n")
            continue
        pixels = image_tensor(combined).unsqueeze(0).to(device)
        started = time.perf_counter()
        generated = generate_parallel_chemsvg_v6(
            model,
            pixels,
            vertex_threshold=args.vertex_threshold,
            edge_threshold=args.edge_threshold,
            ocr_roi_radius=args.ocr_roi_radius,
        )
        baseline = evaluate_repair(generated.sequence, combined, expected_sequence)
        oracle = evaluate_repair(generated.sequence, bond_ink, expected_sequence)
        endpoint_boxes = endpoint_boxes_for_sequence(
            generated.sequence,
            layers,
            combined.size,
        )
        oracle_bbox = evaluate_repair(
            generated.sequence,
            bond_ink,
            expected_sequence,
            endpoint_boxes=endpoint_boxes,
        )
        record = {
            "sample_key": row["sample_key"],
            "molecule_id": row["molecule_id"],
            "split": row["split"],
            "variant": row["variant"],
            "font_family_id": row["font_family_id"],
            "font_size": int(row["font_size"]),
            "font_bucket": row["font_bucket"],
            "visible_text_count": sum(atom["visible_text"] for atom in layers["atoms"]),
            "visible_texts": [
                atom["text"] for atom in layers["atoms"] if atom["visible_text"]
            ],
            "raw_vertices": len(generated.vertices),
            "raw_strokes": len(generated.strokes),
            "mapped_text_boxes": len(endpoint_boxes),
            "seconds": time.perf_counter() - started,
            "baseline": baseline,
            "oracle_bond": oracle,
            "oracle_bbox_bond": oracle_bbox,
        }
        clean = serializable_record(record)
        records.append(clean)
        completed.add(row["sample_key"])
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")

        changed = (
            baseline["false_edges"] != oracle["false_edges"]
            or baseline["false_edges"] != oracle_bbox["false_edges"]
        )
        if changed and preview_count < args.failure_previews:
            preview_dir = output_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            expected = parse_chemsvg_v4(expected_sequence)
            draw_overlay(combined, expected, baseline["predicted"]).save(
                preview_dir / f"{row['sample_key']}.baseline.png"
            )
            draw_overlay(combined, expected, oracle["predicted"]).save(
                preview_dir / f"{row['sample_key']}.oracle-bond.png"
            )
            draw_overlay(combined, expected, oracle_bbox["predicted"]).save(
                preview_dir / f"{row['sample_key']}.oracle-bbox-bond.png"
            )
            bond_ink.save(preview_dir / f"{row['sample_key']}.bond.png")
            preview_count += 4
        if index % 25 == 0 or index == len(rows):
            summary = summarize_records(records, excluded_samples=len(excluded))
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"hard": summary["delta"], "bbox": summary["bbox_delta"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = summarize_records(records, excluded_samples=len(excluded))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
