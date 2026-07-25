#!/usr/bin/env python3
"""Evaluate the predicted-Junction v0.8 ChemScene pipeline end to end."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import islice
import json
import math
from pathlib import Path
import time

from rdkit import Chem
import torch

from Data.sdf.chemscene_v0_1 import mol_from_chemscene_v0_1
from chemvector.audit_bond_first_decomposition import sample_keys_sha256
from chemvector.benchmark_public_ocsr import sdf_graph_match
from chemvector.data import (
    ChemVectorIterableDataset,
    rendered_atom_labels,
    structure_labels,
)
from chemvector.evaluate_node_conditioned_bond import match_vertices
from chemvector.infer_chemscene_hybrid import (
    ChemSceneHybridPipeline,
    compile_scene_with_aromatic_fallback,
)
from chemvector.pretrain_chemscene_bond_query import ChemSceneBondCollator
from chemvector.train import expand_shards


def summarize(records: list[dict]) -> dict:
    samples = max(len(records), 1)
    successful = [row for row in records if row["sdf_valid"]]
    matched_text = sum(row["matched_text_junctions"] for row in records)
    edge_true_positives = sum(row["topology_true_positives"] for row in records)
    edge_false_positives = sum(row["topology_false_positives"] for row in records)
    edge_false_negatives = sum(row["topology_false_negatives"] for row in records)
    edge_precision = edge_true_positives / max(
        edge_true_positives + edge_false_positives, 1
    )
    edge_recall = edge_true_positives / max(
        edge_true_positives + edge_false_negatives, 1
    )
    return {
        "samples": len(records),
        "scene_valid_rate": sum(row["scene_valid"] for row in records) / samples,
        "sdf_valid_rate": len(successful) / samples,
        "junction_count_accuracy": sum(row["junction_count_match"] for row in records)
        / samples,
        "junction_exact_6px": sum(row["junction_exact_6px"] for row in records)
        / samples,
        "bond_count_accuracy": sum(row["bond_count_match"] for row in records)
        / samples,
        "topology_exact_rate": sum(row["topology_exact"] for row in records)
        / samples,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": 2.0 * edge_precision * edge_recall
        / max(edge_precision + edge_recall, 1e-12),
        "nonstereo_structure_match_rate": sum(
            row["nonstereo_structure_match"] for row in records
        )
        / samples,
        "chemical_match_rate": sum(row["chemical_match"] for row in records)
        / samples,
        "text_ocr_accuracy_on_matched_junctions": sum(
            row["correct_text_elements"] for row in records
        )
        / max(matched_text, 1),
        "aromatic_fallback_rate": sum(row["aromatic_fallback"] for row in records)
        / samples,
        "average_self_loop_objects": sum(row["self_loop_objects"] for row in records)
        / samples,
        "average_duplicate_pair_objects": sum(
            row["duplicate_pair_objects"] for row in records
        )
        / samples,
        "average_endpoint_to_junction_distance_mean_pixels": sum(
            row["endpoint_distance_mean_pixels"] for row in records
        )
        / samples,
        "endpoint_to_junction_distance_max_pixels": max(
            (row["endpoint_distance_max_pixels"] for row in records), default=0.0
        ),
        "average_seconds": sum(row["seconds"] for row in records) / samples,
        "atom_repaired_samples": sum(bool(row["atom_decode_changes"]) for row in records),
        "atom_state_changes": sum(len(row["atom_decode_changes"]) for row in records),
        "average_atom_decode_states": sum(
            row["atom_decode_attempted_states"] for row in records
        )
        / samples,
        "failure_categories": dict(
            sorted(
                Counter(
                    row["failure_category"]
                    for row in records
                    if row["failure_category"]
                ).items()
            )
        ),
    }


def failure_category(error: Exception) -> str:
    message = str(error).lower()
    if "junction" in message:
        return "junction"
    if "valence" in message:
        return "valence"
    if "kekul" in message or "aromatic" in message:
        return "aromatic"
    if "connected" in message or "component" in message:
        return "disconnected"
    return "other"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-checkpoint",
        default="outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
    )
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument("--assembler-checkpoint")
    parser.add_argument("--graph-corrector-checkpoint")
    parser.add_argument("--ocr-checkpoint")
    parser.add_argument("--ctc-ocr-checkpoint")
    parser.add_argument("--text-bbox-checkpoint")
    parser.add_argument("--ctc-anchor-fallback-margin", type=float)
    parser.add_argument("--bond-set-selector-checkpoint")
    parser.add_argument("--endpoint-grouper-checkpoint")
    parser.add_argument("--line-band-checkpoint")
    parser.add_argument(
        "--junction-projection-score-source",
        choices=("detector", "selector"),
        default="detector",
    )
    parser.add_argument(
        "--graph-correction-mode",
        choices=("none", "type_only", "full"),
        default="none",
    )
    parser.add_argument("--graph-keep-threshold", type=float)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--bond-threshold", type=float, default=0.25)
    parser.add_argument(
        "--junction-source",
        choices=("vertex", "bond_endpoints", "junction_aware_bonds"),
        default="vertex",
    )
    parser.add_argument("--endpoint-cluster-radius-pixels", type=float, default=12.0)
    parser.add_argument("--text-attach-radius-pixels", type=float, default=24.0)
    parser.add_argument(
        "--atom-decode-mode",
        choices=("greedy", "structured_invalid_only"),
        default="greedy",
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("nearest", "global_unique", "learned", "learned_rescue"),
        default="nearest",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint),
        Path(args.bond_checkpoint),
        device,
        Path(args.assembler_checkpoint) if args.assembler_checkpoint else None,
        Path(args.graph_corrector_checkpoint)
        if args.graph_corrector_checkpoint
        else None,
        Path(args.ocr_checkpoint) if args.ocr_checkpoint else None,
        Path(args.ctc_ocr_checkpoint) if args.ctc_ocr_checkpoint else None,
        Path(args.text_bbox_checkpoint) if args.text_bbox_checkpoint else None,
        args.ctc_anchor_fallback_margin,
        Path(args.bond_set_selector_checkpoint)
        if args.bond_set_selector_checkpoint
        else None,
        Path(args.endpoint_grouper_checkpoint)
        if args.endpoint_grouper_checkpoint
        else None,
        Path(args.line_band_checkpoint) if args.line_band_checkpoint else None,
        args.junction_projection_score_source,
    )
    records = list(
        islice(
            iter(
                ChemVectorIterableDataset(
                    expand_shards(args.validation_shards), sequence_version="0.4"
                )
            ),
            args.samples,
        )
    )
    if len(records) != args.samples:
        raise ValueError("Validation dataset contains fewer samples than requested")
    collator = ChemSceneBondCollator()
    output_rows = []
    for index, sample in enumerate(records, start=1):
        batch = collator([sample])
        pixels = batch["pixel_values"].to(device)
        oracle_scene = batch["scenes"][0]
        expected = mol_from_chemscene_v0_1(oracle_scene)
        labels = structure_labels(sample)
        atoms = labels["atoms"]
        atom_index_by_id = {
            int(atom["atom_id"]): index for index, atom in enumerate(atoms)
        }
        visible_labels = rendered_atom_labels(sample)
        expected_points = torch.tensor(
            [
                (
                    atom["center"][0] / labels["width"],
                    atom["center"][1] / labels["height"],
                )
                for atom in atoms
            ],
            dtype=torch.float32,
        )
        expected_elements = [atom["element"] for atom in atoms]
        expected_text = [
            bool(visible_labels.get(int(atom["atom_id"]), ("", False))[1])
            for atom in atoms
        ]
        expected_edges = {
            tuple(
                sorted(
                    (
                        atom_index_by_id[int(bond["begin_atom"])],
                        atom_index_by_id[int(bond["end_atom"])],
                    )
                )
            )
            for bond in labels["bonds"]
        }
        slices = ["all", f"variant_{labels.get('variant', 'unknown')}"]
        display_types = {str(bond["display_type"]).lower() for bond in labels["bonds"]}
        slices.extend(f"display_{name}" for name in sorted(display_types))
        maximum_bond_length = max(
            (
                math.dist(bond["begin"], bond["end"])
                for bond in labels["bonds"]
            ),
            default=0.0,
        )
        if maximum_bond_length >= 96.0:
            slices.append("has_bond_at_least_96px")
        stress_operation = labels.get("render_parameters", {}).get(
            "stress_operation"
        )
        if stress_operation:
            slices.append(f"stress_{stress_operation}")
        row = {
            "key": sample["key"],
            "expected_junctions": len(expected_points),
            "expected_bonds": expected.GetNumBonds(),
            "predicted_junctions": 0,
            "predicted_bonds": 0,
            "scene_valid": False,
            "sdf_valid": False,
            "junction_count_match": False,
            "junction_exact_6px": False,
            "bond_count_match": False,
            "topology_exact": False,
            "topology_true_positives": 0,
            "topology_false_positives": 0,
            "topology_false_negatives": len(expected_edges),
            "matched_junctions": 0,
            "matched_text_junctions": 0,
            "correct_text_elements": 0,
            "nonstereo_structure_match": False,
            "chemical_match": False,
            "aromatic_fallback": False,
            "self_loop_objects": 0,
            "duplicate_pair_objects": 0,
            "endpoint_distance_mean_pixels": 0.0,
            "endpoint_distance_max_pixels": 0.0,
            "failure_category": None,
            "error": None,
            "atom_decode_changes": [],
            "atom_decode_attempted_states": 0,
            "variant": labels.get("variant"),
            "stress_operation": stress_operation,
            "slices": slices,
        }
        started = time.perf_counter()
        try:
            scene, diagnostics = pipeline.predict(
                pixels,
                sample["key"],
                real_threshold=args.real_threshold,
                bond_threshold=args.bond_threshold,
                assignment_mode=args.assignment_mode,
                graph_correction_mode=args.graph_correction_mode,
                graph_keep_threshold=args.graph_keep_threshold,
                junction_source=args.junction_source,
                endpoint_cluster_radius_pixels=args.endpoint_cluster_radius_pixels,
                text_attach_radius_pixels=args.text_attach_radius_pixels,
                atom_decode_mode=args.atom_decode_mode,
            )
            row["scene_valid"] = True
            row["predicted_junctions"] = len(scene["junctions"])
            row["predicted_bonds"] = len(scene["bond_objects"])
            row["junction_count_match"] = len(scene["junctions"]) == len(expected_points)
            row["bond_count_match"] = len(scene["bond_objects"]) == expected.GetNumBonds()
            row["self_loop_objects"] = diagnostics["self_loop_objects"]
            row["duplicate_pair_objects"] = diagnostics["duplicate_pair_objects"]
            row["endpoint_distance_mean_pixels"] = diagnostics[
                "endpoint_to_junction_distance_mean_pixels"
            ]
            row["endpoint_distance_max_pixels"] = diagnostics[
                "endpoint_to_junction_distance_max_pixels"
            ]
            row["atom_decode_changes"] = diagnostics.get("changes", [])
            row["atom_decode_attempted_states"] = int(
                diagnostics.get("attempted_states", 0)
            )
            predicted_points = torch.tensor(
                [
                    (item["coordinate"][0] / 1023.0, item["coordinate"][1] / 1023.0)
                    for item in scene["junctions"]
                ],
                dtype=torch.float32,
            )
            assignment = match_vertices(predicted_points, expected_points)
            row["matched_junctions"] = len(assignment)
            row["junction_exact_6px"] = (
                len(assignment) == len(predicted_points) == len(expected_points)
            )
            coordinate_to_junction = {
                tuple(item["coordinate"]): junction_index
                for junction_index, item in enumerate(scene["junctions"])
            }
            mapped_edge_rows = []
            unmapped_edges = 0
            for bond in scene["bond_objects"]:
                begin = coordinate_to_junction.get(tuple(bond["endpoints"][0]))
                end = coordinate_to_junction.get(tuple(bond["endpoints"][1]))
                if (
                    begin is None
                    or end is None
                    or begin not in assignment
                    or end not in assignment
                ):
                    unmapped_edges += 1
                    continue
                mapped_edge_rows.append(
                    tuple(sorted((assignment[begin], assignment[end])))
                )
            mapped_edges = set(mapped_edge_rows)
            duplicate_edges = len(mapped_edge_rows) - len(mapped_edges)
            row["topology_true_positives"] = len(mapped_edges & expected_edges)
            row["topology_false_positives"] = (
                len(mapped_edges - expected_edges) + unmapped_edges + duplicate_edges
            )
            row["topology_false_negatives"] = len(expected_edges - mapped_edges)
            row["topology_exact"] = bool(row["junction_exact_6px"]) and not (
                row["topology_false_positives"]
                or row["topology_false_negatives"]
            )
            for predicted_index, expected_index in assignment.items():
                if not expected_text[expected_index]:
                    continue
                row["matched_text_junctions"] += 1
                row["correct_text_elements"] += int(
                    scene["junctions"][predicted_index]["atom"]["element"]
                    == expected_elements[expected_index]
                )
            sdf, compiled_scene, aromatic_fallback = compile_scene_with_aromatic_fallback(
                scene
            )
            row["aromatic_fallback"] = aromatic_fallback
            predicted = mol_from_chemscene_v0_1(compiled_scene)
            row["sdf_valid"] = bool(sdf) and predicted is not None
            row["nonstereo_structure_match"] = sdf_graph_match(
                expected, predicted, use_chirality=False
            )
            row["chemical_match"] = sdf_graph_match(
                expected, predicted, use_chirality=True
            )
        except Exception as error:
            row["failure_category"] = failure_category(error)
            row["error"] = str(error)
        row["seconds"] = time.perf_counter() - started
        output_rows.append(row)
        if index % 25 == 0 or index == len(records):
            partial = summarize(output_rows)
            print(
                f"evaluated={index}/{len(records)} valid={partial['sdf_valid_rate']:.4f} "
                f"junction_exact={partial['junction_exact_6px']:.4f} "
                f"sdf_exact={partial['chemical_match_rate']:.4f}",
                flush=True,
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    slice_rows: dict[str, list[dict]] = defaultdict(list)
    for row in output_rows:
        for name in row["slices"]:
            slice_rows[name].append(row)
    summary = {
        **summarize(output_rows),
        "sample_keys_sha256": sample_keys_sha256(output_rows),
        "slice_metrics": {
            name: summarize(rows) for name, rows in sorted(slice_rows.items())
        },
        "protocol": {
            "selector_checkpoint": args.selector_checkpoint,
            "bond_checkpoint": args.bond_checkpoint,
            "assembler_checkpoint": args.assembler_checkpoint,
            "graph_corrector_checkpoint": args.graph_corrector_checkpoint,
            "ocr_checkpoint": args.ocr_checkpoint,
            "ctc_ocr_checkpoint": args.ctc_ocr_checkpoint,
            "text_bbox_checkpoint": args.text_bbox_checkpoint,
            "ctc_anchor_fallback_margin": args.ctc_anchor_fallback_margin,
            "bond_set_selector_checkpoint": args.bond_set_selector_checkpoint,
            "endpoint_grouper_checkpoint": args.endpoint_grouper_checkpoint,
            "line_band_checkpoint": args.line_band_checkpoint,
            "junction_projection_score_source": args.junction_projection_score_source,
            "graph_correction_mode": args.graph_correction_mode,
            "graph_keep_threshold": args.graph_keep_threshold,
            "real_threshold": args.real_threshold,
            "bond_threshold": args.bond_threshold,
            "assignment_mode": args.assignment_mode,
            "junction_source": args.junction_source,
            "endpoint_cluster_radius_pixels": args.endpoint_cluster_radius_pixels,
            "text_attach_radius_pixels": args.text_attach_radius_pixels,
            "atom_decode_mode": args.atom_decode_mode,
            "uses_oracle_junctions": False,
            "uses_oracle_ocr": False,
            "uses_mr_tadf_for_tuning": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
