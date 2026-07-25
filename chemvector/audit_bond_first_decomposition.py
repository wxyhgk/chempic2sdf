#!/usr/bin/env python3
"""Decompose Bond-first errors into count, geometry, topology, type, and SDF stages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import islice
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import mol_from_chemscene_v0_1
from chemvector.bond_object import BOND_TYPE_TO_ID, DIRECTION_TO_ID
from chemvector.data import (
    ChemVectorIterableDataset,
    rendered_atom_labels,
    structure_labels,
)
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
    decode_dense_bond_objects,
)
from chemvector.experiment_loop.artifacts import sha256_file
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_chemscene_bond_query import ChemSceneBondCollator
from chemvector.pretrain_dense_bond_object import (
    _apply_detector,
    _extract_detector_features,
    _forward_detector,
    _scene_from_dense_objects,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.structure_gate import StructureStyleGate
from chemvector.train import expand_shards


def sample_slices(sample: dict) -> list[str]:
    layers = structure_labels(sample)
    names = ["all"]
    display_types = {bond["display_type"] for bond in layers["bonds"]}
    names.extend(f"display_{name.lower()}" for name in sorted(display_types))
    if layers.get("text_supervision_available", True):
        visible_texts = [
            (bool(atom["visible_text"]), str(atom.get("text", "")))
            for atom in layers["atoms"]
        ]
    else:
        visible_texts = [
            (visible, text)
            for _, (text, visible) in sorted(rendered_atom_labels(sample).items())
        ]
    if any(visible for visible, _ in visible_texts):
        names.append("has_text")
    if any(visible and len(text) > 1 for visible, text in visible_texts):
        names.append("has_multichar_text")
    stress = layers.get("render_parameters", {}).get("stress_operation")
    if stress:
        names.append(f"stress_{stress}")
    if any(_bond_length(bond) >= 96.0 for bond in layers["bonds"]):
        names.append("has_bond_at_least_96px")
    if len(layers["atoms"]) >= 40:
        names.append("atoms_at_least_40")
    try:
        scene = ChemSceneBondCollator()([sample])["scenes"][0]
        mol = mol_from_chemscene_v0_1(scene)
        rings = mol.GetRingInfo().AtomRings()
        if any(len(set(first) & set(second)) >= 2 for offset, first in enumerate(rings) for second in rings[offset + 1 :]):
            names.append("has_fused_rings")
    except Exception:
        pass
    return names


def _bond_length(bond: dict) -> float:
    return math.dist(bond["begin"], bond["end"])


def _expected_bonds(sample: dict) -> list[dict]:
    layers = structure_labels(sample)
    return [
        {
            "pair": tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))),
            "endpoints": torch.tensor(
                (bond["begin"], bond["end"]), dtype=torch.float32
            )
            / torch.tensor(
                (layers["width"], layers["height"]),
                dtype=torch.float32,
            ),
            "display": BOND_TYPE_TO_ID[bond["display_type"]],
            "chemical": BOND_TYPE_TO_ID[bond["chemical_type"]],
            "direction": DIRECTION_TO_ID[bond["direction"]],
        }
        for bond in layers["bonds"]
    ]


def _atom_centers(sample: dict) -> torch.Tensor:
    layers = structure_labels(sample)
    width = float(layers["width"])
    height = float(layers["height"])
    atoms = sorted(layers["atoms"], key=lambda atom: int(atom["atom_id"]))
    if [int(atom["atom_id"]) for atom in atoms] != list(range(len(atoms))):
        raise ValueError(f"Atom IDs are not dense in {sample['key']}")
    return torch.tensor(
        [(atom["center"][0] / width, atom["center"][1] / height) for atom in atoms],
        dtype=torch.float32,
    )


def _aligned_endpoint_errors(
    predicted: torch.Tensor, expected: torch.Tensor, image_size: float
) -> torch.Tensor:
    forward = torch.linalg.vector_norm((predicted - expected) * image_size, dim=-1)
    reverse = torch.linalg.vector_norm(
        (predicted - expected.flip(0)) * image_size, dim=-1
    )
    return forward if float(forward.sum()) <= float(reverse.sum()) else reverse


def analyze_objects(
    sample: dict,
    objects: list[dict],
    *,
    endpoint_radius_pixels: float = 6.0,
) -> dict:
    expected = _expected_bonds(sample)
    expected_count = len(expected)
    predicted_count = len(objects)
    expected_by_pair = {item["pair"]: item for item in expected}
    expected_pairs = set(expected_by_pair)

    geometry_matches = 0
    endpoint_errors: list[float] = []
    if expected and objects:
        predicted_endpoints = torch.stack([item["endpoints"].float().cpu() for item in objects])
        expected_endpoints = torch.stack([item["endpoints"] for item in expected])
        forward = (
            predicted_endpoints[:, None, 0].sub(expected_endpoints[None, :, 0]).abs().sum(-1)
            + predicted_endpoints[:, None, 1].sub(expected_endpoints[None, :, 1]).abs().sum(-1)
        )
        reverse = (
            predicted_endpoints[:, None, 0].sub(expected_endpoints[None, :, 1]).abs().sum(-1)
            + predicted_endpoints[:, None, 1].sub(expected_endpoints[None, :, 0]).abs().sum(-1)
        )
        rows, columns = linear_sum_assignment(torch.minimum(forward, reverse).numpy())
        for row, column in zip(rows, columns):
            errors = _aligned_endpoint_errors(
                predicted_endpoints[row], expected_endpoints[column], 768.0
            )
            endpoint_errors.extend(float(value) for value in errors)
            geometry_matches += int(bool((errors <= endpoint_radius_pixels).all()))

    centers = _atom_centers(sample)
    predicted_pairs: set[tuple[int, int]] = set()
    candidates_by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    topology_true_positives = 0
    display_correct = chemical_correct = direction_correct = 0
    self_loops = duplicate_pairs = nonbond_pairs = 0
    for obj in sorted(objects, key=lambda item: float(item["score"]), reverse=True):
        endpoints = obj["endpoints"].float().cpu()
        nearest = torch.cdist(endpoints, centers).argmin(-1).tolist()
        pair = tuple(sorted((int(nearest[0]), int(nearest[1]))))
        if pair[0] == pair[1]:
            self_loops += 1
            continue
        candidates_by_pair[pair].append(obj)
        if pair in predicted_pairs:
            duplicate_pairs += 1
            continue
        predicted_pairs.add(pair)
        target = expected_by_pair.get(pair)
        if target is None:
            nonbond_pairs += 1
            continue
        topology_true_positives += 1
        display_correct += int(int(obj["display"]) == target["display"])
        chemical_correct += int(int(obj["chemical"]) == target["chemical"])
        direction_correct += int(int(obj["direction"]) == target["direction"])

    topology_false_positives = len(predicted_pairs - expected_pairs)
    topology_false_negatives = len(expected_pairs - predicted_pairs)
    topology_candidate_oracle = expected_pairs.issubset(predicted_pairs)
    chemical_candidate_oracle = topology_candidate_oracle and all(
        any(int(obj["chemical"]) == target["chemical"] for obj in candidates_by_pair[pair])
        for pair, target in expected_by_pair.items()
    )
    recoverable_topology_exact = predicted_pairs == expected_pairs
    count_exact = predicted_count == expected_count
    strict_topology_exact = (
        count_exact
        and recoverable_topology_exact
        and self_loops == 0
        and duplicate_pairs == 0
    )
    geometry_exact = count_exact and geometry_matches == expected_count
    recoverable_display_exact = (
        recoverable_topology_exact and display_correct == expected_count
    )
    recoverable_chemical_exact = (
        recoverable_topology_exact and chemical_correct == expected_count
    )
    display_exact = strict_topology_exact and display_correct == expected_count
    chemical_exact = strict_topology_exact and chemical_correct == expected_count
    direction_exact = strict_topology_exact and direction_correct == expected_count
    return {
        "expected_bonds": expected_count,
        "predicted_bonds": predicted_count,
        "count_exact": count_exact,
        "geometry_matches": geometry_matches,
        "geometry_exact": geometry_exact,
        "topology_true_positives": topology_true_positives,
        "topology_false_positives": topology_false_positives,
        "strict_topology_false_positives": (
            topology_false_positives + self_loops + duplicate_pairs
        ),
        "topology_false_negatives": topology_false_negatives,
        "topology_candidate_oracle": topology_candidate_oracle,
        "chemical_candidate_oracle": chemical_candidate_oracle,
        "recoverable_topology_exact": recoverable_topology_exact,
        "topology_exact": strict_topology_exact,
        "display_correct": display_correct,
        "recoverable_display_exact": recoverable_display_exact,
        "display_exact": display_exact,
        "chemical_correct": chemical_correct,
        "recoverable_chemical_exact": recoverable_chemical_exact,
        "chemical_exact": chemical_exact,
        "direction_correct": direction_correct,
        "direction_exact": direction_exact,
        "self_loops": self_loops,
        "duplicate_pairs": duplicate_pairs,
        "nonbond_pairs": nonbond_pairs,
        "endpoint_error_mean_pixels": (
            float(np.mean(endpoint_errors)) if endpoint_errors else None
        ),
        "endpoint_error_max_pixels": max(endpoint_errors, default=None),
    }


def failure_stage(metrics: dict, compiled_sdf_exact: bool) -> str:
    if compiled_sdf_exact:
        return "exact_sdf"
    if not metrics["count_exact"]:
        return "count"
    if not metrics["recoverable_topology_exact"]:
        return "topology_or_endpoint_assignment"
    if metrics["self_loops"] or metrics["duplicate_pairs"]:
        return "duplicate_or_self_loop"
    if not metrics["chemical_exact"]:
        return "chemical_type"
    return "compile_or_stereo"


def summarize(records: list[dict]) -> dict:
    samples = max(len(records), 1)
    expected_bonds = sum(row["predicted"]["expected_bonds"] for row in records)
    predicted_bonds = sum(row["predicted"]["predicted_bonds"] for row in records)
    true_positives = sum(
        row["predicted"]["topology_true_positives"] for row in records
    )
    recoverable_false_positives = sum(
        row["predicted"]["topology_false_positives"] for row in records
    )
    strict_false_positives = sum(
        row["predicted"]["strict_topology_false_positives"] for row in records
    )
    false_negatives = sum(
        row["predicted"]["topology_false_negatives"] for row in records
    )
    recoverable_precision = true_positives / max(
        true_positives + recoverable_false_positives, 1
    )
    strict_precision = true_positives / max(
        true_positives + strict_false_positives, 1
    )
    recall = true_positives / max(true_positives + false_negatives, 1)
    summary = {
        "samples": len(records),
        "average_expected_bonds": expected_bonds / samples,
        "average_predicted_bonds": predicted_bonds / samples,
        "predicted_count_accuracy": sum(
            row["predicted"]["count_exact"] for row in records
        )
        / samples,
        "known_count_geometry_exact": sum(
            row["known_count"]["geometry_exact"] for row in records
        )
        / samples,
        "predicted_geometry_exact": sum(
            row["predicted"]["geometry_exact"] for row in records
        )
        / samples,
        "known_count_topology_exact": sum(
            row["known_count"]["topology_exact"] for row in records
        )
        / samples,
        "predicted_recoverable_topology_exact": sum(
            row["predicted"]["recoverable_topology_exact"] for row in records
        )
        / samples,
        "topology_candidate_oracle": sum(
            row["predicted"]["topology_candidate_oracle"] for row in records
        )
        / samples,
        "predicted_topology_exact": sum(
            row["predicted"]["topology_exact"] for row in records
        )
        / samples,
        "known_count_chemical_exact": sum(
            row["known_count"]["chemical_exact"] for row in records
        )
        / samples,
        "predicted_chemical_exact": sum(
            row["predicted"]["chemical_exact"] for row in records
        )
        / samples,
        "predicted_recoverable_chemical_exact": sum(
            row["predicted"]["recoverable_chemical_exact"] for row in records
        )
        / samples,
        "chemical_candidate_oracle": sum(
            row["predicted"]["chemical_candidate_oracle"] for row in records
        )
        / samples,
        "predicted_display_exact": sum(
            row["predicted"]["display_exact"] for row in records
        )
        / samples,
        "compiled_sdf_exact": sum(row["compiled_sdf_exact"] for row in records)
        / samples,
        "recoverable_topology_precision": recoverable_precision,
        "topology_precision": strict_precision,
        "topology_recall": recall,
        "recoverable_topology_f1": (
            2.0
            * recoverable_precision
            * recall
            / max(recoverable_precision + recall, 1e-12)
        ),
        "topology_f1": (
            2.0 * strict_precision * recall / max(strict_precision + recall, 1e-12)
        ),
        "average_self_loops": sum(
            row["predicted"]["self_loops"] for row in records
        )
        / samples,
        "average_duplicate_pairs": sum(
            row["predicted"]["duplicate_pairs"] for row in records
        )
        / samples,
        "failure_stages": dict(sorted(Counter(row["failure_stage"] for row in records).items())),
    }
    if records and "selected_bond_expert" in records[0]:
        expert_routes = sum(
            row["selected_bond_expert"] == "style_expert" for row in records
        )
        summary["style_expert_routes"] = expert_routes
        summary["style_expert_route_rate"] = expert_routes / samples
    return summary


def summarize_slices(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        for name in row["slices"]:
            grouped[name].append(row)
    return {name: summarize(rows) for name, rows in sorted(grouped.items())}


def _load_detector_checkpoint(bond_checkpoint: Path, device: torch.device):
    payload = torch.load(bond_checkpoint, map_location="cpu", weights_only=False)
    config = DenseBondObjectConfig(**payload["config"])
    state = payload["dense_bond_object"]
    detector = (
        MultiScaleDenseBondObjectDetector(config).to(device)
        if any(key.startswith("base.") for key in state)
        else DenseBondObjectDetector(config).to(device)
    )
    detector.load_state_dict(state)
    detector.eval().requires_grad_(False)
    return detector, payload


def _load_models(
    bond_checkpoint: Path, device: torch.device, grounded_graph_checkpoint: Path | None
):
    detector, payload = _load_detector_checkpoint(bond_checkpoint, device)
    graph_path = grounded_graph_checkpoint or Path(payload["grounded_graph_checkpoint"])
    _, graph_payload = load_grounded_graph_checkpoint(graph_path, device)
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    model.eval().requires_grad_(False)
    return model, detector, graph_path


def select_routed_object_batches(
    base_batches: list[list[dict]],
    expert_batches: list[list[dict]],
    expert_probabilities: torch.Tensor,
    threshold: float,
) -> tuple[list[list[dict]], list[bool]]:
    """Choose one complete BondObject expert per image."""

    if len(base_batches) != len(expert_batches) or len(base_batches) != len(
        expert_probabilities
    ):
        raise ValueError("Routed BondObject batch sizes differ")
    routes = [float(value) >= threshold for value in expert_probabilities]
    selected = [
        expert if use_expert else base
        for base, expert, use_expert in zip(base_batches, expert_batches, routes)
    ]
    return selected, routes


def sample_keys_sha256(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["key"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument("--grounded-graph-checkpoint")
    parser.add_argument("--expert-bond-checkpoint")
    parser.add_argument("--style-gate-checkpoint")
    parser.add_argument("--style-gate-threshold", type=float, default=0.99)
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=[
            "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-*.tar"
        ],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/audits/chemvector-v0813-bond-first-decomposition-498",
    )
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--endpoint-radius-pixels", type=float, default=6.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, detector, graph_path = _load_models(
        Path(args.bond_checkpoint),
        device,
        Path(args.grounded_graph_checkpoint) if args.grounded_graph_checkpoint else None,
    )
    expert_detector = None
    style_gate = None
    if bool(args.expert_bond_checkpoint) != bool(args.style_gate_checkpoint):
        raise ValueError(
            "--expert-bond-checkpoint and --style-gate-checkpoint must be used together"
        )
    if args.expert_bond_checkpoint:
        if not 0.0 <= args.style_gate_threshold <= 1.0:
            raise ValueError("Style gate threshold must be in [0, 1]")
        expert_path = Path(args.expert_bond_checkpoint)
        expert_detector, expert_payload = _load_detector_checkpoint(expert_path, device)
        expert_graph_path = Path(expert_payload["grounded_graph_checkpoint"])
        if expert_graph_path.resolve() != graph_path.resolve():
            raise ValueError("Base and style expert use different grounded graph checkpoints")
        if type(expert_detector) is not type(detector):
            raise ValueError("Base and style expert detector types differ")
        gate_path = Path(args.style_gate_checkpoint)
        gate_payload = torch.load(gate_path, map_location="cpu", weights_only=False)
        if Path(gate_payload["base_bond_checkpoint"]).resolve() != Path(
            args.bond_checkpoint
        ).resolve():
            raise ValueError("Style gate base BondObject checkpoint does not match")
        if Path(gate_payload["expert_bond_checkpoint"]).resolve() != expert_path.resolve():
            raise ValueError("Style gate expert BondObject checkpoint does not match")
        style_gate = StructureStyleGate(
            gate_payload["input_channels"], hidden_dim=gate_payload["hidden_dim"]
        ).to(device)
        style_gate.load_state_dict(gate_payload["gate"])
        style_gate.eval().requires_grad_(False)
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
        raise ValueError(
            f"Requested {args.samples} validation samples, loaded {len(records)}"
        )
    collator = ChemSceneBondCollator()
    output_rows = []
    for start in range(0, len(records), args.batch_size):
        samples = records[start : start + args.batch_size]
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        route_probabilities = torch.zeros(len(samples))
        route_flags = [False] * len(samples)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if expert_detector is None:
                output = _forward_detector(model, detector, pixels)
                ranked_batches = decode_dense_bond_objects(
                    output, threshold=0.0, maximum_objects=128
                )
            else:
                features = _extract_detector_features(model, detector, pixels)
                base_batches = decode_dense_bond_objects(
                    _apply_detector(detector, features),
                    threshold=0.0,
                    maximum_objects=128,
                )
                expert_batches = decode_dense_bond_objects(
                    _apply_detector(expert_detector, features),
                    threshold=0.0,
                    maximum_objects=128,
                )
                route_probabilities = style_gate(features[0]).float().sigmoid().cpu()
                ranked_batches, route_flags = select_routed_object_batches(
                    base_batches,
                    expert_batches,
                    route_probabilities,
                    args.style_gate_threshold,
                )
        for batch_index, (sample, ranked) in enumerate(zip(samples, ranked_batches)):
            labels = structure_labels(sample)
            expected_count = len(labels["bonds"])
            known_objects = ranked[:expected_count]
            predicted_objects = [
                item for item in ranked if float(item["score"]) >= args.threshold
            ]
            known_metrics = analyze_objects(
                sample,
                known_objects,
                endpoint_radius_pixels=args.endpoint_radius_pixels,
            )
            predicted_metrics = analyze_objects(
                sample,
                predicted_objects,
                endpoint_radius_pixels=args.endpoint_radius_pixels,
            )
            compiled_sdf_exact = False
            predicted_scene = _scene_from_dense_objects(
                batch["scenes"][batch_index], predicted_objects
            )
            if predicted_scene is not None:
                try:
                    compiled_sdf_exact = (
                        Chem.MolToInchiKey(mol_from_chemscene_v0_1(predicted_scene))
                        == batch["expected_inchikeys"][batch_index]
                    )
                except Exception:
                    compiled_sdf_exact = False
            row = {
                    "key": sample["key"],
                    "variant": labels["variant"],
                    "stress_operation": labels["render_parameters"].get(
                        "stress_operation"
                    ),
                    "slices": sample_slices(sample),
                    "known_count": known_metrics,
                    "predicted": predicted_metrics,
                    "compiled_sdf_exact": compiled_sdf_exact,
                    "failure_stage": failure_stage(
                        predicted_metrics, compiled_sdf_exact
                    ),
                }
            if expert_detector is not None:
                row.update(
                    {
                        "style_expert_probability": float(
                            route_probabilities[batch_index]
                        ),
                        "selected_bond_expert": (
                            "style_expert" if route_flags[batch_index] else "base"
                        ),
                    }
                )
            output_rows.append(row)
        completed = min(start + args.batch_size, len(records))
        if completed % 50 == 0 or completed == len(records):
            current = summarize(output_rows)
            print(
                f"audited={completed}/{len(records)} "
                f"count={current['predicted_count_accuracy']:.4f} "
                f"topology={current['predicted_topology_exact']:.4f} "
                f"chemical={current['predicted_chemical_exact']:.4f} "
                f"sdf={current['compiled_sdf_exact']:.4f}",
                flush=True,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "bond-first-decomposition-0.1",
        "overall": summarize(output_rows),
        "slices": summarize_slices(output_rows),
        "protocol": {
            "bond_checkpoint": args.bond_checkpoint,
            "bond_checkpoint_sha256": sha256_file(Path(args.bond_checkpoint)),
            "grounded_graph_checkpoint": str(graph_path),
            "grounded_graph_checkpoint_sha256": sha256_file(graph_path),
            "validation_shards": args.validation_shards,
            "samples": args.samples,
            "sample_keys_sha256": sample_keys_sha256(records),
            "threshold": args.threshold,
            "endpoint_radius_pixels": args.endpoint_radius_pixels,
            "uses_oracle_junction_coordinates_for_topology_audit": True,
            "uses_mr_tadf_for_tuning": False,
            "style_moe": {
                "enabled": expert_detector is not None,
                "expert_bond_checkpoint": args.expert_bond_checkpoint,
                "expert_bond_checkpoint_sha256": (
                    sha256_file(Path(args.expert_bond_checkpoint))
                    if args.expert_bond_checkpoint
                    else None
                ),
                "style_gate_checkpoint": args.style_gate_checkpoint,
                "style_gate_checkpoint_sha256": (
                    sha256_file(Path(args.style_gate_checkpoint))
                    if args.style_gate_checkpoint
                    else None
                ),
                "style_gate_threshold": args.style_gate_threshold,
                "uses_uspto_for_gate_training_or_threshold_selection": False,
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
