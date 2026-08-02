#!/usr/bin/env python3
"""Build internal teacher data for BondObject expert-utility routing."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

import torch

from chemvector.audit_bond_first_decomposition import (
    _load_detector_checkpoint,
    _load_models,
    analyze_objects,
)
from chemvector.bond_expert_utility import combined_utility_features
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import decode_dense_bond_objects
from chemvector.pretrain_chemscene_bond_query import ChemSceneBondCollator
from chemvector.pretrain_dense_bond_object import (
    _apply_detector,
    _extract_detector_features,
)
from chemvector.structure_gate import StructureStyleGate
from chemvector.train import expand_shards


def topology_f1(metrics: dict) -> float:
    true_positive = float(metrics["topology_true_positives"])
    false_positive = float(metrics["strict_topology_false_positives"])
    false_negative = float(metrics["topology_false_negatives"])
    return 2.0 * true_positive / max(2.0 * true_positive + false_positive + false_negative, 1.0)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bond-checkpoint", required=True)
    parser.add_argument("--expert-bond-checkpoint", required=True)
    parser.add_argument("--style-gate-checkpoint")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--object-threshold", type=float, default=0.25)
    parser.add_argument("--endpoint-radius-pixels", type=float, default=12.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.samples, args.batch_size) < 1 or args.skip_samples < 0:
        raise ValueError("Utility data sizes must be positive")

    device = torch.device(args.device)
    model, base_detector, graph_path = _load_models(
        Path(args.base_bond_checkpoint), device, None
    )
    expert_detector, expert_payload = _load_detector_checkpoint(
        Path(args.expert_bond_checkpoint), device
    )
    if Path(expert_payload["grounded_graph_checkpoint"]).resolve() != graph_path.resolve():
        raise ValueError("Bond experts use different grounded vision checkpoints")
    gate = None
    if args.style_gate_checkpoint:
        payload = torch.load(
            args.style_gate_checkpoint, map_location="cpu", weights_only=False
        )
        gate = StructureStyleGate(
            payload["input_channels"], hidden_dim=payload["hidden_dim"]
        ).to(device)
        gate.load_state_dict(payload["gate"])
        gate.eval().requires_grad_(False)

    records = list(
        islice(
            iter(
                ChemVectorIterableDataset(
                    expand_shards(args.shards), sequence_version="0.4"
                )
            ),
            args.skip_samples,
            args.skip_samples + args.samples,
        )
    )
    if len(records) != args.samples:
        raise ValueError("Utility source contains fewer samples than requested")
    collator = ChemSceneBondCollator()
    output_rows = []
    for start in range(0, len(records), args.batch_size):
        samples = records[start : start + args.batch_size]
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            detector_features = _extract_detector_features(
                model, base_detector, pixels
            )
            base_batches = decode_dense_bond_objects(
                _apply_detector(base_detector, detector_features),
                threshold=0.0,
                maximum_objects=128,
            )
            expert_batches = decode_dense_bond_objects(
                _apply_detector(expert_detector, detector_features),
                threshold=0.0,
                maximum_objects=128,
            )
            style_probabilities = (
                gate(detector_features[0]).float().sigmoid().cpu()
                if gate is not None
                else torch.zeros(len(samples))
            )
        for sample, base_ranked, expert_ranked, style_probability in zip(
            samples, base_batches, expert_batches, style_probabilities
        ):
            base_selected = [
                item
                for item in base_ranked
                if float(item["score"]) >= args.object_threshold
            ]
            expert_selected = [
                item
                for item in expert_ranked
                if float(item["score"]) >= args.object_threshold
            ]
            base_metrics = analyze_objects(sample, base_selected)
            expert_metrics = analyze_objects(sample, expert_selected)
            base_exact = bool(base_metrics["topology_exact"])
            expert_exact = bool(expert_metrics["topology_exact"])
            utility = (
                "expert_rescue"
                if expert_exact and not base_exact
                else "expert_break"
                if base_exact and not expert_exact
                else "both_exact"
                if base_exact
                else "both_wrong"
            )
            output_rows.append(
                {
                    "key": sample["key"],
                    "domain": args.domain,
                    "features": combined_utility_features(
                        base_ranked,
                        expert_ranked,
                        style_probability=float(style_probability),
                        object_threshold=args.object_threshold,
                        radius_pixels=args.endpoint_radius_pixels,
                    ),
                    "label": {
                        "topology_utility": utility,
                        "base_topology_exact": base_exact,
                        "expert_topology_exact": expert_exact,
                        "base_topology_f1": topology_f1(base_metrics),
                        "expert_topology_f1": topology_f1(expert_metrics),
                    },
                }
            )
        completed = min(start + args.batch_size, len(records))
        if completed % 100 == 0 or completed == len(records):
            counts = {}
            for row in output_rows:
                label = row["label"]["topology_utility"]
                counts[label] = counts.get(label, 0) + 1
            print(f"built={completed}/{len(records)} labels={counts}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
