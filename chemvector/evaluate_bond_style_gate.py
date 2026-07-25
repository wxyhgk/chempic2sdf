#!/usr/bin/env python3
"""Audit a frozen BondObject style gate on an external image manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from chemvector.audit_bond_first_decomposition import _load_models
from chemvector.evaluate_structure_gate import manifest_scores, summarize_scores
from chemvector.experiment_loop.artifacts import sha256_file
from chemvector.structure_gate import StructureStyleGate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prepare-uspto", action="store_true")
    parser.add_argument("--route-threshold", type=float, default=0.90)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for style-gate evaluation")
    if min(args.samples, args.batch_size) < 1:
        raise ValueError("Evaluation sizes must be positive")

    gate_path = Path(args.gate_checkpoint)
    manifest_path = Path(args.manifest)
    payload = torch.load(gate_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model, _, _ = _load_models(
        Path(payload["base_bond_checkpoint"]), device, None
    )
    gate = StructureStyleGate(
        payload["input_channels"], hidden_dim=payload["hidden_dim"]
    ).to(device)
    gate.load_state_dict(payload["gate"])
    gate.eval().requires_grad_(False)
    scores = manifest_scores(
        model.vision_encoder,
        gate,
        manifest_path,
        device,
        samples=args.samples,
        batch_size=args.batch_size,
        prepare_uspto=args.prepare_uspto,
    )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "status" in rows[0]:
        rows = [row for row in rows if row.get("status") == "ready"]
    rows = rows[: len(scores)]
    score_records = [
        {
            "molecule_id": row.get("molecule_id") or row.get("key"),
            "image_file": row["image_file"],
            "style_expert_probability": float(score),
            "selected_bond_expert": (
                "style_expert" if float(score) >= args.route_threshold else "base"
            ),
        }
        for row, score in zip(rows, scores)
    ]
    result = {
        "schema_version": "chemvector-bond-style-gate-audit-0.1",
        "gate_checkpoint": str(gate_path),
        "gate_checkpoint_sha256": sha256_file(gate_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "prepare_uspto": args.prepare_uspto,
        "uses_manifest_labels": False,
        "uses_manifest_for_threshold_selection": False,
        "route_threshold": args.route_threshold,
        "scores": summarize_scores(
            scores, thresholds=(0.5, 0.75, 0.9, 0.95, 0.99)
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    records_path = output_path.with_suffix(".records.jsonl")
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_records),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
