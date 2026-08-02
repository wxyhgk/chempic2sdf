#!/usr/bin/env python3
"""Evaluate one frozen duplicate-affinity winner on route-confirm only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_duplicate_atom_suppression import evaluate_sample
from chemvector.duplicate_atom_affinity import (
    DuplicateAtomAffinity,
    DuplicateAtomAffinityConfig,
)
from chemvector.infer_direct_atom_pair_bond import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import load_stratified_fresh_records
from chemvector.train_duplicate_atom_affinity import collect, compact, summarize
from chemvector.train_joint_bond_set_multidomain import verify_data_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-select-samples", type=int, default=4_000)
    parser.add_argument("--route-confirm-samples", type=int, default=3_974)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-selected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Duplicate affinity evaluation requires CUDA")
    torch.manual_seed(20260731)
    torch.cuda.manual_seed_all(20260731)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "duplicate-atom-affinity-0.1":
        raise ValueError("Duplicate affinity checkpoint schema differs")
    model = DuplicateAtomAffinity(DuplicateAtomAffinityConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)
    direct_path = payload["training_args"]["direct_checkpoint"]
    direct, direct_payload = load_direct_model(direct_path, device)
    direct_training = direct_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        direct_training["source_checkpoint"],
        direct_training["grounded_graph_checkpoint"],
        device,
    )
    for module in (direct, vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds,
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    total = args.route_select_samples + args.route_confirm_samples
    records, fresh_audit = load_stratified_fresh_records(total)
    confirm = records[args.route_select_samples :]
    samples = collect(
        model, direct, vision, atom, bond, set_head, confirm, collator, config,
        atom_threshold, presence_threshold, device, args.batch_size,
    )
    threshold = float(payload["validation"]["threshold"])
    evaluation = summarize(samples, threshold, presence_threshold)
    baseline_skeleton = sum(
        int(
            evaluate_sample(
                sample,
                torch.ones(len(sample["anchors"]), dtype=torch.bool),
                presence_threshold,
            )["skeleton_exact"]
        )
        for sample in samples
    ) / max(len(samples), 1)
    gate = {
        "baseline_skeleton_exact_rate": baseline_skeleton,
        "skeleton_exact_delta": evaluation["skeleton_exact_rate"] - baseline_skeleton,
        "passed": evaluation["skeleton_exact_rate"] - baseline_skeleton >= 0.005
        and evaluation["skeleton_break"] == 0
        and evaluation["atom_break"] == 0
        and evaluation["protected_close_merges"] == 0
        and evaluation["pair_precision"] >= 0.999,
    }
    report = {
        "checkpoint": args.checkpoint,
        "selected_step": payload["samples_seen"],
        "threshold": threshold,
        "fresh_audit": fresh_audit,
        "data_gate": verify_data_gate(),
        "route_select_read": False,
        "route_confirm_samples": len(confirm),
        "evaluation": evaluation,
        "gate": gate,
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.export_selected:
        payload["fresh_evaluation"] = report
        torch.save(payload, args.export_selected)
    print(
        json.dumps(
            {"evaluation": {**report, "evaluation": compact(evaluation)}},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
