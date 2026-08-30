#!/usr/bin/env python3
"""Calibrate graph-level aromatic semantics for v0.10.3 Direct AtomPair Bond."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings
from chemvector.direct_scene_inference import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    collect_predictions,
    load_stratified_fresh_records,
    summarize_predictions,
)
from chemvector.train_joint_bond_set_multidomain import verify_data_gate


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
)
DEFAULT_OUTPUT = "outputs/evaluations/chemvector-v0104-direct-aromatic-calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--evaluation-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ring_variant(
    predictions: list[dict],
    *,
    presence_threshold: float,
    mean_threshold: float,
    minimum_threshold: float,
) -> tuple[dict, dict]:
    updated = []
    totals = {
        "candidate_cycles": 0,
        "accepted_cycles": 0,
        "promoted_bonds": 0,
        "changed_bond_types": 0,
        "changed_samples": 0,
    }
    for sample in predictions:
        chemical, diagnostics = promote_complete_aromatic_rings(
            sample["pair_indices"],
            sample["probabilities"],
            sample["chemical_logits"],
            presence_threshold=presence_threshold,
            mean_threshold=mean_threshold,
            minimum_threshold=minimum_threshold,
        )
        row = {**sample, "chemical": chemical}
        updated.append(row)
        for name in (
            "candidate_cycles",
            "accepted_cycles",
            "promoted_bonds",
            "changed_bond_types",
        ):
            totals[name] += int(diagnostics[name])
        totals["changed_samples"] += int(diagnostics["changed_bond_types"] > 0)
    return summarize_predictions(updated, presence_threshold), totals


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Direct aromatic calibration requires CUDA")
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data_gate = verify_data_gate()
    device = torch.device("cuda")
    direct, direct_payload = load_direct_model(args.checkpoint, device)
    training_args = direct_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        training_args["source_checkpoint"],
        training_args["grounded_graph_checkpoint"],
        device,
    )
    for module in (vision, atom, bond, set_head, direct):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    records, fresh_audit = load_stratified_fresh_records(
        args.calibration_samples + args.evaluation_samples
    )
    calibration_records = records[: args.calibration_samples]
    evaluation_records = records[args.calibration_samples :]
    calibration_predictions = collect_predictions(
        direct,
        vision,
        atom,
        bond,
        set_head,
        calibration_records,
        collator,
        proposal_config,
        atom_threshold,
        device,
        args.batch_size,
    )
    baseline_calibration = summarize_predictions(
        calibration_predictions, presence_threshold
    )
    grid = []
    for mean_threshold in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        for minimum_threshold in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
            if minimum_threshold > mean_threshold:
                continue
            metrics, changes = ring_variant(
                calibration_predictions,
                presence_threshold=presence_threshold,
                mean_threshold=mean_threshold,
                minimum_threshold=minimum_threshold,
            )
            grid.append(
                {
                    "mean_threshold": mean_threshold,
                    "minimum_threshold": minimum_threshold,
                    "metrics": metrics,
                    "changes": changes,
                }
            )
    selected = max(
        grid,
        key=lambda row: (
            row["metrics"]["chemical_bond_exact_rate"],
            row["metrics"]["full_chemical_exact_rate"],
            row["metrics"]["chemical_accuracy"],
            -row["changes"]["changed_bond_types"],
        ),
    )
    calibration_report = {
        "checkpoint": args.checkpoint,
        "presence_threshold": presence_threshold,
        "fresh_audit": fresh_audit,
        "data_gate": data_gate,
        "baseline": baseline_calibration,
        "selected": selected,
        "grid": grid,
        "sealed_test_read": False,
    }
    (output / "calibration.json").write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"calibration": calibration_report["selected"]}), flush=True)

    evaluation_predictions = collect_predictions(
        direct,
        vision,
        atom,
        bond,
        set_head,
        evaluation_records,
        collator,
        proposal_config,
        atom_threshold,
        device,
        args.batch_size,
    )
    baseline_evaluation = summarize_predictions(
        evaluation_predictions, presence_threshold
    )
    candidate_evaluation, changes = ring_variant(
        evaluation_predictions,
        presence_threshold=presence_threshold,
        mean_threshold=selected["mean_threshold"],
        minimum_threshold=selected["minimum_threshold"],
    )
    exact_delta = (
        candidate_evaluation["chemical_bond_exact_rate"]
        - baseline_evaluation["chemical_bond_exact_rate"]
    )
    accuracy_delta = (
        candidate_evaluation["chemical_accuracy"]
        - baseline_evaluation["chemical_accuracy"]
    )
    report = {
        "checkpoint": args.checkpoint,
        "presence_threshold": presence_threshold,
        "mean_threshold": selected["mean_threshold"],
        "minimum_threshold": selected["minimum_threshold"],
        "baseline": baseline_evaluation,
        "candidate": candidate_evaluation,
        "changes": changes,
        "gate": {
            "chemical_bond_exact_delta": exact_delta,
            "chemical_accuracy_delta": accuracy_delta,
            "passed": exact_delta >= 0.02 and accuracy_delta >= -0.002,
        },
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"evaluation": report}), flush=True)


if __name__ == "__main__":
    main()
