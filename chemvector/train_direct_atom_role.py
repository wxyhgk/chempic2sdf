#!/usr/bin/env python3
"""Train a multi-scale raw-pixel AtomObject TEXT/IMPLICIT role gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.direct_atom_role import (
    DirectAtomRoleConfig,
    DirectAtomRoleGate,
    enumerate_selected_atoms,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    frozen_atom_objects,
    load_stratified_fresh_records,
    match_atom_targets,
)
from chemvector.train_joint_bond_set_multidomain import mixed_batches, verify_data_gate


DEFAULT_SOURCE = "outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--grounded-graph-checkpoint",
        default="outputs/checkpoints/chemvector-v0729-whole-set-stable-distance-overfit100/last.pt",
    )
    parser.add_argument("--style-train-samples", type=int, default=6_668)
    parser.add_argument("--chemvision-replay-samples", type=int, default=1_666)
    parser.add_argument("--font-v2-replay-samples", type=int, default=1_666)
    parser.add_argument("--fresh-calibration-samples", type=int, default=500)
    parser.add_argument("--fresh-eval-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every-total-samples", type=int, default=2_500)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def matched_role_rows(atoms, targets, mapping):
    atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
    mapped = mapping[sample_indices, atom_indices]
    valid = mapped >= 0
    atom_indices = atom_indices[valid]
    sample_indices = sample_indices[valid]
    mapped = mapped[valid]
    labels = targets["atom_roles"][sample_indices, mapped]
    return atom_indices, sample_indices, labels


def summarize(records: list[dict], threshold: float, probability_key: str) -> dict:
    totals = defaultdict(int)
    for sample in records:
        targets = sample["targets"].bool()
        predicted = sample[probability_key] >= threshold
        tp = predicted & targets
        totals["samples"] += 1
        totals["atoms"] += len(targets)
        totals["correct"] += int((predicted == targets).sum())
        totals["text_targets"] += int(targets.sum())
        totals["text_predicted"] += int(predicted.sum())
        totals["text_true_positive"] += int(tp.sum())
        representable = (
            sample["target_atoms"] == sample["matched_atoms"] == sample["selected_atoms"]
        )
        totals["representable"] += int(representable)
        totals["role_exact"] += int(representable and bool((predicted == targets).all()))
    precision = totals["text_true_positive"] / max(totals["text_predicted"], 1)
    recall = totals["text_true_positive"] / max(totals["text_targets"], 1)
    samples = max(totals["samples"], 1)
    return {
        **dict(totals),
        "threshold": threshold,
        "role_accuracy": totals["correct"] / max(totals["atoms"], 1),
        "text_precision": precision,
        "text_recall": recall,
        "text_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "representable_rate": totals["representable"] / samples,
        "role_exact_rate": totals["role_exact"] / samples,
    }


def select_threshold(records: list[dict]) -> dict:
    candidates = [summarize(records, value / 100.0, "probabilities") for value in range(5, 100, 5)]
    return max(
        candidates,
        key=lambda item: (
            item["role_exact_rate"],
            item["text_f1"],
            item["text_precision"],
            item["threshold"],
        ),
    )


@torch.no_grad()
def collect(model, vision, atom, bond, set_head, records, collator, config, atom_threshold, device, batch_size):
    model.eval()
    results = []
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        _, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, config, atom_threshold
        )
        mapping, _ = match_atom_targets(atoms, targets, config)
        atom_indices, sample_indices, labels = matched_role_rows(atoms, targets, mapping)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                pixels,
                atoms["atom_geometry"][..., :2],
                atom_indices,
                sample_indices,
            )
        probabilities = logits.sigmoid().float().cpu()
        baseline = atoms["atom_geometry"][
            sample_indices, atom_indices, 3
        ].float().cpu()
        labels = labels.cpu()
        for batch_index in range(len(pixels)):
            rows = torch.where(sample_indices == batch_index)[0].cpu()
            mapped_values = mapping[batch_index]
            results.append(
                {
                    "probabilities": probabilities[rows],
                    "baseline_probabilities": baseline[rows],
                    "targets": labels[rows],
                    "target_atoms": int(targets["atom_mask"][batch_index].sum()),
                    "selected_atoms": int(atoms["atom_mask"][batch_index].sum()),
                    "matched_atoms": int(mapped_values[mapped_values >= 0].unique().numel()),
                }
            )
    return results


def save(path, model, optimizer, args, config, seen, validation, audit):
    torch.save(
        {
            "checkpoint_schema_version": "direct-atom-role-0.1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "training_args": vars(args),
            "samples_seen": seen,
            "validation": validation,
            "audit": audit,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Direct Atom role training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_gate = verify_data_gate()
    vision, atom, bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=proposal_config.maximum_atoms,
            bond_queries=proposal_config.maximum_bonds,
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    fresh, fresh_audit = load_stratified_fresh_records(
        args.fresh_calibration_samples + args.fresh_eval_samples
    )
    calibration_records = fresh[: args.fresh_calibration_samples]
    evaluation_records = fresh[args.fresh_calibration_samples :]
    config = DirectAtomRoleConfig()
    model = DirectAtomRoleGate(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    audit = {"data_gate": data_gate, "fresh": fresh_audit, "sealed_test_read": False}
    domain_counts = {
        "style_train": args.style_train_samples,
        "chemvision_replay": args.chemvision_replay_samples,
        "font_v2_replay": args.font_v2_replay_samples,
    }
    scene_config = collator.config
    total_target = sum(domain_counts.values())
    seen = 0
    best_score = None
    started = time.perf_counter()
    for samples, domain_seen in mixed_batches(
        domain_sample_counts=domain_counts,
        domain_batch_counts={"style_train": 4, "chemvision_replay": 1, "font_v2_replay": 1},
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=1024,
    ):
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        _, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, proposal_config, atom_threshold
        )
        mapping, _ = match_atom_targets(atoms, targets, proposal_config)
        atom_indices, sample_indices, labels = matched_role_rows(atoms, targets, mapping)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices
            )
            positives = (labels == 1).sum().clamp_min(1)
            negatives = (labels == 0).sum().clamp_min(1)
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), labels.float(), pos_weight=(negatives / positives).clamp(1.0, 8.0)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        seen += len(samples)
        if seen % 1000 < len(samples):
            print(json.dumps({"samples_seen": seen, "loss": float(loss), "domain_seen": domain_seen}), flush=True)
        if seen % args.validate_every_total_samples < len(samples) or seen == total_target:
            predictions = collect(
                model, vision, atom, bond, set_head, calibration_records, collator,
                proposal_config, atom_threshold, device, args.batch_size
            )
            validation = select_threshold(predictions)
            validation.update({"samples_seen": seen, "elapsed_seconds": time.perf_counter() - started})
            score = (validation["role_exact_rate"], validation["text_f1"], validation["text_precision"])
            save(output / "last.pt", model, optimizer, args, config, seen, validation, audit)
            if best_score is None or score > best_score:
                best_score = score
                save(output / "best.pt", model, optimizer, args, config, seen, validation, audit)
            print(json.dumps({"validation": validation}), flush=True)
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    predictions = collect(
        model, vision, atom, bond, set_head, evaluation_records, collator,
        proposal_config, atom_threshold, device, args.batch_size
    )
    baseline = summarize(predictions, 0.5, "baseline_probabilities")
    candidate = summarize(predictions, payload["validation"]["threshold"], "probabilities")
    gate = {
        "role_exact_delta": candidate["role_exact_rate"] - baseline["role_exact_rate"],
        "passed": candidate["role_exact_rate"] - baseline["role_exact_rate"] >= 0.10
        and candidate["text_precision"] >= 0.95 and candidate["text_recall"] >= 0.80,
    }
    report = {"baseline": baseline, "candidate": candidate, "gate": gate, "sealed_test_read": False}
    (output / "fresh_eval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["fresh_evaluation"] = report
    torch.save(payload, output / "selected.pt")
    print(json.dumps({"final": report}), flush=True)


if __name__ == "__main__":
    main()
