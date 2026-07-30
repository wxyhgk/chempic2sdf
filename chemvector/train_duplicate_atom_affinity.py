#!/usr/bin/env python3
"""Train v0.10.11 graph-aware duplicate AtomObject affinity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_duplicate_atom_suppression import (
    evaluate_sample,
    nearest_targets,
)
from chemvector.duplicate_atom_affinity import (
    DuplicateAtomAffinity,
    DuplicateAtomAffinityConfig,
    duplicate_atom_affinity_loss,
    duplicate_pair_context,
    enumerate_close_atom_pairs,
)
from chemvector.infer_direct_atom_pair_bond import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    frozen_atom_objects,
    load_stratified_fresh_records,
)
from chemvector.train_joint_bond_set_multidomain import mixed_batches, verify_data_gate
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs


DEFAULT_DIRECT = (
    "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
)
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v01011-duplicate-atom-affinity-style10k"
THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-checkpoint", default=DEFAULT_DIRECT)
    parser.add_argument("--style-train-samples", type=int, default=6_668)
    parser.add_argument("--chemvision-replay-samples", type=int, default=1_666)
    parser.add_argument("--font-v2-replay-samples", type=int, default=1_666)
    parser.add_argument("--fresh-calibration-samples", type=int, default=500)
    parser.add_argument("--fresh-eval-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every-total-samples", type=int, default=2_500)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def target_ids_for_proposals(atoms, targets, config) -> torch.Tensor:
    target_ids = torch.full_like(atoms["atom_indices"], -1, dtype=torch.long)
    for batch_index in range(targets["atom_mask"].size(0)):
        proposals = torch.where(atoms["atom_mask"][batch_index])[0]
        target_count = int(targets["atom_mask"][batch_index].sum())
        if not len(proposals) or not target_count:
            continue
        distance = torch.cdist(
            atoms["atom_geometry"][batch_index, proposals, :2].float(),
            targets["atom_anchors"][batch_index, :target_count].float(),
        ) * config.image_size
        values, nearest = distance.min(dim=1)
        target_ids[batch_index, proposals] = torch.where(
            values <= config.atom_match_radius_px,
            nearest,
            torch.full_like(nearest, -1),
        )
    return target_ids


def duplicate_targets(
    pairs: torch.Tensor, samples: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    left = target_ids[samples, pairs[:, 0]]
    right = target_ids[samples, pairs[:, 1]]
    return (left >= 0) & (left == right)


def suppress_learned(sample: dict, threshold: float) -> tuple[torch.Tensor, list[dict]]:
    kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
    ranked = sorted(
        zip(sample["duplicate_probabilities"].tolist(), sample["duplicate_pairs"].tolist()),
        reverse=True,
    )
    changes = []
    for probability, pair in ranked:
        if probability < threshold:
            break
        left, right = (int(value) for value in pair)
        if not kept[left] or not kept[right]:
            continue
        removed = right if float(sample["scores"][left]) >= float(sample["scores"][right]) else left
        retained = left if removed == right else right
        kept[removed] = False
        changes.append(
            {
                "retained": retained,
                "removed": removed,
                "probability": probability,
            }
        )
    return kept, changes


def summarize(samples: list[dict], threshold: float, presence_threshold: float) -> dict:
    totals = defaultdict(int)
    cases = []
    for sample in samples:
        baseline_kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
        baseline = evaluate_sample(sample, baseline_kept, presence_threshold)
        kept, changes = suppress_learned(sample, threshold)
        candidate = evaluate_sample(sample, kept, presence_threshold)
        targets = sample["duplicate_targets"]
        predicted = sample["duplicate_probabilities"] >= threshold
        totals["samples"] += 1
        totals["pair_targets"] += int(targets.sum())
        totals["pair_predicted"] += int(predicted.sum())
        totals["pair_true_positive"] += int((predicted & targets).sum())
        totals["changes"] += len(changes)
        totals["atom_exact"] += int(candidate["atom_exact"])
        totals["skeleton_exact"] += int(candidate["skeleton_exact"])
        totals["atom_rescue"] += int(candidate["atom_exact"] and not baseline["atom_exact"])
        totals["atom_break"] += int(baseline["atom_exact"] and not candidate["atom_exact"])
        totals["skeleton_rescue"] += int(
            candidate["skeleton_exact"] and not baseline["skeleton_exact"]
        )
        totals["skeleton_break"] += int(
            baseline["skeleton_exact"] and not candidate["skeleton_exact"]
        )
        nearest = nearest_targets(sample["anchors"], sample["target_anchors"])
        protected = 0
        true_duplicates = 0
        for change in changes:
            retained, removed = change["retained"], change["removed"]
            if nearest[retained] >= 0 and nearest[retained] == nearest[removed]:
                true_duplicates += 1
            elif nearest[retained] >= 0 and nearest[removed] >= 0:
                protected += 1
        totals["true_duplicate_changes"] += true_duplicates
        totals["protected_close_merges"] += protected
        if changes:
            cases.append(
                {
                    "key": sample["key"],
                    "changes": changes,
                    "true_duplicate_changes": true_duplicates,
                    "protected_close_merges": protected,
                    "baseline": baseline,
                    "candidate": candidate,
                }
            )
    samples_count = max(totals["samples"], 1)
    precision = totals["pair_true_positive"] / max(totals["pair_predicted"], 1)
    recall = totals["pair_true_positive"] / max(totals["pair_targets"], 1)
    return {
        **dict(totals),
        "threshold": threshold,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "atom_exact_rate": totals["atom_exact"] / samples_count,
        "skeleton_exact_rate": totals["skeleton_exact"] / samples_count,
        "cases": cases,
    }


def select_threshold(samples: list[dict], presence_threshold: float) -> tuple[dict, list[dict]]:
    grid = [summarize(samples, threshold, presence_threshold) for threshold in THRESHOLDS]
    eligible = [
        row
        for row in grid
        if row["protected_close_merges"] == 0
        and row["atom_break"] == 0
        and row["skeleton_break"] == 0
        and row["pair_precision"] >= 0.999
    ]
    selected = max(
        eligible or grid,
        key=lambda row: (
            row["skeleton_exact_rate"],
            row["atom_exact_rate"],
            row["pair_precision"],
            row["threshold"],
        ),
    )
    return selected, grid


def compact(metrics: dict) -> dict:
    return {name: value for name, value in metrics.items() if name != "cases"}


@torch.no_grad()
def collect(
    model, direct, vision, atom, bond, set_head, records, collator, config,
    atom_threshold, presence_threshold, device, batch_size,
) -> list[dict]:
    model.eval()
    samples = []
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        batch = collator(rows)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        features, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, config, atom_threshold
        )
        direct_pairs, direct_samples = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            direct_output = direct(
                features, atoms["atom_features"], atoms["atom_geometry"][..., :2],
                atoms["atom_scores"], direct_pairs, direct_samples,
            )
        presence = direct_output["presence_logits"].sigmoid().float()
        candidate_pairs, candidate_samples = enumerate_close_atom_pairs(
            atoms["atom_mask"], atoms["atom_geometry"][..., :2],
            maximum_distance_px=model.config.maximum_distance_px,
        )
        context = duplicate_pair_context(
            candidate_pairs, candidate_samples, atoms, direct_pairs, direct_samples, presence,
            presence_threshold=presence_threshold,
            maximum_distance_px=model.config.maximum_distance_px,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            duplicate_probabilities = model(
                atoms["atom_features"], atoms["atom_geometry"], candidate_pairs,
                candidate_samples, context,
            ).sigmoid().float().cpu()
        target_ids = target_ids_for_proposals(atoms, targets, config)
        candidate_targets = duplicate_targets(candidate_pairs, candidate_samples, target_ids).cpu()
        direct_probabilities = presence.cpu()
        direct_chemical = direct_output["chemical_logits"].float().argmax(-1).cpu()
        for batch_index, record in enumerate(rows):
            proposal_indices = torch.where(atoms["atom_mask"][batch_index])[0]
            remap = {int(value): index for index, value in enumerate(proposal_indices.tolist())}
            direct_rows = torch.where(direct_samples == batch_index)[0]
            candidate_rows = torch.where(candidate_samples == batch_index)[0]
            atom_count = int(targets["atom_mask"][batch_index].sum())
            bond_count = int(targets["bond_mask"][batch_index].sum())
            samples.append(
                {
                    "key": record["key"],
                    "anchors": atoms["atom_geometry"][batch_index, proposal_indices, :2].float().cpu(),
                    "scores": atoms["atom_scores"][batch_index, proposal_indices].float().cpu(),
                    "pairs": torch.tensor(
                        [[remap[int(left)], remap[int(right)]] for left, right in direct_pairs[direct_rows].tolist()],
                        dtype=torch.long,
                    ),
                    "probabilities": direct_probabilities[direct_rows.cpu()],
                    "chemical": direct_chemical[direct_rows.cpu()],
                    "target_anchors": targets["atom_anchors"][batch_index, :atom_count].float().cpu(),
                    "target_bonds": {
                        tuple(sorted(int(value) for value in pair))
                        for pair in targets["bond_atom_refs"][batch_index, :bond_count].cpu().tolist()
                    },
                    "duplicate_pairs": torch.tensor(
                        [[remap[int(left)], remap[int(right)]] for left, right in candidate_pairs[candidate_rows].tolist()],
                        dtype=torch.long,
                    ),
                    "duplicate_probabilities": duplicate_probabilities[candidate_rows.cpu()],
                    "duplicate_targets": candidate_targets[candidate_rows.cpu()],
                }
            )
    return samples


def save(path, model, optimizer, args, seen, validation, audit):
    torch.save(
        {
            "checkpoint_schema_version": "duplicate-atom-affinity-0.1",
            "model": model.state_dict(),
            "config": asdict(model.config),
            "optimizer": optimizer.state_dict(),
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
        raise RuntimeError("Duplicate Atom affinity training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    direct, direct_payload = load_direct_model(args.direct_checkpoint, device)
    training_args = direct_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        training_args["source_checkpoint"], training_args["grounded_graph_checkpoint"], device
    )
    for module in (direct, vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=proposal_config.maximum_atoms,
            bond_queries=proposal_config.maximum_bonds,
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    fresh, fresh_audit = load_stratified_fresh_records(
        args.fresh_calibration_samples + args.fresh_eval_samples
    )
    calibration_records = fresh[: args.fresh_calibration_samples]
    evaluation_records = fresh[args.fresh_calibration_samples :]
    model = DuplicateAtomAffinity(DuplicateAtomAffinityConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    audit = {
        "data_gate": verify_data_gate(),
        "fresh": fresh_audit,
        "direct_checkpoint": args.direct_checkpoint,
        "parameters": model.parameter_count(),
        "sealed_test_read": False,
    }
    domain_counts = {
        "style_train": args.style_train_samples,
        "chemvision_replay": args.chemvision_replay_samples,
        "font_v2_replay": args.font_v2_replay_samples,
    }
    total_target = sum(domain_counts.values())
    seen = positives_seen = negatives_seen = 0
    best_score = None
    for samples, domain_seen in mixed_batches(
        domain_sample_counts=domain_counts,
        domain_batch_counts={"style_train": 4, "chemvision_replay": 1, "font_v2_replay": 1},
        batch_size=args.batch_size,
        config=collator.config,
        seed=args.seed,
        shuffle_buffer=1024,
    ):
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        features, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, proposal_config, atom_threshold
        )
        direct_pairs, direct_samples = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            direct_output = direct(
                features, atoms["atom_features"], atoms["atom_geometry"][..., :2],
                atoms["atom_scores"], direct_pairs, direct_samples,
            )
        presence = direct_output["presence_logits"].sigmoid().float()
        candidate_pairs, candidate_samples = enumerate_close_atom_pairs(
            atoms["atom_mask"], atoms["atom_geometry"][..., :2],
            maximum_distance_px=model.config.maximum_distance_px,
        )
        if len(candidate_pairs):
            context = duplicate_pair_context(
                candidate_pairs, candidate_samples, atoms, direct_pairs, direct_samples, presence,
                presence_threshold=presence_threshold,
                maximum_distance_px=model.config.maximum_distance_px,
            )
            target_ids = target_ids_for_proposals(atoms, targets, proposal_config)
            labels = duplicate_targets(candidate_pairs, candidate_samples, target_ids)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    atoms["atom_features"], atoms["atom_geometry"], candidate_pairs,
                    candidate_samples, context,
                )
                loss = duplicate_atom_affinity_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            positives_seen += int(labels.sum())
            negatives_seen += int((~labels).sum())
        seen += len(samples)
        if seen % 1000 < len(samples):
            print(
                json.dumps(
                    {
                        "samples_seen": seen,
                        "positives_seen": positives_seen,
                        "negatives_seen": negatives_seen,
                        "loss": float(loss.detach()) if len(candidate_pairs) else None,
                        "domain_seen": domain_seen,
                    }
                ),
                flush=True,
            )
        if seen % args.validate_every_total_samples < len(samples) or seen == total_target:
            records = collect(
                model, direct, vision, atom, bond, set_head, calibration_records, collator,
                proposal_config, atom_threshold, presence_threshold, device, args.batch_size,
            )
            validation, grid = select_threshold(records, presence_threshold)
            validation.update(
                {
                    "samples_seen": seen,
                    "positives_seen": positives_seen,
                    "negatives_seen": negatives_seen,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            score = (
                validation["skeleton_exact_rate"],
                validation["atom_exact_rate"],
                validation["pair_precision"],
            )
            save(output_dir / "last.pt", model, optimizer, args, seen, validation, audit)
            if best_score is None or score > best_score:
                best_score = score
                save(output_dir / "best.pt", model, optimizer, args, seen, validation, audit)
                (output_dir / "best_calibration.json").write_text(
                    json.dumps(
                        {"selected": validation, "grid": grid}, ensure_ascii=False, indent=2
                    ) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps({"validation": compact(validation)}), flush=True)
    payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    calibration_gate = {
        "passed": payload["validation"]["skeleton_exact_rate"] >= 0.47
        and payload["validation"]["skeleton_break"] == 0
        and payload["validation"]["protected_close_merges"] == 0
        and payload["validation"]["pair_precision"] >= 0.999
    }
    if not calibration_gate["passed"]:
        report = {
            "calibration": payload["validation"],
            "calibration_gate": calibration_gate,
            "fresh_evaluation_read": False,
            "sealed_test_read": False,
        }
        (output_dir / "calibration_no_go.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        payload["fresh_evaluation"] = None
        torch.save(payload, output_dir / "selected.pt")
        print(json.dumps({"final": {**report, "calibration": compact(payload["validation"])}}), flush=True)
        return
    records = collect(
        model, direct, vision, atom, bond, set_head, evaluation_records, collator,
        proposal_config, atom_threshold, presence_threshold, device, args.batch_size,
    )
    evaluation = summarize(records, payload["validation"]["threshold"], presence_threshold)
    gate = {
        "passed": evaluation["skeleton_exact_rate"] >= 0.462
        and evaluation["skeleton_break"] == 0
        and evaluation["protected_close_merges"] == 0
        and evaluation["pair_precision"] >= 0.999
    }
    report = {"evaluation": evaluation, "gate": gate, "sealed_test_read": False}
    (output_dir / "fresh_eval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["fresh_evaluation"] = report
    torch.save(payload, output_dir / "selected.pt")
    print(json.dumps({"final": {"evaluation": compact(evaluation), "gate": gate}}), flush=True)


if __name__ == "__main__":
    main()
