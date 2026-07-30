#!/usr/bin/env python3
"""Audit graph-aware suppression of duplicate predicted AtomObjects."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.infer_direct_atom_pair_bond import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    frozen_atom_objects,
    load_stratified_fresh_records,
)
from chemvector.train_joint_bond_set_multidomain import verify_data_gate


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
)
DEFAULT_OUTPUT = "outputs/audits/chemvector-v01011-duplicate-atom-suppression"


def match_nodes(
    anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    kept: torch.Tensor,
    *,
    radius_px: float = 6.0,
) -> dict[int, int]:
    selected = torch.where(kept)[0]
    if not len(selected) or not len(target_anchors):
        return {}
    cost = torch.cdist(anchors[selected].float(), target_anchors.float()) * 768.0
    rows, columns = linear_sum_assignment(cost.numpy())
    return {
        int(selected[row]): int(column)
        for row, column in zip(rows, columns)
        if float(cost[row, column]) <= radius_px
    }


def nearest_targets(
    anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    *,
    radius_px: float = 6.0,
) -> torch.Tensor:
    if not len(target_anchors):
        return torch.full((len(anchors),), -1, dtype=torch.long)
    distance = torch.cdist(anchors.float(), target_anchors.float()) * 768.0
    values, indices = distance.min(dim=1)
    return torch.where(values <= radius_px, indices, torch.full_like(indices, -1))


def predicted_adjacency(sample: dict, threshold: float) -> list[set[int]]:
    adjacency = [set() for _ in range(len(sample["anchors"]))]
    selected = sample["probabilities"] >= threshold
    for pair in sample["pairs"][selected].tolist():
        begin, end = (int(value) for value in pair)
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    return adjacency


def graph_duplicate_candidates(
    sample: dict,
    *,
    presence_threshold: float,
    distance_px: float,
    minimum_degree: int,
) -> list[tuple[float, int, int]]:
    anchors = sample["anchors"]
    adjacency = predicted_adjacency(sample, presence_threshold)
    candidates = []
    for left in range(len(anchors)):
        for right in range(left + 1, len(anchors)):
            distance = float(torch.linalg.vector_norm(anchors[left] - anchors[right]) * 768.0)
            if distance > distance_px or right in adjacency[left]:
                continue
            left_neighbors = adjacency[left] - {right}
            right_neighbors = adjacency[right] - {left}
            if (
                len(left_neighbors) >= minimum_degree
                and left_neighbors == right_neighbors
            ):
                candidates.append((distance, left, right))
    return sorted(candidates)


def suppress_graph_duplicates(
    sample: dict,
    *,
    presence_threshold: float,
    distance_px: float,
    minimum_degree: int,
) -> tuple[torch.Tensor, list[dict]]:
    kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
    changes = []
    for distance, left, right in graph_duplicate_candidates(
        sample,
        presence_threshold=presence_threshold,
        distance_px=distance_px,
        minimum_degree=minimum_degree,
    ):
        if not kept[left] or not kept[right]:
            continue
        left_score = float(sample["scores"][left])
        right_score = float(sample["scores"][right])
        removed = right if left_score >= right_score else left
        retained = left if removed == right else right
        kept[removed] = False
        changes.append(
            {
                "distance_px": distance,
                "retained": retained,
                "removed": removed,
                "retained_score": float(sample["scores"][retained]),
                "removed_score": float(sample["scores"][removed]),
            }
        )
    return kept, changes


def suppress_oracle_duplicates(sample: dict) -> tuple[torch.Tensor, list[dict]]:
    kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
    nearest = nearest_targets(sample["anchors"], sample["target_anchors"])
    changes = []
    for target in nearest[nearest >= 0].unique(sorted=True).tolist():
        candidates = torch.where(nearest == target)[0]
        if len(candidates) < 2:
            continue
        scores = sample["scores"][candidates]
        retained = int(candidates[scores.argmax()])
        for candidate in candidates.tolist():
            if candidate == retained:
                continue
            kept[candidate] = False
            changes.append(
                {"target": int(target), "retained": retained, "removed": int(candidate)}
            )
    return kept, changes


def evaluate_sample(sample: dict, kept: torch.Tensor, threshold: float) -> dict:
    mapping = match_nodes(sample["anchors"], sample["target_anchors"], kept)
    target_bonds = sample["target_bonds"]
    predicted_bonds = set()
    invalid_edges = 0
    selected = sample["probabilities"] >= threshold
    for pair in sample["pairs"][selected].tolist():
        begin, end = (int(value) for value in pair)
        if not kept[begin] or not kept[end]:
            continue
        if begin not in mapping or end not in mapping or mapping[begin] == mapping[end]:
            invalid_edges += 1
            continue
        predicted_bonds.add(tuple(sorted((mapping[begin], mapping[end]))))
    matched_targets = set(mapping.values())
    atom_exact = len(mapping) == int(kept.sum()) == len(sample["target_anchors"])
    skeleton_exact = atom_exact and not invalid_edges and predicted_bonds == target_bonds
    return {
        "selected_atoms": int(kept.sum()),
        "matched_atoms": len(matched_targets),
        "target_atoms": len(sample["target_anchors"]),
        "atom_fp": int(kept.sum()) - len(mapping),
        "atom_fn": len(sample["target_anchors"]) - len(matched_targets),
        "predicted_bonds": len(predicted_bonds) + invalid_edges,
        "target_bonds": len(target_bonds),
        "bond_fp": len(predicted_bonds - target_bonds) + invalid_edges,
        "bond_fn": len(target_bonds - predicted_bonds),
        "atom_exact": atom_exact,
        "skeleton_exact": skeleton_exact,
    }


def summarize_variant(
    samples: list[dict],
    *,
    threshold: float,
    mode: str,
    distance_px: float | None = None,
    minimum_degree: int | None = None,
) -> dict:
    totals = defaultdict(int)
    cases = []
    for sample in samples:
        baseline_kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
        baseline = evaluate_sample(sample, baseline_kept, threshold)
        if mode == "baseline":
            kept, changes = baseline_kept, []
        elif mode == "oracle":
            kept, changes = suppress_oracle_duplicates(sample)
        elif mode == "graph":
            kept, changes = suppress_graph_duplicates(
                sample,
                presence_threshold=threshold,
                distance_px=float(distance_px),
                minimum_degree=int(minimum_degree),
            )
        else:
            raise ValueError(f"Unknown duplicate suppression mode: {mode}")
        candidate = evaluate_sample(sample, kept, threshold)
        totals["samples"] += 1
        totals["changes"] += len(changes)
        for name in ("atom_fp", "atom_fn", "bond_fp", "bond_fn"):
            totals[name] += candidate[name]
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
        if changes:
            nearest = nearest_targets(sample["anchors"], sample["target_anchors"])
            protected = 0
            true_duplicates = 0
            for change in changes:
                retained = change["retained"]
                removed = change["removed"]
                if nearest[retained] >= 0 and nearest[retained] == nearest[removed]:
                    true_duplicates += 1
                elif nearest[retained] >= 0 and nearest[removed] >= 0:
                    protected += 1
            totals["true_duplicate_changes"] += true_duplicates
            totals["protected_close_merges"] += protected
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
    count = max(totals["samples"], 1)
    return {
        **dict(totals),
        "mode": mode,
        "distance_px": distance_px,
        "minimum_degree": minimum_degree,
        "atom_exact_rate": totals["atom_exact"] / count,
        "skeleton_exact_rate": totals["skeleton_exact"] / count,
        "cases": cases,
    }


@torch.no_grad()
def collect_samples(
    direct,
    vision,
    atom,
    bond,
    set_head,
    records,
    collator,
    config,
    atom_threshold,
    device,
    batch_size,
) -> list[dict]:
    samples = []
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        batch = collator(rows)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        features, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, config, atom_threshold
        )
        pairs, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = direct(
                features,
                atoms["atom_features"],
                atoms["atom_geometry"][..., :2],
                atoms["atom_scores"],
                pairs,
                sample_indices,
            )
        probabilities = output["presence_logits"].sigmoid().float().cpu()
        chemical = output["chemical_logits"].float().argmax(-1).cpu()
        for batch_index, record in enumerate(rows):
            proposal_indices = torch.where(atoms["atom_mask"][batch_index])[0]
            remap = {int(value): index for index, value in enumerate(proposal_indices.tolist())}
            pair_rows = torch.where(sample_indices == batch_index)[0]
            local_pairs = torch.tensor(
                [
                    [remap[int(left)], remap[int(right)]]
                    for left, right in pairs[pair_rows].tolist()
                ],
                dtype=torch.long,
            )
            atom_count = int(targets["atom_mask"][batch_index].sum())
            bond_count = int(targets["bond_mask"][batch_index].sum())
            target_bonds = {
                tuple(sorted(int(value) for value in pair))
                for pair in targets["bond_atom_refs"][batch_index, :bond_count].cpu().tolist()
            }
            samples.append(
                {
                    "key": record["key"],
                    "anchors": atoms["atom_geometry"][
                        batch_index, proposal_indices, :2
                    ].float().cpu(),
                    "scores": atoms["atom_scores"][batch_index, proposal_indices].float().cpu(),
                    "pairs": local_pairs,
                    "probabilities": probabilities[pair_rows.cpu()],
                    "chemical": chemical[pair_rows.cpu()],
                    "target_anchors": targets["atom_anchors"][
                        batch_index, :atom_count
                    ].float().cpu(),
                    "target_bonds": target_bonds,
                }
            )
    return samples


def select_graph_variant(rows: list[dict]) -> dict:
    eligible = [
        row
        for row in rows
        if row["protected_close_merges"] == 0
        and row["atom_break"] == 0
        and row["skeleton_break"] == 0
    ]
    if not eligible:
        return min(rows, key=lambda row: (row["distance_px"], -row["minimum_degree"]))
    return max(
        eligible,
        key=lambda row: (
            row["skeleton_exact_rate"],
            row["atom_exact_rate"],
            row["true_duplicate_changes"],
            -row["distance_px"],
            row["minimum_degree"],
        ),
    )


def compact_metrics(metrics: dict) -> dict:
    return {name: value for name, value in metrics.items() if name != "cases"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--evaluation-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Duplicate AtomObject audit requires CUDA")
    torch.manual_seed(20260730)
    torch.cuda.manual_seed_all(20260730)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda")
    direct, direct_payload = load_direct_model(args.checkpoint, device)
    training_args = direct_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        training_args["source_checkpoint"], training_args["grounded_graph_checkpoint"], device
    )
    for module in (direct, vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    records, fresh_audit = load_stratified_fresh_records(
        args.calibration_samples + args.evaluation_samples
    )
    calibration = collect_samples(
        direct, vision, atom, bond, set_head, records[: args.calibration_samples],
        collator, config, atom_threshold, device, args.batch_size,
    )
    baseline = summarize_variant(calibration, threshold=presence_threshold, mode="baseline")
    oracle = summarize_variant(calibration, threshold=presence_threshold, mode="oracle")
    grid = [
        summarize_variant(
            calibration,
            threshold=presence_threshold,
            mode="graph",
            distance_px=distance,
            minimum_degree=degree,
        )
        for distance in (1.0, 2.0, 3.0, 4.0)
        for degree in (1, 2, 3)
    ]
    selected = select_graph_variant(grid)
    calibration_report = {
        "checkpoint": args.checkpoint,
        "presence_threshold": presence_threshold,
        "data_gate": verify_data_gate(),
        "fresh_audit": fresh_audit,
        "baseline": baseline,
        "oracle": oracle,
        "selected": selected,
        "grid": grid,
        "oracle_gate": {
            "skeleton_exact_delta": oracle["skeleton_exact_rate"] - baseline["skeleton_exact_rate"],
            "passed": oracle["skeleton_exact_rate"] - baseline["skeleton_exact_rate"] >= 0.01,
        },
        "sealed_test_read": False,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration": {
                    "baseline": compact_metrics(baseline),
                    "oracle": compact_metrics(oracle),
                    "selected": compact_metrics(selected),
                }
            }
        ),
        flush=True,
    )

    evaluation = collect_samples(
        direct, vision, atom, bond, set_head, records[args.calibration_samples :],
        collator, config, atom_threshold, device, args.batch_size,
    )
    evaluation_baseline = summarize_variant(
        evaluation, threshold=presence_threshold, mode="baseline"
    )
    evaluation_oracle = summarize_variant(
        evaluation, threshold=presence_threshold, mode="oracle"
    )
    evaluation_candidate = summarize_variant(
        evaluation,
        threshold=presence_threshold,
        mode="graph",
        distance_px=selected["distance_px"],
        minimum_degree=selected["minimum_degree"],
    )
    report = {
        "checkpoint": args.checkpoint,
        "presence_threshold": presence_threshold,
        "selected_rule": {
            "distance_px": selected["distance_px"],
            "minimum_degree": selected["minimum_degree"],
        },
        "baseline": evaluation_baseline,
        "oracle": evaluation_oracle,
        "candidate": evaluation_candidate,
        "gate": {
            "oracle_skeleton_delta": (
                evaluation_oracle["skeleton_exact_rate"]
                - evaluation_baseline["skeleton_exact_rate"]
            ),
            "candidate_skeleton_delta": (
                evaluation_candidate["skeleton_exact_rate"]
                - evaluation_baseline["skeleton_exact_rate"]
            ),
            "passed": (
                evaluation_candidate["skeleton_exact_rate"]
                - evaluation_baseline["skeleton_exact_rate"] >= 0.01
                and evaluation_candidate["skeleton_break"] == 0
                and evaluation_candidate["atom_break"] == 0
                and evaluation_candidate["protected_close_merges"] == 0
            ),
        },
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "evaluation": {
                    **report,
                    "baseline": compact_metrics(evaluation_baseline),
                    "oracle": compact_metrics(evaluation_oracle),
                    "candidate": compact_metrics(evaluation_candidate),
                }
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
