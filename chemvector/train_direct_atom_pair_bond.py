#!/usr/bin/env python3
"""Train a direct all-pairs Bond classifier on frozen predicted AtomObjects."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import random
import subprocess
import time

from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemsvg_sequence_v4 import (
    ChemSVGCoordinateCollisionError,
    serialize_chemsvg_v4,
)
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    direct_atom_pair_loss,
    enumerate_unordered_atom_pairs,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_bond_set_multidomain import (
    FRESH_DEV_SHARDS,
    mixed_batches,
    verify_data_gate,
)
from chemvector.train_proposal_object_set import evaluate as evaluate_source
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head


DEFAULT_SOURCE = "outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-train-samples", type=int, default=6_668)
    parser.add_argument("--chemvision-replay-samples", type=int, default=1_666)
    parser.add_argument("--font-v2-replay-samples", type=int, default=1_666)
    parser.add_argument("--fresh-calibration-samples", type=int, default=500)
    parser.add_argument("--fresh-eval-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--negative-ratio", type=int, default=6)
    parser.add_argument("--minimum-negatives-per-image", type=int, default=48)
    parser.add_argument("--validate-every-total-samples", type=int, default=2_500)
    parser.add_argument("--log-every-total-samples", type=int, default=1_000)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def load_stratified_fresh_records(count: int) -> tuple[list[dict], dict]:
    """Keep the globally smallest key hashes without loading all images in memory."""

    heap: list[tuple[int, str, dict]] = []
    total = 0
    coordinate_collisions = 0
    for sample in ChemVectorIterableDataset(
        FRESH_DEV_SHARDS,
        sequence_version="0.4",
        serialize_target=False,
        load_raster_layers=False,
    ):
        total += 1
        try:
            serialize_chemsvg_v4(sample["chemmol"], sample["render"])
        except ChemSVGCoordinateCollisionError:
            coordinate_collisions += 1
            continue
        score = int.from_bytes(
            hashlib.sha256(sample["key"].encode("utf-8")).digest()[:8], "big"
        )
        item = (-score, sample["key"], sample)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    if total != 8_000 or coordinate_collisions != 26 or len(heap) != count:
        raise ValueError(
            "Fresh stratified coverage differs: "
            f"total={total} collisions={coordinate_collisions} selected={len(heap)}"
        )
    selected = [item[2] for item in sorted(heap, key=lambda item: -item[0])]
    return selected, {
        "method": "smallest_sha256_prefix",
        "fresh_total": total,
        "coordinate_collisions_excluded": coordinate_collisions,
        "selected": len(selected),
        "selected_keys_sha256": hashlib.sha256(
            "\n".join(sample["key"] for sample in selected).encode("utf-8")
        ).hexdigest(),
    }


def frozen_atom_objects(
    vision,
    atom,
    bond,
    set_head,
    pixels: torch.Tensor,
    proposal_config: PredictedProposalConfig,
    atom_decision_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom(features)
        bond_output = forward_frozen_bond_head(vision, bond, features)
        atoms = decode_predicted_atom_proposals(atom_output, proposal_config)
        bonds = decode_predicted_bond_proposals(bond_output, proposal_config)
        set_output = set_head(
            atoms["atom_features"],
            atoms["atom_geometry"],
            atoms["atom_scores"],
            atoms["atom_mask"],
            bonds["bond_features"],
            bonds["bond_endpoints"],
            bonds["bond_types"],
            bonds["bond_scores"],
            bonds["bond_mask"],
        )
    return features, {
        **atoms,
        "atom_mask": atoms["atom_mask"]
        & (set_output["atom_presence_logits"] >= atom_decision_threshold),
    }


def match_atom_targets(
    atoms: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_config: PredictedProposalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = torch.full_like(atoms["atom_indices"], -1, dtype=torch.long)
    matched = torch.zeros_like(targets["atom_mask"])
    for batch_index in range(targets["atom_mask"].size(0)):
        proposal_indices = torch.where(atoms["atom_mask"][batch_index])[0]
        target_count = int(targets["atom_mask"][batch_index].sum())
        if not len(proposal_indices) or not target_count:
            continue
        predicted = atoms["atom_geometry"][batch_index, proposal_indices, :2]
        expected = targets["atom_anchors"][batch_index, :target_count]
        cost = torch.cdist(predicted.float(), expected.float()) * proposal_config.image_size
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        for row, column in zip(rows, columns):
            if float(cost[row, column]) <= proposal_config.atom_match_radius_px:
                proposal_index = proposal_indices[row]
                mapping[batch_index, proposal_index] = int(column)
                matched[batch_index, column] = True
    return mapping, matched


def build_pair_targets(
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    atom_mapping: torch.Tensor,
    targets: dict[str, torch.Tensor],
    atoms: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    presence = torch.zeros(len(pair_indices), device=pair_indices.device, dtype=torch.bool)
    chemical = torch.zeros(len(pair_indices), device=pair_indices.device, dtype=torch.long)
    structured = torch.zeros_like(presence)
    target_bonds = torch.zeros(
        targets["bond_mask"].size(0), device=pair_indices.device, dtype=torch.long
    )
    target_atoms = torch.zeros_like(target_bonds)
    matched_atoms = torch.zeros_like(target_bonds)
    selected_atoms = torch.zeros_like(target_bonds)
    for batch_index in range(targets["bond_mask"].size(0)):
        bond_count = int(targets["bond_mask"][batch_index].sum())
        atom_count = int(targets["atom_mask"][batch_index].sum())
        target_bonds[batch_index] = bond_count
        target_atoms[batch_index] = atom_count
        selected_atoms[batch_index] = int(atoms["atom_mask"][batch_index].sum())
        mapped_values = atom_mapping[batch_index]
        matched_atoms[batch_index] = int(
            mapped_values[mapped_values >= 0].unique().numel()
        )
        bond_lookup = {}
        adjacency = [set() for _ in range(atom_count)]
        for bond_index in range(bond_count):
            begin, end = targets["bond_atom_refs"][batch_index, bond_index].tolist()
            key = tuple(sorted((int(begin), int(end))))
            bond_lookup[key] = int(targets["bond_chemical"][batch_index, bond_index])
            adjacency[begin].add(end)
            adjacency[end].add(begin)
        rows = torch.where(sample_indices == batch_index)[0]
        for row in rows.tolist():
            pair = pair_indices[row]
            mapped = atom_mapping[batch_index, pair]
            if bool((mapped < 0).any()) or int(mapped[0]) == int(mapped[1]):
                continue
            key = tuple(sorted((int(mapped[0]), int(mapped[1]))))
            if key in bond_lookup:
                presence[row] = True
                chemical[row] = bond_lookup[key]
            elif adjacency[key[0]].intersection(adjacency[key[1]]):
                structured[row] = True
    return {
        "presence": presence,
        "chemical": chemical,
        "structured_negative": structured,
        "target_bonds": target_bonds,
        "target_atoms": target_atoms,
        "matched_atoms": matched_atoms,
        "selected_atoms": selected_atoms,
    }


def select_training_pairs(
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    atoms: dict[str, torch.Tensor],
    pair_targets: dict[str, torch.Tensor],
    *,
    negative_ratio: int,
    minimum_negatives: int,
) -> torch.Tensor:
    selected_rows = []
    for batch_index in sample_indices.unique(sorted=True).tolist():
        sample = sample_indices == batch_index
        positives = torch.where(sample & pair_targets["presence"])[0]
        negatives = torch.where(sample & ~pair_targets["presence"])[0]
        desired = min(
            len(negatives),
            max(minimum_negatives, negative_ratio * max(len(positives), 1)),
        )
        if desired:
            anchors = atoms["atom_geometry"][batch_index, :, :2]
            negative_pairs = pair_indices[negatives]
            distances = torch.linalg.vector_norm(
                anchors[negative_pairs[:, 0]] - anchors[negative_pairs[:, 1]], dim=-1
            )
            hard = pair_targets["structured_negative"][negatives].float() * 2.0
            scores = hard - distances + torch.rand_like(distances) * 0.01
            negatives = negatives[scores.topk(desired).indices]
        selected_rows.extend((positives, negatives))
    return torch.cat(selected_rows) if selected_rows else pair_indices.new_empty(0)


def collect_predictions(
    model,
    vision,
    atom,
    bond,
    set_head,
    records,
    collator,
    proposal_config,
    atom_decision_threshold,
    device,
    batch_size,
) -> list[dict]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = collator(records[start : start + batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            features, atoms = frozen_atom_objects(
                vision,
                atom,
                bond,
                set_head,
                pixels,
                proposal_config,
                atom_decision_threshold,
            )
            pairs, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
            mapping, _ = match_atom_targets(atoms, targets, proposal_config)
            pair_targets = build_pair_targets(
                pairs, sample_indices, mapping, targets, atoms
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    features,
                    atoms["atom_features"],
                    atoms["atom_geometry"][..., :2],
                    atoms["atom_scores"],
                    pairs,
                    sample_indices,
                )
            probabilities = output["presence_logits"].sigmoid().float().cpu()
            chemical = output["chemical_logits"].argmax(-1).cpu()
            for batch_index in range(len(pixels)):
                rows = torch.where(sample_indices == batch_index)[0]
                predictions.append(
                    {
                        "probabilities": probabilities[rows.cpu()],
                        "chemical": chemical[rows.cpu()],
                        "presence_targets": pair_targets["presence"][rows].cpu(),
                        "chemical_targets": pair_targets["chemical"][rows].cpu(),
                        "target_bonds": int(pair_targets["target_bonds"][batch_index]),
                        "target_atoms": int(pair_targets["target_atoms"][batch_index]),
                        "matched_atoms": int(pair_targets["matched_atoms"][batch_index]),
                        "selected_atoms": int(pair_targets["selected_atoms"][batch_index]),
                    }
                )
    return predictions


def summarize_predictions(predictions: list[dict], threshold: float) -> dict:
    totals = defaultdict(int)
    for sample in predictions:
        expected = sample["presence_targets"]
        selected = sample["probabilities"] >= threshold
        true_positive = selected & expected
        chemical_correct = sample["chemical"] == sample["chemical_targets"]
        representable = int(expected.sum())
        pair_exact = bool((selected == expected).all())
        bond_exact = pair_exact and representable == sample["target_bonds"]
        atom_exact = (
            sample["matched_atoms"] == sample["target_atoms"]
            and sample["selected_atoms"] == sample["target_atoms"]
        )
        type_exact = bond_exact and bool(chemical_correct[expected].all())
        totals["samples"] += 1
        totals["pairs"] += len(expected)
        totals["target_bonds"] += sample["target_bonds"]
        totals["representable_bonds"] += representable
        totals["selected_bonds"] += int(selected.sum())
        totals["true_positive"] += int(true_positive.sum())
        totals["atom_exact"] += int(atom_exact)
        totals["pair_candidate_exact"] += int(pair_exact)
        totals["bond_deployed_exact"] += int(bond_exact)
        totals["skeleton_exact"] += int(atom_exact and bond_exact)
        totals["chemical_bond_exact"] += int(type_exact)
        totals["full_chemical_exact"] += int(atom_exact and type_exact)
        totals["chemical_correct"] += int(chemical_correct[expected].sum())
    precision = totals["true_positive"] / max(totals["selected_bonds"], 1)
    recall = totals["true_positive"] / max(totals["target_bonds"], 1)
    samples = max(totals["samples"], 1)
    return {
        **dict(totals),
        "threshold": threshold,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "atom_exact_rate": totals["atom_exact"] / samples,
        "pair_candidate_exact_rate": totals["pair_candidate_exact"] / samples,
        "bond_deployed_exact_rate": totals["bond_deployed_exact"] / samples,
        "skeleton_exact_rate": totals["skeleton_exact"] / samples,
        "chemical_bond_exact_rate": totals["chemical_bond_exact"] / samples,
        "full_chemical_exact_rate": totals["full_chemical_exact"] / samples,
        "chemical_accuracy": totals["chemical_correct"]
        / max(totals["representable_bonds"], 1),
        "atom_pair_bond_coverage": totals["representable_bonds"]
        / max(totals["target_bonds"], 1),
    }


def select_threshold(predictions: list[dict]) -> dict:
    candidates = [
        summarize_predictions(predictions, value / 20.0)
        for value in range(2, 19)
    ]
    return max(
        candidates,
        key=lambda item: (
            item["bond_deployed_exact_rate"],
            item["skeleton_exact_rate"],
            item["edge_f1"],
        ),
    )


def save_checkpoint(path, model, optimizer, args, config, seen, validation, audit):
    torch.save(
        {
            "checkpoint_schema_version": "direct-atom-pair-bond-0.1",
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
        raise RuntimeError("Direct AtomPair training requires CUDA")
    random.seed(args.seed)
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
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    vision, atom, bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    fresh_records, fresh_audit = load_stratified_fresh_records(
        args.fresh_calibration_samples + args.fresh_eval_samples
    )
    calibration_records = fresh_records[: args.fresh_calibration_samples]
    eval_records = fresh_records[args.fresh_calibration_samples :]

    source_baseline = evaluate_source(
        vision,
        atom,
        bond,
        set_head,
        eval_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        atom_threshold,
        float(source["training_args"]["bond_decision_threshold"]),
    )
    audit = {
        "data_gate": data_gate,
        "fresh": fresh_audit,
        "source_baseline": source_baseline,
        "code_git_revision": code_revision,
        "candidate_gate": {
            "bond_deployed_exact_delta_min": 0.05,
            "edge_recall_delta_min": -0.01,
        },
        "sealed_test_read": False,
    }
    (output / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_baseline": source_baseline}, separators=(",", ":")), flush=True)

    config = DirectAtomPairBondConfig()
    model = DirectAtomPairBond(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    domain_counts = {
        "style_train": args.style_train_samples,
        "chemvision_replay": args.chemvision_replay_samples,
        "font_v2_replay": args.font_v2_replay_samples,
    }
    total_target = sum(domain_counts.values())
    total_seen = 0
    next_log = args.log_every_total_samples
    next_validation = args.validate_every_total_samples
    window = defaultdict(float)
    window_seen = 0
    best_score = None
    best_validation = None
    started = time.perf_counter()

    for samples, domain_seen in mixed_batches(
        domain_sample_counts=domain_counts,
        domain_batch_counts={
            "style_train": 4,
            "chemvision_replay": 1,
            "font_v2_replay": 1,
        },
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = stack_targets(batch, device)
        features, atoms = frozen_atom_objects(
            vision,
            atom,
            bond,
            set_head,
            pixels,
            proposal_config,
            atom_threshold,
        )
        pairs, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        mapping, _ = match_atom_targets(atoms, targets, proposal_config)
        pair_targets = build_pair_targets(pairs, sample_indices, mapping, targets, atoms)
        selected = select_training_pairs(
            pairs,
            sample_indices,
            atoms,
            pair_targets,
            negative_ratio=args.negative_ratio,
            minimum_negatives=args.minimum_negatives_per_image,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pair_output = model(
                features,
                atoms["atom_features"],
                atoms["atom_geometry"][..., :2],
                atoms["atom_scores"],
                pairs[selected],
                sample_indices[selected],
            )
            losses = direct_atom_pair_loss(
                pair_output,
                pair_targets["presence"][selected],
                pair_targets["chemical"][selected],
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = len(samples)
        total_seen += count
        window_seen += count
        for name in ("loss", "presence_loss", "chemical_loss"):
            window[name] += float(losses[name].detach()) * count

        if total_seen >= next_log:
            print(
                json.dumps(
                    {
                        "samples_seen": total_seen,
                        "domain_seen": domain_seen,
                        "sampled_pairs": len(selected),
                        "mean_losses": {
                            name: value / max(window_seen, 1)
                            for name, value in window.items()
                        },
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            window.clear()
            window_seen = 0
            next_log += args.log_every_total_samples

        if total_seen >= next_validation or total_seen == total_target:
            calibration = collect_predictions(
                model,
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
            validation = select_threshold(calibration)
            validation.update(
                {
                    "samples_seen": total_seen,
                    "domain_seen": domain_seen,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            save_checkpoint(
                output / f"samples-{total_seen:06d}.pt",
                model,
                optimizer,
                args,
                config,
                total_seen,
                validation,
                audit,
            )
            save_checkpoint(
                output / "last.pt",
                model,
                optimizer,
                args,
                config,
                total_seen,
                validation,
                audit,
            )
            score = (
                validation["bond_deployed_exact_rate"],
                validation["skeleton_exact_rate"],
                validation["edge_f1"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_validation = validation
                save_checkpoint(
                    output / "best.pt",
                    model,
                    optimizer,
                    args,
                    config,
                    total_seen,
                    validation,
                    audit,
                )
            (output / "metrics.json").write_text(
                json.dumps(
                    {"latest": validation, "best": best_validation},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"validation": validation}, separators=(",", ":")), flush=True)
            next_validation += args.validate_every_total_samples

    selected_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected_payload["model"], strict=True)
    evaluation_predictions = collect_predictions(
        model,
        vision,
        atom,
        bond,
        set_head,
        eval_records,
        collator,
        proposal_config,
        atom_threshold,
        device,
        args.batch_size,
    )
    evaluation = summarize_predictions(
        evaluation_predictions, float(selected_payload["validation"]["threshold"])
    )
    bond_delta = (
        evaluation["bond_deployed_exact_rate"]
        - source_baseline["bond_deployed_exact_rate"]
    )
    recall_delta = evaluation["edge_recall"] - source_baseline["bond_recall"]
    gate = {
        "bond_deployed_exact_delta": bond_delta,
        "edge_recall_delta": recall_delta,
        "passed": bond_delta >= 0.05 and recall_delta >= -0.01,
    }
    report = {
        "source_baseline": source_baseline,
        "selected_calibration": selected_payload["validation"],
        "evaluation": evaluation,
        "candidate_gate": gate,
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "fresh_eval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selected_payload["fresh_evaluation"] = evaluation
    selected_payload["candidate_gate"] = gate
    torch.save(selected_payload, output / "selected.pt")
    print(json.dumps({"final": report}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
