#!/usr/bin/env python3
"""Learn train-only linear potentials for joint graph MILP decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.joint_graph_milp import JointGraphMilpConfig, solve_joint_graph_milp
from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_cached_proposal_object_set import (
    _forward,
    load_cache,
    stack_cache_records,
)


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0949-cached-proposal-object-set-train8974-step3000/"
    "best.pt"
)
DEFAULT_TRAIN_CACHE = "outputs/checkpoints/chemvector-v0949-proposal-cache-train9k"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0960-joint-graph-potentials/potentials.json"
POTENTIAL_NAMES = (
    "atom_presence_weight",
    "atom_bias",
    "bond_presence_weight",
    "bond_bias",
    "incidence_weight",
)


def _relative_incidence(
    log_probabilities: torch.Tensor, bond: int, begin: int, end: int, atom_mask: torch.Tensor
) -> float:
    values = log_probabilities[bond].clone()
    values[:, ~atom_mask] = -1e9
    best = float(values[0].max() + values[1].max())
    return float(values[0, begin] + values[1, end]) - best


def _graph_features(
    *,
    atoms: list[int] | set[int],
    bonds: dict[int, tuple[int, int]],
    atom_presence_logits: torch.Tensor,
    bond_presence_logits: torch.Tensor,
    incidence_logits: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_threshold: float,
    bond_threshold: float,
) -> np.ndarray:
    log_probabilities = incidence_logits.float().log_softmax(-1)
    return np.asarray(
        (
            sum(float(atom_presence_logits[index]) - atom_threshold for index in atoms),
            len(atoms),
            sum(float(bond_presence_logits[index]) - bond_threshold for index in bonds),
            len(bonds),
            sum(
                _relative_incidence(
                    log_probabilities, bond, endpoints[0], endpoints[1], atom_mask
                )
                for bond, endpoints in bonds.items()
            ),
        ),
        dtype=np.float64,
    )


def _ground_truth_graph(record: dict, incidence_logits: torch.Tensor, endpoint_topk: int):
    if record["atom_matched_count"] != record["atom_target_count"]:
        return None
    if record["bond_matched_count"] != record["bond_target_count"]:
        return None
    if record["incidence_eligible_count"] != record["bond_target_count"]:
        return None
    atom_mask = record["atom_mask"]
    atom_targets = record["atom_presence_targets"] & atom_mask
    bond_targets = record["bond_presence_targets"] & record["bond_mask"]
    supervised = record["bond_supervision_mask"]
    if not torch.equal(supervised, bond_targets):
        return None
    atoms = set(torch.where(atom_targets)[0].tolist())
    bonds = {}
    pairs = set()
    incident_atoms = set()
    topk = min(endpoint_topk, int(atom_mask.sum()))
    for bond in torch.where(bond_targets)[0].tolist():
        begin, end = (int(value) for value in record["target_atom_refs"][bond])
        if begin == end or begin not in atoms or end not in atoms:
            return None
        for endpoint, target in enumerate((begin, end)):
            values = incidence_logits[bond, endpoint].clone()
            values[~atom_mask] = -1e9
            if target not in values.topk(topk).indices.tolist():
                return None
        pair = tuple(sorted((begin, end)))
        if pair in pairs:
            return None
        pairs.add(pair)
        incident_atoms.update(pair)
        bonds[bond] = (begin, end)
    if atoms != incident_atoms:
        return None
    return atoms, bonds


def _config(weights: np.ndarray, args: argparse.Namespace) -> JointGraphMilpConfig:
    return JointGraphMilpConfig(
        endpoint_topk=args.endpoint_topk,
        atom_presence_weight=float(weights[0]),
        atom_bias=float(weights[1]),
        bond_presence_weight=float(weights[2]),
        bond_bias=float(weights[3]),
        incidence_weight=float(weights[4]),
        time_limit_seconds=args.time_limit_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--endpoint-topk", type=int, default=2)
    parser.add_argument("--atom-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-threshold", type=float, default=-0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--time-limit-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Joint graph potential training requires CUDA")
    records, manifest = load_cache(args.train_cache, "train")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["proposal_config"] != manifest["proposal_config"]:
        raise ValueError("Checkpoint and train cache proposal configurations differ")
    rng = random.Random(args.seed)
    candidate_indices = [
        index
        for index, record in enumerate(records)
        if record["atom_matched_count"] == record["atom_target_count"]
        and record["bond_matched_count"] == record["bond_target_count"]
        and record["incidence_eligible_count"] == record["bond_target_count"]
    ]
    rng.shuffle(candidate_indices)
    candidate_indices = candidate_indices[: args.candidate_samples]
    incidence_config = SpatialIncidenceConfig(**payload["config"]["incidence"])
    model = ProposalObjectSetHead(
        ProposalObjectSetConfig(
            incidence=incidence_config, hidden_dim=payload["config"]["hidden_dim"]
        )
    ).cuda()
    model.load_state_dict(payload["head"], strict=True)
    model.eval()
    examples = []
    for start in range(0, len(candidate_indices), args.batch_size):
        indices = candidate_indices[start : start + args.batch_size]
        selected_records = [records[index] for index in indices]
        batch = stack_cache_records(selected_records, torch.device("cuda"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = _forward(model, batch)
        for local_index, record in enumerate(selected_records):
            incidence_logits = output["incidence_logits"][local_index].float().cpu()
            truth = _ground_truth_graph(record, incidence_logits, args.endpoint_topk)
            if truth is None:
                continue
            examples.append(
                {
                    "record": record,
                    "atom_logits": output["atom_presence_logits"][local_index]
                    .float()
                    .cpu(),
                    "bond_logits": output["bond_presence_logits"][local_index]
                    .float()
                    .cpu(),
                    "incidence_logits": incidence_logits,
                    "truth": truth,
                }
            )
    rng.shuffle(examples)
    weights = np.asarray((1.0, 0.0, 1.0, 0.0, 0.5), dtype=np.float64)
    averaged = np.zeros_like(weights)
    updates = exact_before = exact_after = 0
    started = time.perf_counter()
    for step, example in enumerate(examples, 1):
        record = example["record"]
        truth_atoms, truth_bonds = example["truth"]
        predicted_atoms, predicted_bonds, _ = solve_joint_graph_milp(
            atom_presence_logits=example["atom_logits"],
            atom_mask=record["atom_mask"],
            bond_presence_logits=example["bond_logits"],
            bond_mask=record["bond_mask"],
            incidence_logits=example["incidence_logits"],
            atom_threshold=args.atom_threshold,
            bond_threshold=args.bond_threshold,
            config=_config(weights, args),
        )
        truth_pairs = {bond: tuple(sorted(pair)) for bond, pair in truth_bonds.items()}
        predicted_pairs = {
            bond: tuple(sorted(pair)) for bond, pair in predicted_bonds.items()
        }
        exact = set(predicted_atoms) == truth_atoms and predicted_pairs == truth_pairs
        exact_before += int(exact)
        if not exact:
            truth_features = _graph_features(
                atoms=truth_atoms,
                bonds=truth_bonds,
                atom_presence_logits=example["atom_logits"],
                bond_presence_logits=example["bond_logits"],
                incidence_logits=example["incidence_logits"],
                atom_mask=record["atom_mask"],
                atom_threshold=args.atom_threshold,
                bond_threshold=args.bond_threshold,
            )
            predicted_features = _graph_features(
                atoms=predicted_atoms,
                bonds=predicted_bonds,
                atom_presence_logits=example["atom_logits"],
                bond_presence_logits=example["bond_logits"],
                incidence_logits=example["incidence_logits"],
                atom_mask=record["atom_mask"],
                atom_threshold=args.atom_threshold,
                bond_threshold=args.bond_threshold,
            )
            scale = max(len(truth_atoms) + len(truth_bonds), 1)
            weights += args.learning_rate * (truth_features - predicted_features) / scale
            weights[[0, 2, 4]] = np.clip(weights[[0, 2, 4]], 0.05, 3.0)
            weights[[1, 3]] = np.clip(weights[[1, 3]], -2.0, 2.0)
            updates += 1
        averaged += weights
    averaged /= max(len(examples), 1)

    for example in examples:
        record = example["record"]
        truth_atoms, truth_bonds = example["truth"]
        predicted_atoms, predicted_bonds, _ = solve_joint_graph_milp(
            atom_presence_logits=example["atom_logits"],
            atom_mask=record["atom_mask"],
            bond_presence_logits=example["bond_logits"],
            bond_mask=record["bond_mask"],
            incidence_logits=example["incidence_logits"],
            atom_threshold=args.atom_threshold,
            bond_threshold=args.bond_threshold,
            config=_config(averaged, args),
        )
        exact_after += int(
            set(predicted_atoms) == truth_atoms
            and {bond: tuple(sorted(pair)) for bond, pair in predicted_bonds.items()}
            == {bond: tuple(sorted(pair)) for bond, pair in truth_bonds.items()}
        )
    result = {
        "schema_version": "joint-graph-potentials-0.1",
        "protocol": "train-only-averaged-structured-perceptron",
        "checkpoint": args.checkpoint,
        "train_cache": args.train_cache,
        "candidate_records": len(candidate_indices),
        "eligible_examples": len(examples),
        "updates": updates,
        "train_exact_before": exact_before,
        "train_exact_before_rate": exact_before / max(len(examples), 1),
        "train_exact_after": exact_after,
        "train_exact_after_rate": exact_after / max(len(examples), 1),
        "elapsed_seconds": time.perf_counter() - started,
        "potentials": dict(zip(POTENTIAL_NAMES, averaged.tolist())),
        "final_potentials": dict(zip(POTENTIAL_NAMES, weights.tolist())),
        "training_args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
