#!/usr/bin/env python3
"""Sweep global Atom/Bond presence thresholds on internal validation only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import (
    ProposalObjectSetConfig,
    ProposalObjectSetHead,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_predicted_incidence import DEFAULT_VALIDATION_SHARD, _load_records
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    load_frozen_components,
)


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0947-proposal-object-set-train100-val100-step1200/"
    "best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/diagnostics/chemvector-v0947-proposal-object-set-thresholds/"
    "validation100.json"
)


def sweep_thresholds(records: list[dict], thresholds: list[float]) -> dict:
    atom_results: dict[float, list[bool]] = {}
    bond_results: dict[float, list[bool]] = {}
    for threshold in thresholds:
        atom_results[threshold] = [
            bool(record["atom_covered"])
            and bool((record["atom_logits"] >= threshold).eq(record["atom_targets"]).all())
            for record in records
        ]
        bond_results[threshold] = [
            bool(record["bond_covered"])
            and bool((record["bond_logits"] >= threshold).eq(record["bond_targets"]).all())
            for record in records
        ]
    best = None
    frontier = []
    for atom_threshold in thresholds:
        atom_exact = sum(atom_results[atom_threshold])
        for bond_threshold in thresholds:
            bond_exact = sum(bond_results[bond_threshold])
            full_exact = sum(
                atom and bond and bool(record["incidence_exact"])
                for atom, bond, record in zip(
                    atom_results[atom_threshold],
                    bond_results[bond_threshold],
                    records,
                )
            )
            score = (full_exact, min(atom_exact, bond_exact), atom_exact + bond_exact)
            candidate = {
                "atom_threshold": atom_threshold,
                "bond_threshold": bond_threshold,
                "atom_deployed_exact": atom_exact,
                "bond_deployed_exact": bond_exact,
                "full_object_exact": full_exact,
                "score": score,
            }
            if best is None or score > best[0]:
                best = (score, candidate)
            frontier.append(candidate)
    assert best is not None
    atom_best = max(
        (
            {
                "threshold": threshold,
                "deployed_exact": sum(values),
            }
            for threshold, values in atom_results.items()
        ),
        key=lambda item: item["deployed_exact"],
    )
    bond_best = max(
        (
            {
                "threshold": threshold,
                "deployed_exact": sum(values),
            }
            for threshold, values in bond_results.items()
        ),
        key=lambda item: item["deployed_exact"],
    )
    return {
        "samples": len(records),
        "incidence_exact": sum(bool(record["incidence_exact"]) for record in records),
        "atom_best": atom_best,
        "bond_best": bond_best,
        "joint_best": best[1],
        "frontier": frontier,
    }


@torch.no_grad()
def collect_records(
    vision,
    atom_head,
    bond_head,
    set_head,
    samples,
    collator,
    proposal_config,
    device,
    batch_size,
) -> list[dict]:
    collected = []
    set_head.eval()
    for start in range(0, len(samples), batch_size):
        batch = collator(samples[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = bond_head(features)
            proposals = associate_predicted_proposals(
                decode_predicted_atom_proposals(atom_output, proposal_config),
                decode_predicted_bond_proposals(bond_output, proposal_config),
                targets,
                proposal_config,
            )
            output = set_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_scores"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_scores"],
                proposals["bond_mask"],
            )
        for batch_index in range(len(pixels)):
            atom_mask = proposals["atom_mask"][batch_index]
            bond_mask = proposals["bond_mask"][batch_index]
            atom_count = int(targets["atom_mask"][batch_index].sum())
            bond_count = int(targets["bond_mask"][batch_index].sum())
            atom_covered = (
                int(proposals["atom_target_matched"][batch_index].sum())
                == atom_count
            )
            bond_covered = (
                int(proposals["bond_target_matched"][batch_index].sum())
                == bond_count
            )
            supervised = proposals["bond_supervision_mask"][batch_index]
            incidence_correct = (
                output["incidence_logits"][batch_index, supervised].argmax(-1)
                == proposals["target_atom_refs"][batch_index, supervised]
            )
            incidence_exact = (
                int(proposals["incidence_target_eligible"][batch_index].sum())
                == bond_count
                and bool(incidence_correct.all())
            )
            collected.append(
                {
                    "atom_logits": output["atom_presence_logits"][
                        batch_index, atom_mask
                    ].float().cpu(),
                    "atom_targets": proposals["atom_presence_targets"][
                        batch_index, atom_mask
                    ].cpu(),
                    "atom_covered": atom_covered,
                    "bond_logits": output["bond_presence_logits"][
                        batch_index, bond_mask
                    ].float().cpu(),
                    "bond_targets": proposals["bond_presence_targets"][
                        batch_index, bond_mask
                    ].cpu(),
                    "bond_covered": bond_covered,
                    "incidence_exact": incidence_exact,
                }
            )
    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--threshold-min", type=float, default=-6.0)
    parser.add_argument("--threshold-max", type=float, default=6.0)
    parser.add_argument("--threshold-step", type=float, default=0.25)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal threshold audit requires CUDA")
    device = torch.device("cuda")
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incidence_config = SpatialIncidenceConfig(**payload["config"]["incidence"])
    config = ProposalObjectSetConfig(
        incidence=incidence_config,
        hidden_dim=payload["config"]["hidden_dim"],
    )
    set_head = ProposalObjectSetHead(config).to(device)
    set_head.load_state_dict(payload["head"], strict=True)
    proposal_config = PredictedProposalConfig(**payload["proposal_config"])
    samples = _load_records(
        args.validation_shard, args.validation_samples, "validation"
    )
    records = collect_records(
        vision,
        atom_head,
        bond_head,
        set_head,
        samples,
        AtomBondSceneCollator(AtomBondSceneConfig()),
        proposal_config,
        device,
        args.batch_size,
    )
    count = round((args.threshold_max - args.threshold_min) / args.threshold_step)
    thresholds = [
        args.threshold_min + index * args.threshold_step
        for index in range(count + 1)
    ]
    result = sweep_thresholds(records, thresholds)
    report = {
        "schema_version": "proposal-object-set-threshold-audit-0.1",
        "protocol": "global-logit-thresholds-no-per-image-count",
        "checkpoint": args.checkpoint,
        "checkpoint_step": payload["global_step"],
        "validation_shard": args.validation_shard,
        "proposal_config": asdict(proposal_config),
        "thresholds": {
            "minimum": args.threshold_min,
            "maximum": args.threshold_max,
            "step": args.threshold_step,
        },
        "sources": sources,
        "result": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    concise = dict(report)
    concise["result"] = {
        key: value for key, value in result.items() if key != "frontier"
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
