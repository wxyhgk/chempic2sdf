#!/usr/bin/env python3
"""Train target-independent Atom/Bond proposal set selection with incidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.proposal_object_set import (
    ProposalObjectSetConfig,
    ProposalObjectSetHead,
    proposal_object_set_loss,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_predicted_incidence import (
    DEFAULT_TRAIN_SHARD,
    DEFAULT_VALIDATION_SHARD,
    _load_records,
)
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    forward_frozen_bond_head,
    load_frozen_components,
)


DEFAULT_INCIDENCE = (
    "outputs/checkpoints/chemvector-v0946-predicted-incidence-train100-val100-step1000/"
    "best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0947-proposal-object-set-train100-val100"
)


def _sample_set_metrics(
    output: dict[str, torch.Tensor],
    proposals: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    batch_index: int,
    *,
    atom_threshold: float = 0.0,
    bond_threshold: float = 0.0,
) -> dict[str, int | bool]:
    atom_mask = proposals["atom_mask"][batch_index]
    bond_mask = proposals["bond_mask"][batch_index]
    atom_expected = proposals["atom_presence_targets"][batch_index, atom_mask]
    bond_expected = proposals["bond_presence_targets"][batch_index, bond_mask]
    atom_predicted = (
        output["atom_presence_logits"][batch_index, atom_mask] >= atom_threshold
    )
    bond_predicted = (
        output["bond_presence_logits"][batch_index, bond_mask] >= bond_threshold
    )
    atom_targets = int(targets["atom_mask"][batch_index].sum())
    bond_targets = int(targets["bond_mask"][batch_index].sum())
    atom_matched = int(proposals["atom_target_matched"][batch_index].sum())
    bond_matched = int(proposals["bond_target_matched"][batch_index].sum())
    atom_true_positive = int((atom_predicted & atom_expected).sum())
    atom_false_positive = int((atom_predicted & ~atom_expected).sum())
    bond_true_positive = int((bond_predicted & bond_expected).sum())
    bond_false_positive = int((bond_predicted & ~bond_expected).sum())
    atom_candidate_exact = bool((atom_predicted == atom_expected).all())
    bond_candidate_exact = bool((bond_predicted == bond_expected).all())
    atom_deployed_exact = atom_candidate_exact and atom_matched == atom_targets
    bond_deployed_exact = bond_candidate_exact and bond_matched == bond_targets

    supervised = proposals["bond_supervision_mask"][batch_index]
    incidence_predicted = output["incidence_logits"][
        batch_index, supervised
    ].argmax(-1)
    incidence_expected = proposals["target_atom_refs"][batch_index, supervised]
    incidence_correct = incidence_predicted == incidence_expected
    incidence_eligible = int(
        proposals["incidence_target_eligible"][batch_index].sum()
    )
    incidence_exact = (
        incidence_eligible == bond_targets and bool(incidence_correct.all())
    )
    return {
        "atom_targets": atom_targets,
        "atom_matched": atom_matched,
        "atom_true_positive": atom_true_positive,
        "atom_false_positive": atom_false_positive,
        "atom_false_negative": atom_targets - atom_true_positive,
        "atom_selected": int(atom_predicted.sum()),
        "atom_candidate_exact": atom_candidate_exact,
        "atom_deployed_exact": atom_deployed_exact,
        "bond_targets": bond_targets,
        "bond_matched": bond_matched,
        "bond_true_positive": bond_true_positive,
        "bond_false_positive": bond_false_positive,
        "bond_false_negative": bond_targets - bond_true_positive,
        "bond_selected": int(bond_predicted.sum()),
        "bond_candidate_exact": bond_candidate_exact,
        "bond_deployed_exact": bond_deployed_exact,
        "incidence_endpoints": incidence_correct.numel(),
        "incidence_endpoint_correct": int(incidence_correct.sum()),
        "incidence_bonds": len(incidence_correct),
        "incidence_bond_correct": int(incidence_correct.all(-1).sum()),
        "incidence_eligible": incidence_eligible,
        "incidence_exact": incidence_exact,
        "full_object_exact": (
            atom_deployed_exact and bond_deployed_exact and incidence_exact
        ),
    }


@torch.no_grad()
def evaluate(
    vision,
    atom_head,
    bond_head,
    set_head: ProposalObjectSetHead,
    records: list[dict],
    collator: AtomBondSceneCollator,
    proposal_config: PredictedProposalConfig,
    device: torch.device,
    batch_size: int,
    atom_decision_threshold: float = 0.0,
    bond_decision_threshold: float = 0.0,
) -> dict:
    vision.eval()
    atom_head.eval()
    bond_head.eval()
    set_head.eval()
    totals: dict[str, int] = {
        "samples": 0,
        "atom_targets": 0,
        "atom_matched": 0,
        "atom_true_positive": 0,
        "atom_false_positive": 0,
        "atom_false_negative": 0,
        "atom_selected": 0,
        "atom_candidate_exact": 0,
        "atom_deployed_exact": 0,
        "bond_targets": 0,
        "bond_matched": 0,
        "bond_true_positive": 0,
        "bond_false_positive": 0,
        "bond_false_negative": 0,
        "bond_selected": 0,
        "bond_candidate_exact": 0,
        "bond_deployed_exact": 0,
        "incidence_endpoints": 0,
        "incidence_endpoint_correct": 0,
        "incidence_bonds": 0,
        "incidence_bond_correct": 0,
        "incidence_eligible": 0,
        "incidence_exact": 0,
        "full_object_exact": 0,
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = forward_frozen_bond_head(vision, bond_head, features)
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
            metrics = _sample_set_metrics(
                output,
                proposals,
                targets,
                batch_index,
                atom_threshold=atom_decision_threshold,
                bond_threshold=bond_decision_threshold,
            )
            totals["samples"] += 1
            for name, value in metrics.items():
                totals[name] += int(value)
    samples = max(totals["samples"], 1)

    def rate(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(totals[denominator], 1)

    atom_precision_denominator = (
        totals["atom_true_positive"] + totals["atom_false_positive"]
    )
    bond_precision_denominator = (
        totals["bond_true_positive"] + totals["bond_false_positive"]
    )
    return {
        **totals,
        "atom_proposal_recall": rate("atom_matched", "atom_targets"),
        "atom_precision": totals["atom_true_positive"]
        / max(atom_precision_denominator, 1),
        "atom_recall": rate("atom_true_positive", "atom_targets"),
        "atom_candidate_exact_rate": totals["atom_candidate_exact"] / samples,
        "atom_deployed_exact_rate": totals["atom_deployed_exact"] / samples,
        "bond_proposal_recall": rate("bond_matched", "bond_targets"),
        "bond_precision": totals["bond_true_positive"]
        / max(bond_precision_denominator, 1),
        "bond_recall": rate("bond_true_positive", "bond_targets"),
        "bond_candidate_exact_rate": totals["bond_candidate_exact"] / samples,
        "bond_deployed_exact_rate": totals["bond_deployed_exact"] / samples,
        "incidence_endpoint_accuracy": rate(
            "incidence_endpoint_correct", "incidence_endpoints"
        ),
        "incidence_bond_accuracy": rate(
            "incidence_bond_correct", "incidence_bonds"
        ),
        "incidence_exact_rate": totals["incidence_exact"] / samples,
        "full_object_exact_rate": totals["full_object_exact"] / samples,
        "mean_atom_selected": totals["atom_selected"] / samples,
        "mean_bond_selected": totals["bond_selected"] / samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", default=DEFAULT_TRAIN_SHARD)
    parser.add_argument("--validation-shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-evaluation-samples", type=int, default=100)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--incidence-checkpoint", default=DEFAULT_INCIDENCE)
    parser.add_argument("--warmstart-set-checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--maximum-bonds", type=int, default=128)
    parser.add_argument("--atom-threshold", type=float, default=0.01)
    parser.add_argument("--bond-threshold", type=float, default=0.01)
    parser.add_argument("--atom-match-radius-px", type=float, default=6.0)
    parser.add_argument("--bond-match-radius-px", type=float, default=12.0)
    parser.add_argument("--atom-decision-threshold", type=float, default=0.0)
    parser.add_argument("--bond-decision-threshold", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--presence-learning-rate", type=float, default=3e-4)
    parser.add_argument("--incidence-learning-rate", type=float, default=3e-5)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal object set training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    if args.warmstart_set_checkpoint:
        warmstart_set = torch.load(
            args.warmstart_set_checkpoint, map_location="cpu", weights_only=False
        )
        incidence_config = SpatialIncidenceConfig(
            **warmstart_set["config"]["incidence"]
        )
        config = ProposalObjectSetConfig(
            incidence=incidence_config,
            hidden_dim=warmstart_set["config"]["hidden_dim"],
        )
        set_head = ProposalObjectSetHead(config).to(device)
        set_head.load_state_dict(warmstart_set["head"], strict=True)
    else:
        incidence_payload = torch.load(
            args.incidence_checkpoint, map_location="cpu", weights_only=False
        )
        incidence_config = SpatialIncidenceConfig(**incidence_payload["config"])
        config = ProposalObjectSetConfig(incidence=incidence_config)
        set_head = ProposalObjectSetHead(config).to(device)
        set_head.incidence.load_state_dict(incidence_payload["head"], strict=True)
    proposal_config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=args.atom_threshold,
        bond_threshold=args.bond_threshold,
        atom_match_radius_px=args.atom_match_radius_px,
        bond_match_radius_px=args.bond_match_radius_px,
    )
    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter
        for parameter in set_head.parameters()
        if id(parameter) not in incidence_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": presence_parameters, "lr": args.presence_learning_rate},
            {"params": incidence_parameters, "lr": args.incidence_learning_rate},
        ],
        weight_decay=0.0,
    )
    train_records = _load_records(args.train_shard, args.train_samples, "train")
    validation_records = _load_records(
        args.validation_shard, args.validation_samples, "validation"
    )
    train_evaluation_records = train_records[: args.train_evaluation_samples]
    collator = AtomBondSceneCollator(AtomBondSceneConfig())
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rng = random.Random(args.seed)
    best: tuple[float, ...] | None = None
    running_loss = 0.0
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        set_head.train()
        selected = [
            train_records[rng.randrange(len(train_records))]
            for _ in range(args.batch_size)
        ]
        batch = collator(selected)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = bond_head(features)
            proposals = associate_predicted_proposals(
                decode_predicted_atom_proposals(atom_output, proposal_config),
                decode_predicted_bond_proposals(bond_output, proposal_config),
                targets,
                proposal_config,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            losses = proposal_object_set_loss(
                output,
                proposals["atom_presence_targets"],
                proposals["bond_presence_targets"],
                proposals["atom_mask"],
                proposals["bond_mask"],
                proposals["target_atom_refs"],
                proposals["bond_supervision_mask"],
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(set_head.parameters(), 1.0)
        optimizer.step()
        running_loss += float(losses["loss"].detach())
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": running_loss / args.log_every,
                        "atom_presence_loss": float(
                            losses["atom_presence_loss"].detach()
                        ),
                        "bond_presence_loss": float(
                            losses["bond_presence_loss"].detach()
                        ),
                        "endpoint_accuracy": float(
                            losses["endpoint_accuracy"].detach()
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0
        if step % args.validate_every == 0 or step == args.max_steps:
            train_metrics = evaluate(
                vision,
                atom_head,
                bond_head,
                set_head,
                train_evaluation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
                args.atom_decision_threshold,
                args.bond_decision_threshold,
            )
            validation_metrics = evaluate(
                vision,
                atom_head,
                bond_head,
                set_head,
                validation_records,
                collator,
                proposal_config,
                device,
                args.batch_size,
                args.atom_decision_threshold,
                args.bond_decision_threshold,
            )
            report = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            print(json.dumps({"evaluation": report}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "proposal-object-set-checkpoint-0.1",
                "head": set_head.state_dict(),
                "config": {
                    "incidence": asdict(incidence_config),
                    "hidden_dim": config.hidden_dim,
                },
                "proposal_config": asdict(proposal_config),
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "evaluation": report,
                "sources": sources,
                "incidence_checkpoint": args.incidence_checkpoint,
                "warmstart_set_checkpoint": args.warmstart_set_checkpoint,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                validation_metrics["full_object_exact_rate"],
                min(
                    validation_metrics["atom_deployed_exact_rate"],
                    validation_metrics["bond_deployed_exact_rate"],
                ),
                validation_metrics["atom_candidate_exact_rate"]
                + validation_metrics["bond_candidate_exact_rate"],
                validation_metrics["incidence_exact_rate"],
            )
            if best is None or score > best:
                best = score
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
