#!/usr/bin/env python3
"""Train the proposal object set head from frozen compact proposal caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from chemvector.build_proposal_object_cache import TENSOR_FIELDS
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.proposal_object_set import (
    ProposalObjectSetConfig,
    ProposalObjectSetHead,
    proposal_object_set_loss,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig


DEFAULT_TRAIN_CACHE = "outputs/checkpoints/chemvector-v0949-proposal-cache-train9k"
DEFAULT_VALIDATION_CACHE = (
    "outputs/checkpoints/chemvector-v0949-proposal-cache-validation100"
)
DEFAULT_WARMSTART = (
    "outputs/checkpoints/chemvector-v0948-proposal-object-set-train1000-cont1200/"
    "best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0949-cached-proposal-object-set-train8974"
)


def proposal_configs_compatible_for_warmstart(
    warmstart: dict,
    cached: dict,
) -> bool:
    """Set/incidence heads are sequence-length agnostic across Atom capacities."""

    ignored = {"maximum_atoms"}
    return {
        key: value for key, value in warmstart.items() if key not in ignored
    } == {key: value for key, value in cached.items() if key not in ignored}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(directory: str | Path, expected_split: str) -> tuple[list[dict], dict]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "proposal-object-cache-manifest-0.1":
        raise ValueError("Unsupported proposal object cache manifest")
    if manifest.get("expected_split") != expected_split:
        raise ValueError("Proposal object cache split mismatch")
    records = []
    for item in manifest["cache_files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"Proposal object cache integrity failure: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "proposal-object-cache-shard-0.1":
            raise ValueError(f"Unsupported proposal object cache shard: {path}")
        records.extend(payload["records"])
    if len(records) != manifest["valid_records"]:
        raise ValueError("Proposal object cache record count mismatch")
    if any(record.get("source_split") != expected_split for record in records):
        raise ValueError("Proposal object cache contains an unexpected split")
    return records, manifest


def stack_cache_records(records: list[dict], device: torch.device) -> dict:
    batch = {
        name: torch.stack([record[name] for record in records]).to(
            device, non_blocking=True
        )
        for name in TENSOR_FIELDS
    }
    for name in (
        "atom_target_count",
        "bond_target_count",
        "atom_matched_count",
        "bond_matched_count",
        "incidence_eligible_count",
    ):
        batch[name] = torch.tensor(
            [record[name] for record in records], device=device, dtype=torch.long
        )
    return batch


def _forward(set_head: ProposalObjectSetHead, batch: dict) -> dict[str, torch.Tensor]:
    return set_head(
        batch["atom_features"],
        batch["atom_geometry"],
        batch["atom_scores"],
        batch["atom_mask"],
        batch["bond_features"],
        batch["bond_endpoints"],
        batch["bond_types"],
        batch["bond_scores"],
        batch["bond_mask"],
    )


def _cached_sample_metrics(
    output: dict[str, torch.Tensor],
    batch: dict,
    batch_index: int,
    atom_threshold: float,
    bond_threshold: float,
) -> dict[str, int | bool]:
    atom_mask = batch["atom_mask"][batch_index]
    bond_mask = batch["bond_mask"][batch_index]
    atom_expected = batch["atom_presence_targets"][batch_index, atom_mask]
    bond_expected = batch["bond_presence_targets"][batch_index, bond_mask]
    atom_predicted = (
        output["atom_presence_logits"][batch_index, atom_mask] >= atom_threshold
    )
    bond_predicted = (
        output["bond_presence_logits"][batch_index, bond_mask] >= bond_threshold
    )
    atom_targets = int(batch["atom_target_count"][batch_index])
    bond_targets = int(batch["bond_target_count"][batch_index])
    atom_matched = int(batch["atom_matched_count"][batch_index])
    bond_matched = int(batch["bond_matched_count"][batch_index])
    atom_true_positive = int((atom_predicted & atom_expected).sum())
    atom_false_positive = int((atom_predicted & ~atom_expected).sum())
    bond_true_positive = int((bond_predicted & bond_expected).sum())
    bond_false_positive = int((bond_predicted & ~bond_expected).sum())
    atom_candidate_exact = bool((atom_predicted == atom_expected).all())
    bond_candidate_exact = bool((bond_predicted == bond_expected).all())
    atom_deployed_exact = atom_candidate_exact and atom_matched == atom_targets
    bond_deployed_exact = bond_candidate_exact and bond_matched == bond_targets
    supervised = batch["bond_supervision_mask"][batch_index]
    incidence_correct = (
        output["incidence_logits"][batch_index, supervised].argmax(-1)
        == batch["target_atom_refs"][batch_index, supervised]
    )
    incidence_exact = (
        int(batch["incidence_eligible_count"][batch_index]) == bond_targets
        and bool(incidence_correct.all())
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
        "incidence_exact": incidence_exact,
        "full_object_exact": atom_deployed_exact and bond_deployed_exact and incidence_exact,
    }


@torch.no_grad()
def evaluate_cached(
    set_head: ProposalObjectSetHead,
    records: list[dict],
    device: torch.device,
    batch_size: int,
    atom_threshold: float,
    bond_threshold: float,
) -> dict:
    set_head.eval()
    total_names = (
        "atom_targets",
        "atom_matched",
        "atom_true_positive",
        "atom_false_positive",
        "atom_false_negative",
        "atom_selected",
        "atom_candidate_exact",
        "atom_deployed_exact",
        "bond_targets",
        "bond_matched",
        "bond_true_positive",
        "bond_false_positive",
        "bond_false_negative",
        "bond_selected",
        "bond_candidate_exact",
        "bond_deployed_exact",
        "incidence_endpoints",
        "incidence_endpoint_correct",
        "incidence_bonds",
        "incidence_bond_correct",
        "incidence_exact",
        "full_object_exact",
    )
    totals = {name: 0 for name in total_names}
    totals["samples"] = 0
    for start in range(0, len(records), batch_size):
        batch = stack_cache_records(records[start : start + batch_size], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward(set_head, batch)
        for batch_index in range(len(batch["atom_mask"])):
            metrics = _cached_sample_metrics(
                output,
                batch,
                batch_index,
                atom_threshold,
                bond_threshold,
            )
            totals["samples"] += 1
            for name, value in metrics.items():
                totals[name] += int(value)
    samples = max(totals["samples"], 1)

    def rate(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(totals[denominator], 1)

    return {
        **totals,
        "atom_proposal_recall": rate("atom_matched", "atom_targets"),
        "atom_precision": totals["atom_true_positive"]
        / max(totals["atom_true_positive"] + totals["atom_false_positive"], 1),
        "atom_recall": rate("atom_true_positive", "atom_targets"),
        "atom_deployed_exact_rate": totals["atom_deployed_exact"] / samples,
        "bond_proposal_recall": rate("bond_matched", "bond_targets"),
        "bond_precision": totals["bond_true_positive"]
        / max(totals["bond_true_positive"] + totals["bond_false_positive"], 1),
        "bond_recall": rate("bond_true_positive", "bond_targets"),
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


@torch.no_grad()
def select_representable_hard_records(
    set_head: ProposalObjectSetHead,
    records: list[dict],
    device: torch.device,
    batch_size: int,
    atom_threshold: float,
    bond_threshold: float,
) -> list[dict]:
    set_head.eval()
    hard_records = []
    for start in range(0, len(records), batch_size):
        source = records[start : start + batch_size]
        batch = stack_cache_records(source, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward(set_head, batch)
        for batch_index, record in enumerate(source):
            representable = (
                record["atom_matched_count"] == record["atom_target_count"]
                and record["bond_matched_count"] == record["bond_target_count"]
                and record["incidence_eligible_count"] == record["bond_target_count"]
            )
            if not representable:
                continue
            metrics = _cached_sample_metrics(
                output,
                batch,
                batch_index,
                atom_threshold,
                bond_threshold,
            )
            if not metrics["full_object_exact"]:
                hard_records.append(record)
    return hard_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--validation-cache", default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--warmstart-checkpoint", default=DEFAULT_WARMSTART)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Use the first N cached train samples; 0 keeps the full cache.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--train-evaluation-samples", type=int, default=200)
    parser.add_argument("--presence-learning-rate", type=float, default=5e-5)
    parser.add_argument("--incidence-learning-rate", type=float, default=5e-6)
    parser.add_argument("--topology-closure-weight", type=float, default=0.0)
    parser.add_argument("--graph-exact-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--atom-decision-threshold", type=float, default=0.0)
    parser.add_argument("--bond-decision-threshold", type=float, default=-0.5)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--selection-split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--hard-example-replay-probability", type=float, default=0.0)
    parser.add_argument("--hard-example-scan-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cached proposal object set training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not 0.0 <= args.hard_example_replay_probability <= 1.0:
        raise ValueError("Hard example replay probability must be between 0 and 1")
    device = torch.device("cuda")
    train_records, train_manifest = load_cache(args.train_cache, "train")
    if args.train_samples > 0:
        train_records = train_records[: args.train_samples]
    validation_records, validation_manifest = load_cache(
        args.validation_cache, "validation"
    )
    if train_manifest["proposal_config"] != validation_manifest["proposal_config"]:
        raise ValueError("Train and validation proposal cache configurations differ")
    warmstart = torch.load(
        args.warmstart_checkpoint, map_location="cpu", weights_only=False
    )
    if not proposal_configs_compatible_for_warmstart(
        warmstart["proposal_config"], train_manifest["proposal_config"]
    ):
        raise ValueError("Warmstart and proposal cache configurations differ")
    incidence_config = SpatialIncidenceConfig(**warmstart["config"]["incidence"])
    config = ProposalObjectSetConfig(
        incidence=incidence_config,
        hidden_dim=warmstart["config"]["hidden_dim"],
    )
    set_head = ProposalObjectSetHead(config).to(device)
    set_head.load_state_dict(warmstart["head"], strict=True)
    hard_records = []
    if args.hard_example_replay_probability > 0.0:
        hard_records = select_representable_hard_records(
            set_head,
            train_records,
            device,
            args.hard_example_scan_batch_size,
            args.atom_decision_threshold,
            args.bond_decision_threshold,
        )
        if not hard_records:
            raise ValueError("Hard example replay requested but no hard records were found")
    incidence_parameters = list(set_head.incidence.parameters())
    incidence_ids = {id(parameter) for parameter in incidence_parameters}
    presence_parameters = [
        parameter for parameter in set_head.parameters() if id(parameter) not in incidence_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": presence_parameters, "lr": args.presence_learning_rate},
            {"params": incidence_parameters, "lr": args.incidence_learning_rate},
        ],
        weight_decay=0.0,
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "hard_example_summary.json").write_text(
        json.dumps(
            {
                "train_records": len(train_records),
                "representable_hard_records": len(hard_records),
                "replay_probability": args.hard_example_replay_probability,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rng = random.Random(args.seed)
    best: tuple[float, ...] | None = None
    running_loss = 0.0
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        set_head.train()
        selected = []
        for _ in range(args.batch_size):
            pool = (
                hard_records
                if hard_records
                and rng.random() < args.hard_example_replay_probability
                else train_records
            )
            selected.append(pool[rng.randrange(len(pool))])
        batch = stack_cache_records(selected, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _forward(set_head, batch)
            losses = proposal_object_set_loss(
                output,
                batch["atom_presence_targets"],
                batch["bond_presence_targets"],
                batch["atom_mask"],
                batch["bond_mask"],
                batch["target_atom_refs"],
                batch["bond_supervision_mask"],
                topology_closure_weight=args.topology_closure_weight,
                graph_exact_surrogate_weight=args.graph_exact_surrogate_weight,
                atom_decision_threshold=args.atom_decision_threshold,
                bond_decision_threshold=args.bond_decision_threshold,
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
                        "topology_closure_loss": float(
                            losses["topology_closure_loss"].detach()
                        ),
                        "graph_exact_surrogate_loss": float(
                            losses["graph_exact_surrogate_loss"].detach()
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0
        if step % args.validate_every == 0 or step == args.max_steps:
            train_metrics = evaluate_cached(
                set_head,
                train_records[: args.train_evaluation_samples],
                device,
                args.evaluation_batch_size,
                args.atom_decision_threshold,
                args.bond_decision_threshold,
            )
            validation_metrics = evaluate_cached(
                set_head,
                validation_records,
                device,
                args.evaluation_batch_size,
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
                "checkpoint_schema_version": "cached-proposal-object-set-checkpoint-0.1",
                "head": set_head.state_dict(),
                "config": {
                    "incidence": asdict(incidence_config),
                    "hidden_dim": config.hidden_dim,
                },
                "proposal_config": train_manifest["proposal_config"],
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "evaluation": report,
                "train_cache": args.train_cache,
                "validation_cache": args.validation_cache,
                "warmstart_checkpoint": args.warmstart_checkpoint,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            selection_metrics = (
                train_metrics if args.selection_split == "train" else validation_metrics
            )
            score = (
                selection_metrics["full_object_exact_rate"],
                min(
                    selection_metrics["atom_deployed_exact_rate"],
                    selection_metrics["bond_deployed_exact_rate"],
                ),
                selection_metrics["incidence_exact_rate"],
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
