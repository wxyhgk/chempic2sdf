#!/usr/bin/env python3
"""Train paired rendering consistency over a frozen AtomPair proposal stack."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_graph_context import (
    atom_pair_graph_context_loss,
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
)
from chemvector.atom_pair_style_consistency import (
    AtomPairStyleResidualAdapter,
    flatten_target_atom_pairs,
    fold_style_residual_adapter,
    paired_style_consistency_loss,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.style_sensitivity import summarize_style_sensitivity
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import (
    TRAIN_DOMAINS,
    VALIDATION_DOMAINS,
    frozen_groups,
)
from chemvector.train_atom_pair_graph_context import load_group_head, validate
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import (
    load_validation_records,
    rejection_reason,
    sample_split,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH
from chemvector.train_stable_bond_presence import valid_stream


DEFAULT_CONTEXT = (
    "outputs/checkpoints/"
    "chemvector-v0987-atom-pair-graph-context-positive-margin-calibrated/best.pt"
)
DEFAULT_STYLE_ROOT = Path("Data/sdf/chemvision_layers_v0_2/font_pilot_10k")
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0994-atom-pair-style-consistency-pilot250"
)
VARIANT_ORDER = {name: index for index, name in enumerate(("original", "rdkit", "coordgen", "stress"))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-context-checkpoint", default=DEFAULT_CONTEXT)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-root", default=str(DEFAULT_STYLE_ROOT))
    parser.add_argument("--style-molecules", type=int, default=250)
    parser.add_argument("--style-candidate-reserve", type=int, default=64)
    parser.add_argument("--replay-samples-per-domain", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--style-molecules-per-batch", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument(
        "--style-validation-samples",
        type=int,
        default=0,
        help="Accepted style validation limit; 0 reads the complete shard",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--edge-consistency-weight", type=float, default=0.25)
    parser.add_argument("--count-consistency-weight", type=float, default=0.05)
    parser.add_argument("--degree-consistency-weight", type=float, default=0.1)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--count-weight", type=float, default=0.25)
    parser.add_argument("--degree-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--positive-margin", type=float, default=0.5)
    parser.add_argument("--positive-margin-weight", type=float, default=0.25)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def _style_shards(root: Path, split: str) -> list[str]:
    return [str(path) for path in sorted((root / "shards").glob(f"{split}-*.tar"))]


def load_paired_style_training_records(
    root: Path,
    *,
    molecule_count: int,
    reserve: int,
    seed: int,
    config: AtomBondSceneConfig,
) -> tuple[list[list[dict]], dict]:
    if min(molecule_count, reserve + 1) < 1:
        raise ValueError("Paired style selection sizes must be positive")
    with (root / "samples.csv").open(newline="", encoding="utf-8") as handle:
        molecule_ids = sorted(
            {
                row["molecule_id"]
                for row in csv.DictReader(handle)
                if row["split"] == "train"
            },
            key=lambda value: (stable_rank(seed, value), value),
        )
    candidates = molecule_ids[: molecule_count + reserve]
    candidate_set = set(candidates)
    grouped: dict[str, list[dict]] = defaultdict(list)
    rejected = Counter()
    for sample in ChemVectorIterableDataset(
        _style_shards(root, "train"), sequence_version="0.4"
    ):
        molecule_id = str(sample["chemmol"]["molecule_id"])
        if molecule_id not in candidate_set:
            continue
        if sample_split(sample) != "train":
            raise ValueError("Paired style stream contains a non-train sample")
        reason = rejection_reason(sample, config)
        if reason is not None:
            rejected[reason] += 1
            continue
        grouped[molecule_id].append(sample)

    selected = []
    incomplete = Counter()
    for molecule_id in candidates:
        rows = grouped.get(molecule_id, [])
        variants = {str(row["render"]["variant"]) for row in rows}
        if len(rows) != 4 or variants != set(VARIANT_ORDER):
            incomplete[str(len(rows))] += 1
            continue
        selected.append(sorted(rows, key=lambda row: VARIANT_ORDER[row["render"]["variant"]]))
        if len(selected) >= molecule_count:
            break
    if len(selected) != molecule_count:
        raise RuntimeError(
            f"Paired style selection yielded {len(selected)} of {molecule_count} molecules"
        )
    return selected, {
        "molecules": len(selected),
        "samples": sum(map(len, selected)),
        "candidate_molecules": len(candidates),
        "rejected": dict(rejected),
        "incomplete_candidate_molecules": dict(incomplete),
        "molecule_sha256": hashlib.sha256(
            "\n".join(str(rows[0]["chemmol"]["molecule_id"]) for rows in selected).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def load_replay_records(
    *,
    samples_per_domain: int,
    seed: int,
    config: AtomBondSceneConfig,
) -> tuple[dict[str, list[dict]], dict]:
    records = {}
    for offset, (domain, shards) in enumerate(TRAIN_DOMAINS.items()):
        rows = list(
            itertools.islice(
                valid_stream(
                    shards,
                    config=config,
                    seed=seed + offset,
                    shuffle_buffer=1024,
                ),
                samples_per_domain,
            )
        )
        if len(rows) != samples_per_domain:
            raise RuntimeError(
                f"Replay domain {domain} yielded {len(rows)} of {samples_per_domain}"
            )
        records[domain] = rows
    return records, {
        domain: {
            "samples": len(rows),
            "molecules": len({row["chemmol"]["molecule_id"] for row in rows}),
        }
        for domain, rows in records.items()
    }


def _scaled_output(base: dict, context: dict, residual_scale: float) -> dict:
    return {
        **context,
        "presence_logits": base["presence_logits"]
        + residual_scale * context["residual_logits"],
    }


def _supervised_loss(
    output: dict,
    groups: dict,
    pairs: torch.Tensor,
    *,
    batch_size: int,
    maximum_atoms: int,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return atom_pair_graph_context_loss(
        output,
        groups["targets"],
        groups["sample_indices"],
        pairs,
        batch_size=batch_size,
        maximum_atoms=maximum_atoms,
        count_weight=args.count_weight,
        degree_weight=args.degree_weight,
        residual_weight=args.residual_weight,
        positive_margin=args.positive_margin,
        positive_margin_weight=args.positive_margin_weight,
    )


def forward_training_batch(
    samples: list[dict],
    *,
    vision,
    atom,
    bond,
    set_head,
    group_head,
    teacher_context,
    style_adapter,
    collator,
    proposal_config,
    residual_scale: float,
    device: torch.device,
    args: argparse.Namespace,
    paired: bool,
) -> dict[str, torch.Tensor]:
    batch = collator(samples)
    pixels = batch["pixel_values"].to(device)
    targets = stack_targets(batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        proposals, groups = frozen_groups(
            vision, atom, bond, set_head, pixels, targets, proposal_config
        )
        base = group_head(groups["features"])
        pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
        teacher_raw = teacher_context(
            groups["features"],
            base["presence_logits"],
            groups["sample_indices"],
            pairs,
            batch_size=len(samples),
        )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        delta = style_adapter(teacher_raw["residual_hidden"])
        student_raw = {
            **teacher_raw,
            "residual_logits": teacher_raw["residual_logits"] + delta,
        }
        student = _scaled_output(base, student_raw, residual_scale)
        supervised = _supervised_loss(
            student,
            groups,
            pairs,
            batch_size=len(samples),
            maximum_atoms=proposal_config.maximum_atoms,
            args=args,
        )
        teacher_loss = delta.float().square().mean()
        if paired:
            target_pairs = flatten_target_atom_pairs(
                groups["target_atom_pairs"], device=device
            )
            molecule_groups = torch.arange(
                len(samples) // 4, device=device, dtype=torch.long
            ).repeat_interleave(4)
            representable = []
            for batch_index in range(len(samples)):
                sample = groups["sample_indices"] == batch_index
                target_count = int(targets["bond_mask"][batch_index].sum())
                representable.append(
                    int(proposals["bond_target_matched"][batch_index].sum())
                    == target_count
                    and int(groups["targets"][sample].sum()) == target_count
                )
            consistency = paired_style_consistency_loss(
                student["presence_logits"],
                groups["sample_indices"],
                target_pairs,
                molecule_groups,
                batch_size=len(samples),
                maximum_atoms=proposal_config.maximum_atoms,
                valid_samples=torch.tensor(
                    representable, device=device, dtype=torch.bool
                ),
                normalization_counts=targets["bond_mask"].sum(-1),
            )
        else:
            zero = supervised["loss"] * 0.0
            consistency = {
                "edge_consistency_loss": zero,
                "count_consistency_loss": zero,
                "degree_consistency_loss": zero,
                "aligned_target_pairs": zero,
            }
    return {**supervised, **consistency, "teacher_loss": teacher_loss}


@torch.no_grad()
def validate_style_context(
    *,
    vision,
    atom,
    bond,
    set_head,
    group_head,
    teacher_context,
    student_context,
    records: list[dict],
    collator,
    proposal_config,
    residual_scale: float,
    device: torch.device,
    batch_size: int,
) -> dict:
    details = []
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        batch = collator(rows)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals, groups = frozen_groups(
                vision, atom, bond, set_head, pixels, targets, proposal_config
            )
            base = group_head(groups["features"])
            pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
            teacher = _scaled_output(
                base,
                teacher_context(
                    groups["features"],
                    base["presence_logits"],
                    groups["sample_indices"],
                    pairs,
                    batch_size=len(rows),
                ),
                residual_scale,
            )
            student = _scaled_output(
                base,
                student_context(
                    groups["features"],
                    base["presence_logits"],
                    groups["sample_indices"],
                    pairs,
                    batch_size=len(rows),
                ),
                residual_scale,
            )
        chemical_correct = base["chemical_logits"].argmax(-1) == groups[
            "chemical_targets"
        ]
        for batch_index, row in enumerate(rows):
            sample = groups["sample_indices"] == batch_index
            expected = groups["targets"][sample]
            type_mask = groups["type_mask"][sample]
            target_count = int(targets["bond_mask"][batch_index].sum())
            matched_count = int(proposals["bond_target_matched"][batch_index].sum())
            representable = (
                matched_count == target_count
                and int(expected.sum()) == target_count
                and int((expected & type_mask).sum()) == target_count
            )
            item = {
                "molecule_id": str(row["chemmol"]["molecule_id"]),
                "variant": str(row["render"]["variant"]),
                "representable": bool(representable),
                "style": dict(row["render"].get("render_parameters", {})),
            }
            for prefix, logits in (
                ("baseline_", teacher["presence_logits"][sample]),
                ("", student["presence_logits"][sample]),
            ):
                selected = logits >= 0.0
                presence_exact = representable and bool((selected == expected).all())
                typed_exact = bool(chemical_correct[sample][expected & type_mask].all())
                chemical_exact = presence_exact and typed_exact
                item[f"{prefix}presence_exact"] = presence_exact
                item[f"{prefix}chemical_exact"] = chemical_exact
                item[f"{prefix}false_positive"] = int((selected & ~expected).sum())
                item[f"{prefix}false_negative"] = int((~selected & expected).sum())
                item[f"{prefix}wrong_type"] = int(
                    (selected & expected & ~chemical_correct[sample]).sum()
                )
            details.append(item)
    return {
        "baseline_style": summarize_style_sensitivity(details, baseline=True),
        "candidate_style": summarize_style_sensitivity(details, baseline=False),
        "samples": len(details),
    }


def selection_key(official: dict, style: dict, baseline_official: dict) -> tuple:
    official_domains = official["domains"]
    baseline_domains = baseline_official["domains"]
    domain_deltas = [
        min(
            official_domains[name]["deployed_exact"]
            - baseline_domains[name]["deployed_exact"],
            official_domains[name]["chemical_deployed_exact"]
            - baseline_domains[name]["chemical_deployed_exact"],
        )
        for name in baseline_domains
    ]
    paired = style["candidate_style"]["paired_renderings"]["presence_exact"]
    overall = style["candidate_style"]["overall"]
    safe = min(domain_deltas) >= 0
    return (
        int(safe),
        paired["all_renderings_exact"] if safe else min(domain_deltas),
        -paired["mixed_outcome"],
        overall["presence_exact"],
        official["pooled_deployed_exact_rate"],
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Paired style consistency training requires CUDA")
    if args.replay_batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Replay batch size must be divisible by the domain count")
    if min(args.style_molecules_per_batch, args.epochs, args.style_molecules) < 1:
        raise ValueError("Paired style training sizes must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    teacher_context, context_payload = load_atom_pair_graph_context_head(
        args.init_context_checkpoint, device
    )
    teacher_context.eval().requires_grad_(False)
    style_adapter = AtomPairStyleResidualAdapter(
        teacher_context.config.hidden_dim
    ).to(device)
    group_head, group_payload = load_group_head(context_payload["group_checkpoint"], device)
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head, group_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    residual_scale = float(context_payload.get("residual_scale", 1.0))

    style_training, style_training_audit = load_paired_style_training_records(
        Path(args.style_root),
        molecule_count=args.style_molecules,
        reserve=args.style_candidate_reserve,
        seed=args.seed,
        config=scene_config,
    )
    replay_records, replay_audit = load_replay_records(
        samples_per_domain=args.replay_samples_per_domain,
        seed=args.seed,
        config=scene_config,
    )
    style_validation, style_validation_audit = load_style_records(
        str(Path(args.style_root) / "shards" / "validation-000000.tar"),
        split="validation",
        samples=args.style_validation_samples,
        config=scene_config,
    )
    official_records = {}
    official_audit = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        rows, audit = load_validation_records(
            [shard], per_shard=args.validation_samples_per_domain, config=scene_config
        )
        official_records[domain] = rows
        official_audit[domain] = audit
    data_audit = {
        "style_training": style_training_audit,
        "replay": replay_audit,
        "style_validation": style_validation_audit,
        "official_validation": official_audit,
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(data_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_official = validate(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        teacher_context,
        official_records,
        collator,
        proposal_config,
        device,
        args.batch_size,
        residual_scale=residual_scale,
    )
    baseline_style = validate_style_context(
        vision=vision,
        atom=atom,
        bond=bond,
        set_head=set_head,
        group_head=group_head,
        teacher_context=teacher_context,
        student_context=teacher_context,
        records=style_validation,
        collator=collator,
        proposal_config=proposal_config,
        residual_scale=residual_scale,
        device=device,
        batch_size=args.batch_size,
    )
    initial = {"official": baseline_official, "style": baseline_style}
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"initial": initial}, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        style_adapter.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    best_key = selection_key(baseline_official, baseline_style, baseline_official)
    best_metrics = initial
    history = []
    total_style_seen = 0
    total_replay_seen = 0
    started = time.perf_counter()

    def checkpoint_payload(epoch: int, metrics: dict) -> dict:
        folded_context = fold_style_residual_adapter(teacher_context, style_adapter)
        return {
            "checkpoint_schema_version": "atom-pair-graph-context-0.1",
            "head": folded_context.state_dict(),
            "config": context_payload["config"],
            "group_checkpoint": context_payload["group_checkpoint"],
            "source_checkpoint": context_payload["source_checkpoint"],
            "samples_seen": total_style_seen + total_replay_seen,
            "style_samples_seen": total_style_seen,
            "replay_samples_seen": total_replay_seen,
            "residual_scale": residual_scale,
            "base_positive_protection_threshold": context_payload.get(
                "base_positive_protection_threshold"
            ),
            "initial_context_checkpoint": args.init_context_checkpoint,
            "style_adapter": style_adapter.state_dict(),
            "style_adapter_parameters": sum(
                parameter.numel() for parameter in style_adapter.parameters()
            ),
            "validation": metrics["official"],
            "style_validation": metrics["style"],
            "training_args": vars(args),
            "epoch": epoch,
        }

    torch.save(checkpoint_payload(0, initial), output_dir / "initial.pt")
    torch.save(checkpoint_payload(0, initial), output_dir / "best.pt")
    per_domain_replay = args.replay_batch_size // len(TRAIN_DOMAINS)

    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        style_order = list(range(len(style_training)))
        rng.shuffle(style_order)
        replay_orders = {}
        for offset, (domain, rows) in enumerate(replay_records.items()):
            order = list(range(len(rows)))
            random.Random(args.seed + epoch * 17 + offset).shuffle(order)
            replay_orders[domain] = order
        replay_offsets = {domain: 0 for domain in replay_records}
        running = defaultdict(float)
        steps = 0

        for start in range(0, len(style_order), args.style_molecules_per_batch):
            molecule_indices = style_order[
                start : start + args.style_molecules_per_batch
            ]
            if len(molecule_indices) != args.style_molecules_per_batch:
                continue
            style_samples = [
                sample
                for molecule_index in molecule_indices
                for sample in style_training[molecule_index]
            ]
            replay_samples = []
            for domain, rows in replay_records.items():
                order = replay_orders[domain]
                offset = replay_offsets[domain]
                indices = [
                    order[(offset + index) % len(order)]
                    for index in range(per_domain_replay)
                ]
                replay_offsets[domain] = (offset + per_domain_replay) % len(order)
                replay_samples.extend(rows[index] for index in indices)

            style_adapter.train()
            style_losses = forward_training_batch(
                style_samples,
                vision=vision,
                atom=atom,
                bond=bond,
                set_head=set_head,
                group_head=group_head,
                teacher_context=teacher_context,
                style_adapter=style_adapter,
                collator=collator,
                proposal_config=proposal_config,
                residual_scale=residual_scale,
                device=device,
                args=args,
                paired=True,
            )
            replay_losses = forward_training_batch(
                replay_samples,
                vision=vision,
                atom=atom,
                bond=bond,
                set_head=set_head,
                group_head=group_head,
                teacher_context=teacher_context,
                style_adapter=style_adapter,
                collator=collator,
                proposal_config=proposal_config,
                residual_scale=residual_scale,
                device=device,
                args=args,
                paired=False,
            )
            loss = (
                style_losses["loss"]
                + args.edge_consistency_weight
                * style_losses["edge_consistency_loss"]
                + args.count_consistency_weight
                * style_losses["count_consistency_loss"]
                + args.degree_consistency_weight
                * style_losses["degree_consistency_loss"]
                + args.replay_weight * replay_losses["loss"]
                + args.teacher_weight
                * (style_losses["teacher_loss"] + replay_losses["teacher_loss"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(style_adapter.parameters(), 1.0)
            optimizer.step()
            total_style_seen += len(style_samples)
            total_replay_seen += len(replay_samples)
            steps += 1
            running["loss"] += float(loss.detach())
            for prefix, losses in (("style", style_losses), ("replay", replay_losses)):
                for name, value in losses.items():
                    running[f"{prefix}_{name}"] += float(value.detach())

        style_adapter.eval()
        candidate_context = fold_style_residual_adapter(
            teacher_context, style_adapter
        ).to(device)
        candidate_context.eval().requires_grad_(False)
        official = validate(
            vision,
            atom,
            bond,
            set_head,
            group_head,
            candidate_context,
            official_records,
            collator,
            proposal_config,
            device,
            args.batch_size,
            residual_scale=residual_scale,
        )
        style = validate_style_context(
            vision=vision,
            atom=atom,
            bond=bond,
            set_head=set_head,
            group_head=group_head,
            teacher_context=teacher_context,
            student_context=candidate_context,
            records=style_validation,
            collator=collator,
            proposal_config=proposal_config,
            residual_scale=residual_scale,
            device=device,
            batch_size=args.batch_size,
        )
        metrics = {"official": official, "style": style}
        key = selection_key(official, style, baseline_official)
        selected = key > best_key
        if selected:
            best_key = key
            best_metrics = metrics
            torch.save(checkpoint_payload(epoch, metrics), output_dir / "best.pt")
        torch.save(checkpoint_payload(epoch, metrics), output_dir / f"epoch-{epoch}.pt")
        torch.save(checkpoint_payload(epoch, metrics), output_dir / "last.pt")
        row = {
            "epoch": epoch,
            "selected_best": selected,
            "selection_key": list(key),
            "style_seen": total_style_seen,
            "replay_seen": total_replay_seen,
            "mean_losses": {name: value / max(steps, 1) for name, value in running.items()},
            "official": official,
            "style": style,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {"initial": initial, "best": best_metrics, "history": history},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"epoch_result": row}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
