#!/usr/bin/env python3
"""Calibrate a frozen graph-context residual on the two validation domains."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS, frozen_groups
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


@torch.no_grad()
def collect_records(
    vision,
    atom,
    bond,
    set_head,
    group_head,
    context_head,
    records,
    collator,
    proposal_config,
    device,
    batch_size,
) -> dict[str, list[dict]]:
    collected = {}
    for domain, rows in records.items():
        domain_rows = []
        for start in range(0, len(rows), batch_size):
            batch = collator(rows[start : start + batch_size])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals, groups = frozen_groups(
                    vision, atom, bond, set_head, pixels, targets, proposal_config
                )
                base = group_head(groups["features"])
                pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
                context = context_head(
                    groups["features"],
                    base["presence_logits"],
                    groups["sample_indices"],
                    pairs,
                    batch_size=len(pixels),
                )
            chemical = base["chemical_logits"].argmax(-1)
            chemical_correct = chemical == groups["chemical_targets"]
            for batch_index in range(len(pixels)):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                positive_types = sample & groups["type_mask"]
                domain_rows.append(
                    {
                        "base_logits": base["presence_logits"][sample].float().cpu(),
                        "residual_logits": context["residual_logits"][sample]
                        .float()
                        .cpu(),
                        "targets": expected.cpu(),
                        "target_count": target_count,
                        "matched_count": int(
                            proposals["bond_target_matched"][batch_index].sum()
                        ),
                        "positive_count": int(expected.sum()),
                        "type_coverage": int(positive_types.sum()) == target_count,
                        "chemical_exact": bool(
                            chemical_correct[positive_types].all()
                        ),
                    }
                )
        collected[domain] = domain_rows
    return collected


def evaluate_scale(
    records: dict[str, list[dict]],
    scale: float,
    base_positive_protection_threshold: float | None = None,
) -> dict:
    domains = {}
    for domain, rows in records.items():
        totals = defaultdict(int)
        for row in rows:
            logits = row["base_logits"] + float(scale) * row["residual_logits"]
            if base_positive_protection_threshold is not None:
                protected = row["base_logits"] >= float(
                    base_positive_protection_threshold
                )
                logits = torch.where(protected, row["base_logits"], logits)
            selected = logits >= 0.0
            expected = row["targets"]
            exact = bool((selected == expected).all())
            deployed = (
                exact
                and row["matched_count"] == row["target_count"]
                and row["positive_count"] == row["target_count"]
            )
            chemical = (
                deployed and row["type_coverage"] and row["chemical_exact"]
            )
            totals["samples"] += 1
            totals["true_positive"] += int((selected & expected).sum())
            totals["false_positive"] += int((selected & ~expected).sum())
            totals["false_negative"] += int((~selected & expected).sum())
            totals["count_exact"] += int(
                int(selected.sum()) == row["positive_count"]
            )
            totals["deployed_exact"] += int(deployed)
            totals["chemical_deployed_exact"] += int(chemical)
        samples = max(totals["samples"], 1)
        domains[domain] = {
            **dict(totals),
            "precision": totals["true_positive"]
            / max(totals["true_positive"] + totals["false_positive"], 1),
            "recall": totals["true_positive"]
            / max(totals["true_positive"] + totals["false_negative"], 1),
            "count_exact_rate": totals["count_exact"] / samples,
            "deployed_exact_rate": totals["deployed_exact"] / samples,
            "chemical_deployed_exact_rate": totals["chemical_deployed_exact"]
            / samples,
        }
    samples = sum(values["samples"] for values in domains.values())
    return {
        "scale": float(scale),
        "base_positive_protection_threshold": base_positive_protection_threshold,
        "domains": domains,
        "minimum_domain_chemical_deployed_exact": min(
            values["chemical_deployed_exact"] for values in domains.values()
        ),
        "pooled_chemical_deployed_exact": sum(
            values["chemical_deployed_exact"] for values in domains.values()
        ),
        "minimum_domain_deployed_exact": min(
            values["deployed_exact"] for values in domains.values()
        ),
        "pooled_deployed_exact": sum(
            values["deployed_exact"] for values in domains.values()
        ),
        "samples": samples,
    }


def selection_key(result: dict) -> tuple:
    return (
        result["minimum_domain_chemical_deployed_exact"],
        result["pooled_chemical_deployed_exact"],
        result["minimum_domain_deployed_exact"],
        result["pooled_deployed_exact"],
        -abs(result["scale"] - 1.0),
        -(
            result["base_positive_protection_threshold"]
            if result["base_positive_protection_threshold"] is not None
            else 1e9
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[index / 10.0 for index in range(16)],
    )
    parser.add_argument(
        "--protection-thresholds",
        nargs="+",
        type=float,
        default=[-1.0],
        help="Negative means disabled; non-negative protects base-positive logits above it",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Graph-context calibration requires CUDA")
    if not args.scales or min(args.scales) < 0.0:
        raise ValueError("Residual scales must be non-negative")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    context_head, context_payload = load_atom_pair_graph_context_head(
        args.checkpoint, device
    )
    group_head, group_payload = load_group_head(
        context_payload["group_checkpoint"], device
    )
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    records = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        rows, _ = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )
        records[domain] = rows
    cached = collect_records(
        vision,
        atom,
        bond,
        set_head,
        group_head,
        context_head,
        records,
        collator,
        proposal_config,
        device,
        args.batch_size,
    )
    thresholds = [
        None if value < 0.0 else float(value)
        for value in sorted(set(args.protection_thresholds))
    ]
    results = [
        evaluate_scale(cached, scale, threshold)
        for scale in sorted(set(args.scales))
        for threshold in thresholds
    ]
    best = max(results, key=selection_key)
    result = {
        "schema_version": "atom-pair-graph-context-calibration-0.1",
        "source_checkpoint": args.checkpoint,
        "selection_protocol": (
            "minimum-domain chemical exact, pooled chemical exact, "
            "minimum-domain presence exact, pooled presence exact"
        ),
        "uses_internal_test": False,
        "results": results,
        "best": best,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    calibrated = dict(context_payload)
    calibrated["residual_scale"] = float(best["scale"])
    calibrated["base_positive_protection_threshold"] = best[
        "base_positive_protection_threshold"
    ]
    calibrated["calibration"] = {
        "schema_version": result["schema_version"],
        "selection_protocol": result["selection_protocol"],
        "uses_internal_test": False,
        "metrics": best,
    }
    checkpoint = Path(args.output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
