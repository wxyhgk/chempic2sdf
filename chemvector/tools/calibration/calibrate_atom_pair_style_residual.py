#!/usr/bin/env python3
"""Calibrate a frozen paired-style residual without rerunning training."""

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
from chemvector.atom_pair_style_consistency import (
    AtomPairStyleResidualAdapter,
    fold_style_residual_adapter,
)
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.style_sensitivity import summarize_style_sensitivity
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS, frozen_groups
from chemvector.train_atom_pair_graph_context import load_group_head
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-checkpoint", required=True)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-validation-shard", required=True)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[index / 10.0 for index in range(11)],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    return parser.parse_args()


@torch.no_grad()
def collect_records(
    *,
    vision,
    atom,
    bond,
    set_head,
    group_head,
    teacher_context,
    adapter,
    records: dict[str, list[dict]],
    collator,
    proposal_config,
    device: torch.device,
    batch_size: int,
) -> dict[str, list[dict]]:
    collected = {}
    for domain, rows in records.items():
        domain_rows = []
        for start in range(0, len(rows), batch_size):
            samples = rows[start : start + batch_size]
            batch = collator(samples)
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                proposals, groups = frozen_groups(
                    vision, atom, bond, set_head, pixels, targets, proposal_config
                )
                base = group_head(groups["features"])
                pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
                context = teacher_context(
                    groups["features"],
                    base["presence_logits"],
                    groups["sample_indices"],
                    pairs,
                    batch_size=len(samples),
                )
                delta = adapter(context["residual_hidden"])
            chemical_correct = base["chemical_logits"].argmax(-1) == groups[
                "chemical_targets"
            ]
            for batch_index, sample_row in enumerate(samples):
                sample = groups["sample_indices"] == batch_index
                expected = groups["targets"][sample]
                type_mask = groups["type_mask"][sample]
                target_count = int(targets["bond_mask"][batch_index].sum())
                domain_rows.append(
                    {
                        "molecule_id": str(sample_row["chemmol"]["molecule_id"]),
                        "variant": str(sample_row["render"]["variant"]),
                        "style": dict(
                            sample_row["render"].get("render_parameters", {})
                        ),
                        "base_logits": base["presence_logits"][sample].float().cpu(),
                        "teacher_residual": context["residual_logits"][sample]
                        .float()
                        .cpu(),
                        "style_delta": delta[sample].float().cpu(),
                        "targets": expected.cpu(),
                        "chemical_correct": chemical_correct[sample].cpu(),
                        "type_mask": type_mask.cpu(),
                        "target_count": target_count,
                        "matched_count": int(
                            proposals["bond_target_matched"][batch_index].sum()
                        ),
                        "positive_count": int(expected.sum()),
                    }
                )
        collected[domain] = domain_rows
    return collected


def evaluate_domain_rows(
    rows: list[dict], *, adapter_scale: float, residual_scale: float
) -> dict:
    totals = defaultdict(int)
    for row in rows:
        logits = row["base_logits"] + residual_scale * (
            row["teacher_residual"] + adapter_scale * row["style_delta"]
        )
        selected = logits >= 0.0
        expected = row["targets"]
        representable = (
            row["matched_count"] == row["target_count"]
            and row["positive_count"] == row["target_count"]
            and int((expected & row["type_mask"]).sum()) == row["target_count"]
        )
        presence_exact = representable and bool((selected == expected).all())
        chemical_exact = presence_exact and bool(
            row["chemical_correct"][expected & row["type_mask"]].all()
        )
        totals["samples"] += 1
        totals["presence_exact"] += int(presence_exact)
        totals["chemical_exact"] += int(chemical_exact)
        totals["false_positive"] += int((selected & ~expected).sum())
        totals["false_negative"] += int((~selected & expected).sum())
        totals["wrong_type"] += int(
            (selected & expected & ~row["chemical_correct"]).sum()
        )
    samples = max(totals["samples"], 1)
    return {
        **dict(totals),
        "presence_exact_rate": totals["presence_exact"] / samples,
        "chemical_exact_rate": totals["chemical_exact"] / samples,
    }


def evaluate_style_rows(
    rows: list[dict], *, adapter_scale: float, residual_scale: float
) -> dict:
    details = []
    for row in rows:
        item = {
            "molecule_id": row["molecule_id"],
            "variant": row["variant"],
            "style": row["style"],
        }
        for prefix, scale in (("baseline_", 0.0), ("", adapter_scale)):
            logits = row["base_logits"] + residual_scale * (
                row["teacher_residual"] + scale * row["style_delta"]
            )
            selected = logits >= 0.0
            expected = row["targets"]
            representable = (
                row["matched_count"] == row["target_count"]
                and row["positive_count"] == row["target_count"]
                and int((expected & row["type_mask"]).sum()) == row["target_count"]
            )
            presence_exact = representable and bool((selected == expected).all())
            chemical_exact = presence_exact and bool(
                row["chemical_correct"][expected & row["type_mask"]].all()
            )
            item["representable"] = representable
            item[f"{prefix}presence_exact"] = presence_exact
            item[f"{prefix}chemical_exact"] = chemical_exact
            item[f"{prefix}false_positive"] = int((selected & ~expected).sum())
            item[f"{prefix}false_negative"] = int((~selected & expected).sum())
            item[f"{prefix}wrong_type"] = int(
                (selected & expected & ~row["chemical_correct"]).sum()
            )
        details.append(item)
    return summarize_style_sensitivity(details, baseline=False)


def selection_key(result: dict, baseline: dict) -> tuple:
    deltas = []
    for domain, values in result["official"].items():
        base = baseline["official"][domain]
        deltas.extend(
            (
                values["presence_exact"] - base["presence_exact"],
                values["chemical_exact"] - base["chemical_exact"],
            )
        )
    paired = result["style"]["paired_renderings"]["presence_exact"]
    safe = min(deltas) >= 0
    return (
        int(safe),
        paired["all_renderings_exact"] if safe else min(deltas),
        -paired["mixed_outcome"],
        result["style"]["overall"]["presence_exact"],
        sum(values["presence_exact"] for values in result["official"].values()),
        -result["adapter_scale"],
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Style residual calibration requires CUDA")
    if not args.scales or min(args.scales) < 0.0:
        raise ValueError("Style residual scales must be non-negative")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    style_payload = torch.load(
        args.style_checkpoint, map_location="cpu", weights_only=False
    )
    teacher_context, teacher_payload = load_atom_pair_graph_context_head(
        style_payload["initial_context_checkpoint"], device
    )
    adapter = AtomPairStyleResidualAdapter(teacher_context.config.hidden_dim).to(device)
    adapter.load_state_dict(style_payload["style_adapter"], strict=True)
    adapter.eval().requires_grad_(False)
    group_head, group_payload = load_group_head(teacher_payload["group_checkpoint"], device)
    vision, atom, bond, set_head, source = load_components(
        group_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head, group_head, teacher_context):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    records = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        rows, _ = load_validation_records(
            [shard], per_shard=args.validation_samples_per_domain, config=scene_config
        )
        records[domain] = rows
    style_rows, style_audit = load_style_records(
        args.style_validation_shard,
        split="validation",
        samples=0,
        config=scene_config,
    )
    records["style"] = style_rows
    cached = collect_records(
        vision=vision,
        atom=atom,
        bond=bond,
        set_head=set_head,
        group_head=group_head,
        teacher_context=teacher_context,
        adapter=adapter,
        records=records,
        collator=collator,
        proposal_config=proposal_config,
        device=device,
        batch_size=args.batch_size,
    )
    residual_scale = float(teacher_payload.get("residual_scale", 1.0))
    results = []
    for scale in sorted(set(args.scales)):
        official = {
            domain: evaluate_domain_rows(
                cached[domain],
                adapter_scale=scale,
                residual_scale=residual_scale,
            )
            for domain in VALIDATION_DOMAINS
        }
        style = evaluate_style_rows(
            cached["style"],
            adapter_scale=scale,
            residual_scale=residual_scale,
        )
        results.append(
            {"adapter_scale": float(scale), "official": official, "style": style}
        )
    baseline = next(result for result in results if result["adapter_scale"] == 0.0)
    best = max(results, key=lambda result: selection_key(result, baseline))
    result = {
        "schema_version": "atom-pair-style-residual-calibration-0.1",
        "source_checkpoint": args.style_checkpoint,
        "uses_internal_test": False,
        "style_validation_audit": style_audit,
        "baseline": baseline,
        "results": results,
        "best": best,
        "best_selection_key": list(selection_key(best, baseline)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    calibrated = dict(teacher_payload)
    folded = fold_style_residual_adapter(
        teacher_context, adapter, scale=float(best["adapter_scale"])
    )
    calibrated["head"] = folded.state_dict()
    calibrated["style_residual_calibration"] = {
        "schema_version": result["schema_version"],
        "source_checkpoint": args.style_checkpoint,
        "adapter_scale": best["adapter_scale"],
        "uses_internal_test": False,
        "metrics": best,
    }
    checkpoint = Path(args.output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, checkpoint)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_checkpoint": str(checkpoint),
                "best": best,
                "best_selection_key": result["best_selection_key"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
