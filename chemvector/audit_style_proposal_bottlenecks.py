#!/usr/bin/env python3
"""Audit style sensitivity at each frozen Atom/Bond proposal stage."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.style_sensitivity import style_dimensions
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_deletion_veto import frozen_relation_groups
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import DEFAULT_GRAPH


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_SHARD = (
    "Data/sdf/chemvision_layers_v0_2/font_pilot_10k/shards/"
    "validation-000000.tar"
)
STAGES = (
    "atom_coverage",
    "bond_coverage",
    "incidence_eligibility",
    "incidence_exact",
    "group_coverage",
    "type_coverage",
    "representable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument(
        "--split",
        choices=("validation", "internal_test"),
        default="validation",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Accepted sample limit; 0 reads the complete shard",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize_stage_rows(rows: list[dict]) -> dict:
    def rates(group: list[dict]) -> dict:
        return {
            "samples": len(group),
            **{
                stage: sum(bool(row[stage]) for row in group) / max(len(group), 1)
                for stage in STAGES
            },
        }

    breakdowns = {}
    for dimension in (
        "font_family",
        "font_bucket",
        "line_width_bucket",
        "rotation_bucket",
        "multiple_bond_offset_bucket",
        "layout",
        "kekulize",
        "stress_operation",
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[style_dimensions(row)[dimension]].append(row)
        breakdowns[dimension] = {
            name: rates(group) for name, group in sorted(groups.items())
        }

    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_molecule[row["molecule_id"]].append(row)
    paired = {}
    for stage in STAGES:
        all_exact = none_exact = mixed = incomplete = 0
        for group in by_molecule.values():
            if len(group) != 4:
                incomplete += 1
                continue
            values = [bool(row[stage]) for row in group]
            if all(values):
                all_exact += 1
            elif any(values):
                mixed += 1
            else:
                none_exact += 1
        paired[stage] = {
            "molecules": all_exact + none_exact + mixed,
            "all_renderings": all_exact,
            "no_rendering": none_exact,
            "mixed": mixed,
            "mixed_rate": mixed / max(all_exact + none_exact + mixed, 1),
            "incomplete_excluded": incomplete,
        }
    return {"overall": rates(rows), "breakdowns": breakdowns, "paired": paired}


@torch.no_grad()
def collect_rows(
    *,
    vision,
    atom,
    bond,
    set_head,
    records: list[dict],
    collator,
    proposal_config,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    details = []
    for start in range(0, len(records), batch_size):
        samples = records[start : start + batch_size]
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals, set_output, groups = frozen_relation_groups(
                vision, atom, bond, set_head, pixels, targets, proposal_config
            )
        for batch_index, sample_row in enumerate(samples):
            target_atom_count = int(targets["atom_mask"][batch_index].sum())
            target_bond_count = int(targets["bond_mask"][batch_index].sum())
            atom_coverage = (
                int(proposals["atom_target_matched"][batch_index].sum())
                == target_atom_count
            )
            bond_coverage = (
                int(proposals["bond_target_matched"][batch_index].sum())
                == target_bond_count
            )
            eligibility = (
                int(proposals["incidence_target_eligible"][batch_index].sum())
                == target_bond_count
            )
            supervised = proposals["bond_supervision_mask"][batch_index]
            predicted_refs = set_output["incidence_logits"][
                batch_index, supervised
            ].argmax(-1)
            expected_refs = proposals["target_atom_refs"][batch_index, supervised]
            incidence_exact = eligibility and bool((predicted_refs == expected_refs).all())
            sample = groups["sample_indices"] == batch_index
            group_coverage = int(groups["targets"][sample].sum()) == target_bond_count
            type_coverage = int(
                (groups["targets"][sample] & groups["type_mask"][sample]).sum()
            ) == target_bond_count
            details.append(
                {
                    "sample_key": str(sample_row["key"]),
                    "molecule_id": str(sample_row["chemmol"]["molecule_id"]),
                    "variant": str(sample_row["render"]["variant"]),
                    "style": dict(
                        sample_row["render"].get("render_parameters", {})
                    ),
                    "atom_coverage": atom_coverage,
                    "bond_coverage": bond_coverage,
                    "incidence_eligibility": eligibility,
                    "incidence_exact": incidence_exact,
                    "group_coverage": group_coverage,
                    "type_coverage": type_coverage,
                    "representable": bond_coverage
                    and group_coverage
                    and type_coverage,
                    "target_atoms": target_atom_count,
                    "matched_atoms": int(
                        proposals["atom_target_matched"][batch_index].sum()
                    ),
                    "target_bonds": target_bond_count,
                    "matched_bonds": int(
                        proposals["bond_target_matched"][batch_index].sum()
                    ),
                    "eligible_bonds": int(
                        proposals["incidence_target_eligible"][batch_index].sum()
                    ),
                    "positive_groups": int(groups["targets"][sample].sum()),
                    "typed_positive_groups": int(
                        (groups["targets"][sample] & groups["type_mask"][sample]).sum()
                    ),
                }
            )
    return details


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Style proposal audit requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
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
    records, input_audit = load_style_records(
        args.shard,
        split=args.split,
        samples=args.samples,
        config=scene_config,
    )
    rows = collect_rows(
        vision=vision,
        atom=atom,
        bond=bond,
        set_head=set_head,
        records=records,
        collator=AtomBondSceneCollator(scene_config),
        proposal_config=proposal_config,
        device=device,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    details_output = output.with_name(f"{output.stem}-samples.jsonl")
    details_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "style-proposal-bottleneck-audit-0.1",
        "source_checkpoint": args.source_checkpoint,
        "proposal_config": source["proposal_config"],
        "input_audit": input_audit,
        "summary": summarize_stage_rows(rows),
        "sample_details": str(details_output),
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "overall": result["summary"]["overall"],
                "stress_operation": result["summary"]["breakdowns"][
                    "stress_operation"
                ],
                "paired": result["summary"]["paired"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
