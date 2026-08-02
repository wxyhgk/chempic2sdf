#!/usr/bin/env python3
"""Audit graph-level rescues and breaks caused by a proposal-delta adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.dense_bond_object import BondProposalResidualAdapter
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_bond_group import TRAIN_DOMAINS, VALIDATION_DOMAINS
from chemvector.train_atom_pair_style_consistency import load_replay_records
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_proposal_delta_set_adapter import (
    _candidate_batch,
    load_proposal_delta_set_adapter,
)
from chemvector.train_proposal_object_set import _sample_set_metrics
from chemvector.train_spatial_incidence import DEFAULT_GRAPH
from chemvector.train_stable_bond_presence import valid_stream


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0997-proposal-delta-set-adapter-pilot250/"
    "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--samples-per-domain", type=int, default=498)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--exclude-replay-samples-per-domain", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def action_features(
    output: dict[str, torch.Tensor],
    proposals: dict[str, torch.Tensor],
    batch_index: int,
    *,
    bond_threshold: float,
) -> dict[str, float | int]:
    bond_mask = proposals["bond_mask"][batch_index]
    atom_mask = proposals["atom_mask"][batch_index]
    base_bond = output["proposal_delta_base_bond_presence_logits"][
        batch_index, bond_mask
    ].float()
    adjusted_bond = output["bond_presence_logits"][batch_index, bond_mask].float()
    presence_residual = output["proposal_delta_bond_presence_residual"][
        batch_index, bond_mask
    ].float()
    score_delta = output.get("proposal_delta_score")
    endpoint_delta = output.get("proposal_delta_endpoints")
    base_incidence = output["proposal_delta_base_incidence_logits"][batch_index][
        bond_mask
    ][:, :, atom_mask].float()
    adjusted_incidence = output["incidence_logits"][batch_index][bond_mask][
        :, :, atom_mask
    ].float()
    incidence_residual = output["proposal_delta_incidence_residual"][batch_index][
        bond_mask
    ][:, :, atom_mask].float()
    base_selected = base_bond >= bond_threshold
    adjusted_selected = adjusted_bond >= bond_threshold
    base_refs = base_incidence.argmax(-1)
    adjusted_refs = adjusted_incidence.argmax(-1)

    def mean_abs(values: torch.Tensor) -> float:
        return float(values.abs().mean()) if values.numel() else 0.0

    def max_abs(values: torch.Tensor) -> float:
        return float(values.abs().max()) if values.numel() else 0.0

    return {
        "bond_candidates": int(bond_mask.sum()),
        "atom_candidates": int(atom_mask.sum()),
        "bond_additions": int((~base_selected & adjusted_selected).sum()),
        "bond_deletions": int((base_selected & ~adjusted_selected).sum()),
        "incidence_endpoint_changes": int((base_refs != adjusted_refs).sum()),
        "presence_residual_mean_abs": mean_abs(presence_residual),
        "presence_residual_max_abs": max_abs(presence_residual),
        "incidence_residual_mean_abs": mean_abs(incidence_residual),
        "incidence_residual_max_abs": max_abs(incidence_residual),
        "base_bond_margin_min_abs": (
            float((base_bond - bond_threshold).abs().min()) if base_bond.numel() else 0.0
        ),
        "adjusted_bond_margin_min_abs": (
            float((adjusted_bond - bond_threshold).abs().min())
            if adjusted_bond.numel()
            else 0.0
        ),
        "score_delta_mean_abs": (
            mean_abs(score_delta[batch_index, bond_mask]) if score_delta is not None else 0.0
        ),
        "score_delta_max_abs": (
            max_abs(score_delta[batch_index, bond_mask]) if score_delta is not None else 0.0
        ),
        "endpoint_delta_mean_abs": (
            mean_abs(endpoint_delta[batch_index, bond_mask])
            if endpoint_delta is not None
            else 0.0
        ),
        "endpoint_delta_max_abs": (
            max_abs(endpoint_delta[batch_index, bond_mask])
            if endpoint_delta is not None
            else 0.0
        ),
    }


def outcome_name(base_exact: bool, adjusted_exact: bool) -> str:
    if base_exact and adjusted_exact:
        return "preserved_correct"
    if base_exact and not adjusted_exact:
        return "break"
    if not base_exact and adjusted_exact:
        return "rescue"
    return "preserved_wrong"


@torch.no_grad()
def collect_actions(
    *,
    records: dict[str, list[dict]],
    vision,
    atom,
    candidate_bond,
    original_bond,
    set_head,
    adapter,
    collator,
    proposal_config,
    device: torch.device,
    batch_size: int,
    atom_threshold: float,
    bond_threshold: float,
) -> list[dict]:
    rows = []
    for domain, samples in records.items():
        for start in range(0, len(samples), batch_size):
            batch_rows = samples[start : start + batch_size]
            output, proposals, targets = _candidate_batch(
                batch_rows,
                vision=vision,
                atom=atom,
                candidate_bond=candidate_bond,
                original_bond=original_bond,
                set_head=set_head,
                adapter=adapter,
                collator=collator,
                proposal_config=proposal_config,
                device=device,
            )
            base = {
                **output,
                "bond_presence_logits": output[
                    "proposal_delta_base_bond_presence_logits"
                ],
                "incidence_logits": output["proposal_delta_base_incidence_logits"],
            }
            for batch_index, sample in enumerate(batch_rows):
                base_metrics = _sample_set_metrics(
                    base,
                    proposals,
                    targets,
                    batch_index,
                    atom_threshold=atom_threshold,
                    bond_threshold=bond_threshold,
                )
                adjusted_metrics = _sample_set_metrics(
                    output,
                    proposals,
                    targets,
                    batch_index,
                    atom_threshold=atom_threshold,
                    bond_threshold=bond_threshold,
                )
                base_exact = bool(base_metrics["full_object_exact"])
                adjusted_exact = bool(adjusted_metrics["full_object_exact"])
                rows.append(
                    {
                        "domain": domain,
                        "sample_key": str(sample["key"]),
                        "molecule_id": str(sample["chemmol"]["molecule_id"]),
                        "variant": str(sample["render"]["variant"]),
                        "outcome": outcome_name(base_exact, adjusted_exact),
                        "base_full_exact": base_exact,
                        "adjusted_full_exact": adjusted_exact,
                        "base_bond_exact": bool(base_metrics["bond_deployed_exact"]),
                        "adjusted_bond_exact": bool(
                            adjusted_metrics["bond_deployed_exact"]
                        ),
                        "base_incidence_exact": bool(base_metrics["incidence_exact"]),
                        "adjusted_incidence_exact": bool(
                            adjusted_metrics["incidence_exact"]
                        ),
                        "features": action_features(
                            output,
                            proposals,
                            batch_index,
                            bond_threshold=bond_threshold,
                        ),
                    }
                )
    return rows


def summarize_actions(rows: list[dict]) -> dict:
    domains = {}
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        outcomes = Counter(row["outcome"] for row in selected)
        domains[domain] = {
            "samples": len(selected),
            "outcomes": dict(outcomes),
            "net_full_exact": outcomes["rescue"] - outcomes["break"],
            "action_samples": sum(
                row["features"]["bond_additions"]
                + row["features"]["bond_deletions"]
                + row["features"]["incidence_endpoint_changes"]
                > 0
                for row in selected
            ),
        }
    total = Counter(row["outcome"] for row in rows)
    return {
        "samples": len(rows),
        "outcomes": dict(total),
        "net_full_exact": total["rescue"] - total["break"],
        "domains": domains,
    }


def load_train_action_records(
    *,
    samples_per_domain: int,
    excluded_replay_per_domain: int,
    seed: int,
    config: AtomBondSceneConfig,
) -> tuple[dict[str, list[dict]], dict]:
    excluded, _ = load_replay_records(
        samples_per_domain=excluded_replay_per_domain,
        seed=20260726,
        config=config,
    )
    excluded_keys = {
        domain: {str(row["key"]) for row in rows} for domain, rows in excluded.items()
    }
    records = {}
    for offset, (domain, shards) in enumerate(TRAIN_DOMAINS.items()):
        stream = valid_stream(
            shards,
            config=config,
            seed=seed + offset,
            shuffle_buffer=2048,
        )
        rows = list(
            itertools.islice(
                (
                    row
                    for row in stream
                    if str(row["key"]) not in excluded_keys[domain]
                ),
                samples_per_domain,
            )
        )
        if len(rows) != samples_per_domain:
            raise RuntimeError(
                f"Train action domain {domain} yielded {len(rows)} of "
                f"{samples_per_domain} samples"
            )
        records[domain] = rows
    return records, {
        "samples_per_domain": samples_per_domain,
        "excluded_replay_per_domain": excluded_replay_per_domain,
        "excluded_keys": {domain: len(keys) for domain, keys in excluded_keys.items()},
        "selected_key_overlap": {
            domain: len({str(row["key"]) for row in rows} & excluded_keys[domain])
            for domain, rows in records.items()
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Proposal-delta action audit requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    adapter, adapter_payload = load_proposal_delta_set_adapter(args.checkpoint, device)
    vision, atom, candidate_bond, set_head, source = load_components(
        adapter_payload["source_checkpoint"], args.grounded_graph_checkpoint, device
    )
    if not isinstance(candidate_bond, BondProposalResidualAdapter):
        raise ValueError("Source checkpoint does not contain a Bond proposal residual")
    original_bond = candidate_bond.base_detector
    for module in (vision, atom, candidate_bond, original_bond, set_head, adapter):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    if args.split == "train":
        records, input_audit = load_train_action_records(
            samples_per_domain=args.samples_per_domain,
            excluded_replay_per_domain=args.exclude_replay_samples_per_domain,
            seed=args.seed,
            config=scene_config,
        )
    else:
        records = {
            domain: load_validation_records(
                [shard],
                per_shard=args.samples_per_domain,
                config=scene_config,
            )[0]
            for domain, shard in VALIDATION_DOMAINS.items()
        }
        input_audit = {"samples_per_domain": args.samples_per_domain}
    rows = collect_actions(
        records=records,
        vision=vision,
        atom=atom,
        candidate_bond=candidate_bond,
        original_bond=original_bond,
        set_head=set_head,
        adapter=adapter,
        collator=AtomBondSceneCollator(scene_config),
        proposal_config=proposal_config,
        device=device,
        batch_size=args.batch_size,
        atom_threshold=float(source["training_args"]["atom_decision_threshold"]),
        bond_threshold=float(source["training_args"]["bond_decision_threshold"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    details = output.with_name(f"{output.stem}-samples.jsonl")
    details.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "samples_per_domain": args.samples_per_domain,
        "input_audit": input_audit,
        "summary": summarize_actions(rows),
        "sample_details": str(details),
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
