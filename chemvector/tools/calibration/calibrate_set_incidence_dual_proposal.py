#!/usr/bin/env python3
"""Calibrate a dual-proposal set/incidence update in weight space."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.dense_bond_object import BondProposalResidualAdapter
from chemvector.evaluate_style_sensitivity import load_style_records
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_atom_pair_bond_group import VALIDATION_DOMAINS
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_set_incidence_dual_proposal import (
    DEFAULT_SOURCE,
    DEFAULT_STYLE_ROOT,
    _domain_safety_deltas,
)
from chemvector.train_spatial_incidence import DEFAULT_GRAPH
from chemvector.train_stable_bond_presence import domain_validation


DEFAULT_TRAINED = (
    "outputs/checkpoints/chemvector-v0996-set-incidence-dual-proposal-"
    "teacher10-pilot250/epoch-1.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0996-set-incidence-dual-proposal-"
    "teacher10-scaled"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--trained-checkpoint", default=DEFAULT_TRAINED)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--style-root", default=str(DEFAULT_STYLE_ROOT))
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--style-validation-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def interpolated_state(
    base: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= scale <= 1.0:
        raise ValueError("Calibration scale must be in [0, 1]")
    if base.keys() != trained.keys():
        raise ValueError("Set head checkpoint keys differ")
    result = {}
    for name, base_value in base.items():
        trained_value = trained[name]
        if base_value.shape != trained_value.shape:
            raise ValueError(f"Set head tensor shape differs: {name}")
        result[name] = (
            base_value + scale * (trained_value - base_value)
            if base_value.is_floating_point()
            else base_value.clone()
        )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Dual-proposal scale calibration requires CUDA")
    scales = sorted(set(args.scales))
    if not scales or any(not 0.0 < scale <= 1.0 for scale in scales):
        raise ValueError("Nonzero calibration scales must be in (0, 1]")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    vision, atom, candidate_bond, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    if not isinstance(candidate_bond, BondProposalResidualAdapter):
        raise ValueError("Source checkpoint does not contain a Bond proposal residual")
    original_bond = candidate_bond.base_detector
    for module in (vision, atom, candidate_bond, original_bond):
        module.eval().requires_grad_(False)
    trained = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    base_state = {name: value.detach().cpu() for name, value in set_head.state_dict().items()}
    trained_state = trained["set_head"]

    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    official_records = {}
    for domain, shard in VALIDATION_DOMAINS.items():
        official_records[domain] = load_validation_records(
            [shard],
            per_shard=args.validation_samples_per_domain,
            config=scene_config,
        )[0]
    style_records = load_style_records(
        str(Path(args.style_root) / "shards" / "validation-000000.tar"),
        split="validation",
        samples=args.style_validation_samples,
        config=scene_config,
    )[0]
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    bond_threshold = float(source["training_args"]["bond_decision_threshold"])

    def evaluate(bond, records) -> dict:
        return domain_validation(
            vision,
            atom,
            bond,
            set_head,
            records,
            collator,
            proposal_config,
            device,
            args.batch_size,
            atom_threshold,
            bond_threshold,
        )

    set_head.load_state_dict(base_state, strict=True)
    set_head.eval()
    baseline_candidate = evaluate(candidate_bond, official_records)
    baseline_original = evaluate(original_bond, official_records)
    baseline_style = evaluate(candidate_bond, {"style": style_records})
    rows = [
        {
            "scale": 0.0,
            "safe": True,
            "candidate_official": baseline_candidate,
            "original_official": baseline_original,
            "style": baseline_style,
        }
    ]
    best_scale = 0.0
    best_style = baseline_style["domains"]["style"]["full_object_exact"]

    for scale in scales:
        state = interpolated_state(base_state, trained_state, scale)
        set_head.load_state_dict(state, strict=True)
        set_head.eval()
        candidate = evaluate(candidate_bond, official_records)
        original = evaluate(original_bond, official_records)
        deltas = _domain_safety_deltas(candidate, baseline_candidate)
        deltas.extend(_domain_safety_deltas(original, baseline_original))
        safe = min(deltas) >= 0.0
        style = evaluate(candidate_bond, {"style": style_records}) if safe else None
        row = {
            "scale": scale,
            "safe": safe,
            "minimum_safety_delta": min(deltas),
            "candidate_official": candidate,
            "original_official": original,
            "style": style,
        }
        rows.append(row)
        if safe:
            style_exact = style["domains"]["style"]["full_object_exact"]
            if style_exact > best_style:
                best_style = style_exact
                best_scale = scale
                checkpoint = {
                    **{
                        key: value
                        for key, value in source.items()
                        if key not in {"optimizer", "validation", "samples_seen", "set_head"}
                    },
                    "checkpoint_schema_version": "joint-proposal-objects-checkpoint-0.8",
                    "set_head": state,
                    "set_config": {
                        "incidence": asdict(set_head.config.incidence),
                        "hidden_dim": set_head.config.hidden_dim,
                        "stable_bond_hidden_dim": set_head.config.stable_bond_hidden_dim,
                    },
                    "samples_seen": int(source["samples_seen"]),
                    "validation": candidate,
                    "original_proposal_validation": original,
                    "style_validation": style,
                    "source_joint_checkpoint": args.source_checkpoint,
                    "set_realign_checkpoint": args.trained_checkpoint,
                    "set_realign_scale": scale,
                    "set_realign_scale_args": vars(args),
                }
                torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {"best_scale": best_scale, "best_style_full_exact": best_style, "rows": rows},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "scale": scale,
                    "safe": safe,
                    "minimum_safety_delta": min(deltas),
                    "style_full_exact": (
                        style["domains"]["style"]["full_object_exact"] if style else None
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
