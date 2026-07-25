#!/usr/bin/env python3
"""Export a combined joint proposal checkpoint into legacy component files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.benchmark_joint_mr_tadf import _load_set_head
from chemvector.train_spatial_incidence import load_frozen_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.checkpoint)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not str(source.get("checkpoint_schema_version", "")).startswith(
        "joint-proposal-objects-checkpoint-"
    ):
        raise ValueError("Source is not a joint proposal-object checkpoint")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    training_args = source["training_args"]
    graph_checkpoint = training_args["grounded_graph_checkpoint"]
    step = int(source["samples_seen"])
    validation = source.get("validation", {})

    atom_path = output / "atom.pt"
    torch.save(
        {
            "checkpoint_schema_version": "spatial-atom-object-checkpoint-0.3",
            "head": source["atom_head"],
            "config": source["atom_config"],
            "global_step": step,
            "validation": validation,
            "source_joint_checkpoint": str(source_path),
        },
        atom_path,
    )
    bond_path = output / "bond.pt"
    torch.save(
        {
            "checkpoint_schema_version": "dense-bond-object-checkpoint-0.3",
            "dense_bond_object": source["bond_head"],
            "config": source["bond_config"],
            "global_step": step,
            "samples_seen": step,
            "validation": validation,
            "grounded_graph_checkpoint": graph_checkpoint,
            "source_joint_checkpoint": str(source_path),
        },
        bond_path,
    )
    single_bond_path = None
    if source.get("bond_multiscale"):
        single_state = {
            name.removeprefix("base."): value
            for name, value in source["bond_head"].items()
            if name.startswith("base.")
        }
        if not single_state:
            raise ValueError("Multiscale Bond checkpoint contains no base state")
        single_bond_path = output / "bond-single.pt"
        torch.save(
            {
                "checkpoint_schema_version": "dense-bond-object-checkpoint-0.3",
                "dense_bond_object": single_state,
                "config": source["bond_config"],
                "global_step": step,
                "samples_seen": step,
                "validation": validation,
                "grounded_graph_checkpoint": graph_checkpoint,
                "source_joint_checkpoint": str(source_path),
                "source_component": "multiscale.base",
            },
            single_bond_path,
        )
    set_path = output / "set.pt"
    torch.save(
        {
            "checkpoint_schema_version": "proposal-object-set-checkpoint-0.2",
            "head": source["set_head"],
            "config": source["set_config"],
            "proposal_config": source["proposal_config"],
            "global_step": step,
            "evaluation": {"validation": validation},
            "source_joint_checkpoint": str(source_path),
        },
        set_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_frozen_components(
        Path(graph_checkpoint), atom_path, bond_path, device
    )
    _load_set_head(str(set_path), device)
    manifest = {
        "source_checkpoint": str(source_path),
        "samples_seen": step,
        "graph_checkpoint": graph_checkpoint,
        "atom_checkpoint": str(atom_path),
        "bond_checkpoint": str(bond_path),
        "single_bond_checkpoint": (
            str(single_bond_path) if single_bond_path is not None else None
        ),
        "set_checkpoint": str(set_path),
        "validation": validation,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
