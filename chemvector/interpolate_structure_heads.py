#!/usr/bin/env python3
"""Interpolate structure-head checkpoints while preserving the model backbone."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import shutil

import torch

from chemvector.model import ChemVectorConfig


ALLOWED_PREFIXES = (
    "vertex_head.",
    "relation_head.",
    "stroke_head.",
)


def interpolate_state_dicts(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Interpolate changed structure-head tensors and reject every other change."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if first.keys() != second.keys():
        missing = sorted(first.keys() - second.keys())
        added = sorted(second.keys() - first.keys())
        raise ValueError(f"Checkpoint keys differ: missing={missing}, added={added}")

    result = {}
    changed_keys = []
    for key, first_value in first.items():
        second_value = second[key]
        if first_value.shape != second_value.shape:
            raise ValueError(f"Tensor shape differs for {key}")
        changed = not torch.equal(first_value, second_value)
        allowed = key.startswith(ALLOWED_PREFIXES)
        if changed and not allowed:
            raise ValueError(f"Non-structure-head tensor changed: {key}")
        if changed and not first_value.is_floating_point():
            raise ValueError(f"Changed structure-head tensor is not floating point: {key}")
        if changed:
            result[key] = torch.lerp(first_value, second_value, alpha)
            changed_keys.append(key)
        else:
            result[key] = first_value
    return result, changed_keys


def normalized_config(checkpoint: dict) -> dict:
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    return asdict(
        ChemVectorConfig(
            **{
                key: value
                for key, value in checkpoint["config"].items()
                if key in valid_keys
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="Baseline checkpoint")
    parser.add_argument("--second", required=True, help="Adapted checkpoint")
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    first_path = Path(args.first)
    second_path = Path(args.second)
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    second = torch.load(second_path, map_location="cpu", weights_only=False)
    first_config = normalized_config(first)
    second_config = normalized_config(second)
    if first_config != second_config:
        raise ValueError("Structure-head checkpoint configs differ")

    model, changed_keys = interpolate_state_dicts(
        first["model"], second["model"], args.alpha
    )
    metadata = {
        "first": str(first_path),
        "second": str(second_path),
        "alpha": args.alpha,
        "allowed_prefixes": list(ALLOWED_PREFIXES),
        "changed_tensors": len(changed_keys),
        "changed_parameters": sum(first["model"][key].numel() for key in changed_keys),
        "changed_keys": changed_keys,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "last.pt"
    torch.save(
        {
            "model": model,
            "config": first_config,
            "global_step": second.get("global_step", 0),
            "training_kind": "interpolated_structure_heads",
            "interpolation": metadata,
        },
        output_path,
    )
    for source in (first_path.parent, second_path.parent):
        tokenizer = source / "tokenizer.json"
        if tokenizer.is_file():
            shutil.copy2(tokenizer, output_dir / "tokenizer.json")
            break
    (output_dir / "interpolation.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
