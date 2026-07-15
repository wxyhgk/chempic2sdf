#!/usr/bin/env python3
"""Interpolate OCR-only checkpoints while preserving every non-OCR tensor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import shutil

import torch

from chemvector.model import ChemVectorConfig


def interpolate_state_dicts(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if first.keys() != second.keys():
        missing = sorted(first.keys() - second.keys())
        added = sorted(second.keys() - first.keys())
        raise ValueError(f"Checkpoint keys differ: missing={missing}, added={added}")
    result = {}
    for key, first_value in first.items():
        second_value = second[key]
        if first_value.shape != second_value.shape:
            raise ValueError(f"Tensor shape differs for {key}")
        if not key.startswith("ocr_head."):
            if not torch.equal(first_value, second_value):
                raise ValueError(f"Non-OCR tensor changed between checkpoints: {key}")
            result[key] = first_value
        elif first_value.is_floating_point():
            result[key] = torch.lerp(first_value, second_value, alpha)
        else:
            result[key] = second_value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    first_path = Path(args.first)
    second_path = Path(args.second)
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    second = torch.load(second_path, map_location="cpu", weights_only=False)
    valid_keys = {item.name for item in fields(ChemVectorConfig)}
    first_config = asdict(
        ChemVectorConfig(
            **{key: value for key, value in first["config"].items() if key in valid_keys}
        )
    )
    second_config = asdict(
        ChemVectorConfig(
            **{key: value for key, value in second["config"].items() if key in valid_keys}
        )
    )
    if first_config != second_config:
        raise ValueError("OCR checkpoint configs differ")
    model = interpolate_state_dicts(first["model"], second["model"], args.alpha)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "last.pt"
    torch.save(
        {
            "model": model,
            "config": second_config,
            "global_step": second.get("global_step", 0),
            "training_kind": "interpolated_pixel_ocr",
            "interpolation": {
                "first": str(first_path),
                "second": str(second_path),
                "alpha": args.alpha,
            },
        },
        output_path,
    )
    for source in (first_path.parent, second_path.parent):
        tokenizer = source / "tokenizer.json"
        if tokenizer.is_file():
            shutil.copy2(tokenizer, output_dir / "tokenizer.json")
            break
    (output_dir / "interpolation.json").write_text(
        json.dumps(
            {
                "first": str(first_path),
                "second": str(second_path),
                "alpha": args.alpha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
