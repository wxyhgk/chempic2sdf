#!/usr/bin/env python3
"""Pack a structure MoE as one base checkpoint plus a small expert overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.structure_gate import STRUCTURE_HEAD_PREFIXES, extract_structure_overlay


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--expert-checkpoint", required=True)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_checkpoint)
    expert_path = Path(args.expert_checkpoint)
    gate_path = Path(args.gate_checkpoint)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    expert = torch.load(expert_path, map_location="cpu", weights_only=False)
    gate = torch.load(gate_path, map_location="cpu", weights_only=False)
    if base["config"] != expert["config"]:
        raise ValueError("Base and expert configs differ")
    if Path(gate["base_checkpoint"]).resolve() != base_path.resolve():
        raise ValueError("Gate base checkpoint does not match")
    if Path(gate["expert_checkpoint"]).resolve() != expert_path.resolve():
        raise ValueError("Gate expert checkpoint does not match")

    overlay = extract_structure_overlay(base["model"], expert["model"])
    metadata = {
        "format": "chemvector-structure-moe-0.1",
        "base_checkpoint": str(base_path),
        "base_sha256": sha256_file(base_path),
        "expert_checkpoint": str(expert_path),
        "expert_sha256": sha256_file(expert_path),
        "gate_checkpoint": str(gate_path),
        "gate_sha256": sha256_file(gate_path),
        "structure_prefixes": list(STRUCTURE_HEAD_PREFIXES),
        "changed_tensors": len(overlay),
        "changed_parameters": sum(value.numel() for value in overlay.values()),
    }
    payload = {
        **gate,
        **metadata,
        "expert_structure_tensors": overlay,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "last.pt"
    torch.save(payload, output_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
