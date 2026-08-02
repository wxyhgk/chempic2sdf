#!/usr/bin/env python3
"""Replay every P1-D checkpoint over fixed train5k order and blocks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1c_final import (
    _ranker_from_payload,
    _sha256,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
BLOCK_IMAGES = 1_000


def _empty_totals() -> dict[str, int]:
    return {
        "images": 0,
        "positive_count": 0,
        "matched_positive_count": 0,
        "whole_set_exact_images": 0,
        "boundary_exact_images": 0,
    }


def _add_totals(destination: dict[str, int], source: dict[str, int]) -> None:
    for name in destination:
        destination[name] += int(source[name])


def _rates(totals: dict[str, int]) -> dict:
    images = totals["images"]
    positives = totals["positive_count"]
    return {
        **totals,
        "hybrid_topm_recall": (
            totals["matched_positive_count"] / positives if positives else 0.0
        ),
        "hybrid_topm_whole_set_rate": (
            totals["whole_set_exact_images"] / images if images else 0.0
        ),
        "boundary_exact_rate": (
            totals["boundary_exact_images"] / images if images else 0.0
        ),
    }


@torch.no_grad()
def replay_checkpoint(
    atom_head,
    ranker,
    vision,
    manifest,
    collator,
    *,
    device: torch.device,
) -> dict:
    overall = _empty_totals()
    blocks = [_empty_totals() for _ in range(len(manifest) // BLOCK_IMAGES)]
    digest = hashlib.sha256()
    image_offset = 0
    atom_head.eval()
    ranker.eval()
    vision.eval()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        keys = batch["keys"]
        if len(keys) != 4 or image_offset // BLOCK_IMAGES != (
            image_offset + len(keys) - 1
        ) // BLOCK_IMAGES:
            raise RuntimeError("P1-D trajectory batch crosses a fixed block boundary")
        for key in keys:
            digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
            "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        }
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            ranked = ranker(candidates)
        hybrid = build_hybrid_top128_targets(candidates, targets)
        current = p1d.hybrid_topm_statistics(ranked, hybrid)
        _add_totals(overall, current)
        _add_totals(blocks[image_offset // BLOCK_IMAGES], current)
        image_offset += len(keys)
    if image_offset != len(manifest):
        raise RuntimeError("P1-D trajectory did not consume the complete manifest")
    return {
        "overall": _rates(overall),
        "blocks": [
            {
                "block": index + 1,
                "image_start": index * BLOCK_IMAGES,
                "image_stop": (index + 1) * BLOCK_IMAGES,
                **_rates(totals),
            }
            for index, totals in enumerate(blocks)
        ],
        "sample_keys_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=p1d.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("P1-D trajectory replay requires CUDA BF16")

    manifest, manifest_audit = p1a.load_train_manifest()
    if len(manifest) != 5_000 or len(manifest) % BLOCK_IMAGES:
        raise RuntimeError("P1-D trajectory requires fixed train5k blocks")
    device = torch.device(args.device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    checkpoints = []
    for step in CHECKPOINT_STEPS:
        path = output_dir / f"step-{step:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("global_step") != step:
            raise RuntimeError(f"P1-D trajectory checkpoint step differs: {path}")
        _, ranker = _ranker_from_payload(payload, device)
        result = replay_checkpoint(
            atom_head, ranker, vision, manifest, collator, device=device
        )
        if result["sample_keys_sha256"] != p1a.TRAIN_SEQUENCE_SHA256:
            raise RuntimeError("P1-D trajectory train sequence differs")
        checkpoints.append(
            {
                "step": step,
                "epoch": payload["epoch"],
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                **result,
            }
        )
        print(
            {
                "step": step,
                "boundary_exact_rate": result["overall"]["boundary_exact_rate"],
                "block_boundary_exact": [
                    block["boundary_exact_rate"] for block in result["blocks"]
                ],
            },
            flush=True,
        )
    report = {
        "schema_version": "g3-p1d-train-trajectory-0.1",
        "scope": "read-only-fixed-train5k-all-checkpoint-trajectory",
        "manifest": str(p1a.TRAIN_MANIFEST.resolve()),
        "manifest_sha256": manifest_audit["sha256"],
        "sample_keys_sha256": p1a.TRAIN_SEQUENCE_SHA256,
        "vision_checkpoint_sha256": vision_sha,
        "block_images": BLOCK_IMAGES,
        "checkpoints": checkpoints,
        "forbidden_data_read": False,
        "selection_or_training_effect": False,
    }
    p1d._atomic_json(output_dir / "train_checkpoint_trajectory.json", report)


if __name__ == "__main__":
    main()
