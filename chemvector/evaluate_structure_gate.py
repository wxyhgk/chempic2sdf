#!/usr/bin/env python3
"""Measure conservative structure-gate routing on external image manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
import torch

from chemvector.evaluate import load_model
from chemvector.image_io import image_tensor, prepare_input_image
from chemvector.structure_gate import StructureStyleGate


def summarize_scores(
    scores: torch.Tensor,
    thresholds: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> dict:
    quantiles = torch.tensor((0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0))
    return {
        "samples": scores.numel(),
        "mean": float(scores.mean()) if scores.numel() else 0.0,
        "quantiles": {
            f"{float(quantile):.2f}": float(value)
            for quantile, value in zip(quantiles, torch.quantile(scores, quantiles))
        }
        if scores.numel()
        else {},
        "thresholds": {
            f"{threshold:.4f}": {
                "expert_selected": int((scores >= threshold).sum()),
                "expert_route_rate": float((scores >= threshold).float().mean())
                if scores.numel()
                else 0.0,
            }
            for threshold in thresholds
        },
    }


@torch.inference_mode()
def manifest_scores(
    vision,
    gate: StructureStyleGate,
    manifest_path: Path,
    device: torch.device,
    *,
    samples: int,
    batch_size: int,
    prepare_uspto: bool,
) -> torch.Tensor:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "status" in rows[0]:
        rows = [row for row in rows if row.get("status") == "ready"]
    rows = rows[:samples]
    root = manifest_path.parent
    scores = []
    for start in range(0, len(rows), batch_size):
        images = []
        for row in rows[start : start + batch_size]:
            image = Image.open(root / row["image_file"])
            if prepare_uspto:
                image = prepare_input_image(image, image_size=768, content_size=548)
            elif image.size != (768, 768):
                image = image.convert("RGB").resize((768, 768), Image.Resampling.BICUBIC)
            images.append(image_tensor(image))
        pixels = torch.stack(images).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            scores.append(gate(features).float().sigmoid().cpu())
    return torch.cat(scores) if scores else torch.empty(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--uspto-manifest", required=True)
    parser.add_argument("--mr-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gate evaluation")
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("Gate evaluation sizes must be positive")

    device = torch.device("cuda")
    payload = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
    model, _ = load_model(Path(payload["base_checkpoint"]), device)
    vision = model.vision_encoder.eval()
    gate = StructureStyleGate(
        payload["input_channels"],
        hidden_dim=payload["hidden_dim"],
    ).to(device)
    gate.load_state_dict(payload["gate"])
    gate.eval()
    result = {
        "gate_checkpoint": args.gate_checkpoint,
        "base_checkpoint": payload["base_checkpoint"],
        "expert_checkpoint": payload["expert_checkpoint"],
        "uspto": summarize_scores(
            manifest_scores(
                vision,
                gate,
                Path(args.uspto_manifest),
                device,
                samples=args.samples,
                batch_size=args.batch_size,
                prepare_uspto=True,
            )
        ),
        "mr_tadf": summarize_scores(
            manifest_scores(
                vision,
                gate,
                Path(args.mr_manifest),
                device,
                samples=args.samples,
                batch_size=args.batch_size,
                prepare_uspto=False,
            )
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
