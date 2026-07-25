#!/usr/bin/env python3
"""Train the v0.9.44 spatial complete AtomObject capacity head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from itertools import islice
import json
from pathlib import Path
import random
import time

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.model import ChemVectorConfig, ChemVisionEncoder
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
    decode_spatial_atom_objects,
    spatial_atom_object_loss,
)
from chemvector.train_atom_bond_scene import (
    AtomBondSceneCollator,
    DEFAULT_SHARD,
    DEFAULT_VISION,
    stack_targets,
)


DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0944-spatial-atom-object-capacity100"
)


def load_vision(
    path: Path, device: torch.device
) -> tuple[ChemVisionEncoder, ChemVectorConfig, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    valid = {field.name for field in fields(ChemVectorConfig)}
    config = ChemVectorConfig(
        **{
            name: value
            for name, value in payload["vision_config"].items()
            if name in valid
        }
    )
    vision = ChemVisionEncoder(config)
    state = {
        name.removeprefix("vision_encoder."): value
        for name, value in payload["model"].items()
        if name.startswith("vision_encoder.")
    }
    vision.load_state_dict(state, strict=True)
    vision.requires_grad_(False)
    return vision.to(device).eval(), config, payload


def load_grounded_vision(
    path: Path, device: torch.device
) -> tuple[ChemVisionEncoder, ChemVectorConfig, dict]:
    _, graph_payload = load_grounded_graph_checkpoint(path, device)
    base_model, _, _, selector_payload = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    return base_model.vision_encoder, base_model.config, {
        "grounded_graph_checkpoint": str(path),
        "grounded_selector_checkpoint": graph_payload[
            "grounded_selector_checkpoint"
        ],
        "query_checkpoint": selector_payload["query_checkpoint"],
    }


def _compare(
    decoded: list[dict], targets: dict[str, torch.Tensor], batch_index: int
) -> dict[str, int | bool | float]:
    count = int(targets["atom_mask"][batch_index].sum())
    if len(decoded) != count:
        return {
            "objects": count,
            "anchor_correct": 0,
            "role_correct": 0,
            "text_objects": int(targets["atom_text_mask"][batch_index].sum()),
            "text_center_correct": 0,
            "text_extent_correct": 0,
            "anchor_error_sum": 768.0 * count,
            "exact": False,
        }
    predicted = torch.stack([item["anchor"] for item in decoded])
    expected = targets["atom_anchors"][batch_index, :count]
    rows, columns = linear_sum_assignment(
        torch.cdist(predicted.float(), expected.float(), p=1).cpu().numpy()
    )
    rows = torch.as_tensor(rows, device=predicted.device, dtype=torch.long)
    columns = torch.as_tensor(columns, device=predicted.device, dtype=torch.long)
    errors = torch.linalg.vector_norm(
        (predicted[rows] - expected[columns]) * 768.0, dim=-1
    )
    predicted_roles = torch.tensor(
        [decoded[int(row)]["role"] for row in rows],
        device=predicted.device,
    )
    expected_roles = targets["atom_roles"][batch_index, columns]
    text = targets["atom_text_mask"][batch_index, columns]
    text_centers = torch.stack([decoded[int(row)]["text_center"] for row in rows])
    text_extents = torch.stack([decoded[int(row)]["text_extent"] for row in rows])
    expected_centers = targets["atom_text_center"][batch_index, columns]
    expected_extents = targets["atom_text_bbox"][batch_index, columns, 2:] / 2.0
    center_errors = torch.linalg.vector_norm(
        (text_centers - expected_centers) * 768.0, dim=-1
    )
    extent_errors = (text_extents - expected_extents).abs().amax(dim=-1) * 768.0
    anchor_correct = errors <= 6.0
    role_correct = predicted_roles == expected_roles
    text_center_correct = (center_errors <= 6.0) & text
    text_extent_correct = (extent_errors <= 6.0) & text
    exact = bool(
        anchor_correct.all()
        and role_correct.all()
        and text_center_correct.sum() == text.sum()
        and text_extent_correct.sum() == text.sum()
    )
    return {
        "objects": count,
        "anchor_correct": int(anchor_correct.sum()),
        "role_correct": int(role_correct.sum()),
        "text_objects": int(text.sum()),
        "text_center_correct": int(text_center_correct.sum()),
        "text_extent_correct": int(text_extent_correct.sum()),
        "anchor_error_sum": float(errors.sum()),
        "exact": exact,
    }


@torch.no_grad()
def evaluate(
    vision: ChemVisionEncoder,
    head: SpatialAtomObjectDetector,
    records: list[dict],
    collator: AtomBondSceneCollator,
    device: torch.device,
    batch_size: int,
) -> dict:
    vision.eval()
    head.eval()
    totals = {
        "samples": 0,
        "count_exact": 0,
        "known_exact": 0,
        "predicted_exact": 0,
        "objects": 0,
        "anchor_correct": 0,
        "role_correct": 0,
        "text_objects": 0,
        "text_center_correct": 0,
        "text_extent_correct": 0,
        "anchor_error_sum": 0.0,
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            output = head(features)
        true_counts = targets["atom_mask"].sum(dim=1).long()
        predicted_counts = output["count_logits"].argmax(dim=1)
        known = decode_spatial_atom_objects(output, true_counts)
        predicted = decode_spatial_atom_objects(output, predicted_counts)
        for batch_index in range(len(pixels)):
            known_result = _compare(known[batch_index], targets, batch_index)
            predicted_result = _compare(predicted[batch_index], targets, batch_index)
            totals["samples"] += 1
            totals["count_exact"] += int(
                predicted_counts[batch_index] == true_counts[batch_index]
            )
            totals["known_exact"] += int(known_result["exact"])
            totals["predicted_exact"] += int(predicted_result["exact"])
            for name in (
                "objects",
                "anchor_correct",
                "role_correct",
                "text_objects",
                "text_center_correct",
                "text_extent_correct",
                "anchor_error_sum",
            ):
                totals[name] += known_result[name]
    samples = max(totals["samples"], 1)
    objects = max(totals["objects"], 1)
    text_objects = max(totals["text_objects"], 1)
    return {
        "samples": totals["samples"],
        "count_exact": totals["count_exact"] / samples,
        "known_exact": totals["known_exact"] / samples,
        "predicted_exact": totals["predicted_exact"] / samples,
        "anchor_accuracy_6px": totals["anchor_correct"] / objects,
        "role_accuracy": totals["role_correct"] / objects,
        "anchor_mean_error_px": totals["anchor_error_sum"] / objects,
        "text_center_accuracy_6px": totals["text_center_correct"] / text_objects,
        "text_extent_accuracy_6px": totals["text_extent_correct"] / text_objects,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", default=DEFAULT_SHARD)
    parser.add_argument("--vision-checkpoint", default=DEFAULT_VISION)
    parser.add_argument("--grounded-graph-checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Spatial AtomObject capacity training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if args.grounded_graph_checkpoint:
        vision, vision_config, vision_payload = load_grounded_vision(
            Path(args.grounded_graph_checkpoint), device
        )
    else:
        vision, vision_config, vision_payload = load_vision(
            Path(args.vision_checkpoint), device
        )
    config = SpatialAtomObjectConfig(input_channels=vision_config.vision_dims[0])
    head = SpatialAtomObjectDetector(config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    records = list(
        islice(
            iter(
                ChemVectorIterableDataset(
                    [args.train_shard], sequence_version="0.4"
                )
            ),
            args.samples,
        )
    )
    if len(records) != args.samples:
        raise ValueError("Capacity shard contains fewer samples than requested")
    if any(
        sample.get("meta", {}).get("source_split") != "train"
        for sample in records
    ):
        raise ValueError("Spatial AtomObject capacity training may only read train samples")
    collator = AtomBondSceneCollator(AtomBondSceneConfig())
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rng = random.Random(args.seed)
    best = None
    started = time.perf_counter()
    running = 0.0
    for step in range(1, args.max_steps + 1):
        head.train()
        samples = [records[rng.randrange(len(records))] for _ in range(args.batch_size)]
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = head(features)
            losses = spatial_atom_object_loss(output, targets)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        running += float(losses["loss"].detach())
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {"step": step, "loss": running / args.log_every},
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running = 0.0
        if step % args.validate_every == 0 or step == args.max_steps:
            metrics = evaluate(
                vision, head, records, collator, device, args.batch_size
            )
            metrics["step"] = step
            metrics["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "spatial-atom-object-checkpoint-0.1",
                "head": head.state_dict(),
                "config": asdict(config),
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "validation": metrics,
                "source_vision_checkpoint": args.vision_checkpoint,
                "source_vision": vision_payload,
                "source_vision_global_step": vision_payload.get("global_step"),
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["anchor_accuracy_6px"],
                metrics["role_accuracy"],
                metrics["text_center_accuracy_6px"],
                metrics["text_extent_accuracy_6px"],
            )
            if best is None or score > best:
                best = score
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
