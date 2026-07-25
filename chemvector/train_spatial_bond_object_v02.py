#!/usr/bin/env python3
"""Train spatial BondObject proposals against ChemScene v0.2 visible ports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import time

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig, unordered_port_cost
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    decode_dense_bond_objects,
    dense_bond_object_loss,
)
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train_atom_bond_scene import (
    AtomBondSceneCollator,
    DEFAULT_SHARD,
    DEFAULT_VISION,
    stack_targets,
)


DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0945-spatial-bond-object-v02-capacity100"
)
DEFAULT_INIT = (
    "outputs/checkpoints/chemvector-v082-line-object-9k-5epoch/best.pt"
)


def load_matching_vision(
    grounded_graph_checkpoint: Path, device: torch.device
):
    _, graph_payload = load_grounded_graph_checkpoint(
        grounded_graph_checkpoint, device
    )
    base_model, _, _, selector_payload = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    return base_model.vision_encoder, base_model.config, {
        "grounded_graph_checkpoint": str(grounded_graph_checkpoint),
        "grounded_selector_checkpoint": graph_payload[
            "grounded_selector_checkpoint"
        ],
        "query_checkpoint": selector_payload["query_checkpoint"],
    }


def expected_bond_endpoints(
    targets: dict[str, torch.Tensor], geometry: str
) -> torch.Tensor:
    if geometry == "visible_ports":
        return targets["bond_ports"]
    if geometry != "atom_anchors":
        raise ValueError(f"Unknown BondObject target geometry: {geometry}")
    batch = torch.arange(
        targets["bond_atom_refs"].size(0),
        device=targets["bond_atom_refs"].device,
    )[:, None, None]
    return targets["atom_anchors"][batch, targets["bond_atom_refs"]]


def bond_targets(
    targets: dict[str, torch.Tensor], geometry: str = "visible_ports"
) -> dict[str, torch.Tensor]:
    batch, bonds = targets["bond_mask"].shape
    return {
        "endpoints": expected_bond_endpoints(targets, geometry),
        "bbox": torch.zeros(
            (batch, bonds, 4),
            device=targets["bond_ports"].device,
            dtype=targets["bond_ports"].dtype,
        ),
        "display": targets["bond_display"],
        "chemical": targets["bond_chemical"],
        "direction": targets["bond_direction"],
        "mask": targets["bond_mask"],
    }


def _compare(
    objects: list[dict],
    targets: dict[str, torch.Tensor],
    batch_index: int,
    target_geometry: str,
) -> dict[str, int | bool | float]:
    count = int(targets["bond_mask"][batch_index].sum())
    if len(objects) != count:
        return {
            "objects": count,
            "port_pair_correct": 0,
            "display_correct": 0,
            "chemical_correct": 0,
            "port_error_sum": 2.0 * 768.0 * count,
            "exact": False,
        }
    predicted = torch.stack([item["endpoints"] for item in objects])
    expected = expected_bond_endpoints(targets, target_geometry)[
        batch_index, :count
    ]
    rows, columns = linear_sum_assignment(
        unordered_port_cost(predicted, expected).float().cpu().numpy()
    )
    rows = torch.as_tensor(rows, device=predicted.device, dtype=torch.long)
    columns = torch.as_tensor(columns, device=predicted.device, dtype=torch.long)
    predicted = predicted[rows]
    expected = expected[columns]
    forward = torch.linalg.vector_norm((predicted - expected) * 768.0, dim=-1)
    reverse = torch.linalg.vector_norm(
        (predicted - expected.flip(1)) * 768.0, dim=-1
    )
    use_reverse = reverse.sum(-1) < forward.sum(-1)
    errors = torch.where(use_reverse[:, None], reverse, forward)
    display = torch.tensor(
        [objects[int(row)]["display"] for row in rows],
        device=predicted.device,
    )
    chemical = torch.tensor(
        [objects[int(row)]["chemical"] for row in rows],
        device=predicted.device,
    )
    expected_display = targets["bond_display"][batch_index, columns]
    expected_chemical = targets["bond_chemical"][batch_index, columns]
    port_pair_correct = (errors <= 6.0).all(dim=1)
    display_correct = display == expected_display
    chemical_correct = chemical == expected_chemical
    exact = bool(
        port_pair_correct.all()
        and display_correct.all()
        and chemical_correct.all()
    )
    return {
        "objects": count,
        "port_pair_correct": int(port_pair_correct.sum()),
        "display_correct": int(display_correct.sum()),
        "chemical_correct": int(chemical_correct.sum()),
        "port_error_sum": float(errors.sum()),
        "exact": exact,
    }


@torch.no_grad()
def evaluate(
    vision,
    head: DenseBondObjectDetector,
    records: list[dict],
    collator: AtomBondSceneCollator,
    device: torch.device,
    batch_size: int,
    threshold: float,
    target_geometry: str,
) -> dict:
    vision.eval()
    head.eval()
    totals = {
        "samples": 0,
        "objects": 0,
        "port_pair_correct": 0,
        "display_correct": 0,
        "chemical_correct": 0,
        "port_error_sum": 0.0,
        "known_exact": 0,
        "predicted_count_exact": 0,
        "predicted_exact": 0,
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            output = head(features)
        ranked = decode_dense_bond_objects(
            output, threshold=0.0, maximum_objects=128
        )
        for batch_index, objects in enumerate(ranked):
            count = int(targets["bond_mask"][batch_index].sum())
            known = objects[:count]
            predicted = [item for item in objects if item["score"] >= threshold]
            known_result = _compare(
                known, targets, batch_index, target_geometry
            )
            predicted_result = _compare(
                predicted, targets, batch_index, target_geometry
            )
            totals["samples"] += 1
            totals["known_exact"] += int(known_result["exact"])
            totals["predicted_count_exact"] += int(len(predicted) == count)
            totals["predicted_exact"] += int(predicted_result["exact"])
            for name in (
                "objects",
                "port_pair_correct",
                "display_correct",
                "chemical_correct",
                "port_error_sum",
            ):
                totals[name] += known_result[name]
    samples = max(totals["samples"], 1)
    objects = max(totals["objects"], 1)
    return {
        "samples": totals["samples"],
        "known_exact": totals["known_exact"] / samples,
        "predicted_count_exact": totals["predicted_count_exact"] / samples,
        "predicted_exact": totals["predicted_exact"] / samples,
        "port_pair_accuracy_6px": totals["port_pair_correct"] / objects,
        "display_accuracy": totals["display_correct"] / objects,
        "chemical_accuracy": totals["chemical_correct"] / objects,
        "mean_endpoint_error_px": totals["port_error_sum"] / (2 * objects),
        "threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", default=DEFAULT_SHARD)
    parser.add_argument("--vision-checkpoint", default=DEFAULT_VISION)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--grounded-graph-checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--target-geometry",
        choices=("visible_ports", "atom_anchors"),
        default="visible_ports",
    )
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Spatial BondObject capacity training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    initialization = None
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        graph_checkpoint = Path(
            args.grounded_graph_checkpoint
            or initialization["grounded_graph_checkpoint"]
        )
        vision, vision_config, vision_payload = load_matching_vision(
            graph_checkpoint, device
        )
        config = DenseBondObjectConfig(**initialization["config"])
    else:
        from chemvector.train_spatial_atom_object import load_vision

        vision, vision_config, vision_payload = load_vision(
            Path(args.vision_checkpoint), device
        )
        config = DenseBondObjectConfig(
            input_channels=vision_config.vision_dims[0],
            hidden_channels=128,
            slots=3,
            max_endpoint_offset=0.5,
        )
    head = DenseBondObjectDetector(config).to(device)
    if initialization is not None:
        head.load_state_dict(initialization["dense_bond_object"], strict=True)
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
        raise ValueError("Spatial BondObject capacity training may only read train samples")
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
    running = 0.0
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        head.train()
        selected = [records[rng.randrange(len(records))] for _ in range(args.batch_size)]
        batch = collator(selected)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = head(features)
            losses = dense_bond_object_loss(
                output,
                bond_targets(targets, args.target_geometry),
                endpoint_barrier_weight=0.02,
                endpoint_barrier_margin_pixels=4.0,
                heatmap_set_weight=1.0,
                heatmap_set_margin=1.0,
                slot_matching="hungarian",
                cell_count_weight=0.1,
                duplicate_weight=0.05,
                duplicate_sigma_pixels=8.0,
            )
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
                vision,
                head,
                records,
                collator,
                device,
                args.batch_size,
                args.threshold,
                args.target_geometry,
            )
            metrics["step"] = step
            metrics["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "spatial-bond-object-v02-checkpoint-0.1",
                "head": head.state_dict(),
                "config": asdict(config),
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "validation": metrics,
                "source_vision_checkpoint": args.vision_checkpoint,
                "source_vision": vision_payload,
                "init_checkpoint": args.init_checkpoint,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = (
                metrics["known_exact"],
                metrics["port_pair_accuracy_6px"],
                metrics["display_accuracy"],
                metrics["chemical_accuracy"],
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
