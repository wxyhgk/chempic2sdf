#!/usr/bin/env python3
"""Train GT-associated spatial Bond endpoint to AtomObject incidence."""

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

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
)
from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
)
from chemvector.spatial_incidence import (
    SpatialIncidenceConfig,
    SpatialIncidenceHead,
    spatial_incidence_loss,
)
from chemvector.train_atom_bond_scene import (
    AtomBondSceneCollator,
    DEFAULT_SHARD,
    stack_targets,
)
from chemvector.train_spatial_atom_object import load_grounded_vision


DEFAULT_GRAPH = (
    "outputs/checkpoints/chemvector-v0729-whole-set-stable-distance-overfit100/last.pt"
)
DEFAULT_ATOM = (
    "outputs/checkpoints/chemvector-v0945-shared-vision-spatial-atom-capacity100-step1300/"
    "best.pt"
)
DEFAULT_BOND = (
    "outputs/checkpoints/chemvector-v082-line-object-9k-5epoch/best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0945-spatial-incidence-capacity100"
)


def load_frozen_components(
    graph_checkpoint: Path,
    atom_checkpoint: Path,
    bond_checkpoint: Path,
    device: torch.device,
):
    vision, vision_config, vision_payload = load_grounded_vision(
        graph_checkpoint, device
    )
    atom_payload = torch.load(atom_checkpoint, map_location="cpu", weights_only=False)
    atom_config = SpatialAtomObjectConfig(**atom_payload["config"])
    if atom_config.input_channels != vision_config.vision_dims[0]:
        raise ValueError("AtomObject head and shared vision channels differ")
    atom_head = SpatialAtomObjectDetector(atom_config).to(device)
    atom_head.load_state_dict(atom_payload["head"], strict=True)
    bond_payload = torch.load(bond_checkpoint, map_location="cpu", weights_only=False)
    if Path(bond_payload["grounded_graph_checkpoint"]) != graph_checkpoint:
        raise ValueError("BondObject head was trained with a different vision pipeline")
    bond_config = DenseBondObjectConfig(**bond_payload["config"])
    bond_state = bond_payload["dense_bond_object"]
    bond_head = (
        MultiScaleDenseBondObjectDetector(bond_config).to(device)
        if any(key.startswith("base.") for key in bond_state)
        else DenseBondObjectDetector(bond_config).to(device)
    )
    bond_head.load_state_dict(bond_state, strict=True)
    for module in (vision, atom_head, bond_head):
        module.eval()
        module.requires_grad_(False)
    return vision, atom_head, bond_head, {
        "vision": vision_payload,
        "atom_validation": atom_payload.get("validation"),
        "bond_validation": bond_payload.get("validation"),
    }


def forward_frozen_bond_head(
    vision: torch.nn.Module,
    bond_head: torch.nn.Module,
    stroke_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not isinstance(bond_head, MultiScaleDenseBondObjectDetector):
        return bond_head(stroke_features)
    stride8 = vision.forward_high_resolution_from_stroke(stroke_features)
    stride16 = stride8
    for stage in vision.stages[3:5]:
        stride16 = stage(stride16)
    return bond_head(stroke_features, stride8, stride16)


def _target_bond_endpoints(targets: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = torch.arange(
        targets["bond_atom_refs"].size(0),
        device=targets["bond_atom_refs"].device,
    )[:, None, None]
    return targets["atom_anchors"][batch, targets["bond_atom_refs"]]


def extract_gt_associated_proposals(
    atom_output: dict[str, torch.Tensor],
    bond_output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size = targets["atom_mask"].size(0)
    maximum_atoms = targets["atom_mask"].size(1)
    maximum_bonds = targets["bond_mask"].size(1)
    device = targets["atom_anchors"].device
    dtype = atom_output["object_features"].dtype
    atom_feature_dim = atom_output["object_features"].size(1)
    bond_feature_dim = bond_output["object_features"].size(1)
    atom_features = torch.zeros(
        (batch_size, maximum_atoms, atom_feature_dim), device=device, dtype=dtype
    )
    atom_geometry = torch.zeros(
        (batch_size, maximum_atoms, 8), device=device, dtype=dtype
    )
    bond_features = torch.zeros(
        (batch_size, maximum_bonds, bond_feature_dim), device=device, dtype=dtype
    )
    bond_endpoints = torch.zeros(
        (batch_size, maximum_bonds, 2, 2), device=device, dtype=dtype
    )
    bond_types = torch.zeros(
        (batch_size, maximum_bonds, 8), device=device, dtype=dtype
    )
    target_refs = torch.zeros(
        (batch_size, maximum_bonds, 2), device=device, dtype=torch.long
    )
    target_endpoints = _target_bond_endpoints(targets)
    atom_height, atom_width = atom_output["heatmap_logits"].shape[-2:]
    bond_height, bond_width = bond_output["heatmap_logits"].shape[-2:]

    for batch_index in range(batch_size):
        atom_count = int(targets["atom_mask"][batch_index].sum())
        for atom_index in range(atom_count):
            target_anchor = targets["atom_anchors"][batch_index, atom_index]
            cell_x = min(atom_width - 1, int(float(target_anchor[0]) * atom_width))
            cell_y = min(atom_height - 1, int(float(target_anchor[1]) * atom_height))
            atom_features[batch_index, atom_index] = atom_output[
                "object_features"
            ][batch_index, :, cell_y, cell_x]
            predicted_anchor = atom_output["anchors"][
                batch_index, :, cell_y, cell_x
            ]
            role = atom_output["role_logits"][
                batch_index, :, cell_y, cell_x
            ].softmax(0)
            text_center = atom_output["text_centers"][
                batch_index, :, cell_y, cell_x
            ]
            text_extent = atom_output["text_extents"][
                batch_index, :, cell_y, cell_x
            ]
            atom_geometry[batch_index, atom_index] = torch.cat(
                (predicted_anchor, role, text_center, text_extent)
            )

        bond_count = int(targets["bond_mask"][batch_index].sum())
        cells: dict[tuple[int, int], list[int]] = {}
        for bond_index in range(bond_count):
            center = target_endpoints[batch_index, bond_index].mean(dim=0)
            cell_x = min(bond_width - 1, int(float(center[0]) * bond_width))
            cell_y = min(bond_height - 1, int(float(center[1]) * bond_height))
            cells.setdefault((cell_y, cell_x), []).append(bond_index)
        for (cell_y, cell_x), bond_indices in cells.items():
            predicted = bond_output["endpoints"][
                batch_index, :, :, :, cell_y, cell_x
            ].float()
            expected = target_endpoints[batch_index, bond_indices].float()
            forward = (predicted[:, None] - expected[None]).abs().sum(dim=(-1, -2))
            reverse = (
                predicted[:, None] - expected[None].flip(2)
            ).abs().sum(dim=(-1, -2))
            rows, columns = linear_sum_assignment(
                torch.minimum(forward, reverse).cpu().numpy()
            )
            slot_by_target = {
                bond_indices[column]: row for row, column in zip(rows, columns)
            }
            for bond_index in bond_indices:
                slot = slot_by_target[bond_index]
                endpoints = bond_output["endpoints"][
                    batch_index, slot, :, :, cell_y, cell_x
                ]
                bond_features[batch_index, bond_index] = bond_output[
                    "object_features"
                ][batch_index, :, cell_y, cell_x]
                bond_endpoints[batch_index, bond_index] = endpoints
                display = bond_output["display_logits"][
                    batch_index, slot, :, cell_y, cell_x
                ].softmax(0)
                chemical = bond_output["chemical_logits"][
                    batch_index, slot, :, cell_y, cell_x
                ].softmax(0)
                bond_types[batch_index, bond_index] = torch.cat(
                    (display, chemical)
                )
                expected_endpoints = target_endpoints[batch_index, bond_index]
                forward_error = (endpoints - expected_endpoints).abs().sum()
                reverse_error = (endpoints - expected_endpoints.flip(0)).abs().sum()
                refs = targets["bond_atom_refs"][batch_index, bond_index]
                target_refs[batch_index, bond_index] = (
                    refs.flip(0) if reverse_error < forward_error else refs
                )
    return {
        "atom_features": atom_features,
        "atom_geometry": atom_geometry,
        "atom_mask": targets["atom_mask"],
        "bond_features": bond_features,
        "bond_endpoints": bond_endpoints,
        "bond_types": bond_types,
        "bond_mask": targets["bond_mask"],
        "target_atom_refs": target_refs,
    }


@torch.no_grad()
def evaluate(
    vision,
    atom_head,
    bond_head,
    incidence_head,
    records,
    collator,
    device,
    batch_size: int,
) -> dict:
    incidence_head.eval()
    endpoints = bonds = images = 0
    endpoint_correct = bond_correct = image_correct = 0
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = bond_head(features)
            proposals = extract_gt_associated_proposals(
                atom_output, bond_output, targets
            )
            output = incidence_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_mask"],
            )
        for batch_index in range(len(pixels)):
            valid = proposals["bond_mask"][batch_index]
            predicted = output["incidence_logits"][batch_index, valid].argmax(-1)
            expected = proposals["target_atom_refs"][batch_index, valid]
            correct = predicted == expected
            endpoints += correct.numel()
            bonds += len(correct)
            images += 1
            endpoint_correct += int(correct.sum())
            bond_correct += int(correct.all(dim=1).sum())
            image_correct += int(bool(correct.all()))
    return {
        "samples": images,
        "endpoint_accuracy": endpoint_correct / max(endpoints, 1),
        "bond_accuracy": bond_correct / max(bonds, 1),
        "image_exact": image_correct / max(images, 1),
        "protocol": "gt-associated-proposal-capacity",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", default=DEFAULT_SHARD)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Spatial incidence capacity training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    vision, atom_head, bond_head, sources = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    config = SpatialIncidenceConfig()
    incidence_head = SpatialIncidenceHead(config).to(device)
    optimizer = torch.optim.AdamW(
        incidence_head.parameters(), lr=args.learning_rate, weight_decay=0.0
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
        raise ValueError("Spatial incidence capacity training may only read train samples")
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
    best = -1.0
    running = 0.0
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        incidence_head.train()
        selected = [records[rng.randrange(len(records))] for _ in range(args.batch_size)]
        batch = collator(selected)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            bond_output = bond_head(features)
            proposals = extract_gt_associated_proposals(
                atom_output, bond_output, targets
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = incidence_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_mask"],
            )
            losses = spatial_incidence_loss(
                output,
                proposals["target_atom_refs"],
                proposals["bond_mask"],
            )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(incidence_head.parameters(), 1.0)
        optimizer.step()
        running += float(losses["loss"].detach())
        if step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": running / args.log_every,
                        "endpoint_accuracy": float(losses["endpoint_accuracy"]),
                        "bond_accuracy": float(losses["bond_accuracy"]),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running = 0.0
        if step % args.validate_every == 0 or step == args.max_steps:
            metrics = evaluate(
                vision,
                atom_head,
                bond_head,
                incidence_head,
                records,
                collator,
                device,
                args.batch_size,
            )
            metrics["step"] = step
            metrics["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "spatial-incidence-checkpoint-0.1",
                "head": incidence_head.state_dict(),
                "config": asdict(config),
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "validation": metrics,
                "sources": sources,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            if metrics["image_exact"] > best:
                best = metrics["image_exact"]
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
