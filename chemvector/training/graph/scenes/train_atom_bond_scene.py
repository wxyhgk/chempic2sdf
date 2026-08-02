#!/usr/bin/env python3
"""Train the v0.9.42 joint AtomObject/BondObject capacity model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from itertools import islice
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.atom_bond_scene import (
    ATOM_ROLES,
    AtomBondSceneConfig,
    AtomBondScenePredictor,
    atom_bond_scene_loss,
    unordered_port_cost,
)
from chemvector.bond_object import BOND_TYPE_TO_ID, DIRECTION_TO_ID
from chemvector.data import ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig


DEFAULT_SHARD = (
    "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
DEFAULT_VISION = (
    "outputs/checkpoints/chemvector-v098-primitive-set-pilot9k-fresh-full-step1000/"
    "best.pt"
)
DEFAULT_OUTPUT = (
    "outputs/checkpoints/chemvector-v0942-atom-bond-scene-capacity100"
)
QUANTIZATION_MAX = 1023.0
ROLE_TO_ID = {name: index for index, name in enumerate(ATOM_ROLES)}


def _image_tensor(image: Image.Image, image_size: int = 768) -> torch.Tensor:
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


class AtomBondSceneCollator:
    def __init__(self, config: AtomBondSceneConfig) -> None:
        self.config = config

    def __call__(self, samples: list[dict]) -> dict:
        scenes = [
            build_chemscene_v0_2(
                sample["chemmol"], sample["render"], sample.get("layers")
            )
            for sample in samples
        ]
        batch = len(samples)
        atom_anchors = torch.zeros((batch, self.config.atom_queries, 2))
        atom_roles = torch.zeros((batch, self.config.atom_queries), dtype=torch.long)
        atom_text_center = torch.zeros((batch, self.config.atom_queries, 2))
        atom_text_bbox = torch.zeros((batch, self.config.atom_queries, 4))
        atom_text_mask = torch.zeros((batch, self.config.atom_queries), dtype=torch.bool)
        atom_mask = torch.zeros((batch, self.config.atom_queries), dtype=torch.bool)
        bond_ports = torch.zeros((batch, self.config.bond_queries, 2, 2))
        bond_display = torch.zeros((batch, self.config.bond_queries), dtype=torch.long)
        bond_chemical = torch.zeros((batch, self.config.bond_queries), dtype=torch.long)
        bond_direction = torch.zeros((batch, self.config.bond_queries), dtype=torch.long)
        bond_atom_refs = torch.zeros(
            (batch, self.config.bond_queries, 2), dtype=torch.long
        )
        bond_mask = torch.zeros((batch, self.config.bond_queries), dtype=torch.bool)

        for batch_index, scene in enumerate(scenes):
            atoms = scene["atom_objects"]
            bonds = scene["bond_objects"]
            if len(atoms) > self.config.atom_queries:
                raise ValueError(
                    f"AtomObject count {len(atoms)} exceeds query capacity"
                )
            if len(bonds) > self.config.bond_queries:
                raise ValueError(
                    f"BondObject count {len(bonds)} exceeds query capacity"
                )
            for index, atom in enumerate(atoms):
                atom_anchors[batch_index, index] = (
                    torch.tensor(atom["anchor"], dtype=torch.float32)
                    / QUANTIZATION_MAX
                )
                atom_roles[batch_index, index] = ROLE_TO_ID[atom["role"]]
                atom_mask[batch_index, index] = True
                if atom["text"] is not None:
                    text = atom["text"]
                    center = torch.tensor(text["center"], dtype=torch.float32)
                    x0, y0, x1, y1 = text["bbox"]
                    atom_text_center[batch_index, index] = center / QUANTIZATION_MAX
                    atom_text_bbox[batch_index, index] = torch.tensor(
                        (
                            (x0 + x1) / 2.0,
                            (y0 + y1) / 2.0,
                            x1 - x0,
                            y1 - y0,
                        ),
                        dtype=torch.float32,
                    ) / QUANTIZATION_MAX
                    atom_text_mask[batch_index, index] = True
            for index, bond in enumerate(bonds):
                bond_ports[batch_index, index] = (
                    torch.tensor(bond["ports"], dtype=torch.float32)
                    / QUANTIZATION_MAX
                )
                bond_display[batch_index, index] = BOND_TYPE_TO_ID[
                    bond["display_type"]
                ]
                bond_chemical[batch_index, index] = BOND_TYPE_TO_ID[
                    bond["chemical_type"]
                ]
                bond_direction[batch_index, index] = DIRECTION_TO_ID[
                    bond["direction"]
                ]
                bond_atom_refs[batch_index, index] = torch.tensor(
                    bond["atom_refs"], dtype=torch.long
                )
                bond_mask[batch_index, index] = True

        return {
            "pixel_values": torch.stack(
                [_image_tensor(sample["image"]) for sample in samples]
            ),
            "atom_anchors": atom_anchors,
            "atom_roles": atom_roles,
            "atom_text_center": atom_text_center,
            "atom_text_bbox": atom_text_bbox,
            "atom_text_mask": atom_text_mask,
            "atom_mask": atom_mask,
            "bond_ports": bond_ports,
            "bond_display": bond_display,
            "bond_chemical": bond_chemical,
            "bond_direction": bond_direction,
            "bond_atom_refs": bond_atom_refs,
            "bond_mask": bond_mask,
            "keys": [sample["key"] for sample in samples],
            "scenes": scenes,
        }


TARGET_FIELDS = (
    "atom_anchors",
    "atom_roles",
    "atom_text_center",
    "atom_text_bbox",
    "atom_text_mask",
    "atom_mask",
    "bond_ports",
    "bond_display",
    "bond_chemical",
    "bond_direction",
    "bond_atom_refs",
    "bond_mask",
)


def stack_targets(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True) for name in TARGET_FIELDS
    }


def _vision_config(payload: dict) -> ChemVectorConfig:
    valid = {field.name for field in fields(ChemVectorConfig)}
    return ChemVectorConfig(
        **{
            name: value
            for name, value in payload["vision_config"].items()
            if name in valid
        }
    )


def load_model(
    vision_checkpoint: Path,
    config: AtomBondSceneConfig,
    device: torch.device,
    *,
    freeze_vision: bool,
) -> tuple[AtomBondScenePredictor, dict]:
    payload = torch.load(vision_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "primitive-set-v0.1":
        raise ValueError("Vision source must be a primitive-set-v0.1 checkpoint")
    model = AtomBondScenePredictor(_vision_config(payload), config)
    vision_state = {
        name.removeprefix("vision_encoder."): value
        for name, value in payload["model"].items()
        if name.startswith("vision_encoder.")
    }
    missing, unexpected = model.vision_encoder.load_state_dict(
        vision_state, strict=True
    )
    if missing or unexpected:
        raise ValueError(
            f"Vision initialization differs: missing={missing}, unexpected={unexpected}"
        )
    if freeze_vision:
        for parameter in model.vision_encoder.parameters():
            parameter.requires_grad = False
    return model.to(device), payload


def _target_to_selected_query(
    selected_queries: torch.Tensor,
    predicted_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cost = torch.cdist(
        predicted_anchors[selected_queries].float(), target_anchors.float(), p=1
    )
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    rows = torch.as_tensor(rows, device=selected_queries.device, dtype=torch.long)
    columns = torch.as_tensor(columns, device=selected_queries.device, dtype=torch.long)
    target_to_query = torch.full(
        (len(target_anchors),), -1, device=selected_queries.device, dtype=torch.long
    )
    target_to_query[columns] = selected_queries[rows]
    return rows, columns, target_to_query


def _evaluate_selection(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    batch_index: int,
    atom_count: int,
    bond_count: int,
) -> dict[str, bool]:
    true_atom_count = int(targets["atom_mask"][batch_index].sum())
    true_bond_count = int(targets["bond_mask"][batch_index].sum())
    if atom_count != true_atom_count or bond_count != true_bond_count:
        return {
            "atom_exact": False,
            "bond_exact": False,
            "incidence_exact": False,
            "skeleton_exact": False,
            "atom_objects": true_atom_count,
            "atom_coordinate_correct": 0,
            "atom_role_correct": 0,
            "atom_error_sum": 768.0 * true_atom_count,
            "bond_objects": true_bond_count,
            "bond_port_correct": 0,
            "bond_display_correct": 0,
            "bond_chemical_correct": 0,
            "incidence_endpoints": 2 * true_bond_count,
            "incidence_endpoint_correct": 0,
        }
    atom_queries = output["atom_presence_logits"][batch_index].topk(atom_count).indices
    bond_queries = output["bond_presence_logits"][batch_index].topk(bond_count).indices
    target_atoms = targets["atom_anchors"][batch_index, :true_atom_count]
    atom_rows, atom_columns, target_to_query = _target_to_selected_query(
        atom_queries,
        output["atom_anchors"][batch_index],
        target_atoms,
    )
    atom_errors = torch.linalg.vector_norm(
        (
            output["atom_anchors"][batch_index, atom_queries[atom_rows]]
            - target_atoms[atom_columns]
        )
        * 768.0,
        dim=-1,
    )
    atom_roles = output["atom_role_logits"][
        batch_index, atom_queries[atom_rows]
    ].argmax(-1)
    expected_roles = targets["atom_roles"][batch_index, atom_columns]
    coordinate_correct = atom_errors <= 6.0
    atom_exact = bool((atom_errors <= 6.0).all() and (atom_roles == expected_roles).all())

    target_ports = targets["bond_ports"][batch_index, :true_bond_count]
    cost = unordered_port_cost(
        output["bond_ports"][batch_index, bond_queries], target_ports
    )
    bond_rows, bond_columns = linear_sum_assignment(cost.float().cpu().numpy())
    bond_rows = torch.as_tensor(
        bond_rows, device=bond_queries.device, dtype=torch.long
    )
    bond_columns = torch.as_tensor(
        bond_columns, device=bond_queries.device, dtype=torch.long
    )
    matched_queries = bond_queries[bond_rows]
    predicted_ports = output["bond_ports"][batch_index, matched_queries]
    expected_ports = target_ports[bond_columns]
    forward = torch.linalg.vector_norm(
        (predicted_ports - expected_ports) * 768.0, dim=-1
    )
    reverse = torch.linalg.vector_norm(
        (predicted_ports - expected_ports.flip(1)) * 768.0, dim=-1
    )
    use_reverse = reverse.sum(-1) < forward.sum(-1)
    port_errors = torch.where(use_reverse[:, None], reverse, forward)
    display = output["bond_display_logits"][batch_index, matched_queries].argmax(-1)
    chemical = output["bond_chemical_logits"][batch_index, matched_queries].argmax(-1)
    expected_display = targets["bond_display"][batch_index, bond_columns]
    expected_chemical = targets["bond_chemical"][batch_index, bond_columns]
    bond_exact = bool(
        (port_errors <= 6.0).all()
        and (display == expected_display).all()
        and (chemical == expected_chemical).all()
    )

    refs = targets["bond_atom_refs"][batch_index, bond_columns]
    expected_query_refs = target_to_query[refs]
    expected_query_refs = torch.where(
        use_reverse[:, None], expected_query_refs.flip(1), expected_query_refs
    )
    incidence = output["incidence_logits"][batch_index, matched_queries]
    selected_incidence = incidence[:, :, atom_queries]
    predicted_offsets = selected_incidence.argmax(-1)
    predicted_query_refs = atom_queries[predicted_offsets]
    incidence_exact = bool((predicted_query_refs == expected_query_refs).all())
    return {
        "atom_exact": atom_exact,
        "bond_exact": bond_exact,
        "incidence_exact": incidence_exact,
        "skeleton_exact": atom_exact and bond_exact and incidence_exact,
        "atom_objects": true_atom_count,
        "atom_coordinate_correct": int(coordinate_correct.sum()),
        "atom_role_correct": int((atom_roles == expected_roles).sum()),
        "atom_error_sum": float(atom_errors.sum()),
        "bond_objects": true_bond_count,
        "bond_port_correct": int((port_errors <= 6.0).all(dim=1).sum()),
        "bond_display_correct": int((display == expected_display).sum()),
        "bond_chemical_correct": int((chemical == expected_chemical).sum()),
        "incidence_endpoints": 2 * true_bond_count,
        "incidence_endpoint_correct": int(
            (predicted_query_refs == expected_query_refs).sum()
        ),
    }


@torch.no_grad()
def evaluate(
    model: AtomBondScenePredictor,
    records: list[dict],
    collator: AtomBondSceneCollator,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    totals = {
        "samples": 0,
        "atom_count_exact": 0,
        "bond_count_exact": 0,
        "known_atom_exact": 0,
        "known_bond_exact": 0,
        "known_incidence_exact": 0,
        "known_skeleton_exact": 0,
        "predicted_atom_exact": 0,
        "predicted_bond_exact": 0,
        "predicted_incidence_exact": 0,
        "predicted_skeleton_exact": 0,
        "known_atom_objects": 0,
        "known_atom_coordinate_correct": 0,
        "known_atom_role_correct": 0,
        "known_atom_error_sum": 0.0,
        "known_bond_objects": 0,
        "known_bond_port_correct": 0,
        "known_bond_display_correct": 0,
        "known_bond_chemical_correct": 0,
        "known_incidence_endpoints": 0,
        "known_incidence_endpoint_correct": 0,
    }
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(pixels)
        for batch_index in range(len(pixels)):
            true_atoms = int(targets["atom_mask"][batch_index].sum())
            true_bonds = int(targets["bond_mask"][batch_index].sum())
            predicted_atoms = int(output["atom_count_logits"][batch_index].argmax())
            predicted_bonds = int(output["bond_count_logits"][batch_index].argmax())
            known = _evaluate_selection(
                output, targets, batch_index, true_atoms, true_bonds
            )
            predicted = _evaluate_selection(
                output, targets, batch_index, predicted_atoms, predicted_bonds
            )
            totals["samples"] += 1
            totals["atom_count_exact"] += int(predicted_atoms == true_atoms)
            totals["bond_count_exact"] += int(predicted_bonds == true_bonds)
            for name in ("atom_exact", "bond_exact", "incidence_exact", "skeleton_exact"):
                totals[f"known_{name}"] += int(known[name])
                totals[f"predicted_{name}"] += int(predicted[name])
            for name in (
                "atom_objects",
                "atom_coordinate_correct",
                "atom_role_correct",
                "atom_error_sum",
                "bond_objects",
                "bond_port_correct",
                "bond_display_correct",
                "bond_chemical_correct",
                "incidence_endpoints",
                "incidence_endpoint_correct",
            ):
                totals[f"known_{name}"] += known[name]
    samples = max(totals["samples"], 1)
    internal_totals = {
        "known_atom_objects",
        "known_atom_coordinate_correct",
        "known_atom_role_correct",
        "known_atom_error_sum",
        "known_bond_objects",
        "known_bond_port_correct",
        "known_bond_display_correct",
        "known_bond_chemical_correct",
        "known_incidence_endpoints",
        "known_incidence_endpoint_correct",
    }
    metrics = {
        name: (value if name == "samples" else value / samples)
        for name, value in totals.items()
        if name not in internal_totals
    }
    atom_objects = max(totals["known_atom_objects"], 1)
    bond_objects = max(totals["known_bond_objects"], 1)
    incidence_endpoints = max(totals["known_incidence_endpoints"], 1)
    metrics.update(
        {
            "known_atom_coordinate_accuracy": totals[
                "known_atom_coordinate_correct"
            ]
            / atom_objects,
            "known_atom_role_accuracy": totals["known_atom_role_correct"]
            / atom_objects,
            "known_atom_mean_error_px": totals["known_atom_error_sum"]
            / atom_objects,
            "known_bond_port_pair_accuracy": totals["known_bond_port_correct"]
            / bond_objects,
            "known_bond_display_accuracy": totals[
                "known_bond_display_correct"
            ]
            / bond_objects,
            "known_bond_chemical_accuracy": totals[
                "known_bond_chemical_correct"
            ]
            / bond_objects,
            "known_incidence_endpoint_accuracy": totals[
                "known_incidence_endpoint_correct"
            ]
            / incidence_endpoints,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", default=DEFAULT_SHARD)
    parser.add_argument("--vision-checkpoint", default=DEFAULT_VISION)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--atom-queries", type=int, default=128)
    parser.add_argument("--bond-queries", type=int, default=128)
    parser.add_argument("--memory-grid-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--decoder-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v0.9.42 capacity training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    config = AtomBondSceneConfig(
        atom_queries=args.atom_queries,
        bond_queries=args.bond_queries,
        memory_grid_size=args.memory_grid_size,
        d_model=args.d_model,
        decoder_layers=args.decoder_layers,
        decoder_heads=6 if args.d_model % 6 == 0 else 4,
        decoder_ffn_dim=4 * args.d_model,
    )
    model, vision_payload = load_model(
        Path(args.vision_checkpoint),
        config,
        device,
        freeze_vision=not args.train_vision,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0
    )
    start_step = 0
    resume_payload = None
    if args.resume_checkpoint:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        expected_config = AtomBondSceneConfig(
            **resume_payload["atom_bond_scene_config"]
        )
        if expected_config != config:
            raise ValueError(
                f"Resume config differs: checkpoint={expected_config}, requested={config}"
            )
        model.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["global_step"])
        if start_step >= args.max_steps:
            raise ValueError("Resume checkpoint has already reached --max-steps")
    dataset = ChemVectorIterableDataset([args.train_shard], sequence_version="0.4")
    records = list(islice(iter(dataset), args.samples))
    if len(records) != args.samples:
        raise ValueError(f"Capacity shard yielded only {len(records)} samples")
    if any(
        sample.get("meta", {}).get("source_split") != "train"
        for sample in records
    ):
        raise ValueError("Capacity training may only read train samples")
    collator = AtomBondSceneCollator(config)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **model.config_dict(),
                "parameter_count": model.parameter_count(),
                "trainable_parameter_count": sum(p.numel() for p in trainable),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    best = (
        float(resume_payload.get("validation", {}).get("predicted_skeleton_exact", -1.0))
        if resume_payload is not None
        else -1.0
    )
    started = time.perf_counter()
    running_loss = 0.0
    for step in range(start_step + 1, args.max_steps + 1):
        model.train()
        if not args.train_vision:
            model.vision_encoder.eval()
        batch_records = [records[rng.randrange(len(records))] for _ in range(args.batch_size)]
        batch = collator(batch_records)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = stack_targets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixels)
            losses = atom_bond_scene_loss(output, targets)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        running_loss += float(losses["loss"].detach())

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": running_loss / args.log_every,
                        "samples_per_second": (
                            (step - start_step) * args.batch_size / max(elapsed, 1e-6)
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            running_loss = 0.0

        if step % args.validate_every == 0 or step == args.max_steps:
            metrics = evaluate(model, records, collator, device, args.batch_size)
            metrics["step"] = step
            metrics["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps({"validation": metrics}, separators=(",", ":")), flush=True)
            checkpoint = {
                "checkpoint_schema_version": "atom-bond-scene-checkpoint-0.1",
                "model_type": model.model_type,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vision_config": asdict(model.vision_config),
                "atom_bond_scene_config": asdict(config),
                "global_step": step,
                "validation": metrics,
                "source_vision_checkpoint": args.vision_checkpoint,
                "source_vision_global_step": vision_payload.get("global_step"),
                "freeze_vision": not args.train_vision,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, output_dir / "last.pt")
            score = metrics["predicted_skeleton_exact"]
            if score > best:
                best = score
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
