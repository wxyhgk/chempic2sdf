#!/usr/bin/env python3
"""Train a whole-set selector over frozen Bond Object proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import shutil

from rdkit import Chem
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Data.sdf.chemscene_v0_1 import mol_from_chemscene_v0_1
from chemvector.audit_bond_first_decomposition import (
    _atom_centers,
    _expected_bonds,
    _load_models,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS
from chemvector.bond_set_selector import (
    BondSetSelector,
    BondSetSelectorConfig,
    bond_set_selector_loss,
    decode_bond_set,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import decode_dense_bond_objects
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
)
from chemvector.pretrain_chemscene_bond_query import ChemSceneBondCollator
from chemvector.pretrain_dense_bond_object import (
    _forward_detector,
    _scene_from_dense_objects,
)
from chemvector.train import expand_shards


def build_candidate_example(
    sample: dict,
    objects: list[dict],
    scene: dict,
    expected_inchikey: str,
    line_band_features: torch.Tensor | None = None,
) -> dict:
    if not objects:
        raise ValueError(f"Bond candidate cache is empty for {sample['key']}")
    endpoints = torch.stack([item["endpoints"].float().cpu() for item in objects])
    scores = torch.tensor([float(item["score"]) for item in objects])
    display_ids = torch.tensor([int(item["display"]) for item in objects])
    chemical_ids = torch.tensor([int(item["chemical"]) for item in objects])
    direction_ids = torch.tensor([int(item["direction"]) for item in objects])
    visual_features = torch.stack(
        [item["feature"].float().cpu() for item in objects]
    )
    if line_band_features is not None:
        if line_band_features.shape[0] != len(objects):
            raise ValueError("Line-band features differ from Bond candidates")
        visual_features = torch.cat(
            (visual_features, line_band_features.float().cpu()), dim=-1
        )
    centers = _atom_centers(sample)
    distances = torch.cdist(endpoints, centers)
    nearest_distances, nearest_atoms = distances.min(-1)
    pairs = nearest_atoms.sort(-1).values
    expected = _expected_bonds(sample)
    expected_by_pair = {item["pair"]: item for item in expected}
    pair_to_candidates: dict[tuple[int, int], list[int]] = {}
    for index, pair_tensor in enumerate(pairs):
        pair = tuple(int(value) for value in pair_tensor.tolist())
        if pair[0] == pair[1]:
            continue
        pair_to_candidates.setdefault(pair, []).append(index)

    keep_targets = torch.zeros(len(objects), dtype=torch.bool)
    display_targets = display_ids.clone()
    chemical_targets = chemical_ids.clone()
    for pair, target in expected_by_pair.items():
        candidates = pair_to_candidates.get(pair, [])
        if not candidates:
            continue
        expected_endpoints = target["endpoints"]
        best = min(
            candidates,
            key=lambda index: float(
                torch.minimum(
                    (endpoints[index] - expected_endpoints).abs().sum(),
                    (endpoints[index] - expected_endpoints.flip(0)).abs().sum(),
                )
            ),
        )
        keep_targets[best] = True
        display_targets[best] = int(target["display"])
        chemical_targets[best] = int(target["chemical"])

    center = endpoints.mean(1)
    delta = endpoints[:, 1] - endpoints[:, 0]
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    self_loop = pairs[:, 0].eq(pairs[:, 1]).float()[:, None]
    features = torch.cat(
        (
            torch.logit(scores.clamp(1e-5, 1.0 - 1e-5))[:, None],
            endpoints.flatten(1),
            center,
            delta.abs(),
            length,
            F.one_hot(display_ids, num_classes=len(BOND_TYPES)).float(),
            F.one_hot(chemical_ids, num_classes=len(BOND_TYPES)).float(),
            F.one_hot(direction_ids, num_classes=len(DIRECTIONS)).float(),
            nearest_distances * 8.0,
            self_loop,
            visual_features,
        ),
        dim=-1,
    )
    expected_pairs = sorted(expected_by_pair)
    return {
        "key": sample["key"],
        "features": features,
        "mask": torch.ones(len(objects), dtype=torch.bool),
        "pairs": pairs,
        "endpoints": endpoints,
        "scores": scores,
        "display_ids": display_ids,
        "chemical_ids": chemical_ids,
        "direction_ids": direction_ids,
        "keep_targets": keep_targets,
        "count_target": len(expected),
        "display_targets": display_targets,
        "chemical_targets": chemical_targets,
        "expected_pairs": expected_pairs,
        "expected_display": [expected_by_pair[pair]["display"] for pair in expected_pairs],
        "expected_chemical": [expected_by_pair[pair]["chemical"] for pair in expected_pairs],
        "candidate_oracle": all(pair in pair_to_candidates for pair in expected_pairs),
        "chemical_candidate_oracle": all(
            any(
                int(chemical_ids[index]) == expected_by_pair[pair]["chemical"]
                for index in pair_to_candidates.get(pair, [])
            )
            for pair in expected_pairs
        ),
        "scene": scene,
        "expected_inchikey": expected_inchikey,
    }


def load_line_band_reader(
    checkpoint_path: Path | None,
    device: torch.device,
) -> NodeConditionedBondObject | None:
    if checkpoint_path is None:
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = NodeConditionedBondConfig(**payload["config"])
    reader = NodeConditionedBondObject(config).to(device)
    reader.load_state_dict(payload["node_conditioned_bond"])
    reader.eval()
    reader.requires_grad_(False)
    return reader


def read_line_band_features(
    reader: NodeConditionedBondObject,
    pixel_values: torch.Tensor,
    object_batches: list[list[dict]],
) -> list[torch.Tensor]:
    """Read every candidate as one complete 2-D line-band object."""

    counts = [len(objects) for objects in object_batches]
    maximum = max(counts, default=0)
    if maximum == 0:
        return [torch.empty((0, reader.config.hidden_dim + 14)) for _ in counts]
    points = torch.zeros(
        (len(object_batches), maximum * 2, 2),
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )
    pair_indices = torch.zeros(
        (len(object_batches), maximum, 2),
        device=pixel_values.device,
        dtype=torch.long,
    )
    pair_mask = torch.zeros(
        (len(object_batches), maximum),
        device=pixel_values.device,
        dtype=torch.bool,
    )
    base_pairs = torch.arange(
        maximum * 2, device=pixel_values.device, dtype=torch.long
    ).reshape(maximum, 2)
    for batch_index, objects in enumerate(object_batches):
        count = len(objects)
        if not count:
            continue
        endpoints = torch.stack(
            [item["endpoints"].to(pixel_values.device) for item in objects]
        ).to(pixel_values.dtype)
        points[batch_index, : count * 2] = endpoints.reshape(count * 2, 2)
        pair_indices[batch_index, :count] = base_pairs[:count]
        pair_mask[batch_index, :count] = True
    output = reader(pixel_values, points, pair_indices, pair_mask)
    features = torch.cat(
        (
            output["edge_logits"].sigmoid()[:, None],
            output["display_logits"].softmax(-1),
            output["chemical_logits"].softmax(-1),
            output["direction_logits"].softmax(-1),
            output["hidden"],
        ),
        dim=-1,
    ).float().cpu()
    return list(features.split(counts))


@torch.inference_mode()
def build_cache(
    model,
    detector,
    line_band_reader: NodeConditionedBondObject | None,
    shards: list[str],
    *,
    samples: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    records = list(
        islice(
            iter(ChemVectorIterableDataset(shards, sequence_version="0.4")),
            samples,
        )
    )
    if len(records) != samples:
        raise ValueError(f"Requested {samples} cache samples, loaded {len(records)}")
    collator = ChemSceneBondCollator()
    examples = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch = collator(batch_records)
        pixels = batch["pixel_values"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = _forward_detector(model, detector, pixels)
        object_batches = decode_dense_bond_objects(
            output, threshold=0.0, maximum_objects=128
        )
        line_band_batches = (
            read_line_band_features(line_band_reader, pixels, object_batches)
            if line_band_reader is not None
            else [None] * len(object_batches)
        )
        examples.extend(
            build_candidate_example(
                sample, objects, scene, inchikey, line_band_features
            )
            for sample, objects, scene, inchikey, line_band_features in zip(
                batch_records,
                object_batches,
                batch["scenes"],
                batch["expected_inchikeys"],
                line_band_batches,
            )
        )
        completed = min(start + batch_size, len(records))
        if completed % 100 == 0 or completed == len(records):
            print(f"cached={completed}/{len(records)}", flush=True)
    return examples


def load_or_build_cache(path: Path, build) -> list[dict]:
    if path.exists():
        print(f"loading_cache={path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    examples = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, path)
    return examples


def collate_examples(examples: list[dict]) -> dict:
    return {
        "features": torch.stack([item["features"] for item in examples]),
        "mask": torch.stack([item["mask"] for item in examples]),
        "pairs": torch.stack([item["pairs"] for item in examples]),
        "display_ids": torch.stack([item["display_ids"] for item in examples]),
        "chemical_ids": torch.stack([item["chemical_ids"] for item in examples]),
        "keep_targets": torch.stack([item["keep_targets"] for item in examples]),
        "count_targets": torch.tensor([item["count_target"] for item in examples]),
        "display_targets": torch.stack([item["display_targets"] for item in examples]),
        "chemical_targets": torch.stack([item["chemical_targets"] for item in examples]),
        "examples": examples,
    }


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def evaluate(
    model: BondSetSelector,
    examples: list[dict],
    device: torch.device,
    *,
    batch_size: int,
    compile_sdf: bool,
) -> dict:
    model.eval()
    totals = {
        "samples": 0,
        "candidate_oracle": 0,
        "chemical_candidate_oracle": 0,
        "count_exact": 0,
        "topology_exact": 0,
        "topology_exact_on_candidate_oracle": 0,
        "chemical_exact": 0,
        "compiled_sdf_exact": 0,
    }
    loader = DataLoader(
        examples, batch_size=batch_size, shuffle=False, collate_fn=collate_examples
    )
    for batch in loader:
        tensors = to_device(batch, device)
        output = model(
            tensors["features"],
            tensors["mask"],
            tensors["display_ids"],
            tensors["chemical_ids"],
        )
        selected_batches = decode_bond_set(
            output, tensors["pairs"], tensors["mask"]
        )
        predicted_display = output["display_logits"].argmax(-1).cpu()
        predicted_chemical = output["chemical_logits"].argmax(-1).cpu()
        predicted_counts = output["count_logits"].argmax(-1).cpu()
        for batch_index, (example, selected) in enumerate(
            zip(batch["examples"], selected_batches)
        ):
            expected_pair_list = [
                tuple(pair) for pair in example["expected_pairs"]
            ]
            expected_pairs = set(expected_pair_list)
            expected_chemical = dict(
                zip(expected_pair_list, example["expected_chemical"])
            )
            selected_pairs = {
                tuple(int(value) for value in example["pairs"][index].tolist())
                for index in selected
            }
            topology_exact = selected_pairs == expected_pairs
            chemical_exact = topology_exact and all(
                int(predicted_chemical[batch_index, index])
                == int(expected_chemical[tuple(example["pairs"][index].tolist())])
                for index in selected
            )
            totals["samples"] += 1
            totals["candidate_oracle"] += int(example["candidate_oracle"])
            totals["chemical_candidate_oracle"] += int(
                example["chemical_candidate_oracle"]
            )
            totals["count_exact"] += int(
                int(predicted_counts[batch_index]) == int(example["count_target"])
            )
            totals["topology_exact"] += int(topology_exact)
            totals["topology_exact_on_candidate_oracle"] += int(
                topology_exact and example["candidate_oracle"]
            )
            totals["chemical_exact"] += int(chemical_exact)
            if not compile_sdf:
                continue
            objects = [
                {
                    "endpoints": example["endpoints"][index],
                    "score": float(example["scores"][index]),
                    "display": int(predicted_display[batch_index, index]),
                    "chemical": int(predicted_chemical[batch_index, index]),
                    "direction": int(example["direction_ids"][index]),
                }
                for index in selected
            ]
            scene = _scene_from_dense_objects(example["scene"], objects)
            if scene is None:
                continue
            try:
                exact = (
                    Chem.MolToInchiKey(mol_from_chemscene_v0_1(scene))
                    == example["expected_inchikey"]
                )
            except Exception:
                exact = False
            totals["compiled_sdf_exact"] += int(exact)
    samples = max(totals["samples"], 1)
    oracle = max(totals["candidate_oracle"], 1)
    return {
        **totals,
        "candidate_oracle_rate": totals["candidate_oracle"] / samples,
        "chemical_candidate_oracle_rate": totals["chemical_candidate_oracle"]
        / samples,
        "count_accuracy": totals["count_exact"] / samples,
        "topology_exact_rate": totals["topology_exact"] / samples,
        "selector_efficiency_on_candidate_oracle": totals[
            "topology_exact_on_candidate_oracle"
        ]
        / oracle,
        "chemical_exact_rate": totals["chemical_exact"] / samples,
        "compiled_sdf_exact_rate": totals["compiled_sdf_exact"] / samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bond-checkpoint",
        default="outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
    )
    parser.add_argument("--line-band-checkpoint")
    parser.add_argument("--grounded-graph-checkpoint")
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-*.tar"],
    )
    parser.add_argument("--validation-shards", nargs="+")
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/chemvector-v0813-bond-set-selector-capacity500",
    )
    parser.add_argument("--train-cache-samples", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_base, detector, graph_path = _load_models(
        Path(args.bond_checkpoint),
        device,
        Path(args.grounded_graph_checkpoint)
        if args.grounded_graph_checkpoint
        else None,
    )
    line_band_path = (
        Path(args.line_band_checkpoint) if args.line_band_checkpoint else None
    )
    line_band_reader = load_line_band_reader(line_band_path, device)
    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards or args.train_shards)
    train_examples = load_or_build_cache(
        output_dir / "train-candidates.pt",
        lambda: build_cache(
            model_base,
            detector,
            line_band_reader,
            train_shards,
            samples=args.train_cache_samples,
            batch_size=args.cache_batch_size,
            device=device,
        ),
    )
    validation_examples = load_or_build_cache(
        output_dir / "validation-candidates.pt",
        lambda: build_cache(
            model_base,
            detector,
            line_band_reader,
            validation_shards,
            samples=args.validation_samples,
            batch_size=args.cache_batch_size,
            device=device,
        ),
    )
    del model_base, detector, line_band_reader
    torch.cuda.empty_cache()

    initial_epoch = 0
    initial_metrics = None
    initial = None
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        config = BondSetSelectorConfig(**initial["config"])
    else:
        config = BondSetSelectorConfig(
            input_dim=int(train_examples[0]["features"].size(-1))
        )
    for split, examples in (
        ("train", train_examples),
        ("validation", validation_examples),
    ):
        widths = {int(example["features"].size(-1)) for example in examples}
        if widths != {config.input_dim}:
            raise ValueError(
                f"{split} cache feature widths {sorted(widths)} differ from "
                f"selector input_dim={config.input_dim}"
            )
    selector = BondSetSelector(config).to(device)
    if initial is not None:
        selector.load_state_dict(initial["bond_set_selector"])
        initial_epoch = int(initial.get("epoch", 0))
        initial_metrics = initial.get("validation")
    if args.evaluate_only:
        metrics = evaluate(
            selector,
            validation_examples,
            device,
            batch_size=args.batch_size,
            compile_sdf=True,
        )
        result = {
            "schema_version": "bond-set-selector-evaluation-0.1",
            "checkpoint": args.init_checkpoint,
            "metrics": metrics,
        }
        (output_dir / "evaluation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    continue_same_run = bool(
        args.init_checkpoint
        and (output_dir / "best.pt").exists()
        and Path(args.init_checkpoint).resolve() == (output_dir / "best.pt").resolve()
    )
    best_score = (
        (
            initial_metrics["topology_exact_rate"],
            initial_metrics["count_accuracy"],
            initial_metrics["chemical_exact_rate"],
        )
        if initial_metrics is not None and continue_same_run
        else (-1.0, -1.0, -1.0)
    )
    best_metrics = initial_metrics if continue_same_run else None
    for epoch in range(initial_epoch + 1, initial_epoch + args.epochs + 1):
        selector.train()
        loss_sum = 0.0
        batches = 0
        for batch in loader:
            tensors = to_device(batch, device)
            output = selector(
                tensors["features"],
                tensors["mask"],
                tensors["display_ids"],
                tensors["chemical_ids"],
            )
            losses = bond_set_selector_loss(
                output,
                tensors["mask"],
                tensors["keep_targets"],
                tensors["count_targets"],
                tensors["display_targets"],
                tensors["chemical_targets"],
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(losses["loss"].detach())
            batches += 1
        if epoch % args.validate_every and epoch != initial_epoch + args.epochs:
            continue
        metrics = evaluate(
            selector,
            validation_examples,
            device,
            batch_size=args.batch_size,
            compile_sdf=False,
        )
        print(
            f"epoch={epoch} loss={loss_sum / max(batches, 1):.6f} "
            f"count={metrics['count_accuracy']:.4f} "
            f"topology={metrics['topology_exact_rate']:.4f} "
            f"efficiency={metrics['selector_efficiency_on_candidate_oracle']:.4f} "
            f"chemical={metrics['chemical_exact_rate']:.4f}",
            flush=True,
        )
        checkpoint = output_dir / f"checkpoint-{epoch:03d}.pt"
        torch.save(
            {
                "bond_set_selector": selector.state_dict(),
                "config": asdict(config),
                "epoch": epoch,
                "validation": metrics,
                "bond_checkpoint": args.bond_checkpoint,
                "line_band_checkpoint": args.line_band_checkpoint,
                "grounded_graph_checkpoint": str(graph_path),
            },
            checkpoint,
        )
        score = (
            metrics["topology_exact_rate"],
            metrics["count_accuracy"],
            metrics["chemical_exact_rate"],
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            shutil.copy2(checkpoint, output_dir / "best.pt")

    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    selector.load_state_dict(best["bond_set_selector"])
    final_metrics = evaluate(
        selector,
        validation_examples,
        device,
        batch_size=args.batch_size,
        compile_sdf=True,
    )
    summary = {
        "schema_version": "bond-set-selector-capacity-0.1",
        "best_epoch": best["epoch"],
        "best_selection_metrics": best_metrics,
        "final_metrics": final_metrics,
        "go": (
            final_metrics["selector_efficiency_on_candidate_oracle"] >= 0.95
            and final_metrics["count_accuracy"] >= 0.95
        ),
        "protocol": {
            **vars(args),
            "grounded_graph_checkpoint": str(graph_path),
            "uses_oracle_junctions": True,
            "uses_mr_tadf_for_tuning": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
