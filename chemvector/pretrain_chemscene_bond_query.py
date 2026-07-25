#!/usr/bin/env python3
"""Train v0.8 Bond Queries against ChemScene object targets."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import random
import shutil

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import (
    build_chemscene_v0_1,
    mol_from_chemscene_v0_1,
    parse_chemscene_v0_1,
)
from chemvector.bond_object import (
    BOND_TYPES,
    BOND_TYPE_TO_ID,
    DIRECTIONS,
    DIRECTION_TO_ID,
    BondObjectConfig,
    BondObjectDetector,
    bond_object_loss,
    unordered_endpoint_cost,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train import expand_shards


QUANTIZATION_MAX = 1023.0


class ChemSceneBondCollator:
    def __call__(self, samples: list[dict]) -> dict:
        scenes = [
            build_chemscene_v0_1(
                sample["chemmol"], sample["render"], sample.get("layers")
            )
            for sample in samples
        ]
        maximum = max(len(scene["bond_objects"]) for scene in scenes)
        endpoints = torch.zeros((len(samples), maximum, 2, 2))
        boxes = torch.zeros((len(samples), maximum, 4))
        display = torch.zeros((len(samples), maximum), dtype=torch.long)
        chemical = torch.zeros((len(samples), maximum), dtype=torch.long)
        direction = torch.zeros((len(samples), maximum), dtype=torch.long)
        mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
        pixels = []
        for batch_index, (sample, scene) in enumerate(zip(samples, scenes)):
            image = sample["image"]
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            pixels.append((torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5)
            for index, bond in enumerate(scene["bond_objects"]):
                endpoints[batch_index, index] = (
                    torch.tensor(bond["endpoints"], dtype=torch.float32)
                    / QUANTIZATION_MAX
                )
                if bond["bbox"] is None:
                    (begin_x, begin_y), (end_x, end_y) = bond["endpoints"]
                    x0, y0 = min(begin_x, end_x), min(begin_y, end_y)
                    x1, y1 = max(begin_x, end_x), max(begin_y, end_y)
                else:
                    x0, y0, x1, y1 = bond["bbox"]
                boxes[batch_index, index] = torch.tensor(
                    (
                        (x0 + x1) / 2 / QUANTIZATION_MAX,
                        (y0 + y1) / 2 / QUANTIZATION_MAX,
                        (x1 - x0) / QUANTIZATION_MAX,
                        (y1 - y0) / QUANTIZATION_MAX,
                    )
                )
                display[batch_index, index] = BOND_TYPE_TO_ID[bond["display_type"]]
                chemical[batch_index, index] = BOND_TYPE_TO_ID[bond["chemical_type"]]
                direction[batch_index, index] = DIRECTION_TO_ID[bond["direction"]]
                mask[batch_index, index] = True
        return {
            "pixel_values": torch.stack(pixels),
            "endpoints": endpoints,
            "bbox": boxes,
            "display": display,
            "chemical": chemical,
            "direction": direction,
            "mask": mask,
            "keys": [sample["key"] for sample in samples],
            "scenes": scenes,
            "expected_inchikeys": [sample["chemmol"]["inchikey"] for sample in samples],
        }


def stack_targets(batch: dict, device: torch.device) -> dict:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in ("endpoints", "bbox", "display", "chemical", "direction", "mask")
    }


def _oriented_endpoint_errors(
    predicted: torch.Tensor, expected: torch.Tensor, image_size: float
) -> torch.Tensor:
    forward = torch.linalg.vector_norm((predicted - expected) * image_size, dim=-1)
    reverse = torch.linalg.vector_norm(
        (predicted - expected.flip(0)) * image_size, dim=-1
    )
    return forward if float(forward.sum()) <= float(reverse.sum()) else reverse


def _nearest_junction(
    endpoint: torch.Tensor, coordinates: list[tuple[int, int]]
) -> tuple[int, int]:
    x = float(endpoint[0]) * QUANTIZATION_MAX
    y = float(endpoint[1]) * QUANTIZATION_MAX
    return min(coordinates, key=lambda item: (item[0] - x) ** 2 + (item[1] - y) ** 2)


def _oracle_junction_scene(
    oracle_scene: dict,
    output: dict[str, torch.Tensor],
    batch_index: int,
    queries: torch.Tensor,
) -> dict | None:
    scene = copy.deepcopy(oracle_scene)
    parsed = parse_chemscene_v0_1(scene)
    coordinates = [junction.coordinate for junction in parsed.junctions]
    oracle_by_pair = {
        tuple(sorted((bond.begin, bond.end))): bond for bond in parsed.bond_objects
    }
    predicted_bonds = []
    seen_pairs = set()
    for query in queries.tolist():
        endpoints = output["endpoints"][batch_index, query]
        begin = _nearest_junction(endpoints[0], coordinates)
        end = _nearest_junction(endpoints[1], coordinates)
        pair = tuple(sorted((begin, end)))
        if begin == end or pair in seen_pairs:
            return None
        seen_pairs.add(pair)
        oracle = oracle_by_pair.get(pair)
        output_begin, output_end = (
            (oracle.begin, oracle.end) if oracle is not None else (begin, end)
        )
        item = {
            "endpoints": [list(output_begin), list(output_end)],
            "display_type": BOND_TYPES[
                int(output["display_logits"][batch_index, query].argmax())
            ],
            "chemical_type": BOND_TYPES[
                int(output["chemical_logits"][batch_index, query].argmax())
            ],
            "direction": DIRECTIONS[
                int(output["direction_logits"][batch_index, query].argmax())
            ],
            "stereo": oracle.stereo if oracle is not None else "STEREONONE",
            "bbox": None,
        }
        if oracle is not None and oracle.stereo_reference_0 is not None:
            item["stereo_references"] = [
                list(oracle.stereo_reference_0),
                list(oracle.stereo_reference_1),
            ]
        predicted_bonds.append(item)
    scene["bond_objects"] = predicted_bonds
    try:
        parse_chemscene_v0_1(scene)
    except Exception:
        return None
    return scene


@torch.no_grad()
def evaluate(model, detector, records, collator, device, batch_size: int) -> dict:
    detector.eval()
    totals = {
        "images": 0,
        "bonds": 0,
        "endpoint_pairs": 0,
        "object_correct": 0,
        "known_object_exact": 0,
        "predicted_count_exact": 0,
        "predicted_object_exact": 0,
        "oracle_junction_sdf_exact": 0,
    }
    maximum_errors = []
    object_failure_keys = []
    sdf_failure_keys = []
    display_count_exact = {name: 0 for name in BOND_TYPES}
    display_count_support = {name: 0 for name in BOND_TYPES}
    all_display_counts_exact = 0
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model.vision_encoder.forward_stroke_resolution(pixels)
            output = detector(features)
        for batch_index in range(len(pixels)):
            valid = targets["mask"][batch_index]
            expected = targets["endpoints"][batch_index, valid]
            count = len(expected)
            scores = output["presence_logits"][batch_index]
            known_queries = scores.topk(count).indices
            predicted_queries = torch.where(scores.sigmoid() >= 0.5)[0]

            def compare(queries: torch.Tensor) -> tuple[int, bool, float]:
                if len(queries) != count or not count:
                    return 0, False, float("inf")
                predicted = output["endpoints"][batch_index, queries]
                rows, columns = linear_sum_assignment(
                    unordered_endpoint_cost(predicted, expected).float().cpu().numpy()
                )
                valid_indices = torch.where(valid)[0]
                correct = 0
                worst = 0.0
                for row, column in zip(rows, columns):
                    errors = _oriented_endpoint_errors(
                        predicted[row], expected[column], 768.0
                    )
                    worst = max(worst, float(errors.max()))
                    target_index = valid_indices[column]
                    classes_match = (
                        int(output["display_logits"][batch_index, queries[row]].argmax())
                        == int(targets["display"][batch_index, target_index])
                        and int(output["chemical_logits"][batch_index, queries[row]].argmax())
                        == int(targets["chemical"][batch_index, target_index])
                        and int(output["direction_logits"][batch_index, queries[row]].argmax())
                        == int(targets["direction"][batch_index, target_index])
                    )
                    correct += int(bool((errors <= 6.0).all()) and classes_match)
                return correct, correct == count, worst

            known_correct, known_exact, worst = compare(known_queries)
            predicted_correct, predicted_exact, _ = compare(predicted_queries)
            sdf_exact = False
            if predicted_exact:
                predicted_scene = _oracle_junction_scene(
                    batch["scenes"][batch_index], output, batch_index, predicted_queries
                )
                if predicted_scene is not None:
                    try:
                        mol = mol_from_chemscene_v0_1(predicted_scene)
                        sdf_exact = (
                            Chem.MolToInchiKey(mol)
                            == batch["expected_inchikeys"][batch_index]
                        )
                    except Exception:
                        sdf_exact = False
            expected_display = targets["display"][batch_index, valid]
            predicted_display = (
                output["display_logits"][batch_index, predicted_queries].argmax(-1)
                if len(predicted_queries)
                else torch.empty(0, device=device, dtype=torch.long)
            )
            per_type_exact = []
            for type_id, name in enumerate(BOND_TYPES):
                expected_count = int((expected_display == type_id).sum())
                predicted_count = int((predicted_display == type_id).sum())
                exact_count = expected_count == predicted_count
                display_count_exact[name] += int(exact_count)
                display_count_support[name] += int(expected_count > 0)
                per_type_exact.append(exact_count)
            all_display_counts_exact += int(all(per_type_exact))
            totals["images"] += 1
            totals["bonds"] += count
            totals["endpoint_pairs"] += sum(
                int(
                    bool(
                        _oriented_endpoint_errors(
                            output["endpoints"][batch_index, known_queries[row]],
                            expected[column],
                            768.0,
                        ).max()
                        <= 6.0
                    )
                )
                for row, column in zip(
                    *linear_sum_assignment(
                        unordered_endpoint_cost(
                            output["endpoints"][batch_index, known_queries], expected
                        ).float().cpu().numpy()
                    )
                )
            )
            totals["object_correct"] += known_correct
            totals["known_object_exact"] += int(known_exact)
            totals["predicted_count_exact"] += int(len(predicted_queries) == count)
            totals["predicted_object_exact"] += int(predicted_exact)
            totals["oracle_junction_sdf_exact"] += int(sdf_exact)
            maximum_errors.append(worst)
            if not predicted_exact and len(object_failure_keys) < 50:
                object_failure_keys.append(batch["keys"][batch_index])
            if not sdf_exact and len(sdf_failure_keys) < 50:
                sdf_failure_keys.append(batch["keys"][batch_index])
    images = max(totals["images"], 1)
    bonds = max(totals["bonds"], 1)
    finite_errors = [value for value in maximum_errors if np.isfinite(value)]
    return {
        "samples": totals["images"],
        "endpoint_pair_recall": totals["endpoint_pairs"] / bonds,
        "object_accuracy": totals["object_correct"] / bonds,
        "known_object_exact": totals["known_object_exact"] / images,
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "predicted_object_exact": totals["predicted_object_exact"] / images,
        "oracle_junction_sdf_exact": totals["oracle_junction_sdf_exact"] / images,
        "display_count_exact": all_display_counts_exact / images,
        "display_count_accuracy_by_type": {
            name: display_count_exact[name] / images for name in BOND_TYPES
        },
        "display_count_positive_images": display_count_support,
        "image_max_endpoint_error_p95": (
            float(np.percentile(finite_errors, 95)) if finite_errors else None
        ),
        "image_max_endpoint_error_max": max(finite_errors) if finite_errors else None,
        "object_failure_keys": object_failure_keys,
        "sdf_failure_keys": sdf_failure_keys,
    }


def _set_vision_training(model, mode: str) -> list[torch.nn.Parameter]:
    model.vision_encoder.requires_grad_(False)
    if mode == "none":
        return []
    if mode == "stage0":
        modules = [model.vision_encoder.stages[0]]
    elif mode == "stem-stage0":
        modules = [model.vision_encoder.stem, model.vision_encoder.stages[0]]
    else:
        raise ValueError(f"Unknown vision training mode: {mode}")
    for module in modules:
        module.requires_grad_(True)
    return [parameter for module in modules for parameter in module.parameters()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=3e-6)
    parser.add_argument(
        "--train-vision", choices=("none", "stage0", "stem-stage0"), default="none"
    )
    parser.add_argument("--endpoint-tail-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-barrier-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-barrier-margin-pixels", type=float, default=4.0)
    parser.add_argument("--validate-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    device = torch.device("cuda")
    _, payload = load_grounded_graph_checkpoint(args.grounded_graph_checkpoint, device)
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(payload["grounded_selector_checkpoint"]), device
    )
    config = BondObjectConfig(input_channels=model.config.vision_dims[0])
    detector = BondObjectDetector(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        detector.load_state_dict(initial["bond_object"])
        if initial.get("vision_encoder") is not None:
            model.vision_encoder.load_state_dict(initial["vision_encoder"])

    vision_parameters = _set_vision_training(model, args.train_vision)
    parameter_groups = [{"params": detector.parameters(), "lr": args.learning_rate}]
    if vision_parameters:
        parameter_groups.append(
            {"params": vision_parameters, "lr": args.vision_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    collator = ChemSceneBondCollator()

    def cache(shards: list[str], count: int) -> list[dict]:
        dataset = ChemVectorIterableDataset(
            expand_shards(shards), sequence_version="0.4"
        )
        return list(islice(iter(dataset), count))

    train_records = cache(args.train_shards, args.train_cache_samples)
    validation_records = cache(args.validation_shards, args.validation_samples)
    if len(train_records) != args.train_cache_samples or len(validation_records) != args.validation_samples:
        raise ValueError("ChemScene Bond Query cache is incomplete")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best = (-1.0, -1.0, -float("inf"))
    running_loss = 0.0
    running_samples = 0
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            detector.train()
            if vision_parameters:
                model.vision_encoder.train()
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = collator([train_records[index] for index in indices])
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            vision_context = (
                torch.enable_grad() if vision_parameters else torch.no_grad()
            )
            with vision_context, torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                features = model.vision_encoder.forward_stroke_resolution(pixels)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = detector(features)
                losses = bond_object_loss(
                    output,
                    targets,
                    endpoint_tail_weight=args.endpoint_tail_weight,
                    endpoint_barrier_weight=args.endpoint_barrier_weight,
                    endpoint_barrier_margin_pixels=args.endpoint_barrier_margin_pixels,
                )
            losses["loss"].backward()
            trainable = list(detector.parameters()) + vision_parameters
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_loss += float(losses["loss"].detach()) * len(indices)
            running_samples += len(indices)
            if samples_seen // args.log_every != (samples_seen - len(indices)) // args.log_every:
                print(
                    f"samples={samples_seen} loss={running_loss / running_samples:.6f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
                metrics = evaluate(
                    model, detector, validation_records, collator, device, args.batch_size
                )
                print("validation=" + json.dumps(metrics), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                torch.save(
                    {
                        "bond_object": detector.state_dict(),
                        "vision_encoder": (
                            model.vision_encoder.state_dict() if vision_parameters else None
                        ),
                        "config": asdict(config),
                        "samples_seen": samples_seen,
                        "validation": metrics,
                        "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
                    },
                    checkpoint,
                )
                score = (
                    metrics["oracle_junction_sdf_exact"],
                    metrics["predicted_object_exact"],
                    -metrics["image_max_endpoint_error_p95"],
                )
                if score > best:
                    best = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    metrics = evaluate(
        model, detector, validation_records, collator, device, args.batch_size
    )
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "bond_object": detector.state_dict(),
            "vision_encoder": (
                model.vision_encoder.state_dict() if vision_parameters else None
            ),
            "config": asdict(config),
            "samples_seen": samples_seen,
            "validation": metrics,
            "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
        },
        output_dir / "last.pt",
    )
    print("validation=" + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
