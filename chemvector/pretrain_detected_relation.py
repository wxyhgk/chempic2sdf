#!/usr/bin/env python3
"""Overfit relation heads on frozen Atom Query plus dense-snapped vertices."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from Data.sdf.chemsvg_sequence_v4 import ParsedChemSVGSequenceV4, parse_chemsvg_v4
from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    apply_atom_query_dense_head,
    apply_atom_query_vision_updates,
    decode_atom_queries_by_set_count,
    load_atom_query_state,
    snap_queries_to_dense_candidates,
)
from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.evaluate_atom_query_skeleton import vertex_records
from chemvector.evaluate_skeleton import (
    compare_skeleton,
    parse_thresholds,
    skeleton_collate,
    summarize,
)
from chemvector.pretrain_relation import load_relation_model, set_relation_trainable
from chemvector.relation import (
    factorized_relation_probabilities,
    hard_mined_factorized_relation_loss,
    new_relation_totals,
    summarize_relation_metrics,
    update_relation_metrics,
)
from chemvector.stroke import STROKE_CLASS_TO_ID, all_pair_indices
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def detected_pair_labels(
    expected: ParsedChemSVGSequenceV4,
    vertices,
    pairs: list[tuple[int, int]],
    *,
    maximum_mapping_distance_pixels: float | None = None,
    image_size: int = 768,
) -> tuple[torch.Tensor, float]:
    if not expected.vertices or not vertices:
        return torch.zeros(len(pairs), dtype=torch.long), 0.0
    expected_points = np.asarray(
        [(vertex.x, vertex.y) for vertex in expected.vertices],
        dtype=np.float32,
    )
    predicted_points = np.asarray(
        [(vertex.x, vertex.y) for vertex in vertices],
        dtype=np.float32,
    )
    distances = np.linalg.norm(
        predicted_points[:, None, :] - expected_points[None, :, :],
        axis=-1,
    )
    predicted_indices, expected_indices = linear_sum_assignment(distances)
    assigned_distances_pixels = {
        int(predicted_index): (
            float(distances[predicted_index, expected_index]) / 1023.0 * image_size
        )
        for predicted_index, expected_index in zip(predicted_indices, expected_indices)
    }
    mapping = {
        int(predicted_index): int(expected_index)
        for predicted_index, expected_index in zip(predicted_indices, expected_indices)
        if maximum_mapping_distance_pixels is None
        or assigned_distances_pixels[int(predicted_index)]
        <= maximum_mapping_distance_pixels
    }
    expected_coordinate_to_index = {
        (vertex.x, vertex.y): index for index, vertex in enumerate(expected.vertices)
    }
    expected_edges = {
        tuple(
            sorted(
                (
                    expected_coordinate_to_index[stroke.begin],
                    expected_coordinate_to_index[stroke.end],
                )
            )
        ): STROKE_CLASS_TO_ID[stroke.display_type]
        for stroke in expected.strokes
    }
    labels = []
    for begin, end in pairs:
        if begin not in mapping or end not in mapping:
            labels.append(STROKE_CLASS_TO_ID["NONE"])
            continue
        mapped = tuple(sorted((mapping[begin], mapping[end])))
        labels.append(expected_edges.get(mapped, STROKE_CLASS_TO_ID["NONE"]))
    maximum_distance = max(
        (
            assigned_distances_pixels[int(predicted_index)]
            for predicted_index in predicted_indices
        ),
        default=0.0,
    )
    return torch.tensor(labels, dtype=torch.long), maximum_distance


@torch.inference_mode()
def prepare_cache(
    relation_model,
    query_detector,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    samples: int,
    cache_name: str,
) -> list[dict]:
    relation_model.eval()
    query_detector.eval()
    records = []
    for batch in loader:
        if len(records) >= samples:
            break
        pixels = batch["pixel_values"].to(device)
        expected = parse_chemsvg_v4(batch["target_sequences"][0])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stroke_features = relation_model.vision_encoder.forward_stroke_resolution(
                pixels
            )
            if args.feature_level == "stroke":
                query_map = stroke_features
            else:
                query_map = (
                    relation_model.vision_encoder.forward_high_resolution_from_stroke(
                        stroke_features
                    )
                )
            query_output = query_detector(query_map)
            dense_output = relation_model.vertex_head(stroke_features)
        queries = decode_atom_queries_by_set_count(query_output)[0]
        candidates = decode_vertices(
            dense_output,
            threshold=args.dense_threshold,
            maximum_vertices=128,
        )[0]
        detections = snap_queries_to_dense_candidates(
            queries,
            candidates,
            image_size=768,
            maximum_distance_pixels=args.snap_distance_pixels,
        )
        vertices = vertex_records(detections)
        pairs = all_pair_indices(len(vertices))
        anchors = torch.tensor(
            [[[vertex.x / 1023.0, vertex.y / 1023.0] for vertex in vertices]],
            device=device,
            dtype=torch.float32,
        )
        pair_indices = torch.tensor([pairs], device=device, dtype=torch.long)
        pair_mask = torch.ones((1, len(pairs)), device=device, dtype=torch.bool)
        pair_features = None
        if not args.train_pair_encoder:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pair_features = relation_model.stroke_head.encode_pairs(
                    stroke_features,
                    anchors,
                    pair_indices,
                    pair_mask,
                    pixels.shape[-2:],
                )
        labels, maximum_mapping_distance = detected_pair_labels(
            expected,
            vertices,
            pairs,
            maximum_mapping_distance_pixels=args.maximum_mapping_distance_pixels,
        )
        records.append(
            {
                "key": batch["keys"][0],
                "expected": expected,
                "vertices": vertices,
                "pairs": pairs,
                "pair_features": (
                    None if pair_features is None else pair_features.float().cpu()
                ),
                "stroke_features": (
                    stroke_features.bfloat16().cpu()
                    if args.train_pair_encoder
                    else None
                ),
                "anchors": anchors.cpu(),
                "pair_indices": pair_indices.cpu(),
                "pair_mask": pair_mask.cpu(),
                "labels": labels,
                "maximum_mapping_distance_pixels": maximum_mapping_distance,
            }
        )
        if len(records) % 20 == 0:
            print(f"{cache_name}_cached={len(records)}", flush=True)
    if len(records) != samples:
        raise ValueError(f"Requested {samples} {cache_name} samples, found {len(records)}")
    return records


def record_pair_features(model, record: dict, device: torch.device, *, train: bool):
    if record.get("stroke_features") is not None:
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            return model.stroke_head.encode_pairs(
                record["stroke_features"].to(device, non_blocking=True),
                record["anchors"].to(device, non_blocking=True),
                record["pair_indices"].to(device, non_blocking=True),
                record["pair_mask"].to(device, non_blocking=True),
                (768, 768),
            )
    return record["pair_features"].to(device, non_blocking=True)


@torch.no_grad()
def validate_cached(model, records: list[dict], device, args) -> dict:
    model.relation_head.eval()
    comparisons = []
    relation_totals = {
        threshold: new_relation_totals() for threshold in args.edge_thresholds
    }
    for record in records:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = record_pair_features(model, record, device, train=False)
            output = model.relation_head(features)
        probabilities = factorized_relation_probabilities(
            output["edge_logits"],
            output["bond_logits"],
        )
        labels = record["labels"].to(device, non_blocking=True)
        edge_logits = output["edge_logits"].reshape(-1)
        bond_logits = output["bond_logits"].reshape(-1, 3)
        for threshold in args.edge_thresholds:
            update_relation_metrics(
                relation_totals[threshold],
                edge_logits,
                bond_logits,
                labels,
                edge_threshold=threshold,
            )
        comparisons.append(
            {
                "key": record["key"],
                "modes": {
                    "detected_relation": {
                        f"{threshold:.4f}": asdict(
                            compare_skeleton(
                                record["expected"],
                                record["vertices"],
                                record["pairs"],
                                probabilities,
                                edge_threshold=threshold,
                                tolerance_pixels=args.vertex_tolerance_pixels,
                                image_size=768,
                            )
                        )
                        for threshold in args.edge_thresholds
                    }
                },
            }
        )
    summary = summarize(
        comparisons,
        ("detected_relation",),
        args.edge_thresholds,
    )
    threshold_rows = summary["modes"]["detected_relation"]
    best_threshold, best_metrics = max(
        threshold_rows.items(),
        key=lambda item: validation_score(item[1]),
    )
    summary["best_threshold"] = best_threshold
    summary["best_metrics"] = best_metrics
    relation_summary = {
        f"{threshold:.4f}": summarize_relation_metrics(relation_totals[threshold])
        for threshold in args.edge_thresholds
    }
    best_relation_threshold, best_relation_metrics = max(
        relation_summary.items(),
        key=lambda item: (
            item[1]["edge_f1"],
            item[1]["visible_bond_accuracy"],
            item[1]["edge_precision"],
        ),
    )
    summary["relation_supervision"] = relation_summary
    summary["best_relation_threshold"] = best_relation_threshold
    summary["best_relation_metrics"] = best_relation_metrics
    return summary


def validation_score(metrics: dict) -> tuple[float, float, float, float]:
    precision = metrics["edge_precision"]
    recall = metrics["edge_recall"]
    edge_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return (
        metrics["skeleton_exact_rate"],
        metrics["edge_set_exact_rate"],
        edge_f1,
        precision,
    )


def save_checkpoint(path, model, optimizer, args, *, step, samples, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "global_step": step,
            "samples_seen": samples,
            "validation": validation,
            "training_kind": "shared_dense_snapped_detected_relation",
            "vertex_pipeline": {
                "query_checkpoint": args.query_checkpoint,
                "query_decode_mode": "set_count",
                "dense_head": "packed_in_query_checkpoint",
                "shared_stroke_feature": True,
                "dense_threshold": args.dense_threshold,
                "snap_distance_pixels": args.snap_distance_pixels,
                "maximum_mapping_distance_pixels": (
                    args.maximum_mapping_distance_pixels
                ),
            },
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--minimum-negatives", type=int, default=32)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--edge-pos-weight", type=float, default=1.0)
    parser.add_argument("--edge-focal-gamma", type=float, default=2.0)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--bond-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--bond-class-weights",
        type=float,
        nargs=3,
        default=(1.0, 2.0, 6.0),
    )
    parser.add_argument("--feature-level", choices=("stroke", "high"), default="stroke")
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--snap-distance-pixels", type=float, default=64.0)
    parser.add_argument("--maximum-mapping-distance-pixels", type=float, default=12.0)
    parser.add_argument("--vertex-tolerance-pixels", type=float, default=6.0)
    parser.add_argument(
        "--edge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.50,0.60,0.70,0.80,0.90"),
    )
    parser.add_argument("--stroke-band", type=int, default=9)
    parser.add_argument("--stroke-band-radius", type=float, default=12.0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--disable-relation-dropout", action="store_true")
    parser.add_argument("--train-pair-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for detected relation training")
    if min(
        args.train_cache_samples,
        args.validation_samples,
        args.train_samples,
        args.epochs,
    ) < 1:
        raise ValueError("Training counts must be positive")
    if args.maximum_mapping_distance_pixels <= 0:
        raise ValueError("maximum-mapping-distance-pixels must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_path = Path(args.init_checkpoint)
    model = load_relation_model(
        checkpoint_path,
        device,
        stroke_band=args.stroke_band,
        stroke_band_radius=args.stroke_band_radius,
    )
    query_payload = torch.load(
        args.query_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    query_config = AtomQueryConfig(**query_payload["atom_query_config"])
    query_detector = AtomQueryDetector(query_config).to(device)
    load_atom_query_state(query_detector, query_payload["atom_query"])
    query_detector.eval()
    apply_atom_query_vision_updates(model, query_payload)
    apply_atom_query_dense_head(model, query_payload)
    reject_keys = load_reject_keys(args.reject_keys_file)
    print(f"reject_keys={len(reject_keys)}")
    def make_loader(shards):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
                reject_keys=reject_keys,
            ),
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda samples: skeleton_collate(samples, 768),
        )

    train_cache = prepare_cache(
        model,
        query_detector,
        make_loader(args.train_shards),
        device,
        args,
        samples=args.train_cache_samples,
        cache_name="train",
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
    ):
        validation_cache = train_cache
    else:
        validation_cache = prepare_cache(
            model,
            query_detector,
            make_loader(args.validation_shards),
            device,
            args,
            samples=args.validation_samples,
            cache_name="validation",
        )
    for name, cache in (("train", train_cache), ("validation", validation_cache)):
        print(
            f"{name}_cache="
            + json.dumps(
                {
                    "samples": len(cache),
                    "pairs": sum(len(record["pairs"]) for record in cache),
                    "maximum_mapping_distance_pixels": max(
                        record["maximum_mapping_distance_pixels"] for record in cache
                    ),
                },
                ensure_ascii=False,
            )
        )

    trainable = set_relation_trainable(
        model,
        train_pair_encoder=args.train_pair_encoder,
    )
    if args.disable_relation_dropout:
        dropout_roots = [model.relation_head]
        if args.train_pair_encoder:
            dropout_roots.append(model.stroke_head)
        for root in dropout_roots:
            for module in root.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(model.config), indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer = checkpoint_path.parent / "tokenizer.json"
    if tokenizer.is_file():
        shutil.copy2(tokenizer, output_dir / "tokenizer.json")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(parameter.numel() for parameter in trainable):,}")
    rng = random.Random(args.seed)
    step = 0
    samples_seen = 0
    best_score = (-1.0, -1.0, -1.0, -1.0)
    started = time.perf_counter()
    running = {name: 0.0 for name in ("loss", "edge_loss", "bond_loss", "ranking_loss")}
    interval = 0
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        for record_index in order:
            step += 1
            samples_seen += 1
            interval += 1
            model.eval()
            model.relation_head.train()
            if args.train_pair_encoder:
                model.stroke_head.train()
            labels = train_cache[record_index]["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = record_pair_features(
                    model,
                    train_cache[record_index],
                    device,
                    train=args.train_pair_encoder,
                )
                output = model.relation_head(features)
                losses = hard_mined_factorized_relation_loss(
                    output["edge_logits"],
                    output["bond_logits"],
                    labels,
                    negative_ratio=args.negative_ratio,
                    minimum_negatives=args.minimum_negatives,
                    ranking_weight=args.ranking_weight,
                    ranking_margin=args.ranking_margin,
                    edge_pos_weight=args.edge_pos_weight,
                    edge_focal_gamma=args.edge_focal_gamma,
                    bond_class_weights=tuple(args.bond_class_weights),
                    edge_loss_weight=args.edge_loss_weight,
                    bond_loss_weight=args.bond_loss_weight,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for name in running:
                running[name] += float(losses[name].detach())
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples_seen} "
                    + " ".join(
                        f"{name}={running[name] / interval:.6f}" for name in running
                    )
                    + f" samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                interval = 0
            if args.validate_every > 0 and step % args.validate_every == 0:
                validation = validate_cached(model, validation_cache, device, args)
                print(
                    "validation=" + json.dumps(validation["best_metrics"], ensure_ascii=False),
                    flush=True,
                )
                save_checkpoint(
                    output_dir / f"checkpoint-{step:07d}.pt",
                    model,
                    optimizer,
                    args,
                    step=step,
                    samples=samples_seen,
                    validation=validation,
                )
                current_score = validation_score(validation["best_metrics"])
                if current_score > best_score:
                    best_score = current_score
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        args,
                        step=step,
                        samples=samples_seen,
                        validation=validation,
                    )
            if samples_seen >= args.train_samples:
                stop = True
                break
        if stop:
            break

    validation = validate_cached(model, validation_cache, device, args)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        args,
        step=step,
        samples=samples_seen,
        validation=validation,
    )
    print("validation=" + json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
