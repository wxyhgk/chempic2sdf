#!/usr/bin/env python3
"""Train a lightweight selector after all-query dense peak snapping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    apply_atom_query_dense_head,
    apply_atom_query_vision_updates,
    load_atom_query_state,
)
from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.grounded_selector import (
    GroundedQuerySelector,
    GroundedSelectorConfig,
    decode_grounded_queries,
    dense_candidate_targets,
    fuse_dense_candidates_with_queries,
    ground_all_queries,
    grounded_selector_loss,
    grounded_selector_targets,
)
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator, decode_vertices


def load_frozen_pipeline(query_checkpoint: Path, device: torch.device):
    payload = torch.load(query_checkpoint, map_location="cpu", weights_only=False)
    base_model = load_vertex_model(Path(payload["base_checkpoint"]), device).eval()
    apply_atom_query_vision_updates(base_model, payload)
    apply_atom_query_dense_head(base_model, payload)
    query_config = AtomQueryConfig(**payload["atom_query_config"])
    detector = AtomQueryDetector(query_config).to(device)
    load_atom_query_state(detector, payload["atom_query"])
    detector.eval()
    return base_model, detector, query_config


@torch.inference_mode()
def prepare_cache(
    base_model,
    detector,
    loader,
    device: torch.device,
    *,
    samples: int,
    dense_threshold: float,
    snap_distance_pixels: float,
    cache_name: str,
    candidate_centric: bool,
) -> list[dict]:
    records = []
    for batch in loader:
        if len(records) >= samples:
            break
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stroke_features = base_model.vision_encoder.forward_stroke_resolution(pixels)
            output = detector(stroke_features)
            dense_output = base_model.vertex_head(stroke_features)
        dense_candidates = decode_vertices(
            dense_output,
            threshold=dense_threshold,
            maximum_vertices=128,
        )
        anchors = batch["anchor_points"].to(device, non_blocking=True)
        mask = batch["anchor_mask"].to(device, non_blocking=True)
        if candidate_centric:
            grounded = fuse_dense_candidates_with_queries(
                output,
                dense_candidates,
                image_size=pixels.shape[-1],
            )
            targets = dense_candidate_targets(
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                anchors,
                mask,
            )
        else:
            grounded = ground_all_queries(
                output,
                dense_candidates,
                image_size=pixels.shape[-1],
                maximum_distance_pixels=snap_distance_pixels,
            )
            grounded["candidate_mask"] = torch.ones_like(
                grounded["base_logits"],
                dtype=torch.bool,
            )
            targets = grounded_selector_targets(
                grounded["grounded_centers"],
                anchors,
                mask,
            )
        for batch_index in range(pixels.size(0)):
            valid = mask[batch_index]
            valid_cpu = batch["anchor_mask"][batch_index]
            records.append(
                {
                    "key": batch["keys"][batch_index],
                    "query_features": grounded["query_features"][batch_index]
                    .bfloat16()
                    .cpu(),
                    "scalar_features": grounded["scalar_features"][batch_index]
                    .half()
                    .cpu(),
                    "grounded_centers": grounded["grounded_centers"][batch_index]
                    .float()
                    .cpu(),
                    "base_logits": grounded["base_logits"][batch_index].half().cpu(),
                    "aromatic_logits": grounded["aromatic_logits"][batch_index]
                    .half()
                    .cpu(),
                    "candidate_mask": grounded["candidate_mask"][batch_index].cpu(),
                    "targets": targets[batch_index].cpu(),
                    "anchor_points": anchors[batch_index][valid].cpu(),
                    "aromatic_targets": batch["aromatic_targets"][batch_index][valid_cpu]
                    .cpu(),
                }
            )
            if len(records) >= samples:
                break
        if len(records) % 100 == 0:
            print(f"{cache_name}_cached={len(records)}", flush=True)
    if len(records) != samples:
        raise ValueError(f"Requested {samples} {cache_name} samples, found {len(records)}")
    return records


def stack_records(records: list[dict], indices: list[int], device: torch.device) -> dict:
    names = (
        "query_features",
        "scalar_features",
        "grounded_centers",
        "base_logits",
        "aromatic_logits",
        "candidate_mask",
        "targets",
    )
    return {
        name: torch.stack([records[index][name] for index in indices]).to(
            device,
            non_blocking=True,
        )
        for name in names
    }


@torch.no_grad()
def validate_cached(
    selector,
    records: list[dict],
    device: torch.device,
    *,
    batch_size: int,
    tolerance_pixels: float,
    loss_kwargs: dict,
) -> dict:
    selector.eval()
    totals = {
        name: new_detection_totals()
        for name in ("set_count", "known_count", "assignment_oracle")
    }
    loss_totals = {name: 0.0 for name in (
        "loss",
        "selection_loss",
        "ranking_loss",
        "set_count_loss",
        "cardinality_loss",
    )}
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_records(records, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = selector(
                batch["query_features"],
                batch["scalar_features"],
                batch["grounded_centers"],
                batch["base_logits"],
                batch["candidate_mask"],
            )
            losses = grounded_selector_loss(
                logits,
                batch["targets"],
                candidate_mask=batch["candidate_mask"],
                **loss_kwargs,
            )
        for name in loss_totals:
            loss_totals[name] += float(losses[name]) * len(indices)
        expected_counts = batch["targets"].sum(dim=1).long()
        decoded = {
            "set_count": decode_grounded_queries(
                logits,
                batch["grounded_centers"],
                batch["aromatic_logits"],
                candidate_mask=batch["candidate_mask"],
            ),
            "known_count": decode_grounded_queries(
                logits,
                batch["grounded_centers"],
                batch["aromatic_logits"],
                counts=expected_counts,
                candidate_mask=batch["candidate_mask"],
            ),
            "assignment_oracle": decode_grounded_queries(
                (batch["targets"] * 2.0 - 1.0) * 100.0,
                batch["grounded_centers"],
                batch["aromatic_logits"],
                counts=expected_counts,
                candidate_mask=batch["candidate_mask"],
            ),
        }
        for offset, record_index in enumerate(indices):
            record = records[record_index]
            anchors = record["anchor_points"].unsqueeze(0)
            mask = torch.ones((1, anchors.size(1)), dtype=torch.bool)
            aromatic = record["aromatic_targets"].unsqueeze(0)
            for name in totals:
                update_detection_metrics(
                    totals[name],
                    [decoded[name][offset]],
                    anchors,
                    mask,
                    aromatic,
                    768,
                    tolerance_pixels,
                )
    return {
        "samples": len(records),
        "modes": {
            name: summarize_detection_metrics(mode_totals)
            for name, mode_totals in totals.items()
        },
        "losses": {
            name: value / max(len(records), 1) for name, value in loss_totals.items()
        },
    }


def validation_score(validation: dict) -> tuple[float, float, float, float]:
    deployed = validation["modes"]["set_count"]
    known = validation["modes"]["known_count"]
    return (
        deployed["f1"],
        deployed["count_exact_rate"],
        known["f1"],
        -deployed["count_mae"],
    )


def save_checkpoint(path, selector, optimizer, config, args, *, samples, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "grounded_selector": selector.state_dict(),
            "grounded_selector_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "query_checkpoint": args.query_checkpoint,
            "samples_seen": samples,
            "validation": validation,
            "training_kind": (
                "dense_candidate_query_fusion_selector"
                if args.candidate_centric
                else "post_snap_grounded_query_selector"
            ),
            "candidate_centric": args.candidate_centric,
            "dense_threshold": args.dense_threshold,
            "snap_distance_pixels": args.snap_distance_pixels,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--set-count-weight", type=float, default=1.0)
    parser.add_argument("--cardinality-weight", type=float, default=0.1)
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--snap-distance-pixels", type=float, default=64.0)
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--candidate-centric", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grounded selector training")
    if min(
        args.train_cache_samples,
        args.validation_samples,
        args.train_samples,
        args.epochs,
        args.batch_size,
    ) < 1:
        raise ValueError("Grounded selector training counts must be positive")
    if args.learning_rate <= 0 or min(
        args.ranking_weight,
        args.set_count_weight,
        args.cardinality_weight,
    ) < 0:
        raise ValueError("Grounded selector learning rate or loss weights are invalid")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    base_model, detector, query_config = load_frozen_pipeline(
        Path(args.query_checkpoint),
        device,
    )
    selector_config = GroundedSelectorConfig(query_dim=query_config.d_model)
    selector = GroundedQuerySelector(selector_config).to(device)
    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_reject_keys = load_reject_keys(args.train_reject_keys_file)
    validation_reject_keys = load_reject_keys(args.validation_reject_keys_file)

    def make_loader(shards, reject_keys):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
                reject_keys=reject_keys,
            ),
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=VertexCollator(),
        )

    train_cache = prepare_cache(
        base_model,
        detector,
        make_loader(args.train_shards, train_reject_keys),
        device,
        samples=args.train_cache_samples,
        dense_threshold=args.dense_threshold,
        snap_distance_pixels=args.snap_distance_pixels,
        cache_name="train",
        candidate_centric=args.candidate_centric,
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
        and train_reject_keys == validation_reject_keys
    ):
        validation_cache = train_cache
    else:
        validation_cache = prepare_cache(
            base_model,
            detector,
            make_loader(args.validation_shards, validation_reject_keys),
            device,
            samples=args.validation_samples,
            dense_threshold=args.dense_threshold,
            snap_distance_pixels=args.snap_distance_pixels,
            cache_name="validation",
            candidate_centric=args.candidate_centric,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(selector_config), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    loss_kwargs = {
        "ranking_weight": args.ranking_weight,
        "set_count_weight": args.set_count_weight,
        "cardinality_weight": args.cardinality_weight,
    }
    initial = validate_cached(
        selector,
        validation_cache,
        device,
        batch_size=args.batch_size,
        tolerance_pixels=args.tolerance_pixels,
        loss_kwargs=loss_kwargs,
    )
    print("initial=" + json.dumps(initial, ensure_ascii=False), flush=True)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(p.numel() for p in selector.parameters()):,}")

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, -1.0, -float("inf"))
    running = {name: 0.0 for name in initial["losses"]}
    running_samples = 0
    started = time.perf_counter()
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            indices = indices[:remaining]
            batch = stack_records(train_cache, indices, device)
            selector.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = selector(
                    batch["query_features"],
                    batch["scalar_features"],
                    batch["grounded_centers"],
                    batch["base_logits"],
                    batch["candidate_mask"],
                )
                losses = grounded_selector_loss(
                    logits,
                    batch["targets"],
                    candidate_mask=batch["candidate_mask"],
                    **loss_kwargs,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_samples += len(indices)
            for name in running:
                running[name] += float(losses[name].detach()) * len(indices)
            if samples_seen // args.log_every != (
                samples_seen - len(indices)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} "
                    + " ".join(
                        f"{name}={running[name] / max(running_samples, 1):.6f}"
                        for name in running
                    )
                    + f" samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                running_samples = 0
            if args.validate_every > 0 and samples_seen >= next_validation:
                validation = validate_cached(
                    selector,
                    validation_cache,
                    device,
                    batch_size=args.batch_size,
                    tolerance_pixels=args.tolerance_pixels,
                    loss_kwargs=loss_kwargs,
                )
                print("validation=" + json.dumps(validation, ensure_ascii=False), flush=True)
                save_checkpoint(
                    output_dir / f"checkpoint-{samples_seen:07d}.pt",
                    selector,
                    optimizer,
                    selector_config,
                    args,
                    samples=samples_seen,
                    validation=validation,
                )
                score = validation_score(validation)
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        output_dir / "best.pt",
                        selector,
                        optimizer,
                        selector_config,
                        args,
                        samples=samples_seen,
                        validation=validation,
                    )
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    validation = validate_cached(
        selector,
        validation_cache,
        device,
        batch_size=args.batch_size,
        tolerance_pixels=args.tolerance_pixels,
        loss_kwargs=loss_kwargs,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir / "last.pt",
        selector,
        optimizer,
        selector_config,
        args,
        samples=samples_seen,
        validation=validation,
    )
    print("validation=" + json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
