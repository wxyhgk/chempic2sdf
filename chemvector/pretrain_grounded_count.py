#!/usr/bin/env python3
"""Train independent cardinality calibration over frozen grounded graph outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.grounded_count import (
    GroundedCountCalibrator,
    GroundedCountConfig,
    grounded_count_loss,
    predict_grounded_counts,
)
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.grounded_selector import decode_grounded_queries
from chemvector.pretrain_grounded_graph import (
    frozen_candidates,
    load_frozen_grounded_pipeline,
)
from chemvector.pretrain_vertex import (
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.train import expand_shards


@torch.inference_mode()
def prepare_cache(
    base_model,
    detector,
    selector,
    graph,
    loader,
    device: torch.device,
    *,
    samples: int,
    dense_threshold: float,
    candidate_nms_kernel_size: int,
    cache_name: str,
) -> tuple[list[dict], torch.Tensor]:
    records = []
    pair_indices = None
    graph.eval()
    for batch in loader:
        if len(records) >= samples:
            break
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        grounded = frozen_candidates(
            base_model,
            detector,
            selector,
            pixels,
            dense_threshold=dense_threshold,
            maximum_candidates=graph.config.maximum_candidates,
            candidate_nms_kernel_size=candidate_nms_kernel_size,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = graph(
                grounded["candidate_hidden"],
                pixels,
                grounded["grounded_centers"],
                grounded["candidate_mask"],
                grounded["selector_logits"],
            )
        if pair_indices is None:
            pair_indices = output["pair_indices"].cpu()
        elif not torch.equal(pair_indices, output["pair_indices"].cpu()):
            raise ValueError("Grounded graph pair ordering changed inside the cache")
        for batch_index in range(pixels.size(0)):
            valid = batch["anchor_mask"][batch_index]
            records.append(
                {
                    "key": batch["keys"][batch_index],
                    "node_logits": output["node_logits"][batch_index].half().cpu(),
                    "candidate_mask": output["candidate_mask"][batch_index].cpu(),
                    "edge_logits": output["edge_logits"][batch_index].half().cpu(),
                    "centers": grounded["grounded_centers"][batch_index].float().cpu(),
                    "aromatic_logits": grounded["aromatic_logits"][batch_index].half().cpu(),
                    "target_count": int(valid.sum()),
                    "anchor_points": batch["anchor_points"][batch_index][valid].cpu(),
                    "aromatic_targets": batch["aromatic_targets"][batch_index][valid].cpu(),
                }
            )
            if len(records) >= samples:
                break
        if len(records) % 100 == 0:
            print(f"{cache_name}_cached={len(records)}", flush=True)
    if len(records) != samples or pair_indices is None:
        raise ValueError(f"Requested {samples} {cache_name} samples, found {len(records)}")
    return records, pair_indices


def stack_records(records: list[dict], indices: list[int], device: torch.device) -> dict:
    return {
        name: torch.stack([records[index][name] for index in indices]).to(
            device, non_blocking=True
        )
        for name in ("node_logits", "candidate_mask", "edge_logits")
    }


@torch.no_grad()
def evaluate_cached(
    calibrator,
    records: list[dict],
    pair_indices: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> dict:
    calibrator.eval()
    calibrated_totals = new_detection_totals()
    known_totals = new_detection_totals()
    loss_total = 0.0
    count_correct = 0
    count_error = 0
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_records(records, indices, device)
        targets = torch.tensor(
            [records[index]["target_count"] for index in indices],
            device=device,
            dtype=torch.long,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            count_logits = calibrator(
                batch["node_logits"],
                batch["candidate_mask"],
                batch["edge_logits"],
                pair_indices,
            )
            loss = grounded_count_loss(count_logits, targets)
        counts = predict_grounded_counts(count_logits)
        calibrated = decode_grounded_queries(
            batch["node_logits"],
            torch.stack([records[index]["centers"] for index in indices]).to(device),
            torch.stack(
                [records[index]["aromatic_logits"] for index in indices]
            ).to(device),
            counts=counts,
            candidate_mask=batch["candidate_mask"],
        )
        known = decode_grounded_queries(
            batch["node_logits"],
            torch.stack([records[index]["centers"] for index in indices]).to(device),
            torch.stack(
                [records[index]["aromatic_logits"] for index in indices]
            ).to(device),
            counts=targets,
            candidate_mask=batch["candidate_mask"],
        )
        count_correct += int((counts == targets).sum())
        count_error += int((counts - targets).abs().sum())
        loss_total += float(loss) * len(indices)
        for offset, record_index in enumerate(indices):
            record = records[record_index]
            anchors = record["anchor_points"].unsqueeze(0)
            mask = torch.ones((1, anchors.size(1)), dtype=torch.bool)
            aromatic = record["aromatic_targets"].unsqueeze(0)
            update_detection_metrics(
                calibrated_totals,
                [calibrated[offset]],
                anchors,
                mask,
                aromatic,
                768,
                6.0,
            )
            update_detection_metrics(
                known_totals,
                [known[offset]],
                anchors,
                mask,
                aromatic,
                768,
                6.0,
            )
    return {
        "samples": len(records),
        "loss": loss_total / max(len(records), 1),
        "count_accuracy": count_correct / max(len(records), 1),
        "count_mae": count_error / max(len(records), 1),
        "calibrated": summarize_detection_metrics(calibrated_totals),
        "known_count": summarize_detection_metrics(known_totals),
    }


def validation_key(result: dict) -> tuple[float, ...]:
    return (
        result["calibrated"]["vertex_exact_rate"],
        result["count_accuracy"],
        result["calibrated"]["f1"],
        -result["count_mae"],
    )


def save_checkpoint(path, calibrator, optimizer, config, args, *, samples_seen, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "grounded_count": calibrator.state_dict(),
            "grounded_count_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
            "samples_seen": samples_seen,
            "validation": validation,
            "training_kind": "grounded_graph_independent_count_calibration",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--candidate-nms-kernel-size", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grounded count calibration")
    if min(
        args.train_cache_samples,
        args.validation_samples,
        args.train_samples,
        args.epochs,
        args.batch_size,
    ) < 1:
        raise ValueError("Grounded count training counts must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    graph, graph_payload = load_grounded_graph_checkpoint(
        args.grounded_graph_checkpoint, device
    )
    graph.requires_grad_(False)
    base_model, detector, selector, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    config = GroundedCountConfig(
        maximum_candidates=graph.config.maximum_candidates,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    calibrator = GroundedCountCalibrator(config).to(device)
    optimizer = torch.optim.AdamW(
        calibrator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def loader(shards, reject_file):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards),
                sequence_version="0.4",
                reject_keys=load_reject_keys(reject_file),
            ),
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    train_cache, pair_indices = prepare_cache(
        base_model,
        detector,
        selector,
        graph,
        loader(args.train_shards, args.train_reject_keys_file),
        device,
        samples=args.train_cache_samples,
        dense_threshold=args.dense_threshold,
        candidate_nms_kernel_size=args.candidate_nms_kernel_size,
        cache_name="train",
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
        and load_reject_keys(args.train_reject_keys_file)
        == load_reject_keys(args.validation_reject_keys_file)
    ):
        validation_cache = train_cache
    else:
        validation_cache, validation_pairs = prepare_cache(
            base_model,
            detector,
            selector,
            graph,
            loader(args.validation_shards, args.validation_reject_keys_file),
            device,
            samples=args.validation_samples,
            dense_threshold=args.dense_threshold,
            candidate_nms_kernel_size=args.candidate_nms_kernel_size,
            cache_name="validation",
        )
        if not torch.equal(pair_indices, validation_pairs):
            raise ValueError("Train and validation pair ordering differs")
    pair_indices = pair_indices.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    initial = evaluate_cached(
        calibrator,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
    )
    print("initial=" + json.dumps(initial), flush=True)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"trainable_parameters={sum(p.numel() for p in calibrator.parameters()):,}")

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    started = time.perf_counter()
    running_loss = 0.0
    running_samples = 0
    stop = False
    for _ in range(args.epochs):
        order = list(range(len(train_cache)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            indices = order[start : start + min(args.batch_size, remaining)]
            batch = stack_records(train_cache, indices, device)
            targets = torch.tensor(
                [train_cache[index]["target_count"] for index in indices],
                device=device,
                dtype=torch.long,
            )
            calibrator.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = calibrator(
                    batch["node_logits"],
                    batch["candidate_mask"],
                    batch["edge_logits"],
                    pair_indices,
                )
                loss = grounded_count_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(indices)
            running_samples += len(indices)
            running_loss += float(loss.detach()) * len(indices)
            if samples_seen // args.log_every != (
                samples_seen - len(indices)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} loss={running_loss / running_samples:.6f} "
                    f"samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if args.validate_every > 0 and samples_seen >= next_validation:
                validation = evaluate_cached(
                    calibrator,
                    validation_cache,
                    pair_indices,
                    device,
                    batch_size=args.batch_size,
                )
                print("validation=" + json.dumps(validation), flush=True)
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                save_checkpoint(
                    checkpoint,
                    calibrator,
                    optimizer,
                    config,
                    args,
                    samples_seen=samples_seen,
                    validation=validation,
                )
                if validation_key(validation) > best_score:
                    best_score = validation_key(validation)
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    validation = evaluate_cached(
        calibrator,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "last.pt",
        calibrator,
        optimizer,
        config,
        args,
        samples_seen=samples_seen,
        validation=validation,
    )
    print("validation=" + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
