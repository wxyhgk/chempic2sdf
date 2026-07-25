#!/usr/bin/env python3
"""Train count calibration on a frozen dual-branch vertex selector."""

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

from chemvector.data import ChemVectorIterableDataset
from chemvector.dual_vertex_selector import ROLE_COUNT
from chemvector.dual_vertex_selector import (
    role_aware_match_count,
    role_suppressed_topk_indices,
)
from chemvector.grounded_count import (
    GroundedCountConfig,
    GroundedRoleCountCalibrator,
    grounded_role_count_loss,
    predict_grounded_counts,
)
from chemvector.pretrain_dual_vertex_selector import (
    load_dual_pipeline,
    prepare_cache as prepare_selector_cache,
    selector_forward,
    stack_records as stack_selector_records,
)
from chemvector.proposal_graph import ProposalGraphCollator
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
)
from chemvector.train import expand_shards


def load_pipeline(selector_checkpoint: str, device: torch.device):
    payload = torch.load(selector_checkpoint, map_location="cpu", weights_only=False)
    config = StructuredSelectorConfig(**payload["structured_selector_config"])
    selector = GroundedStructuredSelector(config).to(device)
    selector.load_state_dict(payload["dual_vertex_selector"])
    selector.eval().requires_grad_(False)
    pipeline = load_dual_pipeline(
        payload["text_object_checkpoint"],
        payload["implicit_vertex_checkpoint"],
        device,
        maximum_candidates=config.maximum_candidates,
    )
    return selector, payload, pipeline


@torch.inference_mode()
def compact_selector_cache(
    selector,
    records: list[dict],
    pair_indices: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> list[dict]:
    compact = []
    selector.eval()
    device_pairs = pair_indices.to(device)
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_selector_records(records, indices, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            node_logits = selector_forward(selector, batch, device_pairs)
        for offset, record_index in enumerate(indices):
            source = records[record_index]
            compact.append(
                {
                    "key": source["key"],
                    "node_logits": node_logits[offset].half().cpu(),
                    "candidate_mask": source["candidate_mask"],
                    "candidate_roles": source["candidate_roles"],
                    "edge_logits": source["edge_logits"],
                    "targets": source["targets"],
                    "target_count": source["target_count"],
                    "target_role_counts": source["target_role_counts"],
                    "points": source["points"],
                    "anchor_points": source["anchor_points"],
                    "anchor_roles": source["anchor_roles"],
                }
            )
    return compact


COUNT_STACK_FIELDS = (
    "node_logits",
    "candidate_mask",
    "candidate_roles",
    "edge_logits",
    "targets",
)


def stack_count_records(records, indices, device):
    return {
        name: torch.stack([records[index][name] for index in indices]).to(
            device, non_blocking=True
        )
        for name in COUNT_STACK_FIELDS
    }


def summarize_strict(totals: dict[str, int], samples: int) -> dict[str, float]:
    precision = totals["matches"] / max(totals["predicted"], 1)
    recall = totals["matches"] / max(totals["expected"], 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "exact_rate": totals["exact"] / max(samples, 1),
    }


@torch.no_grad()
def evaluate_cached(
    calibrator,
    records,
    pair_indices,
    device,
    *,
    batch_size: int,
) -> dict:
    calibrator.eval()
    count_correct = 0
    count_error = 0
    per_role_correct = torch.zeros(ROLE_COUNT, dtype=torch.long)
    loss_total = 0.0
    calibrated_totals = {"matches": 0, "predicted": 0, "expected": 0, "exact": 0}
    known_totals = {"matches": 0, "predicted": 0, "expected": 0, "exact": 0}
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = stack_count_records(records, indices, device)
        expected = torch.stack(
            [records[index]["target_role_counts"] for index in indices]
        ).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            count_logits = calibrator(
                batch["node_logits"],
                batch["candidate_mask"],
                batch["edge_logits"],
                pair_indices,
                batch["candidate_roles"],
            )
            loss = grounded_role_count_loss(count_logits, expected)
        predicted = predict_grounded_counts(count_logits)
        count_correct += int((predicted == expected).all(dim=1).sum())
        per_role_correct += (predicted == expected).sum(dim=0).cpu()
        count_error += int((predicted - expected).abs().sum())
        loss_total += float(loss) * len(indices)
        for offset, record_index in enumerate(indices):
            record = records[record_index]
            for destination, counts in (
                (calibrated_totals, predicted[offset]),
                (known_totals, expected[offset]),
            ):
                selected = role_suppressed_topk_indices(
                    batch["node_logits"][offset],
                    record["points"].to(device),
                    batch["candidate_mask"][offset],
                    batch["candidate_roles"][offset],
                    counts,
                    suppression_radius_pixels=6.0,
                    image_size=768,
                )
                matches = role_aware_match_count(
                    record["points"].to(device)[selected],
                    batch["candidate_roles"][offset, selected],
                    record["anchor_points"].to(device),
                    record["anchor_roles"].to(device),
                    matching_radius_pixels=6.0,
                    image_size=768,
                )
                expected_total = int(expected[offset].sum())
                destination["matches"] += matches
                destination["predicted"] += len(selected)
                destination["expected"] += expected_total
                destination["exact"] += int(
                    len(selected) == expected_total and matches == expected_total
                )
    return {
        "samples": len(records),
        "loss": loss_total / max(len(records), 1),
        "count_accuracy": count_correct / max(len(records), 1),
        "count_mae": count_error / max(len(records), 1),
        "per_role_count_accuracy": [
            int(value) / max(len(records), 1) for value in per_role_correct
        ],
        "calibrated": summarize_strict(calibrated_totals, len(records)),
        "known_count": summarize_strict(known_totals, len(records)),
    }


def validation_key(result: dict) -> tuple[float, float, float, float]:
    return (
        result["calibrated"]["exact_rate"],
        result["count_accuracy"],
        result["calibrated"]["f1"],
        -result["count_mae"],
    )


def save_checkpoint(path, calibrator, optimizer, config, args, samples_seen, validation):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "dual_vertex_count": calibrator.state_dict(),
            "grounded_count_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "dual_vertex_selector_checkpoint": args.dual_vertex_selector_checkpoint,
            "samples_seen": samples_seen,
            "validation": validation,
            "training_kind": "dual_branch_role_aware_count_calibration",
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-vertex-selector-checkpoint", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dual vertex count calibration")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    selector, selector_payload, pipeline = load_pipeline(
        args.dual_vertex_selector_checkpoint, device
    )
    base, detector, grounded_selector, graph, text_head, implicit_head = pipeline
    selector_config = StructuredSelectorConfig(
        **selector_payload["structured_selector_config"]
    )
    config = GroundedCountConfig(
        maximum_candidates=selector_config.maximum_candidates,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        role_count=ROLE_COUNT,
    )
    calibrator = GroundedRoleCountCalibrator(config).to(device)
    optimizer = torch.optim.AdamW(
        calibrator.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def loader(shards):
        return DataLoader(
            ChemVectorIterableDataset(
                expand_shards(shards), sequence_version="0.4"
            ),
            batch_size=args.cache_batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=ProposalGraphCollator(),
        )

    cache_kwargs = {
        "text_candidate_threshold": 0.05,
        "text_filter_threshold": 0.3,
        "implicit_candidate_threshold": 0.05,
        "bbox_expansion_pixels": 8.0,
        "matching_radius_pixels": 6.0,
    }
    train_source, pair_indices = prepare_selector_cache(
        base,
        detector,
        grounded_selector,
        graph,
        text_head,
        implicit_head,
        loader(args.train_shards),
        device,
        samples=args.train_cache_samples,
        cache_name="train",
        **cache_kwargs,
    )
    train_cache = compact_selector_cache(
        selector,
        train_source,
        pair_indices,
        device,
        batch_size=args.batch_size,
    )
    if (
        expand_shards(args.train_shards) == expand_shards(args.validation_shards)
        and args.train_cache_samples == args.validation_samples
    ):
        validation_cache = train_cache
    else:
        validation_source, validation_pairs = prepare_selector_cache(
            base,
            detector,
            grounded_selector,
            graph,
            text_head,
            implicit_head,
            loader(args.validation_shards),
            device,
            samples=args.validation_samples,
            cache_name="validation",
            **cache_kwargs,
        )
        if not torch.equal(pair_indices, validation_pairs):
            raise ValueError("Train and validation pair ordering differs")
        validation_cache = compact_selector_cache(
            selector,
            validation_source,
            pair_indices,
            device,
            batch_size=args.batch_size,
        )
    pair_indices = pair_indices.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    initial = evaluate_cached(
        calibrator,
        validation_cache,
        pair_indices,
        device,
        batch_size=args.batch_size,
    )
    print("initial=" + json.dumps(initial), flush=True)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"trainable_parameters={sum(parameter.numel() for parameter in calibrator.parameters()):,}",
        flush=True,
    )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    running_loss = 0.0
    running_samples = 0
    started = time.perf_counter()
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
            batch = stack_count_records(train_cache, indices, device)
            targets = torch.stack(
                [train_cache[index]["target_role_counts"] for index in indices]
            ).to(device)
            calibrator.train()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = calibrator(
                    batch["node_logits"],
                    batch["candidate_mask"],
                    batch["edge_logits"],
                    pair_indices,
                    batch["candidate_roles"],
                )
                loss = grounded_role_count_loss(logits, targets)
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
                    f"samples={samples_seen} loss={running_loss / max(running_samples, 1):.6f} "
                    f"samples_per_second={samples_seen / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
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
                    samples_seen,
                    validation,
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
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir / "last.pt",
        calibrator,
        optimizer,
        config,
        args,
        samples_seen,
        validation,
    )
    print("validation=" + json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
