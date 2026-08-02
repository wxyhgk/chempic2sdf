#!/usr/bin/env python3
"""Train and evaluate the shadow-mode Atom Query vertex detector."""

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
    apply_atom_query_vision_updates,
    atom_query_loss,
    decode_atom_queries_by_cardinality,
    decode_atom_queries_by_count_head,
    decode_atom_queries_by_count_regression,
    decode_atom_queries_by_selection_known_counts,
    decode_atom_queries_by_duplicate_penalty_known_counts,
    decode_atom_queries_by_set_count,
    decode_atom_queries,
    load_atom_query_state,
)
from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.pretrain_vertex import (
    load_vertex_model,
    new_detection_totals,
    summarize_detection_metrics,
    update_detection_metrics,
)
from chemvector.train import expand_shards
from chemvector.vertex import VertexCollator


def query_features(
    base_model,
    pixels: torch.Tensor,
    *,
    feature_level: str,
    train_vision: bool = False,
) -> torch.Tensor:
    with torch.set_grad_enabled(train_vision):
        stroke = base_model.vision_encoder.forward_stroke_resolution(pixels)
        if feature_level == "stroke":
            return stroke
        if feature_level == "high":
            return base_model.vision_encoder.forward_high_resolution_from_stroke(stroke)
        raise ValueError(f"Unsupported Atom Query feature level: {feature_level}")


def forward_batch(
    base_model,
    detector,
    batch: dict,
    device: torch.device,
    *,
    feature_level: str,
    no_object_weight: float = 1.0,
    center_loss_weight: float = 10.0,
    coarse_center_loss_weight: float = 2.0,
    center_tail_loss_weight: float = 0.0,
    role_loss_weight: float = 0.0,
    role_matching_cost_weight: float = 0.0,
    cardinality_loss_weight: float = 0.01,
    count_loss_weight: float = 0.0,
    count_regression_loss_weight: float = 0.0,
    selection_loss_weight: float = 0.0,
    selection_ranking_loss_weight: float = 0.0,
    duplicate_loss_weight: float = 0.0,
    set_count_loss_weight: float = 0.0,
    auxiliary_loss_weight: float = 0.0,
):
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    anchors = batch["anchor_points"].to(device, non_blocking=True)
    mask = batch["anchor_mask"].to(device, non_blocking=True)
    aromatic = batch["aromatic_targets"].to(device, non_blocking=True)
    roles = batch["role_targets"].to(device, non_blocking=True)
    train_vision = any(
        parameter.requires_grad for parameter in base_model.vision_encoder.parameters()
    )
    features = query_features(
        base_model,
        pixels,
        feature_level=feature_level,
        train_vision=train_vision,
    )
    output = detector(features)
    return output, atom_query_loss(
        output,
        anchors,
        mask,
        aromatic,
        role_targets=roles,
        no_object_weight=no_object_weight,
        center_loss_weight=center_loss_weight,
        coarse_center_loss_weight=coarse_center_loss_weight,
        center_tail_loss_weight=center_tail_loss_weight,
        role_loss_weight=role_loss_weight,
        role_matching_cost_weight=role_matching_cost_weight,
        cardinality_loss_weight=cardinality_loss_weight,
        count_loss_weight=count_loss_weight,
        count_regression_loss_weight=count_regression_loss_weight,
        selection_loss_weight=selection_loss_weight,
        selection_ranking_loss_weight=selection_ranking_loss_weight,
        duplicate_loss_weight=duplicate_loss_weight,
        set_count_loss_weight=set_count_loss_weight,
        auxiliary_loss_weight=auxiliary_loss_weight,
    )


@torch.no_grad()
def evaluate(
    base_model,
    detector,
    loader,
    device: torch.device,
    *,
    threshold: float,
    tolerance_pixels: float,
    maximum_batches: int,
    feature_level: str,
    no_object_weight: float = 1.0,
    center_loss_weight: float = 10.0,
    coarse_center_loss_weight: float = 2.0,
    center_tail_loss_weight: float = 0.0,
    role_loss_weight: float = 0.0,
    role_matching_cost_weight: float = 0.0,
    cardinality_loss_weight: float = 0.01,
    count_loss_weight: float = 0.0,
    count_regression_loss_weight: float = 0.0,
    selection_loss_weight: float = 0.0,
    selection_ranking_loss_weight: float = 0.0,
    duplicate_loss_weight: float = 0.0,
    set_count_loss_weight: float = 0.0,
    auxiliary_loss_weight: float = 0.0,
    decode_mode: str = "threshold",
    selection_duplicate_penalty: float = 1.0,
) -> dict:
    base_model.eval()
    detector.eval()
    totals = new_detection_totals()
    loss_totals = {
        name: 0.0
        for name in (
            "loss",
            "presence_loss",
            "center_loss",
            "center_tail_loss",
            "coarse_center_loss",
            "aromatic_loss",
            "role_loss",
            "role_accuracy",
            "cardinality_loss",
            "count_loss",
            "count_regression_loss",
            "selection_loss",
            "selection_ranking_loss",
            "duplicate_loss",
            "set_count_loss",
            "auxiliary_loss",
        )
    }
    batches = 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output, losses = forward_batch(
                base_model,
                detector,
                batch,
                device,
                feature_level=feature_level,
                no_object_weight=no_object_weight,
                center_loss_weight=center_loss_weight,
                coarse_center_loss_weight=coarse_center_loss_weight,
                center_tail_loss_weight=center_tail_loss_weight,
                role_loss_weight=role_loss_weight,
                role_matching_cost_weight=role_matching_cost_weight,
                cardinality_loss_weight=cardinality_loss_weight,
                count_loss_weight=count_loss_weight,
                count_regression_loss_weight=count_regression_loss_weight,
                selection_loss_weight=selection_loss_weight,
                selection_ranking_loss_weight=selection_ranking_loss_weight,
                duplicate_loss_weight=duplicate_loss_weight,
                set_count_loss_weight=set_count_loss_weight,
                auxiliary_loss_weight=auxiliary_loss_weight,
            )
        if decode_mode == "threshold":
            decoded = decode_atom_queries(output, threshold=threshold)
        elif decode_mode == "cardinality":
            decoded = decode_atom_queries_by_cardinality(output)
        elif decode_mode == "count_head":
            decoded = decode_atom_queries_by_count_head(output)
        elif decode_mode == "count_regression":
            decoded = decode_atom_queries_by_count_regression(output)
        elif decode_mode == "selection_oracle":
            decoded = decode_atom_queries_by_selection_known_counts(
                output,
                batch["anchor_mask"].sum(dim=1),
            )
        elif decode_mode == "selection_dedup_oracle":
            decoded = decode_atom_queries_by_duplicate_penalty_known_counts(
                output,
                batch["anchor_mask"].sum(dim=1),
                duplicate_penalty=selection_duplicate_penalty,
            )
        elif decode_mode == "set_count":
            decoded = decode_atom_queries_by_set_count(output)
        else:
            raise ValueError(f"Unsupported Atom Query decode mode: {decode_mode}")
        update_detection_metrics(
            totals,
            decoded,
            batch["anchor_points"],
            batch["anchor_mask"],
            batch["aromatic_targets"],
            batch["pixel_values"].shape[-1],
            tolerance_pixels,
        )
        for name in loss_totals:
            loss_totals[name] += float(losses[name])
        batches += 1
        if maximum_batches > 0 and batches >= maximum_batches:
            break
    result = summarize_detection_metrics(totals)
    result.update({name: value / max(batches, 1) for name, value in loss_totals.items()})
    result["batches"] = batches
    return result


def trainable_vision_parameters(
    base_model,
    *,
    train_stem: bool,
    unfreeze_stroke_blocks: int,
) -> list[torch.nn.Parameter]:
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    stroke_stage = base_model.vision_encoder.stages[0]
    if not 0 <= unfreeze_stroke_blocks <= len(stroke_stage):
        raise ValueError(
            f"unfreeze_stroke_blocks must be in [0, {len(stroke_stage)}]"
        )
    modules = []
    if train_stem:
        modules.append(base_model.vision_encoder.stem)
    modules.extend(stroke_stage[len(stroke_stage) - unfreeze_stroke_blocks :])
    for module in modules:
        module.requires_grad_(True)
    return [
        parameter
        for parameter in base_model.vision_encoder.parameters()
        if parameter.requires_grad
    ]


def vision_updates(base_model) -> dict[str, torch.Tensor]:
    """Store the effective query vision encoder, including inherited updates."""

    return {
        name: value.detach().cpu()
        for name, value in base_model.vision_encoder.state_dict().items()
    }


def set_trainable_vision_mode(
    base_model,
    *,
    train_stem: bool,
    unfreeze_stroke_blocks: int,
) -> None:
    if train_stem:
        base_model.vision_encoder.stem.train()
    stroke_stage = base_model.vision_encoder.stages[0]
    for block in stroke_stage[len(stroke_stage) - unfreeze_stroke_blocks :]:
        block.train()


def save_checkpoint(
    path: Path,
    base_model,
    detector,
    optimizer,
    config,
    args,
    *,
    step: int,
    samples: int,
    metrics,
):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "atom_query": detector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "atom_query_config": asdict(config),
            "base_checkpoint": args.init_checkpoint,
            "base_vision_updates": vision_updates(base_model),
            "global_step": step,
            "samples_seen": samples,
            "validation": metrics,
            "training_kind": "atom_query_shadow_pretraining",
        },
        temporary,
    )
    temporary.replace(path)


def metric_key(
    metrics: dict[str, float],
    *,
    f1_first: bool = False,
) -> tuple[float, float, float, float, float]:
    if f1_first:
        return (
            metrics["f1"],
            metrics["vertex_exact_rate"],
            metrics["count_exact_rate"],
            -metrics["count_mae"],
            metrics.get("role_accuracy", 0.0),
        )
    return (
        metrics["vertex_exact_rate"],
        metrics["count_exact_rate"],
        metrics["f1"],
        -metrics["count_mae"],
        metrics.get("role_accuracy", 0.0),
    )


def checkpoint_is_eligible(metrics: dict[str, float], minimum_f1: float) -> bool:
    return metrics["f1"] >= minimum_f1


def set_count_decoder_trainable(
    detector: AtomQueryDetector,
    decoder_layers: int,
) -> list[torch.nn.Parameter]:
    if not 1 <= decoder_layers <= len(detector.decoder.layers):
        raise ValueError(
            f"count decoder layers must be in [1, {len(detector.decoder.layers)}]"
        )
    detector.requires_grad_(False)
    detector.presence.requires_grad_(True)
    for layer in detector.decoder.layers[-decoder_layers:]:
        layer.requires_grad_(True)
    if detector.decoder.norm is not None:
        detector.decoder.norm.requires_grad_(True)
    return [parameter for parameter in detector.parameters() if parameter.requires_grad]


def set_global_cardinality_trainable(
    detector: AtomQueryDetector,
    query_decoder_layers: int,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    if not 0 <= query_decoder_layers <= len(detector.decoder.layers):
        raise ValueError(
            "global cardinality query decoder layers must be in "
            f"[0, {len(detector.decoder.layers)}]"
        )
    detector.requires_grad_(False)
    detector.cardinality_token.requires_grad_(True)
    detector.cardinality_decoder.requires_grad_(True)
    detector.cardinality_norm.requires_grad_(True)
    detector.count_head.requires_grad_(True)
    detector.count_regression.requires_grad_(True)
    global_parameters = [
        parameter for parameter in detector.parameters() if parameter.requires_grad
    ]
    query_parameters = []
    if query_decoder_layers:
        detector.presence.requires_grad_(True)
        query_parameters.extend(detector.presence.parameters())
        for layer in detector.decoder.layers[-query_decoder_layers:]:
            layer.requires_grad_(True)
            query_parameters.extend(layer.parameters())
        if detector.decoder.norm is not None:
            detector.decoder.norm.requires_grad_(True)
            query_parameters.extend(detector.decoder.norm.parameters())
    return global_parameters, query_parameters


def set_selection_trainable(
    detector: AtomQueryDetector,
    *,
    context_only: bool,
) -> list[torch.nn.Parameter]:
    detector.requires_grad_(False)
    modules = [
        detector.selection_local_projection,
        detector.selection_context_projection,
        detector.selection_position_projection,
        detector.selection_context,
        detector.selection_context_score,
    ]
    if not context_only:
        modules.insert(0, detector.selection)
    for module in modules:
        module.requires_grad_(True)
    return [parameter for parameter in detector.parameters() if parameter.requires_grad]


def set_duplicate_trainable(detector: AtomQueryDetector) -> list[torch.nn.Parameter]:
    detector.requires_grad_(False)
    detector.duplicate_projection.requires_grad_(True)
    detector.duplicate_geometry.requires_grad_(True)
    detector.duplicate_similarity_scale.requires_grad_(True)
    return [parameter for parameter in detector.parameters() if parameter.requires_grad]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--query-init-checkpoint")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-reject-keys-file")
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-shuffle-buffer", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--no-object-weight", type=float, default=1.0)
    parser.add_argument("--center-loss-weight", type=float, default=10.0)
    parser.add_argument("--coarse-center-loss-weight", type=float, default=2.0)
    parser.add_argument("--center-tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--role-loss-weight", type=float, default=0.0)
    parser.add_argument("--role-matching-cost-weight", type=float, default=0.0)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.01)
    parser.add_argument("--count-loss-weight", type=float, default=0.0)
    parser.add_argument("--count-regression-loss-weight", type=float, default=0.0)
    parser.add_argument("--selection-loss-weight", type=float, default=0.0)
    parser.add_argument("--selection-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--duplicate-loss-weight", type=float, default=0.0)
    parser.add_argument("--set-count-loss-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.0)
    parser.add_argument("--count-head-only", action="store_true")
    parser.add_argument("--role-head-only", action="store_true")
    parser.add_argument("--second-refinement-only", action="store_true")
    parser.add_argument("--presence-head-only", action="store_true")
    parser.add_argument("--selection-head-only", action="store_true")
    parser.add_argument("--selection-context-only", action="store_true")
    parser.add_argument("--duplicate-head-only", action="store_true")
    parser.add_argument("--unfreeze-count-decoder-layers", type=int, default=0)
    parser.add_argument("--global-cardinality-token", action="store_true")
    parser.add_argument("--query-decoder-learning-rate", type=float, default=0.0)
    parser.add_argument("--minimum-validation-f1", type=float, default=0.0)
    parser.add_argument("--checkpoint-f1-first", action="store_true")
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--memory-stride", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--feature-level", choices=("stroke", "high"), default="stroke")
    parser.add_argument("--vision-learning-rate", type=float, default=0.0)
    parser.add_argument("--train-vision-stem", action="store_true")
    parser.add_argument("--unfreeze-stroke-blocks", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--decode-mode",
        choices=(
            "threshold",
            "cardinality",
            "count_head",
            "count_regression",
            "selection_oracle",
            "selection_dedup_oracle",
            "set_count",
        ),
        default="threshold",
    )
    parser.add_argument("--tolerance-pixels", type=float, default=6.0)
    parser.add_argument("--selection-duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Atom Query training")
    if min(args.train_samples, args.epochs, args.batch_size) < 1:
        raise ValueError("training counts must be positive")
    if (
        args.no_object_weight <= 0
        or args.center_loss_weight < 0
        or args.coarse_center_loss_weight < 0
        or args.center_tail_loss_weight < 0
        or args.role_loss_weight < 0
        or args.role_matching_cost_weight < 0
        or args.cardinality_loss_weight < 0
        or args.count_loss_weight < 0
        or args.count_regression_loss_weight < 0
        or args.selection_loss_weight < 0
        or args.selection_ranking_loss_weight < 0
        or args.duplicate_loss_weight < 0
        or args.set_count_loss_weight < 0
        or args.auxiliary_loss_weight < 0
        or args.learning_rate <= 0
        or args.query_decoder_learning_rate < 0
        or args.vision_learning_rate < 0
        or args.selection_duplicate_penalty < 0
        or not 0.0 <= args.minimum_validation_f1 <= 1.0
    ):
        raise ValueError("Atom Query learning rates or loss weights are invalid")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    base_model = load_vertex_model(Path(args.init_checkpoint), device).eval()
    config = AtomQueryConfig(
        input_channels=base_model.config.vision_dims[
            0 if args.feature_level == "stroke" else 1
        ],
        d_model=args.d_model,
        queries=args.queries,
        layers=args.layers,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        memory_stride=args.memory_stride,
    )
    detector = AtomQueryDetector(config).to(device)
    if args.query_init_checkpoint:
        query_payload = torch.load(
            args.query_init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        saved_config = AtomQueryConfig(**query_payload["atom_query_config"])
        if saved_config != config:
            raise ValueError(
                "Atom Query init checkpoint configuration does not match current arguments"
            )
        load_atom_query_state(detector, query_payload["atom_query"])
        apply_atom_query_vision_updates(base_model, query_payload)
    head_only_modes = sum(
        (
            args.count_head_only,
            args.role_head_only,
            args.second_refinement_only,
            args.presence_head_only,
            args.selection_head_only,
            args.selection_context_only,
            args.duplicate_head_only,
            bool(args.unfreeze_count_decoder_layers)
            and not args.global_cardinality_token,
            args.global_cardinality_token,
        )
    )
    if head_only_modes > 1:
        raise ValueError("Atom Query head-only modes are mutually exclusive")
    if args.count_head_only:
        detector.requires_grad_(False)
        detector.count_head.requires_grad_(True)
    if args.role_head_only:
        if args.role_loss_weight <= 0:
            raise ValueError("Role-head-only training requires a positive role loss weight")
        detector.requires_grad_(False)
        detector.role.requires_grad_(True)
    if args.second_refinement_only:
        detector.requires_grad_(False)
        detector.second_local_refinement.requires_grad_(True)
    if args.presence_head_only:
        if args.set_count_loss_weight <= 0:
            raise ValueError(
                "Presence-head-only training requires a positive set count loss weight"
            )
        detector.requires_grad_(False)
        detector.presence.requires_grad_(True)
    if args.selection_head_only or args.selection_context_only:
        if (
            args.selection_loss_weight <= 0
            and args.selection_ranking_loss_weight <= 0
        ):
            raise ValueError(
                "Selection training requires a positive selection loss"
            )
        set_selection_trainable(
            detector,
            context_only=args.selection_context_only,
        )
    if args.duplicate_head_only:
        if args.duplicate_loss_weight <= 0:
            raise ValueError(
                "Duplicate-head-only training requires a positive duplicate loss"
            )
        set_duplicate_trainable(detector)
    query_decoder_trainable = []
    if args.global_cardinality_token:
        if (
            args.count_loss_weight <= 0
            and args.count_regression_loss_weight <= 0
        ):
            raise ValueError(
                "Global cardinality token training requires a positive count loss"
            )
        if args.unfreeze_count_decoder_layers and args.set_count_loss_weight <= 0:
            raise ValueError(
                "Joint global cardinality training requires a positive set count loss weight"
            )
        detector_trainable, query_decoder_trainable = (
            set_global_cardinality_trainable(
                detector,
                args.unfreeze_count_decoder_layers,
            )
        )
        detector_trainable = [
            parameter
            for parameter in detector_trainable
            if all(parameter is not query for query in query_decoder_trainable)
        ]
    elif args.unfreeze_count_decoder_layers:
        if args.set_count_loss_weight <= 0:
            raise ValueError(
                "Count-aware decoder training requires a positive set count loss weight"
            )
        detector_trainable = set_count_decoder_trainable(
            detector,
            args.unfreeze_count_decoder_layers,
        )
    else:
        detector_trainable = [
            parameter for parameter in detector.parameters() if parameter.requires_grad
        ]
    vision_trainable = trainable_vision_parameters(
        base_model,
        train_stem=args.train_vision_stem,
        unfreeze_stroke_blocks=args.unfreeze_stroke_blocks,
    )
    if bool(vision_trainable) != bool(args.vision_learning_rate):
        raise ValueError(
            "A positive vision learning rate and trainable vision modules are required together"
        )
    parameter_groups = [{"params": detector_trainable, "lr": args.learning_rate}]
    if query_decoder_trainable:
        if args.query_decoder_learning_rate <= 0:
            raise ValueError(
                "Joint global cardinality training requires a positive "
                "query decoder learning rate"
            )
        parameter_groups.append(
            {
                "params": query_decoder_trainable,
                "lr": args.query_decoder_learning_rate,
            }
        )
    if vision_trainable:
        parameter_groups.append(
            {"params": vision_trainable, "lr": args.vision_learning_rate}
        )
    trainable = detector_trainable + query_decoder_trainable + vision_trainable
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    train_reject_keys = load_reject_keys(args.train_reject_keys_file)
    validation_reject_keys = load_reject_keys(args.validation_reject_keys_file)
    print(
        f"train_reject_keys={len(train_reject_keys)} "
        f"validation_reject_keys={len(validation_reject_keys)}"
    )
    train_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.train_shards),
            sample_shuffle_buffer=args.sample_shuffle_buffer,
            sequence_version="0.4",
            reject_keys=train_reject_keys,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    validation_loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.validation_shards),
            sequence_version="0.4",
            reject_keys=validation_reject_keys,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=VertexCollator(),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"query_trainable_parameters={sum(p.numel() for p in detector_trainable):,}")
    print(f"vision_trainable_parameters={sum(p.numel() for p in vision_trainable):,}")
    print(f"trainable_parameters={sum(p.numel() for p in trainable):,}")
    samples = 0
    step = 0
    best_score = (-1.0, -1.0, -1.0, float("-inf"), -1.0)
    started = time.perf_counter()
    running = {
        name: 0.0
        for name in (
            "loss",
            "presence_loss",
            "center_loss",
            "center_tail_loss",
            "coarse_center_loss",
            "aromatic_loss",
            "role_loss",
            "role_accuracy",
            "cardinality_loss",
            "count_loss",
            "count_regression_loss",
            "selection_loss",
            "selection_ranking_loss",
            "duplicate_loss",
            "set_count_loss",
            "auxiliary_loss",
        )
    }
    interval_batches = 0
    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            detector.train()
            set_trainable_vision_mode(
                base_model,
                train_stem=args.train_vision_stem,
                unfreeze_stroke_blocks=args.unfreeze_stroke_blocks,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, losses = forward_batch(
                    base_model,
                    detector,
                    batch,
                    device,
                    feature_level=args.feature_level,
                    no_object_weight=args.no_object_weight,
                    center_loss_weight=args.center_loss_weight,
                    coarse_center_loss_weight=args.coarse_center_loss_weight,
                    center_tail_loss_weight=args.center_tail_loss_weight,
                    role_loss_weight=args.role_loss_weight,
                    role_matching_cost_weight=args.role_matching_cost_weight,
                    cardinality_loss_weight=args.cardinality_loss_weight,
                    count_loss_weight=args.count_loss_weight,
                    count_regression_loss_weight=args.count_regression_loss_weight,
                    selection_loss_weight=args.selection_loss_weight,
                    selection_ranking_loss_weight=args.selection_ranking_loss_weight,
                    duplicate_loss_weight=args.duplicate_loss_weight,
                    set_count_loss_weight=args.set_count_loss_weight,
                    auxiliary_loss_weight=args.auxiliary_loss_weight,
                )
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(f"Non-finite Atom Query loss at step {step}")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += int(batch["pixel_values"].size(0))
            interval_batches += 1
            for name in running:
                running[name] += float(losses[name].detach())
            if step % args.log_every == 0:
                print(
                    f"step={step} samples={samples} "
                    + " ".join(
                        f"{name}={running[name] / interval_batches:.6f}"
                        for name in running
                    )
                    + f" samples_per_second={samples / (time.perf_counter() - started):.2f}",
                    flush=True,
                )
                running = {name: 0.0 for name in running}
                interval_batches = 0
            if args.validate_every > 0 and step % args.validate_every == 0:
                metrics = evaluate(
                    base_model,
                    detector,
                    validation_loader,
                    device,
                    threshold=args.threshold,
                    tolerance_pixels=args.tolerance_pixels,
                    maximum_batches=args.validation_batches,
                    feature_level=args.feature_level,
                    no_object_weight=args.no_object_weight,
                    center_loss_weight=args.center_loss_weight,
                    coarse_center_loss_weight=args.coarse_center_loss_weight,
                    center_tail_loss_weight=args.center_tail_loss_weight,
                    role_loss_weight=args.role_loss_weight,
                    role_matching_cost_weight=args.role_matching_cost_weight,
                    cardinality_loss_weight=args.cardinality_loss_weight,
                    count_loss_weight=args.count_loss_weight,
                    count_regression_loss_weight=args.count_regression_loss_weight,
                    selection_loss_weight=args.selection_loss_weight,
                    selection_ranking_loss_weight=args.selection_ranking_loss_weight,
                    duplicate_loss_weight=args.duplicate_loss_weight,
                    set_count_loss_weight=args.set_count_loss_weight,
                    auxiliary_loss_weight=args.auxiliary_loss_weight,
                    decode_mode=args.decode_mode,
                    selection_duplicate_penalty=args.selection_duplicate_penalty,
                )
                print("validation=" + json.dumps(metrics, ensure_ascii=False), flush=True)
                save_checkpoint(
                    output_dir / f"checkpoint-{step:07d}.pt",
                    base_model,
                    detector,
                    optimizer,
                    config,
                    args,
                    step=step,
                    samples=samples,
                    metrics=metrics,
                )
                current_score = metric_key(
                    metrics,
                    f1_first=args.checkpoint_f1_first,
                )
                if (
                    checkpoint_is_eligible(metrics, args.minimum_validation_f1)
                    and current_score > best_score
                ):
                    best_score = current_score
                    save_checkpoint(
                        output_dir / "best.pt",
                        base_model,
                        detector,
                        optimizer,
                        config,
                        args,
                        step=step,
                        samples=samples,
                        metrics=metrics,
                    )
            if samples >= args.train_samples:
                break
        if samples >= args.train_samples:
            break
    metrics = evaluate(
        base_model,
        detector,
        validation_loader,
        device,
        threshold=args.threshold,
        tolerance_pixels=args.tolerance_pixels,
        maximum_batches=args.validation_batches,
        feature_level=args.feature_level,
        no_object_weight=args.no_object_weight,
        center_loss_weight=args.center_loss_weight,
        coarse_center_loss_weight=args.coarse_center_loss_weight,
        center_tail_loss_weight=args.center_tail_loss_weight,
        role_loss_weight=args.role_loss_weight,
        role_matching_cost_weight=args.role_matching_cost_weight,
        cardinality_loss_weight=args.cardinality_loss_weight,
        count_loss_weight=args.count_loss_weight,
        count_regression_loss_weight=args.count_regression_loss_weight,
        selection_loss_weight=args.selection_loss_weight,
        selection_ranking_loss_weight=args.selection_ranking_loss_weight,
        duplicate_loss_weight=args.duplicate_loss_weight,
        set_count_loss_weight=args.set_count_loss_weight,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        decode_mode=args.decode_mode,
        selection_duplicate_penalty=args.selection_duplicate_penalty,
    )
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "last.pt",
        base_model,
        detector,
        optimizer,
        config,
        args,
        step=step,
        samples=samples,
        metrics=metrics,
    )
    print("validation=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
