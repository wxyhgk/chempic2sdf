#!/usr/bin/env python3
"""Capacity-test a set selector over the fixed P1-W multiscale peaks."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.g3_boundary_band_loss import (
    G3_BOUNDARY_BAND_SIZE,
    boundary_band_ranking_loss,
)
from chemvector.components.atoms.g3_multiscale_peak_set_ranker import (
    G3_MULTISCALE_CANDIDATES,
    G3MultiscalePeakSetRanker,
    build_dynamic_multiscale_targets,
    select_known_count_multiscale,
)
from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
    decode_atom_center_field,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1w_multiscale_peak_rank as p1w
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.atoms import train_g3_p1v_continuous_stride1_capacity as p1v
from chemvector.training.graph.scenes.g3_structural_data import (
    OVERFIT32_MANIFEST,
    load_g3_overfit32_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p1x-multiscale-peak-selector-capacity64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
STEPS = 4_800
SAVE_EVERY = 800
EXPECTED_EXPOSURE = 19_200
STRIDE2_K = 96
STRIDE1_K = 32
SOURCE_ID_SCALE_OFFSET = 1_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-X requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P1-X requires output directory {OUTPUT}")
    if args.log_every < 1:
        raise ValueError("P1-X log interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-X requires CUDA BF16")


def _scale_score_features(scores: torch.Tensor) -> torch.Tensor:
    logits = torch.logit(scores.float().clamp(1e-6, 1.0 - 1e-6))
    return (logits - logits.mean()) / logits.std(unbiased=False).clamp_min(1e-4)


@torch.no_grad()
def _collect_scale(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_model_type: str,
    field_factory,
    feature_name: str,
    maximum_objects: int,
    domains: dict[str, list],
    collator,
    device: torch.device,
    batch_size: int = 4,
    accepted_split: str = "train",
) -> tuple[dict[str, dict[str, dict]], dict]:
    if batch_size < 1:
        raise ValueError("P1-X collection batch size must be positive")
    if not accepted_split:
        raise ValueError("P1-X accepted split must be non-empty")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError(f"P1-X checkpoint SHA differs: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != expected_model_type:
        raise ValueError(f"P1-X checkpoint type differs: {checkpoint}")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.load_state_dict(payload["vision_last_stride4_block"], strict=True)
    field = field_factory(atom_head).to(device)
    field.load_state_dict(payload["atom_center_field"], strict=True)
    field.requires_grad_(False).eval()
    records: dict[str, dict[str, dict]] = {}
    sample_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    started = time.perf_counter()
    for domain, manifest in domains.items():
        current = {}
        for batch in p1a.iter_manifest_batches(
            manifest,
            collator,
            batch_size=batch_size,
            accepted_split=accepted_split,
        ):
            objects, stroke = p1t.model_inputs(
                atom_head,
                vision,
                frozen_block,
                batch["pixel_values"],
                device=device,
                training=False,
            )
            with _autocast_context(device, "bf16"):
                output = field(objects, stroke)
            peaks = extract_regional_maximum_objects(
                decode_atom_center_field(output), maximum_objects=maximum_objects
            )
            feature_map = output[feature_name]
            for index, sample_key_value in enumerate(batch["keys"]):
                sample_key = str(sample_key_value)
                peak = peaks[index]
                if len(peak["anchors"]) != maximum_objects:
                    raise RuntimeError(
                        f"P1-X {domain}/{sample_key} has fewer than {maximum_objects} peaks"
                    )
                source_ids = peak["source_ids"].to(device)
                flat_features = feature_map[index].permute(1, 2, 0).reshape(
                    -1, feature_map.size(1)
                )
                selected_features = flat_features[source_ids].float().cpu()
                record = {
                    "source_ids": peak["source_ids"].long().clone(),
                    "candidate_features": selected_features,
                    "candidate_anchors": peak["anchors"].float().clone(),
                    "raw_scores": peak["scores"].float().clone(),
                    "normalized_scores": _scale_score_features(peak["scores"]),
                    "atom_anchors": batch["atom_anchors"][index].float().clone(),
                    "atom_mask": batch["atom_mask"][index].bool().clone(),
                    "regional_maxima_count": int(peak["regional_maxima_count"]),
                }
                current[sample_key] = record
                sample_digest.update(f"{domain}:{sample_key}\n".encode())
                for name in (
                    "source_ids",
                    "candidate_features",
                    "candidate_anchors",
                    "raw_scores",
                    "normalized_scores",
                ):
                    candidate_digest.update(record[name].numpy().tobytes())
        if len(current) != len(manifest):
            raise RuntimeError(f"P1-X collected incomplete {domain} cache")
        records[domain] = current
    lineage = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "model_type": expected_model_type,
        "vision_checkpoint_sha256": vision_sha,
        "atom_state_sha256": module_state_sha256(atom_head),
        "vision_state_sha256": module_state_sha256(vision),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "candidate_tensors_sha256": candidate_digest.hexdigest(),
        "batch_size": batch_size,
        "accepted_split": accepted_split,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del field, block, frozen_block, vision, atom_head, payload
    gc.collect()
    torch.cuda.empty_cache()
    return records, lineage


def merge_scale_records(
    stride2: dict[str, dict[str, dict]],
    stride1: dict[str, dict[str, dict]],
) -> tuple[dict[str, list[dict]], dict]:
    if set(stride2) != set(stride1):
        raise ValueError("P1-X scale domains differ")
    merged: dict[str, list[dict]] = {}
    digest = hashlib.sha256()
    reachable = {}
    for domain in sorted(stride2):
        if set(stride2[domain]) != set(stride1[domain]):
            raise ValueError(f"P1-X {domain} scale sample identities differ")
        current = []
        matched_atoms = target_atoms = exact_images = 0
        for sample_key in sorted(stride2[domain]):
            left = stride2[domain][sample_key]
            right = stride1[domain][sample_key]
            if not torch.equal(left["atom_anchors"], right["atom_anchors"]) or not torch.equal(
                left["atom_mask"], right["atom_mask"]
            ):
                raise ValueError(f"P1-X {domain}/{sample_key} targets differ by scale")
            record = {
                "sample_key": sample_key,
                "candidate_ids": torch.cat(
                    (
                        left["source_ids"],
                        right["source_ids"] + SOURCE_ID_SCALE_OFFSET,
                    )
                ),
                "candidate_features": torch.cat(
                    (left["candidate_features"], right["candidate_features"])
                ),
                "candidate_anchors": torch.cat(
                    (left["candidate_anchors"], right["candidate_anchors"])
                ),
                "raw_scores": torch.cat((left["raw_scores"], right["raw_scores"])),
                "normalized_scores": torch.cat(
                    (left["normalized_scores"], right["normalized_scores"])
                ),
                "scale_ids": torch.cat(
                    (
                        torch.zeros(STRIDE2_K, dtype=torch.long),
                        torch.ones(STRIDE1_K, dtype=torch.long),
                    )
                ),
                "scale_ranks": torch.cat(
                    (
                        torch.arange(STRIDE2_K).float() / (STRIDE2_K - 1),
                        torch.arange(STRIDE1_K).float() / (STRIDE1_K - 1),
                    )
                ),
                "atom_anchors": left["atom_anchors"],
                "atom_mask": left["atom_mask"],
            }
            if len(record["candidate_ids"]) != G3_MULTISCALE_CANDIDATES:
                raise RuntimeError("P1-X merged candidate count differs from 128")
            expected = record["atom_anchors"][record["atom_mask"]]
            matched = p1a.maximum_cardinality_radius_match_count(
                record["candidate_anchors"], expected,
                radius_pixels=6, image_size=768,
            )
            matched_atoms += matched
            target_atoms += len(expected)
            exact_images += int(matched == len(expected))
            for name in (
                "candidate_ids",
                "candidate_features",
                "candidate_anchors",
                "raw_scores",
                "normalized_scores",
                "scale_ids",
                "scale_ranks",
            ):
                digest.update(record[name].numpy().tobytes())
            current.append(record)
        merged[domain] = current
        reachable[domain] = {
            "matched_atoms": matched_atoms,
            "target_atoms": target_atoms,
            "exact_images": exact_images,
        }
    return merged, {
        "candidate_protocol": "stride2_top96_plus_stride1_top32",
        "candidate_count": G3_MULTISCALE_CANDIDATES,
        "candidate_cache_sha256": digest.hexdigest(),
        "reachability": reachable,
    }


def collate_cached(records: list[dict], device: torch.device) -> dict:
    tensor_names = (
        "candidate_ids",
        "candidate_features",
        "candidate_anchors",
        "raw_scores",
        "normalized_scores",
        "scale_ids",
        "scale_ranks",
        "atom_anchors",
        "atom_mask",
    )
    output = {
        name: torch.stack([record[name] for record in records]).to(device)
        for name in tensor_names
    }
    output["sample_keys"] = [record["sample_key"] for record in records]
    return output


def cached_epoch_batches(cache: dict[str, list[dict]]):
    overfit = cache["overfit32"]
    adjacent = cache["adjacent32"]
    if len(overfit) != 32 or len(adjacent) != 32:
        raise ValueError("P1-X capacity cache must contain two 32-image domains")
    for start in range(0, 32, 4):
        yield "overfit32", overfit[start : start + 4]
        yield "adjacent32", adjacent[start : start + 4]


def validate_optimizer(ranker, optimizer) -> None:
    expected = {id(value) for value in ranker.parameters()}
    actual = {
        id(value) for group in optimizer.param_groups for value in group["params"]
    }
    if expected != actual:
        raise ValueError("P1-X optimizer must contain only selector parameters")


def training_step(ranker, optimizer, batch, *, max_grad_norm: float) -> dict:
    validate_optimizer(ranker, optimizer)
    optimizer.zero_grad(set_to_none=True)
    candidates = {
        name: batch[name]
        for name in (
            "candidate_ids",
            "candidate_features",
            "candidate_anchors",
            "raw_scores",
            "normalized_scores",
            "scale_ids",
            "scale_ranks",
        )
    }
    with _autocast_context(batch["candidate_ids"].device, "bf16"):
        ranked = ranker(candidates)
    dynamic = build_dynamic_multiscale_targets(
        ranked, batch["atom_anchors"], batch["atom_mask"]
    )
    losses = boundary_band_ranking_loss(
        ranked, dynamic, boundary_size=G3_BOUNDARY_BAND_SIZE
    )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-X selector gradient norm is not finite")
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "alternative_candidates": float(dynamic["alternative_candidates"].sum()),
    }


@torch.no_grad()
def evaluate(ranker, records: list[dict], *, device: torch.device) -> dict:
    ranker.eval()
    totals = p1a._empty_atom_totals()
    selection_digest = hashlib.sha256()
    for start in range(0, len(records), 4):
        batch = collate_cached(records[start : start + 4], device)
        candidates = {
            name: batch[name]
            for name in (
                "candidate_ids",
                "candidate_features",
                "candidate_anchors",
                "raw_scores",
                "normalized_scores",
                "scale_ids",
                "scale_ranks",
            )
        }
        with _autocast_context(device, "bf16"):
            ranked = ranker(candidates)
        counts = batch["atom_mask"].sum(dim=1).long()
        selected = select_known_count_multiscale(ranked, counts)["selected_mask"]
        for index in range(len(records[start : start + 4])):
            predicted = batch["candidate_anchors"][index, selected[index]].float().cpu()
            expected = batch["atom_anchors"][index, batch["atom_mask"][index]].float().cpu()
            p1a._merge_atom_totals(
                totals, p1a.atom_sample_statistics(predicted, expected)
            )
            selection_digest.update(
                batch["candidate_ids"][index, selected[index]].cpu().numpy().tobytes()
            )
    metrics = p1a._finalize_atom_totals(totals)
    metrics["exact_rate_6px"] = metrics["whole_set_exact_images"] / len(records)
    metrics["selection_ids_sha256"] = selection_digest.hexdigest()
    return metrics


def capacity_gate(metrics: dict[str, dict]) -> tuple[str, dict[str, bool]]:
    checks = {
        "overfit_all_atoms": metrics["overfit32"]["matched_atoms"]
        == metrics["overfit32"]["target_atoms"],
        "overfit_32_of_32": metrics["overfit32"]["whole_set_exact_images"] == 32,
        "adjacent_all_atoms": metrics["adjacent32"]["matched_atoms"]
        == metrics["adjacent32"]["target_atoms"],
        "adjacent_32_of_32": metrics["adjacent32"]["whole_set_exact_images"] == 32,
    }
    return (
        "dual_domain_selector_capacity_go"
        if all(checks.values())
        else "dual_domain_selector_capacity_no_go",
        checks,
    )


def _checkpoint_payload(ranker, optimizer, args, lineage, *, step, exposure, losses, metrics):
    training_args = {**vars(args), "output_dir": str(args.output_dir)}
    return {
        "checkpoint_schema_version": "g3-p1x-multiscale-selector-capacity-0.1",
        "model_type": MODEL_TYPE,
        "multiscale_peak_set_ranker": ranker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_args": training_args,
        "trainable_parameter_count": ranker.parameter_count(),
        "candidate_protocol": {
            "stride2_k": STRIDE2_K,
            "stride1_k": STRIDE1_K,
            "total": G3_MULTISCALE_CANDIDATES,
            "target_dependent_merging": False,
            "pixel_radius_clustering": False,
        },
        "matching": "maximum-cardinality-score-ranked-one-to-one",
        "known_count_is_model_input": False,
        "last_losses": losses,
        "lineage": lineage,
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P1-X output directory is not empty: {args.output_dir}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    overfit_manifest, overfit_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    adjacent_manifest, adjacent_manifest_audit, adjacent_audit = p1u.load_adjacent_manifest()
    domains = {"overfit32": overfit_manifest, "adjacent32": adjacent_manifest}
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        args.output_dir / "training_args.json",
        {**vars(args), "output_dir": str(args.output_dir)},
    )

    stride2, stride2_lineage = _collect_scale(
        checkpoint=p1w.P1U_CHECKPOINT,
        checkpoint_sha256=p1w.P1U_CHECKPOINT_SHA256,
        expected_model_type=p1u.MODEL_TYPE,
        field_factory=ContinuousStride2AtomCenterField,
        feature_name="high_resolution_features",
        maximum_objects=STRIDE2_K,
        domains=domains,
        collator=collator,
        device=device,
    )
    stride1, stride1_lineage = _collect_scale(
        checkpoint=p1w.P1V_CHECKPOINT,
        checkpoint_sha256=p1w.P1V_CHECKPOINT_SHA256,
        expected_model_type=p1v.MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=STRIDE1_K,
        domains=domains,
        collator=collator,
        device=device,
    )
    cache, cache_audit = merge_scale_records(stride2, stride1)
    if not all(
        value["exact_images"] == 32
        and value["matched_atoms"] == value["target_atoms"]
        for value in cache_audit["reachability"].values()
    ):
        raise RuntimeError("P1-X cache no longer reproduces P1-W reachability")
    del stride2, stride1
    gc.collect()

    ranker = G3MultiscalePeakSetRanker().to(device)
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(ranker, optimizer)
    initial_ranker_sha = module_state_sha256(ranker)
    step0 = {
        domain: evaluate(ranker, records, device=device)
        for domain, records in cache.items()
    }
    lineage = {
        "p1w_audit": str((p1w.DEFAULT_OUTPUT / "audit.json").resolve()),
        "p1w_audit_sha256": file_sha256(p1w.DEFAULT_OUTPUT / "audit.json"),
        "p1u": stride2_lineage,
        "p1v": stride1_lineage,
        "overfit32_manifest": overfit_audit,
        "adjacent32_manifest": adjacent_manifest_audit,
        "adjacent_audit_sha256": p1u.ADJACENT_AUDIT_SHA256,
        "adjacent_capacity_audit": adjacent_audit,
        "cache": cache_audit,
        "initial_ranker_state_sha256": initial_ranker_sha,
        "step0_metrics": step0,
    }
    audit = {
        "schema_version": "g3-p1x-training-audit-0.1",
        "scope": "dual-domain-known-count-selector-capacity64",
        "candidate_protocol": cache_audit,
        "training_targets": "dynamic-maximum-cardinality-one-to-one",
        "fixed_candidate_identity_supervision": False,
        "target_dependent_candidate_merging": False,
        "pixel_radius_clustering": False,
        "known_count_is_model_input": False,
        "forbidden_components_loaded": {
            "bond": False,
            "ocr": False,
            "chemscene": False,
            "sdf": False,
        },
        "trainable_parameter_count": ranker.parameter_count(),
        "lineage": lineage,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    _atomic_json(args.output_dir / "step0_metrics.json", step0)

    epoch_order = [
        f"{domain}:{record['sample_key']}"
        for domain, records in cached_epoch_batches(cache)
        for record in records
    ]
    epoch_order_sha = hashlib.sha256(
        "".join(f"{value}\n" for value in epoch_order).encode()
    ).hexdigest()
    step = exposure = epoch = 0
    losses = {}
    running = {}
    started = time.perf_counter()
    log_path = args.output_dir / "train.jsonl"
    while step < STEPS:
        epoch += 1
        current_order = []
        for domain, records in cached_epoch_batches(cache):
            current_order.extend(f"{domain}:{record['sample_key']}" for record in records)
            ranker.train()
            batch = collate_cached(records, device)
            losses = training_step(
                ranker, optimizer, batch, max_grad_norm=args.max_grad_norm
            )
            step += 1
            exposure += len(records)
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "exposure": exposure,
                    "domain": domain,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{
                        name: value / args.log_every
                        for name, value in running.items()
                    },
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = _checkpoint_payload(
                    ranker, optimizer, args, lineage,
                    step=step, exposure=exposure, losses=losses, metrics=None,
                )
                _atomic_torch_save(args.output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(args.output_dir / "last.pt", payload)
        observed_sha = hashlib.sha256(
            "".join(f"{value}\n" for value in current_order).encode()
        ).hexdigest()
        if observed_sha != epoch_order_sha:
            raise RuntimeError("P1-X cached epoch order changed")

    metrics = {
        domain: evaluate(ranker, records, device=device)
        for domain, records in cache.items()
    }
    repeated = {
        domain: evaluate(ranker, records, device=device)
        for domain, records in cache.items()
    }
    decision, checks = capacity_gate(metrics)
    protocol = {
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == 300,
        "fixed_epoch_order": observed_sha == epoch_order_sha,
        "ranker_changed": module_state_sha256(ranker) != initial_ranker_sha,
        "optimizer_ranker_only": validate_optimizer(ranker, optimizer) is None,
        "p1w_dual_domain_reachability": all(
            value["exact_images"] == 32
            for value in cache_audit["reachability"].values()
        ),
        "repeat_metrics": repeated == metrics,
    }
    if not all(protocol.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol,
        "metrics": metrics,
        "step0_metrics": step0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = _checkpoint_payload(
        ranker, optimizer, args, lineage,
        step=step, exposure=exposure, losses=losses, metrics=final,
    )
    checkpoint = args.output_dir / "step-004800.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(args.output_dir / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(args.output_dir / "final_metrics.json", final)
    _atomic_json(
        args.output_dir / "status.json",
        {
            "status": "completed",
            "decision": decision,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": final["checkpoint_sha256"],
        },
    )
    print(json.dumps({"decision": decision, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
