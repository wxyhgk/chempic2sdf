#!/usr/bin/env python3
"""Train the P1-D boundary-band loss on the frozen P1-A top-128 set."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time
import textwrap

import numpy as np
import torch

from chemvector.components.atoms.g3_boundary_band_loss import (
    G3_BOUNDARY_BAND_SIZE,
    boundary_band_ranking_loss,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    G3_TOP128_SET_RANKER_PARAMETERS,
    G3Top128SetRanker,
    build_hybrid_top128_targets,
    extract_frozen_top128,
    select_ranked_top128,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1b_frozen_presence_ranker as p1b
from chemvector.training.atoms import train_g3_p1c_top128_set_ranker as p1c
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


P1A_CHECKPOINT = p1c.P1A_CHECKPOINT
P1A_CHECKPOINT_SHA256 = p1c.P1A_CHECKPOINT_SHA256
P1A_ATOM_STATE_SHA256 = p1c.P1A_ATOM_STATE_SHA256
EXPECTED_ROUTE_SEQUENCE_SHA256 = p1c.EXPECTED_ROUTE_SEQUENCE_SHA256
DEFAULT_OUTPUT = Path(
    "outputs/checkpoints/chemvector-g3-p1d-boundary-band-set-ranker-train5k"
)
MODEL_TYPE = "chemvector-g3-p1d-boundary-band-set-ranker-train5k"
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
P1C_HELPER_SHA256 = (
    "845eabdd90bd386f77ca40744a27aa3b2f08f39978786a4d0f4f18efbeda478f"
)


def load_p1a_boundary_ranker(device: torch.device):
    atom_head, ranker, payload, config = p1c.load_p1a_set_ranker(device)
    if int(torch.count_nonzero(ranker.residual_head.weight)) or int(
        torch.count_nonzero(ranker.residual_head.bias)
    ):
        raise ValueError("P1-D requires a zero-initialized set ranker")
    return atom_head, ranker, payload, config


def validate_optimizer(atom_head, ranker, optimizer) -> None:
    expected = list(ranker.parameters())
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(expected) != len(actual) or {id(value) for value in expected} != {
        id(value) for value in actual
    }:
        raise ValueError("P1-D optimizer must contain only set-ranker parameters")
    if any(parameter.requires_grad for parameter in atom_head.parameters()):
        raise ValueError("P1-D frozen Atom head became trainable")


def target_free_ranker_contract() -> dict[str, object]:
    source = textwrap.dedent(inspect.getsource(G3Top128SetRanker.forward))
    tree = ast.parse(source)
    arguments = [argument.arg for argument in tree.body[0].args.args]
    forbidden = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id in {"target", "targets", "count", "counts", "atom_counts"}
        }
    )
    return {
        "arguments": arguments,
        "forbidden_identifiers": forbidden,
        "passed": arguments == ["self", "candidates"] and not forbidden,
    }


def p1d_training_step(
    atom_head,
    ranker: G3Top128SetRanker,
    vision,
    optimizer,
    batch: dict,
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
) -> tuple[dict[str, float], dict]:
    validate_optimizer(atom_head, ranker, optimizer)
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("P1-D vision must remain frozen")
    optimizer.zero_grad(set_to_none=True)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = {
        "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
        "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
    }
    with torch.no_grad(), _autocast_context(device, precision):
        features = vision.forward_stroke_resolution(pixels)
        dense = atom_head(features.detach())
        candidates = extract_frozen_top128(dense)
    with _autocast_context(device, precision):
        ranked = ranker(candidates)
    hybrid = build_hybrid_top128_targets(candidates, targets)
    losses = boundary_band_ranking_loss(
        ranked, hybrid, boundary_size=G3_BOUNDARY_BAND_SIZE
    )
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), max_grad_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-D set-ranker gradient norm is not finite")
    optimizer.step()
    if any(parameter.grad is not None for parameter in atom_head.parameters()):
        raise RuntimeError("P1-D produced gradients for the frozen Atom head")
    if any(parameter.grad is not None for parameter in vision.parameters()):
        raise RuntimeError("P1-D produced gradients for frozen vision")
    metrics = {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "batch_images": float(len(batch["keys"])),
    }
    return metrics, {"candidates": candidates, "hybrid": hybrid}


def hybrid_topm_statistics(ranked: dict, hybrid: dict) -> dict[str, int]:
    positive = hybrid["positive_mask"].bool()
    counts = positive.sum(dim=1).long()
    selected = select_ranked_top128(ranked, counts)
    matched = whole = boundary_exact = 0
    total = int(positive.sum())
    for index in range(len(positive)):
        selected_ids = set(
            selected["atom_flat_ids"][index, selected["atom_mask"][index]].tolist()
        )
        positive_ids = set(
            ranked["candidate_ids"][index, positive[index]].tolist()
        )
        matched += len(selected_ids & positive_ids)
        whole += int(selected_ids == positive_ids)
        scores = ranked["ranking_logits"][index].float()
        boundary_exact += int(bool(scores[positive[index]].min() > scores[~positive[index]].max()))
    return {
        "images": len(positive),
        "positive_count": total,
        "matched_positive_count": matched,
        "whole_set_exact_images": whole,
        "boundary_exact_images": boundary_exact,
    }


@torch.no_grad()
def evaluate_train_fixed_final(
    atom_head,
    ranker,
    vision,
    manifest,
    collator,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> dict:
    atom_head.eval()
    ranker.eval()
    vision.eval()
    totals = {
        "images": 0,
        "positive_count": 0,
        "matched_positive_count": 0,
        "whole_set_exact_images": 0,
        "boundary_exact_images": 0,
    }
    sample_digest = hashlib.sha256()
    set_audit = p1c.FrozenSetAuditAccumulator()
    target_free_forward_bytes = True
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=batch_size):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
            "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        }
        with _autocast_context(device, precision):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            ranked = ranker(candidates)
            repeated = ranker(candidates)
        target_free_forward_bytes &= torch.equal(
            ranked["ranking_logits"], repeated["ranking_logits"]
        )
        hybrid = build_hybrid_top128_targets(candidates, targets)
        set_audit.add(batch["keys"], candidates, hybrid)
        current = hybrid_topm_statistics(ranked, hybrid)
        for name in totals:
            totals[name] += current[name]
        del pixels, features, dense, candidates, ranked, hybrid
    positive_total = totals["positive_count"]
    return {
        "scope": "train5k-fixed-final-boundary-diagnostic",
        **totals,
        "hybrid_topm_recall": (
            totals["matched_positive_count"] / positive_total
            if positive_total
            else 0.0
        ),
        "hybrid_topm_whole_set_rate": (
            totals["whole_set_exact_images"] / totals["images"]
            if totals["images"]
            else 0.0
        ),
        "boundary_exact_rate": (
            totals["boundary_exact_images"] / totals["images"]
            if totals["images"]
            else 0.0
        ),
        "sample_keys_sha256": sample_digest.hexdigest(),
        "target_free_forward_bytes": target_free_forward_bytes,
        "frozen_set_audit": set_audit.finalize(),
    }


def selector_gate(protocol_checks: dict[str, bool], metrics: dict) -> dict:
    return p1c.selector_gate(protocol_checks, metrics)


def checkpoint_payload(
    *,
    ranker,
    optimizer,
    args,
    config,
    lineage,
    audit,
    step,
    epoch,
    exposure,
    losses,
    step0_metrics,
    final_metrics,
    train_final_diagnostic,
    gate,
    epoch_audits,
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p1d-boundary-band-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "set_ranker": ranker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "samples_seen": exposure,
        "training_args": vars(args),
        "g3_config": asdict(config),
        "lineage": lineage,
        "audit": audit,
        "trainer_sha256": file_sha256(__file__),
        "last_losses": losses,
        "step0_route_metrics": step0_metrics,
        "final_route_metrics": final_metrics,
        "train_fixed_final_diagnostic": train_final_diagnostic,
        "p1d_gate": gate,
        "completed_epoch_audits": [dict(value) for value in epoch_audits],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(p1a.TRAIN_MANIFEST))
    parser.add_argument("--route-manifest", default=str(p1a.ROUTE_SELECT_MANIFEST))
    parser.add_argument("--p1a-checkpoint", default=str(P1A_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6_250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1_250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "epochs": 5,
        "max_steps": 6_250,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_every": 1_250,
        "num_workers": 0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-D requires fixed {name}={expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-D requires fixed CUDA BF16 training")
    if args.log_every < 1:
        raise ValueError("P1-D log interval must be positive")
    paths = {
        "train_manifest": p1a.TRAIN_MANIFEST,
        "route_manifest": p1a.ROUTE_SELECT_MANIFEST,
        "p1a_checkpoint": P1A_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-D requires fixed {name}: {expected}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if file_sha256(p1c.__file__) != P1C_HELPER_SHA256:
        raise RuntimeError("P1-D fixed P1-C helper source SHA differs")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-D CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-D output directory is not empty: {output_dir}")
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

    train_manifest, train_manifest_audit = p1a.load_train_manifest(args.train_manifest)
    route_manifest, route_manifest_audit = p1a.load_route_manifest(args.route_manifest)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_preflight = p1a.stream_preflight(train_manifest, collator)
    route_preflight = p1a.stream_preflight(route_manifest, collator)
    p1a.validate_occupancy(train_preflight, name="P1-D train5k")
    p1a.validate_occupancy(route_preflight, name="P1-D route-select100")
    overlap = p1a.identity_overlap_audit(
        train_manifest, route_manifest, train_preflight, route_preflight
    )
    if not overlap["all_three_identity_zero"]:
        raise ValueError("P1-D train and route identities overlap")
    if (
        train_manifest_audit["sha256"] != p1a.TRAIN_MANIFEST_SHA256
        or train_preflight["sample_keys_sha256"] != p1a.TRAIN_SEQUENCE_SHA256
        or route_preflight["sample_keys_sha256"] != EXPECTED_ROUTE_SEQUENCE_SHA256
    ):
        raise ValueError("P1-D manifest identity/order differs from protocol")

    atom_head, ranker, p1a_payload, config = load_p1a_boundary_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    if vision_sha != p1b.DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P1-D vision checkpoint SHA differs from protocol")
    vision.requires_grad_(False).eval()
    initial_atom_sha = module_state_sha256(atom_head)
    initial_vision_sha = module_state_sha256(vision)
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    validate_optimizer(atom_head, ranker, optimizer)
    lineage = {
        "p1a": {
            "path": str(P1A_CHECKPOINT.resolve()),
            "sha256": P1A_CHECKPOINT_SHA256,
            "global_step": 6_250,
            "atom_head_state_sha256": initial_atom_sha,
        },
        "vision": {
            "path": str(vision_path.resolve()),
            "sha256": vision_sha,
            "state_sha256": initial_vision_sha,
        },
        "p1c_checkpoint_loaded": False,
    }
    audit = {
        "schema_version": "g3-p1d-training-audit-0.1",
        "scope": "train-distribution-boundary-band-loss-only-train5k",
        "manifests": {
            "train5k": train_manifest_audit,
            "route_select100": route_manifest_audit,
            "identity_overlap": overlap,
        },
        "preflight": {"train5k": train_preflight, "route_select100": route_preflight},
        "lineage": lineage,
        "implementation": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(__file__),
            "ranker_sha256": file_sha256(
                Path(__file__).parents[2] / "components/atoms/g3_top128_set_ranker.py"
            ),
            "loss_sha256": file_sha256(
                Path(__file__).parents[2] / "components/atoms/g3_boundary_band_loss.py"
            ),
            "p1c_helper_path": str(Path(p1c.__file__).resolve()),
            "p1c_helper_sha256": file_sha256(p1c.__file__),
        },
        "optimizer": {
            "name": "AdamW",
            "parameter_count": G3_TOP128_SET_RANKER_PARAMETERS,
            "set_ranker_only": True,
            "restored_state": False,
        },
        "loss": {
            "name": "boundary-band-pairwise-softplus",
            "boundary_size": G3_BOUNDARY_BAND_SIZE,
            "margin": 0.0,
            "temperature": 1.0,
            "only_loss": True,
        },
        "streaming": {
            "shuffle": False,
            "num_workers": 0,
            "drop_last": False,
            "whole_dataset_cache": False,
        },
        "target_free_forward": tuple(inspect.signature(ranker.forward).parameters)
        == ("candidates",),
        "target_free_source_contract": target_free_ranker_contract(),
        "forbidden_components_loaded": {
            "p1b_ranker": False,
            "p1c_checkpoint": False,
            "bond": False,
            "atom_pair": False,
            "ocr": False,
            "chemscene": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    route_steps = [0]
    step0 = p1c.evaluate_route(
        atom_head,
        ranker,
        vision,
        route_manifest,
        collator,
        device=device,
        precision=args.precision,
        batch_size=4,
        require_zero_parity=True,
    )
    step0["step"] = 0
    if not all(
        (
            step0["fixed_positive_localization"]["matched"] == 3_080,
            step0["deploy_count"]["count_exact_images"] == 13,
            step0["deploy_count"]["count_mae"] == 3.83,
            step0["zero_score_parity"],
            step0["count_invariant_ranked_scores"],
            step0["target_free_forward_bytes"],
        )
    ):
        raise RuntimeError("P1-D step0 differs from the fixed P1-A state")
    _atomic_json(output_dir / "step0_route_metrics.json", step0)
    ranker.train()

    step = exposure = 0
    saved_steps = []
    epoch_audits = []
    reference_set_hashes = None
    losses = {}
    running = {}
    started = time.perf_counter()
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(1, 6):
        key_digest = hashlib.sha256()
        set_accumulator = p1c.FrozenSetAuditAccumulator()
        epoch_images = epoch_steps = 0
        for batch in p1a.iter_manifest_batches(train_manifest, collator, batch_size=4):
            if len(batch["keys"]) != 4:
                raise RuntimeError("P1-D encountered a residual training batch")
            for key in batch["keys"]:
                key_digest.update(f"{key}\n".encode("utf-8"))
            losses, evidence = p1d_training_step(
                atom_head,
                ranker,
                vision,
                optimizer,
                batch,
                device=device,
                precision=args.precision,
                max_grad_norm=args.max_grad_norm,
            )
            set_accumulator.add(batch["keys"], evidence["candidates"], evidence["hybrid"])
            step += 1
            epoch_steps += 1
            exposure += 4
            epoch_images += 4
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "samples_seen": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{name: value / args.log_every for name, value in running.items()},
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
        frozen_set = set_accumulator.finalize()
        epoch_audit = {
            "epoch": epoch,
            "images": epoch_images,
            "steps": epoch_steps,
            "sample_keys_sha256": key_digest.hexdigest(),
            "frozen_set": frozen_set,
        }
        if (
            epoch_images != 5_000
            or epoch_steps != 1_250
            or key_digest.hexdigest() != p1a.TRAIN_SEQUENCE_SHA256
        ):
            raise RuntimeError("P1-D epoch stream differs from fixed protocol")
        current_hashes = {
            key: value for key, value in frozen_set.items() if key.endswith("sha256")
        }
        if reference_set_hashes is None:
            reference_set_hashes = current_hashes
        elif current_hashes != reference_set_hashes:
            raise RuntimeError("P1-D frozen candidates or hybrid labels changed")
        epoch_audits.append(epoch_audit)
        if step in CHECKPOINT_STEPS[:-1]:
            payload = checkpoint_payload(
                ranker=ranker,
                optimizer=optimizer,
                args=args,
                config=config,
                lineage=lineage,
                audit=audit,
                step=step,
                epoch=epoch,
                exposure=exposure,
                losses=losses,
                step0_metrics=step0,
                final_metrics=None,
                train_final_diagnostic=None,
                gate=None,
                epoch_audits=epoch_audits,
            )
            _atomic_torch_save(output_dir / f"step-{step:06d}.pt", payload)
            _atomic_torch_save(output_dir / "last.pt", payload)
            saved_steps.append(step)
    if step != 6_250 or exposure != 25_000:
        raise RuntimeError("P1-D final step/exposure differs from protocol")

    train_final = evaluate_train_fixed_final(
        atom_head,
        ranker,
        vision,
        train_manifest,
        collator,
        device=device,
        precision=args.precision,
        batch_size=4,
    )
    final = p1c.evaluate_route(
        atom_head,
        ranker,
        vision,
        route_manifest,
        collator,
        device=device,
        precision=args.precision,
        batch_size=4,
        require_zero_parity=False,
    )
    final["step"] = step
    route_steps.append(step)
    final_atom_sha = module_state_sha256(atom_head)
    final_vision_sha = module_state_sha256(vision)
    step0_set = step0["frozen_set_audit"]
    final_set = final["frozen_set_audit"]
    protocol_checks = {
        "train_manifest": train_manifest_audit["sha256"] == p1a.TRAIN_MANIFEST_SHA256,
        "train_sequence": len(epoch_audits) == 5
        and all(
            value["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256
            for value in epoch_audits
        ),
        "frozen_train_set_identity": all(
            {
                key: value
                for key, value in epoch["frozen_set"].items()
                if key.endswith("sha256")
            }
            == reference_set_hashes
            for epoch in epoch_audits
        ),
        "train_final_diagnostic": train_final["images"] == 5_000
        and train_final["sample_keys_sha256"] == p1a.TRAIN_SEQUENCE_SHA256,
        "train_final_frozen_set": all(
            train_final["frozen_set_audit"][key] == value
            for key, value in reference_set_hashes.items()
        ),
        "route_sequence": step0["sample_keys_sha256"]
        == final["sample_keys_sha256"]
        == EXPECTED_ROUTE_SEQUENCE_SHA256,
        "route_candidate_identity": all(
            step0_set[key] == final_set[key]
            for key in step0_set
            if key.endswith("sha256")
        ),
        "identity_disjoint": overlap["all_three_identity_zero"],
        "p1a_checkpoint": file_sha256(P1A_CHECKPOINT) == P1A_CHECKPOINT_SHA256,
        "p1c_checkpoint_absent": lineage["p1c_checkpoint_loaded"] is False,
        "p1c_helper_source_fixed": (
            audit["implementation"]["p1c_helper_sha256"]
            == P1C_HELPER_SHA256
            == file_sha256(p1c.__file__)
        ),
        "atom_state_unchanged": initial_atom_sha
        == final_atom_sha
        == P1A_ATOM_STATE_SHA256,
        "vision_state_unchanged": initial_vision_sha == final_vision_sha,
        "set_ranker_only_optimizer": validate_optimizer(atom_head, ranker, optimizer)
        is None,
        "boundary_loss_only": audit["loss"]["only_loss"]
        and audit["loss"]["boundary_size"] == 8,
        "fixed_exposure": step == 6_250 and exposure == 25_000,
        "checkpoint_schedule": saved_steps + [step] == list(CHECKPOINT_STEPS),
        "route_steps": route_steps == [0, 6_250],
        "count_vector_unchanged": step0["deploy_count_vector"]
        == final["deploy_count_vector"],
        "count_logits_unchanged": step0["count_logits_sha256"]
        == final["count_logits_sha256"],
        "deploy_count_metrics_unchanged": final["deploy_count"]["count_exact_images"]
        == 13
        and final["deploy_count"]["count_mae"] == 3.83,
        "count_invariant_scores": step0["count_invariant_ranked_scores"]
        and final["count_invariant_ranked_scores"],
        "target_free_forward": (
            audit["target_free_forward"]
            and audit["target_free_source_contract"]["passed"]
            and step0["count_invariant_ranked_scores"]
            and final["count_invariant_ranked_scores"]
        ),
        "forbidden_components_absent": not any(
            audit["forbidden_components_loaded"].values()
        ),
    }
    gate = selector_gate(protocol_checks, final)
    final_payload = checkpoint_payload(
        ranker=ranker,
        optimizer=optimizer,
        args=args,
        config=config,
        lineage=lineage,
        audit=audit,
        step=step,
        epoch=5,
        exposure=exposure,
        losses=losses,
        step0_metrics=step0,
        final_metrics=final,
        train_final_diagnostic=train_final,
        gate=gate,
        epoch_audits=epoch_audits,
    )
    final_path = output_dir / "step-006250.pt"
    _atomic_torch_save(final_path, final_payload)
    _atomic_torch_save(output_dir / "last.pt", final_payload)
    saved_steps.append(step)
    expected_names = [f"step-{value:06d}.pt" for value in CHECKPOINT_STEPS]
    if sorted(path.name for path in output_dir.glob("step-*.pt")) != expected_names:
        raise RuntimeError("P1-D checkpoint file set differs from protocol")
    _atomic_json(output_dir / "epoch_audit.json", epoch_audits)
    _atomic_json(output_dir / "train_fixed_final_diagnostic.json", train_final)
    _atomic_json(output_dir / "final_route_metrics.json", final)
    _atomic_json(output_dir / "p1d_gate.json", gate)
    _atomic_json(
        output_dir / "status.json",
        {
            "status": "completed",
            "decision": gate["decision"],
            "global_step": step,
            "samples_seen": exposure,
            "route_evaluation_steps": route_steps,
            "train_diagnostic_steps": [step],
            "checkpoint": str(final_path.resolve()),
            "checkpoint_sha256": file_sha256(final_path),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
