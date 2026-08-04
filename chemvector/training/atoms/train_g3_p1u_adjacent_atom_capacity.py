#!/usr/bin/env python3
"""Run the dual-domain P1-U adjacent-Atom capacity gate from P1-T initialization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride2AtomCenterField,
    decode_atom_center_field,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1s_phase_residual_center_field as p1s
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.graph.scenes.g3_structural_data import (
    OVERFIT32_MANIFEST,
    load_g3_overfit32_manifest,
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p1u-adjacent-atom-capacity64"
OUTPUT = Path("outputs/smoke/chemvector-g3-p1u-adjacent-atom-capacity64")
ADJACENT_AUDIT = Path(
    "outputs/audits/chemvector-g3-adjacent-atom-capacity32/audit.json"
)
ADJACENT_MANIFEST = Path(
    "outputs/audits/chemvector-g3-adjacent-atom-capacity32/manifest.jsonl"
)
ADJACENT_MANIFEST_SHA256 = (
    "6e4eca6fcace2e9e20a29c0414972cc5b967d71da1df9f7e31034c083f07442f"
)
ADJACENT_AUDIT_SHA256 = (
    "5b9f82fe958beafdf5c48e4a5c607d8787768263c3094690859f8106152aa8e7"
)
P1T_CHECKPOINT = p1t.OUTPUT / "step-002400.pt"
STEPS = 4_800
SAVE_EVERY = 800
EXPECTED_EXPOSURE = 19_200
EXPECTED_COLLISION_IMAGES = 7


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


def validate_args(args) -> None:
    fixed = {
        "batch_size": 4,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-U requires fixed {name}={expected}")
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError(f"P1-U requires output directory {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-U requires CUDA BF16")


def load_adjacent_manifest() -> tuple[list, dict, dict]:
    if file_sha256(ADJACENT_AUDIT) != ADJACENT_AUDIT_SHA256:
        raise ValueError("P1-U adjacent audit SHA256 differs")
    audit = json.loads(ADJACENT_AUDIT.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("collision_images") != EXPECTED_COLLISION_IMAGES
        or audit.get("manifest_sha256") != ADJACENT_MANIFEST_SHA256
    ):
        raise ValueError("P1-U adjacent audit contract differs")
    manifest, manifest_audit = load_g3_sample_manifest(
        ADJACENT_MANIFEST,
        expected_sha256=ADJACENT_MANIFEST_SHA256,
        expected_samples=32,
        expected_molecules=29,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    return manifest, manifest_audit, audit


def alternating_domain_batches(overfit_manifest, adjacent_manifest, collator):
    overfit = p1a.iter_manifest_batches(overfit_manifest, collator, batch_size=4)
    adjacent = p1a.iter_manifest_batches(adjacent_manifest, collator, batch_size=4)
    for overfit_batch, adjacent_batch in zip(overfit, adjacent, strict=True):
        yield "overfit32", overfit_batch
        yield "adjacent32", adjacent_batch


def _sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def domain_capacity_checks(metrics: dict, *, images: int) -> dict[str, bool]:
    known = metrics["known_full_count"]
    reach = metrics["top128_reachability"]
    return {
        "known_all_atoms": known["matched_atoms"] == known["target_atoms"],
        "known_all_images": known["whole_set_exact_images"] == images,
        "reach_all_atoms": reach["matched_targets"] == reach["targets"],
        "reach_all_images": reach["exact_images"] == images,
        "no_peak_shortfall": metrics["regional_maxima"]["peak_shortfall_images"] == 0,
    }


@torch.no_grad()
def evaluate_collision_pairs(
    field, atom_head, vision, frozen_block, manifest, collator, adjacent_audit, *, device
) -> dict:
    collision_records = {
        item["sample_key"]: item
        for item in adjacent_audit["records"]
        if item["same_stride4_cell"]
    }
    results = []
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        objects, stroke = p1t.model_inputs(
            atom_head, vision, frozen_block, batch["pixel_values"],
            device=device, training=False,
        )
        with _autocast_context(device, "bf16"):
            output = field(objects, stroke)
        peaks = extract_regional_maximum_objects(decode_atom_center_field(output))
        for index, sample_key in enumerate(batch["keys"]):
            record = collision_records.get(str(sample_key))
            if record is None:
                continue
            expected = batch["atom_anchors"][index, batch["atom_mask"][index]].float()
            first, second = record["nearest_atom_indices"]
            pair = expected[torch.tensor([first, second])]
            count = len(expected)
            known = p1a.atom_sample_statistics(peaks[index]["anchors"][:count], pair)
            reach = p1a.atom_sample_statistics(peaks[index]["anchors"], pair)
            results.append(
                {
                    "sample_key": str(sample_key),
                    "minimum_distance_pixels": record["minimum_distance_pixels"],
                    "known_distinct_peaks": int(known["matched_atoms"]) == 2,
                    "top128_distinct_peaks": int(reach["matched_atoms"]) == 2,
                }
            )
    results.sort(key=lambda item: item["sample_key"])
    return {
        "collision_images": len(results),
        "known_distinct_images": sum(item["known_distinct_peaks"] for item in results),
        "top128_distinct_images": sum(item["top128_distinct_peaks"] for item in results),
        "records": results,
    }


def capacity_gate(overfit_metrics, adjacent_metrics, collision_metrics):
    overfit_checks = domain_capacity_checks(overfit_metrics, images=32)
    adjacent_checks = domain_capacity_checks(adjacent_metrics, images=32)
    collision_checks = {
        "all_collision_images_seen": collision_metrics["collision_images"]
        == EXPECTED_COLLISION_IMAGES,
        "known_pairs_use_distinct_peaks": collision_metrics["known_distinct_images"]
        == EXPECTED_COLLISION_IMAGES,
        "top128_pairs_use_distinct_peaks": collision_metrics["top128_distinct_images"]
        == EXPECTED_COLLISION_IMAGES,
    }
    checks = {
        "overfit32": overfit_checks,
        "adjacent32": adjacent_checks,
        "collision_pairs": collision_checks,
    }
    passed = all(
        value
        for group in checks.values()
        for value in group.values()
    )
    return ("dual_capacity_go" if passed else "dual_capacity_no_go", checks)


def checkpoint_payload(
    field, vision_block, optimizer, args, lineage, *, step, exposure, losses, metrics=None
):
    payload = p1t.checkpoint_payload(
        field, vision_block, optimizer, args, lineage,
        step=step, exposure=exposure, losses=losses, metrics=metrics,
    )
    payload.update(
        {
            "checkpoint_schema_version": "g3-p1u-adjacent-atom-capacity-0.1",
            "model_type": MODEL_TYPE,
            "training_domains": ["overfit32", "adjacent32"],
            "alternating_domain_batches": True,
        }
    )
    return payload


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-U output directory is not empty")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.requires_grad_(True)
    atom_sha = module_state_sha256(atom_head)
    vision_initial_sha = module_state_sha256(vision)
    block_initial_sha = module_state_sha256(block)
    frozen_block_sha = module_state_sha256(frozen_block)
    frozen_vision_initial_sha = p1t.frozen_vision_sha256(vision)
    field = ContinuousStride2AtomCenterField(atom_head).to(device)
    field_initial_state = {
        name: value.detach().cpu().clone() for name, value in field.named_parameters()
    }
    field_initial_sha = module_state_sha256(field)
    parameters = list(field.parameters()) + list(block.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    p1t.validate_optimizer(field, atom_head, vision, optimizer)

    overfit_manifest, overfit_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    adjacent_manifest, adjacent_manifest_audit, adjacent_audit = load_adjacent_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    first_batch = next(p1a.iter_manifest_batches(overfit_manifest, collator, batch_size=4))
    first_objects, first_stroke = p1t.model_inputs(
        atom_head, vision, frozen_block, first_batch["pixel_values"],
        device=device, training=False,
    )
    parity = p1s.step0_parity(field, atom_head, first_objects, first_stroke)
    if not parity["exact"]:
        raise RuntimeError("P1-U zero-start P1-A base parity failed")
    p1t_payload = torch.load(P1T_CHECKPOINT, map_location="cpu", weights_only=False)
    p1t_lineage = p1t_payload["lineage"]
    initialization_checks = {
        "field_matches_p1t": field_initial_sha
        == p1t_lineage["field_initialization_sha256"],
        "vision_matches_p1t": vision_initial_sha
        == p1t_lineage["vision_initial_state_sha256"],
        "block_matches_p1t": block_initial_sha
        == p1t_lineage["vision_block_initial_sha256"],
        "frozen_base_matches_p1t": frozen_block_sha
        == p1t_lineage["frozen_base_block_sha256"],
    }
    if not all(initialization_checks.values()):
        raise RuntimeError("P1-U initialization differs from P1-T")

    first_epoch = list(
        alternating_domain_batches(overfit_manifest, adjacent_manifest, collator)
    )
    epoch_domains = [domain for domain, _ in first_epoch]
    epoch_keys = [key for _, batch in first_epoch for key in batch["keys"]]
    if len(first_epoch) != 16 or epoch_domains != ["overfit32", "adjacent32"] * 8:
        raise RuntimeError("P1-U alternating epoch protocol differs")
    epoch_keys_sha = _sequence_sha256(epoch_keys)
    lineage = {
        "p1a_checkpoint": str(p1d.P1A_CHECKPOINT.resolve()),
        "p1a_checkpoint_sha256": file_sha256(p1d.P1A_CHECKPOINT),
        "atom_state_sha256": atom_sha,
        "vision_checkpoint_sha256": vision_sha,
        "vision_initial_state_sha256": vision_initial_sha,
        "vision_block_initial_sha256": block_initial_sha,
        "frozen_base_block_sha256": frozen_block_sha,
        "frozen_vision_initial_sha256": frozen_vision_initial_sha,
        "field_initialization_sha256": field_initial_sha,
        "step0_parity": parity,
        "initialization_checks": initialization_checks,
        "p1t_checkpoint": str(P1T_CHECKPOINT.resolve()),
        "p1t_checkpoint_sha256": file_sha256(P1T_CHECKPOINT),
        "overfit32_manifest": overfit_audit,
        "adjacent32_manifest": adjacent_manifest_audit,
        "adjacent32_audit_sha256": ADJACENT_AUDIT_SHA256,
        "training_epoch_domains": epoch_domains,
        "training_epoch_keys_sha256": epoch_keys_sha,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    step = exposure = epoch = 0
    last_losses = {}
    observed_epoch_sha = None
    started = time.perf_counter()
    while step < STEPS:
        epoch += 1
        current_keys = []
        for domain, batch in alternating_domain_batches(
            overfit_manifest, adjacent_manifest, collator
        ):
            current_keys.extend(batch["keys"])
            field.train()
            vision.eval()
            last_losses = p1t.training_step(
                field, atom_head, vision, frozen_block, optimizer, batch,
                device=device, max_grad_norm=args.max_grad_norm,
            )
            step += 1
            exposure += len(batch["keys"])
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "exposure": exposure,
                    "epoch": epoch,
                    "domain": domain,
                    **last_losses,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
            if step % SAVE_EVERY == 0 and step < STEPS:
                payload = checkpoint_payload(
                    field, block, optimizer, args, lineage,
                    step=step, exposure=exposure, losses=last_losses,
                )
                _atomic_torch_save(args.output_dir / f"step-{step:06d}.pt", payload)
                _atomic_torch_save(args.output_dir / "last.pt", payload)
        current_sha = _sequence_sha256(current_keys)
        if current_sha != epoch_keys_sha:
            raise RuntimeError("P1-U training epoch sample order changed")
        observed_epoch_sha = current_sha

    overfit_metrics = p1t.evaluate(
        field, atom_head, vision, frozen_block, overfit_manifest, collator, device=device
    )
    adjacent_metrics = p1t.evaluate(
        field, atom_head, vision, frozen_block, adjacent_manifest, collator, device=device
    )
    collision_metrics = evaluate_collision_pairs(
        field, atom_head, vision, frozen_block, adjacent_manifest, collator,
        adjacent_audit, device=device,
    )
    repeated = {
        "overfit32": p1t.evaluate(
            field, atom_head, vision, frozen_block, overfit_manifest, collator,
            device=device,
        ),
        "adjacent32": p1t.evaluate(
            field, atom_head, vision, frozen_block, adjacent_manifest, collator,
            device=device,
        ),
        "collision_pairs": evaluate_collision_pairs(
            field, atom_head, vision, frozen_block, adjacent_manifest, collator,
            adjacent_audit, device=device,
        ),
    }
    decision, checks = capacity_gate(
        overfit_metrics, adjacent_metrics, collision_metrics
    )
    metrics = {
        "overfit32": overfit_metrics,
        "adjacent32": adjacent_metrics,
        "collision_pairs": collision_metrics,
    }
    changed = {
        name: not torch.equal(field_initial_state[name], value.detach().cpu())
        for name, value in field.named_parameters()
    }
    protocol = {
        "initialization_matches_p1t": all(initialization_checks.values()),
        "parameter_count_92035": sum(value.numel() for value in parameters)
        == p1t.TRAINABLE_PARAMETER_COUNT,
        "optimizer_field_and_last_block_only": p1t.validate_optimizer(
            field, atom_head, vision, optimizer
        ) is None,
        "all_field_tensors_changed": all(changed.values()),
        "atom_state_unchanged": module_state_sha256(atom_head) == atom_sha,
        "vision_state_changed": module_state_sha256(vision) != vision_initial_sha,
        "last_block_changed": module_state_sha256(block) != block_initial_sha,
        "frozen_base_block_unchanged": module_state_sha256(frozen_block)
        == frozen_block_sha,
        "frozen_vision_unchanged": p1t.frozen_vision_sha256(vision)
        == frozen_vision_initial_sha,
        "step0_base_exact": parity["exact"],
        "fixed_steps": step == STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "fixed_epochs": epoch == 300,
        "alternating_epoch_order": observed_epoch_sha == epoch_keys_sha,
        "repeat_metrics": repeated == metrics,
    }
    if not all(protocol.values()):
        decision = "protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol,
        "changed_field_parameter_tensors": changed,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = checkpoint_payload(
        field, block, optimizer, args, lineage,
        step=step, exposure=exposure, losses=last_losses, metrics=final,
    )
    final_path = args.output_dir / "step-004800.pt"
    _atomic_torch_save(final_path, payload)
    _atomic_torch_save(args.output_dir / "last.pt", payload)
    final["checkpoint"] = str(final_path.resolve())
    final["checkpoint_sha256"] = file_sha256(final_path)
    _atomic_json(args.output_dir / "final_metrics.json", final)
    _atomic_json(args.output_dir / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps({"decision": decision, "checkpoint": str(final_path)}, indent=2))


if __name__ == "__main__":
    main()
