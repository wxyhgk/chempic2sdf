#!/usr/bin/env python3
"""Train v0.10.8 anchor-local TEXT bbox regression with frozen structure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.direct_atom_role import enumerate_selected_atoms
from chemvector.direct_atom_text_bbox import (
    DirectAtomTextBBox,
    DirectAtomTextBBoxConfig,
    direct_atom_text_bbox_loss,
)
from chemvector.evaluate_direct_atom_ocr import load_role
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    frozen_atom_objects,
    load_stratified_fresh_records,
    match_atom_targets,
)
from chemvector.train_joint_bond_set_multidomain import mixed_batches, verify_data_gate


DEFAULT_ROLE = "outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k/selected.pt"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0108-direct-atom-text-bbox-style10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    parser.add_argument("--style-train-samples", type=int, default=6_668)
    parser.add_argument("--chemvision-replay-samples", type=int, default=1_666)
    parser.add_argument("--font-v2-replay-samples", type=int, default=1_666)
    parser.add_argument("--fresh-calibration-samples", type=int, default=500)
    parser.add_argument("--fresh-eval-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every-total-samples", type=int, default=2_500)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def matched_text_rows(atoms, targets, mapping):
    atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
    mapped = mapping[sample_indices, atom_indices]
    valid = mapped >= 0
    atom_indices = atom_indices[valid]
    sample_indices = sample_indices[valid]
    mapped = mapped[valid]
    text = targets["atom_text_mask"][sample_indices, mapped]
    return atom_indices[text], sample_indices[text], mapped[text]


def _box_iou(centers, half_extents, targets):
    predicted_min = centers - half_extents
    predicted_max = centers + half_extents
    target_half = targets[:, 2:] / 2.0
    target_min = targets[:, :2] - target_half
    target_max = targets[:, :2] + target_half
    intersection = (
        (torch.minimum(predicted_max, target_max) - torch.maximum(predicted_min, target_min))
        .clamp_min(0.0)
        .prod(dim=-1)
    )
    predicted_area = (2.0 * half_extents).prod(dim=-1)
    target_area = (2.0 * target_half).prod(dim=-1)
    return intersection / (predicted_area + target_area - intersection).clamp_min(1e-9)


@torch.no_grad()
def collect(
    model, vision, atom, bond, set_head, records, collator, config,
    atom_threshold, device, batch_size,
):
    totals = {
        name: 0.0
        for name in (
            "objects", "candidate_iou_sum", "candidate_iou50", "candidate_center_error_sum",
            "baseline_iou_sum", "baseline_iou50", "baseline_center_error_sum",
        )
    }
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        _, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, config, atom_threshold
        )
        mapping, _ = match_atom_targets(atoms, targets, config)
        atom_indices, sample_indices, mapped = matched_text_rows(atoms, targets, mapping)
        if not len(atom_indices):
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices
            )
        expected = targets["atom_text_bbox"][sample_indices, mapped].float()
        centers = output["centers"].float()
        extents = output["half_extents"].float()
        baseline_centers = atoms["atom_geometry"][sample_indices, atom_indices, :2].float()
        baseline_extents = atoms["atom_geometry"][sample_indices, atom_indices, 6:8].float() * 1.5
        candidate_iou = _box_iou(centers, extents, expected)
        baseline_iou = _box_iou(baseline_centers, baseline_extents, expected)
        totals["objects"] += len(atom_indices)
        totals["candidate_iou_sum"] += float(candidate_iou.sum())
        totals["candidate_iou50"] += int((candidate_iou >= 0.5).sum())
        totals["candidate_center_error_sum"] += float(
            torch.linalg.vector_norm((centers - expected[:, :2]) * 768.0, dim=-1).sum()
        )
        totals["baseline_iou_sum"] += float(baseline_iou.sum())
        totals["baseline_iou50"] += int((baseline_iou >= 0.5).sum())
        totals["baseline_center_error_sum"] += float(
            torch.linalg.vector_norm((baseline_centers - expected[:, :2]) * 768.0, dim=-1).sum()
        )
    count = max(int(totals["objects"]), 1)
    return {
        **totals,
        "candidate_mean_iou": totals["candidate_iou_sum"] / count,
        "candidate_iou50_rate": totals["candidate_iou50"] / count,
        "candidate_center_error_px": totals["candidate_center_error_sum"] / count,
        "baseline_mean_iou": totals["baseline_iou_sum"] / count,
        "baseline_iou50_rate": totals["baseline_iou50"] / count,
        "baseline_center_error_px": totals["baseline_center_error_sum"] / count,
    }


def save(path, model, optimizer, args, config, seen, validation, audit):
    torch.save(
        {
            "checkpoint_schema_version": "direct-atom-text-bbox-0.1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "training_args": vars(args),
            "samples_seen": seen,
            "validation": validation,
            "audit": audit,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Direct Atom text bbox training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    role, role_payload = load_role(args.role_checkpoint, device)
    training_args = role_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        training_args["source_checkpoint"], training_args["grounded_graph_checkpoint"], device
    )
    for module in (role, vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=proposal_config.maximum_atoms,
            bond_queries=proposal_config.maximum_bonds,
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    fresh, fresh_audit = load_stratified_fresh_records(
        args.fresh_calibration_samples + args.fresh_eval_samples
    )
    calibration = fresh[: args.fresh_calibration_samples]
    evaluation = fresh[args.fresh_calibration_samples :]
    bbox_config = DirectAtomTextBBoxConfig()
    model = DirectAtomTextBBox(bbox_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    audit = {
        "data_gate": verify_data_gate(),
        "fresh": fresh_audit,
        "label_protocol": "chemscene-atom-object-order",
        "sealed_test_read": False,
    }
    domain_counts = {
        "style_train": args.style_train_samples,
        "chemvision_replay": args.chemvision_replay_samples,
        "font_v2_replay": args.font_v2_replay_samples,
    }
    total_target = sum(domain_counts.values())
    seen = 0
    best_score = None
    started = time.perf_counter()
    for samples, domain_seen in mixed_batches(
        domain_sample_counts=domain_counts,
        domain_batch_counts={"style_train": 4, "chemvision_replay": 1, "font_v2_replay": 1},
        batch_size=args.batch_size,
        config=collator.config,
        seed=args.seed,
        shuffle_buffer=1024,
    ):
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        _, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, proposal_config, atom_threshold
        )
        mapping, _ = match_atom_targets(atoms, targets, proposal_config)
        atom_indices, sample_indices, mapped = matched_text_rows(atoms, targets, mapping)
        if len(atom_indices):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted = model(
                    pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices
                )
                losses = direct_atom_text_bbox_loss(
                    predicted,
                    atoms["atom_geometry"][sample_indices, atom_indices, :2],
                    targets["atom_text_bbox"][sample_indices, mapped],
                    bbox_config,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        seen += len(samples)
        if seen % 1000 < len(samples):
            print(json.dumps({"samples_seen": seen, "loss": float(losses["loss"].detach()), "domain_seen": domain_seen}), flush=True)
        if seen % args.validate_every_total_samples < len(samples) or seen == total_target:
            metrics = collect(
                model, vision, atom, bond, set_head, calibration, collator,
                proposal_config, atom_threshold, device, args.batch_size,
            )
            metrics.update({"samples_seen": seen, "elapsed_seconds": time.perf_counter() - started})
            score = (metrics["candidate_mean_iou"], -metrics["candidate_center_error_px"])
            save(output_dir / "last.pt", model, optimizer, args, bbox_config, seen, metrics, audit)
            if best_score is None or score > best_score:
                best_score = score
                save(output_dir / "best.pt", model, optimizer, args, bbox_config, seen, metrics, audit)
            print(json.dumps({"validation": metrics}), flush=True)
    payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    metrics = collect(
        model, vision, atom, bond, set_head, evaluation, collator,
        proposal_config, atom_threshold, device, args.batch_size,
    )
    metrics["gate"] = {
        "passed": metrics["candidate_mean_iou"] >= metrics["baseline_mean_iou"] + 0.10
        and metrics["candidate_center_error_px"] <= metrics["baseline_center_error_px"] * 0.60
    }
    metrics["sealed_test_read"] = False
    (output_dir / "fresh_eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["fresh_evaluation"] = metrics
    torch.save(payload, output_dir / "selected.pt")
    print(json.dumps({"final": metrics}), flush=True)


if __name__ == "__main__":
    main()
