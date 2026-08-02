#!/usr/bin/env python3
"""Fine-tune bbox CTC on deployed v0.10.5 predicted TEXT AtomObjects."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from PIL import Image
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.ctc_ocr import ctc_loss, encode_text
from chemvector.data import chemscene_atom_labels
from chemvector.direct_atom_role import enumerate_selected_atoms
from chemvector.evaluate_direct_atom_ocr import (
    collect,
    load_bbox,
    load_ctc,
    load_role,
    summarize,
)
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    frozen_atom_objects,
    load_stratified_fresh_records,
    match_atom_targets,
)
from chemvector.train_joint_bond_set_multidomain import mixed_batches, verify_data_gate
from chemvector.evaluate_direct_atom_ocr import crop_tensor


DEFAULT_ROLE = "outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k/selected.pt"
DEFAULT_CTC = "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt"
DEFAULT_BBOX = "outputs/checkpoints/chemvector-v0108-direct-atom-text-bbox-style10k/selected.pt"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    parser.add_argument("--initial-ctc-checkpoint", default=DEFAULT_CTC)
    parser.add_argument("--bbox-checkpoint", default=DEFAULT_BBOX)
    parser.add_argument("--style-train-samples", type=int, default=6_668)
    parser.add_argument("--chemvision-replay-samples", type=int, default=1_666)
    parser.add_argument("--font-v2-replay-samples", type=int, default=1_666)
    parser.add_argument("--fresh-calibration-samples", type=int, default=500)
    parser.add_argument("--fresh-eval-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--extent-scale", type=float, default=1.0)
    parser.add_argument(
        "--true-text-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-every-total-samples", type=int, default=2_500)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def training_crops(
    samples,
    scenes,
    atoms,
    mapping,
    atom_indices,
    sample_indices,
    text_rows,
    extent_scale,
    learned_boxes=None,
):
    images = [
        sample["image"].convert("RGB").resize((768, 768), Image.Resampling.BICUBIC)
        for sample in samples
    ]
    labels_by_sample = [chemscene_atom_labels(scene) for scene in scenes]
    crops = []
    labels = []
    for output_index, row in enumerate(text_rows.tolist()):
        batch_index = int(sample_indices[row])
        atom_index = int(atom_indices[row])
        if learned_boxes is None:
            geometry = atoms["atom_geometry"][batch_index, atom_index].float().cpu() * 768
            x, y = geometry[4:6].tolist()
            extent_x, extent_y = geometry[6:8].tolist()
        else:
            x, y = (learned_boxes["centers"][output_index].float().cpu() * 768).tolist()
            extent_x, extent_y = (
                learned_boxes["half_extents"][output_index].float().cpu() * 768
            ).tolist()
        half_width = max(extent_x * extent_scale, 1.0)
        half_height = max(extent_y * extent_scale, 1.0)
        crops.append(
            crop_tensor(
                images[batch_index],
                (x - half_width, y - half_height, x + half_width, y + half_height),
            )
        )
        target_index = int(mapping[batch_index, atom_index])
        labels.append(
            labels_by_sample[batch_index][target_index][0]
            if target_index >= 0
            else ""
        )
    return crops, labels


def save(path, model, optimizer, args, samples_seen, validation, audit):
    torch.save(
        {
            "checkpoint_schema_version": "direct-atom-ocr-0.1",
            "pretrained_ctc_ocr": model.state_dict(),
            "config": asdict(model.config),
            "optimizer": optimizer.state_dict(),
            "training_args": vars(args),
            "samples_seen": samples_seen,
            "validation": validation,
            "audit": audit,
        },
        path,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Deployed Atom OCR training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    role, role_payload = load_role(args.role_checkpoint, device)
    ocr, _ = load_ctc(args.initial_ctc_checkpoint, device)
    bbox, _ = load_bbox(args.bbox_checkpoint, device)
    ocr.requires_grad_(True)
    ocr.unfreeze_pixel_encoder()
    initial_pixel_state = {
        name: value.detach().cpu().clone()
        for name, value in ocr.pixel_encoder.state_dict().items()
    }
    training_args = role_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        training_args["source_checkpoint"], training_args["grounded_graph_checkpoint"], device
    )
    for module in (vision, atom, bond, set_head, role, bbox):
        module.eval().requires_grad_(False)
    config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds)
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    role_threshold = float(role_payload["validation"]["threshold"])
    fresh, fresh_audit = load_stratified_fresh_records(
        args.fresh_calibration_samples + args.fresh_eval_samples
    )
    calibration_records = fresh[: args.fresh_calibration_samples]
    evaluation_records = fresh[args.fresh_calibration_samples :]
    optimizer = torch.optim.AdamW(ocr.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    audit = {
        "data_gate": verify_data_gate(),
        "fresh": fresh_audit,
        "label_protocol": "chemscene-atom-object-order",
        "bbox_checkpoint": args.bbox_checkpoint,
        "pixel_encoder_runtime_unfrozen": True,
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
    modules = (vision, atom, bond, set_head, atom_threshold)
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
            vision, atom, bond, set_head, pixels, config, atom_threshold
        )
        mapping, _ = match_atom_targets(atoms, targets, config)
        atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            role_probabilities = role(
                pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices
            ).sigmoid()
        text_rows = torch.where(role_probabilities >= role_threshold)[0]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            learned_boxes = bbox(
                pixels,
                atoms["atom_geometry"][..., :2],
                atom_indices[text_rows],
                sample_indices[text_rows],
            )
        crops, labels = training_crops(
            samples,
            batch["scenes"],
            atoms,
            mapping,
            atom_indices,
            sample_indices,
            text_rows,
            args.extent_scale,
            learned_boxes,
        )
        if args.true_text_only:
            retained = [index for index, label in enumerate(labels) if label]
            crops = [crops[index] for index in retained]
            labels = [labels[index] for index in retained]
        if not crops:
            continue
        encoded = [encode_text(label, ocr.config.maximum_bytes) for label in labels]
        lengths = torch.tensor([len(value) for value in encoded], device=device)
        targets_flat = torch.cat(encoded).to(device) if encoded else torch.empty(0, device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        ocr.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = ocr(torch.stack(crops).to(device))
        loss = ctc_loss(
            logits.float().cpu(), targets_flat.cpu(), lengths.cpu()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ocr.parameters(), 1.0)
        optimizer.step()
        seen += len(samples)
        if seen % 1000 < len(samples):
            print(json.dumps({"samples_seen": seen, "crops": len(crops), "loss": float(loss.detach()), "domain_seen": domain_seen}), flush=True)
        if seen % args.validate_every_total_samples < len(samples) or seen == total_target:
            records = collect(
                role, ocr, modules, calibration_records, collator, config,
                role_threshold, device, args.batch_size, bbox
            )
            validation = summarize(records, "learned_bbox")
            validation.update({"samples_seen": seen, "elapsed_seconds": time.perf_counter() - started})
            score = (validation["state_exact_rate"], validation["state_accuracy"], validation["text_raw_accuracy"])
            save(output / "last.pt", ocr, optimizer, args, seen, validation, audit)
            if best_score is None or score > best_score:
                best_score = score
                save(output / "best.pt", ocr, optimizer, args, seen, validation, audit)
            print(json.dumps({"validation": validation}), flush=True)
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    ocr.load_state_dict(payload["pretrained_ctc_ocr"])
    pixel_encoder_max_abs_delta = max(
        float((value.detach().cpu() - initial_pixel_state[name]).abs().max())
        for name, value in ocr.pixel_encoder.state_dict().items()
    )
    calibration_gate = {
        "state_accuracy_min": 0.95,
        "state_exact_rate_min": 0.20,
        "passed": payload["validation"]["state_accuracy"] >= 0.95
        and payload["validation"]["state_exact_rate"] >= 0.20
        and pixel_encoder_max_abs_delta > 0.0,
        "pixel_encoder_max_abs_delta": pixel_encoder_max_abs_delta,
    }
    if not calibration_gate["passed"]:
        report = {
            "calibration": payload["validation"],
            "calibration_gate": calibration_gate,
            "fresh_evaluation_read": False,
            "sealed_test_read": False,
        }
        (output / "calibration_no_go.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        payload["fresh_evaluation"] = None
        payload["calibration_gate"] = calibration_gate
        torch.save(payload, output / "selected.pt")
        print(json.dumps({"final": report}), flush=True)
        return
    records = collect(
        role, ocr, modules, evaluation_records, collator, config,
        role_threshold, device, args.batch_size, bbox
    )
    evaluation = summarize(records, "learned_bbox")
    gate = {
        "passed": evaluation["state_accuracy"] >= 0.95
        and evaluation["state_exact_rate"] >= 0.20
    }
    report = {"evaluation": evaluation, "gate": gate, "sealed_test_read": False}
    (output / "fresh_eval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["fresh_evaluation"] = report
    torch.save(payload, output / "selected.pt")
    print(json.dumps({"final": report}), flush=True)


if __name__ == "__main__":
    main()
