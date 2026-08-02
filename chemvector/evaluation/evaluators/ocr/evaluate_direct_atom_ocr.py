#!/usr/bin/env python3
"""Calibrate anchor-centered bbox CTC OCR behind the v0.10.5 role gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.ctc_ocr import (
    PretrainedPixelCTCConfig,
    PretrainedPixelCTCOCR,
    bbox_crop,
    decode_ctc_with_score,
)
from chemvector.data import chemscene_atom_labels
from chemvector.direct_atom_role import (
    DirectAtomRoleConfig,
    DirectAtomRoleGate,
    enumerate_selected_atoms,
)
from chemvector.direct_atom_text_bbox import (
    DirectAtomTextBBox,
    DirectAtomTextBBoxConfig,
)
from chemvector.ocr import parse_atom_display_text
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import (
    frozen_atom_objects,
    load_stratified_fresh_records,
    match_atom_targets,
)
from chemvector.train_joint_bond_set_multidomain import verify_data_gate


DEFAULT_ROLE = "outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k/selected.pt"
DEFAULT_CTC = "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt"
DEFAULT_OUTPUT = "outputs/evaluations/chemvector-v0106-anchor-ctc-calibration"
RADII = (6, 8, 10, 12, 16)
EXTENT_SCALES = (0.75, 1.0, 1.25, 1.5)
MODES = tuple(f"radius_{value}" for value in RADII) + tuple(
    f"extent_{value:.2f}" for value in EXTENT_SCALES
) + tuple(
    f"bbox_extent_{value:.2f}" for value in EXTENT_SCALES
)
ORACLE_MODE = "oracle_bbox"
LEARNED_MODE = "learned_bbox"


def atom_state(text: str) -> tuple | None:
    if text == "":
        return ("C", 0, 0, 0, 0)
    parsed = parse_atom_display_text(text)
    if parsed is None:
        return None
    return (
        parsed["element"],
        parsed["formal_charge"],
        parsed["isotope"],
        parsed["explicit_h_count"],
        parsed["radical_electrons"],
    )


def load_role(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "direct-atom-role-0.1":
        raise ValueError("Role checkpoint schema differs")
    model = DirectAtomRoleGate(DirectAtomRoleConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    return model.eval().requires_grad_(False), payload


def load_ctc(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PretrainedPixelCTCOCR(PretrainedPixelCTCConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["pretrained_ctc_ocr"])
    model.pixel_encoder_loaded = True
    return model.eval().requires_grad_(False), payload


def load_bbox(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "direct-atom-text-bbox-0.1":
        raise ValueError("Text bbox checkpoint schema differs")
    model = DirectAtomTextBBox(DirectAtomTextBBoxConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    return model.eval().requires_grad_(False), payload


def crop_tensor(image, bbox: tuple[float, float, float, float]) -> torch.Tensor:
    crop = bbox_crop(image, bbox, context=4.0)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


@torch.no_grad()
def decode_crops(model, crops: list[torch.Tensor], device, chunk_size=256):
    decoded = []
    for start in range(0, len(crops), chunk_size):
        batch = torch.stack(crops[start : start + chunk_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch)
        decoded.extend(decode_ctc_with_score(row.float()) for row in logits)
    return decoded


@torch.no_grad()
def collect(
    role, ctc, source_modules, records, collator, config, role_threshold,
    device, batch_size, bbox=None,
):
    vision, atom, bond, set_head, atom_threshold = source_modules
    results = []
    for start in range(0, len(records), batch_size):
        samples = records[start : start + batch_size]
        batch = collator(samples)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        _, atoms = frozen_atom_objects(
            vision, atom, bond, set_head, pixels, config, atom_threshold
        )
        mapping, _ = match_atom_targets(atoms, targets, config)
        atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            role_probabilities = role(
                pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices
            ).sigmoid()
        text_rows = torch.where(role_probabilities >= role_threshold)[0]
        active_modes = (*MODES, ORACLE_MODE) + ((LEARNED_MODE,) if bbox is not None else ())
        deployment_modes = (*MODES,) + ((LEARNED_MODE,) if bbox is not None else ())
        crop_keys = []
        crops_by_mode = {mode: [] for mode in active_modes}
        learned_lookup = {}
        if bbox is not None and len(text_rows):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                learned = bbox(
                    pixels,
                    atoms["atom_geometry"][..., :2],
                    atom_indices[text_rows],
                    sample_indices[text_rows],
                )
            for index, row in enumerate(text_rows.tolist()):
                key = (int(sample_indices[row]), int(atom_indices[row]))
                learned_lookup[key] = (
                    learned["centers"][index].float().cpu() * 768,
                    learned["half_extents"][index].float().cpu() * 768,
                )
        prepared = [
            sample["image"].convert("RGB").resize(
                (768, 768), Image.Resampling.BICUBIC
            )
            for sample in samples
        ]
        for row in text_rows.tolist():
            batch_index = int(sample_indices[row])
            atom_index = int(atom_indices[row])
            x, y = (
                atoms["atom_geometry"][batch_index, atom_index, :2].float().cpu() * 768
            ).tolist()
            extent_x, extent_y = (
                atoms["atom_geometry"][batch_index, atom_index, 6:8].float().cpu() * 768
            ).tolist()
            bbox_x, bbox_y = (
                atoms["atom_geometry"][batch_index, atom_index, 4:6].float().cpu() * 768
            ).tolist()
            crop_keys.append((batch_index, atom_index))
            for radius in RADII:
                crops_by_mode[f"radius_{radius}"].append(
                    crop_tensor(
                        prepared[batch_index],
                        (x - radius, y - radius, x + radius, y + radius),
                    )
                )
            for scale in EXTENT_SCALES:
                half_width = max(extent_x * scale, 1.0)
                half_height = max(extent_y * scale, 1.0)
                crops_by_mode[f"extent_{scale:.2f}"].append(
                    crop_tensor(
                        prepared[batch_index],
                        (x - half_width, y - half_height, x + half_width, y + half_height),
                    )
                )
                crops_by_mode[f"bbox_extent_{scale:.2f}"].append(
                    crop_tensor(
                        prepared[batch_index],
                        (
                            bbox_x - half_width,
                            bbox_y - half_height,
                            bbox_x + half_width,
                            bbox_y + half_height,
                        ),
                    )
                )
            target_index = int(mapping[batch_index, atom_index])
            if target_index >= 0 and bool(
                targets["atom_text_mask"][batch_index, target_index]
            ):
                center_x, center_y, width, height = (
                    targets["atom_text_bbox"][batch_index, target_index].float().cpu()
                    * 768
                ).tolist()
                oracle_bbox = (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                )
            else:
                oracle_bbox = (
                    x - max(extent_x * 1.5, 1.0),
                    y - max(extent_y * 1.5, 1.0),
                    x + max(extent_x * 1.5, 1.0),
                    y + max(extent_y * 1.5, 1.0),
                )
            crops_by_mode[ORACLE_MODE].append(
                crop_tensor(prepared[batch_index], oracle_bbox)
            )
            if bbox is not None:
                learned_center, learned_extent = learned_lookup[(batch_index, atom_index)]
                learned_x, learned_y = learned_center.tolist()
                learned_width, learned_height = learned_extent.tolist()
                crops_by_mode[LEARNED_MODE].append(
                    crop_tensor(
                        prepared[batch_index],
                        (
                            learned_x - learned_width,
                            learned_y - learned_height,
                            learned_x + learned_width,
                            learned_y + learned_height,
                        ),
                    )
                )
        decoded = {
            mode: decode_crops(ctc, crops, device) if crops else []
            for mode, crops in crops_by_mode.items()
        }
        text_lookup = {
            key: {mode: decoded[mode][index] for mode in active_modes}
            for index, key in enumerate(crop_keys)
        }
        for batch_index, sample in enumerate(samples):
            selected = torch.where(atoms["atom_mask"][batch_index])[0]
            mapped = mapping[batch_index, selected]
            valid = mapped >= 0
            selected = selected[valid]
            mapped = mapped[valid]
            labels = chemscene_atom_labels(batch["scenes"][batch_index])
            expected = [labels[int(target)][0] for target in mapped.tolist()]
            predicted = {mode: [] for mode in active_modes}
            predicted["best_score"] = []
            for atom_index in selected.tolist():
                candidates = text_lookup.get((batch_index, atom_index))
                if candidates is None:
                    for name in predicted:
                        predicted[name].append(("", 0.0))
                    continue
                for mode in active_modes:
                    predicted[mode].append(candidates[mode])
                predicted["best_score"].append(
                    max(
                        (candidates[mode] for mode in deployment_modes),
                        key=lambda item: item[1],
                    )
                )
            mapped_values = mapping[batch_index]
            results.append(
                {
                    "expected": expected,
                    "predicted": predicted,
                    "target_atoms": int(targets["atom_mask"][batch_index].sum()),
                    "selected_atoms": int(atoms["atom_mask"][batch_index].sum()),
                    "matched_atoms": int(mapped_values[mapped_values >= 0].unique().numel()),
                }
            )
    return results


def summarize(records: list[dict], mode: str) -> dict:
    totals = defaultdict(int)
    for sample in records:
        expected = sample["expected"]
        predicted = [item[0] for item in sample["predicted"][mode]]
        raw = [left == right for left, right in zip(expected, predicted)]
        states = [atom_state(left) == atom_state(right) for left, right in zip(expected, predicted)]
        text = [bool(value) for value in expected]
        totals["samples"] += 1
        totals["atoms"] += len(expected)
        totals["raw_correct"] += sum(raw)
        totals["state_correct"] += sum(states)
        totals["text_atoms"] += sum(text)
        totals["text_raw_correct"] += sum(ok and is_text for ok, is_text in zip(raw, text))
        representable = (
            sample["target_atoms"] == sample["matched_atoms"] == sample["selected_atoms"]
        )
        totals["representable"] += int(representable)
        totals["raw_exact"] += int(representable and all(raw))
        totals["state_exact"] += int(representable and all(states))
    samples = max(totals["samples"], 1)
    return {
        **dict(totals),
        "mode": mode,
        "raw_accuracy": totals["raw_correct"] / max(totals["atoms"], 1),
        "state_accuracy": totals["state_correct"] / max(totals["atoms"], 1),
        "text_raw_accuracy": totals["text_raw_correct"] / max(totals["text_atoms"], 1),
        "raw_exact_rate": totals["raw_exact"] / samples,
        "state_exact_rate": totals["state_exact"] / samples,
        "representable_rate": totals["representable"] / samples,
    }


def deployment_gate(result: dict, *, learned_bbox: bool) -> dict:
    criteria = {
        "state_accuracy": 0.95,
        "state_exact_rate": 0.15 if learned_bbox else 0.20,
    }
    if learned_bbox:
        criteria["text_raw_accuracy"] = 0.78
    return {
        "kind": "bbox_component" if learned_bbox else "combined_ocr",
        "criteria": criteria,
        "passed": all(result[name] >= threshold for name, threshold in criteria.items()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-checkpoint", default=DEFAULT_ROLE)
    parser.add_argument("--ctc-checkpoint", default=DEFAULT_CTC)
    parser.add_argument("--bbox-checkpoint")
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--evaluation-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OCR calibration requires CUDA")
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    role, role_payload = load_role(args.role_checkpoint, device)
    ctc, _ = load_ctc(args.ctc_checkpoint, device)
    bbox = load_bbox(args.bbox_checkpoint, device)[0] if args.bbox_checkpoint else None
    training_args = role_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        training_args["source_checkpoint"], training_args["grounded_graph_checkpoint"], device
    )
    config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds)
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    role_threshold = float(role_payload["validation"]["threshold"])
    records, audit = load_stratified_fresh_records(args.calibration_samples + args.evaluation_samples)
    modules = (vision, atom, bond, set_head, atom_threshold)
    calibration = collect(
        role, ctc, modules, records[: args.calibration_samples], collator, config,
        role_threshold, device, args.batch_size, bbox
    )
    modes = list(MODES) + ([LEARNED_MODE] if bbox is not None else []) + ["best_score"]
    calibration_metrics = [summarize(calibration, mode) for mode in modes]
    oracle_metrics = summarize(calibration, ORACLE_MODE)
    selected = max(
        calibration_metrics,
        key=lambda item: (item["state_exact_rate"], item["raw_exact_rate"], item["state_accuracy"]),
    )
    (output / "calibration.json").write_text(
        json.dumps(
            {
                "metrics": calibration_metrics,
                "selected": selected,
                "oracle_bbox": oracle_metrics,
                "audit": audit,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"calibration": selected, "oracle_bbox": oracle_metrics}), flush=True)
    evaluation = collect(
        role, ctc, modules, records[args.calibration_samples :], collator, config,
        role_threshold, device, args.batch_size, bbox
    )
    result = summarize(evaluation, selected["mode"])
    report = {
        "role_checkpoint": args.role_checkpoint,
        "ctc_checkpoint": args.ctc_checkpoint,
        "bbox_checkpoint": args.bbox_checkpoint,
        "role_threshold": role_threshold,
        "selected_mode": selected["mode"],
        "evaluation": result,
        "gate": deployment_gate(result, learned_bbox=bbox is not None),
        "data_gate": verify_data_gate(),
        "sealed_test_read": False,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"evaluation": report}), flush=True)


if __name__ == "__main__":
    main()
