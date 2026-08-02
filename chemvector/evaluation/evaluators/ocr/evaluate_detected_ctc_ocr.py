#!/usr/bin/env python3
"""Evaluate bbox CTC OCR with deployed text detection and vertex selection."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import random

import torch

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.ctc_ocr import decode_ctc_with_score, detected_text_crops
from chemvector.dual_vertex_selector import TEXT_ROLE
from chemvector.image_io import image_tensor
from chemvector.infer_chemscene_hybrid import ChemSceneHybridPipeline
from chemvector.layer_ocr import colorize_visible_text
from chemvector.pretrain_ctc_ocr import is_modifier_label, stable_seed
from chemvector.text_object import decode_text_objects


def match_text_predictions(
    points: torch.Tensor,
    roles: torch.Tensor,
    labels: list[str],
    atoms: list[dict],
    *,
    image_size: int,
    tolerance_pixels: float,
) -> list[tuple[str, str]]:
    expected = [atom for atom in atoms if atom.get("visible_text")]
    candidates = []
    for point_index, (point, role) in enumerate(zip(points, roles)):
        if int(role) != TEXT_ROLE:
            continue
        for atom_index, atom in enumerate(expected):
            center = torch.tensor(atom["center"], dtype=point.dtype) / image_size
            distance = float(torch.linalg.vector_norm(point.cpu() - center)) * image_size
            if distance <= tolerance_pixels:
                candidates.append((distance, point_index, atom_index))
    matches = []
    assigned_points = set()
    assigned_atoms = set()
    for _, point_index, atom_index in sorted(candidates):
        if point_index in assigned_points or atom_index in assigned_atoms:
            continue
        assigned_points.add(point_index)
        assigned_atoms.add(atom_index)
        matches.append((str(expected[atom_index]["text"]), labels[point_index]))
    return matches


def expand_shards(patterns: list[str]) -> list[str]:
    shards = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No OCR evaluation shards matched: {pattern}")
        shards.extend(matches)
    return shards


@torch.inference_mode()
def predict_ctc_for_objects(
    pipeline: ChemSceneHybridPipeline,
    pixels: torch.Tensor,
    batches: list[list[dict[str, float]]],
    *,
    extent_scale: float = 1.0,
    crop_context: float = 4.0,
    anchor_fallback_margin: float | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    if pipeline.ctc_ocr is None:
        raise ValueError("Text-object OCR evaluation requires a CTC checkpoint")
    results = []
    for batch_index, objects in enumerate(batches):
        scaled_objects = [
            {
                **item,
                "half_width": float(item["half_width"]) * extent_scale,
                "half_height": float(item["half_height"]) * extent_scale,
            }
            for item in objects
        ]
        points = torch.tensor(
            [(item["x"], item["y"]) for item in scaled_objects],
            device=pipeline.device,
            dtype=pixels.dtype,
        ).reshape(-1, 2)
        roles = torch.full(
            (len(points),), TEXT_ROLE, device=pipeline.device, dtype=torch.long
        )
        crops, indices = detected_text_crops(
            pixels[batch_index],
            points,
            roles,
            scaled_objects,
            text_role=TEXT_ROLE,
            context=crop_context,
        )
        labels = [""] * len(points)
        if len(crops):
            with torch.autocast(device_type=pipeline.device.type, dtype=torch.bfloat16):
                logits = pipeline.ctc_ocr(crops)
            shifted = [decode_ctc_with_score(row.float()) for row in logits]
            if anchor_fallback_margin is not None:
                anchor_objects = [
                    {key: value for key, value in item.items() if key not in ("bbox_x", "bbox_y")}
                    for item in scaled_objects
                ]
                anchor_crops, anchor_indices = detected_text_crops(
                    pixels[batch_index],
                    points,
                    roles,
                    anchor_objects,
                    text_role=TEXT_ROLE,
                    context=crop_context,
                )
                if anchor_indices != indices:
                    raise ValueError("Anchor and shifted crop assignments differ")
                with torch.autocast(
                    device_type=pipeline.device.type, dtype=torch.bfloat16
                ):
                    anchor_logits = pipeline.ctc_ocr(anchor_crops)
                anchor = [
                    decode_ctc_with_score(row.float()) for row in anchor_logits
                ]
                shifted = [
                    shifted_item
                    if shifted_item[1] >= anchor_item[1] + anchor_fallback_margin
                    else anchor_item
                    for shifted_item, anchor_item in zip(shifted, anchor)
                ]
            for index, (text, _) in zip(indices, shifted):
                labels[index] = text
        results.append((points, roles, labels))
    return results


@torch.inference_mode()
def predict_from_text_objects(
    pipeline: ChemSceneHybridPipeline,
    pixels: torch.Tensor,
    *,
    extent_scale: float = 1.0,
    crop_context: float = 4.0,
    anchor_fallback_margin: float | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    with torch.autocast(device_type=pipeline.device.type, dtype=torch.bfloat16):
        features = pipeline.base.vision_encoder.forward_stroke_resolution(pixels)
        output = pipeline.text_head(features)
    batches = decode_text_objects(
        output,
        threshold=float(pipeline.protocol["text_threshold"]),
        maximum_objects=pipeline.selector_config.maximum_candidates,
    )
    return predict_ctc_for_objects(
        pipeline,
        pixels,
        batches,
        extent_scale=extent_scale,
        crop_context=crop_context,
        anchor_fallback_margin=anchor_fallback_margin,
    )


def oracle_text_objects(samples: list[tuple[dict, object]]) -> list[list[dict[str, float]]]:
    batches = []
    for sample, _ in samples:
        width = float(sample["layers"]["width"])
        height = float(sample["layers"]["height"])
        objects = []
        for atom in sample["layers"]["atoms"]:
            if not atom.get("visible_text") or not atom.get("bbox"):
                continue
            x0, y0, x1, y1 = (float(value) for value in atom["bbox"])
            objects.append(
                {
                    "x": float(atom["center"][0]) / width,
                    "y": float(atom["center"][1]) / height,
                    "half_width": (x1 - x0) / (2.0 * width),
                    "half_height": (y1 - y0) / (2.0 * height),
                }
            )
        batches.append(objects)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--bond-checkpoint", required=True)
    parser.add_argument("--graph-corrector-checkpoint")
    parser.add_argument("--ctc-ocr-checkpoint", required=True)
    parser.add_argument("--text-bbox-checkpoint")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--color-probability", type=float, default=0.5)
    parser.add_argument("--tolerance-pixels", type=float, default=8.0)
    parser.add_argument("--extent-scale", type=float, default=1.0)
    parser.add_argument("--crop-context", type=float, default=4.0)
    parser.add_argument("--anchor-fallback-margin", type=float)
    parser.add_argument(
        "--candidate-source",
        choices=("deployed_vertices", "text_objects", "oracle_text_objects"),
        default="deployed_vertices",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint),
        Path(args.bond_checkpoint),
        device,
        graph_corrector_checkpoint=(
            Path(args.graph_corrector_checkpoint)
            if args.graph_corrector_checkpoint
            else None
        ),
        ctc_ocr_checkpoint=Path(args.ctc_ocr_checkpoint),
        text_bbox_checkpoint=(
            Path(args.text_bbox_checkpoint) if args.text_bbox_checkpoint else None
        ),
        ctc_anchor_fallback_margin=args.anchor_fallback_margin,
    )
    dataset = ChemMolWebDataset(
        expand_shards(args.shards),
        validate_legacy_hash=False,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        sequence_version="0.4",
    )
    totals = {
        "images": 0,
        "visible_text": 0,
        "matched_text": 0,
        "ocr_exact": 0,
        "modifier_matched": 0,
        "modifier_exact": 0,
        "simple_matched": 0,
        "simple_exact": 0,
        "false_modifier": 0,
    }
    labels: dict[str, list[int]] = {}

    def consume(current: list[tuple[dict, object]]) -> None:
        if not current:
            return
        pixels = torch.stack([image_tensor(item[1]) for item in current]).to(device)
        if args.candidate_source == "oracle_text_objects":
            predictions = [
                (*prediction, None)
                for prediction in predict_ctc_for_objects(
                    pipeline,
                    pixels,
                    oracle_text_objects(current),
                    extent_scale=args.extent_scale,
                    crop_context=args.crop_context,
                    anchor_fallback_margin=args.anchor_fallback_margin,
                )
            ]
        elif args.candidate_source == "text_objects":
            predictions = [
                (*prediction, None)
                for prediction in predict_from_text_objects(
                    pipeline,
                    pixels,
                    extent_scale=args.extent_scale,
                    crop_context=args.crop_context,
                    anchor_fallback_margin=args.anchor_fallback_margin,
                )
            ]
        else:
            predictions = pipeline.predict_components_batch(pixels, include_ocr=True)
        for (sample, _), (points, roles, predicted_labels, _) in zip(
            current, predictions
        ):
            atoms = sample["layers"]["atoms"]
            visible = sum(bool(atom.get("visible_text")) for atom in atoms)
            matches = match_text_predictions(
                points,
                roles,
                predicted_labels,
                atoms,
                image_size=pixels.size(-1),
                tolerance_pixels=args.tolerance_pixels,
            )
            totals["images"] += 1
            totals["visible_text"] += visible
            totals["matched_text"] += len(matches)
            for expected, predicted in matches:
                exact = expected == predicted
                modifier = is_modifier_label(expected)
                totals["ocr_exact"] += int(exact)
                bucket = "modifier" if modifier else "simple"
                totals[f"{bucket}_matched"] += 1
                totals[f"{bucket}_exact"] += int(exact)
                totals["false_modifier"] += int(
                    not modifier and is_modifier_label(predicted)
                )
                values = labels.setdefault(expected, [0, 0])
                values[0] += int(exact)
                values[1] += 1

    batch = []
    for sample in dataset:
        if totals["images"] + len(batch) >= args.samples:
            consume(batch)
            batch = []
            break
        rng = random.Random(stable_seed(args.seed, sample["key"]))
        image = sample["image"]
        if rng.random() < args.color_probability:
            image = colorize_visible_text(image, sample["layers"]["atoms"])
        batch.append((sample, image))
        if len(batch) < args.batch_size:
            continue
        consume(batch)
        batch = []
    consume(batch)

    result = {
        **totals,
        "text_detection_recall": totals["matched_text"] / max(totals["visible_text"], 1),
        "matched_ocr_exact_rate": totals["ocr_exact"] / max(totals["matched_text"], 1),
        "end_to_end_visible_exact_rate": totals["ocr_exact"] / max(totals["visible_text"], 1),
        "modifier_exact_rate": totals["modifier_exact"] / max(totals["modifier_matched"], 1),
        "simple_exact_rate": totals["simple_exact"] / max(totals["simple_matched"], 1),
        "false_modifier_rate": totals["false_modifier"] / max(totals["simple_matched"], 1),
        "labels": {
            label: {"exact_rate": correct / count, "samples": count}
            for label, (correct, count) in sorted(labels.items())
        },
        "protocol": {
            "predicted_vertices": args.candidate_source == "deployed_vertices",
            "predicted_text_extents": args.candidate_source != "oracle_text_objects",
            "candidate_source": args.candidate_source,
            "color_probability": args.color_probability,
            "tolerance_pixels": args.tolerance_pixels,
            "extent_scale": args.extent_scale,
            "crop_context": args.crop_context,
            "anchor_fallback_margin": args.anchor_fallback_margin,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
