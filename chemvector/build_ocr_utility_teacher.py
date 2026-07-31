#!/usr/bin/env python3
"""Build internal teacher rows for conservative old/new OCR routing."""

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
from chemvector.evaluate_direct_atom_ocr import load_ctc
from chemvector.image_io import image_tensor
from chemvector.infer_chemscene_hybrid import ChemSceneHybridPipeline
from chemvector.layer_ocr import colorize_visible_text
from chemvector.ocr_utility_router import ocr_utility_features
from chemvector.pretrain_ctc_ocr import stable_seed
from chemvector.text_object import decode_text_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--bond-checkpoint", required=True)
    parser.add_argument("--text-bbox-checkpoint", required=True)
    parser.add_argument("--old-ocr-checkpoint", required=True)
    parser.add_argument("--new-ocr-checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--color-probability", type=float, default=0.5)
    parser.add_argument("--anchor-fallback-margin", type=float, default=0.25)
    parser.add_argument("--tolerance-pixels", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def expand_shards(patterns: list[str]) -> list[str]:
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No OCR teacher shards matched: {pattern}")
        result.extend(matches)
    return result


def match_expected_text(
    points: torch.Tensor,
    roles: torch.Tensor,
    atoms: list[dict],
    *,
    image_size: int,
    tolerance_pixels: float,
) -> dict[int, str]:
    expected = [atom for atom in atoms if atom.get("visible_text")]
    candidates = []
    for point_index in torch.where(roles == TEXT_ROLE)[0].tolist():
        for atom_index, atom in enumerate(expected):
            center = torch.tensor(atom["center"], dtype=points.dtype) / image_size
            distance = float(torch.linalg.vector_norm(points[point_index].cpu() - center)) * image_size
            if distance <= tolerance_pixels:
                candidates.append((distance, point_index, atom_index))
    assigned_points = set()
    assigned_atoms = set()
    matches = {}
    for _, point_index, atom_index in sorted(candidates):
        if point_index in assigned_points or atom_index in assigned_atoms:
            continue
        assigned_points.add(point_index)
        assigned_atoms.add(atom_index)
        matches[point_index] = str(expected[atom_index]["text"])
    return matches


@torch.inference_mode()
def decode_with_fallback(model, shifted, anchor, margin: float):
    if not len(shifted):
        return []
    combined = torch.cat((shifted, anchor), dim=0)
    with torch.autocast(device_type=combined.device.type, dtype=torch.bfloat16):
        logits = model(combined).float()
    shifted_logits, anchor_logits = logits.chunk(2)
    shifted_values = [decode_ctc_with_score(row) for row in shifted_logits]
    anchor_values = [decode_ctc_with_score(row) for row in anchor_logits]
    return [
        shifted_item
        if shifted_item[1] >= anchor_item[1] + margin
        else anchor_item
        for shifted_item, anchor_item in zip(shifted_values, anchor_values)
    ]


def main() -> None:
    args = parse_args()
    if min(args.samples, args.batch_size) < 1:
        raise ValueError("Teacher sample counts must be positive")
    device = torch.device(args.device)
    pipeline = ChemSceneHybridPipeline(
        Path(args.selector_checkpoint),
        Path(args.bond_checkpoint),
        device,
        ctc_ocr_checkpoint=Path(args.new_ocr_checkpoint),
        text_bbox_checkpoint=Path(args.text_bbox_checkpoint),
        ctc_anchor_fallback_margin=args.anchor_fallback_margin,
    )
    old_ocr, _ = load_ctc(args.old_ocr_checkpoint, device)
    dataset = ChemMolWebDataset(
        expand_shards(args.shards),
        validate_legacy_hash=False,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        sequence_version="0.4",
    )
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Teacher output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        "images": 0,
        "matched": 0,
        "disagreements": 0,
        "new_rescue": 0,
        "new_break": 0,
        "both_exact": 0,
        "both_wrong": 0,
    }
    rows = []

    def consume(current: list[tuple[dict, object]]) -> None:
        if not current:
            return
        pixels = torch.stack([image_tensor(image) for _, image in current]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = pipeline.base.vision_encoder.forward_stroke_resolution(pixels)
            text_output = pipeline.text_head(features)
        text_batches = decode_text_objects(
            text_output,
            threshold=float(pipeline.protocol["text_threshold"]),
            maximum_objects=pipeline.selector_config.maximum_candidates,
        )
        junctions = pipeline._junctions_batch(
            pixels,
            features,
            real_threshold=0.2,
            bbox_expansion_pixels=8.0,
            include_ocr=False,
        )
        for batch_index, ((sample, _), (points, roles, _), text_objects) in enumerate(
            zip(current, junctions, text_batches)
        ):
            shifted, point_indices = detected_text_crops(
                pixels[batch_index], points, roles, text_objects, text_role=TEXT_ROLE
            )
            anchor_objects = [
                {key: value for key, value in item.items() if key not in ("bbox_x", "bbox_y")}
                for item in text_objects
            ]
            anchor, anchor_indices = detected_text_crops(
                pixels[batch_index], points, roles, anchor_objects, text_role=TEXT_ROLE
            )
            if point_indices != anchor_indices:
                raise ValueError("Shifted and anchor OCR assignments differ")
            old_values = decode_with_fallback(
                old_ocr, shifted, anchor, args.anchor_fallback_margin
            )
            new_values = decode_with_fallback(
                pipeline.ctc_ocr, shifted, anchor, args.anchor_fallback_margin
            )
            expected = match_expected_text(
                points,
                roles,
                sample["layers"]["atoms"],
                image_size=pixels.size(-1),
                tolerance_pixels=args.tolerance_pixels,
            )
            for point_index, old_value, new_value in zip(
                point_indices, old_values, new_values
            ):
                target = expected.get(point_index)
                if target is None:
                    continue
                old_text, old_score = old_value
                new_text, new_score = new_value
                old_exact = old_text == target
                new_exact = new_text == target
                label = (
                    "new_rescue"
                    if new_exact and not old_exact
                    else "new_break"
                    if old_exact and not new_exact
                    else "both_exact"
                    if old_exact and new_exact
                    else "both_wrong"
                )
                totals["matched"] += 1
                totals[label] += 1
                totals["disagreements"] += int(old_text != new_text)
                rows.append(
                    {
                        "sample_key": sample["key"],
                        "point_index": point_index,
                        "domain": args.domain,
                        "expected": target,
                        "old_text": old_text,
                        "new_text": new_text,
                        "old_score": old_score,
                        "new_score": new_score,
                        "label": label,
                        "features": ocr_utility_features(
                            old_text, old_score, new_text, new_score
                        ),
                    }
                )
            totals["images"] += 1
        if totals["images"] % 250 < len(current):
            print(json.dumps(totals), flush=True)

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
        if len(batch) == args.batch_size:
            consume(batch)
            batch = []
    consume(batch)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {**totals, "rows": len(rows), "domain": args.domain}
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
