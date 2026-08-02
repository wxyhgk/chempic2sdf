#!/usr/bin/env python3
"""Evaluate free-running raster-to-primitive generation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
from itertools import permutations
import json
import math
from pathlib import Path

import numpy as np
import cairosvg
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_sequence_v0_1 import (
    ParsedVisualSVGSequenceV01,
    parse_visual_svg_v0_1,
)
from Data.sdf.visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
from Data.sdf.visual_primitives_v0_1 import primitives_to_bond_svg
from chemvector.generation_visual_svg_v0_1 import generate_visual_svg_v0_1
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.visual_prefix_model import (
    VisualPrefixChemVectorModel,
    VisualPrefixMemory,
)
from chemvector.visual_svg_data import VisualSVGCollator, VisualSVGWebDataset
from chemvector.visual_svg_losses import coordinate_expectation_metrics


def image_tensor(image: Image.Image, size: int = 768) -> torch.Tensor:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def line_distance(left, right) -> float:
    direct = max(
        math.hypot(left.x0 - right.x0, left.y0 - right.y0),
        math.hypot(left.x1 - right.x1, left.y1 - right.y1),
    )
    reversed_distance = max(
        math.hypot(left.x0 - right.x1, left.y0 - right.y1),
        math.hypot(left.x1 - right.x0, left.y1 - right.y0),
    )
    return min(direct, reversed_distance)


def optimal_matches(cost_matrix: np.ndarray, threshold: float) -> int:
    if not cost_matrix.size:
        return 0
    rows, columns = linear_sum_assignment(cost_matrix)
    return sum(float(cost_matrix[row, column]) <= threshold for row, column in zip(rows, columns))


def line_true_positives(
    predicted: ParsedVisualSVGSequenceV01,
    expected: ParsedVisualSVGSequenceV01,
    tolerance: float,
) -> int:
    costs = np.asarray(
        [
            [line_distance(left, right) for right in expected.lines]
            for left in predicted.lines
        ],
        dtype=np.float32,
    )
    return optimal_matches(costs, tolerance)


def box_iou(left, right) -> float:
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = (left.x1 - left.x0) * (left.y1 - left.y0)
    right_area = (right.x1 - right.x0) * (right.y1 - right.y0)
    return intersection / max(left_area + right_area - intersection, 1)


def text_box_true_positives(
    predicted: ParsedVisualSVGSequenceV01,
    expected: ParsedVisualSVGSequenceV01,
    threshold: float,
) -> int:
    costs = np.asarray(
        [
            [1.0 - box_iou(left, right) for right in expected.text_boxes]
            for left in predicted.text_boxes
        ],
        dtype=np.float32,
    )
    return optimal_matches(costs, 1.0 - threshold)


def polygon_points(record) -> tuple[tuple[int, int], ...]:
    values = [
        getattr(record, name)
        for name in record.__dataclass_fields__
        if name != "stroke_width"
    ]
    return tuple(zip(values[::2], values[1::2]))


def polygon_distance(left, right) -> float:
    left_points = polygon_points(left)
    right_points = polygon_points(right)
    if len(left_points) != len(right_points):
        return float("inf")
    return max(
        math.hypot(left_x - right_x, left_y - right_y)
        for (left_x, left_y), (right_x, right_y) in zip(left_points, right_points)
    )


def polygon_true_positives(predicted, expected, tolerance: float) -> int:
    matches = 0
    for predicted_records, expected_records in (
        (predicted.triangles, expected.triangles),
        (predicted.quads, expected.quads),
    ):
        costs = np.asarray(
            [
                [polygon_distance(left, right) for right in expected_records]
                for left in predicted_records
            ],
            dtype=np.float32,
        )
        matches += optimal_matches(costs, tolerance)
    return matches


def render_bond_mask(parsed: ParsedVisualSVGSequenceV01, size: int = 768) -> np.ndarray:
    primitives = []
    for record in parsed.lines:
        primitives.append(
            {
                "kind": "line",
                "quantized_points": [[record.x0, record.y0], [record.x1, record.y1]],
                "stroke_width_q": record.stroke_width,
            }
        )
    for records in (parsed.triangles, parsed.quads):
        for record in records:
            primitives.append(
                {
                    "kind": "triangle" if len(polygon_points(record)) == 3 else "quad",
                    "quantized_points": [list(point) for point in polygon_points(record)],
                    "stroke_width_q": record.stroke_width,
                }
            )
    payload = {
        "schema_version": "visual-primitives-0.1",
        "width": size,
        "height": size,
        "primitives": primitives,
    }
    rendered = cairosvg.svg2png(
        bytestring=primitives_to_bond_svg(payload),
        output_width=size,
        output_height=size,
    )
    with Image.open(io.BytesIO(rendered)) as image:
        return np.asarray(image.convert("RGBA").getchannel("A")) > 0


def bond_iou(parsed: ParsedVisualSVGSequenceV01, target: Image.Image) -> float:
    predicted = render_bond_mask(parsed, target.width)
    expected = np.asarray(target.convert("L")) > 0
    return float((predicted & expected).sum() / max(int((predicted | expected).sum()), 1))


def content_free_visual_memory(
    model: VisualPrefixChemVectorModel,
    reference: VisualPrefixMemory,
) -> VisualPrefixMemory:
    """Keep visual slot identity while removing CNN and raw-ink content."""
    positions = torch.arange(model.visual_prefix_tokens, device=reference.prefix.device)
    layout = (
        model.visual_position_embedding(positions).unsqueeze(0)
        + model.visual_modality_embedding.view(1, 1, -1)
    )
    return VisualPrefixMemory(
        prefix=layout.expand(reference.prefix.size(0), -1, -1),
        values=torch.zeros_like(reference.values),
    )


def different_molecule_permutation(
    molecule_ids: list[str],
) -> tuple[int, ...] | None:
    """Return the first deterministic assignment with no same-molecule pairs."""
    if not molecule_ids or any(not molecule_id for molecule_id in molecule_ids):
        raise ValueError("Every evaluation sample must provide molecule_id")
    for candidate in permutations(range(len(molecule_ids))):
        if all(
            molecule_ids[source_index] != molecule_ids[target_index]
            for target_index, source_index in enumerate(candidate)
        ):
            return candidate
    return None


@torch.no_grad()
def teacher_metrics(
    model: torch.nn.Module,
    shard: str | list[str],
    tokenizer: VisualSVGTokenizerV01,
    device: torch.device,
    max_length: int,
    limit: int,
) -> dict[str, float]:
    loader = DataLoader(
        VisualSVGWebDataset(shard, limit=limit),
        batch_size=4,
        num_workers=0,
        collate_fn=VisualSVGCollator(tokenizer, max_length=max_length),
    )
    totals = Counter()
    model.eval()
    for batch in loader:
        pixels = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mismatch_permutation = different_molecule_permutation(
            [str(meta.get("molecule_id", "")) for meta in batch["meta"]]
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            zero_grounded_values = None
            zero_content = None
            white_image = None
            if isinstance(model, VisualPrefixChemVectorModel):
                memory = model.encode_images(pixels)
                output = model(pixels, ids, mask, memory=memory)
                zero_grounded_values = model(
                    pixels,
                    ids,
                    mask,
                    memory=VisualPrefixMemory(
                        prefix=memory.prefix,
                        values=torch.zeros_like(memory.values),
                    ),
                )
                zero_content = model(
                    pixels,
                    ids,
                    mask,
                    memory=content_free_visual_memory(model, memory),
                )
                white_image = model(torch.ones_like(pixels), ids, mask)
            else:
                output = model(pixels, ids, mask)
            mismatched = None
            if mismatch_permutation is not None:
                indices = torch.tensor(mismatch_permutation, device=pixels.device)
                mismatched = model(pixels.index_select(0, indices), ids, mask)
        labels = ids[:, 1:]
        predictions = output["logits"].argmax(dim=-1)
        coordinate_metrics = coordinate_expectation_metrics(
            output["logits"],
            labels,
            tokenizer,
        )
        valid = labels != model.config.pad_token_id
        exact = ((predictions == labels) | ~valid).all(dim=1)
        totals["loss"] += float(output["loss"])
        if mismatched is not None:
            totals["mismatch_loss"] += float(mismatched["loss"])
            totals["mismatch_batches"] += 1
            totals["mismatch_samples"] += pixels.size(0)
        else:
            totals["skipped_mismatch_samples"] += pixels.size(0)
        if zero_grounded_values is not None:
            totals["zero_grounded_values_loss"] += float(
                zero_grounded_values["loss"]
            )
            totals["zero_grounded_values_logits_mae"] += float(
                (output["logits"] - zero_grounded_values["logits"]).abs().mean()
            )
            totals["zero_grounded_values_batches"] += 1
        if zero_content is not None:
            totals["zero_content_loss"] += float(zero_content["loss"])
            totals["zero_content_logits_mae"] += float(
                (output["logits"] - zero_content["logits"]).abs().mean()
            )
            totals["zero_content_batches"] += 1
        if white_image is not None:
            totals["white_image_loss"] += float(white_image["loss"])
            totals["white_image_logits_mae"] += float(
                (output["logits"] - white_image["logits"]).abs().mean()
            )
            totals["white_image_batches"] += 1
        totals["batches"] += 1
        totals["correct"] += int(((predictions == labels) & valid).sum())
        totals["tokens"] += int(valid.sum())
        totals["exact"] += int(exact.sum())
        totals["samples"] += int(exact.numel())
        coordinate_tokens = int(coordinate_metrics["tokens"])
        totals["coordinate_tokens"] += coordinate_tokens
        for source, target in (
            ("loss", "coordinate_loss_sum"),
            ("soft_mae_q", "coordinate_soft_mae_q_sum"),
            ("hard_exact", "coordinate_hard_exact_sum"),
            ("hard_valid_rate", "coordinate_hard_valid_sum"),
        ):
            totals[target] += float(coordinate_metrics[source]) * coordinate_tokens
    matched = totals["loss"] / max(totals["batches"], 1)
    mismatched = totals["mismatch_loss"] / max(totals["mismatch_batches"], 1)
    zero_grounded_values = totals["zero_grounded_values_loss"] / max(
        totals["zero_grounded_values_batches"], 1
    )
    zero_content = totals["zero_content_loss"] / max(
        totals["zero_content_batches"], 1
    )
    white_image = totals["white_image_loss"] / max(
        totals["white_image_batches"], 1
    )
    return {
        "loss": matched,
        "token_accuracy": totals["correct"] / max(totals["tokens"], 1),
        "sequence_exact": totals["exact"] / max(totals["samples"], 1),
        "coordinate_loss": totals["coordinate_loss_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_soft_mae_q": totals["coordinate_soft_mae_q_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_hard_exact": totals["coordinate_hard_exact_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_hard_valid_rate": totals["coordinate_hard_valid_sum"]
        / max(totals["coordinate_tokens"], 1),
        "coordinate_tokens": totals["coordinate_tokens"],
        "mismatch_loss": mismatched,
        "mismatch_loss_gap": mismatched - matched,
        "mismatch_loss_ratio": mismatched / max(matched, 1e-8),
        "mismatch_samples": totals["mismatch_samples"],
        "skipped_mismatch_samples": totals["skipped_mismatch_samples"],
        "zero_grounded_values_loss": zero_grounded_values,
        "zero_grounded_values_loss_ratio": zero_grounded_values
        / max(matched, 1e-8),
        "zero_grounded_values_logits_mae": totals[
            "zero_grounded_values_logits_mae"
        ]
        / max(totals["zero_grounded_values_batches"], 1),
        "zero_content_loss": zero_content,
        "zero_content_loss_ratio": zero_content / max(matched, 1e-8),
        "zero_content_logits_mae": totals["zero_content_logits_mae"]
        / max(totals["zero_content_batches"], 1),
        "white_image_loss": white_image,
        "white_image_loss_ratio": white_image / max(matched, 1e-8),
        "white_image_logits_mae": totals["white_image_logits_mae"]
        / max(totals["white_image_batches"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/chemvector-v090-visual-svg-capacity-100-stage3/last.pt",
    )
    parser.add_argument(
        "--shard",
        nargs="+",
        default=["Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visual-svg-capacity-100-eval",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--line-tolerance", type=float, default=8.0)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--max-generation-length", type=int)
    parser.add_argument("--teacher-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Visual SVG evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("sequence_schema_version") != "visual-svg-0.1":
        raise ValueError("Checkpoint does not use Visual SVG v0.1")
    config = ChemVectorConfig(**checkpoint["config"])
    model_type = checkpoint.get(
        "model_type",
        "chemvector-cross-attention-v0.1",
    )
    if model_type == VisualPrefixChemVectorModel.model_type:
        visual_grid_size = checkpoint.get("visual_grid_size")
        if not isinstance(visual_grid_size, int):
            raise ValueError("Visual-prefix checkpoint is missing visual_grid_size")
        model = VisualPrefixChemVectorModel(
            config,
            visual_grid_size=visual_grid_size,
            ink_subgrid=checkpoint.get("ink_subgrid", 2),
        )
    elif model_type == "chemvector-cross-attention-v0.1":
        model = ChemVectorModel(config)
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device).eval()
    tokenizer_path = Path(args.checkpoint).parent / "tokenizer.json"
    tokenizer = VisualSVGTokenizerV01.load(tokenizer_path)
    vocabulary_hash = hashlib.sha256(
        json.dumps(
            tokenizer.vocabulary,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if checkpoint.get("tokenizer_sha256") != vocabulary_hash:
        raise ValueError("Checkpoint tokenizer hash does not match tokenizer.json")
    maximum = args.max_generation_length or config.max_target_length
    if not 1 <= maximum <= config.max_target_length:
        raise ValueError("Generation length must be within the model context")
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    teacher = teacher_metrics(model, args.shard, tokenizer, device, maximum, args.limit)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.teacher_only:
        metrics = {"schema_version": "visual-svg-evaluation-0.1", "teacher": teacher}
        (output / "generation_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2))
        return

    totals = Counter()
    ious = []
    conditional_ious = []
    rows = []
    dataset = VisualSVGWebDataset(args.shard, limit=args.limit)
    for index, sample in enumerate(dataset, 1):
        if index > args.limit:
            break
        pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
        result = generate_visual_svg_v0_1(
            model,
            pixels,
            tokenizer,
            max_length=maximum,
        )
        expected = parse_visual_svg_v0_1(sample["target_sequence"])
        predicted = None
        error = result.error
        if result.sequence is not None:
            try:
                predicted = parse_visual_svg_v0_1(result.sequence)
            except Exception as parse_error:
                error = f"{type(parse_error).__name__}: {parse_error}"
        totals["samples"] += 1
        totals["completed"] += int(result.completed)
        totals["grammar_valid"] += int(result.grammar_valid and predicted is not None)
        totals["line_expected"] += len(expected.lines)
        totals["text_expected"] += len(expected.text_boxes)
        totals["polygon_expected"] += len(expected.triangles) + len(expected.quads)
        row = {
            "sample_key": sample["key"],
            "completed": result.completed,
            "grammar_valid": result.grammar_valid and predicted is not None,
            "expected_sequence": sample["target_sequence"],
            "predicted_sequence": result.sequence,
            "error": error,
        }
        visual_iou = 0.0
        if predicted is not None:
            totals["valid_samples"] += 1
            totals["line_expected_valid"] += len(expected.lines)
            totals["text_expected_valid"] += len(expected.text_boxes)
            totals["polygon_expected_valid"] += len(expected.triangles) + len(expected.quads)
            exact = predicted == expected
            totals["primitive_exact"] += int(exact)
            totals["line_count_exact"] += int(len(predicted.lines) == len(expected.lines))
            totals["structural_count_exact"] += int(
                len(predicted.lines) + len(predicted.triangles) + len(predicted.quads)
                == len(expected.lines) + len(expected.triangles) + len(expected.quads)
            )
            totals["triangle_count_exact"] += int(
                len(predicted.triangles) == len(expected.triangles)
            )
            totals["quad_count_exact"] += int(len(predicted.quads) == len(expected.quads))
            totals["text_count_exact"] += int(
                len(predicted.text_boxes) == len(expected.text_boxes)
            )
            line_tp = line_true_positives(predicted, expected, args.line_tolerance)
            text_tp = text_box_true_positives(predicted, expected, args.text_box_iou)
            polygon_tp = polygon_true_positives(
                predicted,
                expected,
                args.line_tolerance,
            )
            totals["line_tp"] += line_tp
            totals["line_predicted"] += len(predicted.lines)
            totals["text_tp"] += text_tp
            totals["text_predicted"] += len(predicted.text_boxes)
            totals["polygon_tp"] += polygon_tp
            totals["polygon_predicted"] += len(predicted.triangles) + len(predicted.quads)
            visual_iou = bond_iou(predicted, sample["bond_layer"])
            conditional_ious.append(visual_iou)
            row.update(
                {
                    "primitive_exact": exact,
                    "expected_counts": {
                        "line": len(expected.lines),
                        "triangle": len(expected.triangles),
                        "quad": len(expected.quads),
                        "text_box": len(expected.text_boxes),
                    },
                    "predicted_counts": {
                        "line": len(predicted.lines),
                        "triangle": len(predicted.triangles),
                        "quad": len(predicted.quads),
                        "text_box": len(predicted.text_boxes),
                    },
                    "line_true_positives": line_tp,
                    "text_box_true_positives": text_tp,
                    "polygon_true_positives": polygon_tp,
                    "bond_iou": visual_iou,
                }
            )
        ious.append(visual_iou)
        rows.append(row)
        if index % 5 == 0:
            print(f"generated={index}/{args.limit}", flush=True)

    samples = max(totals["samples"], 1)
    line_precision = totals["line_tp"] / max(totals["line_predicted"], 1)
    line_recall = totals["line_tp"] / max(totals["line_expected"], 1)
    text_precision = totals["text_tp"] / max(totals["text_predicted"], 1)
    text_recall = totals["text_tp"] / max(totals["text_expected"], 1)
    polygon_precision = totals["polygon_tp"] / max(totals["polygon_predicted"], 1)
    polygon_recall = totals["polygon_tp"] / max(totals["polygon_expected"], 1)
    line_recall_valid = totals["line_tp"] / max(totals["line_expected_valid"], 1)
    text_recall_valid = totals["text_tp"] / max(totals["text_expected_valid"], 1)
    polygon_recall_valid = totals["polygon_tp"] / max(
        totals["polygon_expected_valid"], 1
    )
    metrics = {
        "schema_version": "visual-svg-evaluation-0.1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_type": model_type,
        "samples": totals["samples"],
        "teacher": teacher,
        "free_generation": {
            "completed": totals["completed"] / samples,
            "grammar_valid": totals["grammar_valid"] / samples,
            "primitive_exact": totals["primitive_exact"] / samples,
            "line_count_exact": totals["line_count_exact"] / samples,
            "structural_primitive_count_exact": totals["structural_count_exact"] / samples,
            "triangle_count_exact": totals["triangle_count_exact"] / samples,
            "quad_count_exact": totals["quad_count_exact"] / samples,
            "text_box_count_exact": totals["text_count_exact"] / samples,
            "line_precision": line_precision,
            "line_recall": line_recall,
            "line_f1": 2 * line_precision * line_recall / max(line_precision + line_recall, 1e-8),
            "text_box_precision": text_precision,
            "text_box_recall": text_recall,
            "text_box_f1": 2 * text_precision * text_recall / max(text_precision + text_recall, 1e-8),
            "polygon_precision": polygon_precision,
            "polygon_recall": polygon_recall,
            "polygon_f1": 2 * polygon_precision * polygon_recall / max(polygon_precision + polygon_recall, 1e-8),
            "bond_iou_mean": sum(ious) / max(len(ious), 1),
            "bond_iou_min": min(ious) if ious else 0.0,
            "conditional_on_valid": {
                "samples": totals["valid_samples"],
                "line_recall": line_recall_valid,
                "line_f1": 2 * line_precision * line_recall_valid / max(line_precision + line_recall_valid, 1e-8),
                "text_box_recall": text_recall_valid,
                "text_box_f1": 2 * text_precision * text_recall_valid / max(text_precision + text_recall_valid, 1e-8),
                "polygon_recall": polygon_recall_valid,
                "polygon_f1": 2 * polygon_precision * polygon_recall_valid / max(polygon_precision + polygon_recall_valid, 1e-8),
                "bond_iou_mean": sum(conditional_ious) / max(len(conditional_ious), 1),
                "bond_iou_min": min(conditional_ious) if conditional_ious else 0.0,
            },
        },
    }
    (output / "generation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
