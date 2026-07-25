#!/usr/bin/env python3
"""Evaluate unordered raster-to-Visual-SVG primitive-set predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from Data.sdf.visual_svg_sequence_v0_1 import (
    COORDINATE_MAX,
    STROKE_WIDTH_MAX,
    STROKE_WIDTH_MIN,
    LineRecordV01,
    ParsedVisualSVGSequenceV01,
    QuadRecordV01,
    TextBoxRecordV01,
    TriangleRecordV01,
    VisualSVGSequenceError,
    parse_visual_svg_v0_1,
    serialize_parsed_visual_svg_v0_1,
    validate_parsed_visual_svg_v0_1,
)
from chemvector.evaluate_visual_svg import (
    bond_iou,
    line_true_positives,
    polygon_true_positives,
    text_box_true_positives,
)
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    LINE,
    NO_OBJECT,
    QUAD,
    TEXT_BOX,
    TRIANGLE,
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
    decode_primitive_set,
)
from chemvector.primitive_set_data import PrimitiveSetCollator
from chemvector.visual_svg_data import VisualSVGWebDataset


EVALUATION_SCHEMA_VERSION = "primitive-set-evaluation-0.1"
CLASS_NAMES = ("line", "triangle", "quad", "text_box")
DECODE_MODES = ("threshold", "count_head")
TYPE_TO_CLASS = {
    LINE: "line",
    TRIANGLE: "triangle",
    QUAD: "quad",
    TEXT_BOX: "text_box",
}


@dataclass(frozen=True, slots=True)
class DecodedRowsResult:
    """A decoded set plus every object that could not enter the renderable set."""

    parsed: ParsedVisualSVGSequenceV01
    sequence: str | None
    decoded_counts: dict[str, int]
    renderable_counts: dict[str, int]
    renderable_objects: tuple[dict[str, Any], ...]
    invalid_objects: tuple[dict[str, Any], ...]
    serialization_error: str | None

    @property
    def valid(self) -> bool:
        return not self.invalid_objects and self.serialization_error is None


class PrimitiveSetEvaluationCollator(PrimitiveSetCollator):
    """Retain sidecars needed by evaluation in addition to model inputs."""

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(samples)
        batch["target_sequences"] = [sample["target_sequence"] for sample in samples]
        batch["primitives"] = [sample["primitives"] for sample in samples]
        return batch


def _clamped_unit(value: Any, field: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} is not finite: {value!r}")
    return min(max(value, 0.0), 1.0)


def _coordinate(value: Any, field: str) -> int:
    return int(round(_clamped_unit(value, field) * COORDINATE_MAX))


def _stroke_width(value: Any) -> int:
    quantized = int(round(_clamped_unit(value, "stroke") * STROKE_WIDTH_MAX))
    return min(max(quantized, STROKE_WIDTH_MIN), STROKE_WIDTH_MAX)


def _positive_interval(center: Any, size: Any, axis: str) -> tuple[int, int]:
    center_value = _clamped_unit(center, f"text_{axis}_center")
    size_value = _clamped_unit(size, f"text_{axis}_size")
    lower = int(round(max(center_value - size_value * 0.5, 0.0) * COORDINATE_MAX))
    upper = int(round(min(center_value + size_value * 0.5, 1.0) * COORDINATE_MAX))
    if upper <= lower:
        anchor = int(round(center_value * COORDINATE_MAX))
        if anchor < COORDINATE_MAX:
            lower, upper = anchor, anchor + 1
        else:
            lower, upper = COORDINATE_MAX - 1, COORDINATE_MAX
    return lower, upper


def _normalized_geometry(row: dict[str, Any]) -> list[float]:
    geometry = row.get("geometry")
    if isinstance(geometry, torch.Tensor):
        geometry = geometry.detach().float().cpu().tolist()
    elif hasattr(geometry, "tolist"):
        geometry = geometry.tolist()
    if not isinstance(geometry, (list, tuple)) or len(geometry) != 8:
        raise ValueError("geometry must contain eight normalized values")
    return [float(value) for value in geometry]


def _base_object(row: dict[str, Any], primitive_type: int) -> dict[str, Any]:
    return {
        "query_index": int(row.get("query_index", -1)),
        "score": float(row.get("score", 0.0)),
        "type_id": primitive_type,
        "type": TYPE_TO_CLASS.get(primitive_type, "unknown"),
    }


def _record_from_row(
    row: dict[str, Any],
) -> tuple[
    LineRecordV01 | TriangleRecordV01 | QuadRecordV01 | TextBoxRecordV01,
    dict[str, Any],
]:
    primitive_type = int(row.get("type", NO_OBJECT))
    if primitive_type not in TYPE_TO_CLASS:
        raise ValueError(f"unsupported primitive type: {primitive_type}")
    geometry = _normalized_geometry(row)
    base = _base_object(row, primitive_type)
    if primitive_type == TEXT_BOX:
        x0, x1 = _positive_interval(geometry[0], geometry[2], "x")
        y0, y1 = _positive_interval(geometry[1], geometry[3], "y")
        record = TextBoxRecordV01(x0, y0, x1, y1)
        payload = {
            **base,
            "kind": "text_box",
            "quantized_points": [[x0, y0], [x1, y1]],
        }
        return record, payload

    point_count = {LINE: 2, TRIANGLE: 3, QUAD: 4}[primitive_type]
    values = [
        _coordinate(geometry[index], f"geometry[{index}]")
        for index in range(point_count * 2)
    ]
    points = [values[index : index + 2] for index in range(0, len(values), 2)]
    stroke_width = _stroke_width(row.get("stroke", 0.0))
    if primitive_type == LINE:
        record = LineRecordV01(*values, stroke_width)
    elif primitive_type == TRIANGLE:
        record = TriangleRecordV01(*values, stroke_width)
    else:
        record = QuadRecordV01(*values, stroke_width)
    payload = {
        **base,
        "kind": TYPE_TO_CLASS[primitive_type],
        "quantized_points": points,
        "stroke_width_q": stroke_width,
    }
    return record, payload


def _single_record_parsed(
    primitive_type: int,
    record: LineRecordV01 | TriangleRecordV01 | QuadRecordV01 | TextBoxRecordV01,
) -> ParsedVisualSVGSequenceV01:
    return ParsedVisualSVGSequenceV01(
        (record,) if primitive_type == LINE else (),
        (record,) if primitive_type == TRIANGLE else (),
        (record,) if primitive_type == QUAD else (),
        (record,) if primitive_type == TEXT_BOX else (),
    )


def decoded_rows_to_parsed(rows: Iterable[dict[str, Any]]) -> DecodedRowsResult:
    """Quantize one decoded object set without silently accepting invalid geometry."""

    decoded_counts = Counter({name: 0 for name in CLASS_NAMES})
    record_groups: dict[str, list[Any]] = {name: [] for name in CLASS_NAMES}
    renderable_objects: list[dict[str, Any]] = []
    invalid_objects: list[dict[str, Any]] = []
    for row in rows:
        primitive_type = int(row.get("type", NO_OBJECT))
        class_name = TYPE_TO_CLASS.get(primitive_type)
        if class_name is not None:
            decoded_counts[class_name] += 1
        try:
            record, payload = _record_from_row(row)
            if primitive_type == TEXT_BOX:
                if record.x0 >= record.x1 or record.y0 >= record.y1:
                    raise VisualSVGSequenceError("A text box must have positive width and height")
            else:
                validate_parsed_visual_svg_v0_1(
                    _single_record_parsed(primitive_type, record)
                )
            record_groups[TYPE_TO_CLASS[primitive_type]].append(record)
            renderable_objects.append(payload)
        except (TypeError, ValueError, VisualSVGSequenceError) as error:
            invalid = _base_object(row, primitive_type)
            invalid["error"] = f"{type(error).__name__}: {error}"
            try:
                invalid["geometry"] = _normalized_geometry(row)
            except Exception:
                invalid["geometry"] = None
            stroke = row.get("stroke")
            invalid["stroke"] = float(stroke) if stroke is not None else None
            invalid_objects.append(invalid)

    parsed = ParsedVisualSVGSequenceV01(
        tuple(record_groups["line"]),
        tuple(record_groups["triangle"]),
        tuple(record_groups["quad"]),
        tuple(record_groups["text_box"]),
    )
    sequence = None
    serialization_error = None
    try:
        sequence = serialize_parsed_visual_svg_v0_1(parsed)
        parsed = parse_visual_svg_v0_1(sequence)
    except (TypeError, ValueError, VisualSVGSequenceError) as error:
        serialization_error = f"{type(error).__name__}: {error}"

    renderable_counts = {
        "line": len(parsed.lines),
        "triangle": len(parsed.triangles),
        "quad": len(parsed.quads),
        "text_box": len(parsed.text_boxes),
    }
    return DecodedRowsResult(
        parsed=parsed,
        sequence=sequence,
        decoded_counts={name: int(decoded_counts[name]) for name in CLASS_NAMES},
        renderable_counts=renderable_counts,
        renderable_objects=tuple(renderable_objects),
        invalid_objects=tuple(invalid_objects),
        serialization_error=serialization_error,
    )


def _only_polygon_type(
    parsed: ParsedVisualSVGSequenceV01,
    class_name: str,
) -> ParsedVisualSVGSequenceV01:
    return ParsedVisualSVGSequenceV01(
        (),
        parsed.triangles if class_name == "triangle" else (),
        parsed.quads if class_name == "quad" else (),
        (),
    )


def _safe_bond_iou(parsed: ParsedVisualSVGSequenceV01, target) -> tuple[float, str | None]:
    if not (parsed.lines or parsed.triangles or parsed.quads):
        return 0.0, None
    try:
        return bond_iou(parsed, target), None
    except Exception as error:
        return 0.0, f"{type(error).__name__}: {error}"


def evaluate_decoded_sample(
    *,
    sample_key: str,
    decode_mode: str = "threshold",
    threshold: float | None,
    decoded_rows: list[dict[str, Any]],
    expected_sequence: str,
    bond_layer,
    line_tolerance: float,
    text_box_iou: float,
) -> dict[str, Any]:
    expected = parse_visual_svg_v0_1(expected_sequence)
    converted = decoded_rows_to_parsed(decoded_rows)
    predicted = converted.parsed
    expected_counts = {
        "line": len(expected.lines),
        "triangle": len(expected.triangles),
        "quad": len(expected.quads),
        "text_box": len(expected.text_boxes),
    }
    true_positives = {
        "line": line_true_positives(predicted, expected, line_tolerance),
        "triangle": polygon_true_positives(
            _only_polygon_type(predicted, "triangle"),
            _only_polygon_type(expected, "triangle"),
            line_tolerance,
        ),
        "quad": polygon_true_positives(
            _only_polygon_type(predicted, "quad"),
            _only_polygon_type(expected, "quad"),
            line_tolerance,
        ),
        "text_box": text_box_true_positives(predicted, expected, text_box_iou),
    }
    visual_iou, render_error = _safe_bond_iou(predicted, bond_layer)
    exact = (
        converted.valid
        and render_error is None
        and predicted == expected
    )
    invalid_polygon_count = sum(
        row.get("type") in {"triangle", "quad"}
        for row in converted.invalid_objects
    )
    return {
        "sample_key": sample_key,
        "decode_mode": decode_mode,
        "threshold": threshold,
        "primitive_exact": exact,
        "expected_sequence": expected_sequence,
        "predicted_sequence": converted.sequence,
        "expected_counts": expected_counts,
        "decoded_counts": converted.decoded_counts,
        "renderable_counts": converted.renderable_counts,
        "true_positives": true_positives,
        "renderable_objects": list(converted.renderable_objects),
        "invalid_objects": list(converted.invalid_objects),
        "invalid_polygon_count": invalid_polygon_count,
        "serialization_error": converted.serialization_error,
        "render_error": render_error,
        "conditional_valid": converted.valid and render_error is None,
        "bond_iou": visual_iou,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _class_metrics(rows: list[dict[str, Any]], class_name: str) -> dict[str, Any]:
    true_positive = sum(row["true_positives"][class_name] for row in rows)
    predicted = sum(row["decoded_counts"][class_name] for row in rows)
    expected = sum(row["expected_counts"][class_name] for row in rows)
    precision = _ratio(true_positive, predicted)
    recall = _ratio(true_positive, expected)
    errors = [
        abs(row["decoded_counts"][class_name] - row["expected_counts"][class_name])
        for row in rows
    ]
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "expected": expected,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "count_exact": _ratio(sum(error == 0 for error in errors), len(rows)),
        "count_mae": _ratio(sum(errors), len(rows)),
    }


def _total_count_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = []
    for row in rows:
        predicted = sum(row["decoded_counts"].values())
        expected = sum(row["expected_counts"].values())
        errors.append(abs(predicted - expected))
    return {
        "exact": _ratio(sum(error == 0 for error in errors), len(rows)),
        "mae": _ratio(sum(errors), len(rows)),
    }


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = {name: _class_metrics(rows, name) for name in CLASS_NAMES}
    polygon_tp = classes["triangle"]["true_positive"] + classes["quad"]["true_positive"]
    polygon_predicted = classes["triangle"]["predicted"] + classes["quad"]["predicted"]
    polygon_expected = classes["triangle"]["expected"] + classes["quad"]["expected"]
    polygon_precision = _ratio(polygon_tp, polygon_predicted)
    polygon_recall = _ratio(polygon_tp, polygon_expected)
    ious = [float(row["bond_iou"]) for row in rows]
    return {
        "samples": len(rows),
        "primitive_exact": _ratio(sum(row["primitive_exact"] for row in rows), len(rows)),
        "primitive_count": _total_count_metrics(rows),
        "classes": classes,
        "polygon": {
            "true_positive": polygon_tp,
            "predicted": polygon_predicted,
            "expected": polygon_expected,
            "precision": polygon_precision,
            "recall": polygon_recall,
            "f1": _f1(polygon_precision, polygon_recall),
        },
        "bond_iou": {
            "mean": _ratio(sum(ious), len(ious)),
            "min": min(ious) if ious else 0.0,
        },
    }


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditional = [row for row in rows if row["conditional_valid"]]
    summary = _metric_block(rows)
    summary["invalid"] = {
        "objects": sum(len(row["invalid_objects"]) for row in rows),
        "polygon_objects": sum(row["invalid_polygon_count"] for row in rows),
        "samples": sum(bool(row["invalid_objects"]) for row in rows),
        "serialization_errors": sum(row["serialization_error"] is not None for row in rows),
        "render_errors": sum(row["render_error"] is not None for row in rows),
    }
    summary["conditional_on_valid"] = _metric_block(conditional)
    return summary


def normalize_thresholds(values: list[float] | None) -> tuple[float, ...]:
    thresholds = values if values is not None else [0.5]
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Every threshold must be in [0, 1]")
    return tuple(sorted(set(float(value) for value in thresholds)))


def normalize_decode_modes(values: list[str] | None) -> tuple[str, ...]:
    modes = values if values is not None else ["threshold"]
    if not modes:
        raise ValueError("At least one decode mode is required")
    invalid = [mode for mode in modes if mode not in DECODE_MODES]
    if invalid:
        raise ValueError(f"Unsupported decode modes: {invalid}")
    return tuple(dict.fromkeys(modes))


def build_decode_groups(
    decode_modes: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> tuple[tuple[str, float | None], ...]:
    groups = [
        ("threshold", threshold)
        for threshold in thresholds
        if "threshold" in decode_modes
    ]
    if "count_head" in decode_modes:
        groups.append(("count_head", None))
    return tuple(groups)


def validate_shards(values: list[str]) -> tuple[str, ...]:
    if not values:
        raise ValueError("At least one explicit shard is required")
    resolved = []
    for value in values:
        if any(character in value for character in "*?[]{}"):
            raise ValueError(f"Shard patterns are not allowed; pass explicit files: {value}")
        path = Path(value).expanduser().resolve()
        lowered = str(path).lower().replace("-", "_")
        if "internal_test" in lowered:
            raise ValueError(f"internal_test is sealed and cannot be evaluated: {path}")
        if not path.is_file():
            raise ValueError(f"Shard does not exist or is not a file: {path}")
        resolved.append(str(path))
    return tuple(dict.fromkeys(resolved))


def _checkpoint_configs(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nested = checkpoint.get("config")
    nested = nested if isinstance(nested, dict) else {}
    vision = checkpoint.get("vision_config", nested.get("vision_config"))
    primitive = checkpoint.get(
        "primitive_set_config",
        nested.get("primitive_set_config"),
    )
    if not isinstance(vision, dict) or not isinstance(primitive, dict):
        raise ValueError("Primitive Set checkpoint is missing model configurations")
    return vision, primitive


def load_primitive_set_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PrimitiveSetPredictor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != PrimitiveSetPredictor.model_type:
        raise ValueError(
            f"Unsupported checkpoint model_type: {checkpoint.get('model_type')!r}"
        )
    vision_payload, primitive_payload = _checkpoint_configs(checkpoint)
    model = PrimitiveSetPredictor(
        ChemVectorConfig(**vision_payload),
        PrimitiveSetConfig(**primitive_payload),
    )
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Primitive Set checkpoint is missing model state")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def _ensure_source_split(meta: dict[str, Any], sample_key: str) -> None:
    split = str(meta.get("split", meta.get("source_split", ""))).lower().replace("-", "_")
    output_shard = str(meta.get("output_shard", "")).lower().replace("-", "_")
    if split == "internal_test" or "internal_test" in output_shard:
        raise RuntimeError(f"Refusing sealed internal_test sample: {sample_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        help="Repeat to evaluate a validation threshold sweep; default: 0.5",
    )
    parser.add_argument(
        "--decode-mode",
        action="append",
        choices=DECODE_MODES,
        help="Repeat to compare modes; default: threshold",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--line-tolerance", type=float, default=8.0)
    parser.add_argument("--text-box-iou", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    decode_modes = normalize_decode_modes(args.decode_mode)
    thresholds = normalize_thresholds(args.threshold)
    if args.threshold is not None and "threshold" not in decode_modes:
        raise ValueError("--threshold is only meaningful with --decode-mode threshold")
    shards = validate_shards(args.shard)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    if args.image_size < 1 or args.line_tolerance < 0:
        raise ValueError("Image size must be positive and line tolerance nonnegative")
    if not 0.0 <= args.text_box_iou <= 1.0:
        raise ValueError("--text-box-iou must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model, _ = load_primitive_set_model(args.checkpoint, device)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = VisualSVGWebDataset(list(shards), limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=PrimitiveSetEvaluationCollator(image_size=args.image_size),
    )
    decode_groups = build_decode_groups(decode_modes, thresholds)
    rows_by_group = {group: [] for group in decode_groups}
    samples_seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            for key, meta in zip(batch["keys"], batch["meta"]):
                _ensure_source_split(meta, key)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                model_output = model(pixels)
            for decode_mode, threshold in decode_groups:
                decoded = (
                    decode_primitive_set(
                        model_output,
                        threshold=threshold,
                        mode="threshold",
                    )
                    if decode_mode == "threshold"
                    else decode_primitive_set(model_output, mode="count_head")
                )
                for index, decoded_rows in enumerate(decoded):
                    rows_by_group[(decode_mode, threshold)].append(
                        evaluate_decoded_sample(
                            sample_key=batch["keys"][index],
                            decode_mode=decode_mode,
                            threshold=threshold,
                            decoded_rows=decoded_rows,
                            expected_sequence=batch["target_sequences"][index],
                            bond_layer=batch["bond_layers"][index],
                            line_tolerance=args.line_tolerance,
                            text_box_iou=args.text_box_iou,
                        )
                    )
            samples_seen += len(batch["keys"])
            if samples_seen % 10 == 0:
                print(f"evaluated={samples_seen}", flush=True)

    results = [
        {
            "decode_mode": decode_mode,
            "threshold": threshold,
            **summarize_prediction_rows(rows_by_group[(decode_mode, threshold)]),
        }
        for decode_mode, threshold in decode_groups
    ]
    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_type": PrimitiveSetPredictor.model_type,
        "shards": list(shards),
        "decode_results": results,
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for decode_mode, threshold in decode_groups:
            for row in rows_by_group[(decode_mode, threshold)]:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
