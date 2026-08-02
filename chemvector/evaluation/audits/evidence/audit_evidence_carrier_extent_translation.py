#!/usr/bin/env python3
"""Audit learned v0.9.9 extent geometry under paired integer translations.

The audit is deliberately P0-conditional.  OpenCV LSD is rerun independently
on every translated full raster.  GT is shifted by the same pixel-center
transform and is used only after proposal generation to associate equivalent
positive carriers and score reconstructed physical lines.  The untrained P0
keep head is forced active; only learned normal-offset and extent bins affect
the reported geometry.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image
import torch

from chemvector.audit_evidence_carrier_proposals import (
    COORDINATE_MAX,
    extract_targets,
)
from chemvector.evidence_carrier_extent import (
    EvidenceCarrierExtentPredictor,
    decode_evidence_carrier_extent,
)
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentCollator,
    EvidenceCarrierExtentTargetConfig,
    EvidenceCarrierExtentWebDataset,
)
from chemvector.train_evidence_carrier_extent import (
    DEFAULT_TOLERANCES_Q,
    DEFAULT_VALIDATION_SHARDS,
    assert_unscaled_full_raster_batch,
    autocast_context,
    batch_tensors_to_device,
    evaluate_sample_geometry,
    expand_shards,
    file_sha256,
    load_checkpoint,
    reject_internal_test_paths,
    shard_manifest,
    validate_batch_security,
)


SCHEMA_VERSION = "evidence-carrier-extent-translation-audit-0.1"
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v099-evidence-carrier-extent-translation-validation100/"
    "translation_metrics.json"
)
DEFAULT_TRANSLATIONS = ((16, 0), (32, 0))


def _translation(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("Translation must contain dx and dy")
    dx, dy = value
    if isinstance(dx, bool) or isinstance(dy, bool):
        raise ValueError("Translation offsets must be integers")
    if int(dx) != dx or int(dy) != dy:
        raise ValueError("Translation offsets must be integers")
    result = int(dx), int(dy)
    if result == (0, 0):
        raise ValueError("Do not include the implicit zero-translation baseline")
    return result


def parse_translation(value: str) -> tuple[int, int]:
    """Parse one CLI translation in ``DX,DY`` form."""

    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("translation must use DX,DY")
    try:
        return _translation((int(fields[0]), int(fields[1])))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def translation_name(dx: int, dy: int) -> str:
    dx, dy = _translation((dx, dy))
    return f"x{dx:+d}_y{dy:+d}"


def foreground_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return the inclusive non-white RGB bounds, or ``None`` for a blank image."""

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.any(array != 255, axis=-1)
    if not bool(foreground.any()):
        return None
    rows, columns = np.where(foreground)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def translation_preserves_foreground(
    image: Image.Image,
    *,
    dx: int,
    dy: int,
) -> bool:
    """Return whether an integer translation preserves every raster pixel."""

    dx, dy = _translation((dx, dy))
    width, height = image.size
    if abs(dx) >= width or abs(dy) >= height:
        return False
    bounds = foreground_bounds(image)
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    return (
        0 <= x0 + dx
        and x1 + dx <= width - 1
        and 0 <= y0 + dy
        and y1 + dy <= height - 1
    )


def translate_image(
    image: Image.Image,
    *,
    dx: int,
    dy: int,
    mode: str,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    """Translate an image on its fixed canvas without interpolation."""

    dx, dy = _translation((dx, dy))
    if mode not in {"L", "RGB"}:
        raise ValueError("Translation audit supports only L and RGB rasters")
    source = np.asarray(image.convert(mode), dtype=np.uint8)
    height, width = source.shape[:2]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("Translation offset is outside the image canvas")
    translated = np.empty_like(source)
    translated[...] = fill

    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    destination_x0 = source_x0 + dx
    destination_x1 = source_x1 + dx
    destination_y0 = source_y0 + dy
    destination_y1 = source_y1 + dy
    translated[
        destination_y0:destination_y1,
        destination_x0:destination_x1,
    ] = source[source_y0:source_y1, source_x0:source_x1]
    return Image.fromarray(translated, mode=mode)


def _shift_point(value: Any, dx: float, dy: float, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain x/y")
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain finite x/y")
    shifted = point + np.asarray((dx, dy), dtype=np.float64)
    return [float(shifted[0]), float(shifted[1])]


def _shift_points(value: Any, dx: float, dy: float, name: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain a point list")
    return [
        _shift_point(point, dx, dy, f"{name}[{index}]")
        for index, point in enumerate(value)
    ]


def _shift_box(value: Any, dx: float, dy: float, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain x0/y0/x1/y1")
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"{name} must contain finite coordinates")
    shifted = box + np.asarray((dx, dy, dx, dy), dtype=np.float64)
    return [float(item) for item in shifted]


def translate_sidecars(
    primitives: dict[str, Any],
    layers: dict[str, Any],
    *,
    dx: int,
    dy: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate all geometry used by P0 in pixel-center coordinates."""

    dx, dy = _translation((dx, dy))
    shifted_primitives = copy.deepcopy(primitives)
    shifted_layers = copy.deepcopy(layers)
    width = int(shifted_layers.get("width", 0))
    height = int(shifted_layers.get("height", 0))
    if width < 2 or height < 2:
        raise ValueError("Translation audit needs a pixel-center canvas")
    if (
        int(shifted_primitives.get("width", width)) != width
        or int(shifted_primitives.get("height", height)) != height
    ):
        raise ValueError("Primitive and layer canvases differ")

    q_dx = float(dx) * COORDINATE_MAX / float(width - 1)
    q_dy = float(dy) * COORDINATE_MAX / float(height - 1)
    for index, primitive in enumerate(shifted_primitives.get("primitives", [])):
        if "points" in primitive:
            primitive["points"] = _shift_points(
                primitive["points"], dx, dy, f"primitives[{index}].points"
            )
        if "quantized_points" in primitive:
            primitive["quantized_points"] = _shift_points(
                primitive["quantized_points"],
                q_dx,
                q_dy,
                f"primitives[{index}].quantized_points",
            )
            values = np.asarray(primitive["quantized_points"], dtype=np.float64)
            tolerance = 1e-6
            if (
                (values < -tolerance).any()
                or (values[:, 0] > COORDINATE_MAX + tolerance).any()
                or (values[:, 1] > COORDINATE_MAX + tolerance).any()
            ):
                raise ValueError("Translated primitive leaves the q2047 canvas")

    for index, text in enumerate(shifted_primitives.get("text_instances", [])):
        if text.get("bbox") is not None:
            text["bbox"] = _shift_box(
                text["bbox"], dx, dy, f"text_instances[{index}].bbox"
            )

    for index, atom in enumerate(shifted_layers.get("atoms", [])):
        if atom.get("center") is not None:
            atom["center"] = _shift_point(
                atom["center"], dx, dy, f"atoms[{index}].center"
            )
        if atom.get("bbox") is not None:
            atom["bbox"] = _shift_box(
                atom["bbox"], dx, dy, f"atoms[{index}].bbox"
            )
    for index, bond in enumerate(shifted_layers.get("bonds", [])):
        for field in ("begin", "end"):
            if bond.get(field) is not None:
                bond[field] = _shift_point(
                    bond[field], dx, dy, f"bonds[{index}].{field}"
                )
        if bond.get("bbox") is not None:
            bond["bbox"] = _shift_box(
                bond["bbox"], dx, dy, f"bonds[{index}].bbox"
            )
    return shifted_primitives, shifted_layers


def _verify_target_translation(
    primitives: dict[str, Any],
    layers: dict[str, Any],
    shifted_primitives: dict[str, Any],
    shifted_layers: dict[str, Any],
    *,
    dx: int,
    dy: int,
    long_bond_pixels: float,
) -> None:
    base_lines, base_groups, _ = extract_targets(
        primitives, layers, long_bond_pixels=long_bond_pixels
    )
    shifted_lines, shifted_groups, _ = extract_targets(
        shifted_primitives,
        shifted_layers,
        long_bond_pixels=long_bond_pixels,
    )
    if len(base_lines) != len(shifted_lines) or len(base_groups) != len(shifted_groups):
        raise AssertionError("Translation changed the target cardinality")
    width = int(layers["width"])
    height = int(layers["height"])
    expected_shift = np.asarray(
        (
            float(dx) * COORDINATE_MAX / float(width - 1),
            float(dy) * COORDINATE_MAX / float(height - 1),
        ),
        dtype=np.float64,
    )
    for base, shifted in zip(base_lines, shifted_lines):
        if (
            base.group_index != shifted.group_index
            or base.group_kind != shifted.group_kind
            or base.long_group != shifted.long_group
            or base.text_adjacent != shifted.text_adjacent
        ):
            raise AssertionError("Translation changed target semantics")
        difference = np.asarray(shifted.points_q) - np.asarray(base.points_q)
        np.testing.assert_allclose(
            difference,
            np.broadcast_to(expected_shift, difference.shape),
            atol=1e-6,
            rtol=0.0,
        )


def translate_sample(
    sample: dict[str, Any],
    *,
    dx: int,
    dy: int,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> dict[str, Any]:
    """Build one paired translated sample without reusing base proposals."""

    if not translation_preserves_foreground(sample["image"], dx=dx, dy=dy):
        raise ValueError("Translation would crop foreground raster evidence")
    primitives, layers = translate_sidecars(
        sample["primitives"], sample["layers"], dx=dx, dy=dy
    )
    _verify_target_translation(
        sample["primitives"],
        sample["layers"],
        primitives,
        layers,
        dx=dx,
        dy=dy,
        long_bond_pixels=target_config.long_bond_pixels,
    )
    meta = copy.deepcopy(sample["meta"])
    meta["translation_audit"] = {"dx": dx, "dy": dy}
    return {
        **sample,
        "key": str(sample["key"]),
        "meta": meta,
        "image": translate_image(
            sample["image"], dx=dx, dy=dy, mode="RGB", fill=(255, 255, 255)
        ),
        "bond_layer": translate_image(
            sample["bond_layer"], dx=dx, dy=dy, mode="L", fill=0
        ),
        "primitives": primitives,
        "layers": layers,
    }


def select_paired_samples(
    shards: str | list[str],
    translations: Sequence[tuple[int, int]],
    *,
    limit: int,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one raster-defined no-crop set shared by every scenario."""

    if limit < 1:
        raise ValueError("Translation audit limit must be positive")
    normalized = tuple(_translation(value) for value in translations)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Translation audit offsets must be unique")
    selected: list[dict[str, Any]] = []
    rejected = {translation_name(*value): 0 for value in normalized}
    scanned = 0
    for sample in EvidenceCarrierExtentWebDataset(shards):
        scanned += 1
        failed = [
            value
            for value in normalized
            if not translation_preserves_foreground(
                sample["image"], dx=value[0], dy=value[1]
            )
        ]
        if failed:
            for value in failed:
                rejected[translation_name(*value)] += 1
            continue
        # Validate sidecar movement now so all later scenarios have an exact
        # paired target set.  This never supplies geometry to proposal creation.
        for dx, dy in normalized:
            shifted_primitives, shifted_layers = translate_sidecars(
                sample["primitives"], sample["layers"], dx=dx, dy=dy
            )
            _verify_target_translation(
                sample["primitives"],
                sample["layers"],
                shifted_primitives,
                shifted_layers,
                dx=dx,
                dy=dy,
                long_bond_pixels=target_config.long_bond_pixels,
            )
        selected.append(sample)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise RuntimeError(
            f"Only {len(selected)} no-crop paired samples were available; "
            f"requested {limit}"
        )
    key_digest = hashlib.sha256(
        "\n".join(str(sample["key"]) for sample in selected).encode("utf-8")
    ).hexdigest()
    return selected, {
        "requested": limit,
        "selected": len(selected),
        "scanned": scanned,
        "rejected_for_crop_by_translation": rejected,
        "sample_key_sha256": key_digest,
        "selection_uses_target_geometry": False,
        "selection_rule": "first shard-order samples preserving all non-white pixels",
    }


def _variant_batches(
    samples: Sequence[dict[str, Any]],
    *,
    translation: tuple[int, int] | None,
    target_config: EvidenceCarrierExtentTargetConfig,
    image_size: int,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("Translation audit batch_size must be positive")
    collator = EvidenceCarrierExtentCollator(
        image_size=image_size,
        oracle_bond_layer=False,
        target_config=target_config,
    )
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        if translation is None:
            variant = list(chunk)
        else:
            variant = [
                translate_sample(
                    sample,
                    dx=translation[0],
                    dy=translation[1],
                    target_config=target_config,
                )
                for sample in chunk
            ]
        yield collator(variant)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_records(
    records: Sequence[dict[str, Any]],
    tolerances_q: Sequence[float] = DEFAULT_TOLERANCES_Q,
) -> dict[str, Any]:
    """Summarize geometry while excluding empty-target samples from exact."""

    summary: dict[str, Any] = {}
    for tolerance in tolerances_q:
        name = f"q{int(tolerance)}"
        member_matched = 0
        member_expected = 0
        exact = 0
        exact_expected = 0
        group_totals = {
            alias: {"matched": 0, "expected": 0}
            for alias in ("single", "p2", "p3")
        }
        long_matched = 0
        long_expected = 0
        for record in records:
            values = record["tolerances"][name]
            member_matched += int(values["ordinary_members"]["matched"])
            member_expected += int(values["ordinary_members"]["expected"])
            if int(record["ordinary_target_lines"]) > 0:
                exact_expected += 1
                exact += int(values["structural_sample_exact"])
            for alias in group_totals:
                group_totals[alias]["matched"] += int(
                    values["groups"][alias]["matched"]
                )
                group_totals[alias]["expected"] += int(
                    values["groups"][alias]["expected"]
                )
            long_matched += int(values["long_groups"]["matched"])
            long_expected += int(values["long_groups"]["expected"])
        for values in group_totals.values():
            values["coverage"] = _rate(values["matched"], values["expected"])
        summary[name] = {
            "ordinary_members": {
                "matched": member_matched,
                "expected": member_expected,
                "recall": _rate(member_matched, member_expected),
            },
            "groups": group_totals,
            "long_groups": {
                "matched": long_matched,
                "expected": long_expected,
                "coverage": _rate(long_matched, long_expected),
            },
            "structural_samples": {
                "exact": exact,
                "expected": exact_expected,
                "exact_rate": _rate(exact, exact_expected),
                "empty_target_samples_excluded": len(records) - exact_expected,
            },
        }
    return summary


@torch.inference_mode()
def evaluate_scenario(
    model: EvidenceCarrierExtentPredictor,
    batches: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> dict[str, Any]:
    """Evaluate P0 geometry with every valid carrier forced active."""

    model.eval()
    records: list[dict[str, Any]] = []
    carrier_count = 0
    positive_carrier_count = 0
    for batch in batches:
        validate_batch_security(batch)
        assert_unscaled_full_raster_batch(batch)
        tensors = batch_tensors_to_device(batch, device)
        with autocast_context(device):
            output = model(
                tensors["pixel_values"],
                tensors["carrier_segments"],
                tensors["carrier_mask"],
                coordinate_mode="normalized",
            )
        # P0 never trained keep/drop.  A threshold of zero forces all valid LSD
        # carriers active while normal/extent remain learned checkpoint outputs.
        decoded = decode_evidence_carrier_extent(output, keep_threshold=0.0)
        geometry_mask = tensors["geometry_loss_mask"] & tensors["carrier_mask"]
        predicted_lines = decoded["refined_segments_pixels"].detach().float().cpu()
        line_ids = tensors["target_line_id"].detach().cpu()
        geometry_cpu = geometry_mask.detach().cpu()
        batch_size = int(tensors["pixel_values"].size(0))
        for batch_index in range(batch_size):
            record = evaluate_sample_geometry(
                predicted_lines[batch_index],
                line_ids[batch_index],
                geometry_cpu[batch_index],
                batch["primitives"][batch_index],
                batch["layers"][batch_index],
                target_config=target_config,
                tolerances_q=DEFAULT_TOLERANCES_Q,
            )
            record["sample_key"] = str(batch["keys"][batch_index])
            record["carrier_count"] = int(
                tensors["carrier_mask"][batch_index].sum()
            )
            record["positive_carrier_count"] = int(geometry_mask[batch_index].sum())
            records.append(record)
        carrier_count += int(tensors["carrier_mask"].sum())
        positive_carrier_count += int(geometry_mask.sum())
    if not records:
        raise RuntimeError("Translation audit produced no records")
    geometry = summarize_records(records)
    return {
        "samples": len(records),
        "carrier_count": carrier_count,
        "carrier_count_mean": carrier_count / len(records),
        "positive_carrier_count": positive_carrier_count,
        "geometry": geometry,
        "ordinary_q8_member_recall": geometry["q8"]["ordinary_members"]["recall"],
        "structural_q8_sample_exact_rate": geometry["q8"]["structural_samples"][
            "exact_rate"
        ],
        "records": records,
    }


def _regression(base: float | None, translated: float | None) -> dict[str, Any]:
    if base is None or translated is None:
        return {
            "base": base,
            "translated": translated,
            "delta_pp": None,
            "regression_pp": None,
        }
    return {
        "base": base,
        "translated": translated,
        "delta_pp": (translated - base) * 100.0,
        "regression_pp": (base - translated) * 100.0,
    }


def compare_scenarios(
    base: dict[str, Any],
    translated: dict[str, Any],
) -> dict[str, Any]:
    """Compare paired scenario rates and reject changed GT denominators."""

    base_records = base.get("records", [])
    translated_records = translated.get("records", [])
    base_keys = [record.get("sample_key") for record in base_records]
    translated_keys = [record.get("sample_key") for record in translated_records]
    if base_keys != translated_keys:
        raise AssertionError("Translation scenarios do not contain the same sample keys")
    for base_record, translated_record in zip(base_records, translated_records):
        if base_record.get("ordinary_target_lines") != translated_record.get(
            "ordinary_target_lines"
        ):
            raise AssertionError("Translation changed one sample's target cardinality")

    result: dict[str, Any] = {}
    for name in ("q8", "q16", "q32"):
        base_values = base["geometry"][name]
        translated_values = translated["geometry"][name]
        checks = (
            (
                base_values["ordinary_members"]["expected"],
                translated_values["ordinary_members"]["expected"],
            ),
            (
                base_values["structural_samples"]["expected"],
                translated_values["structural_samples"]["expected"],
            ),
            (
                base_values["long_groups"]["expected"],
                translated_values["long_groups"]["expected"],
            ),
        )
        if any(left != right for left, right in checks):
            raise AssertionError("Translation changed a paired target denominator")
        groups = {}
        for alias in ("single", "p2", "p3"):
            base_group = base_values["groups"][alias]
            translated_group = translated_values["groups"][alias]
            if base_group["expected"] != translated_group["expected"]:
                raise AssertionError("Translation changed a group denominator")
            groups[alias] = _regression(
                base_group["coverage"], translated_group["coverage"]
            )
        result[name] = {
            "ordinary_member_recall": _regression(
                base_values["ordinary_members"]["recall"],
                translated_values["ordinary_members"]["recall"],
            ),
            "structural_sample_exact_rate": _regression(
                base_values["structural_samples"]["exact_rate"],
                translated_values["structural_samples"]["exact_rate"],
            ),
            "groups": groups,
            "long_group_coverage": _regression(
                base_values["long_groups"]["coverage"],
                translated_values["long_groups"]["coverage"],
            ),
        }
    return result


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=list(DEFAULT_VALIDATION_SHARDS),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--translations",
        nargs="+",
        type=parse_translation,
        default=list(DEFAULT_TRANSLATIONS),
        metavar="DX,DY",
    )
    parser.add_argument("--max-q8-regression-pp", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1 or args.batch_size < 1 or args.image_size < 2:
        raise ValueError("limit, batch-size and image-size must be positive")
    if not math.isfinite(args.max_q8_regression_pp) or args.max_q8_regression_pp < 0:
        raise ValueError("max-q8-regression-pp must be finite and nonnegative")
    translations = tuple(_translation(value) for value in args.translations)
    if len(set(translations)) != len(translations):
        raise ValueError("translations must be unique")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    reject_internal_test_paths((checkpoint_path,))
    device = _device(args.device)
    model, target_config, checkpoint = load_checkpoint(checkpoint_path, device=device)
    validation_shards = expand_shards(args.validation_shards)
    samples, selection = select_paired_samples(
        validation_shards,
        translations,
        limit=args.limit,
        target_config=target_config,
    )

    base = evaluate_scenario(
        model,
        _variant_batches(
            samples,
            translation=None,
            target_config=target_config,
            image_size=args.image_size,
            batch_size=args.batch_size,
        ),
        device,
        target_config=target_config,
    )
    translated_rows = []
    q8_regressions = []
    for translation in translations:
        metrics = evaluate_scenario(
            model,
            _variant_batches(
                samples,
                translation=translation,
                target_config=target_config,
                image_size=args.image_size,
                batch_size=args.batch_size,
            ),
            device,
            target_config=target_config,
        )
        comparison = compare_scenarios(base, metrics)
        regression = comparison["q8"]["ordinary_member_recall"]["regression_pp"]
        if regression is None:
            raise RuntimeError("Translation audit has no q8 ordinary target lines")
        q8_regressions.append(float(regression))
        translated_rows.append(
            {
                "name": translation_name(*translation),
                "dx": translation[0],
                "dy": translation[1],
                "metrics": metrics,
                "comparison_to_base": comparison,
            }
        )

    maximum_regression = max(q8_regressions)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "audit_script_sha256": file_sha256(__file__),
        "device": str(device),
        "target_config": asdict(target_config),
        "validation_shards": shard_manifest(validation_shards),
        "selection": selection,
        "base": base,
        "translations": translated_rows,
        "q8_translation_gate": {
            "criterion": "ordinary member recall regression < threshold",
            "threshold_pp": float(args.max_q8_regression_pp),
            "maximum_regression_pp": maximum_regression,
            "passed": maximum_regression < args.max_q8_regression_pp,
        },
        "protocol": {
            "paired_sample_set": True,
            "crop_policy": "reject sample if any non-white full-raster pixel is lost",
            "raster_translation": "integer pixels without interpolation",
            "physical_coordinate_translation": "q += pixels * 2047 / (size - 1)",
            "raster_source": "full_png",
            "oracle_bond_layer": False,
            "lsd_rerun_per_scenario": True,
            "base_carriers_reused_for_translation": False,
            "gt_used_for_proposal_generation": False,
            "gt_used_after_proposal_generation_for_association_and_scoring": True,
            "conditional_positive_carriers": True,
            "keep_head_in_scope": False,
            "keep_decode": "force all valid carriers active with threshold 0.0",
            "prediction_geometry": "learned normal/left/right bins only",
            "empty_target_samples_counted_as_structural_exact": False,
            "internal_test_read": False,
        },
        "uses_internal_test": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "base_q8_recall": base["ordinary_q8_member_recall"],
                "translations": [
                    {
                        "name": row["name"],
                        "q8_recall": row["metrics"]["ordinary_q8_member_recall"],
                        "q8_regression_pp": row["comparison_to_base"]["q8"][
                            "ordinary_member_recall"
                        ]["regression_pp"],
                        "structural_q8_exact_rate": row["metrics"][
                            "structural_q8_sample_exact_rate"
                        ],
                    }
                    for row in translated_rows
                ],
                "gate": payload["q8_translation_gate"],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
