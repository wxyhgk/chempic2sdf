#!/usr/bin/env python3
"""Build a fixed 100-image raster-to-primitive capacity dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
from typing import Any
import xml.etree.ElementTree as ET

import cairosvg
import fastjsonschema
import numpy as np
from PIL import Image

try:
    from . import build_chemvision_layers as layer_builder
    from .visual_primitives_v0_1 import (
        extract_visual_primitives_v0_1,
        primitive_counts,
        primitives_to_bond_svg,
    )
    from .visual_svg_sequence_v0_1 import (
        parse_visual_svg_v0_1,
        parsed_visual_svg_from_primitives,
        serialize_visual_svg_v0_1,
    )
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
except ImportError:
    import build_chemvision_layers as layer_builder
    from visual_primitives_v0_1 import (
        extract_visual_primitives_v0_1,
        primitive_counts,
        primitives_to_bond_svg,
    )
    from visual_svg_sequence_v0_1 import (
        parse_visual_svg_v0_1,
        parsed_visual_svg_from_primitives,
        serialize_visual_svg_v0_1,
    )
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


SEED = 20260721
TARGET_SAMPLES = 100
FORCED_KEYS = ("000000046", "000000080", "000000081", "000001157", "000002408")
FEATURE_QUOTAS = {
    "double": 30,
    "triple": 10,
    "dash_wedge": 10,
    "solid_wedge": 10,
    "quad": 3,
    "text_dense": 15,
    "crossed_bonds": 8,
    "long_bond": 8,
}
MEMBER_SUFFIXES = (
    ".png",
    ".bond.png",
    ".primitives.json",
    ".visualsvg.txt",
    ".layers.json",
    ".render.json",
    ".chemmol.json",
    ".meta.json",
)
GROUP_STRUCTURAL_PROTOCOL = "visual-svg-group-structural-gate-0.1"


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def tar_add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(value))


def load_rows(pilot: Path) -> dict[str, dict[str, str]]:
    with (pilot / "samples.csv").open(newline="", encoding="utf-8") as handle:
        return {row["sample_key"]: row for row in csv.DictReader(handle)}


class PilotArchives:
    def __init__(self, pilot: Path, rows: dict[str, dict[str, str]]) -> None:
        self.pilot = pilot
        self.rows = rows
        self.archives: dict[str, tarfile.TarFile] = {}

    def read(self, key: str, suffix: str) -> bytes:
        shard = self.rows[key]["output_shard"]
        archive = self.archives.get(shard)
        if archive is None:
            archive = tarfile.open(self.pilot / shard, "r")
            self.archives[shard] = archive
        extracted = archive.extractfile(key + suffix)
        if extracted is None:
            raise RuntimeError(f"Cannot read {key}{suffix} from {shard}")
        return extracted.read()

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()


def semantic_features(row: dict[str, str], layers: dict[str, Any]) -> set[str]:
    features = set()
    displays = {str(bond["display_type"]) for bond in layers["bonds"]}
    directions = {str(bond["direction"]) for bond in layers["bonds"]}
    visible_text = sum(bool(atom["visible_text"]) for atom in layers["atoms"])
    if "DOUBLE" in displays:
        features.add("double")
    if "TRIPLE" in displays:
        features.add("triple")
    if any("DASH" in value for value in directions):
        features.add("dash_wedge")
    if any("WEDGE" in value for value in directions):
        features.add("solid_wedge")
    if visible_text >= 8:
        features.add("text_dense")
    if row.get("stress_operation") in {"crossed_bonds", "long_bond"}:
        features.add(row["stress_operation"])
    return features


def build_task(row: dict[str, str]) -> layer_builder.RenderTask:
    fonts = {item["font_family_id"]: item for item in layer_builder.FONT_SPECS}
    font = fonts[row["font_family_id"]]
    return layer_builder.RenderTask(
        sample_key=row["sample_key"],
        molecule_id=row["molecule_id"],
        split=row["split"],
        variant=row["variant"],
        source_sample_key=row["source_sample_key"],
        source_shard=row["source_shard"],
        output_shard=row["output_shard"],
        font_family_id=row["font_family_id"],
        font_source_file=font["source_file"],
        font_file_sha256=layer_builder.file_sha256(font["source_file"]),
        font_size=int(row["font_size"]),
        font_bucket=row["font_bucket"],
        bond_line_width=float(row["bond_line_width"]),
        rotation_degrees=float(row["rotation_degrees"]),
        repeat_check=False,
    )


def binary_bond_iou(target_png: bytes, predicted_svg: bytes) -> float:
    rendered = cairosvg.svg2png(
        bytestring=predicted_svg,
        output_width=layer_builder.WIDTH,
        output_height=layer_builder.HEIGHT,
    )
    with Image.open(io.BytesIO(target_png)) as image:
        target = np.asarray(image.convert("L")) > 0
    with Image.open(io.BytesIO(rendered)) as image:
        predicted = np.asarray(image.convert("RGBA").getchannel("A")) > 0
    return float((target & predicted).sum() / max(int((target | predicted).sum()), 1))


def svg_alpha(svg: bytes) -> np.ndarray:
    rendered = cairosvg.svg2png(
        bytestring=svg,
        output_width=layer_builder.WIDTH,
        output_height=layer_builder.HEIGHT,
    )
    with Image.open(io.BytesIO(rendered)) as image:
        return np.asarray(image.convert("RGBA").getchannel("A")) > 0


def svg_iou(left: bytes, right: bytes) -> float:
    left_alpha = svg_alpha(left)
    right_alpha = svg_alpha(right)
    return float(
        (left_alpha & right_alpha).sum()
        / max(int((left_alpha | right_alpha).sum()), 1)
    )


def dilate_one_pixel(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for y_offset in range(3):
        for x_offset in range(3):
            result |= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return result


def svg_pair_metrics(left: bytes, right: bytes) -> tuple[float, float]:
    left_alpha = svg_alpha(left)
    right_alpha = svg_alpha(right)
    union = int((left_alpha | right_alpha).sum())
    iou = float((left_alpha & right_alpha).sum() / max(union, 1))
    precision = float(
        (right_alpha & dilate_one_pixel(left_alpha)).sum()
        / max(int(right_alpha.sum()), 1)
    )
    recall = float(
        (left_alpha & dilate_one_pixel(right_alpha)).sum()
        / max(int(left_alpha.sum()), 1)
    )
    alignment_f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return iou, alignment_f1


def bond_only_svg(svg: bytes, bond_id: int) -> bytes:
    root = ET.fromstring(svg)
    for element in list(root):
        name = element.tag.rsplit("}", 1)[-1]
        classes = tuple(element.attrib.get("class", "").split())
        class_bond = next(
            (
                int(value.removeprefix("bond-"))
                for value in classes
                if value.startswith("bond-") and value.removeprefix("bond-").isdigit()
            ),
            None,
        )
        if name in {"defs", "style"}:
            continue
        if class_bond != bond_id:
            root.remove(element)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def group_iou_metrics(svg: bytes, primitives: dict[str, Any]) -> dict[str, float]:
    by_id = {int(item["primitive_id"]): item for item in primitives["primitives"]}
    all_ious = []
    nonadjusted_ious = []
    polygon_ious = []
    nonadjusted_alignment = []
    polygon_alignment = []
    for group in primitives["bond_groups"]:
        selected = [by_id[int(primitive_id)] for primitive_id in group["primitive_ids"]]
        subset = dict(primitives)
        subset["primitives"] = selected
        value, alignment = svg_pair_metrics(
            bond_only_svg(svg, int(group["bond_id"])),
            primitives_to_bond_svg(subset),
        )
        all_ious.append(value)
        if not any(item.get("quantization_adjusted", False) for item in selected):
            nonadjusted_ious.append(value)
            nonadjusted_alignment.append(alignment)
        if any(item["kind"] in {"triangle", "quad"} for item in selected):
            polygon_ious.append(value)
            polygon_alignment.append(alignment)
    return {
        "group_iou_min": min(all_ious),
        "group_iou_mean": sum(all_ious) / len(all_ious),
        "group_iou_min_nonadjusted": min(nonadjusted_ious, default=1.0),
        "polygon_group_iou_min": min(polygon_ious, default=1.0),
        "group_alignment_f1_min_nonadjusted": min(
            nonadjusted_alignment, default=1.0
        ),
        "polygon_alignment_f1_min": min(polygon_alignment, default=1.0),
    }


def group_structural_metrics(primitives: dict[str, Any]) -> dict[str, Any]:
    """Audit per-bond conversion without rasterizing every group.

    The primitive extractor is the source-to-target conversion boundary: every
    bond path is assigned to exactly one chemical bond before coordinates and
    stroke widths are quantized.  This gate verifies that ownership partition
    and bounds the only lossy operation (quantization).  A single whole-image
    raster IoU remains responsible for the aggregate pixel check.
    """

    width = int(primitives["width"])
    height = int(primitives["height"])
    coordinate_max = int(primitives["coordinate_quantization"])
    stroke_width_scale = int(primitives["stroke_width_scale"])
    if width <= 0 or height <= 0 or coordinate_max <= 0 or stroke_width_scale <= 0:
        raise ValueError("Visual primitive dimensions and quantization must be positive")

    items = primitives.get("primitives")
    groups = primitives.get("bond_groups")
    if not isinstance(items, list) or not isinstance(groups, list) or not groups:
        raise ValueError("Visual primitives must contain non-empty bond groups")
    by_id = {int(item["primitive_id"]): item for item in items}
    if len(by_id) != len(items):
        raise ValueError("Visual primitive IDs must be unique")

    referenced: list[int] = []
    max_coordinate_error = 0.0
    max_coordinate_error_nonadjusted = 0.0
    max_stroke_width_error = 0.0
    adjusted_count = 0
    polygon_count = 0

    for group in groups:
        bond_id = int(group["bond_id"])
        primitive_ids = [int(value) for value in group.get("primitive_ids", [])]
        if not primitive_ids:
            raise ValueError(f"Bond group {bond_id} has no physical primitives")
        for primitive_id in primitive_ids:
            if primitive_id not in by_id:
                raise ValueError(
                    f"Bond group {bond_id} references missing primitive {primitive_id}"
                )
            primitive = by_id[primitive_id]
            if primitive.get("owner_kind") != "bond" or int(
                primitive.get("owner_id", -1)
            ) != bond_id:
                raise ValueError(
                    f"Bond group {bond_id} references foreign primitive {primitive_id}"
                )
            if primitive.get("kind") == "text_box":
                raise ValueError(
                    f"Bond group {bond_id} references text primitive {primitive_id}"
                )
            referenced.append(primitive_id)

            raw_points = primitive.get("points")
            quantized_points = primitive.get("quantized_points")
            if (
                not isinstance(raw_points, list)
                or not isinstance(quantized_points, list)
                or len(raw_points) != len(quantized_points)
                or not raw_points
            ):
                raise ValueError(f"Primitive {primitive_id} has inconsistent points")
            if any(
                not isinstance(point, (list, tuple)) or len(point) != 2
                for point in (*raw_points, *quantized_points)
            ):
                raise ValueError(f"Primitive {primitive_id} has a malformed point")
            adjusted = bool(primitive.get("quantization_adjusted", False))
            if adjusted:
                if primitive.get("kind") != "line" or len(quantized_points) != 2:
                    raise ValueError(
                        f"Primitive {primitive_id} has invalid quantization adjustment"
                    )
                q0, q1 = quantized_points
                if abs(int(q0[0]) - int(q1[0])) + abs(int(q0[1]) - int(q1[1])) != 1:
                    raise ValueError(
                        f"Adjusted primitive {primitive_id} must span one quantized unit"
                    )
                adjusted_count += 1

            expected_quantized_points = [
                [
                    int(
                        round(
                            min(float(width), max(0.0, float(raw[0])))
                            / float(width)
                            * coordinate_max
                        )
                    ),
                    int(
                        round(
                            min(float(height), max(0.0, float(raw[1])))
                            / float(height)
                            * coordinate_max
                        )
                    ),
                ]
                for raw in raw_points
            ]
            if adjusted:
                if (
                    expected_quantized_points[0] != expected_quantized_points[1]
                    or expected_quantized_points[0] not in quantized_points
                ):
                    raise ValueError(
                        f"Adjusted primitive {primitive_id} is not a one-unit rescue"
                    )
            elif expected_quantized_points != quantized_points:
                raise ValueError(
                    f"Primitive {primitive_id} coordinates do not match exact quantization"
                )

            for raw, quantized in zip(raw_points, quantized_points):
                if len(raw) != 2 or len(quantized) != 2:
                    raise ValueError(f"Primitive {primitive_id} has a malformed point")
                qx, qy = int(quantized[0]), int(quantized[1])
                if not (0 <= qx <= coordinate_max and 0 <= qy <= coordinate_max):
                    raise ValueError(
                        f"Primitive {primitive_id} has an out-of-range coordinate"
                    )
                decoded_x = qx / float(coordinate_max) * width
                decoded_y = qy / float(coordinate_max) * height
                error = math.hypot(
                    float(raw[0]) - decoded_x,
                    float(raw[1]) - decoded_y,
                )
                max_coordinate_error = max(max_coordinate_error, error)
                if not adjusted:
                    max_coordinate_error_nonadjusted = max(
                        max_coordinate_error_nonadjusted, error
                    )

            if primitive.get("kind") in {"triangle", "quad"}:
                polygon_count += 1
                area2 = sum(
                    int(left[0]) * int(right[1]) - int(right[0]) * int(left[1])
                    for left, right in zip(
                        quantized_points,
                        (*quantized_points[1:], quantized_points[0]),
                    )
                )
                if area2 == 0:
                    raise ValueError(
                        f"Polygon primitive {primitive_id} collapsed after quantization"
                    )

            raw_stroke = float(primitive["stroke_width"])
            quantized_stroke = int(primitive["stroke_width_q"])
            if not 1 <= quantized_stroke <= 63:
                raise ValueError(
                    f"Primitive {primitive_id} has an out-of-range stroke width"
                )
            expected_stroke = min(
                63,
                max(1, int(round(raw_stroke * stroke_width_scale))),
            )
            if quantized_stroke != expected_stroke:
                raise ValueError(
                    f"Primitive {primitive_id} stroke width does not match exact quantization"
                )
            max_stroke_width_error = max(
                max_stroke_width_error,
                abs(raw_stroke - quantized_stroke / float(stroke_width_scale)),
            )

    bond_primitive_ids = sorted(
        int(item["primitive_id"])
        for item in items
        if item.get("owner_kind") == "bond"
    )
    if sorted(referenced) != bond_primitive_ids or len(referenced) != len(
        set(referenced)
    ):
        raise ValueError("Bond groups must partition every bond primitive exactly once")

    rounding_slack = 1e-3
    nonadjusted_limit = math.hypot(
        width / (2.0 * coordinate_max),
        height / (2.0 * coordinate_max),
    ) + rounding_slack
    adjusted_limit = math.hypot(
        1.5 * width / coordinate_max,
        1.5 * height / coordinate_max,
    ) + rounding_slack
    if max_coordinate_error_nonadjusted > nonadjusted_limit:
        raise ValueError(
            "Non-adjusted primitive coordinate error exceeds the quantization bound: "
            f"{max_coordinate_error_nonadjusted:.8f} > {nonadjusted_limit:.8f}"
        )
    if max_coordinate_error > adjusted_limit:
        raise ValueError(
            "Primitive coordinate error exceeds the adjusted quantization bound: "
            f"{max_coordinate_error:.8f} > {adjusted_limit:.8f}"
        )
    if max_stroke_width_error > 0.5 / stroke_width_scale + 1e-4:
        raise ValueError(
            "Primitive stroke-width error exceeds the 0.1 px quantization bound: "
            f"{max_stroke_width_error:.8f}"
        )

    return {
        "protocol": GROUP_STRUCTURAL_PROTOCOL,
        "group_count": len(groups),
        "bond_primitive_count": len(bond_primitive_ids),
        "polygon_primitive_count": polygon_count,
        "quantization_adjusted_primitive_count": adjusted_count,
        "max_coordinate_error_px": max_coordinate_error,
        "max_coordinate_error_nonadjusted_px": max_coordinate_error_nonadjusted,
        "coordinate_error_limit_nonadjusted_px": nonadjusted_limit,
        "coordinate_error_limit_adjusted_px": adjusted_limit,
        "max_stroke_width_error_px": max_stroke_width_error,
        "partition_exact": True,
    }


def prepare_sample(
    key: str,
    row: dict[str, str],
    archives: PilotArchives,
    tokenizer: VisualSVGTokenizerV01,
    validator,
    max_tokens: int,
    *,
    validate_group_alignment: bool = True,
) -> tuple[dict[str, bytes], dict[str, Any], set[str]]:
    original_png = archives.read(key, ".png")
    bond_png = archives.read(key, ".bond.png")
    layers_bytes = archives.read(key, ".layers.json")
    render_bytes = archives.read(key, ".render.json")
    chemmol_bytes = archives.read(key, ".chemmol.json")
    layers = json.loads(layers_bytes)
    render = json.loads(render_bytes)
    chemmol = json.loads(chemmol_bytes)
    svg, _, _, _ = layer_builder.draw_semantic_svg(
        build_task(row),
        chemmol,
        render,
    )
    rerendered, _ = layer_builder.rasterize_combined(svg)
    if rerendered != original_png:
        raise RuntimeError(f"PNG byte mismatch after deterministic redraw for {key}")

    primitives = extract_visual_primitives_v0_1(svg, layers)
    validator(primitives)
    sequence = serialize_visual_svg_v0_1(primitives)
    parsed = parse_visual_svg_v0_1(sequence)
    if parsed != parsed_visual_svg_from_primitives(primitives):
        raise RuntimeError(f"Primitive-to-sequence semantic mismatch for {key}")
    token_ids = tokenizer.encode(sequence)
    if tokenizer.decode(token_ids) != sequence:
        raise RuntimeError(f"Tokenizer roundtrip mismatch for {key}")
    if len(token_ids) > max_tokens:
        raise RuntimeError(f"Sequence length {len(token_ids)} exceeds {max_tokens}")
    iou = binary_bond_iou(bond_png, primitives_to_bond_svg(primitives))
    if iou < 0.90:
        raise RuntimeError(f"Primitive bond IoU is too low for {key}: {iou:.6f}")
    if validate_group_alignment:
        group_metrics = group_iou_metrics(svg, primitives)
        if group_metrics["group_alignment_f1_min_nonadjusted"] < 0.90:
            raise RuntimeError(
                f"Per-bond primitive alignment is too low for {key}: "
                f"{group_metrics['group_alignment_f1_min_nonadjusted']:.6f}"
            )
        if group_metrics["polygon_alignment_f1_min"] < 0.95:
            raise RuntimeError(
                f"Polygon primitive alignment is too low for {key}: "
                f"{group_metrics['polygon_alignment_f1_min']:.6f}"
            )
    else:
        group_metrics = {
            "group_iou_min": None,
            "group_iou_mean": None,
            "group_iou_min_nonadjusted": None,
            "polygon_group_iou_min": None,
            "group_alignment_f1_min_nonadjusted": None,
            "polygon_alignment_f1_min": None,
        }

    counts = primitive_counts(primitives)
    features = semantic_features(row, layers)
    if counts.get("quad", 0):
        features.add("quad")
    primitives_bytes = json_bytes(primitives)
    sequence_bytes = sequence.encode("utf-8")
    members = {
        ".png": original_png,
        ".bond.png": bond_png,
        ".primitives.json": primitives_bytes,
        ".visualsvg.txt": sequence_bytes,
        ".layers.json": layers_bytes,
        ".render.json": render_bytes,
        ".chemmol.json": chemmol_bytes,
    }
    hashes = {suffix.removeprefix("."): sha256_bytes(value) for suffix, value in members.items()}
    meta = {
        "schema_version": "visual-svg-capacity-meta-0.1",
        "sample_key": key,
        "molecule_id": row["molecule_id"],
        "source_pilot": "chemvision_layers_v0_1/pilot_10k",
        "source_shard": row["output_shard"],
        "source_split": row["split"],
        "source_variant": row["variant"],
        "sequence_schema_version": "visual-svg-0.1",
        "sequence_tokens": len(token_ids),
        "primitive_counts": counts,
        "bond_iou": round(iou, 8),
        "group_alignment_audited": validate_group_alignment,
        "group_iou": {
            name: None if value is None else round(value, 8)
            for name, value in group_metrics.items()
        },
        "sha256": hashes,
    }
    members[".meta.json"] = json_bytes(meta)
    result = {
        "sample_key": key,
        "molecule_id": row["molecule_id"],
        "source_split": row["split"],
        "source_variant": row["variant"],
        "source_shard": row["output_shard"],
        "features": ";".join(sorted(features)),
        "line_count": counts.get("line", 0),
        "triangle_count": counts.get("triangle", 0),
        "quad_count": counts.get("quad", 0),
        "text_box_count": counts.get("text_box", 0),
        "primitive_count": parsed.primitive_count,
        "quantization_adjusted_lines": sum(
            primitive.get("quantization_adjusted", False)
            for primitive in primitives["primitives"]
        ),
        "sequence_tokens": len(token_ids),
        "bond_iou": iou,
        **group_metrics,
        "png_sha256": sha256_bytes(original_png),
        "primitives_sha256": sha256_bytes(primitives_bytes),
        "sequence_sha256": sha256_bytes(sequence_bytes),
    }
    return members, result, features


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_shard(
    shard_path: Path,
    selected: list[tuple[str, dict[str, bytes], dict[str, Any]]],
) -> dict[str, Any]:
    expected_names = {
        key + suffix
        for key, _, _ in selected
        for suffix in MEMBER_SUFFIXES
    }
    with tarfile.open(shard_path, "r") as archive:
        names = [member.name for member in archive]
        if len(names) != len(set(names)):
            raise RuntimeError("Capacity shard contains duplicate member names")
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))[:3]
            extra = sorted(set(names) - expected_names)[:3]
            raise RuntimeError(f"Capacity shard manifest mismatch: missing={missing} extra={extra}")
        checked_hashes = 0
        for key, _, _ in selected:
            meta_file = archive.extractfile(key + ".meta.json")
            if meta_file is None:
                raise RuntimeError(f"Cannot read capacity metadata for {key}")
            meta = json.load(meta_file)
            if meta["source_split"] != "train":
                raise RuntimeError(f"Non-training source leaked into capacity shard: {key}")
            for name, expected_hash in meta["sha256"].items():
                member = archive.extractfile(key + "." + name)
                if member is None:
                    raise RuntimeError(f"Cannot read {key}.{name}")
                actual_hash = sha256_bytes(member.read())
                if actual_hash != expected_hash:
                    raise RuntimeError(f"Post-write hash mismatch for {key}.{name}")
                checked_hashes += 1
    return {
        "members": len(names),
        "checked_hashes": checked_hashes,
        "sha256": sha256_bytes(shard_path.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        default="Data/sdf/chemvision_layers_v0_1/pilot_10k",
    )
    parser.add_argument(
        "--output",
        default="Data/sdf/visual_svg_v0_1/capacity_100",
    )
    parser.add_argument("--samples", type=int, default=TARGET_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < len(FORCED_KEYS):
        raise ValueError("--samples is smaller than the forced regression set")

    pilot = Path(args.pilot).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output already exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "shards").mkdir(parents=True)

    rows = load_rows(pilot)
    for key in FORCED_KEYS:
        if key not in rows:
            raise RuntimeError(f"Forced regression sample is missing: {key}")
        if rows[key]["split"] != "train":
            raise RuntimeError(f"Forced regression sample is not from train: {key}")
    archives = PilotArchives(pilot, rows)
    tokenizer = VisualSVGTokenizerV01()
    schema = json.loads(
        (Path(__file__).parent / "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = fastjsonschema.compile(schema)
    train_keys = [key for key, row in rows.items() if row["split"] == "train"]
    layer_cache = {
        key: json.loads(archives.read(key, ".layers.json"))
        for key in train_keys
    }
    candidates = sorted(
        train_keys,
        key=lambda key: stable_int(SEED, "candidate", key),
    )
    selected: list[tuple[str, dict[str, bytes], dict[str, Any]]] = []
    selected_keys: set[str] = set()
    selected_molecules: set[str] = set()
    feature_counts = Counter()
    variant_counts = Counter()
    variants = ("original", "rdkit", "coordgen", "stress")
    variant_targets = {
        variant: args.samples // len(variants) + int(index < args.samples % len(variants))
        for index, variant in enumerate(variants)
    }
    rejected_by_key: dict[str, dict[str, str]] = {}
    prepared_cache: dict[
        str,
        tuple[dict[str, bytes], dict[str, Any], set[str]],
    ] = {}

    def prepared(key: str):
        if key in prepared_cache:
            return prepared_cache[key]
        row = rows[key]
        try:
            value = prepare_sample(
                key,
                row,
                archives,
                tokenizer,
                validator,
                args.max_tokens,
            )
        except Exception as error:
            rejected_by_key[key] = {
                "sample_key": key,
                "error": f"{type(error).__name__}: {error}",
            }
            return None
        prepared_cache[key] = value
        return value

    def add_key(key: str, required_feature: str | None = None, *, forced: bool = False) -> bool:
        row = rows[key]
        if key in selected_keys or row["molecule_id"] in selected_molecules:
            return False
        variant = row["variant"]
        if not forced and variant_counts[variant] >= variant_targets[variant]:
            return False
        value = prepared(key)
        if value is None:
            return False
        members, result, actual_features = value
        if required_feature is not None and required_feature not in actual_features:
            return False
        selected.append((key, members, result))
        selected_keys.add(key)
        selected_molecules.add(row["molecule_id"])
        feature_counts.update(actual_features)
        variant_counts[variant] += 1
        return True

    for key in FORCED_KEYS:
        if not add_key(key, forced=True):
            error = rejected_by_key.get(key, {}).get("error", "selection conflict")
            raise RuntimeError(f"Forced regression sample failed: {key}: {error}")

    feature_priority = (
        "quad",
        "triple",
        "dash_wedge",
        "solid_wedge",
        "crossed_bonds",
        "long_bond",
        "double",
        "text_dense",
    )
    for feature in feature_priority:
        target = FEATURE_QUOTAS[feature]
        if feature_counts[feature] >= target:
            continue
        semantic_feature = "solid_wedge" if feature == "quad" else feature
        feature_candidates = [
            key
            for key in candidates
            if semantic_feature in semantic_features(rows[key], layer_cache[key])
        ]
        feature_candidates.sort(
            key=lambda key: (
                variant_counts[rows[key]["variant"]]
                / max(variant_targets[rows[key]["variant"]], 1),
                stable_int(SEED, feature, key),
            )
        )
        for key in feature_candidates:
            if feature_counts[feature] >= target:
                break
            add_key(key, required_feature=feature)
        if feature_counts[feature] < target:
            raise RuntimeError(
                f"Cannot meet {feature} quota: {feature_counts[feature]}/{target}"
            )

    for variant in variants:
        for key in candidates:
            if variant_counts[variant] >= variant_targets[variant]:
                break
            if rows[key]["variant"] != variant:
                continue
            add_key(key)
        if variant_counts[variant] != variant_targets[variant]:
            raise RuntimeError(
                f"Cannot fill {variant}: {variant_counts[variant]}/{variant_targets[variant]}"
            )

    archives.close()
    if len(selected) != args.samples:
        raise RuntimeError(f"Selected {len(selected)} samples, expected {args.samples}")
    missing = {
        name: target - feature_counts[name]
        for name, target in FEATURE_QUOTAS.items()
        if feature_counts[name] < target
    }
    if missing:
        raise RuntimeError(f"Capacity feature quotas were not met: {missing}")
    if selected_keys & set(FORCED_KEYS) != set(FORCED_KEYS):
        raise RuntimeError("Not all forced regression samples reached the final selection")
    if any(row["source_split"] != "train" for _, _, row in selected):
        raise RuntimeError("Capacity selection contains a non-training source sample")

    shard_path = output / "shards/capacity-000000.tar"
    with tarfile.open(shard_path, "w") as archive:
        for key, members, _ in sorted(selected):
            for suffix in MEMBER_SUFFIXES:
                tar_add_bytes(archive, key + suffix, members[suffix])
    shard_audit = audit_shard(shard_path, selected)
    sample_rows = [result for _, _, result in sorted(selected)]
    write_csv(output / "samples.csv", sample_rows)
    tokenizer.save(output / "tokenizer.json")
    summary = {
        "schema_version": "visual-svg-capacity-summary-0.1",
        "status": "passed",
        "samples": len(sample_rows),
        "molecules": len({row["molecule_id"] for row in sample_rows}),
        "feature_counts": dict(sorted(feature_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "primitive_counts": {
            name: sum(int(row[f"{name}_count"]) for row in sample_rows)
            for name in ("line", "triangle", "quad", "text_box")
        },
        "quantization_adjusted_lines": sum(
            int(row["quantization_adjusted_lines"]) for row in sample_rows
        ),
        "sequence_tokens": {
            "min": min(int(row["sequence_tokens"]) for row in sample_rows),
            "max": max(int(row["sequence_tokens"]) for row in sample_rows),
            "mean": sum(int(row["sequence_tokens"]) for row in sample_rows) / len(sample_rows),
        },
        "bond_iou": {
            "min": min(float(row["bond_iou"]) for row in sample_rows),
            "mean": sum(float(row["bond_iou"]) for row in sample_rows) / len(sample_rows),
        },
        "tar_members": len(sample_rows) * len(MEMBER_SUFFIXES),
        "shard_bytes": shard_path.stat().st_size,
        "shard_sha256": shard_audit["sha256"],
        "rejected_candidates": len(rejected_by_key),
    }
    validation = {
        "schema_version": "visual-svg-capacity-validation-0.1",
        "status": "passed",
        "checks": {
            "source_split": "train",
            "forced_regression_keys": list(FORCED_KEYS),
            "variant_targets": variant_targets,
            "post_write_members": shard_audit["members"],
            "post_write_hashes": shard_audit["checked_hashes"],
            "minimum_bond_iou": summary["bond_iou"]["min"],
            "minimum_group_iou_nonadjusted": min(
                float(row["group_iou_min_nonadjusted"]) for row in sample_rows
            ),
            "minimum_group_alignment_f1_nonadjusted": min(
                float(row["group_alignment_f1_min_nonadjusted"])
                for row in sample_rows
            ),
            "minimum_polygon_group_iou": min(
                float(row["polygon_group_iou_min"]) for row in sample_rows
            ),
            "minimum_polygon_alignment_f1": min(
                float(row["polygon_alignment_f1_min"]) for row in sample_rows
            ),
            "feature_quotas": FEATURE_QUOTAS,
            "unique_molecules": len(selected_molecules),
        },
        "rejected": list(rejected_by_key.values()),
    }
    (output / "summary.json").write_bytes(json_bytes(summary))
    (output / "validation.json").write_bytes(json_bytes(validation))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
