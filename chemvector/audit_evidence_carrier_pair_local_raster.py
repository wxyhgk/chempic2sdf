#!/usr/bin/env python3
"""Render v0.9.18 OOF cross-pair local-raster previews for v0.9.19."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalOrdinaryPairWebDataset,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_pair_local_raster import (
    PairLocalRasterConfig,
    extract_pair_local_rasters,
)


SCHEMA_VERSION = "evidence-carrier-pair-local-raster-preview-0.1"
DEFAULT_SHARDS = [
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-000000.tar"
]
DEFAULT_REPORTS = [
    f"outputs/checkpoints/chemvector-v0918-endpoint-oof/fold{fold}/oof.json"
    for fold in range(3)
]
DEFAULT_SIDECARS = [
    f"outputs/cache/chemvector-v0918-endpoint-oof-fold{fold}.pt"
    for fold in range(3)
]
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0919-pair-local-raster/previews"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_changed_cross_pairs(
    report: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, fold: int
) -> list[dict[str, Any]]:
    by_key = {str(record["identity"]["sample_key"]): record for record in records}
    result = []
    for changed in report.get("changed_samples", []):
        identity = changed["identity"]
        sample_key = str(identity["sample_key"])
        record = by_key.get(sample_key)
        if record is None:
            raise ValueError(f"fold {fold} report sample {sample_key} lacks sidecar record")
        labels = torch.as_tensor(record["target_aware"]["pair_labels"], dtype=torch.long)
        target_free = record["target_free"]
        pairs = torch.as_tensor(target_free["pair_indices"], dtype=torch.long)
        tokens = torch.as_tensor(target_free["pair_tokens"], dtype=torch.float32)
        baseline_selected = set(changed["baseline"]["selected_candidate_indices"])
        for candidate_index in changed["candidate"]["selected_candidate_indices"]:
            candidate_index = int(candidate_index)
            if int(labels[candidate_index]) != 0:
                continue
            result.append(
                {
                    "fold": fold,
                    "sample_key": sample_key,
                    "render_variant": str(identity["render_variant"]),
                    "candidate_index": candidate_index,
                    "anchor_pair": [int(value) for value in pairs[candidate_index]],
                    "raw_gain": float(tokens[candidate_index, -1]),
                    "introduced_by_residual": candidate_index not in baseline_selected,
                }
            )
    return result


def _annotated_image(
    image: Image.Image,
    first: torch.Tensor,
    second: torch.Tensor,
    centerline: torch.Tensor,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for segment, color in ((first, "#e53935"), (second, "#1565c0")):
        points = [tuple(map(float, point)) for point in segment.tolist()]
        draw.line(points, fill=color, width=5)
        for point in points:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
    draw.line(
        [tuple(map(float, point)) for point in centerline.tolist()],
        fill="#00a152",
        width=3,
    )
    annotated.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return annotated


def _crop_preview(crop: torch.Tensor) -> Image.Image:
    ink = crop[0].clamp(0.0, 1.0)
    mask = crop[1].clamp(0.0, 1.0)
    base = (255.0 * (1.0 - ink)).round().to(torch.uint8)
    rgb = torch.stack((base, base, base), dim=-1)
    strength = (mask * 0.65).unsqueeze(-1)
    overlay = torch.tensor((220.0, 30.0, 30.0)).view(1, 1, 3)
    rgb = (rgb.to(torch.float32) * (1.0 - strength) + overlay * strength).round()
    preview = Image.fromarray(rgb.to(torch.uint8).numpy(), mode="RGB")
    return preview.resize((384, 192), Image.Resampling.NEAREST)


def _write_html(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cards = []
    for row in rows:
        title = (
            f"fold {row['fold']} / {row['sample_key']} / {row['render_variant']} / "
            f"candidate {row['candidate_index']}"
        )
        cards.append(
            "<article><h2>"
            + html.escape(title)
            + "</h2><p>anchor pair "
            + html.escape(str(row["anchor_pair"]))
            + f" · raw gain {row['raw_gain']:.6f} · "
            + (
                "residual 新引入"
                if row["introduced_by_residual"]
                else "baseline 已选择"
            )
            + "</p><div class='images'>"
            + f"<img src='{html.escape(row['annotated_file'])}' alt='原图标线'>"
            + f"<img src='{html.escape(row['crop_file'])}' alt='局部 crop'>"
            + "</div></article>"
        )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v0.9.19 Pair-Local Raster 审计</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#171717}
main{max-width:1200px;margin:auto}article{background:white;border:1px solid #ddd;
border-radius:6px;padding:16px;margin:16px 0}h1{font-size:24px}h2{font-size:16px}
p{color:#555}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center}
img{max-width:100%;border:1px solid #ccc;background:white;image-rendering:auto}
@media(max-width:720px){.images{grid-template-columns:1fr}}
</style></head><body><main>
<h1>v0.9.19 Pair-Local Raster OOF Cross 审计</h1>
<p>红/蓝为候选两条 LSD 边界，绿线为配对中心线。右图红色覆盖为候选几何掩码，
其余黑色墨迹来自完整 PNG，可暴露交叉线、相邻键和文字干扰。</p>
""" + "\n".join(cards) + "\n</main></body></html>\n"
    (output / "index.html").write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.reports) != 3 or len(args.sidecars) != 3:
        raise ValueError("exactly three reports and sidecars are required")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = PairLocalRasterConfig()
    rows = []
    sidecar_records: dict[str, Mapping[str, Any]] = {}
    for fold, (report_path, sidecar_path) in enumerate(
        zip(args.reports, args.sidecars, strict=True)
    ):
        with Path(report_path).open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        records = sidecar["records"]
        fold_rows = collect_changed_cross_pairs(report, records, fold=fold)
        rows.extend(fold_rows)
        sidecar_records.update(
            {str(record["identity"]["sample_key"]): record for record in records}
        )

    required_keys = {str(row["sample_key"]) for row in rows}
    dataset = ConditionalOrdinaryPairWebDataset(
        args.shards,
        split="train",
        render_variant=None,
        allowed_sample_keys=required_keys,
        validate_hashes=True,
    )
    samples = {str(sample["key"]): sample for sample in dataset}
    for row in rows:
        key = str(row["sample_key"])
        sample = samples[key]
        record = sidecar_records[key]
        target = build_conditional_ordinary_pair_target(
            sample["image"],
            sample["primitives"],
            sample["layers"],
            pairing_config=conditional_pair_pairing_config("training_wide"),
        )
        frozen_pairs = torch.as_tensor(
            record["target_free"]["pair_indices"], dtype=torch.long
        )
        if not torch.equal(target["pair_indices"], frozen_pairs):
            raise ValueError(f"sample {key} candidate order differs from frozen sidecar")
        candidate_index = int(row["candidate_index"])
        pair = target["pair_indices"][candidate_index]
        crop = extract_pair_local_rasters(
            sample["image"],
            target["anchor_segments_pixels"],
            pair.view(1, 2),
            pair_centerlines_pixels=target["pair_overlap_centerline_pixels"]
            [candidate_index : candidate_index + 1],
            config=config,
        )[0]
        stem = f"fold{row['fold']}-{key}-pair{candidate_index:04d}"
        annotated_file = f"{stem}-source.png"
        crop_file = f"{stem}-crop.png"
        _annotated_image(
            sample["image"],
            target["anchor_segments_pixels"][int(pair[0])],
            target["anchor_segments_pixels"][int(pair[1])],
            target["pair_overlap_centerline_pixels"][candidate_index],
        ).save(output / annotated_file, optimize=True)
        _crop_preview(crop).save(output / crop_file, optimize=True)
        row["annotated_file"] = annotated_file
        row["crop_file"] = crop_file

    rows.sort(
        key=lambda row: (
            int(row["fold"]), str(row["sample_key"]), int(row["candidate_index"])
        )
    )
    _write_html(output, rows)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PAIR-LOCAL-RASTER-PREVIEW-COMPLETE",
        "cross_pair_count": len(rows),
        "sample_count": len(required_keys),
        "stress_pair_count": sum(row["render_variant"] == "stress" for row in rows),
        "introduced_pair_count": sum(
            bool(row["introduced_by_residual"]) for row in rows
        ),
        "config": config.as_dict(),
        "reports": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for path in args.reports
        ],
        "sidecars": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for path in args.sidecars
        ],
        "rows": rows,
    }
    audit_path = output / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**audit, "index_path": str(output / "index.html")}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--reports", nargs=3, default=DEFAULT_REPORTS)
    parser.add_argument("--sidecars", nargs=3, default=DEFAULT_SIDECARS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    print(
        json.dumps(
            run(build_arg_parser().parse_args()),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["collect_changed_cross_pairs", "run"]
