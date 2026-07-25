#!/usr/bin/env python3
"""Replay and visualize all selected cross-target pairs from v0.9.19 OOF."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import chemvector.train_evidence_carrier_endpoint_oof as base
from chemvector.audit_evidence_carrier_pair_local_raster import (
    _annotated_image,
    _crop_preview,
)
from chemvector.evidence_carrier_conditional_pair_data import (
    ConditionalOrdinaryPairWebDataset,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_pair_evaluation import (
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_local_raster import (
    dequantize_pair_local_rasters,
    extract_pair_local_rasters,
)
from chemvector.evidence_carrier_pair_local_residual import (
    EvidenceCarrierPairLocalResidual,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)
from chemvector.train_evidence_carrier_pair_local_oof import (
    DEFAULT_LOCAL_INDEX,
    DEFAULT_OUTPUT_DIR,
    _forward,
    _to_device,
    collate_pair_local_records,
    load_local_fold,
    load_local_index,
    merge_pair_local_records,
)


SCHEMA_VERSION = "evidence-carrier-pair-local-oof-failure-audit-0.1"
DEFAULT_SHARDS = [
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-000000.tar"
]
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0919-pair-local-oof/failures"
)


def classify_cross_pair(
    pair_index: int,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    baseline_selected: Sequence[int],
) -> dict[str, Any]:
    pair = pairs[pair_index]
    first, second = int(pair[0]), int(pair[1])
    incident = (pairs == first).any(dim=1) | (pairs == second).any(dim=1)
    positive = labels == 1
    first_degree = int(((pairs == first).any(dim=1)).sum())
    second_degree = int(((pairs == second).any(dim=1)).sum())
    return {
        "anchor_pair": [first, second],
        "first_endpoint_degree": first_degree,
        "second_endpoint_degree": second_degree,
        "has_incident_positive_candidate": bool((incident & positive).any()),
        "isolated_endpoint_pair": first_degree == 1 and second_degree == 1,
        "introduced_by_pair_local_model": pair_index not in set(baseline_selected),
    }


def _load_model(path: Path, device: torch.device) -> EvidenceCarrierPairLocalResidual:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = EvidenceCarrierPairLocalResidual()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def replay_failures(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_index, base_metadata = base.load_oof_index(args.base_index, audit_path=args.base_audit)
    local_index, local_metadata = load_local_index(args.local_index)
    device = torch.device(args.device)
    rows = []
    for fold in range(3):
        records, base_fold_meta = base.load_oof_fold(base_index, fold)
        crops, local_fold_meta = load_local_fold(local_index, fold)
        records = merge_pair_local_records(records, crops)
        model = _load_model(
            Path(args.checkpoint_dir).expanduser().resolve() / f"fold{fold}" / "final.pt",
            device,
        )
        for start in range(0, len(records), 32):
            current = records[start : start + 32]
            batch = _to_device(collate_pair_local_records(current), device)
            output = _forward(model, batch)
            for image_index, record in enumerate(current):
                count = int(batch["pair_count"][image_index])
                raw = batch["pair_tokens"][image_index, :count, -1].detach().cpu()
                corrected = output["corrected_gain"][image_index, :count].detach().cpu()
                residual = output["residual"][image_index, :count].detach().cpu()
                target = base._evaluation_target(record)
                baseline = evaluate_evidence_carrier_pair_sample(
                    target,
                    raw,
                    threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
                    verify_permutation=False,
                )
                candidate = evaluate_evidence_carrier_pair_sample(
                    target,
                    corrected,
                    threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
                    verify_permutation=False,
                )
                pairs = torch.as_tensor(
                    record["target_free"]["pair_indices"], dtype=torch.long
                )
                labels = torch.as_tensor(
                    record["target_aware"]["pair_labels"], dtype=torch.long
                )
                for pair_index in candidate["selected_candidate_indices"]:
                    pair_index = int(pair_index)
                    if int(labels[pair_index]) != 0:
                        continue
                    crop = dequantize_pair_local_rasters(
                        record["pair_local_rasters"][pair_index : pair_index + 1]
                    )[0]
                    ink = crop[0]
                    border = torch.cat(
                        (
                            ink[:2].flatten(),
                            ink[-2:].flatten(),
                            ink[2:-2, :2].flatten(),
                            ink[2:-2, -2:].flatten(),
                        )
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "sample_key": str(record["identity"]["sample_key"]),
                            "render_variant": str(record["identity"]["render_variant"]),
                            "candidate_index": pair_index,
                            "raw_gain": float(raw[pair_index]),
                            "residual": float(residual[pair_index]),
                            "corrected_gain": float(corrected[pair_index]),
                            "raw_above_threshold": bool(
                                float(raw[pair_index]) > SAMPLE_THRESHOLD_BASELINE_TAU
                            ),
                            "crop_mean_ink": float(ink.mean()),
                            "crop_border_mean_ink": float(border.mean()),
                            **classify_cross_pair(
                                pair_index,
                                pairs,
                                labels,
                                baseline["selected_candidate_indices"],
                            ),
                        }
                    )
        del model, records
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows.sort(key=lambda row: (row["fold"], row["sample_key"], row["candidate_index"]))
    return rows, {
        "base_index": base_metadata,
        "local_index": local_metadata,
    }


def _write_html(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cards = []
    for row in rows:
        badges = [row["render_variant"]]
        if row["introduced_by_pair_local_model"]:
            badges.append("新引入")
        if row["has_incident_positive_candidate"]:
            badges.append("有正候选竞争")
        if row["isolated_endpoint_pair"]:
            badges.append("两端 degree=1")
        cards.append(
            "<article><h2>"
            + html.escape(
                f"fold {row['fold']} / {row['sample_key']} / pair {row['candidate_index']}"
            )
            + "</h2><p>"
            + " · ".join(html.escape(str(value)) for value in badges)
            + "</p><p>"
            + html.escape(
                f"anchors={row['anchor_pair']} degree=({row['first_endpoint_degree']},"
                f"{row['second_endpoint_degree']}) raw={row['raw_gain']:.4f} "
                f"residual={row['residual']:.4f} corrected={row['corrected_gain']:.4f}"
            )
            + "</p><div class='images'>"
            + f"<img src='{row['annotated_file']}' alt='source'>"
            + f"<img src='{row['crop_file']}' alt='crop'>"
            + "</div></article>"
        )
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v0.9.19 OOF 剩余 cross</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f4f4f4;color:#171717}
main{max-width:1200px;margin:auto}article{background:#fff;border:1px solid #ddd;
border-radius:6px;padding:16px;margin:16px 0}h1{font-size:24px}h2{font-size:16px}
p{color:#555}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center}
img{max-width:100%;border:1px solid #ccc;background:#fff}@media(max-width:720px){.images{grid-template-columns:1fr}}
</style></head><body><main><h1>v0.9.19 Pair-Local OOF 剩余 cross</h1>
<p>这里只展示 maximum-weight matching 最终选中的跨 target 假配对。</p>
""" + "\n".join(cards) + "\n</main></body></html>\n"
    (output / "index.html").write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, provenance = replay_failures(args)
    required_keys = {row["sample_key"] for row in rows}
    dataset = ConditionalOrdinaryPairWebDataset(
        args.shards,
        split="train",
        render_variant=None,
        allowed_sample_keys=required_keys,
        validate_hashes=True,
    )
    samples = {str(sample["key"]): sample for sample in dataset}
    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_key.setdefault(row["sample_key"], []).append(row)
    for key, key_rows in rows_by_key.items():
        sample = samples[key]
        target = build_conditional_ordinary_pair_target(
            sample["image"],
            sample["primitives"],
            sample["layers"],
            pairing_config=conditional_pair_pairing_config("training_wide"),
        )
        for row in key_rows:
            pair_index = int(row["candidate_index"])
            pair = target["pair_indices"][pair_index]
            stem = f"fold{row['fold']}-{key}-pair{pair_index:04d}"
            row["annotated_file"] = f"{stem}-source.png"
            row["crop_file"] = f"{stem}-crop.png"
            _annotated_image(
                sample["image"],
                target["anchor_segments_pixels"][int(pair[0])],
                target["anchor_segments_pixels"][int(pair[1])],
                target["pair_overlap_centerline_pixels"][pair_index],
            ).save(output / row["annotated_file"], optimize=True)
            # Recompute only the selected pair to keep the preview tied to the PNG.
            preview_crop = extract_pair_local_rasters(
                sample["image"],
                target["anchor_segments_pixels"],
                pair.view(1, 2),
                pair_centerlines_pixels=target["pair_overlap_centerline_pixels"]
                [pair_index : pair_index + 1],
            )[0]
            _crop_preview(preview_crop).save(output / row["crop_file"], optimize=True)

    _write_html(output, rows)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PAIR-LOCAL-OOF-FAILURE-AUDIT-COMPLETE",
        "cross_pair_count": len(rows),
        "sample_count": len(required_keys),
        "stress_pair_count": sum(row["render_variant"] == "stress" for row in rows),
        "introduced_pair_count": sum(
            bool(row["introduced_by_pair_local_model"]) for row in rows
        ),
        "incident_positive_pair_count": sum(
            bool(row["has_incident_positive_candidate"]) for row in rows
        ),
        "isolated_pair_count": sum(bool(row["isolated_endpoint_pair"]) for row in rows),
        "provenance": provenance,
        "rows": rows,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**audit, "index_path": str(output / "index.html")}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", default=base.DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--base-audit", default=base.DEFAULT_AUDIT)
    parser.add_argument("--local-index", default=DEFAULT_LOCAL_INDEX)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["classify_cross_pair", "replay_failures", "run"]
