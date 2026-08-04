#!/usr/bin/env python3
"""Audit frozen P1-U/P1-V peak ranks and fixed multiscale candidate budgets."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
    decode_atom_center_field,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.atoms import train_g3_p1v_continuous_stride1_capacity as p1v
from chemvector.training.graph.scenes.g3_structural_data import (
    OVERFIT32_MANIFEST,
    load_g3_overfit32_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-p1w-multiscale-peak-rank")
P1U_CHECKPOINT = p1u.OUTPUT / "step-004800.pt"
P1U_CHECKPOINT_SHA256 = (
    "0da73a5900c90035be8438e01b872c1fd818a797d03715228e557723966c2c5f"
)
P1V_CHECKPOINT = p1v.OUTPUT / "step-004800.pt"
P1V_CHECKPOINT_SHA256 = (
    "790dead79f17b6857c2f873a9a7d83b98a167aea3cb5768727d3497c80a43b71"
)
MAXIMUM_PEAKS = 1_024
FIXED_K = (32, 64, 96, 128, 192, 256, 512, 1_024)
MULTISCALE_BUDGETS = (
    (128, 0),
    (96, 32),
    (64, 64),
    (32, 96),
    (0, 128),
    (128, 64),
    (64, 128),
    (256, 0),
    (192, 64),
    (128, 128),
    (64, 192),
    (0, 256),
    (256, 128),
    (128, 256),
)
MATCH_RADIUS_PIXELS = 6.0
IMAGE_SIZE = 768.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def match_count(predicted: torch.Tensor, expected: torch.Tensor) -> int:
    return p1a.maximum_cardinality_radius_match_count(
        predicted,
        expected,
        radius_pixels=MATCH_RADIUS_PIXELS,
        image_size=IMAGE_SIZE,
    )


def minimum_sufficient_k(predicted: torch.Tensor, expected: torch.Tensor) -> int | None:
    if match_count(predicted, expected) != len(expected):
        return None
    lower = 1
    upper = len(predicted)
    while lower < upper:
        middle = (lower + upper) // 2
        if match_count(predicted[:middle], expected) == len(expected):
            upper = middle
        else:
            lower = middle + 1
    return lower


def compatible_rank_statistics(predicted: torch.Tensor, expected: torch.Tensor) -> dict:
    distances = torch.cdist(expected.float(), predicted.float()) * IMAGE_SIZE
    compatible = distances <= MATCH_RADIUS_PIXELS
    sentinel = len(predicted) + 1
    ranks = []
    for target in compatible:
        indices = torch.where(target)[0]
        ranks.append(int(indices[0]) + 1 if len(indices) else sentinel)
    top128 = compatible[:, :128]
    return {
        "first_compatible_ranks": ranks,
        "uncovered_targets": sum(rank == sentinel for rank in ranks),
        "targets_with_multiple_top128_peaks": int((top128.sum(dim=1) >= 2).sum()),
        "top128_peaks_compatible_with_multiple_targets": int(
            (top128.sum(dim=0) >= 2).sum()
        ),
    }


def _percentiles(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": int(array.max()),
    }


def summarize_scale(records: dict[str, dict]) -> dict:
    fixed = {
        str(k): {"matched_atoms": 0, "target_atoms": 0, "exact_images": 0}
        for k in FIXED_K
    }
    minimum_k = []
    first_ranks = []
    uncovered = 0
    multiple_targets = shared_peaks = 0
    maxima = []
    per_sample = []
    for sample_key, record in sorted(records.items()):
        predicted = record["anchors"]
        expected = record["expected"]
        target_count = len(expected)
        for k in FIXED_K:
            matched = match_count(predicted[:k], expected)
            fixed[str(k)]["matched_atoms"] += matched
            fixed[str(k)]["target_atoms"] += target_count
            fixed[str(k)]["exact_images"] += int(matched == target_count)
        sufficient = minimum_sufficient_k(predicted, expected)
        ranks = compatible_rank_statistics(predicted, expected)
        if sufficient is not None:
            minimum_k.append(sufficient)
        first_ranks.extend(ranks["first_compatible_ranks"])
        uncovered += ranks["uncovered_targets"]
        multiple_targets += ranks["targets_with_multiple_top128_peaks"]
        shared_peaks += ranks["top128_peaks_compatible_with_multiple_targets"]
        maxima.append(record["regional_maxima_count"])
        per_sample.append(
            {
                "sample_key": sample_key,
                "target_atoms": target_count,
                "minimum_sufficient_k": sufficient,
                "regional_maxima_count": record["regional_maxima_count"],
                "top128_matched": match_count(predicted[:128], expected),
            }
        )
    for value in fixed.values():
        value["atom_recall"] = value["matched_atoms"] / value["target_atoms"]
        value["exact_rate"] = value["exact_images"] / len(records)
    return {
        "images": len(records),
        "fixed_k": fixed,
        "minimum_sufficient_k": {
            "reachable_images": len(minimum_k),
            "unreachable_images": len(records) - len(minimum_k),
            "distribution": _percentiles(minimum_k) if minimum_k else None,
        },
        "first_compatible_rank": {
            "uncovered_targets": uncovered,
            "distribution": _percentiles(first_ranks),
        },
        "top128_duplicate_structure": {
            "targets_with_multiple_compatible_peaks": multiple_targets,
            "peaks_compatible_with_multiple_targets": shared_peaks,
        },
        "regional_maxima": _percentiles(maxima),
        "samples": per_sample,
    }


def evaluate_budget(
    stride2: dict[str, dict],
    stride1: dict[str, dict],
    *,
    stride2_k: int,
    stride1_k: int,
) -> dict:
    if set(stride2) != set(stride1):
        raise ValueError("P1-W multiscale sample identities differ")
    matched_atoms = target_atoms = exact_images = 0
    for key in sorted(stride2):
        left = stride2[key]
        right = stride1[key]
        if not torch.equal(left["expected"], right["expected"]):
            raise ValueError(f"P1-W multiscale targets differ for {key}")
        candidates = torch.cat(
            (left["anchors"][:stride2_k], right["anchors"][:stride1_k]), dim=0
        )
        expected = left["expected"]
        matched = match_count(candidates, expected)
        matched_atoms += matched
        target_atoms += len(expected)
        exact_images += int(matched == len(expected))
    return {
        "stride2_k": stride2_k,
        "stride1_k": stride1_k,
        "total_candidates": stride2_k + stride1_k,
        "matched_atoms": matched_atoms,
        "target_atoms": target_atoms,
        "atom_recall": matched_atoms / target_atoms,
        "exact_images": exact_images,
        "exact_rate": exact_images / len(stride2),
    }


def choose_protocol(budgets: list[dict]) -> dict | None:
    grouped: dict[tuple[int, int], dict[str, dict]] = {}
    for record in budgets:
        key = (record["stride2_k"], record["stride1_k"])
        grouped.setdefault(key, {})[record["domain"]] = record
    for stride2_k, stride1_k in MULTISCALE_BUDGETS:
        domains = grouped.get((stride2_k, stride1_k), {})
        if set(domains) == {"overfit32", "adjacent32"} and all(
            value["exact_images"] == 32 for value in domains.values()
        ):
            return {
                "stride2_k": stride2_k,
                "stride1_k": stride1_k,
                "total_candidates": stride2_k + stride1_k,
                "domains": domains,
            }
    return None


@torch.no_grad()
def collect_candidates(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_model_type: str,
    field_factory,
    domains: dict[str, list],
    collator,
    device,
) -> tuple[dict[str, dict[str, dict]], dict]:
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError(f"P1-W checkpoint SHA differs: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_type") != expected_model_type:
        raise ValueError(f"P1-W checkpoint type differs: {checkpoint}")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    block = p1t.trainable_vision_block(vision)
    frozen_block = copy.deepcopy(block).requires_grad_(False).eval()
    block.load_state_dict(payload["vision_last_stride4_block"], strict=True)
    field = field_factory(atom_head).to(device)
    field.load_state_dict(payload["atom_center_field"], strict=True)
    field.eval()
    outputs: dict[str, dict[str, dict]] = {}
    key_digest = hashlib.sha256()
    started = time.perf_counter()
    for domain, manifest in domains.items():
        current = {}
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
            objects, stroke = p1t.model_inputs(
                atom_head, vision, frozen_block, batch["pixel_values"],
                device=device, training=False,
            )
            with _autocast_context(device, "bf16"):
                dense = field(objects, stroke)
            peaks = extract_regional_maximum_objects(
                decode_atom_center_field(dense), maximum_objects=MAXIMUM_PEAKS
            )
            for index, sample_key_value in enumerate(batch["keys"]):
                sample_key = str(sample_key_value)
                expected = batch["atom_anchors"][
                    index, batch["atom_mask"][index]
                ].float().cpu()
                peak = peaks[index]
                current[sample_key] = {
                    "expected": expected,
                    "anchors": peak["anchors"].float().cpu(),
                    "scores": peak["scores"].float().cpu(),
                    "source_ids": peak["source_ids"].long().cpu(),
                    "regional_maxima_count": int(peak["regional_maxima_count"]),
                }
                key_digest.update(f"{domain}:{sample_key}\n".encode())
        if len(current) != len(manifest):
            raise RuntimeError(f"P1-W collected incomplete {domain} candidates")
        outputs[domain] = current
    lineage = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "model_type": expected_model_type,
        "vision_checkpoint_sha256": vision_sha,
        "atom_state_sha256": module_state_sha256(atom_head),
        "vision_state_sha256": module_state_sha256(vision),
        "sample_keys_sha256": key_digest.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del field, block, frozen_block, vision, atom_head
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, lineage


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-W output directory is not empty")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-W requires CUDA BF16")
    device = torch.device("cuda")
    overfit_manifest, overfit_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    adjacent_manifest, adjacent_audit, _ = p1u.load_adjacent_manifest()
    domains = {
        "overfit32": overfit_manifest,
        "adjacent32": adjacent_manifest,
    }
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    stride2, stride2_lineage = collect_candidates(
        checkpoint=P1U_CHECKPOINT,
        checkpoint_sha256=P1U_CHECKPOINT_SHA256,
        expected_model_type=p1u.MODEL_TYPE,
        field_factory=ContinuousStride2AtomCenterField,
        domains=domains,
        collator=collator,
        device=device,
    )
    stride1, stride1_lineage = collect_candidates(
        checkpoint=P1V_CHECKPOINT,
        checkpoint_sha256=P1V_CHECKPOINT_SHA256,
        expected_model_type=p1v.MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        domains=domains,
        collator=collator,
        device=device,
    )

    scale_summary = {
        "stride2": {domain: summarize_scale(records) for domain, records in stride2.items()},
        "stride1": {domain: summarize_scale(records) for domain, records in stride1.items()},
    }
    budgets = []
    for stride2_k, stride1_k in MULTISCALE_BUDGETS:
        for domain in domains:
            budgets.append(
                {
                    "domain": domain,
                    **evaluate_budget(
                        stride2[domain],
                        stride1[domain],
                        stride2_k=stride2_k,
                        stride1_k=stride1_k,
                    ),
                }
            )
    selected = choose_protocol(budgets)
    decision = (
        "fixed_multiscale_candidate_protocol_go"
        if selected is not None
        else "fixed_budget_coverage_no_go"
    )
    audit = {
        "schema_version": "g3-p1w-multiscale-peak-rank-0.1",
        "decision": decision,
        "maximum_peaks_per_scale": MAXIMUM_PEAKS,
        "fixed_k": list(FIXED_K),
        "budget_order": [
            {"stride2_k": left, "stride1_k": right}
            for left, right in MULTISCALE_BUDGETS
        ],
        "selected_protocol": selected,
        "lineage": {
            "overfit32_manifest": overfit_audit,
            "adjacent32_manifest": adjacent_audit,
            "stride2": stride2_lineage,
            "stride1": stride1_lineage,
        },
        "scales": scale_summary,
        "multiscale_budgets": budgets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({
        "decision": decision,
        "selected_protocol": selected,
        "stride2_fixed_k": {
            domain: summary["fixed_k"] for domain, summary in scale_summary["stride2"].items()
        },
        "stride1_fixed_k": {
            domain: summary["fixed_k"] for domain, summary in scale_summary["stride1"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
