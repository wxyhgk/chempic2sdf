"""Frozen BF16/FP32 peak and dense-score counterfactual for external OCR R0."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_atom_reachability import match_target_anchors
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest
from chemvector.external_ocr_r0_selection import validate_selection
from chemvector.external_ocr_r0_truth import (
    load_truth_manifest,
    rendered_truth_targets,
    validate_reveal_inputs,
    validate_reveal_sample,
)


PEAK_STAGES = (
    "a0_bf16_local_peaks",
    "a1_bf16_fp32_peak_ops",
    "a2_fp32_local_peaks",
)
DENSE_CAPACITIES = (128, 192, 256, 384, 768, 1024)
BF16_DENSE_STAGES = tuple(
    f"a3_bf16_dense_top{capacity}" for capacity in DENSE_CAPACITIES
)
FP32_DENSE_STAGES = tuple(
    f"a3_fp32_dense_top{capacity}" for capacity in DENSE_CAPACITIES
)
FP32_DENSE_ANCHOR_STAGE = "a2_fp32_dense_anchors"
STAGE_NAMES = (
    *PEAK_STAGES,
    FP32_DENSE_ANCHOR_STAGE,
    *BF16_DENSE_STAGES,
    *FP32_DENSE_STAGES,
)
REPRESENTATION_RECOVERY_FRACTION = 0.50
DENSE_TOP1024_RECOVERY_FRACTION = 0.80
PAIRED_TEXT_LOST_MAXIMUM = 5
STRESS_TEXT_RECALL_MINIMUM = 0.98
IMPLICIT_RECALL_REGRESSION_MAXIMUM = 0.005


def stable_dense_order(scores):
    """Rank flattened dense cells by score, breaking ties by spatial identity."""

    import torch

    if scores.ndim != 2:
        raise ValueError("Dense scores must have shape [batch, spatial cells]")
    return torch.argsort(scores.float(), dim=1, descending=True, stable=True)


def summarize_peak_precision(
    target_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    roles = ("text", "implicit")
    variants = sorted({row["variant"] for row in target_rows})
    stages = {
        stage: _stage_summary(target_rows, sample_rows, stage, roles, variants)
        for stage in STAGE_NAMES
    }
    a0 = PEAK_STAGES[0]
    deployed_text_misses = [
        row
        for row in target_rows
        if row["role"] == "text" and row["baseline_deployed_miss"]
    ]
    representation_text_misses = [
        row
        for row in target_rows
        if row["role"] == "text" and row["baseline_representation_miss"]
    ]
    paired = {
        stage: {
            role: {
                "gained": sum(
                    row["role"] == role
                    and not row["stages"][a0]["matched"]
                    and row["stages"][stage]["matched"]
                    for row in target_rows
                ),
                "lost": sum(
                    row["role"] == role
                    and row["stages"][a0]["matched"]
                    and not row["stages"][stage]["matched"]
                    for row in target_rows
                ),
            }
            for role in roles
        }
        for stage in STAGE_NAMES[1:]
    }
    deployed_recovery = {
        stage: sum(row["stages"][stage]["matched"] for row in deployed_text_misses)
        for stage in STAGE_NAMES
    }
    representation_recovery = {
        stage: sum(
            row["stages"][stage]["matched"]
            for row in representation_text_misses
        )
        for stage in STAGE_NAMES
    }
    representation_required = math.ceil(
        len(representation_text_misses) * REPRESENTATION_RECOVERY_FRACTION
    )
    a0_text_recall = stages[a0]["roles"]["text"]["recall"]
    a0_stress_recall = stages[a0]["variants"]["stress"]["text"]["rate"]
    a0_implicit_recall = stages[a0]["roles"]["implicit"]["recall"]
    precision_gates = {}
    for stage in PEAK_STAGES[1:]:
        overall_text_recall = stages[stage]["roles"]["text"]["recall"]
        stress_text_recall = stages[stage]["variants"]["stress"]["text"]["rate"]
        dense_preserved = True
        if stage == "a2_fp32_local_peaks":
            dense_preserved = (
                stages[FP32_DENSE_ANCHOR_STAGE]["roles"]["text"]["recall"]
                == 1.0
                and stages[FP32_DENSE_ANCHOR_STAGE]["roles"]["implicit"]["recall"]
                == 1.0
            )
        precision_gates[stage] = {
            "representation_misses_recovered": representation_recovery[stage],
            "representation_recovery_required": representation_required,
            "paired_text_lost": paired[stage]["text"]["lost"],
            "paired_text_lost_maximum": PAIRED_TEXT_LOST_MAXIMUM,
            "overall_text_recall_delta_pp": (
                overall_text_recall - a0_text_recall
            )
            * 100.0,
            "stress_text_recall_delta_pp": (
                stress_text_recall - a0_stress_recall
            )
            * 100.0,
            "fp32_dense_recall_preserved": dense_preserved,
            "passed": (
                representation_recovery[stage] >= representation_required
                and paired[stage]["text"]["lost"] <= PAIRED_TEXT_LOST_MAXIMUM
                and overall_text_recall >= a0_text_recall
                and stress_text_recall >= a0_stress_recall
                and dense_preserved
            ),
        }
    dense_required = math.ceil(
        len(representation_text_misses) * DENSE_TOP1024_RECOVERY_FRACTION
    )
    dense_top1024_gates = {}
    for stage in (BF16_DENSE_STAGES[-1], FP32_DENSE_STAGES[-1]):
        dense_top1024_gates[stage] = {
            "representation_misses_recovered": representation_recovery[stage],
            "representation_recovery_required": dense_required,
            "paired_text_lost": paired[stage]["text"]["lost"],
            "passed": representation_recovery[stage] >= dense_required,
        }
    deployment_candidate_gates = {}
    deployed_required = math.ceil(len(deployed_text_misses) * 0.80)
    for stage in (*BF16_DENSE_STAGES[:4], *FP32_DENSE_STAGES[:4]):
        stress_recall = stages[stage]["variants"]["stress"]["text"]["rate"]
        implicit_recall = stages[stage]["roles"]["implicit"]["recall"]
        deployment_candidate_gates[stage] = {
            "deployed_misses_recovered": deployed_recovery[stage],
            "deployed_recovery_required": deployed_required,
            "stress_text_recall": stress_recall,
            "paired_text_lost": paired[stage]["text"]["lost"],
            "paired_implicit_lost": paired[stage]["implicit"]["lost"],
            "implicit_recall_delta_pp": (
                implicit_recall - a0_implicit_recall
            )
            * 100.0,
            "diagnostic_reachability_passed": (
                deployed_recovery[stage] >= deployed_required
                and stress_recall >= STRESS_TEXT_RECALL_MINIMUM
                and implicit_recall
                >= a0_implicit_recall - IMPLICIT_RECALL_REGRESSION_MAXIMUM
            ),
            "primary_top128_prefix_preserved": False,
            "promotion_authorized": False,
        }
    if precision_gates["a1_bf16_fp32_peak_ops"]["passed"]:
        recommendation = "bf16_peak_ops_quantization"
    elif precision_gates["a2_fp32_local_peaks"]["passed"]:
        recommendation = "bf16_forward_quantization"
    elif any(gate["passed"] for gate in dense_top1024_gates.values()):
        recommendation = "local_peak_operator_candidate"
    else:
        recommendation = "heatmap_score_anchor_geometry_decoupling"
    representation_labels = Counter(
        row["truth"] for row in representation_text_misses
    )
    representation_variants = Counter(
        row["variant"] for row in representation_text_misses
    )
    unique_logit_summary = {}
    for arm in ("a0_bf16", "a2_fp32"):
        values = [row["heatmap_unique_logits"][arm] for row in sample_rows]
        unique_logit_summary[arm] = {
            "mean": float(np.mean(values)) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return {
        "population": {
            "samples": len(sample_rows),
            "targets": len(target_rows),
            "roles": {
                role: sum(row["role"] == role for row in target_rows)
                for role in roles
            },
            "variants": {
                variant: sum(row["variant"] == variant for row in target_rows)
                for variant in variants
            },
        },
        "baseline_failures": {
            "deployed_text_misses": len(deployed_text_misses),
            "peak_inaccessible_text": len(representation_text_misses),
            "peak_inaccessible_variants": dict(sorted(representation_variants.items())),
            "peak_inaccessible_labels": dict(sorted(representation_labels.items())),
        },
        "stages": stages,
        "paired_vs_a0": paired,
        "deployed_text_miss_recovery": deployed_recovery,
        "representation_text_miss_recovery": representation_recovery,
        "heatmap_unique_logits": unique_logit_summary,
        "gates": {
            "precision": precision_gates,
            "dense_top1024": dense_top1024_gates,
            "deployment_candidates_up_to_384": deployment_candidate_gates,
            "recommendation": recommendation,
            "training_authorized": False,
        },
    }


def run_peak_precision_audit(
    dataset_dir: str | Path,
    truth_manifest_path: str | Path,
    baseline_ladder_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    import torch

    from chemvector.data import ChemVectorIterableDataset
    from chemvector.image_io import image_tensor
    from chemvector.train_bond_presence_calibration import load_components
    from chemvector.train_joint_bond_set_multidomain import FRESH_DEV_SHARDS

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The peak precision audit requires CUDA")

    root, crop_manifest, crops = load_crop_manifest(dataset_dir)
    load_truth_manifest(truth_manifest_path, crops)
    selection_path = Path(crop_manifest["selection"]["path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    selected = set(selection["sample_keys"])
    validate_reveal_inputs(list(FRESH_DEV_SHARDS))
    for path, digest, label in (
        (contract.SOURCE_CHECKPOINT, contract.SOURCE_SHA256, "source checkpoint"),
        (
            contract.GROUNDED_GRAPH_CHECKPOINT,
            contract.GROUNDED_GRAPH_SHA256,
            "grounded graph checkpoint",
        ),
    ):
        contract.validate_frozen_file(path, digest, label)
    baseline_report, baseline_targets = _load_baseline_ladder(baseline_ladder_dir)
    if set(row["sample_key"] for row in baseline_targets.values()) != selected:
        raise ValueError("Baseline ladder and frozen selection differ")

    device = torch.device("cuda")
    vision, atom, bond, set_head, source = load_components(
        str(contract.SOURCE_CHECKPOINT),
        str(contract.GROUNDED_GRAPH_CHECKPOINT),
        device,
    )
    del bond, set_head
    for model in (vision, atom):
        model.eval().requires_grad_(False)
    model_state_before = _model_state_sha256((vision, atom))
    proposal_config = source["proposal_config"]
    match_radius_px = float(proposal_config["atom_match_radius_px"])
    image_size = float(proposal_config["image_size"])

    target_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    output_digests = {
        "a0_bf16_forward": hashlib.sha256(),
        "a2_fp32_forward": hashlib.sha256(),
    }

    def consume(samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        pixels = torch.stack([image_tensor(sample["image"]) for sample in samples]).to(
            device
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            bf16_output = atom(vision.forward_stroke_resolution(pixels))
            bf16_repeat = atom(vision.forward_stroke_resolution(pixels))
        _validate_repeat(bf16_output, bf16_repeat, "A0 BF16")
        del bf16_repeat
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
            with torch.autocast(device_type="cuda", enabled=False):
                fp32_output = atom(vision.forward_stroke_resolution(pixels.float()))
                fp32_repeat = atom(vision.forward_stroke_resolution(pixels.float()))
        _validate_repeat(fp32_output, fp32_repeat, "A2 FP32")
        del fp32_repeat
        _update_output_digest(output_digests["a0_bf16_forward"], samples, bf16_output)
        _update_output_digest(output_digests["a2_fp32_forward"], samples, fp32_output)

        bf16_dense, bf16_peak_mask = _dense_proposals(
            bf16_output, force_float_peak_ops=False
        )
        bf16_fp32_ops_dense, bf16_fp32_ops_peak_mask = _dense_proposals(
            bf16_output, force_float_peak_ops=True
        )
        fp32_dense, fp32_peak_mask = _dense_proposals(
            fp32_output, force_float_peak_ops=True
        )
        bf16_ranked = _ranked_dense_proposals(bf16_dense, DENSE_CAPACITIES[-1])
        fp32_ranked = _ranked_dense_proposals(fp32_dense, DENSE_CAPACITIES[-1])
        prefix_masks = {
            capacity: torch.arange(
                DENSE_CAPACITIES[-1], device=device
            )[None].expand(len(samples), -1)
            < capacity
            for capacity in DENSE_CAPACITIES
        }

        stage_proposals = {
            "a0_bf16_local_peaks": bf16_dense,
            "a1_bf16_fp32_peak_ops": bf16_fp32_ops_dense,
            "a2_fp32_local_peaks": fp32_dense,
            FP32_DENSE_ANCHOR_STAGE: fp32_dense,
            **{stage: bf16_ranked for stage in BF16_DENSE_STAGES},
            **{stage: fp32_ranked for stage in FP32_DENSE_STAGES},
        }
        for batch_index, sample in enumerate(samples):
            key = sample["key"]
            targets = rendered_truth_targets(sample)
            metadata = selection["sample_metadata"][key]
            stage_masks = {
                "a0_bf16_local_peaks": bf16_peak_mask[batch_index],
                "a1_bf16_fp32_peak_ops": bf16_fp32_ops_peak_mask[batch_index],
                "a2_fp32_local_peaks": fp32_peak_mask[batch_index],
                FP32_DENSE_ANCHOR_STAGE: fp32_dense["atom_mask"][batch_index],
                **{
                    stage: prefix_masks[capacity][batch_index]
                    for stage, capacity in zip(BF16_DENSE_STAGES, DENSE_CAPACITIES)
                },
                **{
                    stage: prefix_masks[capacity][batch_index]
                    for stage, capacity in zip(FP32_DENSE_STAGES, DENSE_CAPACITIES)
                },
            }
            matches = {
                stage: _match_stage(
                    stage_proposals[stage],
                    batch_index,
                    mask,
                    targets,
                    image_size=image_size,
                    match_radius_px=match_radius_px,
                )
                for stage, mask in stage_masks.items()
            }
            sample_stages = {
                stage: _sample_stage(mask, matches[stage], targets)
                for stage, mask in stage_masks.items()
            }
            sample_rows.append(
                {
                    "sample_key": key,
                    "molecule_id": metadata["molecule_id"],
                    "variant": metadata["variant"],
                    "targets": len(targets),
                    "text_targets": sum(target["visible"] for target in targets),
                    "heatmap_unique_logits": {
                        "a0_bf16": int(
                            bf16_output["heatmap_logits"][batch_index].unique().numel()
                        ),
                        "a2_fp32": int(
                            fp32_output["heatmap_logits"][batch_index].unique().numel()
                        ),
                    },
                    "stages": sample_stages,
                }
            )
            for target_index, target in enumerate(targets):
                gt_key = f"{key}:t{target_index:03d}"
                baseline = baseline_targets.get(gt_key)
                if baseline is None:
                    raise ValueError(f"Baseline target is missing: {gt_key}")
                a0_matched = target_index in matches["a0_bf16_local_peaks"]
                if a0_matched != baseline["all_local_peaks_matched"]:
                    raise ValueError(f"A0 does not reproduce baseline: {gt_key}")
                role = "text" if target["visible"] else "implicit"
                truth = target["truth"] if target["visible"] else None
                if (
                    role != baseline["role"]
                    or truth != baseline["truth"]
                    or metadata["variant"] != baseline["variant"]
                ):
                    raise ValueError(f"Baseline target identity differs: {gt_key}")
                target_rows.append(
                    {
                        "gt_key": gt_key,
                        "sample_key": key,
                        "molecule_id": metadata["molecule_id"],
                        "variant": metadata["variant"],
                        "source_atom_id": target["source_atom_id"],
                        "anchor": target["anchor"],
                        "role": role,
                        "truth": truth,
                        "label_group": (
                            contract.label_group(truth) if truth is not None else None
                        ),
                        "baseline_deployed_miss": (
                            role == "text" and not baseline["deployed_matched"]
                        ),
                        "baseline_representation_miss": (
                            role == "text" and not baseline["all_local_peaks_matched"]
                        ),
                        "stages": {
                            stage: _target_stage(
                                matches[stage].get(target_index),
                                stage_proposals[stage],
                                batch_index,
                                ranked=stage
                                in (*BF16_DENSE_STAGES, *FP32_DENSE_STAGES),
                            )
                            for stage in STAGE_NAMES
                        },
                    }
                )

    pending: list[dict[str, Any]] = []
    for sample in ChemVectorIterableDataset(
        FRESH_DEV_SHARDS,
        sequence_version="0.4",
        serialize_target=False,
        load_raster_layers=False,
    ):
        if sample["key"] not in selected:
            continue
        validate_reveal_sample(sample, selection["sample_metadata"][sample["key"]])
        seen.add(sample["key"])
        pending.append(sample)
        if len(pending) == 4:
            consume(pending)
            pending = []
            if len(seen) % 100 < 4:
                print(f"audited_images={len(seen)}/{len(selected)}", flush=True)
    consume(pending)
    if seen != selected:
        raise ValueError("Peak precision audit did not cover the frozen selection")
    model_state_after = _model_state_sha256((vision, atom))
    if model_state_after != model_state_before:
        raise ValueError("Frozen model state changed during peak precision audit")

    target_rows.sort(key=lambda row: row["gt_key"])
    sample_rows.sort(key=lambda row: row["sample_key"])
    target_path = output / "targets.jsonl"
    sample_path = output / "samples.jsonl"
    contract.write_new_jsonl(target_path, target_rows)
    contract.write_new_jsonl(sample_path, sample_rows)
    report = summarize_peak_precision(target_rows, sample_rows)
    if report["baseline_failures"]["deployed_text_misses"] != baseline_report[
        "deployment_text_misses"
    ]["count"]:
        raise ValueError("Baseline deployed TEXT miss count differs")
    expected_representation_misses = (
        baseline_report["population"]["roles"]["text"]
        - baseline_report["stages"]["all_local_peaks"]["roles"]["text"]["matched"]
    )
    if report["baseline_failures"]["peak_inaccessible_text"] != (
        expected_representation_misses
    ):
        raise ValueError("Baseline peak-inaccessible TEXT count differs")
    report.update(
        {
            "schema_version": "external-ocr-r0-peak-precision-0.1",
            "audit_id": "external-ocr-r0-peak-precision",
            "protocol": {
                "arms": {
                    "a0": "BF16 forward and BF16 sigmoid/max-pool",
                    "a1": "BF16 forward and FP32 sigmoid/max-pool",
                    "a2": "FP32 forward with TF32 disabled and FP32 peak ops",
                    "a3": "stable dense-score prefixes without local-max",
                },
                "dense_capacities": list(DENSE_CAPACITIES),
                "stable_tie_break": "ascending spatial index",
                "local_peak_kernel": 3,
                "match_radius_px": match_radius_px,
                "image_size": image_size,
                "target_quantization_max": 1023.0,
                "training_performed": False,
                "set_head_run": False,
                "ocr_run": False,
            },
            "inputs": {
                "crop_manifest": {
                    "path": str(root / "manifest.json"),
                    "sha256": contract.sha256_file(root / "manifest.json"),
                },
                "truth_manifest": {
                    "path": str(truth_manifest_path),
                    "sha256": contract.sha256_file(truth_manifest_path),
                },
                "selection": {
                    "path": str(selection_path),
                    "sha256": contract.sha256_file(selection_path),
                },
                "baseline_ladder": {
                    "path": str(Path(baseline_ladder_dir) / "report.json"),
                    "sha256": contract.sha256_file(
                        Path(baseline_ladder_dir) / "report.json"
                    ),
                },
                "source_checkpoint": crop_manifest["checkpoints"]["source"],
                "grounded_graph_checkpoint": crop_manifest["checkpoints"][
                    "grounded_graph"
                ],
            },
            "artifacts": {
                "targets": {
                    "path": str(target_path),
                    "sha256": contract.sha256_file(target_path),
                    "rows": len(target_rows),
                },
                "samples": {
                    "path": str(sample_path),
                    "sha256": contract.sha256_file(sample_path),
                    "rows": len(sample_rows),
                },
            },
            "reproducibility": {
                "repeat_forward_exact": True,
                "model_state_before_sha256": model_state_before,
                "model_state_after_sha256": model_state_after,
                "output_digests": {
                    arm: digest.hexdigest() for arm, digest in output_digests.items()
                },
            },
            "provenance": {
                "implementation_sha256": contract.sha256_file(__file__),
                "code_revision": contract.git_revision(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "finished_utc": contract.utc_now(),
            },
        }
    )
    report_path = output / "report.json"
    contract.write_new_json(report_path, report)
    return {
        "output": str(report_path),
        "recommendation": report["gates"]["recommendation"],
        "targets": len(target_rows),
        "samples": len(sample_rows),
    }


def _load_baseline_ladder(
    baseline_ladder_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(baseline_ladder_dir)
    report_path = root / "report.json"
    target_path = root / "targets.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "external-ocr-r0-atom-reachability-0.2":
        raise ValueError("Baseline ladder schema differs")
    artifact = report["artifacts"]["targets"]
    if contract.sha256_file(target_path) != artifact["sha256"]:
        raise ValueError("Baseline ladder target SHA256 differs")
    targets = {}
    last_key = None
    with target_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["gt_key"]
            if last_key is not None and key <= last_key:
                raise ValueError("Baseline ladder targets are not canonically ordered")
            last_key = key
            targets[key] = {
                "sample_key": row["sample_key"],
                "variant": row["variant"],
                "role": row["role"],
                "truth": row["truth"],
                "deployed_matched": bool(row["stages"]["deployed"]["matched"]),
                "all_local_peaks_matched": bool(
                    row["stages"]["all_local_peaks"]["matched"]
                ),
            }
    if len(targets) != artifact["rows"]:
        raise ValueError("Baseline ladder target row count differs")
    return report, targets


def _dense_proposals(atom_output, *, force_float_peak_ops: bool):
    import torch
    from torch.nn import functional as F

    logits = atom_output["heatmap_logits"]
    probabilities = (
        logits.float().sigmoid() if force_float_peak_ops else logits.sigmoid()
    )
    peaks = probabilities == F.max_pool2d(
        probabilities,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    batch, channels, height, width = probabilities.shape
    if channels != 1:
        raise ValueError("Atom proposal heatmap must have one channel")
    spatial_size = height * width
    return (
        {
            "atom_scores": probabilities[:, 0].flatten(1),
            "atom_indices": torch.arange(
                spatial_size, device=probabilities.device
            )[None].expand(batch, -1),
            "atom_geometry": atom_output["anchors"].permute(0, 2, 3, 1).reshape(
                batch, spatial_size, -1
            ),
            "atom_mask": torch.ones(
                (batch, spatial_size),
                device=probabilities.device,
                dtype=torch.bool,
            ),
        },
        peaks[:, 0].flatten(1),
    )


def _ranked_dense_proposals(dense, maximum: int):
    import torch

    order = stable_dense_order(dense["atom_scores"])[:, :maximum]
    batch = torch.arange(order.size(0), device=order.device)[:, None]
    return {
        "atom_scores": dense["atom_scores"][batch, order],
        "atom_indices": dense["atom_indices"][batch, order],
        "atom_geometry": dense["atom_geometry"][batch, order],
        "atom_mask": torch.ones_like(order, dtype=torch.bool),
    }


def _match_stage(
    proposals,
    batch_index: int,
    mask,
    targets: list[dict[str, Any]],
    *,
    image_size: float,
    match_radius_px: float,
) -> dict[int, dict[str, Any]]:
    import torch

    positions = torch.where(mask)[0].tolist()
    candidate_anchors = (
        proposals["atom_geometry"][batch_index, positions, :2].float().cpu().numpy()
    )
    target_anchors = np.asarray([target["anchor"] for target in targets])
    matches = match_target_anchors(
        candidate_anchors,
        target_anchors,
        image_size=image_size,
        target_quantization_max=1023.0,
        match_radius_px=match_radius_px,
    )
    return {
        target: {"proposal_index": positions[candidate], "distance_px": distance}
        for target, (candidate, distance) in matches.items()
    }


def _sample_stage(mask, matches, targets: list[dict[str, Any]]) -> dict[str, Any]:
    selected = int(mask.sum())
    matched = len(matches)
    return {
        "selected": selected,
        "matched": matched,
        "matched_text": sum(targets[index]["visible"] for index in matches),
        "matched_implicit": sum(not targets[index]["visible"] for index in matches),
        "extra_proposals": selected - matched,
    }


def _target_stage(
    match, proposals, batch_index: int, *, ranked: bool
) -> dict[str, Any]:
    if match is None:
        return {
            "matched": False,
            "proposal_rank": None,
            "spatial_index": None,
            "distance_px": None,
            "proposal_score": None,
        }
    position = int(match["proposal_index"])
    score = float(proposals["atom_scores"][batch_index, position])
    if not math.isfinite(score):
        raise ValueError("Peak precision stage contains a non-finite score")
    return {
        "matched": True,
        "proposal_rank": position if ranked else None,
        "spatial_index": int(proposals["atom_indices"][batch_index, position]),
        "distance_px": float(match["distance_px"]),
        "proposal_score": score,
    }


def _stage_summary(target_rows, sample_rows, stage, roles, variants):
    stage_samples = [row["stages"][stage] for row in sample_rows]
    selected = [row["selected"] for row in stage_samples]
    return {
        "roles": {
            role: _matched_rate(
                [row for row in target_rows if row["role"] == role], stage
            )
            for role in roles
        },
        "variants": {
            variant: {
                role: _matched_rate(
                    [
                        row
                        for row in target_rows
                        if row["variant"] == variant and row["role"] == role
                    ],
                    stage,
                )
                for role in roles
            }
            for variant in variants
        },
        "samples": {
            "count": len(stage_samples),
            "selected_proposals": sum(selected),
            "selected_mean": float(np.mean(selected)) if selected else None,
            "selected_p50": float(np.percentile(selected, 50)) if selected else None,
            "selected_p95": float(np.percentile(selected, 95)) if selected else None,
            "selected_max": max(selected) if selected else None,
            "extra_proposals": sum(row["extra_proposals"] for row in stage_samples),
        },
    }


def _matched_rate(rows, stage):
    matched = sum(row["stages"][stage]["matched"] for row in rows)
    return {
        "targets": len(rows),
        "matched": matched,
        "recall": matched / len(rows) if rows else None,
        "rate": matched / len(rows) if rows else None,
    }


def _validate_repeat(first, second, label: str) -> None:
    for field in ("heatmap_logits", "anchors"):
        if not first[field].equal(second[field]):
            raise ValueError(f"{label} repeat forward differs: {field}")


def _update_output_digest(digest, samples, output) -> None:
    import torch

    for sample_index, sample in enumerate(samples):
        digest.update(sample["key"].encode("utf-8"))
        for field in ("heatmap_logits", "anchors"):
            tensor = output[field][sample_index].detach().contiguous().cpu()
            digest.update(field.encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())


def _model_state_sha256(models) -> str:
    import torch

    digest = hashlib.sha256()
    for model_index, model in enumerate(models):
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().contiguous().cpu()
            digest.update(f"{model_index}:{name}".encode("utf-8"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
