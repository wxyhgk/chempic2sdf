"""Zero-training AtomObject proposal reachability ladder for external OCR R0."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest
from chemvector.external_ocr_r0_selection import validate_selection
from chemvector.external_ocr_r0_truth import (
    load_truth_manifest,
    rendered_truth_targets,
    validate_reveal_inputs,
    validate_reveal_sample,
)


RAW_CAPACITIES = (128, 192, 256, 384)
RAW_STAGE_NAMES = tuple(f"raw_top{capacity}" for capacity in RAW_CAPACITIES)
DIAGNOSTIC_STAGE_NAMES = ("dense_anchors", "all_local_peaks")
STAGE_NAMES = (
    *DIAGNOSTIC_STAGE_NAMES,
    *RAW_STAGE_NAMES,
    "decoded",
    "set_zero",
    "deployed",
)
SET_LOGIT_THRESHOLDS = (
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
)
RAW_RECOVERY_MINIMUM = 0.80
STRESS_TEXT_RECALL_MINIMUM = 0.98
SAFETY_REGRESSION_PP = 0.5
EXPECTED_DEPLOYED_SET_LOGIT_THRESHOLD = -0.25


def match_target_anchors(
    candidate_anchors: np.ndarray,
    target_anchors: np.ndarray,
    *,
    image_size: float,
    target_quantization_max: float,
    match_radius_px: float,
) -> dict[int, tuple[int, float]]:
    """Return one-to-one target-to-candidate matches within the frozen radius."""

    candidates = np.asarray(candidate_anchors, dtype=np.float32).reshape(-1, 2)
    targets = (
        np.asarray(target_anchors, dtype=np.float32).reshape(-1, 2)
        / target_quantization_max
    )
    if not len(candidates) or not len(targets):
        return {}
    distances = np.linalg.norm(
        candidates[:, None, :] - targets[None, :, :], axis=-1
    )
    distances *= image_size
    cardinality = min(len(candidates), len(targets))
    eligible_candidates = np.flatnonzero(
        (distances <= match_radius_px).any(axis=1)
    )
    if not len(eligible_candidates):
        return {}
    eligible_distances = distances[eligible_candidates]
    normalized_cost = eligible_distances / match_radius_px
    cost = np.where(
        eligible_distances <= match_radius_px,
        normalized_cost,
        float(cardinality + 1),
    )
    candidate_rows, target_columns = linear_sum_assignment(cost)
    return {
        int(target): (
            int(eligible_candidates[candidate]),
            float(eligible_distances[candidate, target]),
        )
        for candidate, target in zip(candidate_rows, target_columns)
        if float(eligible_distances[candidate, target]) <= match_radius_px
    }


def legacy_match_target_anchors(
    candidate_anchors: np.ndarray,
    target_anchors: np.ndarray,
    *,
    image_size: float,
    target_quantization_max: float,
    match_radius_px: float,
) -> dict[int, tuple[int, float]]:
    """Reproduce the frozen R0 match-then-filter assignment for audit only."""

    candidates = np.asarray(candidate_anchors, dtype=np.float32).reshape(-1, 2)
    targets = (
        np.asarray(target_anchors, dtype=np.float32).reshape(-1, 2)
        / target_quantization_max
    )
    if not len(candidates) or not len(targets):
        return {}
    distances = np.linalg.norm(
        candidates[:, None, :] - targets[None, :, :], axis=-1
    )
    distances *= image_size
    candidate_rows, target_columns = linear_sum_assignment(distances)
    return {
        int(target): (int(candidate), float(distances[candidate, target]))
        for candidate, target in zip(candidate_rows, target_columns)
        if float(distances[candidate, target]) <= match_radius_px
    }


def summarize_reachability(
    target_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    *,
    deployed_threshold: float,
) -> dict[str, Any]:
    roles = ("text", "implicit")
    variants = sorted({row["variant"] for row in target_rows})
    stages = {
        stage: _stage_summary(target_rows, sample_rows, stage, roles, variants)
        for stage in STAGE_NAMES
    }
    deployment_misses = [
        row
        for row in target_rows
        if row["role"] == "text" and not row["stages"]["deployed"]["matched"]
    ]
    frozen_text = [
        row
        for row in target_rows
        if row["role"] == "text"
        and isinstance(row.get("frozen_deployed_matched"), bool)
    ]
    matching_correction = {
        "available": bool(frozen_text),
        "targets": len(frozen_text),
        "frozen_matched": sum(
            row["frozen_deployed_matched"] for row in frozen_text
        ),
        "corrected_matched": sum(
            row["stages"]["deployed"]["matched"] for row in frozen_text
        ),
        "changed": sum(
            row["frozen_deployed_matched"]
            != row["stages"]["deployed"]["matched"]
            for row in frozen_text
        ),
        "frozen_miss_to_corrected_match": sum(
            not row["frozen_deployed_matched"]
            and row["stages"]["deployed"]["matched"]
            for row in frozen_text
        ),
        "frozen_match_to_corrected_miss": sum(
            row["frozen_deployed_matched"]
            and not row["stages"]["deployed"]["matched"]
            for row in frozen_text
        ),
    }
    recoveries = {
        stage: sum(row["stages"][stage]["matched"] for row in deployment_misses)
        for stage in STAGE_NAMES[:-1]
    }
    raw_recovery_rates = {
        stage: (
            recoveries[stage] / len(deployment_misses)
            if deployment_misses
            else 1.0
        )
        for stage in RAW_STAGE_NAMES
    }
    stress_text = [
        row
        for row in target_rows
        if row["variant"] == "stress" and row["role"] == "text"
    ]
    raw_capacity_gates = {}
    for capacity, stage in zip(RAW_CAPACITIES, RAW_STAGE_NAMES):
        stress_recall = _matched_rate(stress_text, stage)["rate"]
        recovery_passed = raw_recovery_rates[stage] >= RAW_RECOVERY_MINIMUM
        stress_passed = (
            stress_recall is not None
            and stress_recall >= STRESS_TEXT_RECALL_MINIMUM
        )
        raw_capacity_gates[stage] = {
            "capacity": capacity,
            "saturated_samples": sum(
                row["stages"][stage]["selected"] == capacity
                for row in sample_rows
            ),
            "saturation_rate": (
                sum(
                    row["stages"][stage]["selected"] == capacity
                    for row in sample_rows
                )
                / len(sample_rows)
                if sample_rows
                else None
            ),
            "deployment_miss_recovery_rate": raw_recovery_rates[stage],
            "deployment_miss_recovery_passed": recovery_passed,
            "stress_text_recall": stress_recall,
            "stress_text_recall_passed": stress_passed,
            "passed": recovery_passed and stress_passed,
        }
    diagnostic_gates = {}
    for stage in DIAGNOSTIC_STAGE_NAMES:
        stress_recall = _matched_rate(stress_text, stage)["rate"]
        recovery_rate = (
            recoveries[stage] / len(deployment_misses)
            if deployment_misses
            else 1.0
        )
        diagnostic_gates[stage] = {
            "deployment_miss_recovery_rate": recovery_rate,
            "deployment_miss_recovery_passed": (
                recovery_rate >= RAW_RECOVERY_MINIMUM
            ),
            "stress_text_recall": stress_recall,
            "stress_text_recall_passed": (
                stress_recall is not None
                and stress_recall >= STRESS_TEXT_RECALL_MINIMUM
            ),
        }
        diagnostic_gates[stage]["passed"] = (
            diagnostic_gates[stage]["deployment_miss_recovery_passed"]
            and diagnostic_gates[stage]["stress_text_recall_passed"]
        )
    top128_gate = raw_capacity_gates["raw_top128"]
    top384_gate = raw_capacity_gates["raw_top384"]
    if top128_gate["passed"]:
        recommendation = "selector_threshold_or_budget"
    elif top384_gate["passed"]:
        recommendation = "raw_capacity_or_peak_ranking"
    elif diagnostic_gates["all_local_peaks"]["passed"]:
        recommendation = "deeper_raw_capacity_or_peak_ranking"
    elif diagnostic_gates["dense_anchors"]["passed"]:
        recommendation = "peak_precision_and_dense_score_counterfactual"
    else:
        recommendation = "anchor_geometry_generator"
    deployed_implicit = stages["deployed"]["roles"]["implicit"]["recall"]
    deployed_count_exact = stages["deployed"]["samples"]["count_exact_rate"]
    threshold_frontier = {}
    for threshold in SET_LOGIT_THRESHOLDS:
        key = _threshold_key(threshold)
        summary = _stage_summary(
            target_rows,
            sample_rows,
            f"threshold:{key}",
            roles,
            variants,
        )
        implicit_recall = summary["roles"]["implicit"]["recall"]
        count_exact = summary["samples"]["count_exact_rate"]
        summary["safety_vs_deployed"] = {
            "implicit_recall_delta_pp": (
                (implicit_recall - deployed_implicit) * 100.0
                if implicit_recall is not None and deployed_implicit is not None
                else None
            ),
            "count_exact_delta_pp": (
                (count_exact - deployed_count_exact) * 100.0
                if count_exact is not None and deployed_count_exact is not None
                else None
            ),
            "passed": (
                implicit_recall is not None
                and deployed_implicit is not None
                and count_exact is not None
                and deployed_count_exact is not None
                and implicit_recall >= deployed_implicit - SAFETY_REGRESSION_PP / 100.0
                and count_exact >= deployed_count_exact - SAFETY_REGRESSION_PP / 100.0
            ),
        }
        threshold_frontier[key] = summary
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
        "stages": stages,
        "threshold_frontier": threshold_frontier,
        "deployment_text_misses": {
            "count": len(deployment_misses),
            "recovered": recoveries,
            "raw_recovery_rate": raw_recovery_rates["raw_top128"],
            "raw_recovery_rates": raw_recovery_rates,
        },
        "frozen_matching_correction": matching_correction,
        "gates": {
            "raw_recovery_minimum": RAW_RECOVERY_MINIMUM,
            "raw_recovery_passed": top128_gate[
                "deployment_miss_recovery_passed"
            ],
            "stress_text_recall_minimum": STRESS_TEXT_RECALL_MINIMUM,
            "stress_raw_text_recall": top128_gate["stress_text_recall"],
            "stress_recall_passed": top128_gate[
                "stress_text_recall_passed"
            ],
            "raw_capacities": raw_capacity_gates,
            "diagnostic_stages": diagnostic_gates,
            "peak_conditioned_candidate_completeness": (
                "pass"
                if diagnostic_gates["all_local_peaks"]["passed"]
                else "no_go"
            ),
            "safety_regression_pp": SAFETY_REGRESSION_PP,
            "recommendation": recommendation,
            "training_authorized": False,
        },
        "deployed_set_logit_threshold": deployed_threshold,
    }


def run_atom_reachability(
    dataset_dir: str | Path,
    truth_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    import torch

    from chemvector.data import ChemVectorIterableDataset
    from chemvector.image_io import image_tensor
    from chemvector.predicted_incidence import (
        PredictedProposalConfig,
        decode_predicted_atom_proposals,
        decode_predicted_bond_proposals,
    )
    from chemvector.train_bond_presence_calibration import load_components
    from chemvector.train_joint_bond_set_multidomain import FRESH_DEV_SHARDS
    from chemvector.train_spatial_incidence import forward_frozen_bond_head

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The AtomObject reachability audit requires CUDA")

    root, crop_manifest, crops = load_crop_manifest(dataset_dir)
    _, truth_rows = load_truth_manifest(truth_manifest_path, crops)
    selection_path = Path(crop_manifest["selection"]["path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    selected = set(selection["sample_keys"])
    validate_reveal_inputs(list(FRESH_DEV_SHARDS))
    contract.validate_frozen_file(
        contract.GROUNDED_GRAPH_CHECKPOINT,
        contract.GROUNDED_GRAPH_SHA256,
        "grounded graph checkpoint",
    )

    device = torch.device("cuda")
    vision, atom, bond, set_head, source = load_components(
        str(contract.SOURCE_CHECKPOINT),
        str(contract.GROUNDED_GRAPH_CHECKPOINT),
        device,
    )
    for model in (vision, atom, bond, set_head):
        model.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    deployed_threshold = float(source["training_args"]["atom_decision_threshold"])
    if deployed_threshold != EXPECTED_DEPLOYED_SET_LOGIT_THRESHOLD:
        raise ValueError(
            "Frozen deployed set-logit threshold differs: "
            f"{deployed_threshold} != {EXPECTED_DEPLOYED_SET_LOGIT_THRESHOLD}"
        )
    if proposal_config.maximum_atoms != RAW_CAPACITIES[0]:
        raise ValueError(
            "Frozen deployed AtomObject capacity differs: "
            f"{proposal_config.maximum_atoms} != {RAW_CAPACITIES[0]}"
        )
    expected_atoms = _expected_deployed_atoms(root, crop_manifest)
    truth_by_key = {row["gt_key"]: row for row in truth_rows}
    target_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def consume(samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        pixels = torch.stack([image_tensor(sample["image"]) for sample in samples]).to(
            device
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom(features)
            bond_output = forward_frozen_bond_head(vision, bond, features)
            raw_atoms = {
                f"raw_top{capacity}": decode_predicted_atom_proposals(
                    atom_output,
                    replace(
                        proposal_config,
                        maximum_atoms=capacity,
                        atom_threshold=0.0,
                    ),
                )
                for capacity in RAW_CAPACITIES
            }
            _validate_raw_prefixes(raw_atoms)
            dense_atoms, local_peak_mask = _dense_atom_proposals(atom_output)
            atoms = decode_predicted_atom_proposals(atom_output, proposal_config)
            bonds = decode_predicted_bond_proposals(bond_output, proposal_config)
            set_output = set_head(
                atoms["atom_features"],
                atoms["atom_geometry"],
                atoms["atom_scores"],
                atoms["atom_mask"],
                bonds["bond_features"],
                bonds["bond_endpoints"],
                bonds["bond_types"],
                bonds["bond_scores"],
                bonds["bond_mask"],
            )
        presence_logits = set_output["atom_presence_logits"].float()
        if presence_logits.shape != atoms["atom_mask"].shape:
            raise ValueError("Set-head logits do not match deployed AtomObjects")
        for batch_index, sample in enumerate(samples):
            key = sample["key"]
            targets = rendered_truth_targets(sample)
            stage_proposals = {
                "dense_anchors": dense_atoms,
                "all_local_peaks": dense_atoms,
                **raw_atoms,
                "decoded": atoms,
                "set_zero": atoms,
                "deployed": atoms,
            }
            stage_masks = {
                "dense_anchors": dense_atoms["atom_mask"][batch_index],
                "all_local_peaks": local_peak_mask[batch_index],
                **{
                    name: proposals["atom_mask"][batch_index]
                    for name, proposals in raw_atoms.items()
                },
                "decoded": atoms["atom_mask"][batch_index],
                "set_zero": atoms["atom_mask"][batch_index]
                & (presence_logits[batch_index] >= 0.0),
                "deployed": atoms["atom_mask"][batch_index]
                & (presence_logits[batch_index] >= deployed_threshold),
            }
            threshold_masks = {
                _threshold_key(threshold): atoms["atom_mask"][batch_index]
                & (presence_logits[batch_index] >= threshold)
                for threshold in SET_LOGIT_THRESHOLDS
            }
            deployed_threshold_key = _threshold_key(deployed_threshold)
            if not torch.equal(
                threshold_masks[deployed_threshold_key],
                stage_masks["deployed"],
            ):
                raise ValueError("Threshold frontier does not reproduce deployment")
            stage_matches = {
                name: _match_stage(
                    stage_proposals[name],
                    batch_index,
                    mask,
                    targets,
                    proposal_config,
                )
                for name, mask in stage_masks.items()
            }
            legacy_deployed_matches = _match_stage(
                atoms,
                batch_index,
                stage_masks["deployed"],
                targets,
                proposal_config,
                matcher=legacy_match_target_anchors,
            )
            threshold_matches = {
                name: _match_stage(
                    atoms,
                    batch_index,
                    mask,
                    targets,
                    proposal_config,
                )
                for name, mask in threshold_masks.items()
            }
            _validate_deployed_reproduction(
                key,
                atoms,
                batch_index,
                stage_masks["deployed"],
                expected_atoms.get(key, []),
            )
            metadata = selection["sample_metadata"][key]
            sample_stage_rows = {}
            for name, mask in stage_masks.items():
                sample_stage_rows[name] = _sample_stage_row(
                    mask, stage_matches[name], targets
                )
            for name, mask in threshold_masks.items():
                sample_stage_rows[f"threshold:{name}"] = _sample_stage_row(
                    mask, threshold_matches[name], targets
                )
            sample_rows.append(
                {
                    "sample_key": key,
                    "molecule_id": metadata["molecule_id"],
                    "variant": metadata["variant"],
                    "targets": len(targets),
                    "text_targets": sum(target["visible"] for target in targets),
                    "stages": sample_stage_rows,
                }
            )
            for target_index, target in enumerate(targets):
                gt_key = f"{key}:t{target_index:03d}"
                role = "text" if target["visible"] else "implicit"
                stages = {
                    name: _target_stage_row(
                        match=stage_matches[name].get(target_index),
                        proposals=stage_proposals[name],
                        batch_index=batch_index,
                        presence_logits=(
                            None
                            if name in (*DIAGNOSTIC_STAGE_NAMES, *RAW_STAGE_NAMES)
                            else presence_logits
                        ),
                        decoded_mask=(
                            None
                            if name in (*DIAGNOSTIC_STAGE_NAMES, *RAW_STAGE_NAMES)
                            else atoms["atom_mask"][batch_index]
                        ),
                        set_evaluated=name
                        not in (*DIAGNOSTIC_STAGE_NAMES, *RAW_STAGE_NAMES),
                    )
                    for name in STAGE_NAMES
                }
                stages.update(
                    {
                        f"threshold:{name}": _target_stage_row(
                            match=threshold_matches[name].get(target_index),
                            proposals=atoms,
                            batch_index=batch_index,
                            presence_logits=presence_logits,
                            decoded_mask=atoms["atom_mask"][batch_index],
                            set_evaluated=True,
                        )
                        for name in threshold_masks
                    }
                )
                row = {
                    "gt_key": gt_key,
                    "sample_key": key,
                    "molecule_id": metadata["molecule_id"],
                    "variant": metadata["variant"],
                    "source_atom_id": target["source_atom_id"],
                    "anchor": target["anchor"],
                    "role": role,
                    "truth": target["truth"] if target["visible"] else None,
                    "frozen_deployed_matched": None,
                    "stages": stages,
                }
                if role == "text":
                    frozen_truth = truth_by_key.get(gt_key)
                    if frozen_truth is None:
                        raise ValueError(f"Frozen truth row is missing: {gt_key}")
                    frozen_deployed_matched = bool(frozen_truth["matched_atom"])
                    if frozen_truth["truth"] != target["truth"] or (
                        frozen_deployed_matched
                        != bool(legacy_deployed_matches.get(target_index))
                    ):
                        raise ValueError(
                            "Deployed proposal/truth reproduction differs: "
                            f"{gt_key}"
                        )
                    row["frozen_deployed_matched"] = frozen_deployed_matched
                target_rows.append(row)

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
        if len(pending) == 6:
            consume(pending)
            pending = []
            if len(seen) % 100 < 6:
                print(f"audited_images={len(seen)}/{len(selected)}", flush=True)
    consume(pending)
    if seen != selected:
        raise ValueError("Reachability audit did not cover the frozen selection")

    target_rows.sort(key=lambda row: row["gt_key"])
    sample_rows.sort(key=lambda row: row["sample_key"])
    target_path = output / "targets.jsonl"
    sample_path = output / "samples.jsonl"
    contract.write_new_jsonl(target_path, target_rows)
    contract.write_new_jsonl(sample_path, sample_rows)
    report = summarize_reachability(
        target_rows,
        sample_rows,
        deployed_threshold=deployed_threshold,
    )
    report.update(
        {
            "schema_version": "external-ocr-r0-atom-reachability-0.2",
            "audit_id": "external-ocr-r0-atom-reachability",
            "protocol": {
                "stages": list(STAGE_NAMES),
                "set_logit_thresholds": list(SET_LOGIT_THRESHOLDS),
                "raw_capacities": list(RAW_CAPACITIES),
                "atom_match_radius_px": proposal_config.atom_match_radius_px,
                "formal_matcher": "maximum_feasible_cardinality_then_distance",
                "frozen_matcher_reproduced": "minimum_distance_then_radius_filter",
                "raw_atom_threshold": 0.0,
                "local_peak_kernel": 3,
                "diagnostic_stages": {
                    "dense_anchors": "all predicted anchor-field cells",
                    "all_local_peaks": "all 3x3 heatmap local maxima",
                },
                "decoded_atom_threshold": proposal_config.atom_threshold,
                "maximum_atoms": proposal_config.maximum_atoms,
                "training_performed": False,
                "routing_performed": False,
                "replacement_performed": False,
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
            "provenance": {
                "implementation_sha256": contract.sha256_file(__file__),
                "code_revision": contract.git_revision(),
                "python": sys.version,
                "platform": platform.platform(),
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


def _expected_deployed_atoms(
    root: Path, manifest: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    rows = contract.read_jsonl(root / manifest["artifacts"]["atom_objects"]["path"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_key"]].append(row)
    return grouped


def _validate_raw_prefixes(raw_atoms: dict[str, Any]) -> None:
    for smaller, larger in zip(RAW_STAGE_NAMES, RAW_STAGE_NAMES[1:]):
        prefix = raw_atoms[smaller]["atom_indices"].size(1)
        for field in ("atom_indices", "atom_scores", "atom_geometry", "atom_mask"):
            if not raw_atoms[smaller][field].equal(
                raw_atoms[larger][field][:, :prefix]
            ):
                raise ValueError(
                    f"Raw AtomObject ranking is not prefix-stable: "
                    f"{smaller} vs {larger} field={field}"
                )


def _dense_atom_proposals(atom_output):
    import torch
    from torch.nn import functional as F

    heatmap_probabilities = atom_output["heatmap_logits"].sigmoid()
    peak_mask = heatmap_probabilities == F.max_pool2d(
        heatmap_probabilities,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    batch, channels, height, width = heatmap_probabilities.shape
    if channels != 1:
        raise ValueError("Atom proposal heatmap must have one channel")
    spatial_size = height * width
    indices = torch.arange(
        spatial_size,
        device=heatmap_probabilities.device,
    )[None].expand(batch, -1)
    geometry = atom_output["anchors"].permute(0, 2, 3, 1).reshape(
        batch, spatial_size, -1
    )
    return (
        {
            "atom_scores": heatmap_probabilities[:, 0].flatten(1),
            "atom_indices": indices,
            "atom_geometry": geometry,
            "atom_mask": torch.ones(
                (batch, spatial_size),
                device=heatmap_probabilities.device,
                dtype=torch.bool,
            ),
        },
        peak_mask[:, 0].flatten(1),
    )


def _match_stage(
    proposals,
    batch_index: int,
    mask,
    targets: list[dict[str, Any]],
    proposal_config,
    *,
    matcher=match_target_anchors,
) -> dict[int, dict[str, Any]]:
    import torch

    positions = torch.where(mask)[0].tolist()
    candidate_anchors = (
        proposals["atom_geometry"][batch_index, positions, :2].float().cpu().numpy()
    )
    target_anchors = np.asarray([target["anchor"] for target in targets])
    matches = matcher(
        candidate_anchors,
        target_anchors,
        image_size=proposal_config.image_size,
        target_quantization_max=1023.0,
        match_radius_px=proposal_config.atom_match_radius_px,
    )
    return {
        target: {"proposal_index": positions[candidate], "distance_px": distance}
        for target, (candidate, distance) in matches.items()
    }


def _validate_deployed_reproduction(
    sample_key: str,
    atoms,
    batch_index: int,
    deployed_mask,
    expected_rows: list[dict[str, Any]],
) -> None:
    import torch

    positions = torch.where(deployed_mask)[0].tolist()
    expected_positions = [int(row["predicted_atom_index"]) for row in expected_rows]
    if positions != expected_positions:
        raise ValueError(f"Deployed AtomObject indices differ: {sample_key}")
    expected_anchors = {
        int(row["predicted_atom_index"]): np.asarray(row["anchor"], dtype=np.float32)
        for row in expected_rows
    }
    for position in positions:
        actual = (
            atoms["atom_geometry"][batch_index, position, :2].float().cpu().numpy()
        )
        if not np.array_equal(actual, expected_anchors[position]):
            raise ValueError(
                f"Deployed AtomObject anchor differs: {sample_key}:{position}"
            )


def _sample_stage_row(
    mask, matches: dict[int, dict[str, Any]], targets: list[dict[str, Any]]
) -> dict[str, Any]:
    selected = int(mask.sum())
    matched = len(matches)
    target_count = len(targets)
    return {
        "selected": selected,
        "matched": matched,
        "matched_text": sum(targets[index]["visible"] for index in matches),
        "matched_implicit": sum(not targets[index]["visible"] for index in matches),
        "extra_proposals": selected - matched,
        "count_exact": selected == target_count,
        "object_exact": selected == target_count and matched == target_count,
    }


def _target_stage_row(
    *,
    match: dict[str, Any] | None,
    proposals,
    batch_index: int,
    set_evaluated: bool,
    presence_logits=None,
    decoded_mask=None,
) -> dict[str, Any]:
    if match is None:
        return {
            "matched": False,
            "proposal_index": None,
            "proposal_rank": None,
            "spatial_index": None,
            "distance_px": None,
            "proposal_score": None,
            "presence_logit": None,
            "set_evaluated": set_evaluated,
        }
    position = int(match["proposal_index"])
    score = float(proposals["atom_scores"][batch_index, position])
    spatial_index = int(proposals["atom_indices"][batch_index, position])
    logit = None
    if set_evaluated:
        if presence_logits is None or decoded_mask is None:
            raise ValueError("Set-evaluated stage is missing set-head outputs")
        decoded = bool(decoded_mask[position])
        logit = float(presence_logits[batch_index, position]) if decoded else None
    elif presence_logits is not None or decoded_mask is not None:
        raise ValueError("Raw stage must not receive set-head outputs")
    if not math.isfinite(score) or (logit is not None and not math.isfinite(logit)):
        raise ValueError("Reachability stage contains a non-finite score")
    return {
        "matched": True,
        "proposal_index": position,
        "proposal_rank": position,
        "spatial_index": spatial_index,
        "distance_px": float(match["distance_px"]),
        "proposal_score": score,
        "presence_logit": logit,
        "set_evaluated": set_evaluated,
    }


def _stage_summary(
    target_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    stage: str,
    roles: tuple[str, ...],
    variants: list[str],
) -> dict[str, Any]:
    sample_stage = [row["stages"][stage] for row in sample_rows]
    selected = sum(row["selected"] for row in sample_stage)
    return {
        "targets": _matched_rate(target_rows, stage),
        "roles": {
            role: {
                "targets": sum(row["role"] == role for row in target_rows),
                "matched": sum(
                    row["role"] == role and row["stages"][stage]["matched"]
                    for row in target_rows
                ),
                "recall": _matched_rate(
                    [row for row in target_rows if row["role"] == role], stage
                )["rate"],
            }
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
            "count": len(sample_stage),
            "selected_proposals": selected,
            "selected_mean": selected / len(sample_stage) if sample_stage else None,
            "extra_proposals": sum(row["extra_proposals"] for row in sample_stage),
            "count_exact": sum(row["count_exact"] for row in sample_stage),
            "count_exact_rate": (
                sum(row["count_exact"] for row in sample_stage) / len(sample_stage)
                if sample_stage
                else None
            ),
            "object_exact": sum(row["object_exact"] for row in sample_stage),
            "object_exact_rate": (
                sum(row["object_exact"] for row in sample_stage) / len(sample_stage)
                if sample_stage
                else None
            ),
        },
    }


def _matched_rate(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    matched = sum(row["stages"][stage]["matched"] for row in rows)
    return {
        "matched": matched,
        "targets": len(rows),
        "rate": matched / len(rows) if rows else None,
    }


def _threshold_key(value: float) -> str:
    return f"{value:g}"
