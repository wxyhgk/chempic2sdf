#!/usr/bin/env python3
"""Audit regional-max Atom objects from the frozen P1-A spatial field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch

from chemvector.components.atoms.g3_component_set_ranker import (
    build_target_free_components,
    select_ranked_components,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
)
from chemvector.components.vision.regional_maxima import regional_maxima_mask
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1n-regional-maxima-zero-train"
)
MAXIMUM_OBJECTS = 128
MATCH_RADIUS_PIXELS = 6.0
IMAGE_SIZE = 768.0


def collapse_slots_to_cell_field(dense: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Choose one slot per cell without averaging object coordinates."""

    required = {"presence_logits", "anchors"}
    if required - dense.keys():
        raise ValueError("P1-N dense output is missing Atom fields")
    logits = dense["presence_logits"]
    anchors = dense["anchors"]
    if logits.ndim != 4:
        raise ValueError("P1-N presence logits must have shape [batch,slots,h,w]")
    batch, slots, height, width = logits.shape
    if anchors.shape != (batch, slots, 2, height, width):
        raise ValueError("P1-N Atom anchors and logits differ")
    scores, slot_ids = logits.float().max(dim=1)
    gather = slot_ids[:, None, None].expand(batch, 1, 2, height, width)
    cell_anchors = anchors.float().gather(1, gather).squeeze(1)
    cell_ids = torch.arange(
        height * width, device=logits.device, dtype=torch.long
    ).reshape(1, height, width)
    source_ids = slot_ids.long() * (height * width) + cell_ids
    return {
        "scores": scores,
        "anchors": cell_anchors,
        "slot_ids": slot_ids,
        "source_ids": source_ids,
    }


def extract_regional_maximum_objects(
    field: dict[str, torch.Tensor],
    *,
    maximum_objects: int = MAXIMUM_OBJECTS,
) -> list[dict[str, torch.Tensor | list[int] | int]]:
    """Collapse every regional-maximum plateau to one deterministic Atom object."""

    if maximum_objects < 1:
        raise ValueError("P1-N maximum object count must be positive")
    scores = field["scores"]
    anchors = field["anchors"]
    source_ids = field["source_ids"]
    if scores.dtype != torch.float32 or scores.ndim != 3:
        raise ValueError("P1-N cell scores must be FP32 [batch,h,w]")
    batch, height, width = scores.shape
    if anchors.shape != (batch, 2, height, width):
        raise ValueError("P1-N cell anchors and scores differ")
    if source_ids.shape != scores.shape:
        raise ValueError("P1-N source identities and scores differ")

    masks = regional_maxima_mask(scores, connectivity=8).cpu().numpy()
    scores_cpu = scores.detach().cpu().numpy()
    anchors_cpu = anchors.detach().cpu().permute(0, 2, 3, 1).numpy()
    source_cpu = source_ids.detach().cpu().numpy()
    structure = np.ones((3, 3), dtype=np.uint8)
    outputs = []
    for batch_index in range(batch):
        labels, plateau_count = ndimage.label(masks[batch_index], structure=structure)
        objects = []
        flat_labels = labels.reshape(-1)
        active_cells = np.flatnonzero(flat_labels)
        active_labels = flat_labels[active_cells]
        plateau_sizes = np.bincount(
            active_labels, minlength=plateau_count + 1
        )[1:].astype(np.int64, copy=False)
        representatives = np.full(plateau_count, height * width, dtype=np.int64)
        np.minimum.at(representatives, active_labels - 1, active_cells)
        if (
            len(plateau_sizes) != plateau_count
            or bool((plateau_sizes < 1).any())
            or bool((representatives >= height * width).any())
        ):
            raise RuntimeError("P1-N produced an empty regional-max plateau")
        for plateau_index, representative in enumerate(representatives):
            # A regional maximum is an equal-valued flat zone. Row-major cell ID
            # is the deterministic representative and does not inspect targets.
            flat_cell = int(representative)
            y, x = divmod(flat_cell, width)
            objects.append(
                (
                    float(scores_cpu[batch_index, y, x]),
                    int(source_cpu[batch_index, y, x]),
                    anchors_cpu[batch_index, y, x].copy(),
                    int(plateau_sizes[plateau_index]),
                )
            )
        objects.sort(key=lambda item: (-item[0], item[1]))
        selected = objects[:maximum_objects]
        outputs.append(
            {
                "anchors": torch.tensor(
                    np.asarray([item[2] for item in selected], dtype=np.float32)
                    if selected
                    else np.zeros((0, 2), dtype=np.float32)
                ),
                "scores": torch.tensor([item[0] for item in selected]),
                "source_ids": torch.tensor(
                    [item[1] for item in selected], dtype=torch.long
                ),
                "plateau_sizes": [item[3] for item in selected],
                "all_plateau_sizes": plateau_sizes.tolist(),
                "regional_maxima_count": len(objects),
            }
        )
    return outputs


def _empty_mode_totals() -> dict:
    return p1a._empty_atom_totals()


def _finalize_mode(totals: dict) -> dict:
    result = p1a._finalize_atom_totals(totals)
    result["exact_rate_6px"] = result["whole_set_exact_images"] / max(
        result["images"], 1
    )
    result["count_exact_rate"] = result["count_exact_images"] / max(
        result["images"], 1
    )
    return result


def _sample_collision(anchors: torch.Tensor, *, grid_size: int = 192) -> bool:
    cells = (anchors * grid_size).clamp(0, grid_size - 1e-4).long()
    identities = cells[:, 1] * grid_size + cells[:, 0]
    return len(identities.unique()) != len(identities)


def _compatibility_statistics(
    predicted: torch.Tensor, expected: torch.Tensor
) -> tuple[int, int]:
    if not len(predicted) or not len(expected):
        return 0, 0
    compatible = torch.cdist(expected.float(), predicted.float()) * IMAGE_SIZE <= MATCH_RADIUS_PIXELS
    multiple_peaks = int((compatible.sum(dim=1) >= 2).sum())
    shared_peaks = int((compatible.sum(dim=0) >= 2).sum())
    return multiple_peaks, shared_peaks


def _percentile(values: list[int], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


@torch.no_grad()
def evaluate_manifest(atom_head, vision, manifest, collator, *, device, limit=None):
    modes = {
        "regional_top128_full": _empty_mode_totals(),
        "regional_known_full": _empty_mode_totals(),
        "regional_known_representable": _empty_mode_totals(),
        "regional_deploy_full": _empty_mode_totals(),
        "component_known_representable": _empty_mode_totals(),
    }
    occupancy = {
        "ordinary": _empty_mode_totals(),
        "same_cell_collision": _empty_mode_totals(),
    }
    variants: dict[str, dict] = {}
    sample_digest = hashlib.sha256()
    peak_digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    maxima_counts: list[int] = []
    plateau_sizes: list[int] = []
    peak_shortfall_full = 0
    peak_shortfall_deploy = 0
    multiple_peak_targets = 0
    shared_peaks = 0
    compatible_targets = 0
    reachability_targets = 0
    reachability_matched = 0
    reachability_exact_images = 0
    rescue = break_count = both_exact = both_wrong = 0
    images = 0

    selected_manifest = manifest if limit is None else manifest[:limit]
    for batch in p1a.iter_manifest_batches(selected_manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device),
            "atom_mask": batch["atom_mask"].to(device),
        }
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
        # Regional-max topology is evaluated in FP32. BF16 forward remains fixed
        # to preserve the frozen P1-A inference lineage.
        field = collapse_slots_to_cell_field(dense)
        peaks = extract_regional_maximum_objects(field)
        fixed = build_hybrid_top128_targets(candidates, targets)
        components = build_target_free_components(candidates)
        baseline_ranked = {**components, "ranking_logits": components["base_logits"]}
        component_selected = select_ranked_components(
            baseline_ranked, fixed["representable_counts"].long()
        )

        for index, peak in enumerate(peaks):
            images += 1
            all_expected = targets["atom_anchors"][index, targets["atom_mask"][index]].float().cpu()
            fixed_mask = fixed["positive_mask"][index]
            fixed_indices = fixed["matched_target_indices"][index, fixed_mask]
            representable_expected = targets["atom_anchors"][index, fixed_indices].float().cpu()
            peak_anchors = peak["anchors"]
            full_count = len(all_expected)
            representable_count = len(representable_expected)
            deploy_count = min(
                int(dense["count_logits"][index].float().argmax()), MAXIMUM_OBJECTS
            )
            maxima_count = int(peak["regional_maxima_count"])
            maxima_counts.append(maxima_count)
            plateau_sizes.extend(peak["all_plateau_sizes"])
            if maxima_count < full_count:
                peak_shortfall_full += 1
            if maxima_count < deploy_count:
                peak_shortfall_deploy += 1

            predictions = {
                "regional_top128_full": peak_anchors,
                "regional_known_full": peak_anchors[:full_count],
                "regional_known_representable": peak_anchors[:representable_count],
                "regional_deploy_full": peak_anchors[:deploy_count],
            }
            component_mask = component_selected["atom_mask"][index]
            predictions["component_known_representable"] = component_selected[
                "atom_anchors"
            ][index, component_mask].float().cpu()
            expected = {
                "regional_top128_full": all_expected,
                "regional_known_full": all_expected,
                "regional_known_representable": representable_expected,
                "regional_deploy_full": all_expected,
                "component_known_representable": representable_expected,
            }
            current = {}
            for name in modes:
                current[name] = p1a.atom_sample_statistics(
                    predictions[name], expected[name]
                )
                p1a._merge_atom_totals(modes[name], current[name])
            top128_current = current["regional_top128_full"]
            reachability_targets += int(top128_current["target_atoms"])
            reachability_matched += int(top128_current["matched_atoms"])
            reachability_exact_images += int(
                top128_current["matched_atoms"] == top128_current["target_atoms"]
            )

            collision_bucket = (
                "same_cell_collision"
                if _sample_collision(all_expected)
                else "ordinary"
            )
            p1a._merge_atom_totals(
                occupancy[collision_bucket], current["regional_known_full"]
            )
            variant = str(batch["variants"][index])
            variants.setdefault(variant, _empty_mode_totals())
            p1a._merge_atom_totals(
                variants[variant], current["regional_known_full"]
            )
            multi, shared = _compatibility_statistics(
                peak_anchors, all_expected
            )
            multiple_peak_targets += multi
            shared_peaks += shared
            compatible_targets += full_count

            regional_exact = bool(current["regional_known_representable"]["whole_set_exact_images"])
            component_exact = bool(current["component_known_representable"]["whole_set_exact_images"])
            if regional_exact and component_exact:
                both_exact += 1
            elif regional_exact:
                rescue += 1
            elif component_exact:
                break_count += 1
            else:
                both_wrong += 1
            peak_digest.update(peak["source_ids"].numpy().tobytes())
            selected_digest.update(
                peak["source_ids"][:full_count].numpy().tobytes()
            )

    if not images:
        raise RuntimeError("P1-N evaluated no images")
    maxima_array = np.asarray(maxima_counts, dtype=np.float64)
    plateau_array = np.asarray(plateau_sizes, dtype=np.float64)
    return {
        "images": images,
        "modes": {name: _finalize_mode(value) for name, value in modes.items()},
        "occupancy": {
            name: _finalize_mode(value) for name, value in occupancy.items()
        },
        "variants": {
            name: _finalize_mode(value) for name, value in sorted(variants.items())
        },
        "regional_maxima": {
            "mean_per_image": float(maxima_array.mean()),
            "p50_per_image": _percentile(maxima_counts, 50),
            "p90_per_image": _percentile(maxima_counts, 90),
            "maximum_per_image": int(max(maxima_counts)),
            "images_above_128": int((maxima_array > MAXIMUM_OBJECTS).sum()),
            "peak_shortfall_full_count_images": peak_shortfall_full,
            "peak_shortfall_deploy_count_images": peak_shortfall_deploy,
            "plateau_mean_cells": float(plateau_array.mean()),
            "plateau_p90_cells": _percentile(plateau_sizes, 90),
            "plateau_maximum_cells": int(max(plateau_sizes)),
            "multiple_peak_compatible_targets": multiple_peak_targets,
            "multiple_peak_compatible_target_rate": multiple_peak_targets
            / max(compatible_targets, 1),
            "shared_by_multiple_targets_peaks": shared_peaks,
        },
        "top128_reachability": {
            "targets": reachability_targets,
            "matched_targets": reachability_matched,
            "atom_recall": reachability_matched / max(reachability_targets, 1),
            "exact_images": reachability_exact_images,
            "exact_rate": reachability_exact_images / images,
        },
        "paired_component_baseline": {
            "both_exact": both_exact,
            "regional_rescue": rescue,
            "regional_break": break_count,
            "both_wrong": both_wrong,
        },
        "sample_keys_sha256": sample_digest.hexdigest(),
        "peak_source_ids_sha256": peak_digest.hexdigest(),
        "known_full_source_ids_sha256": selected_digest.hexdigest(),
    }


def _decision(train: dict) -> tuple[str, dict[str, bool]]:
    full_top = train["top128_reachability"]
    known = train["modes"]["regional_known_representable"]
    baseline = train["modes"]["component_known_representable"]
    checks = {
        "top128_full_recall_99_7": full_top["atom_recall"] >= 0.997,
        "top128_full_exact_95": full_top["exact_rate"] >= 0.95,
        "known_exact_plus_5pp": known["exact_rate_6px"]
        >= baseline["exact_rate_6px"] + 0.05,
        "known_recall_regression_le_0_2pp": known["recall_6px"]
        >= baseline["recall_6px"] - 0.002,
        "no_full_count_peak_shortfall": train["regional_maxima"][
            "peak_shortfall_full_count_images"
        ]
        == 0,
    }
    return (
        "mechanism_go_requires_fresh_confirmation"
        if all(checks.values())
        else "mechanism_no_go",
        checks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "route", "both"), default="both")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("P1-N limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-N frozen replay requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-N output directory is not empty")

    device = torch.device("cuda")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    atom_head.requires_grad_(False).eval()
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    results = {}
    if args.split in {"train", "both"}:
        results["train5k"] = evaluate_manifest(
            atom_head, vision, train, collator, device=device, limit=args.limit
        )
    if args.split in {"route", "both"}:
        results["route100"] = evaluate_manifest(
            atom_head, vision, route, collator, device=device, limit=args.limit
        )
    if "train5k" in results and args.limit is None:
        decision, checks = _decision(results["train5k"])
    else:
        decision, checks = "diagnostic_only", {}
    report = {
        "schema_version": "g3-p1n-regional-maxima-zero-train-audit-0.1",
        "scope": "frozen-p1a-cell-max-true-regional-maxima",
        "operator": {
            "slot_reduction": "maximum_logit_per_cell_smallest_slot_tie",
            "regional_maxima": "complete_equal_plateau_8_connectivity",
            "plateau_representative": "smallest_row_major_cell",
            "ranking": "logit_desc_source_id_asc",
            "maximum_objects": MAXIMUM_OBJECTS,
            "absolute_threshold": None,
            "pixel_cluster_radius": None,
            "background_backfill": False,
        },
        "lineage": {
            "p1a_checkpoint": p1a_payload["lineage"]["p1a"]
            if "p1a" in p1a_payload.get("lineage", {})
            else str(p1d.P1A_CHECKPOINT.resolve()),
            "atom_state_sha256": module_state_sha256(atom_head),
            "vision_checkpoint_sha256": vision_sha,
            "vision_state_sha256": module_state_sha256(vision),
            "train_manifest_sha256": train_audit["sha256"],
            "route_manifest_sha256": route_audit["sha256"],
        },
        "protocol": {
            "updates": 0,
            "optimizer": None,
            "match_radius_pixels_evaluation_only": MATCH_RADIUS_PIXELS,
            "route100_status": "previously_consumed_historical_diagnostic",
            "fresh_confirmation_required": True,
            "forbidden_data_read": False,
            "split": args.split,
            "limit": args.limit,
        },
        "results": results,
        "checks": checks,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": decision, "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
