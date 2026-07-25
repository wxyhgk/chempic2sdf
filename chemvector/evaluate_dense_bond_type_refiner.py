#!/usr/bin/env python3
"""Evaluate a line-band type refiner without changing dense bond geometry."""

from __future__ import annotations

import argparse
import copy
from itertools import islice
import json
from pathlib import Path

from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import mol_from_chemscene_v0_1
from chemvector.bond_object import BOND_TYPES, unordered_endpoint_cost
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
    decode_dense_bond_objects,
)
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.line_band_display import (
    LineBandDisplayConfig,
    LineBandDisplayRefiner,
)
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
)
from chemvector.pretrain_chemscene_bond_query import (
    ChemSceneBondCollator,
    _oriented_endpoint_errors,
    stack_targets,
)
from chemvector.pretrain_dense_bond_object import (
    _forward_detector,
    _scene_from_dense_objects,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train import expand_shards


def pack_predicted_bond_endpoints(
    ranked_batches: list[list[dict]], limits: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack independent predicted bonds without creating all-pairs candidates."""

    if len(ranked_batches) != len(limits):
        raise ValueError("Bond batches and limits differ")
    maximum = max(limits, default=0)
    reference = next(
        (
            obj["endpoints"]
            for objects, limit in zip(ranked_batches, limits)
            for obj in objects[:limit]
        ),
        None,
    )
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = reference.dtype if reference is not None else torch.float32
    points = torch.zeros((len(limits), maximum * 2, 2), device=device, dtype=dtype)
    pair_indices = torch.zeros(
        (len(limits), maximum, 2), device=device, dtype=torch.long
    )
    pair_mask = torch.zeros(
        (len(limits), maximum), device=device, dtype=torch.bool
    )
    for batch_index, (objects, limit) in enumerate(zip(ranked_batches, limits)):
        if limit > len(objects):
            raise ValueError("Refinement limit exceeds decoded bond count")
        for object_index, obj in enumerate(objects[:limit]):
            begin_index = object_index * 2
            points[batch_index, begin_index : begin_index + 2] = obj["endpoints"]
            pair_indices[batch_index, object_index] = torch.tensor(
                (begin_index, begin_index + 1), device=device
            )
        pair_mask[batch_index, :limit] = True
    return points, pair_indices, pair_mask


def reverse_packed_bond_endpoints(
    points: torch.Tensor, limits: list[int]
) -> torch.Tensor:
    """Reverse each independent bond while keeping its pair index unchanged."""

    reversed_points = points.clone()
    for batch_index, limit in enumerate(limits):
        if not limit:
            continue
        current = points[batch_index, : limit * 2].reshape(limit, 2, 2)
        reversed_points[batch_index, : limit * 2] = current.flip(1).reshape(-1, 2)
    return reversed_points


def apply_type_predictions(
    ranked_batches: list[list[dict]],
    limits: list[int],
    display_logits: torch.Tensor,
    chemical_logits: torch.Tensor,
    *,
    replace_display: bool = True,
    replace_chemical: bool = True,
) -> list[list[dict]]:
    """Replace only display and chemical classes in ranked bond objects."""

    expected = sum(limits)
    if len(display_logits) != expected or len(chemical_logits) != expected:
        raise ValueError("Refiner output count differs from packed bonds")
    refined: list[list[dict]] = []
    offset = 0
    for objects, limit in zip(ranked_batches, limits):
        current = [dict(obj) for obj in objects]
        for object_index in range(limit):
            if replace_display:
                current[object_index]["display"] = int(
                    display_logits[offset + object_index].argmax()
                )
            if replace_chemical:
                current[object_index]["chemical"] = int(
                    chemical_logits[offset + object_index].argmax()
                )
        refined.append(current)
        offset += limit
    return refined


def apply_visible_multiple_overrides(
    ranked_batches: list[list[dict]],
    limits: list[int],
    display_logits: torch.Tensor,
    confidence_threshold: float,
) -> list[list[dict]]:
    """Use line evidence only for confident, directly visible multiple bonds."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must lie in [0, 1]")
    if len(display_logits) != sum(limits):
        raise ValueError("Refiner output count differs from packed bonds")
    probabilities = display_logits.float().softmax(-1)
    confidence, predictions = probabilities.max(-1)
    double_id = BOND_TYPES.index("DOUBLE")
    triple_id = BOND_TYPES.index("TRIPLE")
    aromatic_id = BOND_TYPES.index("AROMATIC")
    refined: list[list[dict]] = []
    offset = 0
    for objects, limit in zip(ranked_batches, limits):
        current = [dict(obj) for obj in objects]
        for object_index in range(limit):
            flat_index = offset + object_index
            prediction = int(predictions[flat_index])
            if (
                prediction in {double_id, triple_id}
                and float(confidence[flat_index]) >= confidence_threshold
                and current[object_index]["display"] != aromatic_id
            ):
                current[object_index]["display"] = prediction
        refined.append(current)
        offset += limit
    return refined


def _empty_totals() -> dict:
    return {
        "images": 0,
        "bonds": 0,
        "geometry_correct": 0,
        "display_correct_given_geometry": 0,
        "chemical_correct_given_geometry": 0,
        "object_correct": 0,
        "geometry_exact": 0,
        "known_object_exact": 0,
        "predicted_count_exact": 0,
        "predicted_object_exact": 0,
        "oracle_junction_sdf_exact": 0,
        "by_display_type": {
            name: {
                "count": 0,
                "geometry_correct": 0,
                "display_correct_given_geometry": 0,
                "chemical_correct_given_geometry": 0,
                "object_correct": 0,
            }
            for name in BOND_TYPES
        },
    }


def _update_known_metrics(
    totals: dict,
    objects: list[dict],
    targets: dict[str, torch.Tensor],
    batch_index: int,
) -> None:
    valid = targets["mask"][batch_index]
    expected = targets["endpoints"][batch_index, valid]
    expected_indices = torch.where(valid)[0]
    count = len(expected)
    totals["images"] += 1
    totals["bonds"] += count
    for target_index in expected_indices:
        class_name = BOND_TYPES[int(targets["display"][batch_index, target_index])]
        totals["by_display_type"][class_name]["count"] += 1
    if len(objects) != count or not count:
        return

    predicted = torch.stack([obj["endpoints"] for obj in objects])
    rows, columns = linear_sum_assignment(
        unordered_endpoint_cost(predicted, expected).float().cpu().numpy()
    )
    geometry_hits = 0
    object_hits = 0
    for row, column in zip(rows, columns):
        target_index = expected_indices[column]
        errors = _oriented_endpoint_errors(predicted[row], expected[column], 768.0)
        geometry_correct = bool((errors <= 6.0).all())
        display_correct = objects[row]["display"] == int(
            targets["display"][batch_index, target_index]
        )
        chemical_correct = objects[row]["chemical"] == int(
            targets["chemical"][batch_index, target_index]
        )
        class_name = BOND_TYPES[int(targets["display"][batch_index, target_index])]
        bucket = totals["by_display_type"][class_name]
        if geometry_correct:
            geometry_hits += 1
            totals["geometry_correct"] += 1
            bucket["geometry_correct"] += 1
            if display_correct:
                totals["display_correct_given_geometry"] += 1
                bucket["display_correct_given_geometry"] += 1
            if chemical_correct:
                totals["chemical_correct_given_geometry"] += 1
                bucket["chemical_correct_given_geometry"] += 1
            if display_correct and chemical_correct:
                object_hits += 1
                totals["object_correct"] += 1
                bucket["object_correct"] += 1
    totals["geometry_exact"] += int(geometry_hits == count)
    totals["known_object_exact"] += int(object_hits == count)


def _update_predicted_metrics(
    totals: dict,
    objects: list[dict],
    oracle_scene: dict,
    expected_inchikey: str,
    targets: dict[str, torch.Tensor],
    batch_index: int,
) -> None:
    count = int(targets["mask"][batch_index].sum())
    totals["predicted_count_exact"] += int(len(objects) == count)
    if len(objects) != count or not count:
        return
    expected = targets["endpoints"][batch_index, targets["mask"][batch_index]]
    predicted = torch.stack([obj["endpoints"] for obj in objects])
    rows, columns = linear_sum_assignment(
        unordered_endpoint_cost(predicted, expected).float().cpu().numpy()
    )
    valid_indices = torch.where(targets["mask"][batch_index])[0]
    exact = True
    for row, column in zip(rows, columns):
        target_index = valid_indices[column]
        errors = _oriented_endpoint_errors(predicted[row], expected[column], 768.0)
        exact = exact and bool((errors <= 6.0).all())
        exact = exact and objects[row]["display"] == int(
            targets["display"][batch_index, target_index]
        )
        exact = exact and objects[row]["chemical"] == int(
            targets["chemical"][batch_index, target_index]
        )
    totals["predicted_object_exact"] += int(exact)
    if not exact:
        return
    predicted_scene = _scene_from_dense_objects(oracle_scene, objects)
    if predicted_scene is None:
        return
    try:
        predicted_inchikey = Chem.MolToInchiKey(
            mol_from_chemscene_v0_1(predicted_scene)
        )
    except Exception:
        return
    totals["oracle_junction_sdf_exact"] += int(predicted_inchikey == expected_inchikey)


def _finalize(totals: dict, threshold: float) -> dict:
    images = max(totals["images"], 1)
    bonds = max(totals["bonds"], 1)
    geometry = max(totals["geometry_correct"], 1)
    result = {
        "samples": totals["images"],
        "bonds": totals["bonds"],
        "threshold": threshold,
        "geometry_accuracy": totals["geometry_correct"] / bonds,
        "display_accuracy_given_geometry": (
            totals["display_correct_given_geometry"] / geometry
        ),
        "chemical_accuracy_given_geometry": (
            totals["chemical_correct_given_geometry"] / geometry
        ),
        "known_object_accuracy": totals["object_correct"] / bonds,
        "geometry_exact": totals["geometry_exact"] / images,
        "known_object_exact": totals["known_object_exact"] / images,
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "predicted_object_exact": totals["predicted_object_exact"] / images,
        "oracle_junction_sdf_exact": totals["oracle_junction_sdf_exact"] / images,
        "by_gt_display_type": {},
    }
    for name, values in totals["by_display_type"].items():
        count = max(values["count"], 1)
        geometry_count = max(values["geometry_correct"], 1)
        result["by_gt_display_type"][name] = {
            "count": values["count"],
            "geometry_recall": values["geometry_correct"] / count,
            "display_accuracy_given_geometry": (
                values["display_correct_given_geometry"] / geometry_count
            ),
            "chemical_accuracy_given_geometry": (
                values["chemical_correct_given_geometry"] / geometry_count
            ),
            "object_recall": values["object_correct"] / count,
        }
    return result


def _load_dense_pipeline(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, graph_payload = load_grounded_graph_checkpoint(
        payload["grounded_graph_checkpoint"], device
    )
    base, _, _, _ = load_frozen_grounded_pipeline(
        Path(graph_payload["grounded_selector_checkpoint"]), device
    )
    base.eval()
    config = DenseBondObjectConfig(**payload["config"])
    state = payload["dense_bond_object"]
    if any(key.startswith("base.") for key in state):
        detector = MultiScaleDenseBondObjectDetector(config).to(device)
    else:
        detector = DenseBondObjectDetector(config).to(device)
    detector.load_state_dict(state)
    detector.eval()
    return base, detector


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--type-refiner-checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda")
    base, detector = _load_dense_pipeline(Path(args.dense_checkpoint), device)
    refiner_payload = torch.load(
        args.type_refiner_checkpoint, map_location="cpu", weights_only=False
    )
    display_only_refiner = "line_band_display" in refiner_payload
    if display_only_refiner:
        refiner = LineBandDisplayRefiner(
            LineBandDisplayConfig(**refiner_payload["config"])
        ).to(device)
        refiner.load_state_dict(refiner_payload["line_band_display"])
    else:
        refiner = NodeConditionedBondObject(
            NodeConditionedBondConfig(**refiner_payload["config"])
        ).to(device)
        refiner.load_state_dict(refiner_payload["node_conditioned_bond"])
    refiner.eval()

    records = list(
        islice(
            iter(
                ChemVectorIterableDataset(
                    expand_shards(args.validation_shards), sequence_version="0.4"
                )
            ),
            args.samples,
        )
    )
    if len(records) != args.samples:
        raise ValueError("Validation dataset contains fewer samples than requested")
    collator = ChemSceneBondCollator()
    variants = {
        "dense_baseline": _empty_totals(),
        "line_band_display_only": _empty_totals(),
    }
    if not display_only_refiner:
        variants["line_band_forward"] = _empty_totals()
        variants["line_band_symmetric"] = _empty_totals()
        variants["line_band_chemical_only"] = _empty_totals()
    override_thresholds = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    for confidence_threshold in override_thresholds:
        variants[f"visible_multiple_override_{confidence_threshold:.2f}"] = _empty_totals()

    for start in range(0, len(records), args.batch_size):
        batch = collator(records[start : start + args.batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            dense_output = _forward_detector(base, detector, pixels)
        ranked_batches = decode_dense_bond_objects(
            dense_output, threshold=0.0, maximum_objects=128
        )
        limits = []
        for batch_index, ranked in enumerate(ranked_batches):
            known_count = int(targets["mask"][batch_index].sum())
            predicted_count = sum(obj["score"] >= args.threshold for obj in ranked)
            limits.append(max(known_count, predicted_count))

        points, pair_indices, pair_mask = pack_predicted_bond_endpoints(
            ranked_batches, limits
        )
        reversed_points = reverse_packed_bond_endpoints(points, limits)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            forward_output = refiner(pixels, points, pair_indices, pair_mask)
            reverse_output = refiner(
                pixels, reversed_points, pair_indices, pair_mask
            )
        symmetric_display = (
            forward_output["display_logits"] + reverse_output["display_logits"]
        ) / 2.0
        symmetric_chemical = None
        if not display_only_refiner:
            symmetric_chemical = (
                forward_output["chemical_logits"]
                + reverse_output["chemical_logits"]
            ) / 2.0
        refined_batches = {
            "dense_baseline": ranked_batches,
            "line_band_display_only": apply_type_predictions(
                ranked_batches,
                limits,
                symmetric_display,
                symmetric_display,
                replace_chemical=False,
            ),
        }
        if not display_only_refiner:
            refined_batches["line_band_forward"] = apply_type_predictions(
                ranked_batches,
                limits,
                forward_output["display_logits"],
                forward_output["chemical_logits"],
            )
            refined_batches["line_band_symmetric"] = apply_type_predictions(
                ranked_batches, limits, symmetric_display, symmetric_chemical
            )
            refined_batches["line_band_chemical_only"] = apply_type_predictions(
                ranked_batches,
                limits,
                symmetric_display,
                symmetric_chemical,
                replace_display=False,
            )
        for confidence_threshold in override_thresholds:
            refined_batches[
                f"visible_multiple_override_{confidence_threshold:.2f}"
            ] = apply_visible_multiple_overrides(
                ranked_batches,
                limits,
                symmetric_display,
                confidence_threshold,
            )

        for name, current_batches in refined_batches.items():
            totals = variants[name]
            for batch_index, ranked in enumerate(current_batches):
                count = int(targets["mask"][batch_index].sum())
                known_objects = ranked[:count]
                predicted_objects = [
                    obj for obj in ranked if obj["score"] >= args.threshold
                ]
                _update_known_metrics(totals, known_objects, targets, batch_index)
                _update_predicted_metrics(
                    totals,
                    predicted_objects,
                    batch["scenes"][batch_index],
                    batch["expected_inchikeys"][batch_index],
                    targets,
                    batch_index,
                )
        completed = min(start + args.batch_size, len(records))
        if completed % 100 == 0 or completed == len(records):
            print(f"evaluated={completed}/{len(records)}", flush=True)

    result = {
        "protocol": {
            "dense_checkpoint": args.dense_checkpoint,
            "type_refiner_checkpoint": args.type_refiner_checkpoint,
            "samples": args.samples,
            "threshold": args.threshold,
            "geometry_and_count_frozen": True,
            "refined_fields": (
                ["display"] if display_only_refiner else ["display", "chemical"]
            ),
            "display_only_refiner": display_only_refiner,
            "endpoint_order_ensemble": "mean forward and reverse logits",
        },
        "variants": {
            name: _finalize(totals, args.threshold)
            for name, totals in variants.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
