"""Post-candidate GT reveal and validation for external OCR R0."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest
from chemvector.external_ocr_r0_recognizers import validate_arm_artifact
from chemvector.external_ocr_r0_selection import (
    validate_fresh_shards,
    validate_selection,
)


def reveal_truth(
    dataset_dir: str | Path,
    arm_paths: dict[str, str | Path],
    output: str | Path,
) -> dict[str, Any]:
    import torch

    from chemvector.data import ChemVectorIterableDataset
    from chemvector.predicted_incidence import PredictedProposalConfig
    from chemvector.train_joint_bond_set_multidomain import FRESH_DEV_SHARDS

    _validate_arm_path_set(arm_paths)
    validate_reveal_inputs(list(FRESH_DEV_SHARDS))
    root, crop_manifest, _ = load_crop_manifest(dataset_dir)
    crop_manifest_path = root / "manifest.json"
    for name, path in arm_paths.items():
        validate_arm_artifact(name, path, crop_manifest_path)

    atom_path = root / crop_manifest["artifacts"]["atom_objects"]["path"]
    atom_rows = contract.read_jsonl(atom_path)
    atoms_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in atom_rows:
        atoms_by_sample[row["sample_key"]].append(row)
    selection_path = Path(crop_manifest["selection"]["path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    selected = set(selection["sample_keys"])
    source = torch.load(
        contract.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    truth_rows = []
    seen = set()
    for sample in ChemVectorIterableDataset(
        FRESH_DEV_SHARDS,
        sequence_version="0.4",
        serialize_target=False,
        load_raster_layers=False,
    ):
        if sample["key"] not in selected:
            continue
        key = sample["key"]
        validate_reveal_sample(sample, selection["sample_metadata"][key])
        seen.add(key)
        targets = rendered_truth_targets(sample)
        predicted_rows = atoms_by_sample.get(key, [])
        target_to_predicted = _match_targets(
            predicted_rows,
            targets,
            proposal_config.atom_match_radius_px,
        )
        for target_index, target in enumerate(targets):
            truth = target["truth"]
            if not target["visible"] or not truth:
                continue
            predicted_index = target_to_predicted.get(target_index)
            predicted_row = (
                predicted_rows[predicted_index]
                if predicted_index is not None
                else None
            )
            object_key = (
                predicted_row["crop_object_key"]
                if predicted_row is not None and predicted_row["selected_as_text"]
                else None
            )
            truth_rows.append(
                {
                    "schema_version": contract.CROP_SCHEMA_VERSION,
                    "gt_key": f"{key}:t{target_index:03d}",
                    "object_key": object_key,
                    "sample_key": key,
                    "molecule_id": selection["sample_metadata"][key]["molecule_id"],
                    "target_index": target_index,
                    "source_atom_id": target["source_atom_id"],
                    "truth": truth,
                    "group": contract.label_group(truth),
                    "matched_atom": predicted_row is not None,
                    "selected_as_text": object_key is not None,
                }
            )
    if seen != selected:
        raise ValueError("GT reveal did not cover the complete frozen selection")
    truth_rows.sort(key=lambda row: row["gt_key"])
    truth_path = Path(output)
    contract.write_new_jsonl(truth_path, truth_rows)
    truth_manifest = {
        "schema_version": contract.TRUTH_SCHEMA_VERSION,
        "audit_id": "external-ocr-r0",
        "phase": "gt_reveal_after_candidate_freeze",
        "crop_manifest": {
            "path": str(crop_manifest_path),
            "sha256": contract.FROZEN_CROP_MANIFEST_SHA256,
        },
        "arm_bundles": dict(contract.FROZEN_ARM_BUNDLE_SHA256),
        "gt_text_objects": len(truth_rows),
        "matched_gt_text_objects": sum(
            row["object_key"] is not None for row in truth_rows
        ),
        "artifact": {
            "path": str(truth_path),
            "sha256": contract.sha256_file(truth_path),
            "rows": len(truth_rows),
        },
    }
    manifest_path = truth_path.with_suffix(".manifest.json")
    contract.write_new_json(manifest_path, truth_manifest)
    return {
        "truth_manifest": str(manifest_path),
        "sha256": contract.sha256_file(manifest_path),
        "gt_text_objects": len(truth_rows),
    }


def validate_reveal_inputs(shards: list[str]) -> None:
    contract.validate_frozen_file(
        contract.SOURCE_CHECKPOINT,
        contract.SOURCE_SHA256,
        "source proposal checkpoint",
    )
    validate_fresh_shards(shards)


def validate_reveal_sample(
    sample: dict[str, Any], expected: dict[str, Any]
) -> None:
    if sample.get("meta", {}).get("molecule_id") != expected["molecule_id"]:
        raise ValueError("GT reveal sample molecule identity differs from selection")
    if sample.get("meta", {}).get("sha256", {}).get("png") != expected["png_sha256"]:
        raise ValueError("GT reveal sample PNG commitment differs from selection")
    if sample.get("layers", {}).get("split") != "fresh_dev":
        raise ValueError("GT reveal sample is not from fresh_dev")


def rendered_truth_targets(sample: dict[str, Any]) -> list[dict[str, Any]]:
    from chemvector.data import rendered_atom_labels

    labels = rendered_atom_labels(sample)
    render_atoms = sorted(
        sample.get("render", {}).get("atoms", []),
        key=lambda atom: int(atom["atom_id"]),
    )
    source_ids = [int(atom["atom_id"]) for atom in render_atoms]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(labels):
        raise ValueError("GT reveal render/layer AtomObject identities differ")
    targets = []
    for target_index, atom in enumerate(render_atoms):
        source_atom_id = int(atom["atom_id"])
        anchor = [atom.get("quantized_x"), atom.get("quantized_y")]
        if len(anchor) != 2 or not all(
            isinstance(value, int) and 0 <= value <= 1023 for value in anchor
        ):
            raise ValueError("GT reveal render AtomObject anchor is invalid")
        text, visible = labels[source_atom_id]
        targets.append(
            {
                "target_index": target_index,
                "source_atom_id": source_atom_id,
                "anchor": anchor,
                "truth": text,
                "visible": visible,
            }
        )
    return targets


def _match_targets(
    predicted_rows: list[dict[str, Any]],
    target_atoms: list[dict[str, Any]],
    match_radius_px: float,
) -> dict[int, int]:
    from scipy.optimize import linear_sum_assignment

    predicted = np.asarray(
        [row["anchor"] for row in predicted_rows], dtype=np.float32
    ).reshape(-1, 2)
    expected = np.asarray(
        [atom["anchor"] for atom in target_atoms], dtype=np.float32
    ).reshape(-1, 2) / 1023.0
    if not len(predicted) or not len(expected):
        return {}
    cost = (
        np.linalg.norm(predicted[:, None, :] - expected[None, :, :], axis=-1)
        * 768.0
    )
    rows, columns = linear_sum_assignment(cost)
    return {
        int(column): int(row)
        for row, column in zip(rows, columns)
        if float(cost[row, column]) <= match_radius_px
    }


def validate_truth_population(
    truth: list[dict[str, Any]],
    manifest: dict[str, Any],
    crops: list[dict[str, Any]],
) -> None:
    if len(truth) != manifest.get("gt_text_objects"):
        raise ValueError("Truth population row count differs from truth manifest")
    gt_keys = [row.get("gt_key") for row in truth]
    if gt_keys != sorted(gt_keys) or len(gt_keys) != len(set(gt_keys)):
        raise ValueError("Truth population keys must be unique and canonically sorted")
    crop_keys = {row["object_key"] for row in crops}
    matched_keys = []
    for row in truth:
        if row.get("schema_version") != contract.CROP_SCHEMA_VERSION:
            raise ValueError("Truth row schema differs")
        truth_text = row.get("truth")
        if not isinstance(truth_text, str) or not truth_text:
            raise ValueError("Truth row must contain non-empty raw UTF-8 text")
        if row.get("group") != contract.label_group(truth_text):
            raise ValueError("Truth row group differs from the frozen grouping function")
        object_key = row.get("object_key")
        if object_key is not None:
            if object_key not in crop_keys:
                raise ValueError("Truth row points outside the crop population")
            matched_keys.append(object_key)
        if bool(row.get("selected_as_text")) != (object_key is not None):
            raise ValueError("Truth row selected_as_text flag differs from object mapping")
    if len(matched_keys) != len(set(matched_keys)):
        raise ValueError("Multiple truth rows map to the same predicted crop")
    if len(matched_keys) != manifest.get("matched_gt_text_objects"):
        raise ValueError("Matched truth count differs from truth manifest")


def load_truth_manifest(
    truth_manifest_path: str | Path,
    crops: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(truth_manifest_path)
    if contract.FROZEN_TRUTH_MANIFEST_SHA256 is None:
        raise RuntimeError("GT reveal manifest has not been frozen")
    if contract.sha256_file(path) != contract.FROZEN_TRUTH_MANIFEST_SHA256:
        raise ValueError("GT reveal manifest SHA256 differs from the frozen protocol")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != contract.TRUTH_SCHEMA_VERSION:
        raise ValueError("GT reveal manifest schema differs")
    if manifest.get("phase") != "gt_reveal_after_candidate_freeze":
        raise ValueError("GT reveal phase differs")
    if (
        manifest.get("crop_manifest", {}).get("sha256")
        != contract.FROZEN_CROP_MANIFEST_SHA256
    ):
        raise ValueError("GT reveal crop commitment differs")
    if manifest.get("arm_bundles") != contract.FROZEN_ARM_BUNDLE_SHA256:
        raise ValueError("GT reveal arm bundle commitments differ")
    truth_path = Path(manifest["artifact"]["path"])
    if contract.sha256_file(truth_path) != manifest["artifact"]["sha256"]:
        raise ValueError("Truth records SHA256 differs")
    truth = contract.read_jsonl(truth_path)
    validate_truth_population(truth, manifest, crops)
    return manifest, truth


def _validate_arm_path_set(arm_paths: dict[str, str | Path]) -> None:
    if set(arm_paths) != set(contract.ARM_NAMES):
        raise ValueError("GT reveal requires exactly the four frozen recognizer arms")
