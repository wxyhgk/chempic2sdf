"""Predicted AtomObject crop production and validation for external OCR R0."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_selection import (
    validate_fresh_shards,
    validate_selection,
)


def prepare_predicted_crops(
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int | str]:
    import torch

    from chemvector.ctc_ocr import bbox_crop
    from chemvector.data import ChemVectorIterableDataset
    from chemvector.direct_atom_role import enumerate_selected_atoms
    from chemvector.evaluate_direct_atom_ocr import load_bbox, load_role
    from chemvector.image_io import image_tensor
    from chemvector.predicted_incidence import PredictedProposalConfig
    from chemvector.train_bond_presence_calibration import load_components
    from chemvector.train_direct_atom_pair_bond import frozen_atom_objects
    from chemvector.train_joint_bond_set_multidomain import FRESH_DEV_SHARDS

    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    selected = set(selection["sample_keys"])
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    (output / "crops").mkdir(parents=True, exist_ok=True)

    frozen = (
        (contract.ROLE_CHECKPOINT, contract.ROLE_SHA256, "role checkpoint"),
        (contract.BBOX_CHECKPOINT, contract.BBOX_SHA256, "bbox checkpoint"),
        (contract.CTC_CHECKPOINT, contract.V0109_CTC_SHA256, "CTC checkpoint"),
        (contract.SOURCE_CHECKPOINT, contract.SOURCE_SHA256, "source proposal checkpoint"),
        (
            contract.GROUNDED_GRAPH_CHECKPOINT,
            contract.GROUNDED_GRAPH_SHA256,
            "grounded graph checkpoint",
        ),
    )
    for path, expected_sha, name in frozen:
        contract.validate_frozen_file(path, expected_sha, name)
    validate_fresh_shards(list(FRESH_DEV_SHARDS))
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen R0 crop producer requires CUDA")
    device = torch.device("cuda")
    role, role_payload = load_role(str(contract.ROLE_CHECKPOINT), device)
    bbox, _ = load_bbox(str(contract.BBOX_CHECKPOINT), device)
    role_args = role_payload["training_args"]
    if Path(role_args["source_checkpoint"]) != contract.SOURCE_CHECKPOINT:
        raise ValueError("Role checkpoint points to a different source proposal checkpoint")
    if Path(role_args["grounded_graph_checkpoint"]) != contract.GROUNDED_GRAPH_CHECKPOINT:
        raise ValueError("Role checkpoint points to a different grounded graph checkpoint")
    vision, atom, bond, set_head, source = load_components(
        role_args["source_checkpoint"],
        role_args["grounded_graph_checkpoint"],
        device,
    )
    for model in (role, bbox, vision, atom, bond, set_head):
        model.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    role_threshold = float(role_payload["validation"]["threshold"])
    crop_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def consume(samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        pixels = torch.stack(
            [image_tensor(sample["image"]) for sample in samples]
        ).to(device)
        _, atoms = frozen_atom_objects(
            vision,
            atom,
            bond,
            set_head,
            pixels,
            proposal_config,
            atom_threshold,
        )
        atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            role_probabilities = role(
                pixels,
                atoms["atom_geometry"][..., :2],
                atom_indices,
                sample_indices,
            ).sigmoid()
        text_rows = torch.where(role_probabilities >= role_threshold)[0]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            boxes = bbox(
                pixels,
                atoms["atom_geometry"][..., :2],
                atom_indices[text_rows],
                sample_indices[text_rows],
            )
        box_lookup = {
            (int(sample_indices[row]), int(atom_indices[row])): (
                boxes["centers"][index].float().cpu() * 768.0,
                boxes["half_extents"][index].float().cpu() * 768.0,
                float(role_probabilities[row]),
            )
            for index, row in enumerate(text_rows.tolist())
        }
        images = [
            sample["image"].convert("RGB").resize(
                (768, 768), Image.Resampling.BICUBIC
            )
            for sample in samples
        ]
        metadata = selection["sample_metadata"]
        for batch_index, sample in enumerate(samples):
            key = sample["key"]
            for atom_index in torch.where(atoms["atom_mask"][batch_index])[0].tolist():
                box_item = box_lookup.get((batch_index, atom_index))
                atom_object_key = f"{key}:p{atom_index:03d}"
                anchor = (
                    atoms["atom_geometry"][batch_index, atom_index, :2]
                    .float()
                    .cpu()
                    .tolist()
                )
                atom_rows.append(
                    {
                        "schema_version": contract.CROP_SCHEMA_VERSION,
                        "atom_object_key": atom_object_key,
                        "sample_key": key,
                        "molecule_id": metadata[key]["molecule_id"],
                        "predicted_atom_index": atom_index,
                        "anchor": anchor,
                        "selected_as_text": box_item is not None,
                        "crop_object_key": (
                            atom_object_key if box_item is not None else None
                        ),
                    }
                )
                if box_item is None:
                    continue
                center, extent, role_score = box_item
                center_x, center_y = center.tolist()
                half_width, half_height = extent.tolist()
                bbox_values = (
                    center_x - half_width,
                    center_y - half_height,
                    center_x + half_width,
                    center_y + half_height,
                )
                if not all(math.isfinite(value) for value in bbox_values):
                    raise ValueError(f"Non-finite predicted bbox for {key}:{atom_index}")
                if bbox_values[2] <= bbox_values[0] or bbox_values[3] <= bbox_values[1]:
                    raise ValueError(f"Degenerate predicted bbox for {key}:{atom_index}")
                relative = Path("crops") / f"{atom_object_key}.png"
                crop = bbox_crop(images[batch_index], bbox_values, context=4.0)
                crop.save(output / relative)
                crop_rows.append(
                    {
                        "schema_version": contract.CROP_SCHEMA_VERSION,
                        "object_key": atom_object_key,
                        "sample_key": key,
                        "molecule_id": metadata[key]["molecule_id"],
                        "variant": metadata[key]["variant"],
                        "font_family_group": metadata[key]["font_family_group"],
                        "predicted_atom_index": atom_index,
                        "role_score": role_score,
                        "bbox": {
                            "source": "predicted_v0108",
                            "x0": bbox_values[0],
                            "y0": bbox_values[1],
                            "x1": bbox_values[2],
                            "y1": bbox_values[3],
                            "context": 4.0,
                        },
                        "crop": {
                            "path": str(relative),
                            "sha256": contract.sha256_file(output / relative),
                            "width": crop.width,
                            "height": crop.height,
                        },
                    }
                )

    pending = []
    for sample in ChemVectorIterableDataset(
        FRESH_DEV_SHARDS,
        sequence_version="0.4",
        serialize_target=False,
        load_raster_layers=False,
    ):
        if sample["key"] not in selected:
            continue
        expected_sample = selection["sample_metadata"][sample["key"]]
        if sample["meta"].get("molecule_id") != expected_sample["molecule_id"]:
            raise ValueError("Runtime sample molecule identity differs from selection")
        if sample["meta"].get("sha256", {}).get("png") != expected_sample["png_sha256"]:
            raise ValueError("Runtime sample PNG commitment differs from selection")
        if sample.get("layers", {}).get("split") != "fresh_dev":
            raise ValueError("Runtime sample is not from fresh_dev")
        seen.add(sample["key"])
        pending.append(sample)
        if len(pending) == 6:
            consume(pending)
            pending = []
            if len(seen) % 100 < 6:
                print(f"prepared_images={len(seen)}/{len(selected)}", flush=True)
    consume(pending)
    if seen != selected:
        missing = sorted(selected - seen)[:10]
        raise ValueError(f"Selection samples were not found or were rejected: {missing}")

    crop_rows.sort(key=lambda row: row["object_key"])
    atom_rows.sort(key=lambda row: row["atom_object_key"])
    contract.write_new_jsonl(output / "crops.jsonl", crop_rows)
    contract.write_new_jsonl(output / "atom-objects.jsonl", atom_rows)
    manifest = {
        "schema_version": contract.CROP_SCHEMA_VERSION,
        "audit_id": "external-ocr-r0",
        "phase": "candidate_coverage_only",
        "selection": {
            "path": str(selection_path),
            "sha256": contract.sha256_file(selection_path),
        },
        "bbox_source": "predicted_v0108",
        "crop_contract": (
            "bbox_crop(context=4.0,output_size=192x48,"
            "aspect_preserving,white_canvas)"
        ),
        "objects": len(crop_rows),
        "atom_objects": len(atom_rows),
        "gt_revealed": False,
        "training_performed": False,
        "routing_performed": False,
        "replacement_performed": False,
        "external_detector_used": False,
        "checkpoints": expected_checkpoint_graph(),
        "artifacts": {
            "crops": {
                "path": "crops.jsonl",
                "sha256": contract.sha256_file(output / "crops.jsonl"),
            },
            "atom_objects": {
                "path": "atom-objects.jsonl",
                "sha256": contract.sha256_file(output / "atom-objects.jsonl"),
            },
        },
    }
    contract.write_new_json(output / "manifest.json", manifest)
    return {
        "output": str(output),
        "objects": len(crop_rows),
        "atom_objects": len(atom_rows),
    }


def expected_checkpoint_graph() -> dict[str, dict[str, str]]:
    return {
        "role": {"path": str(contract.ROLE_CHECKPOINT), "sha256": contract.ROLE_SHA256},
        "bbox": {"path": str(contract.BBOX_CHECKPOINT), "sha256": contract.BBOX_SHA256},
        "ctc": {"path": str(contract.CTC_CHECKPOINT), "sha256": contract.V0109_CTC_SHA256},
        "source": {"path": str(contract.SOURCE_CHECKPOINT), "sha256": contract.SOURCE_SHA256},
        "grounded_graph": {
            "path": str(contract.GROUNDED_GRAPH_CHECKPOINT),
            "sha256": contract.GROUNDED_GRAPH_SHA256,
        },
    }


def load_crop_manifest(
    dataset_dir: str | Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if contract.FROZEN_CROP_MANIFEST_SHA256 is None:
        raise RuntimeError("Crop manifest has not been frozen in the R0 protocol")
    if contract.sha256_file(manifest_path) != contract.FROZEN_CROP_MANIFEST_SHA256:
        raise ValueError("Crop manifest SHA256 differs from the frozen R0 producer artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != contract.CROP_SCHEMA_VERSION:
        raise ValueError("Crop manifest schema differs")
    if (
        manifest.get("audit_id") != "external-ocr-r0"
        or manifest.get("phase") != "candidate_coverage_only"
    ):
        raise ValueError("Crop manifest audit identity or phase differs")
    if manifest.get("bbox_source") != "predicted_v0108":
        raise ValueError("Only the frozen predicted-v0.10.8 bbox source is permitted")
    if manifest.get("gt_revealed") is not False:
        raise ValueError("Crop producer manifest must be frozen before GT reveal")
    for flag in (
        "training_performed",
        "routing_performed",
        "replacement_performed",
        "external_detector_used",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"Crop manifest protocol flag must be false: {flag}")
    if manifest.get("checkpoints") != expected_checkpoint_graph():
        raise ValueError("Crop manifest checkpoint graph differs from the frozen protocol")
    selection_path = Path(manifest["selection"]["path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    if contract.sha256_file(selection_path) != manifest["selection"].get("sha256"):
        raise ValueError("Crop manifest selection SHA256 differs")

    crop_path = root / manifest["artifacts"]["crops"]["path"]
    if contract.sha256_file(crop_path) != manifest["artifacts"]["crops"]["sha256"]:
        raise ValueError("Crop manifest records SHA256 differs")
    rows = contract.read_jsonl(crop_path)
    if len(rows) != manifest.get("objects"):
        raise ValueError("Crop row count differs from the frozen manifest")
    keys = [row["object_key"] for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("Crop manifest keys must be unique and canonically sorted")
    for row in rows:
        _validate_crop_row(root, row, selection)

    atom_path = root / manifest["artifacts"]["atom_objects"]["path"]
    if contract.sha256_file(atom_path) != manifest["artifacts"]["atom_objects"]["sha256"]:
        raise ValueError("Predicted AtomObject records SHA256 differs")
    atom_rows = contract.read_jsonl(atom_path)
    if len(atom_rows) != manifest.get("atom_objects"):
        raise ValueError("Predicted AtomObject row count differs")
    atom_keys = [row.get("atom_object_key") for row in atom_rows]
    if atom_keys != sorted(atom_keys) or len(atom_keys) != len(set(atom_keys)):
        raise ValueError("Predicted AtomObject keys must be unique and sorted")
    selected_crop_keys = set()
    for row in atom_rows:
        if row.get("schema_version") != contract.CROP_SCHEMA_VERSION:
            raise ValueError("Predicted AtomObject row schema differs")
        if row.get("sample_key") not in selection["sample_metadata"]:
            raise ValueError("Predicted AtomObject sample is outside the frozen selection")
        anchor = row.get("anchor")
        if not isinstance(anchor, list) or len(anchor) != 2 or not all(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
            for value in anchor
        ):
            raise ValueError("Predicted AtomObject anchor is invalid")
        crop_object_key = row.get("crop_object_key")
        if bool(row.get("selected_as_text")) != (crop_object_key is not None):
            raise ValueError("Predicted AtomObject text-selection flag differs")
        if crop_object_key is not None:
            if crop_object_key != row["atom_object_key"]:
                raise ValueError("Predicted AtomObject crop identity differs")
            selected_crop_keys.add(crop_object_key)
    if selected_crop_keys != set(keys):
        raise ValueError("Predicted AtomObject text subset differs from crop records")
    return root, manifest, rows


def _validate_crop_row(
    root: Path,
    row: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    if row.get("schema_version") != contract.CROP_SCHEMA_VERSION:
        raise ValueError("Crop row schema differs")
    if "truth" in row or "group" in row:
        raise ValueError("Recognizer crop records must not expose GT fields")
    if row.get("sample_key") not in selection["sample_metadata"]:
        raise ValueError("Crop row sample is outside the frozen selection")
    expected = selection["sample_metadata"][row["sample_key"]]
    if row.get("molecule_id") != expected["molecule_id"]:
        raise ValueError("Crop row molecule identity differs from selection")
    if row.get("variant") != expected["variant"]:
        raise ValueError("Crop row variant differs from selection")
    if row.get("font_family_group") != expected["font_family_group"]:
        raise ValueError("Crop row font group differs from selection")
    if row.get("bbox", {}).get("source") != "predicted_v0108":
        raise ValueError("Crop row bbox source differs")
    crop_path = root / row["crop"]["path"]
    if crop_path.resolve().parent != (root / "crops").resolve():
        raise ValueError("Crop path escapes the frozen crop directory")
    if contract.sha256_file(crop_path) != row["crop"]["sha256"]:
        raise ValueError(f"Crop SHA256 differs: {row['object_key']}")
    if (row["crop"]["width"], row["crop"]["height"]) != (192, 48):
        raise ValueError("Crop dimensions differ from the R0 contract")
