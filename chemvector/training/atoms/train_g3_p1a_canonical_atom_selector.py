#!/usr/bin/env python3
"""Train the canonical G3 Atom selector on the committed train5k stream."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Callable, Iterator

import numpy as np
from PIL import Image
import torch
from torch import nn

from Data.sdf.build_webdataset_sample import mol_from_chemmol, scaffold_smiles
from chemvector.audit_g3_p0i_canonical_cell_slot import (
    config_from_payload,
    model_from_checkpoint,
    validate_p0i_checkpoint,
)
from chemvector.audit_g3_p0j_frozen_composition import (
    P0I_CHECKPOINT,
    P0I_CHECKPOINT_SHA256,
)
from chemvector.audit_g3_p0m_frozen_route_select import (
    EXPECTED_VARIANTS,
    ROUTE_SELECT_MANIFEST,
    ROUTE_SELECT_MANIFEST_SHA256,
)
from chemvector.data import ChemVectorIterableDataset, structure_labels
from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    g3_canonical_atom_only_loss,
)
from chemvector.g3_structural_data import (
    G3SampleManifestRecord,
    file_sha256,
    load_g3_sample_manifest,
)
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    load_frozen_vision_checkpoint,
    maximum_cardinality_radius_match_count,
    module_state_sha256,
)


TRAIN_MANIFEST = Path("Data/sdf/chemtraverse_v0_1_capacity/train_5k.jsonl")
TRAIN_MANIFEST_SHA256 = (
    "0022245e1bbc98a638a94b03426b3cc12c09781ed355baafbd748ad918d66e21"
)
TRAIN_SEQUENCE_SHA256 = (
    "f336398a865d55a65b5efc2a3dd4f3b4e430434356a0fe59bd318ddfac9b6049"
)
DEFAULT_OUTPUT = Path(
    "outputs/checkpoints/chemvector-g3-p1a-canonical-atom-selector-train5k"
)
MODEL_TYPE = "g3-p1a-canonical-atom-selector-train5k-v0.1"
EXPECTED_TRAIN_IMAGES = 5_000
EXPECTED_TRAIN_MOLECULES = 1_250
EXPECTED_ROUTE_IMAGES = 100
EXPECTED_ROUTE_MOLECULES = 25
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0
GRID_SIZE = 192
ATOM_COUNT_BUCKETS = (
    ("1-20", 1, 20),
    ("21-40", 21, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
    ("101-128", 101, 128),
)
CHECKPOINT_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)


def load_train_manifest(
    path: str | Path = TRAIN_MANIFEST,
) -> tuple[list[G3SampleManifestRecord], dict]:
    return load_g3_sample_manifest(
        path,
        expected_sha256=TRAIN_MANIFEST_SHA256,
        expected_samples=EXPECTED_TRAIN_IMAGES,
        expected_molecules=EXPECTED_TRAIN_MOLECULES,
        expected_variants=EXPECTED_VARIANTS,
    )


def load_route_manifest(
    path: str | Path = ROUTE_SELECT_MANIFEST,
) -> tuple[list[G3SampleManifestRecord], dict]:
    return load_g3_sample_manifest(
        path,
        expected_sha256=ROUTE_SELECT_MANIFEST_SHA256,
        expected_samples=EXPECTED_ROUTE_IMAGES,
        expected_molecules=EXPECTED_ROUTE_MOLECULES,
        expected_variants=EXPECTED_VARIANTS,
    )


def _default_dataset_factory(
    shards: list[str], include_keys: set[str]
) -> Iterator[dict]:
    return iter(
        ChemVectorIterableDataset(
            shards,
            shardshuffle=False,
            sample_shuffle_buffer=0,
            sequence_version="0.4",
            include_keys=include_keys,
            serialize_target=False,
            load_raster_layers=False,
        )
    )


def iter_manifest_samples(
    manifest: list[G3SampleManifestRecord],
    *,
    dataset_factory: Callable[[list[str], set[str]], Iterator[dict]] = (
        _default_dataset_factory
    ),
) -> Iterator[dict]:
    """Stream exactly one manifest epoch and prove source order and identity."""

    if not manifest:
        raise ValueError("P1-A manifest stream is empty")
    keys = {record.sample_key for record in manifest}
    if len(keys) != len(manifest):
        raise ValueError("P1-A manifest stream contains duplicate keys")
    shards = list(dict.fromkeys(record.shard for record in manifest))
    source = dataset_factory(shards, keys)
    count = 0
    for count, (record, sample) in enumerate(zip(manifest, source, strict=True), 1):
        key = str(sample["key"])
        molecule_id = str(sample["chemmol"]["molecule_id"])
        inchikey = str(sample["chemmol"]["inchikey"])
        variant = str(sample["render"]["variant"])
        if (
            key != record.sample_key
            or molecule_id != record.molecule_id
            or inchikey != record.inchikey
            or variant != record.variant
        ):
            raise ValueError(f"P1-A stream identity/order differs at sample {count}")
        split = sample.get("meta", {}).get("split")
        source_split = sample.get("meta", {}).get("source_split")
        if split != "train" and source_split != "train":
            raise ValueError(f"P1-A accepts only train samples: {key}")
        yield sample
    if count != len(manifest):
        raise ValueError("P1-A stream did not consume the complete manifest")


class G3AtomCollator:
    """Collate only pixels and Atom supervision; Bond fields are never built."""

    def __init__(self, *, maximum_atoms: int = 128, image_size: int = 768) -> None:
        if min(maximum_atoms, image_size) < 1:
            raise ValueError("P1-A Atom collator dimensions must be positive")
        self.maximum_atoms = maximum_atoms
        self.image_size = image_size

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5

    @staticmethod
    def atom_anchors(sample: dict) -> list[tuple[float, float]]:
        labels = structure_labels(sample)
        atoms = sorted(labels["atoms"], key=lambda item: int(item["atom_id"]))
        render_positions = {
            int(item["atom_id"]): item for item in sample["render"]["atoms"]
        }
        anchors = []
        for atom in atoms:
            atom_id = int(atom["atom_id"])
            rendered = render_positions[atom_id]
            if {"quantized_x", "quantized_y"} <= rendered.keys():
                anchor = (
                    int(rendered["quantized_x"]) / 1023.0,
                    int(rendered["quantized_y"]) / 1023.0,
                )
            else:
                anchor = (
                    float(rendered["image_x"]) / max(int(labels["width"]) - 1, 1),
                    float(rendered["image_y"]) / max(int(labels["height"]) - 1, 1),
                )
            if not 0.0 <= anchor[0] <= 1.0 or not 0.0 <= anchor[1] <= 1.0:
                raise ValueError(f"P1-A Atom anchor is outside canvas: {atom_id}")
            anchors.append(anchor)
        return anchors

    def __call__(self, samples: list[dict]) -> dict:
        if not samples:
            raise ValueError("P1-A Atom collator requires samples")
        atom_anchors = torch.zeros(len(samples), self.maximum_atoms, 2)
        atom_mask = torch.zeros(len(samples), self.maximum_atoms, dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            anchors = self.atom_anchors(sample)
            if len(anchors) > self.maximum_atoms:
                raise ValueError(
                    f"P1-A sample {sample['key']} exceeds Atom capacity"
                )
            if anchors:
                atom_anchors[batch_index, : len(anchors)] = torch.tensor(anchors)
                atom_mask[batch_index, : len(anchors)] = True
        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "atom_anchors": atom_anchors,
            "atom_mask": atom_mask,
            "keys": [str(sample["key"]) for sample in samples],
            "molecule_ids": [
                str(sample["chemmol"]["molecule_id"]) for sample in samples
            ],
            "variants": [str(sample["render"]["variant"]) for sample in samples],
        }


def iter_manifest_batches(
    manifest: list[G3SampleManifestRecord],
    collator: G3AtomCollator,
    *,
    batch_size: int,
    dataset_factory: Callable[[list[str], set[str]], Iterator[dict]] = (
        _default_dataset_factory
    ),
) -> Iterator[dict]:
    if batch_size < 1:
        raise ValueError("P1-A batch size must be positive")
    pending = []
    for sample in iter_manifest_samples(manifest, dataset_factory=dataset_factory):
        pending.append(sample)
        if len(pending) == batch_size:
            yield collator(pending)
            pending = []
    if pending:
        yield collator(pending)


class CanonicalOccupancyAccumulator:
    def __init__(self, *, slots: int = 3, height: int = 192, width: int = 192):
        self.slots = slots
        self.height = height
        self.width = width
        self.cells: dict[int, int] = {}
        self.atoms: dict[int, int] = {}
        self.images: dict[int, set[str]] = {}
        self.sample_maximum: dict[str, int] = {}
        self.overflow = []
        self.collisions = []

    def add(self, sample_key: str, anchors: list[tuple[float, float]]) -> None:
        coordinate_groups: dict[tuple[float, float], list[int]] = {}
        cell_groups: dict[tuple[int, int], list[int]] = {}
        for index, anchor in enumerate(anchors):
            coordinate = (float(anchor[0]), float(anchor[1]))
            coordinate_groups.setdefault(coordinate, []).append(index)
            x = min(self.width - 1, max(0, int(anchor[0] * self.width)))
            y = min(self.height - 1, max(0, int(anchor[1] * self.height)))
            cell_groups.setdefault((y, x), []).append(index)
        for coordinate, indices in coordinate_groups.items():
            if len(indices) > 1:
                self.collisions.append(
                    {
                        "sample_key": sample_key,
                        "normalized_anchor": list(coordinate),
                        "target_indices": indices,
                    }
                )
        maximum = max((len(indices) for indices in cell_groups.values()), default=0)
        self.sample_maximum[sample_key] = maximum
        for (y, x), indices in cell_groups.items():
            occupancy = len(indices)
            self.cells[occupancy] = self.cells.get(occupancy, 0) + 1
            self.atoms[occupancy] = self.atoms.get(occupancy, 0) + occupancy
            self.images.setdefault(occupancy, set()).add(sample_key)
            if occupancy > self.slots:
                self.overflow.append(
                    {
                        "sample_key": sample_key,
                        "cell": {"y": y, "x": x},
                        "occupancy": occupancy,
                        "target_indices": indices,
                    }
                )

    def finalize(self) -> dict:
        return {
            "grid": {
                "slots": self.slots,
                "height": self.height,
                "width": self.width,
            },
            "cell_occupancy": {
                str(value): {
                    "cells": self.cells.get(value, 0),
                    "atoms": self.atoms.get(value, 0),
                    "images": len(self.images.get(value, set())),
                }
                for value in (1, 2, 3)
            },
            "overflow_gt_3": {
                "cells": len(self.overflow),
                "images": len({item["sample_key"] for item in self.overflow}),
                "sample_keys": sorted({item["sample_key"] for item in self.overflow}),
                "details": self.overflow,
            },
            "identical_coordinate_collision": {
                "groups": len(self.collisions),
                "images": len({item["sample_key"] for item in self.collisions}),
                "sample_keys": sorted(
                    {item["sample_key"] for item in self.collisions}
                ),
                "details": self.collisions,
            },
            "sample_maximum_occupancy": self.sample_maximum,
        }


def stream_preflight(
    manifest: list[G3SampleManifestRecord],
    collator: G3AtomCollator,
    *,
    dataset_factory: Callable[[list[str], set[str]], Iterator[dict]] = (
        _default_dataset_factory
    ),
) -> dict:
    occupancy = CanonicalOccupancyAccumulator(slots=3, height=GRID_SIZE, width=GRID_SIZE)
    scaffolds: dict[str, str] = {}
    digest = hashlib.sha256()
    samples = 0
    for sample in iter_manifest_samples(manifest, dataset_factory=dataset_factory):
        key = str(sample["key"])
        molecule_id = str(sample["chemmol"]["molecule_id"])
        anchors = collator.atom_anchors(sample)
        occupancy.add(key, anchors)
        digest.update(f"{key}\n".encode("utf-8"))
        if molecule_id not in scaffolds:
            mol = mol_from_chemmol(sample["chemmol"])
            smiles = str(sample["chemmol"]["canonical_isomeric_smiles"])
            scaffolds[molecule_id] = scaffold_smiles(mol, smiles)
        samples += 1
    result = occupancy.finalize()
    result.update(
        {
            "samples": samples,
            "molecules": len(scaffolds),
            "sample_keys_sha256": digest.hexdigest(),
            "scaffolds": sorted(set(scaffolds.values())),
        }
    )
    return result


def identity_overlap_audit(
    train: list[G3SampleManifestRecord],
    route: list[G3SampleManifestRecord],
    train_preflight: dict,
    route_preflight: dict,
) -> dict:
    fields = {
        "sample_key": lambda item: item.sample_key,
        "molecule_id": lambda item: item.molecule_id,
        "inchikey": lambda item: item.inchikey,
    }
    overlap = {
        name: sorted({getter(item) for item in train} & {getter(item) for item in route})
        for name, getter in fields.items()
    }
    scaffold = sorted(
        set(train_preflight["scaffolds"]) & set(route_preflight["scaffolds"])
    )
    overlap["scaffold"] = scaffold
    return {
        "overlap": overlap,
        "all_three_identity_zero": all(
            not overlap[name] for name in ("sample_key", "molecule_id", "inchikey")
        ),
        "scaffold_overlap_count": len(scaffold),
        "scaffold_hard_gate": False,
    }


def validate_occupancy(preflight: dict, *, name: str) -> None:
    if preflight["overflow_gt_3"]["cells"]:
        raise ValueError(f"P1-A {name} contains occupancy above three")
    if preflight["identical_coordinate_collision"]["groups"]:
        raise ValueError(f"P1-A {name} contains identical Atom coordinates")


def _empty_atom_totals() -> dict[str, float | dict[str, int]]:
    return {
        "images": 0.0,
        "predicted_atoms": 0.0,
        "target_atoms": 0.0,
        "matched_atoms": 0.0,
        "count_exact_images": 0.0,
        "whole_set_exact_images": 0.0,
        "count_absolute_error": 0.0,
        "count_error_distribution": {},
    }


def _merge_atom_totals(destination: dict, source: dict) -> None:
    for name in (
        "images",
        "predicted_atoms",
        "target_atoms",
        "matched_atoms",
        "count_exact_images",
        "whole_set_exact_images",
        "count_absolute_error",
    ):
        destination[name] += source[name]
    for error, count in source["count_error_distribution"].items():
        destination["count_error_distribution"][error] = (
            destination["count_error_distribution"].get(error, 0) + count
        )


def _finalize_atom_totals(totals: dict) -> dict:
    images = max(totals["images"], 1.0)
    predicted = max(totals["predicted_atoms"], 1.0)
    target = max(totals["target_atoms"], 1.0)
    precision = totals["matched_atoms"] / predicted
    recall = totals["matched_atoms"] / target
    return {
        "images": int(totals["images"]),
        "predicted_atoms": int(totals["predicted_atoms"]),
        "target_atoms": int(totals["target_atoms"]),
        "matched_atoms": int(totals["matched_atoms"]),
        "precision_6px": precision,
        "recall_6px": recall,
        "f1_6px": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "count_exact_images": int(totals["count_exact_images"]),
        "count_mae": totals["count_absolute_error"] / images,
        "whole_set_exact_images": int(totals["whole_set_exact_images"]),
        "count_error_distribution": {
            str(error): count
            for error, count in sorted(
                totals["count_error_distribution"].items(), key=lambda item: int(item[0])
            )
        },
    }


def atom_sample_statistics(
    predicted_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
) -> dict:
    matched = maximum_cardinality_radius_match_count(
        predicted_anchors,
        target_anchors,
        radius_pixels=MATCH_RADIUS_PIXELS,
        image_size=IMAGE_SIZE,
    )
    predicted = len(predicted_anchors)
    target = len(target_anchors)
    error = predicted - target
    exact = predicted == target
    return {
        "images": 1.0,
        "predicted_atoms": float(predicted),
        "target_atoms": float(target),
        "matched_atoms": float(matched),
        "count_exact_images": float(exact),
        "whole_set_exact_images": float(exact and matched == target),
        "count_absolute_error": float(abs(error)),
        "count_error_distribution": {error: 1},
    }


def atom_count_bucket(target_count: int) -> str:
    for name, lower, upper in ATOM_COUNT_BUCKETS:
        if lower <= target_count <= upper:
            return name
    raise ValueError(f"P1-A target Atom count is outside 1..128: {target_count}")


@torch.no_grad()
def evaluate_route_streaming(
    model: G3AtomOnlyPredictor,
    vision: nn.Module,
    manifest: list[G3SampleManifestRecord],
    collator: G3AtomCollator,
    occupancy: dict,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    dataset_factory: Callable[[list[str], set[str]], Iterator[dict]] = (
        _default_dataset_factory
    ),
) -> dict:
    was_training = model.training
    model.eval()
    vision.eval()
    totals = _empty_atom_totals()
    by_variant = {
        variant: _empty_atom_totals() for variant in sorted(EXPECTED_VARIANTS)
    }
    by_occupancy = {str(value): _empty_atom_totals() for value in (1, 2, 3)}
    by_atom_count = {
        name: _empty_atom_totals() for name, _, _ in ATOM_COUNT_BUCKETS
    }
    per_sample = []
    digest = hashlib.sha256()
    for batch in iter_manifest_batches(
        manifest,
        collator,
        batch_size=batch_size,
        dataset_factory=dataset_factory,
    ):
        for key in batch["keys"]:
            digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with _autocast_context(device, precision):
            features = vision.forward_stroke_resolution(pixels)
            output = model(features.detach())
        for index, key in enumerate(batch["keys"]):
            predicted_indices = torch.where(output["atom_mask"][index])[0]
            target_indices = torch.where(batch["atom_mask"][index])[0]
            current = atom_sample_statistics(
                output["atom_anchors"][index, predicted_indices].detach().float().cpu(),
                batch["atom_anchors"][index, target_indices].float(),
            )
            _merge_atom_totals(totals, current)
            _merge_atom_totals(by_variant[batch["variants"][index]], current)
            maximum_occupancy = occupancy["sample_maximum_occupancy"][key]
            _merge_atom_totals(by_occupancy[str(maximum_occupancy)], current)
            count_bucket = atom_count_bucket(len(target_indices))
            _merge_atom_totals(by_atom_count[count_bucket], current)
            finalized = _finalize_atom_totals(current)
            per_sample.append(
                {
                    "sample_key": key,
                    "molecule_id": batch["molecule_ids"][index],
                    "variant": batch["variants"][index],
                    "maximum_occupancy": maximum_occupancy,
                    "target_atom_count": len(target_indices),
                    "predicted_atom_count": len(predicted_indices),
                    "matched_atoms": finalized["matched_atoms"],
                    "count_exact": bool(finalized["count_exact_images"]),
                    "whole_set_exact": bool(finalized["whole_set_exact_images"]),
                }
            )
        del pixels, features, output
    if was_training:
        model.train()
    return {
        "scope": "route-select100-streaming-deploy-atom",
        **_finalize_atom_totals(totals),
        "per_variant": {
            name: _finalize_atom_totals(value) for name, value in by_variant.items()
        },
        "per_occupancy": {
            name: _finalize_atom_totals(value)
            for name, value in by_occupancy.items()
        },
        "per_atom_count": {
            name: _finalize_atom_totals(value)
            for name, value in by_atom_count.items()
        },
        "per_sample": per_sample,
        "sample_keys_sha256": digest.hexdigest(),
    }


def atom_capability_gate(metrics: dict) -> dict:
    checks = {
        "count_exact_80": metrics["count_exact_images"] >= 80,
        "count_mae_1": metrics["count_mae"] <= 1.0,
        "precision_97": metrics["precision_6px"] >= 0.97,
        "recall_97": metrics["recall_6px"] >= 0.97,
        "f1_97": metrics["f1_6px"] >= 0.97,
        "whole_set_exact_50": metrics["whole_set_exact_images"] >= 50,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def metric_delta(step0: dict, final: dict) -> dict:
    fields = (
        "matched_atoms",
        "precision_6px",
        "recall_6px",
        "f1_6px",
        "count_exact_images",
        "count_mae",
        "whole_set_exact_images",
    )
    return {name: final[name] - step0[name] for name in fields}


def load_warmstart_p0i(path: Path, device: torch.device):
    if file_sha256(path) != P0I_CHECKPOINT_SHA256:
        raise ValueError("P1-A P0-I warmstart SHA differs from the protocol")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_p0i_checkpoint(payload, step=2_400)
    config = config_from_payload(payload)
    model = model_from_checkpoint(payload, config, device)
    model.requires_grad_(True).train()
    return model, payload, config


def validate_atom_optimizer(
    model: G3AtomOnlyPredictor, optimizer: torch.optim.Optimizer
) -> None:
    expected = list(model.atom_head.parameters())
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(expected) != len(actual) or {id(value) for value in expected} != {
        id(value) for value in actual
    }:
        raise ValueError("P1-A optimizer must contain only Atom-head parameters")


def atom_optimizer_is_exact(
    model: G3AtomOnlyPredictor, optimizer: torch.optim.Optimizer
) -> bool:
    try:
        validate_atom_optimizer(model, optimizer)
    except ValueError:
        return False
    return True


def p1a_training_step(
    model: G3AtomOnlyPredictor,
    vision: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
) -> dict[str, float]:
    validate_atom_optimizer(model, optimizer)
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("P1-A vision must remain frozen")
    optimizer.zero_grad(set_to_none=True)
    pixels = batch["pixel_values"].to(device, non_blocking=True)
    targets = {
        "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
        "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
    }
    with torch.no_grad():
        with _autocast_context(device, precision):
            features = vision.forward_stroke_resolution(pixels)
    with _autocast_context(device, precision):
        output = model(features.detach())
        losses = g3_canonical_atom_only_loss(output, targets)
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.atom_head.parameters(), max_grad_norm
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P1-A Atom gradient norm is not finite")
    optimizer.step()
    if any(parameter.grad is not None for parameter in vision.parameters()):
        raise RuntimeError("P1-A produced gradients for frozen vision")
    return {
        **{name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": float(gradient_norm.detach()),
        "batch_images": float(len(batch["keys"])),
    }


def static_training_contract() -> dict[str, int | bool]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_paths = (
        "dev_route_" + "confirm_400.jsonl",
        "fresh_" + "eval_12k",
        "public_" + "ocsr_v1",
        "MR-" + "TADF_SDF",
    )
    forbidden_path_references = sum(path in source for path in forbidden_paths)
    forbidden_identifiers = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and any(token in node.id.lower() for token in ("bond", "pair", "ocr"))
    }
    training_source = ast.parse(
        inspect_source := __import__("inspect").getsource(p1a_training_step)
    )
    vision_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward_stroke_resolution"
        for node in ast.walk(training_source)
    )
    canonical_loss_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "g3_canonical_atom_only_loss"
        for node in ast.walk(training_source)
    )
    teacher_forced_count_references = sum(
        isinstance(node, ast.keyword) and node.arg == "atom_counts"
        for node in ast.walk(training_source)
    )
    whole_cache_references = sum(
        isinstance(node, ast.Name)
        and node.id == "cache_frozen_vision_features"
        for node in ast.walk(tree)
    )
    if (
        forbidden_path_references
        or forbidden_identifiers
        or vision_calls != 1
        or canonical_loss_calls != 1
        or teacher_forced_count_references
        or whole_cache_references
    ):
        raise ValueError("P1-A source violates the fixed Atom-only protocol")
    return {
        "forbidden_data_path_references": forbidden_path_references,
        "forbidden_model_identifiers": len(forbidden_identifiers),
        "vision_calls_per_training_step": vision_calls,
        "canonical_loss_calls_per_training_step": canonical_loss_calls,
        "teacher_forced_count_references": teacher_forced_count_references,
        "whole_dataset_feature_cache_references": whole_cache_references,
        "whole_dataset_feature_cache": False,
        "source_length": len(inspect_source),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(TRAIN_MANIFEST))
    parser.add_argument("--route-manifest", default=str(ROUTE_SELECT_MANIFEST))
    parser.add_argument("--p0i-warmstart", default=str(P0I_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6_250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1_250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "epochs": 5,
        "max_steps": 6_250,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_every": 1_250,
        "num_workers": 0,
        "seed": 20260713,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"P1-A requires fixed {name}={expected}")
    if args.log_every < 1:
        raise ValueError("P1-A log interval must be positive")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P1-A requires fixed CUDA BF16 training")
    paths = {
        "train_manifest": TRAIN_MANIFEST,
        "route_manifest": ROUTE_SELECT_MANIFEST,
        "p0i_warmstart": P0I_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT,
    }
    for name, expected in paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P1-A requires fixed {name}: {expected}")


def checkpoint_step_files(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.glob("step-*.pt") if path.is_file())


def validate_checkpoint_files(output_dir: Path, steps: list[int]) -> None:
    expected = sorted(f"step-{step:06d}.pt" for step in steps)
    actual = checkpoint_step_files(output_dir)
    if actual != expected:
        raise RuntimeError(
            f"P1-A checkpoint file set differs: expected={expected}, actual={actual}"
        )


def epoch_audits_are_exact(epoch_audits: list[dict]) -> bool:
    return (
        len(epoch_audits) == 5
        and [item.get("epoch") for item in epoch_audits] == [1, 2, 3, 4, 5]
        and all(
            item.get("images") == 5_000
            and item.get("steps") == 1_250
            and item.get("sample_keys_sha256") == TRAIN_SEQUENCE_SHA256
            for item in epoch_audits
        )
    )


def checkpoint_payload(
    *,
    model: G3AtomOnlyPredictor,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    config,
    lineage: dict,
    audit: dict,
    step: int,
    epoch: int,
    exposure: int,
    losses: dict,
    step0_metrics: dict,
    final_metrics: dict | None,
    gate: dict | None,
    completed_epoch_audits: list[dict],
) -> dict:
    return {
        "checkpoint_schema_version": "g3-p1a-atom-selector-checkpoint-0.1",
        "model_type": MODEL_TYPE,
        "atom_head": model.atom_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "samples_seen": exposure,
        "training_args": vars(args),
        "g3_config": asdict(config),
        "lineage": lineage,
        "audit": audit,
        "trainer_sha256": audit.get("implementation", {}).get("trainer_sha256"),
        "last_losses": losses,
        "step0_route_metrics": step0_metrics,
        "final_route_metrics": final_metrics,
        "p1a_gate": gate,
        "completed_epoch_audits": [dict(item) for item in completed_epoch_audits],
    }


def p1a_gate(protocol_checks: dict[str, bool], final_metrics: dict) -> dict:
    capability = atom_capability_gate(final_metrics)
    protocol_passed = all(protocol_checks.values())
    passed = protocol_passed and capability["status"] == "passed"
    return {
        "status": "passed" if passed else "failed",
        "decision": "GO" if passed else "NO-GO",
        "protocol_status": "passed" if protocol_passed else "failed",
        "protocol_checks": protocol_checks,
        "capability": capability,
        "fixed_final_step": 6_250,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("P1-A CUDA training requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P1-A output directory is not empty: {output_dir}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    source_contract = static_training_contract()
    train_manifest, train_manifest_audit = load_train_manifest(args.train_manifest)
    route_manifest, route_manifest_audit = load_route_manifest(args.route_manifest)
    collator = G3AtomCollator(maximum_atoms=128, image_size=768)
    train_preflight = stream_preflight(train_manifest, collator)
    route_preflight = stream_preflight(route_manifest, collator)
    validate_occupancy(train_preflight, name="train5k")
    validate_occupancy(route_preflight, name="route-select100")
    overlap = identity_overlap_audit(
        train_manifest, route_manifest, train_preflight, route_preflight
    )
    if not overlap["all_three_identity_zero"]:
        raise ValueError("P1-A train and route sample/molecule/InChIKey overlap")
    if (
        train_preflight["sample_keys_sha256"]
        != train_manifest_audit["sample_keys_sha256"]
        or train_preflight["sample_keys_sha256"] != TRAIN_SEQUENCE_SHA256
        or route_preflight["sample_keys_sha256"]
        != route_manifest_audit["sample_keys_sha256"]
    ):
        raise ValueError("P1-A streaming preflight changed manifest order")

    model, p0i_payload, config = load_warmstart_p0i(
        Path(args.p0i_warmstart), device
    )
    if config.maximum_atoms != 128 or config.atom_slots_per_cell != 3:
        raise ValueError("P1-A warmstart Atom capacity differs from the protocol")
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        p0i_payload["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P1-A frozen vision SHA differs from the protocol")
    vision.requires_grad_(False).eval()
    model.requires_grad_(True).train()
    initial_vision_sha = module_state_sha256(vision)
    initial_atom_sha = module_state_sha256(model.atom_head)
    optimizer = torch.optim.AdamW(
        model.atom_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    validate_atom_optimizer(model, optimizer)
    parameter_names = [name for name, _ in model.atom_head.named_parameters()]
    parameter_count = sum(
        parameter.numel() for parameter in model.atom_head.parameters()
    )
    lineage = {
        "p0i_warmstart": {
            "path": str(Path(args.p0i_warmstart).resolve()),
            "sha256": file_sha256(args.p0i_warmstart),
            "global_step": p0i_payload["global_step"],
            "atom_head_state_sha256": initial_atom_sha,
        },
        "vision": {
            "path": str(Path(p0i_payload["source_vision_checkpoint"]).resolve()),
            "sha256": vision_sha,
            "state_sha256": initial_vision_sha,
        },
    }
    audit = {
        "schema_version": "g3-p1a-training-audit-0.1",
        "scope": "train-distribution-canonical-atom-selector-train5k",
        "manifests": {
            "train5k": train_manifest_audit,
            "route_select100": route_manifest_audit,
            "identity_overlap": overlap,
        },
        "preflight": {
            "train5k": train_preflight,
            "route_select100": route_preflight,
        },
        "lineage": lineage,
        "implementation": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(__file__),
            "source_contract": source_contract,
        },
        "streaming": {
            "whole_dataset_pixel_cache": False,
            "whole_dataset_feature_cache": False,
            "shuffle": False,
            "num_workers": 0,
            "drop_last": False,
            "batch_size": 4,
            "steps_per_epoch": 1_250,
        },
        "optimizer": {
            "name": "AdamW",
            "parameter_names": parameter_names,
            "parameter_count": parameter_count,
            "atom_head_only": True,
            "restored_state": False,
        },
        "forbidden_components_loaded": {
            "p0g": False,
            "p0k": False,
            "semantic": False,
            "ocr": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit.json", audit)
    _atomic_json(output_dir / "training_args.json", vars(args))
    _atomic_json(output_dir / "config.json", {"g3_config": asdict(config)})

    route_evaluation_steps = []
    step0_metrics = evaluate_route_streaming(
        model,
        vision,
        route_manifest,
        collator,
        route_preflight,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    step0_metrics["step"] = 0
    route_evaluation_steps.append(0)
    _atomic_json(output_dir / "step0_route_metrics.json", step0_metrics)
    model.train()

    step = exposure = 0
    saved_steps = []
    epoch_audits = []
    losses: dict[str, float] = {}
    running: dict[str, float] = {}
    started = time.perf_counter()
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        digest = hashlib.sha256()
        epoch_images = epoch_steps = 0
        for batch in iter_manifest_batches(
            train_manifest, collator, batch_size=args.batch_size
        ):
            if len(batch["keys"]) != args.batch_size:
                raise RuntimeError("P1-A encountered a residual training batch")
            for key in batch["keys"]:
                digest.update(f"{key}\n".encode("utf-8"))
            losses = p1a_training_step(
                model,
                vision,
                optimizer,
                batch,
                device=device,
                precision=args.precision,
                max_grad_norm=args.max_grad_norm,
            )
            step += 1
            epoch_steps += 1
            exposure += len(batch["keys"])
            epoch_images += len(batch["keys"])
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "samples_seen": exposure,
                    "elapsed_seconds": time.perf_counter() - started,
                    **{
                        name: value / args.log_every
                        for name, value in running.items()
                    },
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
                running.clear()
        epoch_audit = {
            "epoch": epoch,
            "images": epoch_images,
            "steps": epoch_steps,
            "sample_keys_sha256": digest.hexdigest(),
        }
        if (
            epoch_images != EXPECTED_TRAIN_IMAGES
            or epoch_steps != 1_250
            or epoch_audit["sample_keys_sha256"] != TRAIN_SEQUENCE_SHA256
        ):
            raise RuntimeError("P1-A epoch stream differs from the fixed manifest")
        epoch_audits.append(epoch_audit)
        if step in CHECKPOINT_STEPS[:-1]:
            if module_state_sha256(vision) != initial_vision_sha:
                raise RuntimeError("P1-A changed frozen vision before checkpoint")
            checkpoint = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                args=args,
                config=config,
                lineage=lineage,
                audit=audit,
                step=step,
                epoch=epoch,
                exposure=exposure,
                losses=losses,
                step0_metrics=step0_metrics,
                final_metrics=None,
                gate=None,
                completed_epoch_audits=list(epoch_audits),
            )
            path = output_dir / f"step-{step:06d}.pt"
            _atomic_torch_save(path, checkpoint)
            _atomic_torch_save(output_dir / "last.pt", checkpoint)
            saved_steps.append(step)
    if step != args.max_steps or exposure != 25_000:
        raise RuntimeError("P1-A final step or exposure differs from the protocol")

    final_metrics = evaluate_route_streaming(
        model,
        vision,
        route_manifest,
        collator,
        route_preflight,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    final_metrics["step"] = step
    route_evaluation_steps.append(step)
    final_metrics["delta_from_step0"] = metric_delta(step0_metrics, final_metrics)
    final_vision_sha = module_state_sha256(vision)
    protocol_checks = {
        "train_manifest": train_manifest_audit["sha256"] == TRAIN_MANIFEST_SHA256,
        "route_manifest": (
            route_manifest_audit["sha256"] == ROUTE_SELECT_MANIFEST_SHA256
        ),
        "identity_disjoint": overlap["all_three_identity_zero"],
        "warmstart_sha": file_sha256(args.p0i_warmstart) == P0I_CHECKPOINT_SHA256,
        "vision_sha": vision_sha == DEFAULT_VISION_CHECKPOINT_SHA256,
        "vision_state_unchanged": final_vision_sha == initial_vision_sha,
        "vision_frozen_eval": (
            not vision.training
            and not any(parameter.requires_grad for parameter in vision.parameters())
        ),
        "optimizer_atom_head_only": atom_optimizer_is_exact(model, optimizer),
        "streaming_epochs": epoch_audits_are_exact(epoch_audits),
        "total_exposure_and_updates": exposure == 25_000 and step == 6_250,
        "checkpoint_schedule": saved_steps + [step] == list(CHECKPOINT_STEPS),
        "route_evaluation_steps": route_evaluation_steps == [0, 6_250],
        "route_stream_order": (
            step0_metrics["sample_keys_sha256"]
            == final_metrics["sample_keys_sha256"]
            == route_manifest_audit["sample_keys_sha256"]
        ),
        "train_occupancy": (
            train_preflight["overflow_gt_3"]["cells"] == 0
            and train_preflight["identical_coordinate_collision"]["groups"] == 0
        ),
        "route_occupancy": (
            route_preflight["overflow_gt_3"]["cells"] == 0
            and route_preflight["identical_coordinate_collision"]["groups"] == 0
        ),
        "source_contract": all(
            (
                source_contract["forbidden_data_path_references"] == 0,
                source_contract["forbidden_model_identifiers"] == 0,
                source_contract["vision_calls_per_training_step"] == 1,
                source_contract["canonical_loss_calls_per_training_step"] == 1,
                not source_contract["whole_dataset_feature_cache"],
            )
        ),
    }
    gate = p1a_gate(protocol_checks, final_metrics)
    final_checkpoint = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        args=args,
        config=config,
        lineage=lineage,
        audit=audit,
        step=step,
        epoch=args.epochs,
        exposure=exposure,
        losses=losses,
        step0_metrics=step0_metrics,
        final_metrics=final_metrics,
        gate=gate,
        completed_epoch_audits=epoch_audits,
    )
    final_path = output_dir / f"step-{step:06d}.pt"
    _atomic_torch_save(final_path, final_checkpoint)
    _atomic_torch_save(output_dir / "last.pt", final_checkpoint)
    saved_steps.append(step)
    validate_checkpoint_files(output_dir, saved_steps)
    _atomic_json(output_dir / "final_route_metrics.json", final_metrics)
    _atomic_json(output_dir / "epoch_audit.json", {"epochs": epoch_audits})
    _atomic_json(output_dir / "p1a_gate.json", gate)
    _atomic_json(
        output_dir / "status.json",
        {
            "schema_version": "g3-p1a-status-0.1",
            "status": gate["status"],
            "decision": gate["decision"],
            "global_step": step,
            "samples_seen": exposure,
            "checkpoint": final_path.name,
            "saved_steps": saved_steps,
            "route_evaluation_steps": route_evaluation_steps,
            "initial_atom_state_sha256": initial_atom_sha,
            "final_atom_state_sha256": module_state_sha256(model.atom_head),
            "initial_vision_state_sha256": initial_vision_sha,
            "final_vision_state_sha256": final_vision_sha,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps({"p1a_gate": gate, "final": final_metrics}), flush=True)


if __name__ == "__main__":
    main()
