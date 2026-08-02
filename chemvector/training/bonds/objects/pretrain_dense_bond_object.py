#!/usr/bin/env python3
"""Train spatially grounded v0.8 Bond Objects."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import random
import shutil
from collections.abc import Iterable, Iterator

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import (
    mol_from_chemscene_v0_1,
    parse_chemscene_v0_1,
)
from chemvector.bond_object import BOND_TYPES, DIRECTIONS, unordered_endpoint_cost
from chemvector.data import ChemVectorIterableDataset
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
    decode_dense_bond_objects,
    dense_bond_object_loss,
)
from chemvector.grounded_graph import load_grounded_graph_checkpoint
from chemvector.pretrain_chemscene_bond_query import (
    ChemSceneBondCollator,
    _nearest_junction,
    _oriented_endpoint_errors,
    stack_targets,
)
from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
from chemvector.train import expand_shards


def _scene_from_dense_objects(oracle_scene: dict, objects: list[dict]) -> dict | None:
    scene = copy.deepcopy(oracle_scene)
    parsed = parse_chemscene_v0_1(scene)
    coordinates = [junction.coordinate for junction in parsed.junctions]
    oracle_by_pair = {
        tuple(sorted((bond.begin, bond.end))): bond for bond in parsed.bond_objects
    }
    predicted_bonds = []
    seen_pairs = set()
    for obj in objects:
        begin = _nearest_junction(obj["endpoints"][0], coordinates)
        end = _nearest_junction(obj["endpoints"][1], coordinates)
        pair = tuple(sorted((begin, end)))
        if begin == end or pair in seen_pairs:
            return None
        seen_pairs.add(pair)
        oracle = oracle_by_pair.get(pair)
        output_begin, output_end = (
            (oracle.begin, oracle.end) if oracle is not None else (begin, end)
        )
        item = {
            "endpoints": [list(output_begin), list(output_end)],
            "display_type": BOND_TYPES[obj["display"]],
            "chemical_type": BOND_TYPES[obj["chemical"]],
            "direction": DIRECTIONS[obj["direction"]],
            "stereo": oracle.stereo if oracle is not None else "STEREONONE",
            "bbox": None,
        }
        if oracle is not None and oracle.stereo_reference_0 is not None:
            item["stereo_references"] = [
                list(oracle.stereo_reference_0),
                list(oracle.stereo_reference_1),
            ]
        predicted_bonds.append(item)
    scene["bond_objects"] = predicted_bonds
    try:
        parse_chemscene_v0_1(scene)
    except Exception:
        return None
    return scene


def _compare_objects(
    objects: list[dict],
    targets: dict[str, torch.Tensor],
    batch_index: int,
) -> tuple[int, bool, float]:
    valid = targets["mask"][batch_index]
    expected = targets["endpoints"][batch_index, valid]
    if len(objects) != len(expected) or not len(expected):
        return 0, False, float("inf")
    predicted = torch.stack([obj["endpoints"] for obj in objects])
    rows, columns = linear_sum_assignment(
        unordered_endpoint_cost(predicted, expected).float().cpu().numpy()
    )
    valid_indices = torch.where(valid)[0]
    correct = 0
    worst = 0.0
    for row, column in zip(rows, columns):
        errors = _oriented_endpoint_errors(predicted[row], expected[column], 768.0)
        worst = max(worst, float(errors.max()))
        target_index = valid_indices[column]
        obj = objects[row]
        classes_match = (
            obj["display"] == int(targets["display"][batch_index, target_index])
            and obj["chemical"] == int(targets["chemical"][batch_index, target_index])
        )
        correct += int(bool((errors <= 6.0).all()) and classes_match)
    return correct, correct == len(expected), worst


@torch.no_grad()
def evaluate(
    model,
    detector,
    records,
    collator,
    device,
    batch_size: int,
    threshold: float,
) -> dict:
    detector.eval()
    totals = {
        "images": 0,
        "bonds": 0,
        "known_correct": 0,
        "known_exact": 0,
        "predicted_count_exact": 0,
        "predicted_exact": 0,
        "oracle_junction_sdf_exact": 0,
    }
    maximum_errors = []
    object_failure_keys = []
    sdf_failure_keys = []
    for start in range(0, len(records), batch_size):
        batch = collator(records[start : start + batch_size])
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _forward_detector(model, detector, pixels)
        all_objects = decode_dense_bond_objects(
            output, threshold=0.0, maximum_objects=128
        )
        for batch_index, ranked in enumerate(all_objects):
            count = int(targets["mask"][batch_index].sum())
            known_objects = ranked[:count]
            predicted_objects = [obj for obj in ranked if obj["score"] >= threshold]
            known_correct, known_exact, worst = _compare_objects(
                known_objects, targets, batch_index
            )
            _, predicted_exact, _ = _compare_objects(
                predicted_objects, targets, batch_index
            )
            sdf_exact = False
            if predicted_exact:
                predicted_scene = _scene_from_dense_objects(
                    batch["scenes"][batch_index], predicted_objects
                )
                if predicted_scene is not None:
                    try:
                        sdf_exact = (
                            Chem.MolToInchiKey(mol_from_chemscene_v0_1(predicted_scene))
                            == batch["expected_inchikeys"][batch_index]
                        )
                    except Exception:
                        sdf_exact = False
            totals["images"] += 1
            totals["bonds"] += count
            totals["known_correct"] += known_correct
            totals["known_exact"] += int(known_exact)
            totals["predicted_count_exact"] += int(len(predicted_objects) == count)
            totals["predicted_exact"] += int(predicted_exact)
            totals["oracle_junction_sdf_exact"] += int(sdf_exact)
            maximum_errors.append(worst)
            if not predicted_exact and len(object_failure_keys) < 50:
                object_failure_keys.append(batch["keys"][batch_index])
            if not sdf_exact and len(sdf_failure_keys) < 50:
                sdf_failure_keys.append(batch["keys"][batch_index])
    images = max(totals["images"], 1)
    bonds = max(totals["bonds"], 1)
    finite = [value for value in maximum_errors if np.isfinite(value)]
    return {
        "samples": totals["images"],
        "known_object_accuracy": totals["known_correct"] / bonds,
        "known_object_exact": totals["known_exact"] / images,
        "predicted_count_accuracy": totals["predicted_count_exact"] / images,
        "predicted_object_exact": totals["predicted_exact"] / images,
        "oracle_junction_sdf_exact": totals["oracle_junction_sdf_exact"] / images,
        "threshold": threshold,
        "image_max_endpoint_error_p95": (
            float(np.percentile(finite, 95)) if finite else None
        ),
        "image_max_endpoint_error_max": max(finite) if finite else None,
        "object_failure_keys": object_failure_keys,
        "sdf_failure_keys": sdf_failure_keys,
    }


def _extract_detector_features(model, detector, pixels: torch.Tensor):
    stride4 = model.vision_encoder.forward_stroke_resolution(pixels)
    if not isinstance(detector, MultiScaleDenseBondObjectDetector):
        return (stride4,)
    stride8 = model.vision_encoder.forward_high_resolution_from_stroke(stride4)
    stride16 = stride8
    for stage in model.vision_encoder.stages[3:5]:
        stride16 = stage(stride16)
    return stride4, stride8, stride16


def _apply_detector(detector, features) -> dict[str, torch.Tensor]:
    return detector(*features)


def _forward_detector(model, detector, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    return _apply_detector(detector, _extract_detector_features(model, detector, pixels))


def expand_compatible_state_dict(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expand slot-indexed output channels while preserving learned prefixes."""

    if set(target_state) != set(source_state):
        missing = sorted(set(target_state) - set(source_state))
        unexpected = sorted(set(source_state) - set(target_state))
        raise ValueError(
            f"Checkpoint keys differ; missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    expanded = {}
    for name, target in target_state.items():
        source = source_state[name]
        if source.shape == target.shape:
            expanded[name] = source
            continue
        if (
            source.ndim == target.ndim
            and source.ndim >= 1
            and source.shape[1:] == target.shape[1:]
            and source.shape[0] < target.shape[0]
        ):
            value = target.clone()
            value[: source.shape[0]].copy_(source)
            expanded[name] = value
            continue
        raise ValueError(
            f"Cannot expand checkpoint tensor {name}: {tuple(source.shape)} -> "
            f"{tuple(target.shape)}"
        )
    return expanded


def stream_training_batches(
    shards: list[str],
    *,
    samples: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
) -> Iterator[list[dict]]:
    if min(samples, batch_size, shuffle_buffer) < 1:
        raise ValueError("Streaming sample, batch, and shuffle sizes must be positive")
    source = islice(
        iter(
            ChemVectorIterableDataset(
                expand_shards(shards), sequence_version="0.4"
            )
        ),
        samples,
    )
    rng = random.Random(seed)
    pending: list[dict] = []
    emitted = 0
    for record in source:
        pending.append(record)
        if len(pending) < shuffle_buffer:
            continue
        rng.shuffle(pending)
        while len(pending) >= batch_size:
            emitted += batch_size
            yield pending[-batch_size:]
            del pending[-batch_size:]
    rng.shuffle(pending)
    while pending:
        batch = pending[-batch_size:]
        del pending[-batch_size:]
        emitted += len(batch)
        yield batch
    if emitted != samples:
        raise ValueError(
            f"Requested {samples} streaming samples, loaded and emitted {emitted}"
        )


def _record_key(record: dict) -> str:
    key = record.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("Training records must contain a non-empty string key")
    return key


def _record_attribute(record: dict, name: str) -> str:
    meta = record.get("meta")
    if isinstance(meta, dict):
        value = meta.get(name)
        if value is not None and str(value):
            return str(value)
    if name == "variant":
        render = record.get("render")
        if isinstance(render, dict):
            value = render.get("variant")
            if value is not None and str(value):
                return str(value)
    if name == "source":
        chemmol = record.get("chemmol")
        if isinstance(chemmol, dict):
            molecule_id = chemmol.get("molecule_id")
            if isinstance(molecule_id, str) and ":" in molecule_id:
                return molecule_id.split(":", 1)[0]
    return "<unknown>"


def _update_key_digest(digest, key: str) -> None:
    digest.update(key.encode("utf-8"))
    digest.update(b"\n")


def _key_set_sha256(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        _update_key_digest(digest, key)
    return digest.hexdigest()


class MixedSamplingAudit:
    """Accumulate a compact, reproducible audit of coverage/replay sampling."""

    def __init__(
        self,
        *,
        coverage_samples: int,
        replay_pool_samples: int,
        replay_samples: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.coverage_samples = coverage_samples
        self.replay_pool_samples = replay_pool_samples
        self.replay_samples = replay_samples
        self.batch_size = batch_size
        self.seed = seed
        self.key_counts = {
            "coverage": Counter(),
            "replay": Counter(),
        }
        self.key_digests = {
            "coverage": hashlib.sha256(),
            "replay": hashlib.sha256(),
        }
        self.attribute_counts = {
            role: {
                "variant": Counter(),
                "source": Counter(),
            }
            for role in ("coverage", "replay")
        }
        self.replay_pool_keys: list[str] = []
        self.replay_pool_attributes = {
            "variant": Counter(),
            "source": Counter(),
        }
        self.batch_compositions = Counter()
        self.batches = 0

    def set_replay_pool(self, descriptors: list[dict]) -> None:
        if self.replay_pool_keys:
            raise ValueError("Replay pool audit has already been initialized")
        for descriptor in descriptors:
            self.replay_pool_keys.append(descriptor["key"])
            self.replay_pool_attributes["variant"][descriptor["variant"]] += 1
            self.replay_pool_attributes["source"][descriptor["source"]] += 1

    def observe_batch(self, tagged_records: list[tuple[str, dict]]) -> None:
        composition = Counter()
        for role, record in tagged_records:
            if role not in self.key_counts:
                raise ValueError(f"Unknown sampling role: {role}")
            key = _record_key(record)
            self.key_counts[role][key] += 1
            _update_key_digest(self.key_digests[role], key)
            for attribute in ("variant", "source"):
                self.attribute_counts[role][attribute][
                    _record_attribute(record, attribute)
                ] += 1
            composition[role] += 1
        self.batch_compositions[
            f"coverage={composition['coverage']},replay={composition['replay']}"
        ] += 1
        self.batches += 1

    def to_dict(self) -> dict:
        coverage_keys = set(self.key_counts["coverage"])
        replay_keys = set(self.key_counts["replay"])
        replay_pool_keys = set(self.replay_pool_keys)
        total_exposures = self.key_counts["coverage"] + self.key_counts["replay"]
        exposure_histogram = Counter(total_exposures.values())
        replay_exposure_histogram = Counter(self.key_counts["replay"].values())
        expected_half_batch = self.batch_size // 2
        expected_batches = (
            self.coverage_samples + self.replay_samples
        ) // self.batch_size
        issues = []
        if sum(self.key_counts["coverage"].values()) != self.coverage_samples:
            issues.append("coverage exposure count differs from request")
        if sum(self.key_counts["replay"].values()) != self.replay_samples:
            issues.append("replay exposure count differs from request")
        if len(coverage_keys) != self.coverage_samples:
            issues.append("coverage keys are not unique")
        if len(replay_pool_keys) != self.replay_pool_samples:
            issues.append("replay pool keys are not unique")
        if replay_keys != replay_pool_keys:
            issues.append("observed replay keys differ from the replay pool")
        if not replay_pool_keys.issubset(coverage_keys):
            issues.append("replay pool is not a subset of coverage")
        if self.batches != expected_batches:
            issues.append("optimizer step count differs from expected full batches")
        expected_composition = (
            f"coverage={expected_half_batch},replay={expected_half_batch}"
        )
        if self.batch_compositions != Counter({expected_composition: expected_batches}):
            issues.append("one or more batches do not have the expected 1:1 composition")

        pool_digest = hashlib.sha256()
        for key in self.replay_pool_keys:
            _update_key_digest(pool_digest, key)
        return {
            "schema_version": "chemvector-mixed-sampling-audit-0.1",
            "status": "passed" if not issues else "failed",
            "issues": issues,
            "seed": self.seed,
            "requested": {
                "coverage_samples": self.coverage_samples,
                "replay_pool_samples": self.replay_pool_samples,
                "replay_samples": self.replay_samples,
                "total_samples": self.coverage_samples + self.replay_samples,
                "batch_size": self.batch_size,
            },
            "actual": {
                "coverage_samples": sum(self.key_counts["coverage"].values()),
                "replay_samples": sum(self.key_counts["replay"].values()),
                "total_samples": sum(total_exposures.values()),
                "optimizer_steps": self.batches,
                "coverage_unique_keys": len(coverage_keys),
                "replay_unique_keys": len(replay_keys),
                "replay_pool_unique_keys": len(replay_pool_keys),
                "coverage_replay_intersection": len(coverage_keys & replay_keys),
                "total_unique_keys": len(total_exposures),
            },
            "key_sha256": {
                "coverage_ordered": self.key_digests["coverage"].hexdigest(),
                "coverage_set": _key_set_sha256(coverage_keys),
                "replay_ordered": self.key_digests["replay"].hexdigest(),
                "replay_set": _key_set_sha256(replay_keys),
                "replay_pool_ordered": pool_digest.hexdigest(),
                "replay_pool_set": _key_set_sha256(replay_pool_keys),
            },
            "exposure_histogram": {
                str(exposures): count
                for exposures, count in sorted(exposure_histogram.items())
            },
            "replay_exposure_histogram": {
                str(exposures): count
                for exposures, count in sorted(replay_exposure_histogram.items())
            },
            "batch_composition_histogram": dict(
                sorted(self.batch_compositions.items())
            ),
            "attribute_counts": {
                role: {
                    attribute: dict(sorted(counts.items()))
                    for attribute, counts in role_counts.items()
                }
                for role, role_counts in self.attribute_counts.items()
            },
            "replay_pool_attribute_counts": {
                attribute: dict(sorted(counts.items()))
                for attribute, counts in self.replay_pool_attributes.items()
            },
        }


def _stream_prefix_records(shards: list[str], samples: int) -> Iterator[dict]:
    emitted = 0
    source = islice(
        iter(
            ChemVectorIterableDataset(
                expand_shards(shards), sequence_version="0.4"
            )
        ),
        samples,
    )
    for record in source:
        emitted += 1
        yield record
    if emitted != samples:
        raise ValueError(
            f"Requested {samples} prefix samples, loaded and emitted {emitted}"
        )


def _buffered_shuffle_records(
    records: Iterable[dict],
    *,
    shuffle_buffer: int,
    seed: int,
) -> Iterator[dict]:
    if shuffle_buffer < 1:
        raise ValueError("Shuffle buffer must be positive")
    rng = random.Random(seed)
    pending = []
    for record in records:
        pending.append(record)
        if len(pending) < shuffle_buffer:
            continue
        rng.shuffle(pending)
        while pending:
            yield pending.pop()
    rng.shuffle(pending)
    while pending:
        yield pending.pop()


def _collect_replay_pool_descriptors(
    shards: list[str], replay_pool_samples: int
) -> list[dict]:
    descriptors = []
    for index, record in enumerate(
        _stream_prefix_records(shards, replay_pool_samples)
    ):
        descriptors.append(
            {
                "index": index,
                "key": _record_key(record),
                "variant": _record_attribute(record, "variant"),
                "source": _record_attribute(record, "source"),
            }
        )
    keys = [descriptor["key"] for descriptor in descriptors]
    if len(set(keys)) != len(keys):
        raise ValueError("Replay pool contains duplicate sample keys")
    return descriptors


def select_stratified_replay_indices(
    descriptors: list[dict], count: int, *, seed: int
) -> set[int]:
    """Choose an exact, stable source-by-variant stratified replay subset."""

    if count < 0 or count > len(descriptors):
        raise ValueError("Replay subset count is outside the replay pool")
    if count == 0:
        return set()
    groups: dict[tuple[str, str], list[dict]] = {}
    for descriptor in descriptors:
        group = (descriptor["source"], descriptor["variant"])
        groups.setdefault(group, []).append(descriptor)
    total = len(descriptors)
    quotas = {
        group: count * len(items) // total for group, items in groups.items()
    }
    leftover = count - sum(quotas.values())
    remainders = sorted(
        groups,
        key=lambda group: (
            -(count * len(groups[group]) % total),
            group,
        ),
    )
    for group in remainders[:leftover]:
        quotas[group] += 1

    selected = set()
    for group in sorted(groups):
        ranked = sorted(
            groups[group],
            key=lambda descriptor: (
                hashlib.sha256(
                    (
                        f"{seed}\0{descriptor['index']}\0{descriptor['key']}"
                    ).encode("utf-8")
                ).digest(),
                descriptor["index"],
            ),
        )
        selected.update(
            descriptor["index"] for descriptor in ranked[: quotas[group]]
        )
    if len(selected) != count:
        raise RuntimeError(
            f"Selected {len(selected)} replay indices instead of requested {count}"
        )
    return selected


def _validated_replay_cycle(
    shards: list[str], descriptors: list[dict]
) -> Iterator[tuple[int, dict]]:
    for index, record in enumerate(_stream_prefix_records(shards, len(descriptors))):
        key = _record_key(record)
        if key != descriptors[index]["key"]:
            raise ValueError(
                "Replay pool order changed between passes: "
                f"index={index} expected={descriptors[index]['key']} actual={key}"
            )
        yield index, record


def _stream_replay_records(
    shards: list[str],
    descriptors: list[dict],
    *,
    samples: int,
    seed: int,
) -> Iterator[dict]:
    full_cycles, remainder = divmod(samples, len(descriptors))
    for _ in range(full_cycles):
        for _, record in _validated_replay_cycle(shards, descriptors):
            yield record
    if remainder:
        selected = select_stratified_replay_indices(
            descriptors, remainder, seed=seed
        )
        emitted = 0
        for index, record in _validated_replay_cycle(shards, descriptors):
            if index not in selected:
                continue
            emitted += 1
            yield record
        if emitted != remainder:
            raise RuntimeError(
                f"Emitted {emitted} replay remainder samples instead of {remainder}"
            )


def stream_mixed_training_batches(
    shards: list[str],
    *,
    coverage_samples: int,
    replay_pool_samples: int,
    replay_samples: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    audit: MixedSamplingAudit,
) -> Iterator[list[dict]]:
    """Stream exact 1:1 coverage/replay batches without pool-boundary tails."""

    if min(
        coverage_samples,
        replay_pool_samples,
        replay_samples,
        batch_size,
        shuffle_buffer,
    ) < 1:
        raise ValueError("Mixed sampling counts and buffers must be positive")
    if coverage_samples != replay_samples:
        raise ValueError("Mixed sampling currently requires a 1:1 exposure ratio")
    if replay_pool_samples > coverage_samples:
        raise ValueError("Replay pool cannot be larger than coverage")
    if batch_size % 2:
        raise ValueError("Mixed sampling requires an even batch size")
    half_batch = batch_size // 2
    if coverage_samples % half_batch:
        raise ValueError(
            "Coverage and replay exposure counts must be divisible by half the batch size"
        )

    descriptors = _collect_replay_pool_descriptors(shards, replay_pool_samples)
    audit.set_replay_pool(descriptors)
    role_shuffle_buffer = max(half_batch, shuffle_buffer // 2)
    coverage = _buffered_shuffle_records(
        _stream_prefix_records(shards, coverage_samples),
        shuffle_buffer=role_shuffle_buffer,
        seed=seed + 104729,
    )
    replay = _buffered_shuffle_records(
        _stream_replay_records(
            shards,
            descriptors,
            samples=replay_samples,
            seed=seed + 130363,
        ),
        shuffle_buffer=role_shuffle_buffer,
        seed=seed + 155921,
    )
    batch_rng = random.Random(seed + 196613)
    batches = coverage_samples // half_batch
    for _ in range(batches):
        coverage_batch = list(islice(coverage, half_batch))
        replay_batch = list(islice(replay, half_batch))
        if len(coverage_batch) != half_batch or len(replay_batch) != half_batch:
            raise ValueError("Mixed sampling source ended before a full batch was built")
        tagged = [
            ("coverage", record) for record in coverage_batch
        ] + [
            ("replay", record) for record in replay_batch
        ]
        batch_rng.shuffle(tagged)
        audit.observe_batch(tagged)
        yield [record for _, record in tagged]
    if next(coverage, None) is not None or next(replay, None) is not None:
        raise ValueError("Mixed sampling source emitted more records than requested")


def prepare_output_directory(output_dir: Path) -> None:
    """Reject stale training artifacts instead of mixing independent runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        output_dir / "training_args.json",
        output_dir / "config.json",
        output_dir / "initial.pt",
        output_dir / "best.pt",
        output_dir / "last.pt",
        output_dir / "initial_validation.json",
        output_dir / "initial_retention_validation.json",
        output_dir / "validation.json",
        output_dir / "retention_validation.json",
        output_dir / "sampling_audit.json",
        *sorted(output_dir.glob("checkpoint-*.pt")),
    ]
    existing = [path.name for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            f"Dense Bond output directory already contains training artifacts: "
            f"{output_dir} ({', '.join(existing[:8])})"
        )


def validation_score(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics["oracle_junction_sdf_exact"]),
        float(metrics["known_object_exact"]),
        -float(metrics["image_max_endpoint_error_p95"]),
    )


def dual_domain_validation_score(
    primary: dict, retention: dict | None = None
) -> tuple[float, ...]:
    if retention is None:
        return validation_score(primary)
    primary_exact = float(primary["known_object_exact"])
    retention_exact = float(retention["known_object_exact"])
    primary_accuracy = float(primary["known_object_accuracy"])
    retention_accuracy = float(retention["known_object_accuracy"])
    return (
        min(primary_exact, retention_exact),
        (primary_exact + retention_exact) * 0.5,
        min(primary_accuracy, retention_accuracy),
        (primary_accuracy + retention_accuracy) * 0.5,
        -max(
            float(primary["image_max_endpoint_error_p95"]),
            float(retention["image_max_endpoint_error_p95"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-graph-checkpoint", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--slots", type=int)
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--multiscale-only", action="store_true")
    parser.add_argument("--preserve-base-types", action="store_true")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--replay-shards", nargs="+")
    parser.add_argument("--replay-cache-samples", type=int)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--retention-validation-shards", nargs="+")
    parser.add_argument("--retention-validation-samples", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-cache-samples", type=int, default=100)
    parser.add_argument("--stream-train-samples", type=int)
    parser.add_argument("--replay-pool-samples", type=int)
    parser.add_argument("--replay-samples", type=int)
    parser.add_argument("--stream-shuffle-buffer", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--endpoint-barrier-weight", type=float, default=0.02)
    parser.add_argument("--endpoint-barrier-margin-pixels", type=float, default=4.0)
    parser.add_argument("--heatmap-set-weight", type=float, default=0.0)
    parser.add_argument("--heatmap-set-margin", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--validate-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if (args.replay_shards is None) != (args.replay_cache_samples is None):
        raise ValueError(
            "--replay-shards and --replay-cache-samples must be used together"
        )
    if (args.retention_validation_shards is None) != (
        args.retention_validation_samples is None
    ):
        raise ValueError(
            "retention validation shards and sample count must be used together"
        )

    if args.slots is not None and args.slots < 1:
        raise ValueError("--slots must be positive")
    replay_mode = (
        args.replay_pool_samples is not None or args.replay_samples is not None
    )
    if replay_mode:
        if args.replay_pool_samples is None or args.replay_samples is None:
            raise ValueError(
                "--replay-pool-samples and --replay-samples must be used together"
            )
        if args.stream_train_samples is None:
            raise ValueError("Replay mode requires --stream-train-samples")
        if min(args.replay_pool_samples, args.replay_samples) < 1:
            raise ValueError("Replay sample counts must be positive")
        if args.stream_train_samples != args.replay_samples:
            raise ValueError("Replay mode currently requires a 1:1 exposure ratio")
        if args.replay_pool_samples > args.stream_train_samples:
            raise ValueError("Replay pool cannot be larger than coverage")
        if args.stream_train_samples + args.replay_samples != args.train_samples:
            raise ValueError(
                "Coverage plus replay samples must equal --train-samples"
            )
        if args.batch_size % 2:
            raise ValueError("Replay mode requires an even batch size")
        if args.stream_train_samples % (args.batch_size // 2):
            raise ValueError(
                "Coverage samples must be divisible by half the batch size"
            )
        if args.epochs != 1:
            raise ValueError("Replay mode is one deterministic logical epoch")
    if args.stream_train_samples is not None:
        if args.stream_train_samples < 1:
            raise ValueError("--stream-train-samples must be positive")
        if (
            not replay_mode
            and args.epochs * args.stream_train_samples < args.train_samples
        ):
            raise ValueError(
                "epochs * stream-train-samples is smaller than train-samples"
            )
    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    _, payload = load_grounded_graph_checkpoint(args.grounded_graph_checkpoint, device)
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(payload["grounded_selector_checkpoint"]), device
    )
    model.vision_encoder.requires_grad_(False)
    initial = (
        torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.init_checkpoint
        else None
    )
    initial_config = (
        DenseBondObjectConfig(**initial["config"])
        if initial is not None
        else DenseBondObjectConfig(input_channels=model.config.vision_dims[0])
    )
    config = DenseBondObjectConfig(
        input_channels=model.config.vision_dims[0],
        hidden_channels=initial_config.hidden_channels,
        slots=args.slots or initial_config.slots,
        max_endpoint_offset=initial_config.max_endpoint_offset,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    detector = (
        MultiScaleDenseBondObjectDetector(
            config, preserve_base_types=args.preserve_base_types
        ).to(device)
        if args.multiscale
        else DenseBondObjectDetector(config).to(device)
    )
    if initial is not None:
        state = initial["dense_bond_object"]
        if isinstance(detector, MultiScaleDenseBondObjectDetector) and not any(
            key.startswith("base.") for key in state
        ):
            detector.base.load_state_dict(
                expand_compatible_state_dict(detector.base.state_dict(), state)
            )
        else:
            detector.load_state_dict(
                expand_compatible_state_dict(detector.state_dict(), state)
            )
    if args.multiscale_only:
        if not isinstance(detector, MultiScaleDenseBondObjectDetector):
            raise ValueError("--multiscale-only requires --multiscale")
        detector.requires_grad_(False)
        detector.stride8_adapter.requires_grad_(True)
        detector.stride16_adapter.requires_grad_(True)
    trainable_parameters = [
        parameter for parameter in detector.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=0.0
    )
    collator = ChemSceneBondCollator()

    def cache(shards: list[str], count: int) -> list[dict]:
        return list(
            islice(
                iter(
                    ChemVectorIterableDataset(
                        expand_shards(shards), sequence_version="0.4"
                    )
                ),
                count,
            )
        )

    train_records = (
        None
        if args.stream_train_samples is not None
        else cache(args.train_shards, args.train_cache_samples)
    )
    if train_records is not None and args.replay_shards is not None:
        train_records.extend(
            cache(args.replay_shards, args.replay_cache_samples)
        )
    validation_records = cache(args.validation_shards, args.validation_samples)
    retention_validation_records = (
        cache(
            args.retention_validation_shards,
            args.retention_validation_samples,
        )
        if args.retention_validation_shards is not None
        else None
    )
    if (
        train_records is not None
        and len(train_records)
        != args.train_cache_samples + (args.replay_cache_samples or 0)
    ) or len(validation_records) != args.validation_samples:
        raise ValueError("Dense Bond Object cache is incomplete")
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "architecture": (
                    "multiscale_dense_bond_object" if args.multiscale else "dense_bond_object"
                ),
                "dense_config": asdict(config),
                "trainable_parameters": sum(p.numel() for p in trainable_parameters),
                "preserve_base_types": args.preserve_base_types,
                "sampling_mode": "coverage_replay" if replay_mode else "legacy",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    initial_metrics = evaluate(
        model,
        detector,
        validation_records,
        collator,
        device,
        args.batch_size,
        args.threshold,
    )
    initial_retention_metrics = (
        evaluate(
            model,
            detector,
            retention_validation_records,
            collator,
            device,
            args.batch_size,
            args.threshold,
        )
        if retention_validation_records is not None
        else None
    )
    initial_checkpoint = output_dir / "initial.pt"
    torch.save(
        {
            "dense_bond_object": detector.state_dict(),
            "config": asdict(config),
            "samples_seen": 0,
            "validation": initial_metrics,
            "retention_validation": initial_retention_metrics,
            "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
            "init_checkpoint": args.init_checkpoint,
            "seed": args.seed,
        },
        initial_checkpoint,
    )
    (output_dir / "initial_validation.json").write_text(
        json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8"
    )
    if initial_retention_metrics is not None:
        (output_dir / "initial_retention_validation.json").write_text(
            json.dumps(initial_retention_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    shutil.copy2(initial_checkpoint, output_dir / "best.pt")
    print("initial_validation=" + json.dumps(initial_metrics), flush=True)
    if initial_retention_metrics is not None:
        print(
            "initial_retention_validation="
            + json.dumps(initial_retention_metrics),
            flush=True,
        )

    rng = random.Random(args.seed)
    samples_seen = 0
    next_validation = args.validate_every
    best = dual_domain_validation_score(
        initial_metrics, initial_retention_metrics
    )
    running_loss = 0.0
    running_samples = 0
    stop = False
    sampling_audit = (
        MixedSamplingAudit(
            coverage_samples=args.stream_train_samples,
            replay_pool_samples=args.replay_pool_samples,
            replay_samples=args.replay_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if replay_mode
        else None
    )
    for epoch in range(args.epochs):
        if samples_seen >= args.train_samples:
            break
        if replay_mode:
            batches = stream_mixed_training_batches(
                args.train_shards,
                coverage_samples=args.stream_train_samples,
                replay_pool_samples=args.replay_pool_samples,
                replay_samples=args.replay_samples,
                batch_size=args.batch_size,
                shuffle_buffer=args.stream_shuffle_buffer,
                seed=args.seed,
                audit=sampling_audit,
            )
        elif train_records is None:
            epoch_samples = min(
                args.stream_train_samples,
                args.train_samples - samples_seen,
            )
            batches = stream_training_batches(
                args.train_shards,
                samples=epoch_samples,
                batch_size=args.batch_size,
                shuffle_buffer=args.stream_shuffle_buffer,
                seed=args.seed + epoch,
            )
        else:
            order = list(range(len(train_records)))
            rng.shuffle(order)
            batches = (
                [train_records[index] for index in order[start : start + args.batch_size]]
                for start in range(0, len(order), args.batch_size)
            )
        for batch_records in batches:
            detector.train()
            remaining = args.train_samples - samples_seen
            if remaining <= 0:
                stop = True
                break
            batch_records = batch_records[:remaining]
            batch = collator(batch_records)
            pixels = batch["pixel_values"].to(device)
            targets = stack_targets(batch, device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                features = _extract_detector_features(model, detector, pixels)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = _apply_detector(detector, features)
                losses = dense_bond_object_loss(
                    output,
                    targets,
                    endpoint_barrier_weight=args.endpoint_barrier_weight,
                    endpoint_barrier_margin_pixels=args.endpoint_barrier_margin_pixels,
                    heatmap_set_weight=args.heatmap_set_weight,
                    heatmap_set_margin=args.heatmap_set_margin,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(batch_records)
            running_loss += float(losses["loss"].detach()) * len(batch_records)
            running_samples += len(batch_records)
            if samples_seen // args.log_every != (
                samples_seen - len(batch_records)
            ) // args.log_every:
                print(
                    f"samples={samples_seen} loss={running_loss / running_samples:.6f}",
                    flush=True,
                )
                running_loss = 0.0
                running_samples = 0
            if samples_seen >= next_validation:
                metrics = evaluate(
                    model,
                    detector,
                    validation_records,
                    collator,
                    device,
                    args.batch_size,
                    args.threshold,
                )
                retention_metrics = (
                    evaluate(
                        model,
                        detector,
                        retention_validation_records,
                        collator,
                        device,
                        args.batch_size,
                        args.threshold,
                    )
                    if retention_validation_records is not None
                    else None
                )
                print("validation=" + json.dumps(metrics), flush=True)
                if retention_metrics is not None:
                    print(
                        "retention_validation=" + json.dumps(retention_metrics),
                        flush=True,
                    )
                checkpoint = output_dir / f"checkpoint-{samples_seen:07d}.pt"
                torch.save(
                    {
                        "dense_bond_object": detector.state_dict(),
                        "config": asdict(config),
                        "samples_seen": samples_seen,
                        "validation": metrics,
                        "retention_validation": retention_metrics,
                        "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
                        "init_checkpoint": args.init_checkpoint,
                        "seed": args.seed,
                    },
                    checkpoint,
                )
                score = dual_domain_validation_score(metrics, retention_metrics)
                if score > best:
                    best = score
                    shutil.copy2(checkpoint, output_dir / "best.pt")
                while next_validation <= samples_seen:
                    next_validation += args.validate_every
        if stop:
            break

    if sampling_audit is not None:
        audit_payload = sampling_audit.to_dict()
        (output_dir / "sampling_audit.json").write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if audit_payload["status"] != "passed":
            raise RuntimeError(
                "Mixed sampling audit failed: " + "; ".join(audit_payload["issues"])
            )

    metrics = evaluate(
        model,
        detector,
        validation_records,
        collator,
        device,
        args.batch_size,
        args.threshold,
    )
    retention_metrics = (
        evaluate(
            model,
            detector,
            retention_validation_records,
            collator,
            device,
            args.batch_size,
            args.threshold,
        )
        if retention_validation_records is not None
        else None
    )
    (output_dir / "validation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    if retention_metrics is not None:
        (output_dir / "retention_validation.json").write_text(
            json.dumps(retention_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    torch.save(
        {
            "dense_bond_object": detector.state_dict(),
            "config": asdict(config),
            "samples_seen": samples_seen,
            "validation": metrics,
            "retention_validation": retention_metrics,
            "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
            "init_checkpoint": args.init_checkpoint,
            "seed": args.seed,
        },
        output_dir / "last.pt",
    )
    print("validation=" + json.dumps(metrics), flush=True)
    if retention_metrics is not None:
        print("retention_validation=" + json.dumps(retention_metrics), flush=True)


if __name__ == "__main__":
    main()
