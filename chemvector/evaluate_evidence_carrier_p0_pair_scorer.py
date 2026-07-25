#!/usr/bin/env python3
"""One-pass validation for the frozen-P0 conditioned pair scorer.

This is an evaluation-only entry point.  The pair threshold comes exclusively
from the train/cal checkpoint.  The upstream P0 checkpoint, base pair cache,
and P0 augmentation cache are all verified against live SHA256 digests before
any validation image is decoded.

The fixed-step upstream P0 checkpoint contains metrics computed on the same
500-image validation split, although that checkpoint was selected by its
explicit step filename rather than those metrics.  Results from this entry
point are therefore conservatively classified as component-conditioned
pipeline regression, not independent generalization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIRING_PROFILE_TRAINING_WIDE,
    ConditionalOrdinaryPairWebDataset,
    ConditionalPairSampleIdentity,
    build_conditional_ordinary_pair_target,
    conditional_pair_pairing_config,
    validate_molecule_disjoint_splits,
)
from chemvector.evidence_carrier_extent_data import (
    EvidenceCarrierExtentTargetConfig,
    reject_internal_test_sample,
    reject_internal_test_shards,
)
from chemvector.evidence_carrier_p0_pair_cache import (
    load_evidence_carrier_p0_pair_cache,
)
from chemvector.evidence_carrier_p0_pair_scorer import (
    EvidenceCarrierP0PairScorer,
    extract_frozen_p0_anchor_evidence,
    p0_decoded_pair_consistency_features,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    score_evidence_carrier_pair_record,
)
from chemvector.evaluate_evidence_carrier_pair_scorer import (
    EXPECTED_RENDER_VARIANTS,
    pair_scorer_validation_gate,
)
from chemvector.train_evidence_carrier_extent import (
    file_sha256,
    load_checkpoint,
)
from chemvector.train_evidence_carrier_p0_pair_scorer import (
    load_p0_pair_scorer_checkpoint,
    verify_frozen_p0_provenance,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    load_train_only_cache,
    pair_cache_records,
)


EVALUATION_SCHEMA_VERSION = "evidence-carrier-p0-pair-scorer-validation-0.1"
FORMAL_VALIDATION_SAMPLE_COUNT = 500
FORMAL_IMAGE_SIZE = 768
VALIDATION_INTERPRETATION = "component-conditioned pipeline regression"


def _contains_internal_test(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return "internal_test" in normalized


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.suffix.lower() != ".json":
        path = path / "validation.json"
    return path.resolve(strict=False)


def _expand_validation_shards(patterns: Sequence[str]) -> list[str]:
    reject_internal_test_shards(patterns)
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        paths.extend(matches or [str(Path(pattern).expanduser())])
    resolved = []
    for value in paths:
        path = Path(value).resolve(strict=False)
        if _contains_internal_test(path):
            raise ValueError("Frozen-P0 pair evaluation cannot read internal_test")
        if not path.is_file():
            raise FileNotFoundError(f"Validation shard does not exist: {path}")
        resolved.append(str(path))
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Validation shard expansion is empty or contains duplicates")
    return resolved


def _shard_descriptors(paths: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(Path(path).resolve()),
            "size_bytes": Path(path).stat().st_size,
        }
        for path in paths
    ]


def _live_source(
    checkpoint: Mapping[str, Any],
    path_name: str,
    sha_name: str,
) -> tuple[Path, str]:
    raw_path = checkpoint.get(path_name)
    expected_sha = checkpoint.get(sha_name)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Frozen-P0 pair checkpoint lacks {path_name}")
    path = Path(raw_path).expanduser().resolve(strict=False)
    if _contains_internal_test(raw_path) or _contains_internal_test(path):
        raise ValueError("Frozen-P0 pair provenance cannot reference internal_test")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen-P0 pair provenance file is missing: {path}")
    actual_sha = file_sha256(path)
    if expected_sha != actual_sha:
        raise ValueError(f"Frozen-P0 pair checkpoint {sha_name} no longer matches")
    return path, actual_sha


def _same_resolved_path(first: Any, second: Any) -> bool:
    if not isinstance(first, str) or not isinstance(second, str):
        return False
    return Path(first).expanduser().resolve(strict=False) == Path(second).expanduser().resolve(
        strict=False
    )


def _train_cache_identities(
    cache: Mapping[str, Any],
) -> list[ConditionalPairSampleIdentity]:
    identities = []
    seen: set[str] = set()
    for partition in ("fit", "calibration"):
        for record in pair_cache_records(cache, partition):
            payload = record.get("identity")
            if not isinstance(payload, Mapping):
                raise ValueError("Frozen-P0 pair base cache lacks identity metadata")
            identity = ConditionalPairSampleIdentity(**dict(payload))
            if identity.source_split != "train" or identity.sample_key in seen:
                raise ValueError("Frozen-P0 pair base cache identity contract differs")
            seen.add(identity.sample_key)
            identities.append(identity)
    return identities


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_frozen_p0_pair_pipeline(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    require_train_cal_go: bool = True,
) -> dict[str, Any]:
    """Load B and its exact train-time P0/cache provenance as one unit."""

    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if _contains_internal_test(checkpoint_path) or _contains_internal_test(path):
        raise ValueError("Frozen-P0 pair checkpoint cannot reference internal_test")
    if not path.is_file():
        raise FileNotFoundError(f"Frozen-P0 pair checkpoint is missing: {path}")
    if path.name != "best.pt":
        raise ValueError("Formal frozen-P0 pair evaluation requires best.pt")
    pair_model, checkpoint = load_p0_pair_scorer_checkpoint(path, device=device)
    threshold = checkpoint.get("gain_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("Frozen-P0 pair checkpoint threshold is invalid")
    if not isinstance(require_train_cal_go, bool):
        raise ValueError("require_train_cal_go must be boolean")
    if (
        require_train_cal_go
        and checkpoint.get("train_cal_gate", {}).get("status") != "GO"
    ):
        raise ValueError("Frozen-P0 pair checkpoint did not pass train/cal gate")

    base_path, base_sha = _live_source(
        checkpoint, "base_cache_path", "base_cache_sha256"
    )
    augmentation_path, augmentation_sha = _live_source(
        checkpoint, "augmentation_cache_path", "augmentation_cache_sha256"
    )
    p0_path, p0_sha = _live_source(
        checkpoint, "frozen_p0_checkpoint_path", "frozen_p0_checkpoint_sha256"
    )

    base_cache = load_train_only_cache(base_path)
    if base_cache.get("cache_sha256") != base_sha:
        raise ValueError("Loaded base cache digest differs from B checkpoint")
    augmentation_cache = load_evidence_carrier_p0_pair_cache(
        augmentation_path,
        base_cache=base_cache,
        require_complete=True,
    )
    if augmentation_cache.get("cache_sha256") != augmentation_sha:
        raise ValueError("Loaded P0 augmentation digest differs from B checkpoint")
    if augmentation_cache.get("base_cache_sha256") != base_sha or not _same_resolved_path(
        augmentation_cache.get("base_cache_path"), str(base_path)
    ):
        raise ValueError("P0 augmentation is not bound to the checkpoint base cache")
    if augmentation_cache.get("p0_checkpoint_sha256") != p0_sha or not _same_resolved_path(
        augmentation_cache.get("p0_checkpoint_path"), str(p0_path)
    ):
        raise ValueError("P0 augmentation is not bound to the checkpoint P0 weights")
    if augmentation_cache.get("pair_consistency_config") != asdict(pair_model.config):
        raise ValueError("B scorer consistency config differs from its P0 augmentation")
    p0_metadata_links = {
        "p0_global_step": augmentation_cache.get("p0_global_step"),
        "p0_model_type": augmentation_cache.get("p0_model_type"),
        "p0_checkpoint_schema_version": augmentation_cache.get(
            "p0_checkpoint_schema_version"
        ),
        "p0_checkpoint_contains_validation_metrics": augmentation_cache.get(
            "p0_checkpoint_contains_validation_metrics"
        ),
        "p0_checkpoint_selected_by_validation": augmentation_cache.get(
            "p0_checkpoint_selected_by_validation"
        ),
    }
    mismatched_links = {
        name: (checkpoint.get(name), expected)
        for name, expected in p0_metadata_links.items()
        if checkpoint.get(name) != expected
    }
    if mismatched_links:
        raise ValueError(
            f"B checkpoint and P0 augmentation metadata differ: {mismatched_links}"
        )

    p0_provenance = verify_frozen_p0_provenance(augmentation_cache)
    expected_upstream_flags = {
        "upstream_p0_validation_read": p0_provenance["upstream_validation_read"],
        "upstream_p0_checkpoint_selected_by_validation": p0_provenance[
            "checkpoint_selected_by_validation"
        ],
    }
    actual_upstream_flags = {
        name: checkpoint.get(name) for name in expected_upstream_flags
    }
    if actual_upstream_flags != expected_upstream_flags:
        raise ValueError("B checkpoint upstream P0 provenance flags differ from live P0")
    if not p0_provenance["upstream_validation_read"]:
        raise ValueError(
            "This formal protocol is specifically classified as component-conditioned; "
            "the live P0 provenance does not record the shared validation dependency"
        )

    p0_model, target_config, p0_checkpoint = load_checkpoint(p0_path, device=device)
    if asdict(target_config) != asdict(EvidenceCarrierExtentTargetConfig()):
        raise ValueError("Frozen-P0 pair evaluation requires the default P0 target protocol")
    if (
        augmentation_cache.get("p0_model_type") != p0_checkpoint.get("model_type")
        or augmentation_cache.get("p0_global_step") != p0_checkpoint.get("global_step")
        or augmentation_cache.get("p0_checkpoint_schema_version")
        != p0_checkpoint.get("checkpoint_schema_version")
    ):
        raise ValueError("Live P0 checkpoint metadata differs from its augmentation cache")
    if int(getattr(p0_model.config, "hidden_dim", -1)) != int(
        checkpoint.get("anchor_feature_dim", -2)
    ):
        raise ValueError("Live P0 feature width differs from the B scorer contract")
    _freeze(pair_model)
    _freeze(p0_model)
    return {
        "pair_model": pair_model,
        "pair_checkpoint": checkpoint,
        "pair_checkpoint_path": path,
        "pair_checkpoint_sha256": file_sha256(path),
        "pair_threshold": float(threshold),
        "p0_model": p0_model,
        "p0_checkpoint": p0_checkpoint,
        "p0_checkpoint_path": p0_path,
        "p0_checkpoint_sha256": p0_sha,
        "target_config": target_config,
        "base_cache": base_cache,
        "base_cache_path": base_path,
        "base_cache_sha256": base_sha,
        "augmentation_cache": augmentation_cache,
        "augmentation_cache_path": augmentation_path,
        "augmentation_cache_sha256": augmentation_sha,
        "p0_provenance": p0_provenance,
    }


def _native_image_tensor(image: Image.Image, *, expected_image_size: int) -> torch.Tensor:
    if image.size != (expected_image_size, expected_image_size):
        raise ValueError(
            "Frozen-P0 pair validation requires the native raster: "
            f"got {image.size}, expected {(expected_image_size, expected_image_size)}"
        )
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


@torch.inference_mode()
def score_frozen_p0_pair_target(
    pair_model: EvidenceCarrierP0PairScorer,
    p0_model: torch.nn.Module,
    image: Image.Image,
    target: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    expected_image_size: int = FORMAL_IMAGE_SIZE,
) -> dict[str, Any]:
    """Extract live frozen-P0 evidence and score one unpadded pair target."""

    anchors = target["anchor_segments"]
    pairs = target["pair_indices"]
    pair_features = target["pair_features"]
    centerlines = target["pair_overlap_centerline_pixels"]
    if anchors.ndim != 3 or anchors.shape[1:] != (2, 2):
        raise ValueError("Frozen-P0 pair anchors must have shape [anchors,2,2]")
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise ValueError("Frozen-P0 pair indices must have shape [pairs,2]")
    pair_count = len(pairs)
    anchor_count = len(anchors)
    if pair_features.shape != (pair_count, 8) or centerlines.shape != (
        pair_count,
        2,
        2,
    ):
        raise ValueError("Frozen-P0 pair target feature shapes differ")

    anchor_capacity = max(anchor_count, 1)
    pair_capacity = max(pair_count, 1)
    padded_anchors = torch.zeros((1, anchor_capacity, 2, 2), dtype=torch.float32)
    anchor_mask = torch.zeros((1, anchor_capacity), dtype=torch.bool)
    padded_pairs = torch.full((1, pair_capacity, 2), -1, dtype=torch.long)
    pair_mask = torch.zeros((1, pair_capacity), dtype=torch.bool)
    padded_features = torch.zeros((1, pair_capacity, 8), dtype=torch.float32)
    padded_centerlines = torch.zeros((1, pair_capacity, 2, 2), dtype=torch.float32)
    if anchor_count:
        padded_anchors[0, :anchor_count] = anchors.float()
        anchor_mask[0, :anchor_count] = True
    if pair_count:
        padded_pairs[0, :pair_count] = pairs.long()
        pair_mask[0, :pair_count] = True
        padded_features[0, :pair_count] = pair_features.float()
        padded_centerlines[0, :pair_count] = centerlines.float()

    pixels = _native_image_tensor(
        image, expected_image_size=expected_image_size
    ).unsqueeze(0)
    canvas = torch.tensor(
        [[float(image.width), float(image.height)]], dtype=torch.float32
    )
    pixels = pixels.to(device)
    padded_anchors = padded_anchors.to(device)
    anchor_mask = anchor_mask.to(device)
    padded_pairs = padded_pairs.to(device)
    pair_mask = pair_mask.to(device)
    padded_features = padded_features.to(device)
    padded_centerlines = padded_centerlines.to(device)
    canvas = canvas.to(device)

    with torch.autocast(device_type=device.type, enabled=False):
        evidence = extract_frozen_p0_anchor_evidence(
            p0_model,
            pixels,
            padded_anchors,
            anchor_mask,
            coordinate_mode="normalized",
        )
        anchor_features = evidence["anchor_features"]
        decoded = evidence["decoded_segments_pixels"]
        if anchor_features.dtype != torch.float32 or decoded.dtype != torch.float32:
            raise RuntimeError("Formal frozen-P0 pair extraction must remain FP32")
        consistency = p0_decoded_pair_consistency_features(
            decoded,
            padded_pairs,
            padded_centerlines,
            pair_mask,
            canvas,
            config=pair_model.config,
        )
        if consistency.dtype != torch.float32:
            raise RuntimeError("Formal frozen-P0 pair consistency must remain FP32")
    output = pair_model(
        anchor_features,
        padded_pairs,
        padded_features,
        consistency,
        pair_mask,
    )
    scores = output["pair_gains"][0, :pair_count].detach().float().cpu()
    return {
        "pair_scores": scores,
        "decoded_segments_pixels": decoded[0, :anchor_count]
        .detach()
        .float()
        .cpu(),
        "anchor_count": anchor_count,
        "pair_count": pair_count,
        "p0_forward_passes": 1,
        "consistency_feature_count": int(consistency.size(-1)),
    }


@torch.inference_mode()
def evaluate_frozen_p0_pair_sample(
    pair_model: EvidenceCarrierP0PairScorer,
    p0_model: torch.nn.Module,
    sample: Mapping[str, Any],
    *,
    threshold: float,
    pairing_config: Any,
    target_config: EvidenceCarrierExtentTargetConfig,
    device: torch.device,
    expected_image_size: int = FORMAL_IMAGE_SIZE,
) -> dict[str, Any]:
    reject_internal_test_sample(dict(sample))
    identity = sample.get("identity")
    if not isinstance(identity, ConditionalPairSampleIdentity):
        raise ValueError("Frozen-P0 pair validation sample lacks a validated identity")
    image = sample.get("image")
    primitives = sample.get("primitives")
    layers = sample.get("layers")
    if not isinstance(image, Image.Image) or not isinstance(primitives, dict) or not isinstance(
        layers, dict
    ):
        raise ValueError("Frozen-P0 pair validation sample lacks image or sidecars")
    target = build_conditional_ordinary_pair_target(
        image,
        primitives,
        layers,
        target_config=target_config,
        pairing_config=pairing_config,
    )
    scored = score_frozen_p0_pair_target(
        pair_model,
        p0_model,
        image,
        target,
        device=device,
        expected_image_size=expected_image_size,
    )
    evaluation = score_evidence_carrier_pair_record(
        target,
        scored["pair_scores"],
        threshold,
        verify_permutation=True,
    )
    return {
        "identity": identity,
        "target": target,
        "pair_evaluation": evaluation,
        "target_builds": 1,
        **scored,
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    validation_paths = _expand_validation_shards(args.validation_shards)
    output = _output_path(args.output)
    if output.exists():
        raise FileExistsError(f"Frozen-P0 pair validation output exists: {output}")
    device = _device(args.device)
    pipeline = load_frozen_p0_pair_pipeline(args.checkpoint, device=device)
    pairing_config = conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
    checkpoint = pipeline["pair_checkpoint"]
    if checkpoint.get("pairing_config") != asdict(pairing_config):
        raise ValueError("Frozen-P0 pair checkpoint candidate profile differs")

    dataset = ConditionalOrdinaryPairWebDataset(
        validation_paths,
        split="validation",
        render_variant=None,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        validate_hashes=True,
        limit=FORMAL_VALIDATION_SAMPLE_COUNT,
    )
    records = []
    identities = []
    seen: set[str] = set()
    for sample in dataset:
        result = evaluate_frozen_p0_pair_sample(
            pipeline["pair_model"],
            pipeline["p0_model"],
            sample,
            threshold=pipeline["pair_threshold"],
            pairing_config=pairing_config,
            target_config=pipeline["target_config"],
            device=device,
        )
        identity = result["identity"]
        if identity.sample_key in seen:
            raise ValueError("Frozen-P0 pair validation returned a duplicate sample")
        seen.add(identity.sample_key)
        identities.append(identity)
        records.append(result)
    if len(records) != FORMAL_VALIDATION_SAMPLE_COUNT:
        raise ValueError(
            f"Formal frozen-P0 pair validation requires exactly "
            f"{FORMAL_VALIDATION_SAMPLE_COUNT} samples, got {len(records)}"
        )

    train_identities = _train_cache_identities(pipeline["base_cache"])
    identity_audit = validate_molecule_disjoint_splits(
        train_identities,
        identities,
        left_name="train_base_cache",
        right_name="validation",
    )
    metrics = aggregate_evidence_carrier_pair_evaluations(
        [record["pair_evaluation"] for record in records]
    )
    render_variants = {identity.render_variant for identity in identities}
    gate = pair_scorer_validation_gate(
        metrics,
        sample_count=len(records),
        expected_sample_count=FORMAL_VALIDATION_SAMPLE_COUNT,
        render_variants=render_variants,
        identity_audit=identity_audit,
        checkpoint=checkpoint,
    )
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": gate["status"],
        "interpretation": {
            "classification": VALIDATION_INTERPRETATION,
            "independent_generalization": False,
            "reason": (
                "The fixed-step upstream P0 checkpoint contains metrics computed "
                "on this same 500-image validation split; its weights were selected "
                "by explicit step, not by those metrics."
            ),
            "upstream_p0_validation_read": True,
            "upstream_p0_checkpoint_selected_by_validation": pipeline[
                "p0_provenance"
            ]["checkpoint_selected_by_validation"],
        },
        "protocol": {
            "evaluation_only": True,
            "formal_sample_count": FORMAL_VALIDATION_SAMPLE_COUNT,
            "validation_passes": 1,
            "target_builds_per_sample": 1,
            "p0_evidence_forwards_per_sample": 1,
            "pairing_profile": PAIRING_PROFILE_TRAINING_WIDE,
            "gain_threshold_source": "frozen_train_cal_checkpoint",
            "threshold_recalibrated_on_validation": False,
            "checkpoint_modified": False,
            "internal_test_read": False,
        },
        "checkpoint": {
            "path": str(pipeline["pair_checkpoint_path"]),
            "sha256": pipeline["pair_checkpoint_sha256"],
            "model_type": checkpoint.get("model_type"),
            "parameter_count": checkpoint.get("parameter_count"),
            "gain_threshold": pipeline["pair_threshold"],
            "train_cal_gate": checkpoint.get("train_cal_gate"),
        },
        "provenance": {
            "base_cache": {
                "path": str(pipeline["base_cache_path"]),
                "sha256": pipeline["base_cache_sha256"],
            },
            "augmentation_cache": {
                "path": str(pipeline["augmentation_cache_path"]),
                "sha256": pipeline["augmentation_cache_sha256"],
                "complete": pipeline["augmentation_cache"].get("complete"),
                "anchor_feature_dtype": pipeline["augmentation_cache"].get(
                    "anchor_feature_dtype"
                ),
                "pair_consistency_config": pipeline["augmentation_cache"].get(
                    "pair_consistency_config"
                ),
                "p0_extraction_provenance": pipeline["augmentation_cache"].get(
                    "p0_extraction_provenance"
                ),
                "p0_checkpoint_selection": pipeline["augmentation_cache"].get(
                    "p0_checkpoint_selection"
                ),
            },
            "frozen_p0_checkpoint": {
                "path": str(pipeline["p0_checkpoint_path"]),
                "sha256": pipeline["p0_checkpoint_sha256"],
                "model_type": pipeline["p0_checkpoint"].get("model_type"),
                "global_step": pipeline["p0_checkpoint"].get("global_step"),
                **pipeline["p0_provenance"],
            },
        },
        "validation_shards": _shard_descriptors(validation_paths),
        "identity": {
            "validation_sample_count": len(records),
            "validation_molecule_count": len({item.molecule_id for item in identities}),
            "validation_source_structure_count": len(
                {item.source_structure_id for item in identities}
            ),
            "render_variants": sorted(render_variants),
            "train_validation_zero_overlap": identity_audit,
        },
        "metrics": metrics,
        "go_gate": gate,
        "sample_evaluations": [record["pair_evaluation"] for record in records],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    values = [args.checkpoint, args.output, *args.validation_shards]
    if any(_contains_internal_test(value) for value in values):
        raise ValueError("Frozen-P0 pair evaluation paths cannot reference internal_test")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen-P0 pair checkpoint is missing: {checkpoint}")
    if _contains_internal_test(_output_path(args.output)):
        raise ValueError("Frozen-P0 pair output cannot reference internal_test")
    _device(args.device)


def main() -> None:
    payload = evaluate(build_parser().parse_args())
    print(json.dumps(_plain_json(payload), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "FORMAL_IMAGE_SIZE",
    "FORMAL_VALIDATION_SAMPLE_COUNT",
    "VALIDATION_INTERPRETATION",
    "build_parser",
    "evaluate",
    "evaluate_frozen_p0_pair_sample",
    "load_frozen_p0_pair_pipeline",
    "score_frozen_p0_pair_target",
]
