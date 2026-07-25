"""Conditional ordinary-anchor pair data for Evidence Carrier v0.9.10.

The experiment deliberately uses a GT router oracle: raw LSD anchors are first
associated with Visual SVG sidecars and only positive ordinary anchors enter
the pair stage.  Pair candidate generation itself accepts only raster pixels,
anchor segments, and a target-free geometry configuration.  Target line IDs
are exposed only afterwards to label compatible candidates as same-target or
distinct-target hard negatives.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
import torch
from torch.utils.data import IterableDataset
import webdataset as wds

from chemvector import evidence_carrier_stroke_pairing as _stroke_pairing
from chemvector.evidence_carrier_extent_data import (
    IGNORE_LABEL,
    POSITIVE_LABEL,
    EvidenceCarrierExtentTargetConfig,
    associate_evidence_carriers,
    detect_evidence_carriers,
    reject_internal_test_sample,
    reject_internal_test_shards,
)
from chemvector.evidence_carrier_stroke_pairing import StrokeBoundaryPairingConfig
from chemvector.visual_svg_data import VisualSVGCollator


SCHEMA_VERSION = "evidence-carrier-conditional-ordinary-pair-0.1"
PAIR_FEATURE_NAMES = (
    "angle_degrees",
    "normal_separation_pixels",
    "overlap_pixels",
    "shorter_interval_overlap",
    "interval_iou",
    "interior_ink_occupancy",
    "interior_mean_darkness",
    "center_mean_darkness",
)
PAIR_FEATURE_DIM = len(PAIR_FEATURE_NAMES)
PAIR_NEGATIVE_LABEL = 0
PAIR_POSITIVE_LABEL = 1
PAIR_PADDING_LABEL = -1
PAIR_PADDING_INDEX = -1
PAIRING_PROFILE_STRICT = "strict"
PAIRING_PROFILE_TRAINING_WIDE = "training_wide"
PAIRING_PROFILES = (
    PAIRING_PROFILE_STRICT,
    PAIRING_PROFILE_TRAINING_WIDE,
)
SplitName = Literal["train", "validation"]


def _contains_internal_test(value: Any) -> bool:
    return "internal_test" in str(value).lower().replace("-", "_")


def _sha256_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_split(value: Any) -> SplitName:
    if not isinstance(value, str) or value not in {"train", "validation"}:
        raise ValueError("Conditional pair split must be 'train' or 'validation'")
    return value


def _strict_variant(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Conditional pair render_variant must be a nonempty string")
    if _contains_internal_test(value):
        raise ValueError("Conditional pair render_variant cannot reference internal_test")
    return value


def conditional_pair_pairing_config(
    profile: str = PAIRING_PROFILE_STRICT,
) -> StrokeBoundaryPairingConfig:
    """Return a named, immutable candidate-generation profile.

    ``strict`` is the original v0.9.9 audit baseline.  ``training_wide`` only
    expands target-free geometry/raster support so hard negatives can enter the
    conditional scorer dataset; it does not inspect target IDs.
    """

    if profile == PAIRING_PROFILE_STRICT:
        return StrokeBoundaryPairingConfig()
    if profile == PAIRING_PROFILE_TRAINING_WIDE:
        return replace(
            StrokeBoundaryPairingConfig(),
            maximum_angle_degrees=8.0,
            minimum_normal_separation_pixels=0.5,
            maximum_normal_separation_pixels=8.0,
            minimum_overlap_pixels=4.0,
            minimum_shorter_interval_overlap=0.35,
            minimum_interior_ink_occupancy=0.25,
        )
    raise ValueError(
        f"Unsupported conditional pair profile {profile!r}; expected one of "
        f"{PAIRING_PROFILES}"
    )


@dataclass(frozen=True, slots=True)
class ConditionalPairSampleIdentity:
    """Leakage-relevant identity for one raster render."""

    sample_key: str
    molecule_id: str
    source_structure_id: str
    source_pilot: str
    source_split: SplitName
    source_record: str
    render_variant: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def conditional_pair_sample_identity(
    sample: dict[str, Any],
    *,
    expected_split: SplitName | None = None,
    expected_render_variant: str | None = None,
) -> ConditionalPairSampleIdentity:
    """Validate and derive molecule/source identity without reading structure GT."""

    reject_internal_test_sample(sample)
    meta = sample.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Conditional pair sample lacks metadata")
    if _contains_internal_test(json.dumps(meta, sort_keys=True, ensure_ascii=True)):
        raise ValueError("Conditional pair metadata references sealed internal_test")

    sample_key = str(sample.get("key", sample.get("__key__", "")))
    if not sample_key:
        raise ValueError("Conditional pair sample_key is empty")
    if str(meta.get("sample_key", sample_key)) != sample_key:
        raise ValueError("Conditional pair sample key and metadata disagree")

    split = _strict_split(str(meta.get("split", "")))
    source_split = _strict_split(str(meta.get("source_split", "")))
    if split != source_split:
        raise ValueError("Conditional pair split and source_split disagree")
    if expected_split is not None and split != _strict_split(expected_split):
        raise ValueError(
            f"Conditional pair sample split {split!r} differs from expected "
            f"{expected_split!r}"
        )

    variant = _strict_variant(str(meta.get("source_variant", "")))
    declared_variant = _strict_variant(str(meta.get("variant", "")))
    if variant != declared_variant:
        raise ValueError("Conditional pair source_variant and variant disagree")
    if (
        expected_render_variant is not None
        and variant != _strict_variant(expected_render_variant)
    ):
        raise ValueError(
            f"Conditional pair variant {variant!r} differs from expected "
            f"{expected_render_variant!r}"
        )

    molecule_id = str(meta.get("molecule_id", ""))
    source_pilot = str(meta.get("source_pilot", ""))
    source_sample_key = str(meta.get("source_sample_key", ""))
    missing = [
        name
        for name, value in (
            ("molecule_id", molecule_id),
            ("source_pilot", source_pilot),
            ("source_sample_key", source_sample_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Conditional pair identity fields are missing: {missing}")
    suffix = f"_{variant}"
    if not source_sample_key.endswith(suffix):
        raise ValueError(
            "Conditional pair source_sample_key does not end in source_variant"
        )
    source_record = source_sample_key[: -len(suffix)]
    if not source_record:
        raise ValueError("Conditional pair derived source record is empty")
    source_structure_id = f"{source_pilot}|{source_split}|{source_record}"
    return ConditionalPairSampleIdentity(
        sample_key=sample_key,
        molecule_id=molecule_id,
        source_structure_id=source_structure_id,
        source_pilot=source_pilot,
        source_split=source_split,
        source_record=source_record,
        render_variant=variant,
    )


@dataclass(frozen=True, slots=True)
class ConditionalPairIdentityPartition:
    fit: tuple[ConditionalPairSampleIdentity, ...]
    calibration: tuple[ConditionalPairSampleIdentity, ...]
    component_count: int
    calibration_component_count: int
    seed: str

    @property
    def fit_sample_keys(self) -> frozenset[str]:
        return frozenset(identity.sample_key for identity in self.fit)

    @property
    def calibration_sample_keys(self) -> frozenset[str]:
        return frozenset(identity.sample_key for identity in self.calibration)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, item: tuple[str, str]) -> tuple[str, str]:
        if item not in self.parent:
            self.parent[item] = item
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: tuple[str, str], second: tuple[str, str]) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        self.parent[upper] = lower


def partition_fit_calibration_identities(
    identities: Sequence[ConditionalPairSampleIdentity],
    *,
    calibration_fraction: float = 0.10,
    seed: str = "chemvector-v0.9.10",
) -> ConditionalPairIdentityPartition:
    """Deterministically split connected molecule/source components.

    Every render sharing either molecule identity or source-structure identity
    remains on one side.  The lowest stable SHA256-ranked components form the
    calibration slice; validation identities must never be passed here.
    """

    if isinstance(calibration_fraction, bool) or not isinstance(
        calibration_fraction, (int, float)
    ):
        raise ValueError("calibration_fraction must be numeric")
    calibration_fraction = float(calibration_fraction)
    if not math.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    if not isinstance(seed, str) or not seed:
        raise ValueError("Conditional pair partition seed must be nonempty")
    if not identities:
        raise ValueError("Conditional pair fit/calibration partition cannot be empty")
    if any(identity.source_split != "train" for identity in identities):
        raise ValueError("Only train identities may be partitioned into fit/calibration")
    sample_keys = [identity.sample_key for identity in identities]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("Conditional pair sample keys must be unique")

    union = _UnionFind()
    for identity in identities:
        union.union(
            ("molecule", identity.molecule_id),
            ("source", identity.source_structure_id),
        )
    components: dict[tuple[str, str], list[ConditionalPairSampleIdentity]] = defaultdict(list)
    for identity in identities:
        root = union.find(("molecule", identity.molecule_id))
        components[root].append(identity)
    if len(components) < 2:
        raise ValueError("Fit/calibration partition requires at least two identity components")

    ranked: list[tuple[str, tuple[ConditionalPairSampleIdentity, ...]]] = []
    for members in components.values():
        molecule_ids = sorted({member.molecule_id for member in members})
        source_ids = sorted({member.source_structure_id for member in members})
        canonical = json.dumps(
            {"molecule_ids": molecule_ids, "source_structure_ids": source_ids},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        rank = hashlib.sha256(f"{seed}\n{canonical}".encode("utf-8")).hexdigest()
        ranked.append((rank, tuple(sorted(members, key=lambda item: item.sample_key))))
    ranked.sort(key=lambda item: item[0])
    calibration_count = int(round(len(ranked) * calibration_fraction))
    calibration_count = min(max(calibration_count, 1), len(ranked) - 1)
    calibration_members = tuple(
        member for _, members in ranked[:calibration_count] for member in members
    )
    fit_members = tuple(
        member for _, members in ranked[calibration_count:] for member in members
    )
    validate_molecule_disjoint_splits(
        fit_members,
        calibration_members,
        left_name="fit",
        right_name="calibration",
    )
    return ConditionalPairIdentityPartition(
        fit=tuple(sorted(fit_members, key=lambda item: item.sample_key)),
        calibration=tuple(sorted(calibration_members, key=lambda item: item.sample_key)),
        component_count=len(ranked),
        calibration_component_count=calibration_count,
        seed=seed,
    )


def validate_molecule_disjoint_splits(
    left: Sequence[ConditionalPairSampleIdentity],
    right: Sequence[ConditionalPairSampleIdentity],
    *,
    left_name: str = "train",
    right_name: str = "validation",
) -> dict[str, Any]:
    """Require zero overlap for both molecule and source-structure identity."""

    left_molecules = {identity.molecule_id for identity in left}
    right_molecules = {identity.molecule_id for identity in right}
    left_sources = {identity.source_structure_id for identity in left}
    right_sources = {identity.source_structure_id for identity in right}
    molecule_overlap = sorted(left_molecules.intersection(right_molecules))
    source_overlap = sorted(left_sources.intersection(right_sources))
    if molecule_overlap or source_overlap:
        raise ValueError(
            f"Conditional pair {left_name}/{right_name} identity leakage: "
            f"molecules={molecule_overlap[:5]}, sources={source_overlap[:5]}"
        )
    return {
        "left_name": left_name,
        "right_name": right_name,
        "left_samples": len(left),
        "right_samples": len(right),
        "left_molecules": len(left_molecules),
        "right_molecules": len(right_molecules),
        "left_source_structures": len(left_sources),
        "right_source_structures": len(right_sources),
        "molecule_overlap": 0,
        "source_structure_overlap": 0,
        "left_molecule_sha256": _sha256_lines(left_molecules),
        "right_molecule_sha256": _sha256_lines(right_molecules),
        "left_source_structure_sha256": _sha256_lines(left_sources),
        "right_source_structure_sha256": _sha256_lines(right_sources),
    }


def _geometry_features(geometry: Any) -> np.ndarray:
    values = np.asarray(
        [
            geometry.angle_degrees,
            geometry.normal_separation_pixels,
            geometry.overlap_pixels,
            geometry.shorter_interval_overlap,
            geometry.interval_iou,
            geometry.interior_ink_occupancy,
            geometry.interior_mean_darkness,
            geometry.center_mean_darkness,
        ],
        dtype=np.float64,
    )
    if values.shape != (PAIR_FEATURE_DIM,) or not np.isfinite(values).all():
        raise ValueError("Conditional pair geometry features must be finite")
    return values


def generate_target_free_compatible_pairs(
    anchor_segments_pixels: Any,
    raster: Image.Image | np.ndarray,
    *,
    pairing_config: StrokeBoundaryPairingConfig = StrokeBoundaryPairingConfig(),
) -> dict[str, np.ndarray]:
    """Generate all compatible pair candidates without accepting any GT input."""

    records = _stroke_pairing._records(anchor_segments_pixels)
    darkness = _stroke_pairing._darkness_image(raster)
    height, width = darkness.shape
    pair_indices: list[tuple[int, int]] = []
    pair_features: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            geometry = _stroke_pairing._measure_pair(
                records[first_index],
                records[second_index],
                darkness=darkness,
                config=pairing_config,
            )
            if not geometry.compatible:
                continue
            pair_indices.append((first_index, second_index))
            pair_features.append(_geometry_features(geometry))
            centerlines.append(
                _stroke_pairing._centerline(
                    records[first_index],
                    records[second_index],
                    width=width,
                    height=height,
                )
            )
    if not pair_indices:
        return {
            "pair_indices": np.empty((0, 2), dtype=np.int64),
            "pair_features": np.empty((0, PAIR_FEATURE_DIM), dtype=np.float64),
            "pair_overlap_centerline_pixels": np.empty(
                (0, 2, 2), dtype=np.float64
            ),
        }
    return {
        "pair_indices": np.asarray(pair_indices, dtype=np.int64),
        "pair_features": np.asarray(pair_features, dtype=np.float64),
        "pair_overlap_centerline_pixels": np.asarray(centerlines, dtype=np.float64),
    }


def label_conditional_pairs(
    pair_indices: Any,
    anchor_target_line_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Label frozen candidates; all distinct ordinary targets are hard negatives."""

    pairs = np.asarray(pair_indices, dtype=np.int64)
    target_ids = np.asarray(anchor_target_line_ids, dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Conditional pair indices must have shape [pairs,2]")
    if target_ids.ndim != 1 or (target_ids < 0).any():
        raise ValueError("Conditional pair anchor target IDs must be nonnegative")
    if len(pairs) and ((pairs < 0).any() or (pairs >= len(target_ids)).any()):
        raise ValueError("Conditional pair indices exceed the anchor array")
    if len(pairs) and (pairs[:, 0] >= pairs[:, 1]).any():
        raise ValueError("Conditional pair indices must be strictly ascending")
    pair_target_ids = target_ids[pairs] if len(pairs) else np.empty((0, 2), np.int64)
    labels = (
        pair_target_ids[:, 0] == pair_target_ids[:, 1]
    ).astype(np.int64, copy=False)
    return labels, pair_target_ids


def build_conditional_ordinary_pair_target(
    image: Image.Image,
    primitives_payload: dict[str, Any],
    layers: dict[str, Any],
    *,
    target_config: EvidenceCarrierExtentTargetConfig = EvidenceCarrierExtentTargetConfig(),
    pairing_config: StrokeBoundaryPairingConfig = StrokeBoundaryPairingConfig(),
    carrier_segments_pixels: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Build one unpadded oracle-routed ordinary pair sample."""

    try:
        width, height = int(layers["width"]), int(layers["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Conditional pair layers require integer width and height") from error
    if image.size != (width, height):
        raise ValueError(
            f"Conditional pair raster size {image.size} differs from layers "
            f"canvas {(width, height)}"
        )
    raw_segments = (
        detect_evidence_carriers(image, target_config=target_config)
        if carrier_segments_pixels is None
        else np.asarray(carrier_segments_pixels, dtype=np.float64)
    )
    association = associate_evidence_carriers(
        raw_segments,
        primitives_payload,
        layers,
        target_config=target_config,
    )
    raw_labels = association["keep_labels"].detach().cpu().numpy()
    positive_source_indices = np.flatnonzero(raw_labels == POSITIVE_LABEL).astype(
        np.int64, copy=False
    )
    raw_segments = np.asarray(raw_segments, dtype=np.float64)
    anchors = raw_segments[positive_source_indices]
    anchor_target_ids = (
        association["target_line_id"][positive_source_indices]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    candidates = generate_target_free_compatible_pairs(
        anchors,
        image,
        pairing_config=pairing_config,
    )
    pair_labels, pair_target_ids = label_conditional_pairs(
        candidates["pair_indices"],
        anchor_target_ids,
    )
    scale = np.asarray((width - 1, height - 1), dtype=np.float64)
    anchor_normalized = anchors / scale if len(anchors) else anchors.copy()
    centerline_pixels = candidates["pair_overlap_centerline_pixels"]
    centerline_normalized = (
        centerline_pixels / scale if len(centerline_pixels) else centerline_pixels.copy()
    )
    return {
        "anchor_segments": torch.tensor(anchor_normalized, dtype=torch.float32),
        "anchor_segments_pixels": torch.tensor(anchors, dtype=torch.float32),
        "anchor_source_indices": torch.tensor(
            positive_source_indices, dtype=torch.long
        ),
        "anchor_target_line_ids": torch.tensor(anchor_target_ids, dtype=torch.long),
        "anchor_mask": torch.ones(len(anchors), dtype=torch.bool),
        "pair_indices": torch.tensor(candidates["pair_indices"], dtype=torch.long),
        "pair_features": torch.tensor(candidates["pair_features"], dtype=torch.float32),
        "pair_labels": torch.tensor(pair_labels, dtype=torch.long),
        "pair_target_line_ids": torch.tensor(pair_target_ids, dtype=torch.long),
        "pair_overlap_centerline": torch.tensor(
            centerline_normalized, dtype=torch.float32
        ),
        "pair_overlap_centerline_pixels": torch.tensor(
            centerline_pixels, dtype=torch.float32
        ),
        "pair_mask": torch.ones(len(pair_labels), dtype=torch.bool),
        "raw_carrier_segments": association["carrier_segments"].clone(),
        "raw_keep_labels": association["keep_labels"].clone(),
        "raw_target_line_ids": association["target_line_id"].clone(),
        "raw_group_ids": association["group_id"].clone(),
        "raw_group_types": association["group_type"].clone(),
        "raw_carrier_mask": association["carrier_mask"].clone(),
    }


def iter_conditional_pair_identities(
    shards: str | list[str],
    *,
    split: SplitName,
    render_variant: str | None,
    limit: int | None = None,
) -> Iterator[ConditionalPairSampleIdentity]:
    """Scan identities in one style, or all styles when render_variant is None."""

    dataset = ConditionalOrdinaryPairWebDataset(
        shards,
        split=split,
        render_variant=render_variant,
        limit=limit,
        validate_hashes=False,
    )
    for _, _, identity in dataset._iter_selected_raw():
        yield identity


class ConditionalOrdinaryPairWebDataset(IterableDataset):
    """Visual SVG stream restricted to one split and an explicit render mode.

    A string selects a single-style capacity dataset.  ``None`` keeps every
    render variant for generalization training; fit/calibration partitioning
    must then use :func:`partition_fit_calibration_identities`, which groups all
    records sharing molecule or source identity on the same side.
    """

    def __init__(
        self,
        shards: str | list[str],
        *,
        split: SplitName,
        render_variant: str | None,
        allowed_sample_keys: Iterable[str] | None = None,
        shardshuffle: bool = False,
        sample_shuffle_buffer: int = 0,
        validate_hashes: bool = True,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        reject_internal_test_shards(shards)
        self.shards = shards
        self.split = _strict_split(split)
        self.render_variant = (
            None if render_variant is None else _strict_variant(render_variant)
        )
        self.allowed_sample_keys = (
            None if allowed_sample_keys is None else frozenset(map(str, allowed_sample_keys))
        )
        if self.allowed_sample_keys is not None and not self.allowed_sample_keys:
            raise ValueError("Conditional pair allowed_sample_keys cannot be empty")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("Conditional pair limit must be a positive integer")
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer
        self.validate_hashes = validate_hashes
        self.limit = limit

    def _iter_selected_raw(self):
        source = wds.WebDataset(
            self.shards,
            shardshuffle=100 if self.shardshuffle else False,
            empty_check=False,
        )
        if self.sample_shuffle_buffer:
            source = source.shuffle(self.sample_shuffle_buffer)
        selected = 0
        seen_samples: set[str] = set()
        seen_source_renders: set[tuple[str, str]] = set()
        for raw in source:
            meta = json.loads(raw["meta.json"])
            sample = {"key": raw["__key__"], "meta": meta}
            identity = conditional_pair_sample_identity(
                sample,
                expected_split=self.split,
            )
            if (
                self.render_variant is not None
                and identity.render_variant != self.render_variant
            ):
                continue
            if (
                self.allowed_sample_keys is not None
                and identity.sample_key not in self.allowed_sample_keys
            ):
                continue
            if identity.sample_key in seen_samples:
                raise ValueError("Conditional pair stream contains a duplicate sample key")
            source_render = (
                identity.source_structure_id,
                identity.render_variant,
            )
            if source_render in seen_source_renders:
                raise ValueError(
                    "Conditional pair stream contains a duplicate source render"
                )
            seen_samples.add(identity.sample_key)
            seen_source_renders.add(source_render)
            selected += 1
            yield raw, meta, identity
            if self.limit is not None and selected >= self.limit:
                break
        if self.allowed_sample_keys is not None and self.limit is None:
            missing = self.allowed_sample_keys - seen_samples
            if missing:
                raise ValueError(
                    "Conditional pair requested sample keys were not found: "
                    + ", ".join(sorted(missing)[:5])
                )

    def __iter__(self):
        for raw, meta, identity in self._iter_selected_raw():
            payloads = {
                "png": raw["png"],
                "bond.png": raw["bond.png"],
                "primitives.json": raw["primitives.json"],
                "layers.json": raw["layers.json"],
            }
            if self.validate_hashes:
                expected = meta.get("sha256", {})
                for name, payload in payloads.items():
                    if (
                        name in expected
                        and hashlib.sha256(payload).hexdigest() != expected[name]
                    ):
                        raise ValueError(
                            f"Hash mismatch for {identity.sample_key}.{name}"
                        )
            image = Image.open(io.BytesIO(payloads["png"])).convert("RGB")
            image.load()
            bond_layer = Image.open(io.BytesIO(payloads["bond.png"])).convert("L")
            bond_layer.load()
            yield {
                "key": identity.sample_key,
                "meta": meta,
                "identity": identity,
                "image": image,
                "bond_layer": bond_layer,
                "primitives": json.loads(payloads["primitives.json"]),
                "layers": json.loads(payloads["layers.json"]),
            }


class ConditionalOrdinaryPairCollator:
    """Build and dynamically pad conditional pair candidates."""

    def __init__(
        self,
        *,
        image_size: int = 768,
        target_config: EvidenceCarrierExtentTargetConfig = EvidenceCarrierExtentTargetConfig(),
        pairing_config: StrokeBoundaryPairingConfig = StrokeBoundaryPairingConfig(),
    ) -> None:
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 2:
            raise ValueError("Conditional pair image_size must be an integer >= 2")
        self.image_size = image_size
        self.target_config = target_config
        self.pairing_config = pairing_config
        self.image_collator = VisualSVGCollator(tokenizer=None, image_size=image_size)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Conditional pair cannot collate an empty batch")
        targets = []
        identities = []
        expected_size = (self.image_size, self.image_size)
        for sample in samples:
            reject_internal_test_sample(sample)
            if sample["image"].size != expected_size:
                raise ValueError(
                    "Conditional pair requires native-size rasters; "
                    f"got {sample['image'].size}, expected {expected_size}"
                )
            identity = sample.get("identity")
            if not isinstance(identity, ConditionalPairSampleIdentity):
                identity = conditional_pair_sample_identity(sample)
            identities.append(identity)
            targets.append(
                build_conditional_ordinary_pair_target(
                    sample["image"],
                    sample["primitives"],
                    sample["layers"],
                    target_config=self.target_config,
                    pairing_config=self.pairing_config,
                )
            )

        anchor_counts = [len(target["anchor_segments"]) for target in targets]
        pair_counts = [len(target["pair_indices"]) for target in targets]
        raw_counts = [len(target["raw_carrier_segments"]) for target in targets]
        anchor_capacity = max(max(anchor_counts), 1)
        pair_capacity = max(max(pair_counts), 1)
        raw_capacity = max(max(raw_counts), 1)
        batch = len(samples)
        result: dict[str, Any] = {
            "pixel_values": torch.stack(
                [self.image_collator._image_tensor(sample["image"]) for sample in samples]
            ),
            "anchor_segments": torch.zeros(
                (batch, anchor_capacity, 2, 2), dtype=torch.float32
            ),
            "anchor_segments_pixels": torch.zeros(
                (batch, anchor_capacity, 2, 2), dtype=torch.float32
            ),
            "anchor_source_indices": torch.full(
                (batch, anchor_capacity), PAIR_PADDING_INDEX, dtype=torch.long
            ),
            "anchor_target_line_ids": torch.full(
                (batch, anchor_capacity), PAIR_PADDING_INDEX, dtype=torch.long
            ),
            "anchor_mask": torch.zeros((batch, anchor_capacity), dtype=torch.bool),
            "pair_indices": torch.full(
                (batch, pair_capacity, 2), PAIR_PADDING_INDEX, dtype=torch.long
            ),
            "pair_features": torch.zeros(
                (batch, pair_capacity, PAIR_FEATURE_DIM), dtype=torch.float32
            ),
            "pair_labels": torch.full(
                (batch, pair_capacity), PAIR_PADDING_LABEL, dtype=torch.long
            ),
            "pair_target_line_ids": torch.full(
                (batch, pair_capacity, 2), PAIR_PADDING_INDEX, dtype=torch.long
            ),
            "pair_overlap_centerline": torch.zeros(
                (batch, pair_capacity, 2, 2), dtype=torch.float32
            ),
            "pair_overlap_centerline_pixels": torch.zeros(
                (batch, pair_capacity, 2, 2), dtype=torch.float32
            ),
            "pair_mask": torch.zeros((batch, pair_capacity), dtype=torch.bool),
            "raw_carrier_segments": torch.zeros(
                (batch, raw_capacity, 2, 2), dtype=torch.float32
            ),
            "raw_keep_labels": torch.full(
                (batch, raw_capacity), IGNORE_LABEL, dtype=torch.long
            ),
            "raw_target_line_ids": torch.full(
                (batch, raw_capacity), PAIR_PADDING_INDEX, dtype=torch.long
            ),
            "raw_group_ids": torch.full(
                (batch, raw_capacity), PAIR_PADDING_INDEX, dtype=torch.long
            ),
            "raw_group_types": torch.zeros((batch, raw_capacity), dtype=torch.long),
            "raw_carrier_mask": torch.zeros((batch, raw_capacity), dtype=torch.bool),
        }
        fields = (
            "anchor_segments",
            "anchor_segments_pixels",
            "anchor_source_indices",
            "anchor_target_line_ids",
            "anchor_mask",
            "pair_indices",
            "pair_features",
            "pair_labels",
            "pair_target_line_ids",
            "pair_overlap_centerline",
            "pair_overlap_centerline_pixels",
            "pair_mask",
            "raw_carrier_segments",
            "raw_keep_labels",
            "raw_target_line_ids",
            "raw_group_ids",
            "raw_group_types",
            "raw_carrier_mask",
        )
        for batch_index, target in enumerate(targets):
            counts = {
                "anchor": anchor_counts[batch_index],
                "pair": pair_counts[batch_index],
                "raw": raw_counts[batch_index],
            }
            for name in fields:
                prefix = "anchor" if name.startswith("anchor_") else (
                    "pair" if name.startswith("pair_") else "raw"
                )
                count = counts[prefix]
                result[name][batch_index, :count] = target[name]
        return {
            **result,
            "anchor_count": torch.tensor(anchor_counts, dtype=torch.long),
            "pair_count": torch.tensor(pair_counts, dtype=torch.long),
            "raw_carrier_count": torch.tensor(raw_counts, dtype=torch.long),
            "keys": [identity.sample_key for identity in identities],
            "identity": [identity.as_dict() for identity in identities],
            "meta": [sample["meta"] for sample in samples],
            "schema_version": SCHEMA_VERSION,
            "pair_feature_names": PAIR_FEATURE_NAMES,
        }


def _maximum_disjoint_positive_pairs(
    anchor_count: int,
    positive_pair_indices: np.ndarray,
) -> int:
    if anchor_count < 1 or len(positive_pair_indices) == 0:
        return 0
    rows = np.concatenate((positive_pair_indices[:, 0], positive_pair_indices[:, 1]))
    columns = np.concatenate(
        (np.arange(len(positive_pair_indices)), np.arange(len(positive_pair_indices)))
    )
    matrix = coo_array(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(anchor_count, len(positive_pair_indices)),
    ).tocsr()
    result = milp(
        c=-np.ones(len(positive_pair_indices), dtype=np.float64),
        integrality=np.ones(len(positive_pair_indices), dtype=np.uint8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            matrix,
            np.zeros(anchor_count, dtype=np.float64),
            np.ones(anchor_count, dtype=np.float64),
        ),
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"Conditional pair oracle matching failed: {result.message}")
    return int(np.rint(result.x).sum())


class ConditionalPairSummary:
    """Streaming class-distribution and candidate-ceiling report."""

    def __init__(self, *, feature_round_decimals: int = 6) -> None:
        if isinstance(feature_round_decimals, bool) or not isinstance(
            feature_round_decimals, int
        ) or feature_round_decimals < 0:
            raise ValueError("feature_round_decimals must be a nonnegative integer")
        self.feature_round_decimals = feature_round_decimals
        self.samples = 0
        self.anchors = 0
        self.pairs = 0
        self.positive_pairs = 0
        self.negative_pairs = 0
        self.samples_with_positive = 0
        self.samples_with_negative = 0
        self.samples_without_pairs = 0
        self.targets = 0
        self.multi_anchor_targets = 0
        self.multi_anchor_count = 0
        self.pairable_multi_anchor_targets = 0
        self.same_target_possible_pairs = 0
        self.same_target_compatible_pairs = 0
        self.oracle_disjoint_positive_pairs = 0
        self.oracle_paired_anchors = 0
        self.samples_with_multi_anchor_targets = 0
        self.samples_oracle_all_multi_anchor_targets_pairable = 0
        self.feature_labels: dict[tuple[float, ...], Counter[int]] = defaultdict(Counter)

    def update(self, target: dict[str, torch.Tensor]) -> None:
        anchor_ids = target["anchor_target_line_ids"].detach().cpu().numpy()
        pair_indices = target["pair_indices"].detach().cpu().numpy()
        labels = target["pair_labels"].detach().cpu().numpy()
        features = target["pair_features"].detach().cpu().numpy()
        if pair_indices.shape != (len(labels), 2):
            raise ValueError("Conditional pair summary received malformed pair indices")
        if features.shape != (len(labels), PAIR_FEATURE_DIM):
            raise ValueError("Conditional pair summary received malformed pair features")
        self.samples += 1
        self.anchors += len(anchor_ids)
        self.pairs += len(labels)
        positives = labels == PAIR_POSITIVE_LABEL
        negatives = labels == PAIR_NEGATIVE_LABEL
        self.positive_pairs += int(positives.sum())
        self.negative_pairs += int(negatives.sum())
        self.samples_with_positive += int(bool(positives.any()))
        self.samples_with_negative += int(bool(negatives.any()))
        self.samples_without_pairs += int(len(labels) == 0)

        counts = Counter(map(int, anchor_ids.tolist()))
        positive_targets = {
            int(anchor_ids[first])
            for first, _ in pair_indices[positives].tolist()
        }
        multi = {target_id for target_id, count in counts.items() if count >= 2}
        self.targets += len(counts)
        self.multi_anchor_targets += len(multi)
        self.multi_anchor_count += sum(counts[target_id] for target_id in multi)
        self.pairable_multi_anchor_targets += len(multi.intersection(positive_targets))
        self.same_target_possible_pairs += sum(
            count * (count - 1) // 2 for count in counts.values()
        )
        self.same_target_compatible_pairs += int(positives.sum())
        oracle_pairs = _maximum_disjoint_positive_pairs(
            len(anchor_ids), pair_indices[positives]
        )
        self.oracle_disjoint_positive_pairs += oracle_pairs
        self.oracle_paired_anchors += 2 * oracle_pairs
        self.samples_with_multi_anchor_targets += int(bool(multi))
        self.samples_oracle_all_multi_anchor_targets_pairable += int(
            bool(multi) and multi.issubset(positive_targets)
        )
        for feature, label in zip(features, labels):
            key = tuple(
                np.round(feature.astype(np.float64), self.feature_round_decimals).tolist()
            )
            self.feature_labels[key][int(label)] += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def result(self) -> dict[str, Any]:
        feature_correct = sum(max(counts.values()) for counts in self.feature_labels.values())
        conflicting = sum(1 for counts in self.feature_labels.values() if len(counts) > 1)
        return {
            "schema_version": SCHEMA_VERSION,
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "samples": self.samples,
            "ordinary_anchors": self.anchors,
            "compatible_pair_candidates": self.pairs,
            "class_distribution": {
                "same_target_positive": self.positive_pairs,
                "distinct_target_hard_negative": self.negative_pairs,
                "positive_fraction": self._ratio(self.positive_pairs, self.pairs),
                "hard_negatives_per_sample": self._ratio(
                    self.negative_pairs, self.samples
                ),
                "samples_with_positive": self.samples_with_positive,
                "samples_with_hard_negative": self.samples_with_negative,
                "samples_without_candidates": self.samples_without_pairs,
                "both_classes_present": bool(
                    self.positive_pairs > 0 and self.negative_pairs > 0
                ),
            },
            "candidate_learnability_ceiling": {
                "ordinary_target_count": self.targets,
                "multi_anchor_target_count": self.multi_anchor_targets,
                "pairable_multi_anchor_target_count": (
                    self.pairable_multi_anchor_targets
                ),
                "pairable_multi_anchor_target_recall": self._ratio(
                    self.pairable_multi_anchor_targets,
                    self.multi_anchor_targets,
                ),
                "compatible_same_target_pair_recall": self._ratio(
                    self.same_target_compatible_pairs,
                    self.same_target_possible_pairs,
                ),
                "oracle_max_disjoint_positive_pairs": (
                    self.oracle_disjoint_positive_pairs
                ),
                "oracle_paired_anchor_count": self.oracle_paired_anchors,
                "oracle_paired_anchor_fraction": self._ratio(
                    self.oracle_paired_anchors,
                    self.anchors,
                ),
                "oracle_paired_multi_anchor_fraction": self._ratio(
                    self.oracle_paired_anchors,
                    self.multi_anchor_count,
                ),
                "samples_with_multi_anchor_targets": (
                    self.samples_with_multi_anchor_targets
                ),
                "sample_all_multi_anchor_targets_pairable": (
                    self.samples_oracle_all_multi_anchor_targets_pairable
                ),
                "sample_all_multi_anchor_targets_pairable_fraction": self._ratio(
                    self.samples_oracle_all_multi_anchor_targets_pairable,
                    self.samples_with_multi_anchor_targets,
                ),
            },
            "empirical_feature_collision_ceiling": {
                "round_decimals": self.feature_round_decimals,
                "unique_feature_rows": len(self.feature_labels),
                "conflicting_feature_rows": conflicting,
                "majority_lookup_accuracy": self._ratio(feature_correct, self.pairs),
            },
        }


__all__ = [
    "PAIR_FEATURE_DIM",
    "PAIR_FEATURE_NAMES",
    "PAIR_NEGATIVE_LABEL",
    "PAIR_PADDING_INDEX",
    "PAIR_PADDING_LABEL",
    "PAIRING_PROFILES",
    "PAIRING_PROFILE_STRICT",
    "PAIRING_PROFILE_TRAINING_WIDE",
    "PAIR_POSITIVE_LABEL",
    "SCHEMA_VERSION",
    "ConditionalOrdinaryPairCollator",
    "ConditionalOrdinaryPairWebDataset",
    "ConditionalPairIdentityPartition",
    "ConditionalPairSampleIdentity",
    "ConditionalPairSummary",
    "build_conditional_ordinary_pair_target",
    "conditional_pair_sample_identity",
    "conditional_pair_pairing_config",
    "generate_target_free_compatible_pairs",
    "iter_conditional_pair_identities",
    "label_conditional_pairs",
    "partition_fit_calibration_identities",
    "validate_molecule_disjoint_splits",
]
