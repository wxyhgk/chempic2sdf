"""Bounded, isolated port-conditioned endpoint-objectness trial.

This experiment does not alter the production model.  ``initial.pt`` and its
bond detector are frozen; their per-endpoint features are cached to CPU and a
small MLP is trained on the cache.  A sidecar row is an endpoint of an already
predicted bond, never a new row in the production atom/set head.  At inference
the sidecar can only claim an unoccupied port once, preserving the baseline
prefix by construction.

The launcher is intentionally conservative: it uses a fixed 480/720
group/molecule/SMILES-disjoint split, caps optimization at 800 steps, chooses a
threshold from train rows only, and writes an audit report rather than a
production checkpoint.  Sealed-test paths are not referenced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from tools.current_initial_cohort import split_group_disjoint


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_PORT_SIDECAR_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-port-endpoint-sidecar-20261008"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
REQUESTED_TRAIN_IMAGES = 480
MAX_STEPS = 800
IMAGE_SIZE = 768.0
PORT_RADIUS_PX = 12.0
LABEL_RADIUS_PX = 6.0
MAX_ACCEPTED_PER_IMAGE = 16
SEED = 20261008
FEATURE_MODE = os.environ.get("CHEMVECTOR_PORT_FEATURE_MODE", "full")
if FEATURE_MODE not in {"full", "explicit"}:
    raise ValueError("CHEMVECTOR_PORT_FEATURE_MODE must be full or explicit")
THRESHOLD_GRID = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
PRECISION_FLOOR = 0.85
MIN_TRAIN_POSITIVES = 50
MIN_TRAIN_AUPRC = 0.50


@dataclass(frozen=True)
class CandidateCache:
    """CPU-only candidate cache; each row corresponds to one bond port."""

    features: torch.Tensor
    labels: torch.Tensor
    text_labels: torch.Tensor
    image_indices: torch.Tensor
    sample_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        n = self.features.size(0)
        if self.features.ndim != 2 or n == 0:
            raise ValueError("candidate features must be a non-empty [N,D] tensor")
        for value, name in (
            (self.labels, "labels"),
            (self.text_labels, "text_labels"),
            (self.image_indices, "image_indices"),
        ):
            if value.ndim != 1 or value.size(0) != n:
                raise ValueError(f"{name} must be [N]")
        if self.labels.dtype != torch.bool or self.text_labels.dtype != torch.bool:
            raise ValueError("candidate labels must be boolean")
        if self.image_indices.dtype != torch.long:
            raise ValueError("image_indices must be int64")

    def to_payload(self) -> dict[str, object]:
        """Return a stable, pickle-free-enough payload for audit caches.

        Saving the dataclass itself from a ``python -m`` invocation records its
        class as ``__main__.CandidateCache``.  Such a cache cannot be loaded by
        an external audit process.  Keep the on-disk contract to tensors and a
        tuple of strings instead.
        """
        return {
            "features": self.features,
            "labels": self.labels,
            "text_labels": self.text_labels,
            "image_indices": self.image_indices,
            "sample_keys": list(self.sample_keys),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CandidateCache":
        required = {"features", "labels", "text_labels", "image_indices", "sample_keys"}
        if set(payload) != required:
            raise ValueError(f"unexpected candidate cache keys: {sorted(payload)}")
        return cls(
            payload["features"],  # type: ignore[arg-type]
            payload["labels"],  # type: ignore[arg-type]
            payload["text_labels"],  # type: ignore[arg-type]
            payload["image_indices"],  # type: ignore[arg-type]
            tuple(payload["sample_keys"]),  # type: ignore[arg-type]
        )


class PortEndpointObjectness(nn.Module):
    """Small independent scorer; parameter count remains below 50k."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("sidecar dimensions must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        if sum(parameter.numel() for parameter in self.parameters()) >= 50_000:
            raise ValueError("port sidecar exceeds 50k parameter budget")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _target_ids_for_ports(
    endpoint_points: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_points: torch.Tensor,
    baseline_ids: set[int],
) -> torch.Tensor:
    """Assign residual endpoint labels without leaking target identity.

    The returned target IDs are used only to create the boolean objectness
    label.  Multiple bond ports may legitimately share one atom (a junction),
    so labels are nearest-target per port rather than one-to-one.  Only targets
    not already owned by the frozen baseline are retained.
    """
    labels = torch.full(
        (endpoint_points.size(0),), -1, dtype=torch.long, device=endpoint_points.device
    )
    candidate_ids = torch.where(candidate_mask)[0]
    remaining = [index for index in range(target_points.size(0)) if index not in baseline_ids]
    if not len(candidate_ids) or not remaining:
        return labels
    distances = torch.cdist(
        endpoint_points[candidate_ids] * IMAGE_SIZE,
        target_points[remaining] * IMAGE_SIZE,
    )
    nearest_distance, nearest_target = distances.min(dim=1)
    for candidate_offset in torch.where(nearest_distance <= LABEL_RADIUS_PX)[0].tolist():
        labels[int(candidate_ids[candidate_offset])] = remaining[int(nearest_target[candidate_offset])]
    return labels


def build_port_features(
    bonds: dict[str, torch.Tensor],
    baseline_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create target-independent features and candidate mask for bond ports."""
    mask = bonds["bond_mask"][0].bool()
    endpoints = bonds["bond_endpoints"][0]
    if endpoints.ndim != 3 or endpoints.size(1) != 2:
        raise ValueError("bond endpoints must be [Q,2,2]")
    endpoint_features = bonds["bond_features"][0]
    endpoint_types = bonds["bond_types"][0]
    endpoint_scores = bonds["bond_scores"][0]
    if endpoint_features.size(0) != endpoints.size(0):
        raise ValueError("bond feature/query count mismatch")
    if endpoint_types.size(0) != endpoints.size(0):
        raise ValueError("bond type/query count mismatch")
    if endpoint_scores.size(0) != endpoints.size(0):
        raise ValueError("bond score/query count mismatch")
    if len(baseline_points):
        nearest = torch.cdist(
            endpoints.reshape(-1, 2) * IMAGE_SIZE,
            baseline_points * IMAGE_SIZE,
        ).amin(dim=1).reshape(-1, 2)
    else:
        nearest = torch.full(
            (endpoints.size(0), 2), float("inf"),
            device=endpoints.device, dtype=endpoints.dtype,
        )
    free = mask[:, None] & (nearest > PORT_RADIUS_PX)
    side = torch.eye(2, device=endpoints.device, dtype=endpoints.dtype)[None].expand(endpoints.size(0), -1, -1)
    repeated_features = endpoint_features[:, None, :].expand(-1, 2, -1)
    repeated_types = endpoint_types[:, None, :].expand(-1, 2, -1)
    repeated_scores = endpoint_scores[:, None, None].expand(-1, 2, -1)
    repeated_distances = nearest[..., None].clamp_max(IMAGE_SIZE) / IMAGE_SIZE
    components = [repeated_types, repeated_scores, endpoints, side, repeated_distances]
    if FEATURE_MODE == "full":
        components.insert(0, repeated_features)
    combined = torch.cat(components, dim=-1)
    features = combined.reshape(-1, combined.size(-1))
    return features, free.reshape(-1)


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.bool()
    if not labels.numel() or not bool(labels.any()):
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order].float()
    precision = sorted_labels.cumsum(0) / torch.arange(
        1, len(sorted_labels) + 1, device=scores.device, dtype=torch.float32
    )
    return float((precision * sorted_labels).sum() / sorted_labels.sum())


def _apply_capacity(
    scores: torch.Tensor,
    selected: torch.Tensor,
    image_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the graph-merge per-image claim capacity to a selection mask."""
    if image_indices is None:
        return selected
    selected = selected.clone()
    for image in image_indices.unique(sorted=True).tolist():
        rows = torch.where(selected & (image_indices == image))[0]
        if len(rows) > MAX_ACCEPTED_PER_IMAGE:
            keep = rows[torch.argsort(scores[rows], descending=True)[:MAX_ACCEPTED_PER_IMAGE]]
            selected[rows] = False
            selected[keep] = True
    return selected


def _select_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_indices: torch.Tensor | None = None,
) -> dict:
    """Choose a train-only threshold, preferring precision then recall."""
    rows = []
    for threshold in THRESHOLD_GRID:
        selected = _apply_capacity(scores, scores >= threshold, image_indices)
        matched = int((selected & labels).sum())
        count = int(selected.sum())
        rows.append({
            "threshold": threshold,
            "selected": count,
            "matched": matched,
            "precision": matched / max(count, 1),
            "recall": matched / max(int(labels.sum()), 1),
        })
    admissible = [row for row in rows if row["precision"] >= PRECISION_FLOOR]
    if not admissible:
        return {"selected": None, "grid": rows, "admissible": False}
    chosen = max(admissible, key=lambda row: (row["recall"], -row["threshold"]))
    return {"selected": chosen, "grid": rows, "admissible": True}


def _evaluate(scores: torch.Tensor, labels: torch.Tensor, image_indices: torch.Tensor, threshold: float | None) -> dict:
    result = {
        "candidates": int(labels.numel()),
        "positives": int(labels.sum()),
        "auprc": _average_precision(scores, labels),
        "threshold": threshold,
    }
    if threshold is None:
        result.update({"selected": 0, "matched": 0, "precision": 0.0, "recall": 0.0})
        return result
    chosen = _apply_capacity(scores, scores >= threshold, image_indices)
    matched = int((chosen & labels).sum())
    count = int(chosen.sum())
    result.update({
        "selected": count,
        "matched": matched,
        "precision": matched / max(count, 1),
        "recall": matched / max(int(labels.sum()), 1),
    })
    return result


def _cache_digest(cache: CandidateCache) -> str:
    digest = hashlib.sha256()
    for value in (cache.features, cache.labels, cache.text_labels, cache.image_indices):
        digest.update(value.numpy().tobytes())
    digest.update("\n".join(cache.sample_keys).encode())
    return digest.hexdigest()


def _load_rows() -> tuple[list[dict], list[dict], dict]:
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    manifest = source_checks()["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in keys]
    if len(rows) != 1_200:
        raise ValueError(f"Expected 1,200 released cohort rows, found {len(rows)}")
    train, holdout, split = split_group_disjoint(rows, REQUESTED_TRAIN_IMAGES)
    if split["train_images"] != REQUESTED_TRAIN_IMAGES or split["holdout_images"] != 1_200 - REQUESTED_TRAIN_IMAGES:
        raise ValueError(f"Group-disjoint split did not produce 480/720 rows: {split}")
    if any(split[key] != 0 for key in ("group_overlap", "molecule_overlap", "canonical_smiles_overlap")):
        raise AssertionError(f"Identity overlap in split: {split}")
    return train, holdout, split


def _build_cache(
    model,
    config,
    payload: dict,
    rows: list[dict],
    loader: RecordLoader,
    collator: CurriculumCollator,
    device: torch.device,
) -> CandidateCache:
    feature_rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    text_labels: list[torch.Tensor] = []
    image_indices: list[torch.Tensor] = []
    sample_keys: list[str] = []
    for image_index, row in enumerate(rows):
        sample = loader(loader.manifest, row)
        batch = collator([sample])
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.no_grad():
            _, _, _, _, bonds, _, selected = structure_forward(
                model,
                tensors["pixel_values"],
                config,
                payload["provenance"]["atom_threshold"],
            )
            mapping, _ = match_atom_targets(selected, tensors, config)
            baseline_ids = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
            baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0], :2]
            features, candidate_mask = build_port_features(bonds, baseline_points)
            endpoint_points = bonds["bond_endpoints"][0].reshape(-1, 2)
            target_points = tensors["atom_anchors"][0, : int(tensors["atom_mask"].sum()), :2]
            target_ids = _target_ids_for_ports(endpoint_points, candidate_mask, target_points, baseline_ids)
            # Endpoint labels are all that the sidecar sees.  Text labels are
            # retained only for subgroup diagnostics and are never fed back.
            text_target_mask = tensors["atom_text_mask"][0].bool()
            row_labels = target_ids >= 0
            row_text = row_labels.clone()
            valid_target_ids = target_ids[row_labels]
            if len(valid_target_ids):
                row_text[row_labels] = text_target_mask[valid_target_ids]
            keep = candidate_mask
            feature_rows.append(features[keep].float().cpu())
            labels.append(row_labels[keep].cpu())
            text_labels.append(row_text[keep].cpu())
            image_indices.append(torch.full((int(keep.sum()),), image_index, dtype=torch.long))
            sample_keys.append(row["sample_key"])
    if not feature_rows:
        raise ValueError("No unoccupied bond-port candidates were produced")
    return CandidateCache(
        torch.cat(feature_rows),
        torch.cat(labels).bool(),
        torch.cat(text_labels).bool(),
        torch.cat(image_indices).long(),
        tuple(sample_keys),
    )


def _train_sidecar(train: CandidateCache, holdout: CandidateCache, device: torch.device) -> tuple[PortEndpointObjectness, dict]:
    torch.manual_seed(SEED)
    model = PortEndpointObjectness(train.features.size(1)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positives = max(int(train.labels.sum()), 1)
    negatives = max(int((~train.labels).sum()), 1)
    pos_weight = torch.tensor(min(negatives / positives, 20.0), device=device)
    features = train.features.to(device)
    labels = train.labels.float().to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    batch_size = min(4096, len(features))
    events = []
    for step in range(1, MAX_STEPS + 1):
        rows = torch.randint(len(features), (batch_size,), generator=generator, device=device)
        logits = model(features[rows])
        loss = F.binary_cross_entropy_with_logits(logits, labels[rows], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == MAX_STEPS:
            events.append({"step": step, "loss": float(loss.detach())})
    with torch.no_grad():
        train_scores = model(train.features.to(device)).sigmoid().cpu()
        holdout_scores = model(holdout.features.to(device)).sigmoid().cpu()
    threshold = _select_threshold(train_scores, train.labels, train.image_indices)
    chosen = threshold["selected"]
    threshold_value = None if chosen is None else float(chosen["threshold"])
    report = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": MAX_STEPS,
        "events": events,
        "train": _evaluate(train_scores, train.labels, train.image_indices, threshold_value),
        "holdout": _evaluate(holdout_scores, holdout.labels, holdout.image_indices, threshold_value),
        "train_threshold": threshold,
        # Text is a subgroup label, not a filter: all non-text candidates stay
        # negatives so this precision is not the tautological 1.0 obtained by
        # evaluating only positive text rows.
        "holdout_text": _evaluate(holdout_scores, holdout.text_labels, holdout.image_indices, threshold_value),
        "holdout_scores": holdout_scores,
    }
    return model, report


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    train_rows, holdout_rows, split = _load_rows()
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to("cuda:0").eval().requires_grad_(False)
    loader = RecordLoader(source_checks()["manifest"])
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    train_cache = _build_cache(model, config, payload, train_rows, loader, collator, torch.device("cuda:0"))
    holdout_cache = _build_cache(model, config, payload, holdout_rows, loader, collator, torch.device("cuda:0"))
    OUTPUT.mkdir(parents=True)
    cache_payload = {
        "schema_version": "chemvector-port-endpoint-cache-0.1",
        "initial_sha256": INITIAL_SHA,
        "train": train_cache.to_payload(),
        "holdout": holdout_cache.to_payload(),
    }
    torch.save(cache_payload, OUTPUT / "candidate_cache.pt")
    sidecar, metrics = _train_sidecar(train_cache, holdout_cache, torch.device("cuda:0"))
    result = {
        "schema_version": "chemvector-port-endpoint-sidecar-trial-0.1",
        "complete": True,
        "training_performed": True,
        "in_memory_sidecar_only": True,
        "production_checkpoint_written": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)),
        "split": split,
        "contract": {
            "frozen_initial_and_bond_detector": True,
            "candidate_source": "unoccupied_frozen_predicted_bond_ports",
            "old_set_head_called_with_residual": False,
            "one_to_one_port_claim": True,
            "port_radius_px": PORT_RADIUS_PX,
            "label_radius_px": LABEL_RADIUS_PX,
            "max_accepted_per_image": MAX_ACCEPTED_PER_IMAGE,
            "max_steps": MAX_STEPS,
            "feature_cache": "candidate_cache.pt",
            "feature_mode": FEATURE_MODE,
        },
        "cache": {
            "train_candidates": int(train_cache.labels.numel()),
            "train_positives": int(train_cache.labels.sum()),
            "train_text_positives": int(train_cache.text_labels.sum()),
            "holdout_candidates": int(holdout_cache.labels.numel()),
            "holdout_positives": int(holdout_cache.labels.sum()),
            "holdout_text_positives": int(holdout_cache.text_labels.sum()),
            "train_digest": _cache_digest(train_cache),
            "holdout_digest": _cache_digest(holdout_cache),
        },
        "metrics": {key: value for key, value in metrics.items() if key != "holdout_scores"},
        "gate": {
            "prefix_immutable": True,
            "old_selected_retention": 1.0,
            "group_disjoint": split["group_overlap"] == 0,
            "molecule_disjoint": split["molecule_overlap"] == 0,
            "canonical_smiles_disjoint": split["canonical_smiles_overlap"] == 0,
            "train_positive_count": int(train_cache.labels.sum()),
            "train_positive_count_sufficient": int(train_cache.labels.sum()) >= MIN_TRAIN_POSITIVES,
            "train_auprc_sufficient": (metrics["train"]["auprc"] or 0.0) >= MIN_TRAIN_AUPRC,
            "holdout_precision_sufficient": metrics["holdout"]["precision"] >= PRECISION_FLOOR,
            "admissible": bool(
                metrics["train_threshold"]["selected"] is not None
                and int(train_cache.labels.sum()) >= MIN_TRAIN_POSITIVES
                and (metrics["train"]["auprc"] or 0.0) >= MIN_TRAIN_AUPRC
                and metrics["holdout"]["precision"] >= PRECISION_FLOOR
                and metrics["holdout"]["matched"] >= 14
            ),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
