#!/usr/bin/env python3
"""Probe whether frozen Bond endpoints improve P4-E9 recovery-action prediction."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.components.graph.joint.predicted_incidence import (
    decode_predicted_bond_proposals,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p4e9_recovery_oracle as p4e9,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e9-bond-evidence-probe-fit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
ORACLE_ROOT = p4e9.OUTPUT
ORACLE_AUDIT = ORACLE_ROOT / "audit.json"
ORACLE_ACTIONS = ORACLE_ROOT / "actions.jsonl"
ORACLE_IMAGES = ORACLE_ROOT / "images.jsonl"
ORACLE_FIT = ORACLE_ROOT / "fit.jsonl"
ORACLE_SHA256 = {
    "audit": "5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698",
    "actions": "30aff390726d2e50388e45b37bd3710be19fab8b1d08a1e5fa81bfd3a4ce5819",
    "images": "b1f6db4da1735e20113a76439c761d0cad3f024e0935aef9747832afde138a9a",
    "fit": "69d7d5c7555041039e83d975f10680b8b1c94e8cc1df65a8d7bfa446b7a4c32f",
}
ROUTES = ("ADD_SPLIT", "ADD_REFINE")
FORMAL_ROUTE = "ADD_REFINE"
BASELINE_DIM = 34
INCIDENT_DIM = len(p3e29.incident_feature_names())
FOLDS = 3
NEGATIVE_RATIO = 3
ENDPOINT_RADIUS_PIXELS = 6.0
MIN_AUC_LIFT = 0.02
MIN_ENDPOINT_COVERAGE = 0.90
LOGISTIC_L2 = 1e-2
LOGISTIC_STEPS = 64
BATCH_SIZE = 2
IMAGE_SIZE = 768
SEED = 20260849


class BondEvidenceProbeError(RuntimeError):
    """Raised when the frozen P4-E9 evidence protocol differs."""


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scaffold_fold(
    scaffold: str,
    molecule_id: str,
    *,
    folds: int = FOLDS,
    seed: int = SEED,
) -> int:
    """Assign all variants of one scaffold to one deterministic OOF fold."""

    if isinstance(folds, bool) or folds < 2:
        raise ValueError("P4-E9 Bond probe requires at least two folds")
    group = str(scaffold).strip() or f"molecule:{molecule_id}"
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def evidence_gate(auc_lift: float, endpoint_coverage: float) -> bool:
    """Apply the preregistered conjunctive evidence gate."""

    values = (float(auc_lift), float(endpoint_coverage))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("P4-E9 Bond evidence gate requires finite metrics")
    return values[0] >= MIN_AUC_LIFT and values[1] >= MIN_ENDPOINT_COVERAGE


def endpoint_target_coverage(
    target_anchors: torch.Tensor,
    bond_endpoints: torch.Tensor,
    bond_mask: torch.Tensor,
    *,
    radius_pixels: float = ENDPOINT_RADIUS_PIXELS,
    image_size: float = IMAGE_SIZE,
) -> dict[str, object]:
    """Count action targets supported by a valid frozen Bond endpoint."""

    if target_anchors.ndim != 2 or target_anchors.size(-1) != 2:
        raise ValueError("P4-E9 Bond coverage targets must have shape [target,2]")
    if bond_endpoints.ndim != 3 or bond_endpoints.shape[1:] != (2, 2):
        raise ValueError("P4-E9 Bond coverage endpoints must have shape [bond,2,2]")
    if bond_mask.shape != (len(bond_endpoints),) or bond_mask.dtype != torch.bool:
        raise ValueError("P4-E9 Bond coverage mask has an invalid shape or dtype")
    if min(float(radius_pixels), float(image_size)) <= 0.0:
        raise ValueError("P4-E9 Bond coverage geometry must be positive")
    if not all(
        value.is_floating_point() and bool(torch.isfinite(value).all())
        for value in (target_anchors, bond_endpoints)
    ):
        raise ValueError("P4-E9 Bond coverage coordinates must be finite floating point")
    if not len(target_anchors):
        return {"targets": 0, "covered": 0, "coverage": 0.0, "covered_mask": []}
    valid = bond_endpoints[bond_mask].reshape(-1, 2).float()
    if not len(valid):
        covered = torch.zeros(len(target_anchors), dtype=torch.bool)
    else:
        distances = torch.cdist(target_anchors.float(), valid) * float(image_size)
        covered = (distances <= float(radius_pixels)).any(dim=1)
    count = int(covered.sum())
    return {
        "targets": len(target_anchors),
        "covered": count,
        "coverage": count / len(target_anchors),
        "covered_mask": covered.tolist(),
    }


def _candidate_statuses(record: Mapping[str, object]) -> list[str]:
    source_ids = record["source_ids"]
    if not isinstance(source_ids, torch.Tensor) or source_ids.ndim != 1:
        raise ValueError("P4-E9 Bond probe source IDs must be a vector")
    if source_ids.dtype != torch.long or len(torch.unique(source_ids)) != len(source_ids):
        raise ValueError("P4-E9 Bond probe source IDs must be unique long values")
    actions = {
        int(source): str(route)
        for source, route in dict(record.get("actions", {})).items()
    }
    if any(route not in ROUTES for route in actions.values()):
        raise ValueError("P4-E9 Bond probe found an unknown Oracle action")
    unsupported = {int(value) for value in record.get("unsupported_source_ids", ())}
    if unsupported & actions.keys():
        raise ValueError("P4-E9 Bond probe action is also marked UNSUPPORTED")
    statuses = []
    for source in source_ids.tolist():
        if source in actions:
            statuses.append(actions[source])
        elif source in unsupported:
            statuses.append("UNSUPPORTED")
        else:
            statuses.append("NONE")
    if set(actions) - set(source_ids.tolist()):
        raise ValueError("P4-E9 Bond probe Oracle action source is absent")
    if unsupported - set(source_ids.tolist()):
        raise ValueError("P4-E9 Bond probe UNSUPPORTED source is absent")
    return statuses


def build_route_dataset(
    records: Sequence[Mapping[str, object]],
    *,
    route: str,
    negative_ratio: int = NEGATIVE_RATIO,
    seed: int = SEED,
) -> dict[str, object]:
    """Keep route positives and a deterministic 1:3 sample of true NONE rows."""

    if route not in ROUTES:
        raise ValueError(f"P4-E9 Bond probe has an unknown route: {route}")
    if isinstance(negative_ratio, bool) or negative_ratio != NEGATIVE_RATIO:
        raise ValueError("P4-E9 Bond probe requires a fixed 1:3 ratio")
    positives: dict[int, list[dict]] = defaultdict(list)
    negatives: dict[int, list[dict]] = defaultdict(list)
    excluded = defaultdict(int)
    for record in records:
        baseline = record["baseline_features"]
        incident = record["incident_features"]
        source_ids = record["source_ids"]
        if (
            not isinstance(baseline, torch.Tensor)
            or baseline.ndim != 2
            or baseline.size(1) != BASELINE_DIM
            or not isinstance(incident, torch.Tensor)
            or incident.shape != (len(baseline), INCIDENT_DIM)
            or not isinstance(source_ids, torch.Tensor)
            or source_ids.shape != (len(baseline),)
        ):
            raise ValueError("P4-E9 Bond probe candidate feature shapes differ")
        if not all(
            value.is_floating_point() and bool(torch.isfinite(value).all())
            for value in (baseline, incident)
        ):
            raise ValueError("P4-E9 Bond probe candidate features must be finite")
        sample_key = str(record["sample_key"])
        fold = scaffold_fold(str(record["scaffold"]), str(record["molecule_id"]))
        statuses = _candidate_statuses(record)
        for index, (source, status) in enumerate(zip(source_ids.tolist(), statuses)):
            entry = {
                "sample_key": sample_key,
                "source_id": int(source),
                "fold": fold,
                "baseline": baseline[index].float(),
                "incident": incident[index].float(),
                "label": status == route,
                "status": status,
            }
            if status == route:
                positives[fold].append(entry)
            elif status == "NONE":
                negatives[fold].append(entry)
            else:
                excluded[status] += 1

    selected = []
    fold_counts = {}
    for fold in range(FOLDS):
        current_positive = positives[fold]
        required = negative_ratio * len(current_positive)
        if not current_positive:
            raise ValueError(f"P4-E9 Bond probe fold {fold} has no {route} positives")
        if len(negatives[fold]) < required:
            raise ValueError(f"P4-E9 Bond probe fold {fold} lacks true NONE negatives")
        current_negative = sorted(
            negatives[fold],
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}:{route}:{item['sample_key']}:{item['source_id']}".encode(
                        "utf-8"
                    )
                ).digest(),
                item["sample_key"],
                item["source_id"],
            ),
        )[:required]
        selected.extend(current_positive)
        selected.extend(current_negative)
        fold_counts[str(fold)] = {
            "positives": len(current_positive),
            "none_negatives": len(current_negative),
        }
    selected.sort(key=lambda item: (item["fold"], item["sample_key"], item["source_id"]))
    labels = torch.tensor([item["label"] for item in selected], dtype=torch.bool)
    folds = torch.tensor([item["fold"] for item in selected], dtype=torch.long)
    baseline = torch.stack([item["baseline"] for item in selected])
    incident = torch.stack([item["incident"] for item in selected])
    return {
        "baseline_features": baseline,
        "augmented_features": torch.cat((baseline, incident), dim=1),
        "labels": labels,
        "folds": folds,
        "identities": [
            (item["sample_key"], item["source_id"], item["status"])
            for item in selected
        ],
        "fold_counts": fold_counts,
        "excluded_counts": dict(sorted(excluded.items())),
        "positives": int(labels.sum()),
        "none_negatives": int((~labels).sum()),
    }


def _logistic_objective(
    design: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, l2: float
) -> torch.Tensor:
    logits = design @ weights
    return F.softplus(logits).sub(labels * logits).sum() + 0.5 * l2 * weights[1:].square().sum()


def fit_logistic_predict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    *,
    l2: float = LOGISTIC_L2,
    steps: int = LOGISTIC_STEPS,
) -> torch.Tensor:
    """Fit one deterministic, standardized, L2 logistic model by damped Newton steps."""

    if (
        train_features.ndim != 2
        or test_features.ndim != 2
        or train_features.size(1) != test_features.size(1)
        or train_labels.shape != (len(train_features),)
        or train_labels.dtype != torch.bool
        or not len(test_features)
    ):
        raise ValueError("P4-E9 Bond probe logistic inputs have invalid shapes")
    if isinstance(steps, bool) or steps < 1 or not math.isfinite(float(l2)) or l2 <= 0.0:
        raise ValueError("P4-E9 Bond probe logistic protocol is invalid")
    if not all(
        value.is_floating_point() and bool(torch.isfinite(value).all())
        for value in (train_features, test_features)
    ):
        raise ValueError("P4-E9 Bond probe logistic features must be finite")
    if min(int(train_labels.sum()), int((~train_labels).sum())) == 0:
        raise ValueError("P4-E9 Bond probe logistic training requires both classes")
    train = train_features.detach().cpu().double()
    test = test_features.detach().cpu().double()
    labels = train_labels.detach().cpu().double()
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False)
    scale = torch.where(scale >= 1e-8, scale, torch.ones_like(scale))
    train = (train - mean) / scale
    test = (test - mean) / scale
    train_design = torch.cat((torch.ones(len(train), 1), train), dim=1)
    test_design = torch.cat((torch.ones(len(test), 1), test), dim=1)
    weights = torch.zeros(train_design.size(1), dtype=torch.float64)
    penalty = torch.zeros_like(weights)
    penalty[1:] = float(l2)
    identity = torch.eye(len(weights), dtype=torch.float64) * 1e-8
    for _ in range(steps):
        logits = (train_design @ weights).clamp(-30.0, 30.0)
        probabilities = logits.sigmoid()
        variance = (probabilities * (1.0 - probabilities)).clamp_min(1e-8)
        gradient = train_design.T @ (probabilities - labels) + penalty * weights
        hessian = train_design.T @ (variance[:, None] * train_design)
        hessian = hessian + torch.diag(penalty) + identity
        direction = torch.linalg.solve(hessian, gradient)
        before = _logistic_objective(train_design, labels, weights, float(l2))
        step_size = 1.0
        candidate = weights - direction
        for _ in range(16):
            candidate = weights - step_size * direction
            after = _logistic_objective(train_design, labels, candidate, float(l2))
            if bool(after <= before + 1e-12):
                break
            step_size *= 0.5
        weights = candidate
        if float(direction.abs().max() * step_size) < 1e-10:
            break
    return (test_design @ weights).sigmoid().float()


def three_fold_logistic_oof(
    features: torch.Tensor,
    labels: torch.Tensor,
    folds: torch.Tensor,
) -> dict[str, object]:
    """Return fixed three-fold held-out probabilities and AUCs."""

    if (
        features.ndim != 2
        or labels.shape != (len(features),)
        or folds.shape != labels.shape
        or labels.dtype != torch.bool
        or folds.dtype != torch.long
    ):
        raise ValueError("P4-E9 Bond probe OOF tensors have invalid shapes or dtypes")
    if set(folds.tolist()) != set(range(FOLDS)):
        raise ValueError("P4-E9 Bond probe OOF requires all three folds")
    scores = torch.empty(len(features), dtype=torch.float32)
    per_fold = {}
    for fold in range(FOLDS):
        test = folds == fold
        train = ~test
        if min(int(labels[test].sum()), int((~labels[test]).sum())) == 0:
            raise ValueError(f"P4-E9 Bond probe OOF fold {fold} test lacks a class")
        scores[test] = fit_logistic_predict(features[train], labels[train], features[test])
        per_fold[str(fold)] = {
            "samples": int(test.sum()),
            "positives": int(labels[test].sum()),
            "auc": p3e29.binary_auc(scores[test], labels[test]),
        }
    return {
        "auc": p3e29.binary_auc(scores, labels),
        "macro_auc": sum(value["auc"] for value in per_fold.values()) / FOLDS,
        "per_fold": per_fold,
        "scores": scores,
    }


class _PixelOnlyCollator:
    """Collate only observable PNG pixels and identities, never GT structure fields."""

    def __init__(self) -> None:
        self._image = p1a.G3AtomCollator(maximum_atoms=128, image_size=IMAGE_SIZE)

    def __call__(self, samples: list[dict]) -> dict[str, object]:
        return {
            "pixel_values": torch.stack(
                [self._image._image_tensor(sample["image"]) for sample in samples]
            ),
            "keys": [str(sample["key"]) for sample in samples],
        }


def _load_oracle() -> tuple[list[dict], dict[str, dict], list, dict]:
    paths = {
        "audit": ORACLE_AUDIT,
        "actions": ORACLE_ACTIONS,
        "images": ORACLE_IMAGES,
        "fit": ORACLE_FIT,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise BondEvidenceProbeError(
            "P4-E9 Oracle is not complete; static probe is waiting for: " + ", ".join(missing)
        )
    for name, path in paths.items():
        actual = file_sha256(path)
        if actual != ORACLE_SHA256[name]:
            raise BondEvidenceProbeError(
                f"P4-E9 Bond probe {name} SHA256 differs: {actual}"
            )
    audit = json.loads(ORACLE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("decision") != "recovery_oracle_go":
        raise BondEvidenceProbeError("P4-E9 Oracle did not pass its protocol gate")
    actions = [row for row in _read_jsonl(ORACLE_ACTIONS) if row.get("split") == "fit"]
    if not actions or any(row.get("route") not in ROUTES for row in actions):
        raise BondEvidenceProbeError("P4-E9 fit actions are empty or invalid")
    action_keys = {str(row["sample_key"]) for row in actions}
    images = {
        str(row["sample_key"]): row
        for row in _read_jsonl(ORACLE_IMAGES)
        if str(row["sample_key"]) in action_keys
    }
    if set(images) != action_keys or any(row.get("split") != "fit" for row in images.values()):
        raise BondEvidenceProbeError("P4-E9 Bond probe may read only complete fit action images")
    fit, fit_audit = load_g3_sample_manifest(
        ORACLE_FIT,
        expected_sha256=ORACLE_SHA256["fit"],
        expected_samples=14_000,
        expected_molecules=3_500,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    selected = [record for record in fit if record.sample_key in action_keys]
    if {record.sample_key for record in selected} != action_keys:
        raise BondEvidenceProbeError("P4-E9 fit manifest lacks an action image")
    return actions, images, selected, {"oracle": audit, "fit_manifest": fit_audit}


@torch.inference_mode()
def collect_frozen_records(
    actions: Sequence[Mapping[str, object]],
    images: Mapping[str, Mapping[str, object]],
    manifest: list,
    *,
    device: torch.device,
) -> tuple[list[dict], dict[str, dict], str, str]:
    """Collect P4-E3 local and P3-E29 endpoint features from PNGs only."""

    field, atom_head, atom_vision, frozen_block, atom_vision_sha = p4e6._load_model(device)
    bond_vision, bond, proposal = p3e29._load_bond_source(device)
    modules = (field, atom_head, atom_vision, frozen_block, bond_vision, bond)
    if any(any(parameter.requires_grad for parameter in module.parameters()) for module in modules):
        raise BondEvidenceProbeError("P4-E9 Bond probe source model is not frozen")
    actions_by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in actions:
        actions_by_key[str(row["sample_key"])].append(row)
    records = []
    coverage = {
        route: {"targets": 0, "covered": 0}
        for route in ROUTES
    }
    digest = hashlib.sha256()
    collator = _PixelOnlyCollator()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=BATCH_SIZE):
        pixels = batch["pixel_values"].to(device)
        objects, stroke = p1t.model_inputs(
            atom_head,
            atom_vision,
            frozen_block,
            pixels,
            device=device,
            training=False,
        )
        with _autocast_context(device, "bf16"):
            field_output = field(objects, stroke)
        peaks = extract_regional_maximum_objects(
            decode_atom_center_field(field_output), maximum_objects=recovery_data.CANDIDATES
        )
        observations = recovery_data.build_local_recovery_observations(
            pixels, field_output, peaks
        )
        with _autocast_context(device, "bf16"):
            bond_stroke = bond_vision.forward_stroke_resolution(pixels)
            bond_output = p3e29.forward_bond_source(bond_vision, bond, bond_stroke)
            decoded_bonds = decode_predicted_bond_proposals(bond_output, proposal)
        for batch_index, key_value in enumerate(batch["keys"]):
            key = str(key_value)
            source_ids = observations["candidate_source_ids"][batch_index].cpu()
            anchors = observations["candidate_anchors"][batch_index].cpu()
            baseline = torch.cat(
                (
                    observations["candidate_features"][batch_index].cpu().float(),
                    observations["candidate_scores"][batch_index, :, None].cpu(),
                    observations["candidate_ranks"][batch_index, :, None].cpu(),
                ),
                dim=1,
            )
            incident = p3e29.continuous_incident_features(
                anchors.to(device),
                decoded_bonds["bond_scores"][batch_index],
                decoded_bonds["bond_endpoints"][batch_index],
                decoded_bonds["bond_types"][batch_index],
                decoded_bonds["bond_mask"][batch_index],
            ).cpu()
            current_actions = actions_by_key[key]
            action_map = {int(row["base_source_id"]): str(row["route"]) for row in current_actions}
            if len(action_map) != len(current_actions):
                raise BondEvidenceProbeError("P4-E9 action source is not mutually exclusive")
            metadata = images[key]
            record = {
                "sample_key": key,
                "molecule_id": str(metadata["molecule_id"]),
                "scaffold": str(metadata["scaffold"]),
                "source_ids": source_ids,
                "baseline_features": baseline,
                "incident_features": incident,
                "actions": action_map,
                "unsupported_source_ids": list(metadata["unsupported_source_ids"]),
            }
            _candidate_statuses(record)
            records.append(record)
            digest.update(key.encode("utf-8"))
            digest.update(baseline.numpy().tobytes())
            digest.update(incident.numpy().tobytes())
            for route in ROUTES:
                target_rows = [row for row in current_actions if row["route"] == route]
                targets = torch.tensor(
                    [row["target_anchor"] for row in target_rows], dtype=torch.float32
                ).reshape(-1, 2)
                current = endpoint_target_coverage(
                    targets,
                    decoded_bonds["bond_endpoints"][batch_index].float().cpu(),
                    decoded_bonds["bond_mask"][batch_index].cpu(),
                )
                coverage[route]["targets"] += int(current["targets"])
                coverage[route]["covered"] += int(current["covered"])
    for route in ROUTES:
        targets = coverage[route]["targets"]
        coverage[route]["coverage"] = coverage[route]["covered"] / max(targets, 1)
    return records, coverage, atom_vision_sha, digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise BondEvidenceProbeError(f"P4-E9 Bond probe output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise BondEvidenceProbeError("P4-E9 Bond probe requires CUDA BF16")
    for path, expected in (
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
        (p3e29.BOND_SOURCE, p3e29.BOND_SOURCE_SHA256),
        (p3e29.VISION_SOURCE, p3e29.VISION_SOURCE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise BondEvidenceProbeError(f"P4-E9 Bond probe source SHA differs: {path}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    actions, images, manifest, input_audit = _load_oracle()
    started = time.perf_counter()
    records, coverage, atom_vision_sha, feature_tensor_sha = collect_frozen_records(
        actions, images, manifest, device=torch.device("cuda")
    )
    oracle = input_audit["oracle"]
    route_results = {}
    for route in ROUTES:
        fit_actions = sum(
            int(row["route"] == route and row["split"] == "fit") for row in actions
        )
        if not bool(oracle["route_capacity"][route]):
            route_results[route] = {
                "status": "skipped_capacity_no_go",
                "oracle_fit_actions": fit_actions,
                "include_bond_evidence": False,
                "endpoint_coverage": coverage[route],
            }
            continue
        dataset = build_route_dataset(records, route=route)
        baseline = three_fold_logistic_oof(
            dataset["baseline_features"], dataset["labels"], dataset["folds"]
        )
        augmented = three_fold_logistic_oof(
            dataset["augmented_features"], dataset["labels"], dataset["folds"]
        )
        lift = float(augmented["auc"] - baseline["auc"])
        include = evidence_gate(lift, float(coverage[route]["coverage"]))
        route_results[route] = {
            "status": "include" if include else "exclude",
            "oracle_fit_actions": fit_actions,
            "include_bond_evidence": include,
            "baseline_34d_oof": {
                key: value for key, value in baseline.items() if key != "scores"
            },
            "augmented_oof": {
                key: value for key, value in augmented.items() if key != "scores"
            },
            "auc_lift": lift,
            "endpoint_coverage": coverage[route],
            "sampling": {
                key: value
                for key, value in dataset.items()
                if key
                in {
                    "fold_counts",
                    "excluded_counts",
                    "positives",
                    "none_negatives",
                }
            },
        }
    formal = route_results[FORMAL_ROUTE]
    include = bool(formal["include_bond_evidence"])
    scaffold_sets = {
        fold: {
            str(record["scaffold"]).strip() or f"molecule:{record['molecule_id']}"
            for record in records
            if scaffold_fold(str(record["scaffold"]), str(record["molecule_id"])) == fold
        }
        for fold in range(FOLDS)
    }
    overlap = sum(
        len(scaffold_sets[left] & scaffold_sets[right])
        for left in range(FOLDS)
        for right in range(left + 1, FOLDS)
    )
    audit = {
        "schema_version": "g3-p4e9-bond-evidence-probe-fit-0.1",
        "decision": (
            "bond_endpoint_evidence_include"
            if include
            else "bond_endpoint_evidence_exclude"
        ),
        "formal_route": FORMAL_ROUTE,
        "include_bond_evidence": include,
        "route_results": route_results,
        "protocol": {
            "data": "Oracle fit action images only",
            "baseline_features": "32D frozen P4-E3 candidate feature + normalized score + rank",
            "augmented_features": list(p3e29.incident_feature_names()),
            "labels": "route positive + exact 1:3 NONE; other route and UNSUPPORTED masked",
            "folds": "SHA256(seed:scaffold) modulo 3",
            "classifier": "fixed standardized L2 logistic, damped Newton, no tuning",
            "logistic_l2": LOGISTIC_L2,
            "logistic_steps": LOGISTIC_STEPS,
            "minimum_auc_lift": MIN_AUC_LIFT,
            "minimum_target_endpoint_coverage": MIN_ENDPOINT_COVERAGE,
            "target_endpoint_radius_pixels": ENDPOINT_RADIUS_PIXELS,
        },
        "protocol_checks": {
            "fit_action_images_only": len(records) == len(images),
            "formal_refine_route_capacity_go": bool(
                oracle["route_capacity"][FORMAL_ROUTE]
            ),
            "split_skipped_capacity_no_go": (
                route_results["ADD_SPLIT"]["status"] == "skipped_capacity_no_go"
            ),
            "scaffold_cross_fold_overlap_zero": overlap == 0,
            "frozen_sources": True,
            "gt_fields_not_forwarded": True,
            "confirm_v2_not_read": True,
            "external_and_sealed_not_read": True,
        },
        "lineage": {
            "oracle": {
                name: {"path": str(path.resolve()), "sha256": ORACLE_SHA256[name]}
                for name, path in {
                    "audit": ORACLE_AUDIT,
                    "actions": ORACLE_ACTIONS,
                    "images": ORACLE_IMAGES,
                    "fit": ORACLE_FIT,
                }.items()
            },
            "p4e3_checkpoint": str(p4e6.P4E3_CHECKPOINT.resolve()),
            "p4e3_checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
            "p4e3_vision_sha256": atom_vision_sha,
            "p3e29_bond_source": str(p3e29.BOND_SOURCE.resolve()),
            "p3e29_bond_source_sha256": p3e29.BOND_SOURCE_SHA256,
            "p3e29_vision_source": str(p3e29.VISION_SOURCE.resolve()),
            "p3e29_vision_source_sha256": p3e29.VISION_SOURCE_SHA256,
            "fit_manifest": input_audit["fit_manifest"],
        },
        "feature_tensor_sha256": feature_tensor_sha,
        "images": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "data_access": {
            "png_read": True,
            "oracle_action_labels_read": True,
            "gt_atom_read": False,
            "gt_bond_read": False,
            "confirm_v2_read": False,
            "external_read": False,
        },
    }
    if not all(audit["protocol_checks"].values()):
        raise BondEvidenceProbeError("P4-E9 Bond probe protocol check failed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
