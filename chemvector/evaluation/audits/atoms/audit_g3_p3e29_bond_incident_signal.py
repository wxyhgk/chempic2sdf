#!/usr/bin/env python3
"""Audit whether frozen BondObject endpoints carry useful top192 Atom evidence."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tarfile
import time

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_gt_candidate_compatibility,
)
from chemvector.components.bonds.objects.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
)
from chemvector.components.graph.joint.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_bond_proposals,
)
from chemvector.evaluation import g3_p3e21_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e23_top192_cycle_recovery as p3e23,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21_cache,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p3e25_hard_latent_occupancy_micro160 as p3e25,
)
from chemvector.training.atoms.train_spatial_atom_object import (
    load_grounded_vision,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_atom_bond_scene import _image_tensor
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e29-bond-incident-signal-micro160"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
EVIDENCE_CACHE = OUTPUT / "evidence.pt"
BOND_SOURCE = Path(
    "outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt"
)
BOND_SOURCE_SHA256 = (
    "e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac"
)
VISION_SOURCE = Path(
    "outputs/checkpoints/chemvector-v0729-whole-set-stable-distance-overfit100/last.pt"
)
VISION_SOURCE_SHA256 = (
    "1eede2e814efc0c440c26eba88bc8e9ed86e0911d7727b770a992c1ea2671c6a"
)
MICRO_SAMPLE_SHA256 = (
    "06f44d30a88013643be6041fe55741fd75f6d58bcf1d875a22224972a7944501"
)
IMAGE_SIZE = 768.0
SIGMAS_PX = (4.0, 8.0, 12.0, 20.0)
BATCH_SIZE = 8
ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MIN_TAIL_RESCUE_IMAGES = 5
MIN_CANDIDATE_AUC = 0.65
MAX_REACHABLE_RECALL_REGRESSION = 0.005
SEED = 20260829


def incident_feature_names(type_count: int = 8) -> tuple[str, ...]:
    names = ["nearest_endpoint_distance", "nearest_endpoint_score"]
    for sigma in SIGMAS_PX:
        names.extend((f"support_max_sigma{sigma:g}", f"support_sum_sigma{sigma:g}"))
    names.extend(f"type_support_{index}" for index in range(type_count))
    return tuple(names)


def continuous_incident_features(
    candidate_anchors: torch.Tensor,
    bond_scores: torch.Tensor,
    bond_endpoints: torch.Tensor,
    bond_types: torch.Tensor,
    bond_mask: torch.Tensor,
    *,
    image_size: float = IMAGE_SIZE,
) -> torch.Tensor:
    """Aggregate continuous endpoint support without assigning bonds to candidates."""

    if candidate_anchors.ndim != 2 or candidate_anchors.size(-1) != 2:
        raise ValueError("P3-E29 candidate anchors must have shape [candidate,2]")
    if bond_endpoints.ndim != 3 or bond_endpoints.shape[1:] != (2, 2):
        raise ValueError("P3-E29 Bond endpoints must have shape [bond,2,2]")
    bonds = len(bond_endpoints)
    if bond_scores.shape != (bonds,) or bond_mask.shape != (bonds,):
        raise ValueError("P3-E29 Bond score/mask shape differs")
    if bond_types.ndim != 2 or bond_types.size(0) != bonds:
        raise ValueError("P3-E29 Bond type shape differs")
    tensors = (candidate_anchors, bond_scores, bond_endpoints, bond_types)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("P3-E29 incident input contains a non-finite value")

    candidates = candidate_anchors.float()
    valid = bond_mask.bool()
    feature_count = len(incident_feature_names(bond_types.size(1)))
    if not bool(valid.any()):
        result = candidates.new_zeros((len(candidates), feature_count))
        result[:, 0] = math.sqrt(2.0)
        return result

    scores = bond_scores[valid].float().clamp(0.0, 1.0)
    endpoints = bond_endpoints[valid].float()
    types = bond_types[valid].float()
    distances = torch.linalg.vector_norm(
        (candidates[:, None, None] - endpoints[None]) * image_size,
        dim=-1,
    ).amin(dim=-1)
    nearest_distance, nearest_bond = distances.min(dim=1)
    features = [
        nearest_distance / image_size,
        scores[nearest_bond],
    ]
    kernels = {}
    for sigma in SIGMAS_PX:
        kernel = torch.exp(-0.5 * (distances / sigma).square()) * scores[None]
        kernels[sigma] = kernel
        features.extend((kernel.max(dim=1).values, kernel.sum(dim=1)))
    type_support = kernels[12.0] @ types
    return torch.cat((torch.stack(features, dim=1), type_support), dim=1)


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("P3-E29 AUC inputs must be equal vectors")
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if min(positives, negatives) == 0:
        raise ValueError("P3-E29 AUC requires both classes")
    ranks = rankdata(scores.detach().double().cpu().numpy(), method="average")
    positive_rank_sum = float(ranks[labels.cpu().numpy()].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    std = values.std(unbiased=False)
    if float(std) < 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def selection_metrics(
    records: list[dict],
    evidence: dict[str, torch.Tensor],
    *,
    alpha: float | None,
) -> dict:
    totals = defaultdict(int)
    for record in records:
        key = str(record["sample_key"])
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        count = len(expected)
        support = evidence[key].float()
        if alpha is None:
            score = _standardize(support)
        else:
            score = _standardize(record["raw_scores"]) + alpha * _standardize(
                support
            )
        selected = torch.argsort(score, descending=True, stable=True)[:count]
        matched = p1a.maximum_cardinality_radius_match_count(
            record["candidate_anchors"][selected],
            expected,
            radius_pixels=6.0,
            image_size=IMAGE_SIZE,
        )
        old_reachable = (
            p1a.maximum_cardinality_radius_match_count(
                record["candidate_anchors"][:128],
                expected,
                radius_pixels=6.0,
                image_size=IMAGE_SIZE,
            )
            == count
        )
        exact = matched == count
        totals["images"] += 1
        totals["matched_atoms"] += matched
        totals["target_atoms"] += count
        totals["exact_images"] += int(exact)
        group = "reachable" if old_reachable else "tail"
        totals[f"{group}_images"] += 1
        totals[f"{group}_matched_atoms"] += matched
        totals[f"{group}_target_atoms"] += count
        totals[f"{group}_exact_images"] += int(exact)
    result = dict(totals)
    for group in ("", "reachable_", "tail_"):
        if result.get(f"{group}target_atoms", 0) == 0:
            result[f"{group}recall"] = 0.0
            continue
        result[f"{group}recall"] = (
            result[f"{group}matched_atoms"] / result[f"{group}target_atoms"]
        )
    return result


def choose_oracle_fusion(results: dict[str, dict]) -> tuple[str, dict]:
    candidates = [(name, value) for name, value in results.items() if name != "bond_only"]
    name, value = max(
        candidates,
        key=lambda item: (
            item[1]["tail_exact_images"],
            item[1]["exact_images"],
            item[1]["matched_atoms"],
            -float(item[0].split("=")[1]),
        ),
    )
    return name, value


def _load_micro160() -> tuple[list[dict], dict, dict[str, object]]:
    for path, expected in (
        (p3e23.TOP256_CACHE, p3e23.TOP256_CACHE_SHA256),
        (p3e23.OLD_CACHE_ROOT / "train.pt", p3e23.OLD_CACHE_SHA256["train"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E29 input SHA256 differs: {path}")
    old = torch.load(
        p3e23.OLD_CACHE_ROOT / "train.pt", map_location="cpu", weights_only=False
    )["records"]
    top256 = torch.load(p3e23.TOP256_CACHE, map_location="cpu", weights_only=False)
    records, input_audit = p3e25.true_global192_records(top256, old)
    micro, selection = p3e25.select_micro160(records, old)
    if selection["sample_keys_sha256"] != MICRO_SAMPLE_SHA256:
        raise ValueError("P3-E29 Micro160 sample digest differs")
    manifest, manifest_audit = load_g3_sample_manifest(
        p3e21_cache.MANIFEST_ROOT / "train.jsonl",
        expected_sha256=p3e21_cache.MANIFESTS["train"][0],
        expected_samples=p3e21_cache.MANIFESTS["train"][1],
        expected_molecules=p3e21_cache.MANIFESTS["train"][2],
        expected_variants=cycle.VARIANTS,
    )
    by_key = {record.sample_key: record for record in manifest}
    selected_manifest = {key: by_key[key] for key in selection["sample_keys"]}
    return micro, {**input_audit, "selection": selection}, {
        "records": selected_manifest,
        "audit": manifest_audit,
    }


def _load_bond_source(device: torch.device):
    if file_sha256(BOND_SOURCE) != BOND_SOURCE_SHA256:
        raise ValueError("P3-E29 frozen Bond source SHA256 differs")
    if file_sha256(VISION_SOURCE) != VISION_SOURCE_SHA256:
        raise ValueError("P3-E29 frozen vision source SHA256 differs")
    payload = torch.load(BOND_SOURCE, map_location="cpu", weights_only=False)
    source_graph = Path(payload["training_args"]["grounded_graph_checkpoint"])
    if source_graph.resolve() != VISION_SOURCE.resolve():
        raise ValueError("P3-E29 Bond source was trained with another vision source")
    vision, _, _ = load_grounded_vision(VISION_SOURCE, device)
    config = DenseBondObjectConfig(**payload["bond_config"])
    bond = (
        MultiScaleDenseBondObjectDetector(config)
        if payload["bond_multiscale"]
        else DenseBondObjectDetector(config)
    ).to(device)
    bond.load_state_dict(payload["bond_head"], strict=True)
    bond.eval().requires_grad_(False)
    proposal = PredictedProposalConfig(**payload["proposal_config"])
    if proposal.maximum_bonds != 192:
        raise ValueError("P3-E29 frozen source must expose 192 Bond proposals")
    return vision.eval().requires_grad_(False), bond, proposal


def forward_bond_source(
    vision: torch.nn.Module,
    bond: torch.nn.Module,
    stroke: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Forward the frozen source without relying on compatibility-module identity."""

    if isinstance(bond, MultiScaleDenseBondObjectDetector):
        stride8 = vision.forward_high_resolution_from_stroke(stroke)
        stride16 = stride8
        for stage in vision.stages[3:5]:
            stride16 = stage(stride16)
        return bond(stroke, stride8, stride16)
    if isinstance(bond, DenseBondObjectDetector):
        return bond(stroke)
    raise TypeError(f"P3-E29 unsupported frozen Bond source: {type(bond)!r}")


def _read_images(manifest: dict[str, object], keys: list[str]) -> torch.Tensor:
    archives: dict[str, tarfile.TarFile] = {}
    images = []
    try:
        for key in keys:
            row = manifest[key]
            archive = archives.get(row.shard)
            if archive is None:
                archive = tarfile.open(row.shard, "r")
                archives[row.shard] = archive
            handle = archive.extractfile(f"{key}.png")
            if handle is None:
                raise FileNotFoundError(f"{key}.png")
            with Image.open(BytesIO(handle.read())) as image:
                images.append(_image_tensor(image))
    finally:
        for archive in archives.values():
            archive.close()
    return torch.stack(images)


@torch.inference_mode()
def collect_evidence(
    records: list[dict], manifest: dict[str, object], device: torch.device
) -> tuple[list[dict], dict]:
    vision, bond, proposal = _load_bond_source(device)
    collected = []
    digest = hashlib.sha256()
    feature_names = incident_feature_names()
    for start in range(0, len(records), BATCH_SIZE):
        current = records[start : start + BATCH_SIZE]
        keys = [str(record["sample_key"]) for record in current]
        pixels = _read_images(manifest, keys).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            stroke = vision.forward_stroke_resolution(pixels)
            output = forward_bond_source(vision, bond, stroke)
            decoded = decode_predicted_bond_proposals(output, proposal)
        for index, (key, record) in enumerate(zip(keys, current)):
            features = continuous_incident_features(
                record["candidate_anchors"].to(device),
                decoded["bond_scores"][index],
                decoded["bond_endpoints"][index],
                decoded["bond_types"][index],
                decoded["bond_mask"][index],
            ).cpu()
            value = {
                "sample_key": key,
                "incident_features": features.to(torch.float16),
                "bond_scores": decoded["bond_scores"][index].float().cpu().to(torch.float16),
                "bond_endpoints": decoded["bond_endpoints"][index].float().cpu().to(torch.float16),
                "bond_types": decoded["bond_types"][index].float().cpu().to(torch.float16),
                "bond_mask": decoded["bond_mask"][index].cpu(),
            }
            digest.update(key.encode())
            digest.update(features.numpy().tobytes())
            collected.append(value)
    return collected, {
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "evidence_tensor_sha256": digest.hexdigest(),
        "images": len(collected),
    }


def audit_signal(records: list[dict], cached: list[dict]) -> dict:
    cache_by_key = {str(record["sample_key"]): record for record in cached}
    support_index = incident_feature_names().index("support_max_sigma12")
    support = {
        key: value["incident_features"][:, support_index].float()
        for key, value in cache_by_key.items()
    }
    all_scores = []
    all_labels = []
    tail_target_percentiles = []
    tail_targets = 0
    for record in records:
        key = str(record["sample_key"])
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        compatibility = g3_gt_candidate_compatibility(
            expected, record["candidate_anchors"].float()
        )
        labels = compatibility.any(dim=0)
        all_scores.append(support[key])
        all_labels.append(labels)
        for target in range(len(expected)):
            if bool(compatibility[target, :128].any()):
                continue
            compatible_tail = compatibility[target, 128:]
            if not bool(compatible_tail.any()):
                raise RuntimeError("P3-E29 selected Micro image is not top192 reachable")
            positive = support[key][128:][compatible_tail].max()
            negatives = support[key][~labels]
            tail_target_percentiles.append(float((negatives <= positive).float().mean()))
            tail_targets += 1
    auc = binary_auc(torch.cat(all_scores), torch.cat(all_labels))
    results = {
        "bond_only": selection_metrics(records, support, alpha=None),
        **{
            f"alpha={alpha:g}": selection_metrics(records, support, alpha=alpha)
            for alpha in ALPHAS
        },
    }
    winner_name, winner = choose_oracle_fusion(results)
    baseline = results["alpha=0"]
    checks = {
        "candidate_auc_min065": auc >= MIN_CANDIDATE_AUC,
        "tail_rescue_min5_of27": winner["tail_exact_images"] >= MIN_TAIL_RESCUE_IMAGES,
        "tail_improves_over_raw": (
            winner["tail_exact_images"] > baseline["tail_exact_images"]
        ),
        "reachable_recall_regression_max005": (
            winner["reachable_recall"]
            >= baseline["reachable_recall"] - MAX_REACHABLE_RECALL_REGRESSION
        ),
        "all_160_have_evidence": len(cache_by_key) == len(records) == 160,
        "all_27_tail_images_present": winner["tail_images"] == 27,
    }
    return {
        "decision": (
            "bond_incident_signal_go" if all(checks.values()) else "bond_incident_signal_no_go"
        ),
        "checks": checks,
        "candidate_positive_negative_auc": auc,
        "tail_only_targets": tail_targets,
        "tail_target_support_percentile_mean": float(np.mean(tail_target_percentiles)),
        "tail_target_support_percentile_min": float(np.min(tail_target_percentiles)),
        "selection_results": results,
        "oracle_fusion": {"name": winner_name, "metrics": winner},
        "raw_baseline": baseline,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E29 output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E29 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    started = time.perf_counter()
    records, input_audit, manifest = _load_micro160()
    cached, evidence_audit = collect_evidence(records, manifest["records"], device)
    signal = audit_signal(records, cached)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "g3-p3e29-bond-incident-evidence-0.1",
        "records": cached,
        "feature_protocol": evidence_audit,
        "source": {
            "bond_checkpoint": str(BOND_SOURCE.resolve()),
            "bond_sha256": BOND_SOURCE_SHA256,
            "vision_checkpoint": str(VISION_SOURCE.resolve()),
            "vision_sha256": VISION_SOURCE_SHA256,
        },
    }
    _atomic_torch_save(EVIDENCE_CACHE, payload)
    audit = {
        "schema_version": "g3-p3e29-bond-incident-signal-audit-0.1",
        **signal,
        "protocol": {
            "candidate_count": 192,
            "bond_proposal_count": 192,
            "sigmas_px": list(SIGMAS_PX),
            "fusion_alphas": list(ALPHAS),
            "oracle_fusion_is_signal_upper_bound_not_deployed_selector": True,
            "tail_rescue_gate": MIN_TAIL_RESCUE_IMAGES,
            "candidate_auc_gate": MIN_CANDIDATE_AUC,
            "maximum_reachable_recall_regression": MAX_REACHABLE_RECALL_REGRESSION,
        },
        "input": input_audit,
        "manifest": manifest["audit"],
        "evidence": {
            **evidence_audit,
            "path": str(EVIDENCE_CACHE.resolve()),
            "sha256": file_sha256(EVIDENCE_CACHE),
        },
        "data_access": {
            "png_read": True,
            "gt_atom_read_for_audit_only": True,
            "gt_bond_read": False,
            "render_json_read": False,
            "bond_layer_read": False,
            "semantic_layer_read": False,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
