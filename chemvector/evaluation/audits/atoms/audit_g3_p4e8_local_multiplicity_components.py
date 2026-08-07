#!/usr/bin/env python3
"""Measure deficient local GT/candidate components on P4-E3 failure images."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e7_offset_radius_ab as p4e7,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21,
)
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


AUDIT_NAME = "chemvector-g3-p4e8-local-multiplicity-components"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P4E7_AUDIT = Path(
    "outputs/audits/chemvector-g3-p4e7-offset-radius-ab/audit.json"
)
P4E7_AUDIT_SHA256 = (
    "1334135d7b32cc312c39947130e50b103c4331d17c2ab337aaaeff9afffb7e75"
)
FAILURES = p4e6.OUTPUT / "failures.jsonl"
FAILURES_SHA256 = (
    "6a1a32937649eacd393bf06be8b32aac755f4a4ebf39cb5c2cc049dfc2b23006"
)
TOP_K = 1024
RADII = (6.0, 10.0)
IMAGE_SIZE = 768
BATCH_SIZE = 8
EXPECTED_IMAGES = 43
EXPECTED_DEFICIENCY_6PX = 58
EXPECTED_DEFICIENCY_10PX = 23
BOUNDED_MAX_GT = 4
BOUNDED_MAX_DEFICIENCY = 2
BOUNDED_COVERAGE = 0.90
SEED = 20260848


def deficient_bipartite_components(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    *,
    radius_pixels: float,
) -> list[dict]:
    if predicted.ndim != 2 or predicted.shape[-1] != 2:
        raise ValueError("P4-E8 predicted anchors must have shape [P,2]")
    if expected.ndim != 2 or expected.shape[-1] != 2:
        raise ValueError("P4-E8 expected anchors must have shape [G,2]")
    distances = torch.cdist(expected.float(), predicted.float()) * IMAGE_SIZE
    eligible = distances <= radius_pixels
    gt_to_candidates = [torch.where(row)[0].tolist() for row in eligible]
    candidate_to_gt = {
        candidate: torch.where(eligible[:, candidate])[0].tolist()
        for candidate in torch.where(eligible.any(dim=0))[0].tolist()
    }
    visited_gt: set[int] = set()
    components = []
    for seed_gt in range(len(expected)):
        if seed_gt in visited_gt:
            continue
        current_gt = {seed_gt}
        current_candidates: set[int] = set()
        pending_gt = [seed_gt]
        pending_candidates: list[int] = []
        visited_gt.add(seed_gt)
        while pending_gt or pending_candidates:
            while pending_gt:
                gt_index = pending_gt.pop()
                for candidate in gt_to_candidates[gt_index]:
                    if candidate not in current_candidates:
                        current_candidates.add(candidate)
                        pending_candidates.append(candidate)
            while pending_candidates:
                candidate = pending_candidates.pop()
                for gt_index in candidate_to_gt[candidate]:
                    if gt_index not in current_gt:
                        current_gt.add(gt_index)
                        visited_gt.add(gt_index)
                        pending_gt.append(gt_index)
        gt_indices = sorted(current_gt)
        candidate_indices = sorted(current_candidates)
        assignments = p4e6.maximum_cardinality_assignments(
            predicted[candidate_indices] if candidate_indices else predicted[:0],
            expected[gt_indices],
            radius_pixels=radius_pixels,
            image_size=IMAGE_SIZE,
        )
        matched = sum(value is not None for value in assignments)
        deficiency = len(gt_indices) - matched
        if deficiency:
            components.append(
                {
                    "gt_indices": gt_indices,
                    "candidate_indices": candidate_indices,
                    "gt_count": len(gt_indices),
                    "candidate_count": len(candidate_indices),
                    "matched_count": matched,
                    "deficiency": deficiency,
                    "kind": (
                        "no_candidate" if not candidate_indices else "shared_deficient"
                    ),
                }
            )
    return components


def choose_decision(components: list[dict]) -> tuple[str, dict[str, bool]]:
    shared = [item for item in components if item["kind"] == "shared_deficient"]
    shared_deficiency = sum(item["deficiency"] for item in shared)
    bounded_deficiency = sum(
        item["deficiency"]
        for item in shared
        if item["gt_count"] <= BOUNDED_MAX_GT
        and item["deficiency"] <= BOUNDED_MAX_DEFICIENCY
    )
    coverage = bounded_deficiency / max(shared_deficiency, 1)
    checks = {
        "shared_deficiency_present": shared_deficiency > 0,
        "bounded_components_cover_90pct_shared_deficiency": coverage >= BOUNDED_COVERAGE,
        "maximum_bounded_gt_count_4": all(
            item["gt_count"] <= BOUNDED_MAX_GT for item in shared
        ),
        "maximum_bounded_deficiency_2": all(
            item["deficiency"] <= BOUNDED_MAX_DEFICIENCY for item in shared
        ),
    }
    return (
        "bounded_local_multiplicity_go"
        if all(checks.values())
        else "unbounded_or_mixed_local_multiplicity",
        checks,
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E8 output is not empty: {OUTPUT}")
    for path, expected in (
        (P4E7_AUDIT, P4E7_AUDIT_SHA256),
        (FAILURES, FAILURES_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E8 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E8 requires CUDA BF16")
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

    failure_keys = {
        json.loads(line)["sample_key"]
        for line in FAILURES.read_text(encoding="utf-8").splitlines()
    }
    dev, manifest_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    failure_manifest = [record for record in dev if record.sample_key in failure_keys]
    if len(failure_manifest) != EXPECTED_IMAGES:
        raise RuntimeError("P4-E8 failure manifest is incomplete")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=IMAGE_SIZE)
    components_by_radius = {f"{radius:g}": [] for radius in RADII}
    started = time.perf_counter()

    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(
            failure_manifest, collator, batch_size=BATCH_SIZE
        ):
            objects, stroke = p1t.model_inputs(
                atom_head,
                vision,
                frozen_block,
                batch["pixel_values"],
                device=device,
                training=False,
            )
            with _autocast_context(device, "bf16"):
                output = field(objects, stroke)
            peaks = extract_regional_maximum_objects(
                decode_atom_center_field(output), maximum_objects=TOP_K
            )
            for index, sample_key_value in enumerate(batch["keys"]):
                expected = batch["atom_anchors"][index][batch["atom_mask"][index]]
                for radius in RADII:
                    components = deficient_bipartite_components(
                        peaks[index]["anchors"], expected, radius_pixels=radius
                    )
                    for component in components:
                        component["sample_key"] = str(sample_key_value)
                        component["radius_pixels"] = radius
                    components_by_radius[f"{radius:g}"].extend(components)

    summaries = {}
    for radius, components in components_by_radius.items():
        summaries[radius] = {
            "components": len(components),
            "deficiency": sum(item["deficiency"] for item in components),
            "kind_counts": dict(Counter(item["kind"] for item in components)),
            "gt_count_distribution": dict(
                sorted(Counter(item["gt_count"] for item in components).items())
            ),
            "candidate_count_distribution": dict(
                sorted(Counter(item["candidate_count"] for item in components).items())
            ),
            "deficiency_distribution": dict(
                sorted(Counter(item["deficiency"] for item in components).items())
            ),
            "maximum_gt_count": max(item["gt_count"] for item in components),
            "maximum_candidate_count": max(
                item["candidate_count"] for item in components
            ),
            "maximum_deficiency": max(item["deficiency"] for item in components),
        }
    decision, capability_checks = choose_decision(components_by_radius["6"])
    protocol_checks = {
        "failure_images_43": len(failure_manifest) == EXPECTED_IMAGES,
        "radius6_deficiency_58": summaries["6"]["deficiency"]
        == EXPECTED_DEFICIENCY_6PX,
        "radius10_deficiency_23": summaries["10"]["deficiency"]
        == EXPECTED_DEFICIENCY_10PX,
        "confirm_v2_not_read": True,
        "external_and_sealed_not_read": True,
    }
    if not all(protocol_checks.values()):
        decision = "local_multiplicity_protocol_failed"
    audit = {
        "schema_version": "g3-p4e8-local-multiplicity-components-audit-0.1",
        "decision": decision,
        "capability_checks": capability_checks,
        "protocol_checks": protocol_checks,
        "protocol": {
            "top_k": TOP_K,
            "radii_pixels": list(RADII),
            "bounded_max_gt": BOUNDED_MAX_GT,
            "bounded_max_deficiency": BOUNDED_MAX_DEFICIENCY,
            "bounded_coverage": BOUNDED_COVERAGE,
            "matching": "maximum-cardinality-Hungarian-per-connected-component",
            "training_performed": False,
        },
        "summaries": summaries,
        "components": components_by_radius,
        "manifest": {
            **manifest_audit,
            "failure_sample_count": len(failure_manifest),
        },
        "lineage": {
            "checkpoint": str(p4e6.P4E3_CHECKPOINT.resolve()),
            "checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "p4e7_audit_sha256": P4E7_AUDIT_SHA256,
            "p4e6_failures_sha256": FAILURES_SHA256,
        },
        "data_access": {
            "dev_failure_png_read": True,
            "other_dev_png_read": False,
            "pilot_train_read": False,
            "confirm_v2_read": False,
            "external_and_sealed_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
