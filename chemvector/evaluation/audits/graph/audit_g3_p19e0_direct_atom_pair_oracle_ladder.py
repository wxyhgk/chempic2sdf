#!/usr/bin/env python3
"""Replay v0.10.3 Direct AtomPair on GT and P4-Oracle Atom sets."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.joint.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    enumerate_unordered_atom_pairs,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0a_stress_bond_ceiling as p18e0a,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p19e0-direct-atom-pair-oracle-ladder-fresh640"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DIRECT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
)
DIRECT_CHECKPOINT_SHA256 = (
    "e0d769b457fdbcac2d405fa234ccd108c80640db63cb35a1d8ed36446eea0614"
)
SOURCE_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac"
)
VISION_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v0729-whole-set-stable-distance-overfit100/last.pt"
)
VISION_CHECKPOINT_SHA256 = (
    "1eede2e814efc0c440c26eba88bc8e9ed86e0911d7727b770a992c1ea2671c6a"
)
E0_CACHE = p18e0.CACHE
E0_CACHE_SHA256 = "009d9dcf77c0d65a17530c2afff6566f7f2265adc9383f750f9b43a7f7cd88cc"
E0_MANIFEST = p18e0.SELECTION_MANIFEST
E0_MANIFEST_SHA256 = "7aeb5df329338faab36b0e5a3baa5bc0e078273034c6cd6718e16f66864bb9bc"
E0A_CACHE = p18e0a.CACHE
E0A_CACHE_SHA256 = "37e84038112a7fdd7a04f1c6a20671f70bafeb3c65e969ab10c96d98898928e8"
BATCH_SIZE = 4
MAXIMUM_ATOMS = 128
SEED = 20260810
GATES = {
    "gt_atom_skeleton_exact_rate_min": 0.80,
    "gt_atom_stress_skeleton_exact_rate_min": 0.60,
    "gt_atom_edge_recall_min": 0.97,
    "p4_atom_coverage_exact": 608,
}


def _sample_atom_features(
    atom_output: dict[str, torch.Tensor],
    anchors: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = atom_output["object_features"]
    scores = atom_output["heatmap_logits"].sigmoid()
    batch, channels, height, width = features.shape
    if anchors.shape != (batch, MAXIMUM_ATOMS, 2) or mask.shape != (
        batch,
        MAXIMUM_ATOMS,
    ):
        raise ValueError("P19-E0 Atom route shapes differ")
    x = (anchors[..., 0] * width).long().clamp(0, width - 1)
    y = (anchors[..., 1] * height).long().clamp(0, height - 1)
    batch_indices = torch.arange(batch, device=anchors.device)[:, None]
    sampled_features = features.permute(0, 2, 3, 1)[batch_indices, y, x]
    sampled_scores = scores[:, 0][batch_indices, y, x]
    sampled_features = sampled_features.masked_fill(~mask[..., None], 0.0)
    sampled_scores = sampled_scores.masked_fill(~mask, 0.0)
    if sampled_features.shape != (batch, MAXIMUM_ATOMS, channels):
        raise RuntimeError("P19-E0 sampled Atom feature shape differs")
    return sampled_features, sampled_scores


def _route_anchors(
    batch: dict,
    p4_records: list[dict],
    *,
    route: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors = torch.zeros(
        len(p4_records), MAXIMUM_ATOMS, 2, device=device, dtype=torch.float32
    )
    mask = torch.zeros(len(p4_records), MAXIMUM_ATOMS, device=device, dtype=torch.bool)
    target_ids = torch.full(
        (len(p4_records), MAXIMUM_ATOMS), -1, device=device, dtype=torch.long
    )
    for index, record in enumerate(p4_records):
        expected = batch["atom_anchors"][index][batch["atom_mask"][index]].float()
        if route == "gt_atom":
            count = len(expected)
            anchors[index, :count] = expected.to(device)
            mask[index, :count] = True
            target_ids[index, :count] = torch.arange(count, device=device)
            continue
        if route != "p4_oracle_atom":
            raise ValueError(f"P19-E0 unknown route: {route}")
        candidates = record["atom_candidate_anchors"][
            record["atom_candidate_mask"]
        ].float()
        assignment = p18e0._maximum_radius_assignment(
            candidates, expected, p18e0.ATOM_RADIUS_PX
        )
        selected_targets = torch.where(assignment >= 0)[0]
        count = len(selected_targets)
        if count:
            anchors[index, :count] = candidates[
                assignment[selected_targets]
            ].to(device)
            mask[index, :count] = True
            target_ids[index, :count] = selected_targets.to(device)
    return anchors, mask, target_ids


def _sample_metrics(
    *,
    probabilities: torch.Tensor,
    chemical: torch.Tensor,
    pairs: torch.Tensor,
    target_ids: torch.Tensor,
    target_atom_count: int,
    target_refs: torch.Tensor,
    target_types: torch.Tensor,
    threshold: float,
) -> dict:
    target_edges = {
        tuple(sorted((int(refs[0]), int(refs[1])))): int(target_types[index])
        for index, refs in enumerate(target_refs.tolist())
    }
    selected = probabilities >= threshold
    predicted_edges = {}
    for row in torch.where(selected)[0].tolist():
        begin, end = pairs[row].tolist()
        mapped = (int(target_ids[begin]), int(target_ids[end]))
        if min(mapped) < 0 or mapped[0] == mapped[1]:
            continue
        predicted_edges[tuple(sorted(mapped))] = int(chemical[row])
    target_keys = set(target_edges)
    predicted_keys = set(predicted_edges)
    true_positive = len(target_keys & predicted_keys)
    atom_values = target_ids[target_ids >= 0]
    atom_exact = (
        len(atom_values) == target_atom_count
        and len(atom_values.unique()) == target_atom_count
    )
    skeleton_exact = atom_exact and predicted_keys == target_keys
    typed_exact = skeleton_exact and all(
        predicted_edges[key] == value for key, value in target_edges.items()
    )
    return {
        "target_atoms": target_atom_count,
        "selected_atoms": len(atom_values),
        "atom_exact": atom_exact,
        "target_bonds": len(target_keys),
        "selected_bonds": len(predicted_keys),
        "true_positive": true_positive,
        "false_positive": len(predicted_keys - target_keys),
        "false_negative": len(target_keys - predicted_keys),
        "skeleton_exact": skeleton_exact,
        "typed_exact": typed_exact,
    }


def _summarize(rows: list[dict]) -> dict:
    integer_names = (
        "target_atoms",
        "selected_atoms",
        "atom_exact",
        "target_bonds",
        "selected_bonds",
        "true_positive",
        "false_positive",
        "false_negative",
        "skeleton_exact",
        "typed_exact",
    )
    totals = {name: sum(int(row[name]) for row in rows) for name in integer_names}
    precision = totals["true_positive"] / max(totals["selected_bonds"], 1)
    recall = totals["true_positive"] / max(totals["target_bonds"], 1)
    images = len(rows)
    return {
        "images": images,
        **totals,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "atom_exact_rate": totals["atom_exact"] / max(images, 1),
        "skeleton_exact_rate": totals["skeleton_exact"] / max(images, 1),
        "typed_exact_rate": totals["typed_exact"] / max(images, 1),
    }


@torch.inference_mode()
def evaluate(
    model,
    vision,
    atom,
    samples: list[dict],
    p4_records: list[dict],
    operations: dict[str, str],
    *,
    threshold: float,
    device: torch.device,
) -> dict:
    collator = G3StructuralCollator(
        maximum_atoms=MAXIMUM_ATOMS,
        maximum_bonds=p18e0.BOND_CAPACITY,
        image_size=768,
    )
    rows_by_route = {"gt_atom": [], "p4_oracle_atom": []}
    prediction_digest = hashlib.sha256()
    for start in range(0, len(samples), BATCH_SIZE):
        current_samples = samples[start : start + BATCH_SIZE]
        current_records = p4_records[start : start + BATCH_SIZE]
        batch = collator(current_samples)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feature_map = vision.forward_stroke_resolution(pixels)
            atom_output = atom(feature_map)
        for route in rows_by_route:
            anchors, mask, target_ids = _route_anchors(
                batch, current_records, route=route, device=device
            )
            atom_features, atom_scores = _sample_atom_features(
                atom_output, anchors, mask
            )
            pairs, sample_indices = enumerate_unordered_atom_pairs(mask)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(
                    feature_map,
                    atom_features,
                    anchors,
                    atom_scores,
                    pairs,
                    sample_indices,
                )
            probabilities = output["presence_logits"].sigmoid().float().cpu()
            chemical = output["chemical_logits"].argmax(-1).cpu()
            prediction_digest.update(route.encode())
            prediction_digest.update(probabilities.numpy().tobytes())
            prediction_digest.update(chemical.numpy().tobytes())
            for batch_index, sample in enumerate(current_samples):
                pair_rows = torch.where(sample_indices == batch_index)[0]
                target_mask = batch["bond_mask"][batch_index]
                metrics = _sample_metrics(
                    probabilities=probabilities[pair_rows.cpu()],
                    chemical=chemical[pair_rows.cpu()],
                    pairs=pairs[pair_rows].cpu(),
                    target_ids=target_ids[batch_index].cpu(),
                    target_atom_count=int(batch["atom_mask"][batch_index].sum()),
                    target_refs=batch["bond_atom_refs"][batch_index][target_mask],
                    target_types=batch["bond_types"][batch_index][target_mask],
                    threshold=threshold,
                )
                key = str(sample["key"])
                rows_by_route[route].append(
                    {
                        "sample_key": key,
                        "variant": str(sample["render"]["variant"]),
                        "stress_operation": operations.get(key),
                        **metrics,
                    }
                )

    result = {}
    for route, rows in rows_by_route.items():
        result[route] = {
            "overall": _summarize(rows),
            "by_variant": {
                variant: _summarize(
                    [row for row in rows if row["variant"] == variant]
                )
                for variant in p18e0.VARIANTS
            },
            "stress_by_operation": {
                operation: _summarize(
                    [row for row in rows if row["stress_operation"] == operation]
                )
                for operation in sorted(set(operations.values()))
            },
            "rows": rows,
        }
    return {"routes": result, "prediction_sha256": prediction_digest.hexdigest()}


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E0 output is not empty: {OUTPUT}")
    for path, expected in (
        (DIRECT_CHECKPOINT, DIRECT_CHECKPOINT_SHA256),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256),
        (VISION_CHECKPOINT, VISION_CHECKPOINT_SHA256),
        (E0_CACHE, E0_CACHE_SHA256),
        (E0_MANIFEST, E0_MANIFEST_SHA256),
        (E0A_CACHE, E0A_CACHE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E0 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E0 requires CUDA BF16")
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

    source_payload = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    graph_checkpoint = Path(source_payload["training_args"]["grounded_graph_checkpoint"])
    if graph_checkpoint.resolve() != VISION_CHECKPOINT.resolve():
        raise ValueError("P19-E0 source is bound to another vision checkpoint")
    vision, atom, bond, set_head, _ = load_components(
        str(SOURCE_CHECKPOINT), str(VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    direct_payload = torch.load(
        DIRECT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    model = DirectAtomPairBond(
        DirectAtomPairBondConfig(**direct_payload["config"])
    ).to(device)
    model.load_state_dict(direct_payload["model"], strict=True)
    model.eval().requires_grad_(False)
    threshold = float(direct_payload["validation"]["threshold"])
    if threshold != 0.85:
        raise ValueError("P19-E0 expected the fixed v0.10.3 threshold 0.85")
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "direct": module_state_sha256(model),
    }

    rows = p18e0._jsonl(E0_MANIFEST)
    manifest = p18e0._manifest_records(rows)
    samples = list(
        p1a.iter_manifest_samples(
            manifest, accepted_split=p18e0.freshdev.manifest_protocol.EXPECTED_SPLIT
        )
    )
    e0 = torch.load(E0_CACHE, map_location="cpu", weights_only=False)
    p4_records = e0["records"]
    stress = torch.load(E0A_CACHE, map_location="cpu", weights_only=False)
    operations = {
        str(record["sample_key"]): str(record["stress_operation"])
        for record in stress["records"]
    }
    if [str(sample["key"]) for sample in samples] != [
        str(record["sample_key"]) for record in p4_records
    ]:
        raise RuntimeError("P19-E0 sample/cache order differs")
    result = evaluate(
        model,
        vision,
        atom,
        samples,
        p4_records,
        operations,
        threshold=threshold,
        device=device,
    )
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "direct": module_state_sha256(model),
    }
    gt = result["routes"]["gt_atom"]
    p4 = result["routes"]["p4_oracle_atom"]
    checks = {
        "gt_atom_skeleton_exact_rate_min80": (
            gt["overall"]["skeleton_exact_rate"]
            >= GATES["gt_atom_skeleton_exact_rate_min"]
        ),
        "gt_atom_stress_skeleton_exact_rate_min60": (
            gt["by_variant"]["stress"]["skeleton_exact_rate"]
            >= GATES["gt_atom_stress_skeleton_exact_rate_min"]
        ),
        "gt_atom_edge_recall_min97": (
            gt["overall"]["edge_recall"] >= GATES["gt_atom_edge_recall_min"]
        ),
        "p4_atom_coverage_matches_e0_608": (
            p4["overall"]["atom_exact"] == GATES["p4_atom_coverage_exact"]
        ),
        "all_frozen_states_unchanged": states_before == states_after,
    }
    decision = (
        "existing_direct_atom_pair_reuse_go"
        if all(checks.values())
        else "existing_direct_atom_pair_reuse_no_go"
    )
    route_rows = {
        route: value.pop("rows") for route, value in result["routes"].items()
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "rows.json", route_rows)
    report = {
        "schema_version": "g3-p19e0-direct-atom-pair-oracle-ladder-0.1",
        "decision": decision,
        "checks": checks,
        "gates": GATES,
        "threshold": threshold,
        "routes": result["routes"],
        "prediction_sha256": result["prediction_sha256"],
        "sources": {
            "direct_checkpoint": str(DIRECT_CHECKPOINT.resolve()),
            "direct_checkpoint_sha256": DIRECT_CHECKPOINT_SHA256,
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "vision_checkpoint": str(VISION_CHECKPOINT.resolve()),
            "vision_checkpoint_sha256": VISION_CHECKPOINT_SHA256,
            "fresh640_cache_sha256": E0_CACHE_SHA256,
            "stress_operation_cache_sha256": E0A_CACHE_SHA256,
        },
        "training_or_calibration": False,
        "gt_atom_is_oracle_audit_only": True,
        "p4_selection_uses_gt_after_target_free_candidate_forward": True,
        "dense_bond_forward": False,
        "incidence_forward": False,
        "sealed_test_read": False,
        "confirm_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
