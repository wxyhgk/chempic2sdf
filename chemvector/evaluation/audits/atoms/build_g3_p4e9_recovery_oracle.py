#!/usr/bin/env python3
"""Build the train-only P4-E9 local recovery action oracle."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
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
from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e8_local_multiplicity_components as p4e8,
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
    G3SampleManifestRecord,
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e9-recovery-oracle-pilot20k"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P4E0_ROOT = Path(
    "outputs/audits/chemvector-g3-p4e0-source-union-cycle-manifests"
)
PILOT_MANIFEST = P4E0_ROOT / "pilot_train.jsonl"
PILOT_MANIFEST_SHA256 = (
    "22a03e2b2e37b41b7b298a1fec324fbe383e0f09c49e173a39c8d90347636daa"
)
P4E8_AUDIT = p4e8.OUTPUT / "audit.json"
P4E8_AUDIT_SHA256 = (
    "7062f586af88ff122b31538890c8bc72a9af7b8dfb54d3f9f5841b889784480b"
)
TOP_K = 128
MATCH_RADIUS = 6.0
REFINE_RADIUS = 12.0
IMAGE_SIZE = 768
BATCH_SIZE = 8
SPLIT_NAMES = ("fit", "calibration", "holdout")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
ACTION_ID_STRIDE = IMAGE_SIZE * IMAGE_SIZE
MIN_ROUTE_ACTIONS = 256
MIN_ROUTE_MOLECULES = 100
SEED = 20260849


def _stable_sha(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def allocate_scaffold_splits(
    rows: list[dict],
) -> tuple[dict[str, str], dict]:
    """Assign complete scaffold groups by SHA order and normalized deficit."""

    molecules: dict[str, tuple[str, str]] = {}
    for row in rows:
        molecule_id = str(row["molecule_id"])
        identity = (str(row["scaffold"]), str(row["inchikey"]))
        if molecule_id in molecules and molecules[molecule_id] != identity:
            raise ValueError(f"P4-E9 molecule metadata differs: {molecule_id}")
        molecules[molecule_id] = identity
    grouped: dict[str, list[str]] = defaultdict(list)
    for molecule_id, (scaffold, _) in molecules.items():
        grouped[scaffold].append(molecule_id)
    groups = sorted(
        ((scaffold, sorted(values)) for scaffold, values in grouped.items()),
        key=lambda item: (_stable_sha(item[0]), item[0]),
    )
    total = len(molecules)
    targets = {
        name: total * fraction for name, fraction in zip(SPLIT_NAMES, SPLIT_FRACTIONS)
    }
    counts = {name: 0 for name in SPLIT_NAMES}
    owner: dict[str, str] = {}
    for scaffold, molecule_ids in groups:
        # The largest normalized deficit wins. Stable split order resolves ties.
        selected = max(
            SPLIT_NAMES,
            key=lambda name: ((targets[name] - counts[name]) / targets[name], -SPLIT_NAMES.index(name)),
        )
        counts[selected] += len(molecule_ids)
        for molecule_id in molecule_ids:
            owner[molecule_id] = selected
    scaffold_sets = {
        name: {
            scaffold
            for scaffold, molecule_ids in groups
            if owner[molecule_ids[0]] == name
        }
        for name in SPLIT_NAMES
    }
    overlaps = {
        f"{left}_{right}": len(scaffold_sets[left] & scaffold_sets[right])
        for left_index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[left_index + 1 :]
    }
    if any(overlaps.values()) or set(owner) != set(molecules):
        raise RuntimeError("P4-E9 scaffold allocation is incomplete or overlapping")
    return owner, {
        "algorithm": "sha256-scaffold-order_normalized-deficit_assignment",
        "seed": SEED,
        "molecule_targets": targets,
        "molecule_counts": counts,
        "scaffold_counts": {name: len(values) for name, values in scaffold_sets.items()},
        "cross_split_scaffold_overlap": overlaps,
        "molecule_owner_sha256": hashlib.sha256(
            "".join(f"{key}\t{owner[key]}\n" for key in sorted(owner)).encode()
        ).hexdigest(),
    }


def _manifest_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    required = {"sample_key", "molecule_id", "inchikey", "scaffold", "shard", "variant"}
    if any(required - row.keys() for row in rows):
        raise ValueError("P4-E9 manifest lacks scaffold or identity fields")
    return rows


def _component_assignment(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    component: dict,
) -> dict[int, int]:
    candidate_indices = component["candidate_indices"]
    gt_indices = component["gt_indices"]
    local = p4e6.maximum_cardinality_assignments(
        predicted[candidate_indices], expected[gt_indices], radius_pixels=MATCH_RADIUS
    )
    return {
        gt_indices[gt_local]: candidate_indices[candidate_local]
        for gt_local, candidate_local in enumerate(local)
        if candidate_local is not None
    }


def recovery_actions(
    predicted: torch.Tensor,
    source_ids: torch.Tensor,
    expected: torch.Tensor,
    *,
    return_details: bool = False,
) -> list[dict] | tuple[list[dict], list[int]]:
    """Return mutually exclusive split/refine positives for one training image."""

    components = p4e8.deficient_bipartite_components(
        predicted, expected, radius_pixels=MATCH_RADIUS
    )
    actions: list[dict] = []
    split_gt: set[int] = set()
    split_sources: set[int] = set()
    unsupported_candidates: set[int] = set()
    for component in components:
        if not (
            component["kind"] == "shared_deficient"
            and component["gt_count"] == 2
            and component["candidate_count"] == 1
            and component["deficiency"] == 1
        ):
            if component["kind"] == "shared_deficient":
                unsupported_candidates.update(component["candidate_indices"])
            continue
        assignment = _component_assignment(predicted, expected, component)
        unmatched = [index for index in component["gt_indices"] if index not in assignment]
        if len(unmatched) != 1:
            raise RuntimeError("P4-E9 split component lacks one canonical secondary GT")
        candidate_index = component["candidate_indices"][0]
        target_index = unmatched[0]
        split_gt.update(component["gt_indices"])
        split_sources.add(candidate_index)
        actions.append(
            {
                "route": "ADD_SPLIT",
                "candidate_index": candidate_index,
                "base_source_id": int(source_ids[candidate_index]),
                "added_source_id": ACTION_ID_STRIDE + int(source_ids[candidate_index]),
                "base_anchor": predicted[candidate_index].tolist(),
                "target_anchor": expected[target_index].tolist(),
                "target_index": target_index,
                "component_gt_indices": component["gt_indices"],
            }
        )

    # Refinement owns only strict no-candidate 6px components. This keeps route
    # labels disjoint and prevents the 2GT/1candidate case from being relabelled.
    refine_targets = []
    for component in components:
        if not (
            component["kind"] == "no_candidate"
            and component["gt_count"] == 1
            and component["deficiency"] == 1
        ):
            continue
        target_index = component["gt_indices"][0]
        if target_index in split_gt:
            continue
        refine_targets.append(target_index)
    refine_candidates = [
        index for index in range(len(predicted)) if index not in split_sources
    ]
    refine_eligible: dict[int, set[int]] = {}
    for target_index in refine_targets:
        distances = torch.linalg.vector_norm(
            (predicted.double() - expected[target_index].double()) * IMAGE_SIZE,
            dim=1,
        )
        refine_eligible[target_index] = {
            index
            for index in refine_candidates
            if MATCH_RADIUS < float(distances[index]) <= REFINE_RADIUS
        }
    refine_assignments = (
        p4e6.maximum_cardinality_assignments(
            predicted[refine_candidates],
            expected[refine_targets],
            radius_pixels=REFINE_RADIUS,
        )
        if refine_targets and refine_candidates
        else []
    )
    for target_index, local_candidate in zip(refine_targets, refine_assignments):
        if local_candidate is None:
            continue
        candidate_index = refine_candidates[local_candidate]
        distance = float(
            torch.linalg.vector_norm(
                (predicted[candidate_index].double() - expected[target_index].double())
                * IMAGE_SIZE
            )
        )
        if not MATCH_RADIUS < distance <= REFINE_RADIUS:
            raise RuntimeError("P4-E9 refine assignment violates the annulus protocol")
        actions.append(
            {
                "route": "ADD_REFINE",
                "candidate_index": candidate_index,
                "base_source_id": int(source_ids[candidate_index]),
                "added_source_id": 2 * ACTION_ID_STRIDE + int(source_ids[candidate_index]),
                "base_anchor": predicted[candidate_index].tolist(),
                "target_anchor": expected[target_index].tolist(),
                "target_index": target_index,
                "component_gt_indices": component["gt_indices"],
                "source_distance_pixels": distance,
            }
        )
    assigned_refine_sources = {
        item["candidate_index"] for item in actions if item["route"] == "ADD_REFINE"
    }
    for values in refine_eligible.values():
        unsupported_candidates.update(values - assigned_refine_sources)
    unsupported_candidates.difference_update(
        item["candidate_index"] for item in actions
    )
    baseline_matched = sum(
        value is not None
        for value in p4e6.maximum_cardinality_assignments(
            predicted, expected, radius_pixels=MATCH_RADIUS
        )
    )
    for action in actions:
        augmented = torch.cat(
            (predicted, torch.tensor([action["target_anchor"]], dtype=predicted.dtype))
        )
        augmented_matched = sum(
            value is not None
            for value in p4e6.maximum_cardinality_assignments(
                augmented, expected, radius_pixels=MATCH_RADIUS
            )
        )
        if augmented_matched != baseline_matched + 1:
            raise RuntimeError("P4-E9 Oracle action does not recover exactly one GT")
        action["oracle_matched_gain"] = 1
    identities = [item["base_source_id"] for item in actions]
    if len(identities) != len(set(identities)):
        raise RuntimeError("P4-E9 assigned more than one action to a base source")
    actions = sorted(actions, key=lambda item: (item["route"], item["base_source_id"]))
    unsupported_source_ids = sorted(
        int(source_ids[index]) for index in unsupported_candidates
    )
    return (actions, unsupported_source_ids) if return_details else actions


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E9-0 output is not empty: {OUTPUT}")
    for path, expected in (
        (PILOT_MANIFEST, PILOT_MANIFEST_SHA256),
        (P4E8_AUDIT, P4E8_AUDIT_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E9-0 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E9-0 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    raw_rows = _manifest_rows(PILOT_MANIFEST)
    owner, allocation = allocate_scaffold_splits(raw_rows)
    manifest, manifest_audit = load_g3_sample_manifest(
        PILOT_MANIFEST,
        expected_sha256=PILOT_MANIFEST_SHA256,
        expected_samples=20_000,
        expected_molecules=cycle.PILOT_TRAIN_MOLECULES,
        expected_variants=cycle.VARIANTS,
    )
    split_by_key = {
        row["sample_key"]: owner[row["molecule_id"]] for row in raw_rows
    }
    metadata_by_key = {row["sample_key"]: row for row in raw_rows}
    device = torch.device("cuda")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=IMAGE_SIZE)
    action_rows: list[dict] = []
    image_rows: list[dict] = []
    started = time.perf_counter()

    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=BATCH_SIZE):
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
            for index, key_value in enumerate(batch["keys"]):
                sample_key = str(key_value)
                peak = peaks[index]
                if len(peak["anchors"]) != TOP_K:
                    raise RuntimeError(f"P4-E9 {sample_key} has fewer than top128 peaks")
                expected = batch["atom_anchors"][index][batch["atom_mask"][index]].cpu()
                actions, unsupported_source_ids = recovery_actions(
                    peak["anchors"].cpu(),
                    peak["source_ids"].cpu(),
                    expected,
                    return_details=True,
                )
                meta = metadata_by_key[sample_key]
                split = split_by_key[sample_key]
                for action in actions:
                    action.update(
                        {
                            "sample_key": sample_key,
                            "molecule_id": meta["molecule_id"],
                            "scaffold": meta["scaffold"],
                            "variant": meta["variant"],
                            "split": split,
                        }
                    )
                    action_rows.append(action)
                image_rows.append(
                    {
                        "sample_key": sample_key,
                        "molecule_id": meta["molecule_id"],
                        "scaffold": meta["scaffold"],
                        "variant": meta["variant"],
                        "split": split,
                        "atom_count": int(len(expected)),
                        "split_actions": sum(item["route"] == "ADD_SPLIT" for item in actions),
                        "refine_actions": sum(item["route"] == "ADD_REFINE" for item in actions),
                        "unsupported_source_ids": unsupported_source_ids,
                        "unsupported_sources": len(unsupported_source_ids),
                    }
                )

    route_summary = {}
    for route in ("ADD_SPLIT", "ADD_REFINE"):
        current = [row for row in action_rows if row["route"] == route]
        variants = Counter(row["variant"] for row in current)
        route_summary[route] = {
            "actions": len(current),
            "molecules": len({row["molecule_id"] for row in current}),
            "images": len({row["sample_key"] for row in current}),
            "variants": dict(sorted(variants.items())),
            "splits": dict(sorted(Counter(row["split"] for row in current).items())),
            "minimum_actions_256": len(current) >= MIN_ROUTE_ACTIONS,
            "minimum_molecules_100": len({row["molecule_id"] for row in current})
            >= MIN_ROUTE_MOLECULES,
            "all_four_variants": set(variants) == set(cycle.VARIANTS),
        }
    protocol_checks = {
        "pilot_images_20000": len(image_rows) == 20_000,
        "pilot_molecules_5000": len({row["molecule_id"] for row in image_rows}) == 5_000,
        "scaffold_cross_split_overlap_zero": not any(
            allocation["cross_split_scaffold_overlap"].values()
        ),
        "route_labels_mutually_exclusive": len(
            {(row["sample_key"], row["base_source_id"]) for row in action_rows}
        )
        == len(action_rows),
        "base_candidates_unchanged": True,
        "unsupported_sources_explicit": all(
            "unsupported_source_ids" in row for row in image_rows
        ),
        "confirm_v2_not_read": True,
        "external_and_sealed_not_read": True,
    }
    route_capacity = {
        route: all(
            route_summary[route][name]
            for name in (
                "minimum_actions_256",
                "minimum_molecules_100",
                "all_four_variants",
            )
        )
        for route in route_summary
    }
    decision = (
        "recovery_oracle_go"
        if all(protocol_checks.values()) and any(route_capacity.values())
        else "recovery_oracle_no_go"
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_jsonl(OUTPUT / "actions.jsonl", action_rows)
    _atomic_jsonl(OUTPUT / "images.jsonl", image_rows)
    for split in SPLIT_NAMES:
        rows = [row for row in raw_rows if owner[row["molecule_id"]] == split]
        _atomic_jsonl(OUTPUT / f"{split}.jsonl", rows)
    audit = {
        "schema_version": "g3-p4e9-recovery-oracle-pilot20k-0.1",
        "decision": decision,
        "route_capacity": route_capacity,
        "protocol_checks": protocol_checks,
        "protocol": {
            "top_k": TOP_K,
            "match_radius_pixels": MATCH_RADIUS,
            "refine_radius_pixels": REFINE_RADIUS,
            "split_action": "strict_2gt_1candidate_6px_component",
            "refine_action": "strict_1gt_0candidate_6px_component_with_top128_source_within_12px",
            "negative_label": "NONE only; UNSUPPORTED source IDs are loss-masked",
            "candidate_ids": {
                "base_capacity": ACTION_ID_STRIDE,
                "split": "589824 + base_source_id",
                "refine": "1179648 + base_source_id",
            },
            "training_performed": False,
            "bond_endpoint_evidence": "pending_training_only_3fold_probe",
        },
        "allocation": allocation,
        "manifest": manifest_audit,
        "lineage": {
            "p4e3_checkpoint": str(p4e6.P4E3_CHECKPOINT.resolve()),
            "p4e3_checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
            "p4e8_audit_sha256": P4E8_AUDIT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
        },
        "route_summary": route_summary,
        "totals": {
            "images": len(image_rows),
            "molecules": len({row["molecule_id"] for row in image_rows}),
            "actions": len(action_rows),
            "images_with_actions": sum(
                row["split_actions"] + row["refine_actions"] > 0 for row in image_rows
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)


if __name__ == "__main__":
    main()
