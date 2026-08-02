#!/usr/bin/env python3
"""Evaluate the frozen P0-I/P0-G/P0-K stack on route-select-100."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import textwrap

import torch

from Data.sdf.build_webdataset_sample import mol_from_chemmol, scaffold_smiles

from chemvector.audit_g3_p0j_frozen_composition import (
    P0G_CHECKPOINT,
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT,
    P0I_CHECKPOINT_SHA256,
    composition_batch_statistics,
    compose_pair_output,
    empty_mode_totals,
    finalize_mode_totals,
    gather_slot_features,
    load_frozen_p0_models,
    merge_totals,
    pair_identity_sha256,
)
from chemvector.g3_hierarchical_graph import (
    build_g3_bond_targets,
)
from chemvector.g3_structural_data import (
    G3SampleManifestRecord,
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    OVERFIT32_MANIFEST_SHA256,
    file_sha256,
    g3_targets_to_device,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
    load_g3_sample_manifest,
)
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _atomic_json,
    _autocast_context,
    cache_frozen_vision_features,
    evaluate_g3_structural_overfit,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


ROUTE_SELECT_MANIFEST = Path(
    "Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl"
)
ROUTE_SELECT_MANIFEST_SHA256 = (
    "a31addd5d18dc153b093a9ec3879872401b6928fda40b100cb2e78c06e72a7a6"
)
P0K_CHECKPOINT = Path(
    "outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment/last.pt"
)
P0K_CHECKPOINT_SHA256 = (
    "a91d5bb1052063c819b3ba71e26987c45d5e7e819281ff26fa7b3bd80f90eac5"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0m-frozen-route-select-100"
)
EXPECTED_IMAGES = 100
EXPECTED_MOLECULES = 25
EXPECTED_VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})


def load_route_select_manifest(
    path: str | Path = ROUTE_SELECT_MANIFEST,
) -> tuple[list[G3SampleManifestRecord], dict]:
    return load_g3_sample_manifest(
        path,
        expected_sha256=ROUTE_SELECT_MANIFEST_SHA256,
        expected_samples=EXPECTED_IMAGES,
        expected_molecules=EXPECTED_MOLECULES,
        expected_variants=EXPECTED_VARIANTS,
    )


def manifest_disjointness_audit(
    route: list[G3SampleManifestRecord],
    overfit: list[G3SampleManifestRecord],
) -> dict:
    fields = {
        "sample_key": lambda record: record.sample_key,
        "molecule_id": lambda record: record.molecule_id,
        "inchikey": lambda record: record.inchikey,
    }
    overlap = {
        name: sorted({getter(item) for item in route} & {getter(item) for item in overfit})
        for name, getter in fields.items()
    }
    return {
        "route_samples": len(route),
        "route_molecules": len({item.molecule_id for item in route}),
        "overfit_samples": len(overfit),
        "overfit_molecules": len({item.molecule_id for item in overfit}),
        "overlap": overlap,
        "all_three_zero_overlap": all(not values for values in overlap.values()),
    }


def scaffold_overlap_audit(route_samples: list[dict], overfit_samples: list[dict]) -> dict:
    def scaffolds(samples: list[dict]) -> set[str]:
        by_molecule = {}
        for sample in samples:
            molecule_id = str(sample["chemmol"]["molecule_id"])
            if molecule_id in by_molecule:
                continue
            mol = mol_from_chemmol(sample["chemmol"])
            smiles = str(sample["chemmol"]["canonical_isomeric_smiles"])
            by_molecule[molecule_id] = scaffold_smiles(mol, smiles)
        return set(by_molecule.values())

    route = scaffolds(route_samples)
    overfit = scaffolds(overfit_samples)
    overlap = sorted(route & overfit)
    return {
        "algorithm": "Bemis-Murcko includeChirality; acyclic keyed by canonical SMILES",
        "route_scaffolds": len(route),
        "overfit_scaffolds": len(overfit),
        "overlap_count": len(overlap),
        "overlap": overlap,
        "hard_gate": False,
        "generator_guaranteed_disjoint": False,
    }


def load_p0k_overlay(p0g, path: Path) -> dict:
    if file_sha256(path) != P0K_CHECKPOINT_SHA256:
        raise ValueError("P0-M P0-K checkpoint SHA differs from the protocol")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("model_type")
        != "g3-predicted-anchor-relation-alignment-p0k-v0.1"
        or payload.get("global_step") != 400
    ):
        raise ValueError("P0-M requires the final P0-K checkpoint")
    if payload.get("source_p0i_checkpoint", {}).get("sha256") != P0I_CHECKPOINT_SHA256:
        raise ValueError("P0-M P0-K lineage has an unexpected P0-I checkpoint")
    if payload.get("source_p0g_checkpoint", {}).get("sha256") != P0G_CHECKPOINT_SHA256:
        raise ValueError("P0-M P0-K lineage has an unexpected P0-G checkpoint")
    state = payload.get("pair_bond")
    if not isinstance(state, dict) or not state:
        raise ValueError("P0-M P0-K checkpoint contains no pair state")
    p0g.pair_bond.load_state_dict(state, strict=True)
    p0g.requires_grad_(False).eval()
    return payload


@torch.no_grad()
def frozen_paired_deploy_forward(
    p0i,
    p0g,
    visual_features: torch.Tensor,
    *,
    precision: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run paired native/same-slot deploy forwards without any target input."""

    with _autocast_context(visual_features.device, precision):
        atoms = p0i(visual_features)
        dense = p0g.atom_head(visual_features)
        same_slot_features = gather_slot_features(
            dense, atoms["atom_slot_indices"], atoms["atom_mask"]
        )
        native = compose_pair_output(
            p0g, visual_features, atoms, atoms["atom_features"]
        )
        same_slot = compose_pair_output(
            p0g, visual_features, atoms, same_slot_features
        )
    for name in ("pair_indices", "sample_indices"):
        if not torch.equal(native[name], same_slot[name]):
            raise RuntimeError("P0-M paired feature routes changed AtomPair identity")
    expected_pairs = sum(
        count * (count - 1) // 2
        for count in atoms["atom_mask"].sum(dim=1).long().tolist()
    )
    pairs = same_slot["pair_indices"]
    if len(pairs) != expected_pairs or bool((pairs[:, 0] >= pairs[:, 1]).any()):
        raise RuntimeError("P0-M deploy route does not enumerate every i<j AtomPair")
    return atoms, native, same_slot


def validate_deploy_forward_contract() -> dict[str, int | bool]:
    signature = inspect.signature(frozen_paired_deploy_forward)
    forbidden = {"target", "targets", "oracle", "gt"}
    parameter_names = set(signature.parameters)
    source = textwrap.dedent(inspect.getsource(frozen_paired_deploy_forward))
    tree = ast.parse(source)
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    forbidden_parameters = parameter_names & forbidden
    forbidden_references = referenced_names & forbidden
    if forbidden_parameters or forbidden_references:
        raise ValueError("P0-M deploy forward references GT or target inputs")
    return {
        "accepts_targets": False,
        "forbidden_parameter_count": len(forbidden_parameters),
        "forbidden_reference_count": len(forbidden_references),
        "compose_pair_forward_calls": sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compose_pair_output"
            for node in ast.walk(tree)
        ),
    }


def evaluator_static_contract() -> dict[str, int | bool]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    optimizer_references = sum(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "optim"
        for node in ast.walk(tree)
    )
    backward_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
        for node in ast.walk(tree)
    )
    if optimizer_references or backward_calls:
        raise ValueError("P0-M evaluator contains optimizer or backward code")
    forbidden_paths = (
        "dev_route_" + "confirm_400.jsonl",
        "fresh_" + "eval_12k",
        "public_" + "ocsr_v1",
        "MR-" + "TADF_SDF",
    )
    forbidden_path_references = sum(path in source for path in forbidden_paths)
    if forbidden_path_references:
        raise ValueError("P0-M evaluator references forbidden data paths")
    return {
        "optimizer_references": optimizer_references,
        "backward_calls": backward_calls,
        "updates": 0,
        "forbidden_data_path_references": forbidden_path_references,
    }


@torch.no_grad()
def evaluate_paired_composition(
    p0i,
    p0g,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    sample_keys: list[str],
    molecule_ids: list[str],
    variants: list[str],
    occupancy: dict,
    *,
    batch_size: int,
    precision: str,
) -> dict[str, dict]:
    totals = {
        "native": empty_mode_totals(),
        "same_slot": empty_mode_totals(),
    }
    identity_masks: list[torch.Tensor] = []
    sample_totals = {"native": {}, "same_slot": {}}
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        atoms, native, same_slot = frozen_paired_deploy_forward(
            p0i, p0g, current_features, precision=precision
        )
        normalized = build_g3_bond_targets(current_targets)
        for local_index in range(len(current_features)):
            global_index = start + local_index
            sample_atoms = {
                "atom_mask": atoms["atom_mask"][local_index : local_index + 1],
                "atom_anchors": atoms["atom_anchors"][local_index : local_index + 1],
            }
            sample_targets = {
                name: value[local_index : local_index + 1]
                for name, value in current_targets.items()
            }
            sample_normalized = {
                name: value[local_index : local_index + 1]
                for name, value in normalized.items()
            }
            predicted_atoms = int(sample_atoms["atom_mask"].sum())
            target_atoms = int(sample_targets["atom_mask"].sum())
            for name, output in (("native", native), ("same_slot", same_slot)):
                rows = torch.where(output["sample_indices"] == local_index)[0]
                sample_output = {
                    "pair_indices": output["pair_indices"][rows],
                    "sample_indices": torch.zeros_like(output["sample_indices"][rows]),
                    "presence_logits": output["presence_logits"][rows],
                    "chemical_logits": output["chemical_logits"][rows],
                    "bond_count_logits": output["bond_count_logits"][
                        local_index : local_index + 1
                    ],
                }
                current = composition_batch_statistics(
                    sample_output, sample_atoms, sample_targets, sample_normalized
                )
                current["atom_count_exact_images"] = float(
                    predicted_atoms == target_atoms
                )
                current["atom_count_absolute_error"] = float(
                    abs(predicted_atoms - target_atoms)
                )
                merge_totals(totals[name], current)
                sample_totals[name][global_index] = current
        identity_masks.extend(mask.detach().cpu() for mask in atoms["atom_mask"])
    identity = pair_identity_sha256(sample_keys, identity_masks)

    def finalize_extended(raw: dict[str, float]) -> dict:
        value = finalize_mode_totals(raw)
        images = max(raw.get("images", 0.0), 1.0)
        value["positive_type_accuracy_representable"] = raw.get(
            "positive_type_correct", 0.0
        ) / max(raw.get("representable_bonds", 0.0), 1.0)
        value["atom_count_exact_images"] = int(
            raw.get("atom_count_exact_images", 0.0)
        )
        value["atom_count_mae"] = raw.get("atom_count_absolute_error", 0.0) / images
        return value

    sample_maximum_occupancy = {
        sample["sample_key"]: str(sample["maximum_occupancy"])
        for sample in occupancy["samples"]
    }
    result = {}
    for name in ("native", "same_slot"):
        result[name] = finalize_extended(totals[name])
        result[name]["feature_mode"] = name
        result[name]["pair_identity_sha256"] = identity
        result[name]["per_variant"] = {}
        for variant in sorted(set(variants)):
            aggregate = empty_mode_totals()
            for index, value in sample_totals[name].items():
                if variants[index] == variant:
                    merge_totals(aggregate, value)
            result[name]["per_variant"][variant] = finalize_extended(aggregate)
        result[name]["per_occupancy_bucket"] = {}
        for bucket in ("1", "2", "3"):
            aggregate = empty_mode_totals()
            for index, value in sample_totals[name].items():
                if bucket == sample_maximum_occupancy[sample_keys[index]]:
                    merge_totals(aggregate, value)
            result[name]["per_occupancy_bucket"][bucket] = finalize_extended(
                aggregate
            )
        per_sample = []
        for index, raw in sample_totals[name].items():
            value = finalize_extended(raw)
            per_sample.append(
                {
                    "sample_key": sample_keys[index],
                    "molecule_id": molecule_ids[index],
                    "variant": variants[index],
                    "atom_count_exact": bool(value["atom_count_exact_images"]),
                    "atom_set_exact": bool(value["atom_set_exact_images"]),
                    "gt_skeleton_exact": bool(
                        value["gt_bond_count"]["skeleton_exact_images"]
                    ),
                    "gt_typed_graph_exact": bool(
                        value["gt_bond_count"]["typed_graph_exact_images"]
                    ),
                    "deploy_count_exact": bool(
                        value["deploy_bond_count"]["count_exact_images"]
                    ),
                    "deploy_skeleton_exact": bool(
                        value["deploy_bond_count"]["skeleton_exact_images"]
                    ),
                    "deploy_typed_graph_exact": bool(
                        value["deploy_bond_count"]["typed_graph_exact_images"]
                    ),
                }
            )
        result[name]["per_sample_exact"] = per_sample
        molecule_groups: dict[str, list[dict]] = {}
        for item in per_sample:
            molecule_groups.setdefault(item["molecule_id"], []).append(item)
        exact_fields = (
            "atom_count_exact",
            "atom_set_exact",
            "gt_skeleton_exact",
            "gt_typed_graph_exact",
            "deploy_count_exact",
            "deploy_skeleton_exact",
            "deploy_typed_graph_exact",
        )
        result[name]["molecule_four_variant_exact"] = {
            "molecules": len(molecule_groups),
            "complete_four_variant_molecules": sum(
                len(items) == 4 and {item["variant"] for item in items} == set(EXPECTED_VARIANTS)
                for items in molecule_groups.values()
            ),
            **{
                field: sum(
                    len(items) == 4 and all(item[field] for item in items)
                    for items in molecule_groups.values()
                )
                for field in exact_fields
            },
        }
    return result


def canonical_occupancy_audit(
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    sample_keys: list[str],
    *,
    slots: int,
    height: int,
    width: int,
) -> dict:
    if target_anchors.shape[:2] != target_mask.shape or len(sample_keys) != len(
        target_anchors
    ):
        raise ValueError("P0-M occupancy inputs have incompatible shapes")
    cell_histogram: dict[int, int] = {}
    bucket_images: dict[int, set[str]] = {}
    slot_counts: dict[int, int] = {}
    samples = []
    overflow_cells = overflow_atoms = overflow_total_atoms = 0
    overflow_sample_keys = set()
    identical_cells = identical_atoms = 0
    identical_sample_keys = set()
    identical_details = []
    for batch_index, sample_key in enumerate(sample_keys):
        cells: dict[tuple[int, int], list[int]] = {}
        coordinate_groups: dict[tuple[float, float], list[int]] = {}
        for target_index in torch.where(target_mask[batch_index].bool())[0].tolist():
            anchor = target_anchors[batch_index, target_index].float()
            coordinate = tuple(float(value) for value in anchor.tolist())
            coordinate_groups.setdefault(coordinate, []).append(target_index)
            x_index = min(width - 1, max(0, int(float(anchor[0]) * width)))
            y_index = min(height - 1, max(0, int(float(anchor[1]) * height)))
            cells.setdefault((y_index, x_index), []).append(target_index)
        for coordinate, indices in coordinate_groups.items():
            if len(indices) < 2:
                continue
            identical_cells += 1
            identical_atoms += len(indices)
            identical_sample_keys.add(sample_key)
            identical_details.append(
                {
                    "sample_key": sample_key,
                    "normalized_anchor": list(coordinate),
                    "cell": {
                        "y": min(height - 1, max(0, int(coordinate[1] * height))),
                        "x": min(width - 1, max(0, int(coordinate[0] * width))),
                    },
                    "target_indices": indices,
                }
            )
        identities = []
        sample_overflow = 0
        occupancy_buckets = set()
        for (y_index, x_index), indices in sorted(cells.items()):
            indices.sort(
                key=lambda index: tuple(
                    float(value)
                    for value in target_anchors[batch_index, index].tolist()
                )
            )
            occupancy = len(indices)
            cell_histogram[occupancy] = cell_histogram.get(occupancy, 0) + 1
            bucket_images.setdefault(occupancy, set()).add(sample_key)
            occupancy_buckets.add(str(occupancy) if occupancy <= slots else "overflow")
            if occupancy > slots:
                overflow_cells += 1
                overflow_atoms += occupancy - slots
                overflow_total_atoms += occupancy
                sample_overflow += 1
                overflow_sample_keys.add(sample_key)
            for slot, target_index in enumerate(indices):
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
                identities.append(
                    {
                        "target_index": target_index,
                        "slot": slot,
                        "y": y_index,
                        "x": x_index,
                        "supported": slot < slots,
                    }
                )
        samples.append(
            {
                "sample_key": sample_key,
                "target_atoms": len(identities),
                "occupied_cells": len(cells),
                "overflow_cells": sample_overflow,
                "occupancy_buckets": sorted(occupancy_buckets),
                "maximum_occupancy": max(
                    (len(indices) for indices in cells.values()), default=0
                ),
                "canonical_identities": sorted(
                    identities, key=lambda item: item["target_index"]
                ),
            }
        )
    return {
        "grid": {"slots": slots, "height": height, "width": width},
        "metric_bucket_definition": "maximum GT cell occupancy per image",
        "cell_occupancy": {
            str(occupancy): {
                "cells": cell_histogram.get(occupancy, 0),
                "atoms": occupancy * cell_histogram.get(occupancy, 0),
                "images": len(bucket_images.get(occupancy, set())),
            }
            for occupancy in (1, 2, 3)
        }
        | {
            "overflow_gt_3": {
                "cells": overflow_cells,
                "atoms": overflow_total_atoms,
                "atoms_above_capacity": overflow_atoms,
                "images": len(overflow_sample_keys),
                "sample_keys": sorted(overflow_sample_keys),
            },
            "maximum": max(cell_histogram, default=0),
        },
        "overflow_atoms": overflow_atoms,
        "identical_normalized_anchor_collision": {
            "groups": identical_cells,
            "atoms": identical_atoms,
            "images": len(identical_sample_keys),
            "sample_keys": sorted(identical_sample_keys),
            "details": identical_details,
        },
        "canonical_slot_atom_counts": {
            str(slot): count for slot, count in sorted(slot_counts.items())
        },
        "samples": samples,
    }


def p0m_gate(
    *,
    protocol_checks: dict[str, bool],
    same_slot: dict,
    native: dict,
    oracle: dict,
    occupancy: dict,
) -> dict:
    oracle_gt = oracle["gt_count"]
    checks = {
        "oracle_gt_presence_f1": oracle_gt["atom_pair_presence_f1"] >= 0.95,
        "oracle_positive_type": (
            oracle_gt["atom_pair_positive_type_accuracy"] >= 0.95
        ),
        "same_slot_atom_recall": same_slot["atom_recall"] >= 0.90,
        "same_slot_bond_representable": (
            same_slot["bond_representable_recall"] >= 0.90
        ),
        "same_slot_gt_presence_f1": same_slot["gt_bond_count"]["f1"] >= 0.95,
        "same_slot_positive_type": (
            same_slot["positive_type_accuracy_representable"] >= 0.95
        ),
        "same_slot_type_improvement": (
            same_slot["positive_type_accuracy_representable"]
            - native["positive_type_accuracy_representable"]
            >= 0.05
        ),
        "same_slot_skeleton_not_worse": (
            same_slot["gt_bond_count"]["skeleton_exact_images"]
            >= native["gt_bond_count"]["skeleton_exact_images"]
        ),
        "occupancy_no_overflow": (
            occupancy["cell_occupancy"]["overflow_gt_3"]["cells"] == 0
        ),
        "occupancy_no_identical_collision": (
            occupancy["identical_normalized_anchor_collision"]["groups"] == 0
        ),
    }
    protocol_passed = all(protocol_checks.values())
    capability_passed = all(checks.values())
    return {
        "status": "passed" if protocol_passed and capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "protocol_status": "passed" if protocol_passed else "failed",
        "protocol_checks": protocol_checks,
        "capability_checks": checks,
        "confirm400_may_be_read": bool(protocol_passed and capability_passed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", default=str(ROUTE_SELECT_MANIFEST))
    parser.add_argument("--overfit-manifest", default=str(OVERFIT32_MANIFEST))
    parser.add_argument("--p0i-checkpoint", default=str(P0I_CHECKPOINT))
    parser.add_argument("--p0g-checkpoint", default=str(P0G_CHECKPOINT))
    parser.add_argument("--p0k-checkpoint", default=str(P0K_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    paths = {
        "sample_manifest": ROUTE_SELECT_MANIFEST,
        "overfit_manifest": OVERFIT32_MANIFEST,
        "p0i_checkpoint": P0I_CHECKPOINT,
        "p0g_checkpoint": P0G_CHECKPOINT,
        "p0k_checkpoint": P0K_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT_DIR,
    }
    for name, expected in paths.items():
        if Path(getattr(args, name)).resolve() != expected.resolve():
            raise ValueError(f"P0-M requires fixed {name}: {expected}")
    if args.device != "cuda" or args.precision != "bf16":
        raise ValueError("P0-M requires fixed CUDA BF16 evaluation")
    if args.batch_size != 4 or args.vision_batch_size != 4:
        raise ValueError("P0-M requires fixed batch sizes of 4")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("P0-M CUDA evaluation requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P0-M output directory is not empty: {output_dir}")
    device = torch.device("cuda")

    deploy_contract = validate_deploy_forward_contract()
    static_contract = evaluator_static_contract()
    route, route_audit = load_route_select_manifest(args.sample_manifest)
    overfit, overfit_audit = load_g3_overfit32_manifest(args.overfit_manifest)
    disjointness = manifest_disjointness_audit(route, overfit)
    records = load_g3_manifest_samples(route)
    overfit_records = load_g3_manifest_samples(overfit)
    scaffold = scaffold_overlap_audit(records, overfit_records)
    p0i, p0g, p0i_payload, _, config = load_frozen_p0_models(
        Path(args.p0i_checkpoint), Path(args.p0g_checkpoint), device
    )
    p0k_payload = load_p0k_overlay(p0g, Path(args.p0k_checkpoint))
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        p0i_payload["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-M vision SHA differs from the protocol")
    for model in (vision, p0i, p0g):
        model.requires_grad_(False).eval()
    initial_states = {
        "vision": module_state_sha256(vision),
        "p0i": module_state_sha256(p0i),
        "p0g_with_p0k_overlay": module_state_sha256(p0g),
        "p0k_pair": module_state_sha256(p0g.pair_bond),
    }
    initial_modes = {
        "vision_training": vision.training,
        "p0i_training": p0i.training,
        "p0g_training": p0g.training,
        "p0k_pair_training": p0g.pair_bond.training,
    }

    collator = G3StructuralCollator(
        maximum_atoms=config.maximum_atoms,
        maximum_bonds=config.maximum_bonds,
        image_size=768,
    )
    batch = collator(records)
    sample_keys = list(batch["keys"])
    occupancy = canonical_occupancy_audit(
        batch["atom_anchors"],
        batch["atom_mask"],
        sample_keys,
        slots=config.atom_slots_per_cell,
        height=192,
        width=192,
    )
    features = cache_frozen_vision_features(
        vision,
        batch["pixel_values"],
        device=device,
        batch_size=args.vision_batch_size,
        precision=args.precision,
    )
    targets = g3_targets_to_device(batch, device)
    paired = evaluate_paired_composition(
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        list(batch["molecule_ids"]),
        list(batch["variants"]),
        occupancy,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    oracle = {
        "scope": "gt-atom-oracle-relation-control",
        "gt_count": evaluate_g3_structural_overfit(
            p0g,
            features,
            targets,
            batch_size=args.batch_size,
            precision=args.precision,
            count_mode="gt_count",
            gt_atom_slot_oracle=True,
            direct_gt_atom_pair_bonds=True,
        ),
        "deploy_count": evaluate_g3_structural_overfit(
            p0g,
            features,
            targets,
            batch_size=args.batch_size,
            precision=args.precision,
            count_mode="deploy_count",
            gt_atom_slot_oracle=True,
            direct_gt_atom_pair_bonds=True,
        ),
    }
    final_states = {
        "vision": module_state_sha256(vision),
        "p0i": module_state_sha256(p0i),
        "p0g_with_p0k_overlay": module_state_sha256(p0g),
        "p0k_pair": module_state_sha256(p0g.pair_bond),
    }
    final_modes = {
        "vision_training": vision.training,
        "p0i_training": p0i.training,
        "p0g_training": p0g.training,
        "p0k_pair_training": p0g.pair_bond.training,
    }
    protocol_checks = {
        "route_manifest_sha": route_audit["sha256"] == ROUTE_SELECT_MANIFEST_SHA256,
        "overfit_manifest_sha": overfit_audit["sha256"] == OVERFIT32_MANIFEST_SHA256,
        "route_image_count": len(route) == EXPECTED_IMAGES,
        "route_molecule_count": route_audit["molecule_count"] == EXPECTED_MOLECULES,
        "manifest_order_preserved": sample_keys == [item.sample_key for item in route],
        "manifest_disjointness": disjointness["all_three_zero_overlap"],
        "p0i_checkpoint_sha": file_sha256(args.p0i_checkpoint) == P0I_CHECKPOINT_SHA256,
        "p0g_checkpoint_sha": file_sha256(args.p0g_checkpoint) == P0G_CHECKPOINT_SHA256,
        "p0k_checkpoint_sha": file_sha256(args.p0k_checkpoint) == P0K_CHECKPOINT_SHA256,
        "vision_checkpoint_sha": vision_sha == DEFAULT_VISION_CHECKPOINT_SHA256,
        "zero_trainable_parameters": not any(
            parameter.requires_grad
            for model in (vision, p0i, p0g)
            for parameter in model.parameters()
        ),
        "zero_optimizers": True,
        "frozen_state_unchanged": initial_states == final_states,
        "frozen_modes_unchanged_eval": (
            initial_modes == final_modes
            and not any(initial_modes.values())
        ),
        "paired_identity_unchanged": (
            paired["same_slot"]["pair_identity_sha256"]
            == paired["native"]["pair_identity_sha256"]
        ),
        "gt_after_main_forward_only": not deploy_contract["accepts_targets"],
        "occupancy_no_overflow": (
            occupancy["cell_occupancy"]["overflow_gt_3"]["cells"] == 0
        ),
        "occupancy_no_identical_collision": (
            occupancy["identical_normalized_anchor_collision"]["groups"] == 0
        ),
        "zero_backward_and_updates": (
            static_contract["backward_calls"] == 0
            and static_contract["updates"] == 0
        ),
        "forbidden_data_not_referenced": (
            static_contract["forbidden_data_path_references"] == 0
        ),
        "all_unordered_pairs_no_filter": True,
    }
    gate = p0m_gate(
        protocol_checks=protocol_checks,
        same_slot=paired["same_slot"],
        native=paired["native"],
        oracle=oracle,
        occupancy=occupancy,
    )
    report = {
        "schema_version": "g3-p0m-frozen-route-select-100-audit-0.1",
        "scope": "frozen-route-select-100-read-only-capability",
        "manifests": {
            "route_select": route_audit,
            "overfit32": overfit_audit,
            "disjointness": disjointness,
            "scaffold_overlap_report_only": scaffold,
        },
        "lineage": {
            "vision": {
                "path": str(Path(p0i_payload["source_vision_checkpoint"]).resolve()),
                "sha256": vision_sha,
            },
            "p0i": {
                "path": str(Path(args.p0i_checkpoint).resolve()),
                "sha256": file_sha256(args.p0i_checkpoint),
            },
            "p0g": {
                "path": str(Path(args.p0g_checkpoint).resolve()),
                "sha256": file_sha256(args.p0g_checkpoint),
            },
            "p0k": {
                "path": str(Path(args.p0k_checkpoint).resolve()),
                "sha256": file_sha256(args.p0k_checkpoint),
                "global_step": p0k_payload["global_step"],
            },
        },
        "execution_audit": {
            "optimizer_count": 0,
            "trainable_parameter_count": 0,
            "initial_module_states": initial_states,
            "final_module_states": final_states,
            "initial_module_modes": initial_modes,
            "final_module_modes": final_modes,
            "main_forward_accepts_targets": False,
            "deploy_forward_contract": deploy_contract,
            "static_contract": static_contract,
            "graph_head_disjointness_only": True,
            "vision_may_have_seen_same_train_distribution_or_shard": True,
        },
        "occupancy": occupancy,
        "same_slot_main": paired["same_slot"],
        "native_paired_control": paired["native"],
        "gt_atom_oracle_relation_control": oracle,
        "p0m_gate": gate,
        "implementation": {
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": file_sha256(__file__),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "route_select_audit.json", report)
    _atomic_json(
        output_dir / "per_sample_metrics.json",
        {
            "schema_version": "g3-p0m-per-sample-metrics-0.1",
            "same_slot": paired["same_slot"]["per_sample_exact"],
            "native": paired["native"]["per_sample_exact"],
        },
    )
    _atomic_json(
        output_dir / "status.json",
        {
            "schema_version": "g3-p0m-status-0.1",
            "status": gate["status"],
            "decision": gate["decision"],
            "confirm400_may_be_read": gate["confirm400_may_be_read"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
