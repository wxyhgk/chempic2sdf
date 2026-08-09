#!/usr/bin/env python3
"""Audit the joint Oracle ceiling of frozen P4 Atom and v0.10.1 Bond proposals."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

from chemvector.components.atoms.objects.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
)
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    DenseBondObjectDetector,
    MultiScaleDenseBondObjectDetector,
)
from chemvector.components.graph.joint.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_bond_proposals,
)
from chemvector.components.graph.proposals.proposal_object_set import (
    ProposalObjectSetConfig,
    ProposalObjectSetHead,
)
from chemvector.components.graph.joint.spatial_incidence import SpatialIncidenceConfig
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e18_freshdev8k as freshdev,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as bond_source,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.atoms import (
    train_g3_p4e3_global_peak_rank_pilot20k as p4e3,
)
from chemvector.training.atoms.train_spatial_atom_object import load_grounded_vision
from chemvector.training.graph.proposals.train_proposal_object_set import (
    evaluate as evaluate_proposal_object_set,
)
from chemvector.components.graph.joint.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.graph.scenes.g3_structural_data import (
    G3SampleManifestRecord,
    G3StructuralCollator,
)
from chemvector.training.graph.scenes.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p18e0-proposal-oracle-ladder-fresh640"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CACHE = OUTPUT / "candidate-cache.pt"
SELECTION_MANIFEST = OUTPUT / "fresh640.jsonl"
FULL_FRESH_MANIFEST = freshdev.manifest_protocol.MANIFEST
FULL_FRESH_MANIFEST_SHA256 = freshdev.MANIFEST_SHA256
PILOT_MANIFEST = Path(
    "outputs/audits/chemvector-g3-p4e0-source-union-cycle-manifests/pilot_train.jsonl"
)
P4_CHECKPOINT = p4e6.P4E3_CHECKPOINT
P4_CHECKPOINT_SHA256 = p4e6.P4E3_CHECKPOINT_SHA256
BOND_CHECKPOINT = bond_source.BOND_SOURCE
BOND_CHECKPOINT_SHA256 = bond_source.BOND_SOURCE_SHA256
BOND_VISION = bond_source.VISION_SOURCE
BOND_VISION_SHA256 = bond_source.VISION_SOURCE_SHA256
BASELINE_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v0971-joint-proposal-objects-full50k/"
    "samples-030000.pt"
)
BASELINE_CHECKPOINT_SHA256 = (
    "f8935e92d64e90a2b03b5f1924813cf19c08e7f6bfe382357b03e0eccd047f75"
)
VARIANTS = ("coordgen", "original", "rdkit", "stress")
MOLECULES = 160
IMAGES = MOLECULES * len(VARIANTS)
ATOM_CAPACITY = 128
BOND_CAPACITY = 192
ATOM_RADIUS_PX = 6.0
BOND_RADIUS_PX = 12.0
IMAGE_SIZE = 768.0
BATCH_SIZE = 8
SEED = 20260810
GATES = {
    "joint_exact_rate_min": 0.70,
    "variant_joint_exact_rate_min": 0.60,
    "baseline_headroom_min": 0.20,
    "compile_loss_max": 0.02,
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_fresh640(rows: list[dict]) -> tuple[list[dict], dict]:
    """Select complete molecule groups by a stable molecule/scaffold digest."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["molecule_id"])].append(row)
    eligible = []
    for molecule_id, group in grouped.items():
        variants = {str(row["variant"]) for row in group}
        scaffolds = {str(row.get("scaffold", "")) for row in group}
        if len(group) != len(VARIANTS) or variants != set(VARIANTS):
            raise ValueError(f"P18-E0 incomplete fresh molecule group: {molecule_id}")
        if len(scaffolds) != 1:
            raise ValueError(f"P18-E0 molecule has multiple scaffolds: {molecule_id}")
        scaffold = next(iter(scaffolds))
        digest = hashlib.sha256(f"{molecule_id}\0{scaffold}".encode()).hexdigest()
        eligible.append((digest, molecule_id, group))
    eligible.sort(key=lambda item: (item[0], item[1]))
    if len(eligible) < MOLECULES:
        raise ValueError("P18-E0 fresh_dev has too few complete molecules")
    chosen = eligible[:MOLECULES]
    chosen_ids = {item[1] for item in chosen}
    # The source manifest SHA pins physical tar order. Keeping that order allows
    # the streaming loader to prove one-pass identity without random tar seeks.
    selected = [row for row in rows if str(row["molecule_id"]) in chosen_ids]
    sample_keys = [str(row["sample_key"]) for row in selected]
    return selected, {
        "policy": (
            "first 160 SHA256(molecule_id\\0scaffold), then committed source "
            "manifest physical order"
        ),
        "molecules": len(chosen),
        "images": len(selected),
        "variants": dict(Counter(str(row["variant"]) for row in selected)),
        "sample_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in sample_keys).encode()
        ).hexdigest(),
        "molecule_ids_sha256": hashlib.sha256(
            "".join(f"{item[1]}\n" for item in chosen).encode()
        ).hexdigest(),
    }


def _manifest_records(rows: list[dict]) -> list[G3SampleManifestRecord]:
    return [
        G3SampleManifestRecord(
            sample_key=str(row["sample_key"]),
            molecule_id=str(row["molecule_id"]),
            inchikey=str(row["inchikey"]),
            shard=str(row["shard"]),
            variant=str(row["variant"]),
        )
        for row in rows
    ]


def _maximum_radius_assignment(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    radius_px: float,
) -> torch.Tensor:
    """Return target-to-prediction assignment with maximum feasible cardinality."""

    result = torch.full((len(expected),), -1, dtype=torch.long)
    if not len(predicted) or not len(expected):
        return result
    distances = torch.cdist(predicted.double(), expected.double()) * IMAGE_SIZE
    compatible = distances <= radius_px
    penalty = float(max(len(predicted), len(expected)) + 1) * IMAGE_SIZE
    cost = torch.where(compatible, distances, torch.full_like(distances, penalty))
    prediction_indices, target_indices = linear_sum_assignment(cost.numpy())
    for prediction, target in zip(prediction_indices.tolist(), target_indices.tolist()):
        if bool(compatible[prediction, target]):
            result[target] = prediction
    return result


def _maximum_bond_assignment(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    radius_px: float,
    *,
    predicted_types: torch.Tensor | None = None,
    expected_types: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match unordered Bond endpoints and return target mapping plus orientation."""

    result = torch.full((len(expected),), -1, dtype=torch.long)
    reversed_result = torch.zeros((len(expected),), dtype=torch.bool)
    if not len(predicted) or not len(expected):
        return result, reversed_result
    forward = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None]) * IMAGE_SIZE, dim=-1
    ).amax(-1)
    reverse = torch.linalg.vector_norm(
        (predicted[:, None] - expected[None].flip(2)) * IMAGE_SIZE, dim=-1
    ).amax(-1)
    distances = torch.minimum(forward, reverse).double()
    compatible = distances <= radius_px
    if predicted_types is not None or expected_types is not None:
        if predicted_types is None or expected_types is None:
            raise ValueError("P18-E0 typed Bond matching needs both type vectors")
        compatible &= predicted_types[:, None] == expected_types[None]
    penalty = float(max(len(predicted), len(expected)) + 1) * IMAGE_SIZE
    cost = torch.where(compatible, distances, torch.full_like(distances, penalty))
    prediction_indices, target_indices = linear_sum_assignment(cost.numpy())
    for prediction, target in zip(prediction_indices.tolist(), target_indices.tolist()):
        if bool(compatible[prediction, target]):
            result[target] = prediction
            reversed_result[target] = reverse[prediction, target] < forward[prediction, target]
    return result, reversed_result


def _record_oracle_metrics(record: dict) -> dict[str, int | bool | float]:
    target_atoms = record["target_atom_anchors"][record["target_atom_mask"]].float()
    target_refs = record["target_bond_atom_refs"][record["target_bond_mask"]].long()
    target_types = record["target_bond_types"][record["target_bond_mask"]].long()
    target_bonds = target_atoms[target_refs]
    atom_candidates = record["atom_candidate_anchors"][record["atom_candidate_mask"]].float()
    bond_mask = record["bond_candidate_mask"]
    bond_candidates = record["bond_candidate_endpoints"][bond_mask].float()
    bond_type_probabilities = record["bond_candidate_types"][bond_mask].float()
    type_count = bond_type_probabilities.size(-1) // 2
    predicted_chemical_types = bond_type_probabilities[:, type_count:].argmax(-1)

    atom_assignment = _maximum_radius_assignment(
        atom_candidates, target_atoms, ATOM_RADIUS_PX
    )
    bond_assignment, bond_reversed = _maximum_bond_assignment(
        bond_candidates, target_bonds, BOND_RADIUS_PX
    )
    typed_assignment, _ = _maximum_bond_assignment(
        bond_candidates,
        target_bonds,
        BOND_RADIUS_PX,
        predicted_types=predicted_chemical_types,
        expected_types=target_types,
    )
    atom_matched = int((atom_assignment >= 0).sum())
    bond_matched = int((bond_assignment >= 0).sum())
    typed_bond_matched = int((typed_assignment >= 0).sum())
    atom_exact = atom_matched == len(target_atoms)
    bond_exact = bond_matched == len(target_bonds)
    typed_bond_exact = typed_bond_matched == len(target_bonds)
    joint_exact = atom_exact and bond_exact

    compiled_edges: set[tuple[int, int]] = set()
    incidence_feasible = joint_exact
    maximum_selected_endpoint_error = 0.0
    if joint_exact:
        for target_bond, refs in enumerate(target_refs.tolist()):
            prediction = int(bond_assignment[target_bond])
            endpoints = bond_candidates[prediction]
            if bool(bond_reversed[target_bond]):
                endpoints = endpoints.flip(0)
            selected_atoms = atom_candidates[atom_assignment[torch.tensor(refs)]]
            endpoint_error = torch.linalg.vector_norm(
                (endpoints - selected_atoms) * IMAGE_SIZE, dim=-1
            )
            maximum_selected_endpoint_error = max(
                maximum_selected_endpoint_error, float(endpoint_error.max())
            )
            begin, end = sorted((int(refs[0]), int(refs[1])))
            if begin == end or (begin, end) in compiled_edges:
                incidence_feasible = False
            compiled_edges.add((begin, end))
    expected_edges = {tuple(sorted(map(int, refs))) for refs in target_refs.tolist()}
    compiled_exact = joint_exact and incidence_feasible and compiled_edges == expected_edges
    return {
        "atom_targets": len(target_atoms),
        "atom_matched": atom_matched,
        "bond_targets": len(target_bonds),
        "bond_matched": bond_matched,
        "typed_bond_matched": typed_bond_matched,
        "atom_exact": atom_exact,
        "bond_exact": bond_exact,
        "typed_bond_exact": typed_bond_exact,
        "joint_exact": joint_exact,
        "incidence_feasible": incidence_feasible,
        "compiled_exact": compiled_exact,
        "maximum_selected_endpoint_error_px": maximum_selected_endpoint_error,
    }


def _summarize(rows: list[dict]) -> dict:
    names = (
        "atom_targets",
        "atom_matched",
        "bond_targets",
        "bond_matched",
        "typed_bond_matched",
        "atom_exact",
        "bond_exact",
        "typed_bond_exact",
        "joint_exact",
        "incidence_feasible",
        "compiled_exact",
    )
    totals = {name: sum(int(row[name]) for row in rows) for name in names}
    images = len(rows)
    return {
        "images": images,
        **totals,
        "atom_recall": totals["atom_matched"] / max(totals["atom_targets"], 1),
        "bond_recall": totals["bond_matched"] / max(totals["bond_targets"], 1),
        "typed_bond_recall": totals["typed_bond_matched"] / max(totals["bond_targets"], 1),
        **{
            f"{name}_rate": totals[name] / max(images, 1)
            for name in (
                "atom_exact",
                "bond_exact",
                "typed_bond_exact",
                "joint_exact",
                "incidence_feasible",
                "compiled_exact",
            )
        },
        "maximum_selected_endpoint_error_px": max(
            (float(row["maximum_selected_endpoint_error_px"]) for row in rows),
            default=0.0,
        ),
    }


def oracle_ladder(records: list[dict]) -> dict:
    rows = []
    for record in records:
        metrics = _record_oracle_metrics(record)
        rows.append(
            {
                "sample_key": record["sample_key"],
                "variant": record["variant"],
                **metrics,
            }
        )
    overall = _summarize(rows)
    by_variant = {
        variant: _summarize([row for row in rows if row["variant"] == variant])
        for variant in VARIANTS
    }
    return {"overall": overall, "by_variant": by_variant, "rows": rows}


def _candidate_digest(records: list[dict]) -> str:
    digest = hashlib.sha256()
    tensor_names = (
        "atom_candidate_ids",
        "atom_candidate_features",
        "atom_candidate_anchors",
        "atom_candidate_scores",
        "atom_candidate_mask",
        "bond_candidate_ids",
        "bond_candidate_features",
        "bond_candidate_endpoints",
        "bond_candidate_types",
        "bond_candidate_scores",
        "bond_candidate_mask",
    )
    for record in records:
        digest.update(f"{record['sample_key']}\n".encode())
        for name in tensor_names:
            value = record[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def collect_candidates(
    manifest: list[G3SampleManifestRecord], device: torch.device
) -> tuple[list[dict], dict]:
    atom_records, atom_lineage = p1x._collect_scale(
        checkpoint=P4_CHECKPOINT,
        checkpoint_sha256=P4_CHECKPOINT_SHA256,
        expected_model_type=p4e3.MODEL_TYPE,
        field_factory=p4e6.ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=ATOM_CAPACITY,
        domains={"fresh640": manifest},
        collator=p1a.G3AtomCollator(maximum_atoms=ATOM_CAPACITY, image_size=768),
        device=device,
        batch_size=BATCH_SIZE,
        accepted_split=freshdev.manifest_protocol.EXPECTED_SPLIT,
    )
    atoms_by_key = atom_records["fresh640"]
    samples = list(
        p1a.iter_manifest_samples(
            manifest, accepted_split=freshdev.manifest_protocol.EXPECTED_SPLIT
        )
    )
    vision, bond_head, proposal = bond_source._load_bond_source(device)
    collator = G3StructuralCollator(
        maximum_atoms=ATOM_CAPACITY, maximum_bonds=BOND_CAPACITY, image_size=768
    )
    bond_state_before = module_state_sha256(bond_head)
    vision_state_before = module_state_sha256(vision)
    records = []
    for start in range(0, len(samples), BATCH_SIZE):
        current = samples[start : start + BATCH_SIZE]
        batch = collator(current)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            stroke = vision.forward_stroke_resolution(pixels)
            output = bond_source.forward_bond_source(vision, bond_head, stroke)
            bonds = decode_predicted_bond_proposals(output, proposal)
        for index, sample in enumerate(current):
            key = str(sample["key"])
            atom = atoms_by_key[key]
            records.append(
                {
                    "sample_key": key,
                    "molecule_id": str(sample["chemmol"]["molecule_id"]),
                    "variant": str(sample["render"]["variant"]),
                    "target_atom_anchors": batch["atom_anchors"][index].clone(),
                    "target_atom_mask": batch["atom_mask"][index].clone(),
                    "target_bond_atom_refs": batch["bond_atom_refs"][index].clone(),
                    "target_bond_types": batch["bond_types"][index].clone(),
                    "target_bond_mask": batch["bond_mask"][index].clone(),
                    "atom_candidate_ids": atom["source_ids"].long().clone(),
                    "atom_candidate_features": atom["candidate_features"].half().clone(),
                    "atom_candidate_anchors": atom["candidate_anchors"].float().clone(),
                    "atom_candidate_scores": atom["raw_scores"].float().clone(),
                    "atom_candidate_mask": torch.ones(ATOM_CAPACITY, dtype=torch.bool),
                    "bond_candidate_ids": bonds["bond_indices"][index].long().cpu().clone(),
                    "bond_candidate_features": bonds["bond_features"][index].half().cpu().clone(),
                    "bond_candidate_endpoints": bonds["bond_endpoints"][index].float().cpu().clone(),
                    "bond_candidate_types": bonds["bond_types"][index].half().cpu().clone(),
                    "bond_candidate_scores": bonds["bond_scores"][index].float().cpu().clone(),
                    "bond_candidate_mask": bonds["bond_mask"][index].cpu().clone(),
                }
            )
    state_checks = {
        "bond_state_unchanged": module_state_sha256(bond_head) == bond_state_before,
        "bond_vision_state_unchanged": module_state_sha256(vision) == vision_state_before,
    }
    lineage = {
        "atom": atom_lineage,
        "bond": {
            "checkpoint": str(BOND_CHECKPOINT.resolve()),
            "checkpoint_sha256": BOND_CHECKPOINT_SHA256,
            "vision_checkpoint": str(BOND_VISION.resolve()),
            "vision_checkpoint_sha256": BOND_VISION_SHA256,
            "proposal_config": asdict(proposal),
            "state_checks": state_checks,
        },
    }
    del samples, vision, bond_head
    gc.collect()
    torch.cuda.empty_cache()
    return records, lineage


def _load_baseline(device: torch.device):
    payload = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    graph_checkpoint = Path(payload["training_args"]["grounded_graph_checkpoint"])
    vision, vision_config, _ = load_grounded_vision(graph_checkpoint, device)
    atom_config = SpatialAtomObjectConfig(**payload["atom_config"])
    if atom_config.input_channels != vision_config.vision_dims[0]:
        raise ValueError("P18-E0 baseline Atom channels differ from vision")
    atom = SpatialAtomObjectDetector(atom_config).to(device)
    atom.load_state_dict(payload["atom_head"], strict=True)
    bond_config = DenseBondObjectConfig(**payload["bond_config"])
    bond = (
        MultiScaleDenseBondObjectDetector(bond_config)
        if payload.get("bond_multiscale")
        else DenseBondObjectDetector(bond_config)
    ).to(device)
    bond.load_state_dict(payload["bond_head"], strict=True)
    incidence = SpatialIncidenceConfig(**payload["set_config"]["incidence"])
    set_config = ProposalObjectSetConfig(
        incidence=incidence,
        hidden_dim=int(payload["set_config"]["hidden_dim"]),
        stable_bond_hidden_dim=int(payload["set_config"].get("stable_bond_hidden_dim", 0)),
    )
    set_head = ProposalObjectSetHead(set_config).to(device)
    set_head.load_state_dict(payload["set_head"], strict=True)
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    return vision, atom, bond, set_head, PredictedProposalConfig(**payload["proposal_config"])


@torch.inference_mode()
def evaluate_baseline(samples: list[dict], device: torch.device) -> dict:
    vision, atom, bond, set_head, proposal = _load_baseline(device)
    config = AtomBondSceneConfig(atom_queries=ATOM_CAPACITY, bond_queries=BOND_CAPACITY)
    metrics = evaluate_proposal_object_set(
        vision,
        atom,
        bond,
        set_head,
        samples,
        AtomBondSceneCollator(config),
        proposal,
        device,
        BATCH_SIZE,
        -0.25,
        -0.5,
    )
    del vision, atom, bond, set_head
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P18-E0 output is not empty: {OUTPUT}")
    for path, expected in (
        (FULL_FRESH_MANIFEST, FULL_FRESH_MANIFEST_SHA256),
        (P4_CHECKPOINT, P4_CHECKPOINT_SHA256),
        (BOND_CHECKPOINT, BOND_CHECKPOINT_SHA256),
        (BOND_VISION, BOND_VISION_SHA256),
        (BASELINE_CHECKPOINT, BASELINE_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P18-E0 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P18-E0 requires CUDA BF16")
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

    selected_rows, selection = select_fresh640(_jsonl(FULL_FRESH_MANIFEST))
    selected_molecules = {str(row["molecule_id"]) for row in selected_rows}
    selected_scaffolds = {str(row.get("scaffold", "")) for row in selected_rows}
    pilot_rows = _jsonl(PILOT_MANIFEST)
    pilot_molecules = {str(row["molecule_id"]) for row in pilot_rows}
    pilot_scaffolds = {str(row.get("scaffold", "")) for row in pilot_rows}
    leakage = {
        "molecule_overlap": len(selected_molecules & pilot_molecules),
        "scaffold_overlap": len((selected_scaffolds - {""}) & (pilot_scaffolds - {""})),
    }
    if any(leakage.values()):
        raise RuntimeError(f"P18-E0 fresh640 overlaps P4 pilot: {leakage}")
    manifest = _manifest_records(selected_rows)
    samples = list(
        p1a.iter_manifest_samples(
            manifest, accepted_split=freshdev.manifest_protocol.EXPECTED_SPLIT
        )
    )

    first, lineage = collect_candidates(manifest, device)
    first_digest = _candidate_digest(first)
    second, repeat_lineage = collect_candidates(manifest, device)
    second_digest = _candidate_digest(second)
    repeat_exact = first_digest == second_digest
    if not repeat_exact:
        raise RuntimeError("P18-E0 frozen proposal replay digest differs")
    baseline = evaluate_baseline(samples, device)
    ladder = oracle_ladder(first)
    overall = ladder["overall"]
    baseline_rate = float(baseline["full_object_exact_rate"])
    compile_loss = overall["joint_exact_rate"] - overall["compiled_exact_rate"]
    checks = {
        "joint_exact_rate_min70": overall["joint_exact_rate"] >= GATES["joint_exact_rate_min"],
        "all_variants_joint_exact_rate_min60": all(
            value["joint_exact_rate"] >= GATES["variant_joint_exact_rate_min"]
            for value in ladder["by_variant"].values()
        ),
        "baseline_headroom_min20pp": (
            overall["joint_exact_rate"] - baseline_rate >= GATES["baseline_headroom_min"]
        ),
        "compile_loss_max2pp": compile_loss <= GATES["compile_loss_max"],
        "repeat_candidate_digest_exact": repeat_exact,
        "no_pilot_molecule_or_scaffold_overlap": not any(leakage.values()),
        "all_frozen_states_unchanged": all(lineage["bond"]["state_checks"].values()),
    }
    decision = "proposal_oracle_ladder_go" if all(checks.values()) else "proposal_oracle_ladder_no_go"

    OUTPUT.mkdir(parents=True, exist_ok=False)
    SELECTION_MANIFEST.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in selected_rows) + "\n",
        encoding="utf-8",
    )
    _atomic_torch_save(
        CACHE,
        {
            "schema_version": "g3-p18e0-frozen-proposal-cache-0.1",
            "records": first,
            "candidate_tensors_sha256": first_digest,
            "selection": selection,
            "lineage": lineage,
        },
    )
    _atomic_json(OUTPUT / "rows.json", {"rows": ladder.pop("rows")})
    report = {
        "schema_version": "g3-p18e0-proposal-oracle-ladder-0.1",
        "decision": decision,
        "checks": checks,
        "gates": GATES,
        "selection": {
            **selection,
            "manifest": str(SELECTION_MANIFEST.resolve()),
            "manifest_sha256": file_sha256(SELECTION_MANIFEST),
            "source_manifest": str(FULL_FRESH_MANIFEST.resolve()),
            "source_manifest_sha256": FULL_FRESH_MANIFEST_SHA256,
            "pilot_leakage": leakage,
        },
        "oracle_ladder": ladder,
        "baseline_v0971_same_fresh640": {
            "checkpoint": str(BASELINE_CHECKPOINT.resolve()),
            "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
            "metrics": baseline,
        },
        "derived": {
            "baseline_headroom": overall["joint_exact_rate"] - baseline_rate,
            "compile_loss": compile_loss,
        },
        "cache": {
            "path": str(CACHE.resolve()),
            "sha256": file_sha256(CACHE),
            "candidate_tensors_sha256": first_digest,
            "repeat_candidate_tensors_sha256": second_digest,
        },
        "lineage": lineage,
        "repeat_lineage": repeat_lineage,
        "data_access": {
            "fresh_dev_read": True,
            "gt_read_after_frozen_source_forward": True,
            "confirm_read": False,
            "sealed_test_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(
        json.dumps(
            {
                "decision": decision,
                "checks": checks,
                "overall": overall,
                "by_variant": ladder["by_variant"],
                "baseline_full_object_exact_rate": baseline_rate,
                "baseline_headroom": report["derived"]["baseline_headroom"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
