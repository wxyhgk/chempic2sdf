#!/usr/bin/env python3
"""Compile v0.9.49 object sets into ChemScene v0.2 with Oracle chemistry."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from scipy.optimize import linear_sum_assignment
import torch

from Data.sdf.chemscene_v0_1 import chemsvg_from_chemscene_v0_1
from Data.sdf.chemscene_v0_2 import (
    QUANTIZATION_MAX,
    SCHEMA_VERSION,
    UNITS,
    _clip_port_to_text_bbox,
    build_chemscene_v0_2,
    chemscene_v0_1_from_v0_2,
    mol_from_chemscene_v0_2,
    parse_chemscene_v0_2,
)
from chemvector.benchmark_mr_tadf import (
    failure_category,
    molecular_similarity,
    write_invalid_prediction_sdf,
    write_prediction_sdf,
)
from chemvector.benchmark_public_ocsr import sdf_graph_match
from chemvector.bond_object import BOND_TYPES
from chemvector.joint_graph_milp import JointGraphMilpConfig, apply_joint_graph_milp
from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.train_cached_proposal_object_set import (
    _forward,
    load_cache,
    stack_cache_records,
)
from chemvector.train_predicted_incidence import _load_records


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0949-cached-proposal-object-set-train8974-step3000/"
    "best.pt"
)
DEFAULT_CACHE = "outputs/checkpoints/chemvector-v0949-proposal-cache-internal-test100"
DEFAULT_DATA_SHARD = (
    "Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/internal_test-000000.tar"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0950-chemscene-v02-oracle-shadow-internal100"
)
AROMATIC_CONSISTENCY_MODES = (
    "none",
    "demote-noncycle",
    "demote-noncycle-clear-atoms",
)
BOND_CHEMISTRY_MODES = ("predicted", "oracle-matched")
OBJECT_PRESENCE_MODES = ("predicted", "oracle")
INCIDENCE_MODES = ("predicted", "oracle-eligible")
JOINT_GRAPH_DECODERS = ("none", "milp")
JOINT_INVALID_FALLBACKS = ("none", "independent")


def _q(value: float) -> int:
    return max(0, min(QUANTIZATION_MAX, round(float(value) * QUANTIZATION_MAX)))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def _default_carbon() -> dict:
    return {
        "element": "C",
        "formal_charge": 0,
        "isotope": 0,
        "aromatic": False,
        "explicit_h_count": 0,
        "no_implicit": False,
        "radical_electrons": 0,
        "canonical_chiral_tag": "CHI_UNSPECIFIED",
    }


def _cycle_membership(scene: dict) -> tuple[set[int], set[int]]:
    bonds = scene.get("bond_objects", [])
    adjacency: dict[int, list[tuple[int, int]]] = {
        int(atom["id"]): [] for atom in scene.get("atom_objects", [])
    }
    for index, bond in enumerate(bonds):
        begin, end = (int(value) for value in bond["atom_refs"])
        adjacency.setdefault(begin, []).append((end, index))
        adjacency.setdefault(end, []).append((begin, index))

    def has_alternate_path(begin: int, end: int, excluded_bond: int) -> bool:
        pending = [begin]
        visited = {begin}
        while pending:
            atom = pending.pop()
            for neighbor, bond_index in adjacency.get(atom, []):
                if bond_index == excluded_bond or neighbor in visited:
                    continue
                if neighbor == end:
                    return True
                visited.add(neighbor)
                pending.append(neighbor)
        return False

    cycle_bonds = {
        index
        for index, bond in enumerate(bonds)
        if has_alternate_path(
            int(bond["atom_refs"][0]), int(bond["atom_refs"][1]), index
        )
    }
    cycle_atoms = {
        int(atom_ref)
        for index in cycle_bonds
        for atom_ref in bonds[index]["atom_refs"]
    }
    return cycle_atoms, cycle_bonds


def demote_noncycle_aromatic_bonds(scene: dict) -> tuple[dict, dict]:
    """Demote predicted aromatic bonds that cannot belong to any graph cycle."""
    repaired = deepcopy(scene)
    bonds = repaired.get("bond_objects", [])
    _, cycle_bonds = _cycle_membership(repaired)
    changed_ids = []
    replacement_counts: Counter[str] = Counter()
    for index, bond in enumerate(bonds):
        if bond.get("chemical_type") != "AROMATIC" or index in cycle_bonds:
            continue
        replacement = bond.get("display_type")
        if replacement not in {"SINGLE", "DOUBLE", "TRIPLE"}:
            replacement = "SINGLE"
        bond["chemical_type"] = replacement
        changed_ids.append(int(bond.get("id", index)))
        replacement_counts[replacement] += 1
    diagnostics = {
        "mode": "demote-noncycle",
        "cycle_bond_count": len(cycle_bonds),
        "demoted_aromatic_bond_count": len(changed_ids),
        "demoted_aromatic_bond_ids": changed_ids,
        "replacement_counts": dict(replacement_counts),
        "cleared_noncycle_aromatic_atom_count": 0,
        "cleared_noncycle_aromatic_atom_ids": [],
    }
    return repaired, diagnostics


def clear_noncycle_aromatic_atoms(scene: dict) -> tuple[dict, dict]:
    """Clear aromatic flags only for atoms outside every topological cycle."""
    repaired = deepcopy(scene)
    cycle_atoms, _ = _cycle_membership(repaired)
    changed_ids = []
    for atom in repaired.get("atom_objects", []):
        atom_id = int(atom["id"])
        chemical = atom.get("chemical", {})
        if chemical.get("aromatic") and atom_id not in cycle_atoms:
            chemical["aromatic"] = False
            changed_ids.append(atom_id)
    return repaired, {
        "cleared_noncycle_aromatic_atom_count": len(changed_ids),
        "cleared_noncycle_aromatic_atom_ids": changed_ids,
    }


def apply_aromatic_consistency(scene: dict, mode: str) -> tuple[dict, dict]:
    if mode == "none":
        return deepcopy(scene), {
            "mode": "none",
            "cycle_bond_count": 0,
            "demoted_aromatic_bond_count": 0,
            "demoted_aromatic_bond_ids": [],
            "replacement_counts": {},
            "cleared_noncycle_aromatic_atom_count": 0,
            "cleared_noncycle_aromatic_atom_ids": [],
        }
    if mode == "demote-noncycle":
        return demote_noncycle_aromatic_bonds(scene)
    if mode == "demote-noncycle-clear-atoms":
        repaired, diagnostics = demote_noncycle_aromatic_bonds(scene)
        repaired, atom_diagnostics = clear_noncycle_aromatic_atoms(repaired)
        diagnostics.update(atom_diagnostics)
        diagnostics["mode"] = mode
        return repaired, diagnostics
    raise ValueError(f"Unknown aromatic consistency mode: {mode}")


def apply_oracle_object_controls(
    *,
    atom_presence_logits: torch.Tensor,
    bond_presence_logits: torch.Tensor,
    incidence_logits: torch.Tensor,
    batch: dict,
    batch_index: int,
    object_presence_mode: str,
    incidence_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if object_presence_mode not in OBJECT_PRESENCE_MODES:
        raise ValueError(f"Unknown object presence mode: {object_presence_mode}")
    if incidence_mode not in INCIDENCE_MODES:
        raise ValueError(f"Unknown incidence mode: {incidence_mode}")
    atom_output = atom_presence_logits.clone()
    bond_output = bond_presence_logits.clone()
    incidence_output = incidence_logits.clone()
    if object_presence_mode == "oracle":
        atom_targets = batch["atom_presence_targets"][batch_index].cpu()
        bond_targets = batch["bond_presence_targets"][batch_index].cpu()
        atom_output = torch.where(
            atom_targets, torch.full_like(atom_output, 10.0), torch.full_like(atom_output, -10.0)
        )
        bond_output = torch.where(
            bond_targets, torch.full_like(bond_output, 10.0), torch.full_like(bond_output, -10.0)
        )
    if incidence_mode == "oracle-eligible":
        supervised = batch["bond_supervision_mask"][batch_index].cpu()
        targets = batch["target_atom_refs"][batch_index].cpu()
        incidence_output[supervised] = -10.0
        supervised_indices = torch.where(supervised)[0]
        for bond_index in supervised_indices.tolist():
            for endpoint in range(2):
                incidence_output[
                    bond_index, endpoint, int(targets[bond_index, endpoint])
                ] = 10.0
    return atom_output, bond_output, incidence_output


def build_oracle_chemistry_shadow_scene(
    *,
    molecule_id: str,
    gt_scene: dict,
    atom_geometry: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_presence_logits: torch.Tensor,
    bond_types: torch.Tensor,
    bond_mask: torch.Tensor,
    bond_presence_logits: torch.Tensor,
    incidence_logits: torch.Tensor,
    atom_threshold: float,
    bond_threshold: float,
    atom_match_radius_px: float = 6.0,
    bond_chemistry_mode: str = "predicted",
    endpoint_atom_rescue_margin: float = 0.0,
) -> tuple[dict, dict]:
    if bond_chemistry_mode not in BOND_CHEMISTRY_MODES:
        raise ValueError(f"Unknown bond chemistry mode: {bond_chemistry_mode}")
    if endpoint_atom_rescue_margin < 0.0:
        raise ValueError("Endpoint atom rescue margin must be non-negative")
    selected_bonds = torch.where(bond_mask & (bond_presence_logits >= bond_threshold))[0]
    selected_bonds = sorted(
        selected_bonds.tolist(),
        key=lambda index: (-float(bond_presence_logits[index]), index),
    )
    initial_selected_atoms = set(
        torch.where(atom_mask & (atom_presence_logits >= atom_threshold))[0].tolist()
    )
    rescued_atoms = set()
    if endpoint_atom_rescue_margin > 0.0:
        rescue_threshold = atom_threshold - endpoint_atom_rescue_margin
        for bond_candidate in selected_bonds:
            for atom_candidate in incidence_logits[bond_candidate].argmax(-1).tolist():
                if (
                    bool(atom_mask[atom_candidate])
                    and float(atom_presence_logits[atom_candidate]) >= rescue_threshold
                    and atom_candidate not in initial_selected_atoms
                ):
                    rescued_atoms.add(atom_candidate)
    selected_atoms = torch.tensor(
        sorted(initial_selected_atoms | rescued_atoms), dtype=torch.long
    )
    selected_atoms = sorted(
        selected_atoms.tolist(),
        key=lambda index: (-float(atom_presence_logits[index]), index),
    )
    candidate_by_anchor: dict[tuple[int, int], int] = {}
    dropped_duplicate_atoms = 0
    for candidate in selected_atoms:
        anchor = (_q(atom_geometry[candidate, 0]), _q(atom_geometry[candidate, 1]))
        if anchor in candidate_by_anchor:
            dropped_duplicate_atoms += 1
            continue
        candidate_by_anchor[anchor] = candidate
    selected_atoms = list(candidate_by_anchor.values())
    if not selected_atoms:
        raise ValueError("Atom presence selector returned an empty set")

    gt_atoms = gt_scene["atom_objects"]
    predicted_anchors = torch.tensor(
        [atom_geometry[index, :2].float().tolist() for index in selected_atoms]
    )
    gt_anchors = torch.tensor(
        [
            [
                atom["anchor"][0] / QUANTIZATION_MAX,
                atom["anchor"][1] / QUANTIZATION_MAX,
            ]
            for atom in gt_atoms
        ],
        dtype=torch.float32,
    )
    oracle_atom_by_candidate: dict[int, int] = {}
    if len(gt_anchors):
        cost = torch.cdist(predicted_anchors, gt_anchors) * 768.0
        rows, columns = linear_sum_assignment(cost.numpy())
        for row, column in zip(rows, columns):
            if float(cost[row, column]) <= atom_match_radius_px:
                oracle_atom_by_candidate[selected_atoms[row]] = int(column)

    selected_atoms.sort(
        key=lambda index: (
            _q(atom_geometry[index, 1]),
            _q(atom_geometry[index, 0]),
            index,
        )
    )
    atom_id_by_candidate = {
        candidate: atom_id for atom_id, candidate in enumerate(selected_atoms)
    }
    atom_objects = []
    for atom_id, candidate in enumerate(selected_atoms):
        anchor = [_q(atom_geometry[candidate, 0]), _q(atom_geometry[candidate, 1])]
        gt_index = oracle_atom_by_candidate.get(candidate)
        if gt_index is None:
            chemical = _default_carbon()
            role = "IMPLICIT"
            text = None
        else:
            gt_atom = gt_atoms[gt_index]
            chemical = dict(gt_atom["chemical"])
            role = gt_atom["role"]
            if gt_atom["text"] is None:
                text = None
            else:
                center = [
                    _q(atom_geometry[candidate, 4]),
                    _q(atom_geometry[candidate, 5]),
                ]
                extent = [
                    _q(atom_geometry[candidate, 6]),
                    _q(atom_geometry[candidate, 7]),
                ]
                text = {
                    "value": gt_atom["text"]["value"],
                    "center": center,
                    "bbox": [
                        max(0, center[0] - extent[0]),
                        max(0, center[1] - extent[1]),
                        min(QUANTIZATION_MAX, center[0] + extent[0]),
                        min(QUANTIZATION_MAX, center[1] + extent[1]),
                    ],
                }
        atom_objects.append(
            {
                "id": atom_id,
                "anchor": anchor,
                "role": role,
                "chemical": chemical,
                "text": text,
                "incident_ports": [],
            }
        )

    accepted = []
    seen_pairs = set()
    dropped_unselected_endpoint = dropped_self_bond = dropped_duplicate_bond = 0
    for candidate in selected_bonds:
        endpoint_candidates = incidence_logits[candidate].argmax(-1).tolist()
        if any(index not in atom_id_by_candidate for index in endpoint_candidates):
            dropped_unselected_endpoint += 1
            continue
        atom_refs = [atom_id_by_candidate[index] for index in endpoint_candidates]
        if atom_refs[0] == atom_refs[1]:
            dropped_self_bond += 1
            continue
        pair = tuple(sorted(atom_refs))
        if pair in seen_pairs:
            dropped_duplicate_bond += 1
            continue
        seen_pairs.add(pair)
        accepted.append((candidate, atom_refs))

    gt_bond_by_pair = {
        tuple(sorted(int(value) for value in bond["atom_refs"])): bond
        for bond in gt_scene.get("bond_objects", [])
    }
    bond_objects = []
    oracle_matched_bond_chemistry = 0
    for bond_id, (candidate, atom_refs) in enumerate(accepted):
        begin_atom, end_atom = atom_refs
        begin_anchor = tuple(atom_objects[begin_atom]["anchor"])
        end_anchor = tuple(atom_objects[end_atom]["anchor"])
        begin_text = atom_objects[begin_atom]["text"]
        end_text = atom_objects[end_atom]["text"]
        begin_port, begin_source = _clip_port_to_text_bbox(
            begin_anchor,
            end_anchor,
            tuple(begin_text["bbox"]) if begin_text is not None else None,
        )
        end_port, end_source = _clip_port_to_text_bbox(
            end_anchor,
            begin_anchor,
            tuple(end_text["bbox"]) if end_text is not None else None,
        )
        atom_objects[begin_atom]["incident_ports"].append(
            {
                "bond_id": bond_id,
                "endpoint": 0,
                "coordinate": list(begin_port),
                "source": begin_source,
            }
        )
        atom_objects[end_atom]["incident_ports"].append(
            {
                "bond_id": bond_id,
                "endpoint": 1,
                "coordinate": list(end_port),
                "source": end_source,
            }
        )
        display = BOND_TYPES[int(bond_types[candidate, :4].argmax())]
        chemical = BOND_TYPES[int(bond_types[candidate, 4:].argmax())]
        if bond_chemistry_mode == "oracle-matched":
            endpoint_candidates = incidence_logits[candidate].argmax(-1).tolist()
            gt_atom_refs = [
                oracle_atom_by_candidate.get(index) for index in endpoint_candidates
            ]
            if all(index is not None for index in gt_atom_refs):
                gt_bond = gt_bond_by_pair.get(tuple(sorted(gt_atom_refs)))
                if gt_bond is not None:
                    chemical = gt_bond["chemical_type"]
                    oracle_matched_bond_chemistry += 1
        bond_objects.append(
            {
                "id": bond_id,
                "atom_refs": atom_refs,
                "ports": [list(begin_port), list(end_port)],
                "display_type": display,
                "chemical_type": chemical,
                "direction": "NONE",
                "stereo": "STEREONONE",
                "bbox": None,
            }
        )
    scene = {
        "schema_version": SCHEMA_VERSION,
        "molecule_id": molecule_id,
        "units": UNITS,
        "canvas": dict(gt_scene["canvas"]),
        "atom_objects": atom_objects,
        "bond_objects": bond_objects,
    }
    parse_chemscene_v0_2(scene)
    diagnostics = {
        "selected_atom_candidates": len(selected_atoms),
        "selected_atom_candidate_indices": selected_atoms,
        "rescued_endpoint_atom_candidates": len(rescued_atoms),
        "oracle_matched_atoms": len(oracle_atom_by_candidate),
        "selected_bond_candidates": len(selected_bonds),
        "compiled_bonds": len(bond_objects),
        "dropped_duplicate_atoms": dropped_duplicate_atoms,
        "dropped_unselected_endpoint_bonds": dropped_unselected_endpoint,
        "dropped_self_bonds": dropped_self_bond,
        "dropped_duplicate_bonds": dropped_duplicate_bond,
        "bond_chemistry_mode": bond_chemistry_mode,
        "oracle_matched_bond_chemistry": oracle_matched_bond_chemistry,
    }
    return scene, diagnostics


def _write_empty_sdf(path: Path, error: str) -> None:
    path.write_text(
        "\n".join(
            (
                "ChemVector empty diagnostic prediction",
                "  ChemVector",
                "",
                "  0  0  0  0  0  0  0  0  0  0999 V2000",
                "M  END",
                ">  <CHEMVECTOR_STATUS>",
                "invalid_empty_prediction",
                "",
                ">  <CHEMVECTOR_ERROR>",
                str(error),
                "",
                "$$$$",
                "",
            )
        ),
        encoding="utf-8",
    )


def _render_sdf(path: Path, output: Path) -> None:
    mol = Chem.MolFromMolFile(
        str(path), sanitize=False, removeHs=False, strictParsing=False
    )
    if mol is None or mol.GetNumAtoms() == 0:
        image = Image.new("RGB", (768, 768), "white")
        ImageDraw.Draw(image).text((24, 24), "Invalid or empty SDF", fill="black")
        image.save(output)
        return
    try:
        mol.UpdatePropertyCache(strict=False)
        image = Draw.MolToImage(mol, size=(768, 768), kekulize=False)
    except Exception:
        image = Image.new("RGB", (768, 768), "white")
        ImageDraw.Draw(image).text((24, 24), "SDF rendering failed", fill="black")
    image.save(output)


def _summary(records: list[dict]) -> dict:
    count = max(len(records), 1)
    keys = (
        "scene_valid",
        "sdf_valid",
        "atom_count_match",
        "bond_count_match",
        "nonstereo_structure_match",
        "chemical_match",
    )
    result = {"samples": len(records)}
    for key in keys:
        total = sum(bool(record.get(key)) for record in records)
        result[key] = total
        result[f"{key}_rate"] = total / count
    result["failure_categories"] = dict(
        Counter(record.get("failure_category", "none") for record in records)
    )
    result["demoted_aromatic_bonds"] = sum(
        record.get("aromatic_consistency", {}).get(
            "demoted_aromatic_bond_count", 0
        )
        for record in records
    )
    result["samples_with_demoted_aromatic_bonds"] = sum(
        bool(
            record.get("aromatic_consistency", {}).get(
                "demoted_aromatic_bond_count", 0
            )
        )
        for record in records
    )
    result["cleared_noncycle_aromatic_atoms"] = sum(
        record.get("aromatic_consistency", {}).get(
            "cleared_noncycle_aromatic_atom_count", 0
        )
        for record in records
    )
    result["samples_with_cleared_noncycle_aromatic_atoms"] = sum(
        bool(
            record.get("aromatic_consistency", {}).get(
                "cleared_noncycle_aromatic_atom_count", 0
            )
        )
        for record in records
    )
    result["oracle_matched_bond_chemistry"] = sum(
        record.get("diagnostics", {}).get("oracle_matched_bond_chemistry", 0)
        for record in records
    )
    milp_records = [record["joint_graph_milp"] for record in records if "joint_graph_milp" in record]
    if milp_records:
        result["joint_graph_milp_samples"] = len(milp_records)
        result["joint_graph_milp_solver_success"] = sum(
            bool(item["solver_success"]) for item in milp_records
        )
        result["joint_graph_milp_elapsed_seconds"] = sum(
            float(item["elapsed_seconds"]) for item in milp_records
        )
        result["joint_invalid_fallbacks"] = sum(
            bool(record.get("joint_invalid_fallback")) for record in records
        )
        result["checkpoint_invalid_fallbacks"] = sum(
            bool(record.get("checkpoint_invalid_fallback")) for record in records
        )
    result["protocol"] = "predicted-objects-oracle-chemistry-no-ocr"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--data-shard", default=DEFAULT_DATA_SHARD)
    parser.add_argument("--expected-split", default="internal_test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--atom-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-threshold", type=float, default=-0.5)
    parser.add_argument(
        "--aromatic-consistency",
        choices=AROMATIC_CONSISTENCY_MODES,
        default="none",
    )
    parser.add_argument(
        "--bond-chemistry",
        choices=BOND_CHEMISTRY_MODES,
        default="predicted",
    )
    parser.add_argument(
        "--object-presence",
        choices=OBJECT_PRESENCE_MODES,
        default="predicted",
    )
    parser.add_argument(
        "--incidence",
        choices=INCIDENCE_MODES,
        default="predicted",
    )
    parser.add_argument("--endpoint-atom-rescue-margin", type=float, default=0.0)
    parser.add_argument(
        "--joint-graph-decoder", choices=JOINT_GRAPH_DECODERS, default="none"
    )
    parser.add_argument("--joint-endpoint-topk", type=int, default=2)
    parser.add_argument("--joint-incidence-weight", type=float, default=0.5)
    parser.add_argument("--joint-time-limit-seconds", type=float, default=2.0)
    parser.add_argument("--joint-potential-file")
    parser.add_argument(
        "--joint-invalid-fallback",
        choices=JOINT_INVALID_FALLBACKS,
        default="none",
    )
    parser.add_argument("--invalid-fallback-checkpoint")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ChemScene v0.2 Oracle shadow requires CUDA")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    directories = {
        name: output_dir / name
        for name in ("input", "expected_sdf", "expected_render", "scene", "sdf", "render")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    cache_records, manifest = load_cache(args.cache, args.expected_split)
    cache_records = cache_records[: args.samples]
    samples = _load_records(args.data_shard, args.samples, args.expected_split)
    if [record["key"] for record in cache_records] != [sample["key"] for sample in samples]:
        raise ValueError("Proposal cache and source dataset sample keys differ")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["proposal_config"] != manifest["proposal_config"]:
        raise ValueError("Checkpoint and shadow proposal cache configurations differ")
    incidence_config = SpatialIncidenceConfig(**payload["config"]["incidence"])
    config = ProposalObjectSetConfig(
        incidence=incidence_config,
        hidden_dim=payload["config"]["hidden_dim"],
    )
    device = torch.device("cuda")
    set_head = ProposalObjectSetHead(config).to(device)
    set_head.load_state_dict(payload["head"], strict=True)
    set_head.eval()
    fallback_set_head = None
    fallback_payload = None
    if args.invalid_fallback_checkpoint:
        fallback_payload = torch.load(
            args.invalid_fallback_checkpoint, map_location="cpu", weights_only=False
        )
        if fallback_payload["proposal_config"] != manifest["proposal_config"]:
            raise ValueError("Fallback checkpoint and proposal cache differ")
        fallback_incidence = SpatialIncidenceConfig(
            **fallback_payload["config"]["incidence"]
        )
        fallback_config = ProposalObjectSetConfig(
            incidence=fallback_incidence,
            hidden_dim=fallback_payload["config"]["hidden_dim"],
        )
        fallback_set_head = ProposalObjectSetHead(fallback_config).to(device)
        fallback_set_head.load_state_dict(fallback_payload["head"], strict=True)
        fallback_set_head.eval()
    joint_config = JointGraphMilpConfig(
        endpoint_topk=args.joint_endpoint_topk,
        incidence_weight=args.joint_incidence_weight,
        time_limit_seconds=args.joint_time_limit_seconds,
    )
    joint_potential_payload = None
    if args.joint_potential_file:
        joint_potential_payload = json.loads(
            Path(args.joint_potential_file).read_text(encoding="utf-8")
        )
        potentials = joint_potential_payload["potentials"]
        joint_config = replace(
            joint_config,
            atom_presence_weight=float(potentials["atom_presence_weight"]),
            atom_bias=float(potentials["atom_bias"]),
            bond_presence_weight=float(potentials["bond_presence_weight"]),
            bond_bias=float(potentials["bond_bias"]),
            incidence_weight=float(potentials["incidence_weight"]),
        )
    records = []
    for start in range(0, len(samples), args.batch_size):
        source_batch = samples[start : start + args.batch_size]
        cached_batch = cache_records[start : start + args.batch_size]
        batch = stack_cache_records(cached_batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = _forward(set_head, batch)
            fallback_output = (
                _forward(fallback_set_head, batch)
                if fallback_set_head is not None
                else None
            )
        for local_index, (sample, cached) in enumerate(zip(source_batch, cached_batch)):
            key = _safe_name(sample["key"])
            molecule_id = str(sample["chemmol"]["molecule_id"])
            sample["image"].convert("RGB").save(directories["input"] / f"{key}.png")
            gt_scene = build_chemscene_v0_2(
                sample["chemmol"], sample["render"], sample.get("layers")
            )
            expected = mol_from_chemscene_v0_2(gt_scene)
            expected_sdf = directories["expected_sdf"] / f"{key}.sdf"
            write_prediction_sdf(expected, expected_sdf, sanitize=True)
            _render_sdf(expected_sdf, directories["expected_render"] / f"{key}.png")
            record = {
                "key": sample["key"],
                "molecule_id": molecule_id,
                "protocol": "predicted-objects-oracle-chemistry-no-ocr",
                "scene_valid": False,
                "sdf_valid": False,
                "atom_count_match": False,
                "bond_count_match": False,
                "nonstereo_structure_match": False,
                "chemical_match": False,
            }
            scene = None
            scene_path = directories["scene"] / f"{key}.json"
            try:
                atom_presence, bond_presence, incidence = apply_oracle_object_controls(
                    atom_presence_logits=output["atom_presence_logits"][local_index]
                    .float()
                    .cpu(),
                    bond_presence_logits=output["bond_presence_logits"][local_index]
                    .float()
                    .cpu(),
                    incidence_logits=output["incidence_logits"][local_index]
                    .float()
                    .cpu(),
                    batch=batch,
                    batch_index=local_index,
                    object_presence_mode=args.object_presence,
                    incidence_mode=args.incidence,
                )
                independent_atom_presence = atom_presence.clone()
                independent_bond_presence = bond_presence.clone()
                independent_incidence = incidence.clone()
                joint_diagnostics = None
                if args.joint_graph_decoder == "milp":
                    atom_presence, bond_presence, incidence, joint_diagnostics = (
                        apply_joint_graph_milp(
                            atom_presence_logits=atom_presence,
                            atom_mask=cached["atom_mask"],
                            bond_presence_logits=bond_presence,
                            bond_mask=cached["bond_mask"],
                            incidence_logits=incidence,
                            atom_threshold=args.atom_threshold,
                            bond_threshold=args.bond_threshold,
                            config=joint_config,
                        )
                    )
                scene, diagnostics = build_oracle_chemistry_shadow_scene(
                    molecule_id=molecule_id,
                    gt_scene=gt_scene,
                    atom_geometry=cached["atom_geometry"].float(),
                    atom_mask=cached["atom_mask"],
                    atom_presence_logits=atom_presence,
                    bond_types=cached["bond_types"].float(),
                    bond_mask=cached["bond_mask"],
                    bond_presence_logits=bond_presence,
                    incidence_logits=incidence,
                    atom_threshold=args.atom_threshold,
                    bond_threshold=args.bond_threshold,
                    bond_chemistry_mode=args.bond_chemistry,
                    endpoint_atom_rescue_margin=args.endpoint_atom_rescue_margin,
                )
                scene, aromatic_diagnostics = apply_aromatic_consistency(
                    scene, args.aromatic_consistency
                )
                record["scene_valid"] = True
                record["diagnostics"] = diagnostics
                if joint_diagnostics is not None:
                    record["joint_graph_milp"] = joint_diagnostics
                record["aromatic_consistency"] = aromatic_diagnostics
                try:
                    predicted = mol_from_chemscene_v0_2(scene)
                except Exception as joint_error:
                    if not (
                        joint_diagnostics is not None
                        and args.joint_invalid_fallback == "independent"
                    ):
                        raise
                    fallback_scene, fallback_diagnostics = (
                        build_oracle_chemistry_shadow_scene(
                            molecule_id=molecule_id,
                            gt_scene=gt_scene,
                            atom_geometry=cached["atom_geometry"].float(),
                            atom_mask=cached["atom_mask"],
                            atom_presence_logits=independent_atom_presence,
                            bond_types=cached["bond_types"].float(),
                            bond_mask=cached["bond_mask"],
                            bond_presence_logits=independent_bond_presence,
                            incidence_logits=independent_incidence,
                            atom_threshold=args.atom_threshold,
                            bond_threshold=args.bond_threshold,
                            bond_chemistry_mode=args.bond_chemistry,
                            endpoint_atom_rescue_margin=(
                                args.endpoint_atom_rescue_margin
                            ),
                        )
                    )
                    fallback_scene, fallback_aromatic = apply_aromatic_consistency(
                        fallback_scene, args.aromatic_consistency
                    )
                    try:
                        fallback_predicted = mol_from_chemscene_v0_2(fallback_scene)
                    except Exception as independent_error:
                        if fallback_output is None:
                            raise joint_error from independent_error
                        checkpoint_atom, checkpoint_bond, checkpoint_incidence = (
                            apply_oracle_object_controls(
                                atom_presence_logits=fallback_output[
                                    "atom_presence_logits"
                                ][local_index]
                                .float()
                                .cpu(),
                                bond_presence_logits=fallback_output[
                                    "bond_presence_logits"
                                ][local_index]
                                .float()
                                .cpu(),
                                incidence_logits=fallback_output["incidence_logits"][
                                    local_index
                                ]
                                .float()
                                .cpu(),
                                batch=batch,
                                batch_index=local_index,
                                object_presence_mode=args.object_presence,
                                incidence_mode=args.incidence,
                            )
                        )
                        checkpoint_independent_atom = checkpoint_atom.clone()
                        checkpoint_independent_bond = checkpoint_bond.clone()
                        checkpoint_independent_incidence = checkpoint_incidence.clone()
                        checkpoint_atom, checkpoint_bond, checkpoint_incidence, _ = (
                            apply_joint_graph_milp(
                                atom_presence_logits=checkpoint_atom,
                                atom_mask=cached["atom_mask"],
                                bond_presence_logits=checkpoint_bond,
                                bond_mask=cached["bond_mask"],
                                incidence_logits=checkpoint_incidence,
                                atom_threshold=args.atom_threshold,
                                bond_threshold=args.bond_threshold,
                                config=joint_config,
                            )
                        )
                        fallback_scene, fallback_diagnostics = (
                            build_oracle_chemistry_shadow_scene(
                                molecule_id=molecule_id,
                                gt_scene=gt_scene,
                                atom_geometry=cached["atom_geometry"].float(),
                                atom_mask=cached["atom_mask"],
                                atom_presence_logits=checkpoint_atom,
                                bond_types=cached["bond_types"].float(),
                                bond_mask=cached["bond_mask"],
                                bond_presence_logits=checkpoint_bond,
                                incidence_logits=checkpoint_incidence,
                                atom_threshold=args.atom_threshold,
                                bond_threshold=args.bond_threshold,
                                bond_chemistry_mode=args.bond_chemistry,
                                endpoint_atom_rescue_margin=(
                                    args.endpoint_atom_rescue_margin
                                ),
                            )
                        )
                        fallback_scene, fallback_aromatic = (
                            apply_aromatic_consistency(
                                fallback_scene, args.aromatic_consistency
                            )
                        )
                        try:
                            fallback_predicted = mol_from_chemscene_v0_2(
                                fallback_scene
                            )
                        except Exception:
                            fallback_scene, fallback_diagnostics = (
                                build_oracle_chemistry_shadow_scene(
                                    molecule_id=molecule_id,
                                    gt_scene=gt_scene,
                                    atom_geometry=cached["atom_geometry"].float(),
                                    atom_mask=cached["atom_mask"],
                                    atom_presence_logits=(
                                        checkpoint_independent_atom
                                    ),
                                    bond_types=cached["bond_types"].float(),
                                    bond_mask=cached["bond_mask"],
                                    bond_presence_logits=(
                                        checkpoint_independent_bond
                                    ),
                                    incidence_logits=(
                                        checkpoint_independent_incidence
                                    ),
                                    atom_threshold=args.atom_threshold,
                                    bond_threshold=args.bond_threshold,
                                    bond_chemistry_mode=args.bond_chemistry,
                                    endpoint_atom_rescue_margin=(
                                        args.endpoint_atom_rescue_margin
                                    ),
                                )
                            )
                            fallback_scene, fallback_aromatic = (
                                apply_aromatic_consistency(
                                    fallback_scene, args.aromatic_consistency
                                )
                            )
                            fallback_predicted = mol_from_chemscene_v0_2(
                                fallback_scene
                            )
                        record["checkpoint_invalid_fallback"] = True
                        record["checkpoint_invalid_error"] = str(independent_error)
                    scene = fallback_scene
                    predicted = fallback_predicted
                    record["diagnostics"] = fallback_diagnostics
                    record["aromatic_consistency"] = fallback_aromatic
                    record["joint_invalid_fallback"] = True
                    record["joint_invalid_error"] = str(joint_error)
                scene_path.write_text(
                    json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                record.update(
                    {
                        "sdf_valid": True,
                        "predicted_atoms": predicted.GetNumAtoms(),
                        "predicted_bonds": predicted.GetNumBonds(),
                        "expected_atoms": expected.GetNumAtoms(),
                        "expected_bonds": expected.GetNumBonds(),
                        "atom_count_match": predicted.GetNumAtoms() == expected.GetNumAtoms(),
                        "bond_count_match": predicted.GetNumBonds() == expected.GetNumBonds(),
                        "nonstereo_structure_match": sdf_graph_match(
                            expected, predicted, use_chirality=False
                        ),
                        "chemical_match": sdf_graph_match(
                            expected, predicted, use_chirality=True
                        ),
                        "fingerprint_similarity": molecular_similarity(expected, predicted),
                        "expected_formula": rdMolDescriptors.CalcMolFormula(expected),
                        "actual_formula": rdMolDescriptors.CalcMolFormula(predicted),
                    }
                )
                prefix = (
                    "correct" if record["nonstereo_structure_match"] else "wrong"
                )
                prediction_sdf = directories["sdf"] / f"{prefix}_{key}.sdf"
                write_prediction_sdf(predicted, prediction_sdf, sanitize=True)
            except Exception as error:
                if scene is not None and not scene_path.exists():
                    scene_path.write_text(
                        json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                record["error"] = str(error)
                record["failure_category"] = failure_category(error)
                prediction_sdf = directories["sdf"] / f"invalid_{key}.sdf"
                if scene is not None:
                    try:
                        sequence = chemsvg_from_chemscene_v0_1(
                            chemscene_v0_1_from_v0_2(scene)
                        )
                        atoms, bonds = write_invalid_prediction_sdf(sequence, prediction_sdf)
                        record["predicted_atoms"] = atoms
                        record["predicted_bonds"] = bonds
                    except Exception as diagnostic_error:
                        record["diagnostic_sdf_error"] = str(diagnostic_error)
                        _write_empty_sdf(prediction_sdf, str(error))
                else:
                    _write_empty_sdf(prediction_sdf, str(error))
            record["prediction_sdf_file"] = str(prediction_sdf.relative_to(output_dir))
            prediction_render = directories["render"] / f"{key}.png"
            _render_sdf(prediction_sdf, prediction_render)
            record["prediction_render_file"] = str(
                prediction_render.relative_to(output_dir)
            )
            records.append(record)
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = _summary(records)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_step": payload["global_step"],
            "atom_threshold": args.atom_threshold,
            "bond_threshold": args.bond_threshold,
            "aromatic_consistency": args.aromatic_consistency,
            "bond_chemistry": args.bond_chemistry,
            "object_presence": args.object_presence,
            "incidence": args.incidence,
            "endpoint_atom_rescue_margin": args.endpoint_atom_rescue_margin,
            "joint_graph_decoder": args.joint_graph_decoder,
            "joint_endpoint_topk": args.joint_endpoint_topk,
            "joint_incidence_weight": args.joint_incidence_weight,
            "joint_time_limit_seconds": args.joint_time_limit_seconds,
            "joint_invalid_fallback": args.joint_invalid_fallback,
            "invalid_fallback_checkpoint": args.invalid_fallback_checkpoint,
            "joint_potential_file": args.joint_potential_file,
            "joint_potentials": {
                "atom_presence_weight": joint_config.atom_presence_weight,
                "atom_bias": joint_config.atom_bias,
                "bond_presence_weight": joint_config.bond_presence_weight,
                "bond_bias": joint_config.bond_bias,
                "incidence_weight": joint_config.incidence_weight,
            },
            "oracle_fields": [
                "element",
                "formal_charge",
                "isotope",
                "aromatic",
                "explicit_h_count",
                "no_implicit",
                "radical_electrons",
                "canonical_chiral_tag",
                "text_value",
            ],
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
