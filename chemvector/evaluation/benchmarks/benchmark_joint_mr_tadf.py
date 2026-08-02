#!/usr/bin/env python3
"""Benchmark the v0.9.63 joint object shadow on MR-TADF images."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import time

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import torch

from Data.sdf.chemscene_v0_1 import chemsvg_from_chemscene_v0_1
from Data.sdf.chemscene_v0_2 import chemscene_v0_1_from_v0_2, mol_from_chemscene_v0_2
from chemvector.benchmark_mr_tadf import (
    ensure_benchmark_config,
    failure_category,
    molecular_similarity,
    sha256_file,
    write_invalid_prediction_sdf,
    write_prediction_sdf,
)
from chemvector.benchmark_public_ocsr import sdf_graph_match
from chemvector.bond_equivalence import collapse_incidence_pair_bonds
from chemvector.atom_pair_bond_group import (
    AtomPairBondGroupConfig,
    AtomPairBondGroupHead,
    build_atom_pair_groups,
    collapse_atom_pair_group_outputs,
)
from chemvector.atom_pair_graph_context import (
    load_atom_pair_graph_context_head,
    refine_atom_pair_group_output,
)
from chemvector.joint_graph_milp import JointGraphMilpConfig, apply_joint_graph_milp
from chemvector.line_segment_group_rescue import (
    LineSegmentGroupRescueConfig,
    rescue_disconnected_atom_pair_groups,
)
from chemvector.line_segment_rescue import detect_atom_pair_line_candidates
from chemvector.line_atom_pair_head import (
    build_line_atom_pair_features,
    load_line_atom_pair_head,
)
from chemvector.line_atom_pair_inference import append_line_atom_pair_proposals
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
    fuse_predicted_atom_proposals,
)
from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.shadow_chemscene_v02_oracle import (
    _render_sdf,
    _safe_name,
    _write_empty_sdf,
    apply_aromatic_consistency,
    build_oracle_chemistry_shadow_scene,
)
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.spatial_atom_object import SpatialAtomObjectConfig, SpatialAtomObjectDetector
from chemvector.train_atom_bond_scene import _image_tensor
from chemvector.train_spatial_incidence import (
    DEFAULT_ATOM,
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    forward_frozen_bond_head,
    load_frozen_components,
)


DEFAULT_PRIMARY = (
    "outputs/checkpoints/chemvector-v0962-representable-hard-replay-step1500/best.pt"
)
DEFAULT_FALLBACK = (
    "outputs/checkpoints/chemvector-v0949-cached-proposal-object-set-train8974-step3000/"
    "best.pt"
)
DEFAULT_MANIFEST = "Data/benchmarks/mr_tadf_v1/manifest.csv"
DEFAULT_OUTPUT = "outputs/evaluations/chemvector-v0963-mr-tadf-v1"


class SceneCompilationFailure(RuntimeError):
    def __init__(self, message: str, scene: dict, diagnostics: dict) -> None:
        super().__init__(message)
        self.scene = scene
        self.diagnostics = diagnostics


def _load_set_head(path: str, device: torch.device) -> tuple[ProposalObjectSetHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = dict(payload["config"])
    config_payload["incidence"] = SpatialIncidenceConfig(
        **config_payload["incidence"]
    )
    config = ProposalObjectSetConfig(
        **config_payload,
    )
    model = ProposalObjectSetHead(config).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, payload


def _load_atom_head(path: str, device: torch.device) -> SpatialAtomObjectDetector:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SpatialAtomObjectDetector(
        SpatialAtomObjectConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model


def _load_atom_pair_group_head(
    path: str, device: torch.device
) -> tuple[AtomPairBondGroupHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = AtomPairBondGroupHead(
        AtomPairBondGroupConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["head"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


def _gt_scene(annotation: dict, mol: Chem.Mol) -> dict:
    atoms = []
    for item in annotation["atoms"]:
        atom_id = int(item["atom_id"])
        atom = mol.GetAtomWithIdx(atom_id)
        element = atom.GetSymbol()
        role = "IMPLICIT" if element == "C" else "TEXT"
        center = [round(float(item["image_x"]) / 768.0 * 1023), round(float(item["image_y"]) / 768.0 * 1023)]
        atoms.append(
            {
                "id": atom_id,
                "anchor": center,
                "role": role,
                "chemical": {
                    "element": element,
                    "formal_charge": atom.GetFormalCharge(),
                    "isotope": atom.GetIsotope(),
                    "aromatic": atom.GetIsAromatic(),
                    "explicit_h_count": atom.GetNumExplicitHs(),
                    "no_implicit": atom.GetNoImplicit(),
                    "radical_electrons": atom.GetNumRadicalElectrons(),
                    "canonical_chiral_tag": str(atom.GetChiralTag()),
                },
                "text": (
                    {
                        "value": element,
                        "center": center,
                        "bbox": [center[0], center[1], center[0], center[1]],
                    }
                    if role == "TEXT"
                    else None
                ),
                "incident_ports": [],
            }
        )
    bonds = [
        {
            "id": int(item["bond_id"]),
            "atom_refs": [int(item["begin_atom"]), int(item["end_atom"])],
            "chemical_type": str(item["bond_type"]),
        }
        for item in annotation["bonds"]
    ]
    return {
        "schema_version": "chemscene-0.2",
        "molecule_id": str(annotation["molecule_id"]),
        "units": "q1023",
        "canvas": {"width": int(annotation["width"]), "height": int(annotation["height"])},
        "atom_objects": atoms,
        "bond_objects": bonds,
    }


def _attempt_scene(
    *,
    molecule_id: str,
    gt_scene: dict,
    proposals: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    batch_index: int,
    atom_threshold: float,
    bond_threshold: float,
    joint: bool,
    joint_config: JointGraphMilpConfig,
    semantic_bond_groups: bool = False,
) -> tuple[dict, Chem.Mol, dict]:
    atom_logits = output["atom_presence_logits"][batch_index].float().cpu()
    bond_logits = output["bond_presence_logits"][batch_index].float().cpu()
    incidence_logits = output["incidence_logits"][batch_index].float().cpu()
    bond_mask = proposals["bond_mask"][batch_index].cpu()
    diagnostics = {"joint": joint}
    if semantic_bond_groups:
        bond_logits, bond_mask, grouping = collapse_incidence_pair_bonds(
            bond_logits,
            bond_mask,
            incidence_logits,
        )
        diagnostics["semantic_bond_groups"] = grouping
    if joint:
        atom_logits, bond_logits, incidence_logits, milp = apply_joint_graph_milp(
            atom_presence_logits=atom_logits,
            atom_mask=proposals["atom_mask"][batch_index].cpu(),
            bond_presence_logits=bond_logits,
            bond_mask=bond_mask,
            incidence_logits=incidence_logits,
            atom_threshold=atom_threshold,
            bond_threshold=bond_threshold,
            config=joint_config,
        )
        diagnostics["milp"] = milp
    scene, compile_diagnostics = build_oracle_chemistry_shadow_scene(
        molecule_id=molecule_id,
        gt_scene=gt_scene,
        atom_geometry=proposals["atom_geometry"][batch_index].float().cpu(),
        atom_mask=proposals["atom_mask"][batch_index].cpu(),
        atom_presence_logits=atom_logits,
        bond_types=proposals["bond_types"][batch_index].float().cpu(),
        bond_mask=bond_mask,
        bond_presence_logits=bond_logits,
        incidence_logits=incidence_logits,
        atom_threshold=atom_threshold,
        bond_threshold=bond_threshold,
    )
    scene, aromatic = apply_aromatic_consistency(
        scene, "demote-noncycle-clear-atoms"
    )
    diagnostics["compile"] = compile_diagnostics
    diagnostics["aromatic"] = aromatic
    try:
        predicted = mol_from_chemscene_v0_2(scene)
    except Exception as error:
        raise SceneCompilationFailure(str(error), scene, diagnostics) from error
    return scene, predicted, diagnostics


def _summary(records: list[dict]) -> dict:
    count = max(len(records), 1)
    result = {"samples": len(records)}
    for key in (
        "sdf_valid",
        "atom_count_match",
        "bond_count_match",
        "nonstereo_structure_match",
        "chemical_match",
    ):
        total = sum(bool(record.get(key)) for record in records)
        result[key] = total
        result[f"{key}_rate"] = total / count
    result["failure_categories"] = dict(
        Counter(record.get("failure_category", "none") for record in records)
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["difficulty"]].append(record)
    result["difficulty_rates"] = {
        name: {
            "samples": len(rows),
            "sdf_valid_rate": sum(bool(row.get("sdf_valid")) for row in rows) / len(rows),
            "nonstereo_structure_match_rate": sum(bool(row.get("nonstereo_structure_match")) for row in rows) / len(rows),
            "chemical_match_rate": sum(bool(row.get("chemical_match")) for row in rows) / len(rows),
        }
        for name, rows in sorted(groups.items())
    }
    result["decode_paths"] = dict(Counter(record.get("decode_path", "invalid") for record in records))
    result["average_seconds"] = sum(float(record["seconds"]) for record in records) / count
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default="v0.9.63")
    parser.add_argument("--benchmark-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-checkpoint", default=DEFAULT_PRIMARY)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--rescue-atom-checkpoint")
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument(
        "--proposal-bond-checkpoint",
        help="Optional single- or multi-scale Bond proposal checkpoint",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maximum-atoms", type=int, default=128)
    parser.add_argument("--primary-atom-capacity", type=int, default=128)
    parser.add_argument("--rescue-atom-capacity", type=int, default=16)
    parser.add_argument("--atom-fusion-radius-px", type=float, default=6.0)
    parser.add_argument("--maximum-bonds", type=int, default=192)
    parser.add_argument("--atom-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-threshold", type=float, default=-0.5)
    parser.add_argument("--endpoint-topk", type=int, default=2)
    parser.add_argument("--incidence-weight", type=float, default=0.5)
    parser.add_argument("--semantic-bond-groups", action="store_true")
    parser.add_argument("--atom-pair-group-checkpoint")
    parser.add_argument("--atom-pair-graph-context-checkpoint")
    parser.add_argument("--atom-pair-group-threshold", type=float, default=0.0)
    parser.add_argument("--line-segment-group-rescue", action="store_true")
    parser.add_argument("--line-rescue-minimum-support", type=int, default=2)
    parser.add_argument("--line-atom-pair-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    if not torch.cuda.is_available():
        raise RuntimeError("MR-TADF joint benchmark requires CUDA")
    manifest = Path(args.benchmark_manifest)
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    rows = rows[args.start_index :]
    if args.samples:
        rows = rows[: args.samples]
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    directories = {
        name: output_dir / name
        for name in ("input", "expected_sdf", "expected_render", "scene", "sdf", "render")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    benchmark_config = {
        "benchmark_manifest": str(manifest),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "benchmark_type": "MR-TADF v1 main structure benchmark",
        "model_version": args.model_version,
        "inference_protocol": "joint object MILP with Oracle atom chemistry",
        "checkpoint": args.primary_checkpoint,
        "checkpoint_sha256": sha256_file(Path(args.primary_checkpoint)),
        "fallback_checkpoint": args.fallback_checkpoint,
        "fallback_checkpoint_sha256": sha256_file(Path(args.fallback_checkpoint)),
        "grounded_graph_checkpoint": args.grounded_graph_checkpoint,
        "atom_checkpoint": args.atom_checkpoint,
        "rescue_atom_checkpoint": args.rescue_atom_checkpoint,
        "bond_checkpoint": args.proposal_bond_checkpoint or args.bond_checkpoint,
        "maximum_atoms": args.maximum_atoms,
        "primary_atom_capacity": args.primary_atom_capacity,
        "rescue_atom_capacity": args.rescue_atom_capacity,
        "atom_fusion_radius_px": args.atom_fusion_radius_px,
        "maximum_bonds": args.maximum_bonds,
        "atom_threshold": args.atom_threshold,
        "bond_threshold": args.bond_threshold,
        "endpoint_topk": args.endpoint_topk,
        "incidence_weight": args.incidence_weight,
        "semantic_bond_groups": args.semantic_bond_groups,
        "atom_pair_group_checkpoint": args.atom_pair_group_checkpoint,
        "atom_pair_graph_context_checkpoint": args.atom_pair_graph_context_checkpoint,
        "atom_pair_group_threshold": args.atom_pair_group_threshold,
        "line_segment_group_rescue": args.line_segment_group_rescue,
        "line_rescue_minimum_support": args.line_rescue_minimum_support,
        "line_atom_pair_checkpoint": args.line_atom_pair_checkpoint,
        "oracle_atom_chemistry": True,
        "primary_comparison": "SDF attributed graph isomorphism",
        "start_index": args.start_index,
        "samples": len(rows),
    }
    ensure_benchmark_config(output_dir / "benchmark_config.json", benchmark_config)
    device = torch.device("cuda")
    vision, atom_head, bond_head, _ = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.proposal_bond_checkpoint or args.bond_checkpoint),
        device,
    )
    rescue_atom_head = (
        _load_atom_head(args.rescue_atom_checkpoint, device)
        if args.rescue_atom_checkpoint
        else None
    )
    if rescue_atom_head is not None:
        if args.maximum_atoms < args.primary_atom_capacity + args.rescue_atom_capacity:
            raise ValueError("maximum-atoms is too small for fused Atom proposals")
        if rescue_atom_head.config.input_channels != atom_head.config.input_channels:
            raise ValueError("Primary and rescue Atom heads use different channels")
    primary, primary_payload = _load_set_head(args.primary_checkpoint, device)
    fallback, fallback_payload = _load_set_head(args.fallback_checkpoint, device)
    atom_pair_group_head = None
    atom_pair_group_payload = None
    if args.atom_pair_group_checkpoint:
        atom_pair_group_head, atom_pair_group_payload = _load_atom_pair_group_head(
            args.atom_pair_group_checkpoint, device
        )
    line_atom_pair_head = None
    line_atom_pair_payload = None
    if args.line_atom_pair_checkpoint:
        line_atom_pair_head, line_atom_pair_payload = load_line_atom_pair_head(
            args.line_atom_pair_checkpoint, device
        )
        if atom_pair_group_head is None or not args.line_segment_group_rescue:
            raise ValueError(
                "Line AtomPair benchmark requires group head and line segment rescue"
            )
    atom_pair_graph_context_head = None
    atom_pair_graph_context_payload = None
    if args.atom_pair_graph_context_checkpoint:
        if atom_pair_group_head is None:
            raise ValueError("AtomPair graph context requires the AtomPair group head")
        atom_pair_graph_context_head, atom_pair_graph_context_payload = (
            load_atom_pair_graph_context_head(
                args.atom_pair_graph_context_checkpoint, device
            )
        )
        if Path(atom_pair_graph_context_payload["group_checkpoint"]) != Path(
            args.atom_pair_group_checkpoint
        ):
            raise ValueError("AtomPair graph context was trained for another group head")
    proposal_config = PredictedProposalConfig(
        maximum_atoms=args.maximum_atoms,
        maximum_bonds=args.maximum_bonds,
        atom_threshold=0.01,
        bond_threshold=0.01,
    )
    joint_config = JointGraphMilpConfig(
        endpoint_topk=args.endpoint_topk,
        incidence_weight=args.incidence_weight,
        time_limit_seconds=2.0,
    )
    records = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images = [Image.open(root / row["image_file"]).convert("RGB") for row in batch_rows]
        pixels = torch.stack([_image_tensor(image) for image in images]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(features)
            rescue_atom_output = (
                rescue_atom_head(features) if rescue_atom_head is not None else None
            )
            bond_output = forward_frozen_bond_head(vision, bond_head, features)
            atom_proposals = (
                fuse_predicted_atom_proposals(
                    atom_output,
                    rescue_atom_output,
                    proposal_config,
                    primary_capacity=args.primary_atom_capacity,
                    rescue_capacity=args.rescue_atom_capacity,
                    dedup_radius_px=args.atom_fusion_radius_px,
                )
                if rescue_atom_output is not None
                else decode_predicted_atom_proposals(atom_output, proposal_config)
            )
            proposals = {
                **atom_proposals,
                **decode_predicted_bond_proposals(bond_output, proposal_config),
            }
            primary_output = primary(
                proposals["atom_features"], proposals["atom_geometry"], proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"], proposals["bond_types"], proposals["bond_scores"], proposals["bond_mask"],
            )
            fallback_output = fallback(
                proposals["atom_features"], proposals["atom_geometry"], proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"], proposals["bond_types"], proposals["bond_scores"], proposals["bond_mask"],
            )
            primary_proposals = fallback_proposals = proposals
            primary_grouping = fallback_grouping = None
            primary_first_line = fallback_first_line = None
            if atom_pair_group_head is not None:
                primary_groups = build_atom_pair_groups(primary_output, proposals)
                primary_group_output = atom_pair_group_head(primary_groups["features"])
                primary_graph_context = None
                if atom_pair_graph_context_head is not None:
                    primary_group_output, primary_graph_context = (
                        refine_atom_pair_group_output(
                            atom_pair_graph_context_head,
                            primary_group_output,
                            primary_groups,
                            batch_size=len(images),
                            residual_scale=float(
                                atom_pair_graph_context_payload.get(
                                    "residual_scale", 1.0
                                )
                            ),
                            base_positive_protection_threshold=(
                                atom_pair_graph_context_payload.get(
                                    "base_positive_protection_threshold"
                                )
                            ),
                        )
                    )
                primary_line_rescue = None
                if args.line_segment_group_rescue:
                    primary_line_candidates = []
                    for batch_index, image in enumerate(images):
                        predicted_pixels = (
                            proposals["atom_geometry"][batch_index, :, :2]
                            .float()
                            .cpu()
                            .numpy()
                            * 768.0
                        )
                        selected_atoms = (
                            proposals["atom_mask"][batch_index]
                            & (
                                primary_output["atom_presence_logits"][batch_index]
                                >= args.atom_threshold
                            )
                        ).cpu().numpy()
                        primary_line_candidates.append(
                            detect_atom_pair_line_candidates(
                                image,
                                predicted_pixels,
                                atom_mask=selected_atoms,
                            )
                        )
                    primary_group_output, primary_line_rescue = (
                        rescue_disconnected_atom_pair_groups(
                            primary_group_output,
                            primary_groups,
                            atom_presence_logits=primary_output["atom_presence_logits"],
                            atom_mask=proposals["atom_mask"],
                            line_candidates=primary_line_candidates,
                            atom_threshold=args.atom_threshold,
                            group_threshold=args.atom_pair_group_threshold,
                            config=LineSegmentGroupRescueConfig(
                                minimum_support_segments=args.line_rescue_minimum_support
                            ),
                        )
                    )
                primary_proposals, primary_output, primary_grouping = (
                    collapse_atom_pair_group_outputs(
                        proposals, primary_output, primary_group_output, primary_groups
                    )
                )
                if primary_graph_context is not None:
                    for grouping in primary_grouping:
                        grouping["graph_context"] = primary_graph_context
                if primary_line_rescue is not None:
                    for grouping, rescue in zip(
                        primary_grouping, primary_line_rescue
                    ):
                        grouping["line_segment_rescue"] = rescue
                if line_atom_pair_head is not None:
                    existing = [set(items) for items in primary_groups["atom_pairs"]]
                    primary_line_only = [
                        tuple(
                            candidate
                            for candidate in candidates
                            if len(candidate.support_segment_indices)
                            >= args.line_rescue_minimum_support
                            and candidate.atom_pair not in existing[batch_index]
                        )
                        for batch_index, candidates in enumerate(primary_line_candidates)
                    ]
                    line_features, line_sample_indices = build_line_atom_pair_features(
                        bond_output["object_features"].float(),
                        proposals["atom_features"].float(),
                        proposals["atom_geometry"][..., :2].float(),
                        primary_line_only,
                        config=line_atom_pair_head.config,
                    )
                    line_output = line_atom_pair_head(line_features)
                    line_proposals, line_model_output, line_diagnostics = (
                        append_line_atom_pair_proposals(
                            primary_proposals,
                            primary_output,
                            line_output,
                            primary_line_only,
                            line_sample_indices,
                            line_threshold=float(line_atom_pair_payload["threshold"]),
                        )
                    )
                    line_grouping = [dict(item) for item in primary_grouping]
                    for grouping, diagnostics in zip(line_grouping, line_diagnostics):
                        grouping["line_atom_pair"] = diagnostics
                    primary_first_line = (
                        line_proposals,
                        line_model_output,
                        line_grouping,
                    )
                fallback_groups = build_atom_pair_groups(fallback_output, proposals)
                fallback_group_output = atom_pair_group_head(fallback_groups["features"])
                fallback_graph_context = None
                if atom_pair_graph_context_head is not None:
                    fallback_group_output, fallback_graph_context = (
                        refine_atom_pair_group_output(
                            atom_pair_graph_context_head,
                            fallback_group_output,
                            fallback_groups,
                            batch_size=len(images),
                            residual_scale=float(
                                atom_pair_graph_context_payload.get(
                                    "residual_scale", 1.0
                                )
                            ),
                            base_positive_protection_threshold=(
                                atom_pair_graph_context_payload.get(
                                    "base_positive_protection_threshold"
                                )
                            ),
                        )
                    )
                fallback_line_rescue = None
                if args.line_segment_group_rescue:
                    fallback_line_candidates = []
                    for batch_index, image in enumerate(images):
                        predicted_pixels = (
                            proposals["atom_geometry"][batch_index, :, :2]
                            .float()
                            .cpu()
                            .numpy()
                            * 768.0
                        )
                        selected_atoms = (
                            proposals["atom_mask"][batch_index]
                            & (
                                fallback_output["atom_presence_logits"][batch_index]
                                >= args.atom_threshold
                            )
                        ).cpu().numpy()
                        fallback_line_candidates.append(
                            detect_atom_pair_line_candidates(
                                image,
                                predicted_pixels,
                                atom_mask=selected_atoms,
                            )
                        )
                    fallback_group_output, fallback_line_rescue = (
                        rescue_disconnected_atom_pair_groups(
                            fallback_group_output,
                            fallback_groups,
                            atom_presence_logits=fallback_output["atom_presence_logits"],
                            atom_mask=proposals["atom_mask"],
                            line_candidates=fallback_line_candidates,
                            atom_threshold=args.atom_threshold,
                            group_threshold=args.atom_pair_group_threshold,
                            config=LineSegmentGroupRescueConfig(
                                minimum_support_segments=args.line_rescue_minimum_support
                            ),
                        )
                    )
                fallback_proposals, fallback_output, fallback_grouping = (
                    collapse_atom_pair_group_outputs(
                        proposals, fallback_output, fallback_group_output, fallback_groups
                    )
                )
                if fallback_graph_context is not None:
                    for grouping in fallback_grouping:
                        grouping["graph_context"] = fallback_graph_context
                if fallback_line_rescue is not None:
                    for grouping, rescue in zip(
                        fallback_grouping, fallback_line_rescue
                    ):
                        grouping["line_segment_rescue"] = rescue
                if line_atom_pair_head is not None:
                    existing = [set(items) for items in fallback_groups["atom_pairs"]]
                    fallback_line_only = [
                        tuple(
                            candidate
                            for candidate in candidates
                            if len(candidate.support_segment_indices)
                            >= args.line_rescue_minimum_support
                            and candidate.atom_pair not in existing[batch_index]
                        )
                        for batch_index, candidates in enumerate(fallback_line_candidates)
                    ]
                    line_features, line_sample_indices = build_line_atom_pair_features(
                        bond_output["object_features"].float(),
                        proposals["atom_features"].float(),
                        proposals["atom_geometry"][..., :2].float(),
                        fallback_line_only,
                        config=line_atom_pair_head.config,
                    )
                    line_output = line_atom_pair_head(line_features)
                    line_proposals, line_model_output, line_diagnostics = (
                        append_line_atom_pair_proposals(
                            fallback_proposals,
                            fallback_output,
                            line_output,
                            fallback_line_only,
                            line_sample_indices,
                            line_threshold=float(line_atom_pair_payload["threshold"]),
                        )
                    )
                    line_grouping = [dict(item) for item in fallback_grouping]
                    for grouping, diagnostics in zip(line_grouping, line_diagnostics):
                        grouping["line_atom_pair"] = diagnostics
                    fallback_first_line = (
                        line_proposals,
                        line_model_output,
                        line_grouping,
                    )
        for index, (row, image) in enumerate(zip(batch_rows, images)):
            started = time.perf_counter()
            key = _safe_name(row["molecule_id"])
            image.save(directories["input"] / f"{key}.png")
            expected_path = root / row["sdf_file"]
            expected = Chem.MolFromMolFile(str(expected_path), sanitize=True, removeHs=False)
            if expected is None:
                raise ValueError(f"Cannot read MR-TADF ground truth: {expected_path}")
            expected_output = directories["expected_sdf"] / f"{key}.sdf"
            shutil.copy2(expected_path, expected_output)
            _render_sdf(expected_output, directories["expected_render"] / f"{key}.png")
            annotation = json.loads((root / row["annotation_file"]).read_text())
            gt_scene = _gt_scene(annotation, expected)
            record = {
                "key": row["molecule_id"],
                "molecule_id": row["molecule_id"],
                "difficulty": row["difficulty"],
                "sdf_valid": False,
                "atom_count_match": False,
                "bond_count_match": False,
                "nonstereo_structure_match": False,
                "chemical_match": False,
            }
            scene = None
            predicted = None
            errors = []
            failed_diagnostics = None
            attempts = []
            if primary_first_line is not None:
                attempts.extend(
                    (
                        ("first_line_primary_joint", *primary_first_line, True),
                        ("first_line_primary_independent", *primary_first_line, False),
                    )
                )
            attempts.extend(
                (
                    ("primary_joint", primary_proposals, primary_output, primary_grouping, True),
                    ("primary_independent", primary_proposals, primary_output, primary_grouping, False),
                )
            )
            if fallback_first_line is not None:
                attempts.extend(
                    (
                        ("first_line_fallback_joint", *fallback_first_line, True),
                        ("first_line_fallback_independent", *fallback_first_line, False),
                    )
                )
            attempts.extend(
                (
                    ("fallback_joint", fallback_proposals, fallback_output, fallback_grouping, True),
                    ("fallback_independent", fallback_proposals, fallback_output, fallback_grouping, False),
                )
            )
            for path_name, attempt_proposals, model_output, grouping, joint in attempts:
                try:
                    scene, predicted, diagnostics = _attempt_scene(
                        molecule_id=row["molecule_id"],
                        gt_scene=gt_scene,
                        proposals=attempt_proposals,
                        output=model_output,
                        batch_index=index,
                        atom_threshold=args.atom_threshold,
                        bond_threshold=(
                            args.atom_pair_group_threshold
                            if atom_pair_group_head is not None
                            else args.bond_threshold
                        ),
                        joint=joint,
                        joint_config=joint_config,
                        semantic_bond_groups=args.semantic_bond_groups,
                    )
                    record["decode_path"] = path_name
                    record["diagnostics"] = diagnostics
                    if grouping is not None:
                        record["diagnostics"]["atom_pair_groups"] = grouping[index]
                    break
                except SceneCompilationFailure as error:
                    scene = error.scene
                    failed_diagnostics = error.diagnostics
                    errors.append(f"{path_name}: {error}")
                except Exception as error:
                    errors.append(f"{path_name}: {error}")
            scene_path = directories["scene"] / f"{key}.json"
            if scene is not None:
                scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
            if predicted is not None:
                record.update(
                    {
                        "sdf_valid": True,
                        "predicted_atoms": predicted.GetNumAtoms(),
                        "predicted_bonds": predicted.GetNumBonds(),
                        "expected_atoms": expected.GetNumAtoms(),
                        "expected_bonds": expected.GetNumBonds(),
                        "atom_count_match": predicted.GetNumAtoms() == expected.GetNumAtoms(),
                        "bond_count_match": predicted.GetNumBonds() == expected.GetNumBonds(),
                        "nonstereo_structure_match": sdf_graph_match(expected, predicted, use_chirality=False),
                        "chemical_match": sdf_graph_match(expected, predicted, use_chirality=True),
                        "fingerprint_similarity": molecular_similarity(expected, predicted),
                        "expected_formula": rdMolDescriptors.CalcMolFormula(expected),
                        "actual_formula": rdMolDescriptors.CalcMolFormula(predicted),
                    }
                )
                prefix = "correct" if record["nonstereo_structure_match"] else "wrong"
                prediction_sdf = directories["sdf"] / f"{prefix}_{key}.sdf"
                write_prediction_sdf(predicted, prediction_sdf, sanitize=True)
            else:
                error = errors[-1] if errors else "All decode paths failed"
                record["error"] = error
                if failed_diagnostics is not None:
                    record["diagnostics"] = failed_diagnostics
                record["failure_category"] = failure_category(ValueError(error))
                prediction_sdf = directories["sdf"] / f"invalid_{key}.sdf"
                if scene is not None:
                    try:
                        sequence = chemsvg_from_chemscene_v0_1(chemscene_v0_1_from_v0_2(scene))
                        write_invalid_prediction_sdf(sequence, prediction_sdf)
                    except Exception:
                        _write_empty_sdf(prediction_sdf, error)
                else:
                    _write_empty_sdf(prediction_sdf, error)
            record["prediction_sdf_file"] = str(prediction_sdf.relative_to(output_dir))
            prediction_render = directories["render"] / f"{key}.png"
            _render_sdf(prediction_sdf, prediction_render)
            record["prediction_render_file"] = str(prediction_render.relative_to(output_dir))
            record["seconds"] = time.perf_counter() - started
            records.append(record)
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    summary = _summary(records)
    summary.update(
        {
            "protocol": f"{args.model_version}-joint-object-oracle-atom-chemistry",
            "primary_checkpoint": args.primary_checkpoint,
            "primary_step": primary_payload["global_step"],
            "fallback_checkpoint": args.fallback_checkpoint,
            "fallback_step": fallback_payload["global_step"],
            "maximum_atoms": args.maximum_atoms,
            "maximum_bonds": args.maximum_bonds,
            "oracle_atom_chemistry": True,
            "atom_pair_group_checkpoint": args.atom_pair_group_checkpoint,
            "atom_pair_group_samples_seen": (
                atom_pair_group_payload["samples_seen"]
                if atom_pair_group_payload is not None
                else None
            ),
            "atom_pair_graph_context_checkpoint": args.atom_pair_graph_context_checkpoint,
            "atom_pair_graph_context_samples_seen": (
                atom_pair_graph_context_payload["samples_seen"]
                if atom_pair_graph_context_payload is not None
                else None
            ),
            "atom_pair_graph_context_residual_scale": (
                float(atom_pair_graph_context_payload.get("residual_scale", 1.0))
                if atom_pair_graph_context_payload is not None
                else None
            ),
            "atom_pair_graph_context_base_positive_protection_threshold": (
                atom_pair_graph_context_payload.get(
                    "base_positive_protection_threshold"
                )
                if atom_pair_graph_context_payload is not None
                else None
            ),
            "line_segment_group_rescue": args.line_segment_group_rescue,
            "line_rescue_minimum_support": args.line_rescue_minimum_support,
            "line_atom_pair_checkpoint": args.line_atom_pair_checkpoint,
            "line_atom_pair_epoch": (
                line_atom_pair_payload["epoch"]
                if line_atom_pair_payload is not None
                else None
            ),
            "start_index": args.start_index,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
