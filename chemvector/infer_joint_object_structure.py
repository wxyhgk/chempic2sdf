#!/usr/bin/env python3
"""Run joint Atom/Bond object inference with carbon placeholder chemistry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from PIL import Image
from rdkit.Chem import rdMolDescriptors
import torch

from chemvector.benchmark_joint_mr_tadf import (
    DEFAULT_FALLBACK,
    DEFAULT_PRIMARY,
    SceneCompilationFailure,
    _attempt_scene,
    _load_atom_pair_group_head,
    _load_set_head,
)
from chemvector.atom_pair_bond_group import (
    build_atom_pair_groups,
    collapse_atom_pair_group_outputs,
)
from chemvector.atom_pair_graph_context import (
    flatten_atom_pairs,
    load_atom_pair_graph_context_head,
    refine_atom_pair_group_output,
)
from chemvector.atom_pair_deletion_veto import (
    AtomPairDeletionVetoConfig,
    build_atom_pair_relation_features,
)
from chemvector.atom_pair_one_move import (
    apply_atom_pair_action,
    choose_atom_pair_action,
    load_atom_pair_one_move_head,
)
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
from chemvector.benchmark_mr_tadf import write_prediction_sdf
from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.joint_graph_milp import JointGraphMilpConfig
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.shadow_chemscene_v02_oracle import _render_sdf, _write_empty_sdf
from chemvector.train_atom_bond_scene import _image_tensor
from chemvector.train_spatial_incidence import (
    DEFAULT_BOND,
    DEFAULT_GRAPH,
    forward_frozen_bond_head,
    load_frozen_components,
)


DEFAULT_ATOM = (
    "outputs/checkpoints/chemvector-v0967-atom-object-full8967-step2000-rerun/"
    "step-000250.pt"
)


def placeholder_scene(molecule_id: str, width: int, height: int) -> dict:
    """Provide only a canvas; compiled atoms intentionally fall back to carbon."""

    return {
        "schema_version": "chemscene-0.2",
        "molecule_id": molecule_id,
        "units": "q1023",
        "canvas": {"width": width, "height": height},
        "atom_objects": [],
        "bond_objects": [],
    }


def prepare_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        rgba = opened.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    image = Image.alpha_composite(white, rgba).convert("RGB")
    return image.resize((768, 768), Image.Resampling.BICUBIC)


def compile_carbon_placeholder(scene: dict) -> tuple[dict, object, int]:
    """Demote aromatic chemistry when placeholder carbon cannot be kekulized."""

    repaired = deepcopy(scene)
    for atom in repaired.get("atom_objects", []):
        atom["chemical"]["aromatic"] = False
        atom["incident_ports"] = []
    demoted = 0
    for bond in repaired.get("bond_objects", []):
        if bond.get("chemical_type") != "AROMATIC":
            pass
        else:
            display = bond.get("display_type")
            bond["chemical_type"] = (
                display if display in {"SINGLE", "DOUBLE", "TRIPLE"} else "SINGLE"
            )
            demoted += 1
        for endpoint, atom_id in enumerate(bond["atom_refs"]):
            atom = repaired["atom_objects"][atom_id]
            coordinate = list(atom["anchor"])
            bond["ports"][endpoint] = coordinate
            atom["incident_ports"].append(
                {
                    "bond_id": bond["id"],
                    "endpoint": endpoint,
                    "coordinate": coordinate,
                    "source": "ANCHOR",
                }
            )
    return repaired, mol_from_chemscene_v0_2(repaired), demoted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-version", default="v0.9.67-structure-carbon")
    parser.add_argument("--primary-checkpoint", default=DEFAULT_PRIMARY)
    parser.add_argument("--fallback-checkpoint", default=DEFAULT_FALLBACK)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--atom-checkpoint", default=DEFAULT_ATOM)
    parser.add_argument("--bond-checkpoint", default=DEFAULT_BOND)
    parser.add_argument("--atom-threshold", type=float, default=-0.25)
    parser.add_argument("--bond-threshold", type=float, default=-0.5)
    parser.add_argument("--atom-pair-group-checkpoint")
    parser.add_argument("--atom-pair-graph-context-checkpoint")
    parser.add_argument("--atom-pair-one-move-checkpoint")
    parser.add_argument("--atom-pair-group-threshold", type=float, default=0.0)
    parser.add_argument("--line-segment-group-rescue", action="store_true")
    parser.add_argument("--line-rescue-minimum-support", type=int, default=2)
    parser.add_argument("--line-atom-pair-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image = prepare_image(Path(args.image))
    image.save(output / "input-prepared.png")
    started = time.perf_counter()
    vision, atom_head, bond_head, _ = load_frozen_components(
        Path(args.grounded_graph_checkpoint),
        Path(args.atom_checkpoint),
        Path(args.bond_checkpoint),
        device,
    )
    primary, _ = _load_set_head(args.primary_checkpoint, device)
    fallback, _ = _load_set_head(args.fallback_checkpoint, device)
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
        if atom_pair_group_head is None:
            raise ValueError("Line AtomPair inference requires the AtomPair group head")
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
    atom_pair_one_move_head = None
    atom_pair_one_move_payload = None
    atom_pair_one_move_relation_config = None
    if args.atom_pair_one_move_checkpoint:
        if atom_pair_graph_context_head is None:
            raise ValueError("AtomPair one-move requires AtomPair graph context")
        atom_pair_one_move_head, atom_pair_one_move_payload = (
            load_atom_pair_one_move_head(args.atom_pair_one_move_checkpoint, device)
        )
        if Path(atom_pair_one_move_payload["base_context_checkpoint"]) != Path(
            args.atom_pair_graph_context_checkpoint
        ):
            raise ValueError("AtomPair one-move was trained for another graph context")
        if Path(atom_pair_one_move_payload["group_checkpoint"]) != Path(
            args.atom_pair_group_checkpoint
        ):
            raise ValueError("AtomPair one-move was trained for another group head")
        atom_pair_one_move_relation_config = AtomPairDeletionVetoConfig(
            **atom_pair_one_move_payload["relation_config"]
        )
    pixels = _image_tensor(image)[None].to(device)
    proposal_config = PredictedProposalConfig(maximum_atoms=128, maximum_bonds=192)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
    ):
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom_head(features)
        bond_output = forward_frozen_bond_head(vision, bond_head, features)
        proposals = {
            **decode_predicted_atom_proposals(atom_output, proposal_config),
            **decode_predicted_bond_proposals(bond_output, proposal_config),
        }
        base_outputs = [
            ("primary", primary(
                proposals["atom_features"], proposals["atom_geometry"],
                proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"],
                proposals["bond_types"], proposals["bond_scores"],
                proposals["bond_mask"],
            )),
            ("fallback", fallback(
                proposals["atom_features"], proposals["atom_geometry"],
                proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"],
                proposals["bond_types"], proposals["bond_scores"],
                proposals["bond_mask"],
            )),
        ]
        attempts = []
        if atom_pair_group_head is not None:
            for name, model_output in base_outputs:
                groups = build_atom_pair_groups(model_output, proposals)
                group_output = atom_pair_group_head(groups["features"])
                graph_context_diagnostics = None
                if atom_pair_graph_context_head is not None:
                    group_output, graph_context_diagnostics = (
                        refine_atom_pair_group_output(
                            atom_pair_graph_context_head,
                            group_output,
                            groups,
                            batch_size=1,
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
                one_move_diagnostics = None
                if atom_pair_one_move_head is not None:
                    pairs = flatten_atom_pairs(groups["atom_pairs"], device=device)
                    relation = build_atom_pair_relation_features(
                        pixels,
                        proposals["atom_geometry"][..., :2],
                        proposals["atom_mask"],
                        model_output["atom_presence_logits"],
                        groups["sample_indices"],
                        pairs,
                        group_output["presence_logits"],
                        batch_size=1,
                        config=atom_pair_one_move_relation_config,
                    )
                    edge_features = torch.cat(
                        (relation, group_output["presence_logits"].float()[:, None]),
                        dim=-1,
                    )
                    action_scores, actions = atom_pair_one_move_head.score_sample(
                        edge_features,
                        relation,
                        group_output["presence_logits"],
                        pairs,
                    )
                    action, action_margin = choose_atom_pair_action(
                        action_scores,
                        actions,
                        minimum_margin=float(
                            atom_pair_one_move_payload["minimum_margin"]
                        ),
                    )
                    selected_before = group_output["presence_logits"] >= 0.0
                    selected_after = apply_atom_pair_action(selected_before, action)
                    refined_logits = group_output["presence_logits"].clone()
                    changed = selected_after != selected_before
                    refined_logits[changed & selected_after] = 1e-4
                    refined_logits[changed & ~selected_after] = -1e-4
                    group_output = {**group_output, "presence_logits": refined_logits}
                    one_move_diagnostics = {
                        "action": action.kind,
                        "delete_index": action.delete_index,
                        "add_index": action.add_index,
                        "delete_atom_pair": (
                            pairs[action.delete_index].tolist()
                            if action.delete_index >= 0
                            else None
                        ),
                        "add_atom_pair": (
                            pairs[action.add_index].tolist()
                            if action.add_index >= 0
                            else None
                        ),
                        "score_margin": action_margin,
                        "minimum_margin": float(
                            atom_pair_one_move_payload["minimum_margin"]
                        ),
                        "candidate_actions": len(actions),
                    }
                plain_proposals, plain_output, plain_grouping = (
                    collapse_atom_pair_group_outputs(
                        proposals, model_output, group_output, groups
                    )
                )
                if graph_context_diagnostics is not None:
                    plain_grouping[0]["graph_context"] = graph_context_diagnostics
                if one_move_diagnostics is not None:
                    plain_grouping[0]["one_move"] = one_move_diagnostics
                sample_line_candidates = None
                if args.line_segment_group_rescue or line_atom_pair_head is not None:
                    predicted_pixels = (
                        proposals["atom_geometry"][0, :, :2].float().cpu().numpy()
                        * 768.0
                    )
                    selected_atoms = (
                        proposals["atom_mask"][0]
                        & (model_output["atom_presence_logits"][0] >= args.atom_threshold)
                    ).cpu().numpy()
                    sample_line_candidates = tuple(
                        candidate
                        for candidate in detect_atom_pair_line_candidates(
                            image,
                            predicted_pixels,
                            atom_mask=selected_atoms,
                        )
                        if len(candidate.support_segment_indices)
                        >= args.line_rescue_minimum_support
                    )
                grouped_proposals = plain_proposals
                grouped_output = plain_output
                grouping = plain_grouping
                if args.line_segment_group_rescue:
                    rescued_group_output, line_rescue = rescue_disconnected_atom_pair_groups(
                        group_output,
                        groups,
                        atom_presence_logits=model_output["atom_presence_logits"],
                        atom_mask=proposals["atom_mask"],
                        line_candidates=[sample_line_candidates or ()],
                        atom_threshold=args.atom_threshold,
                        group_threshold=args.atom_pair_group_threshold,
                        config=LineSegmentGroupRescueConfig(
                            minimum_support_segments=args.line_rescue_minimum_support
                        ),
                    )
                    grouped_proposals, grouped_output, grouping = (
                        collapse_atom_pair_group_outputs(
                            proposals, model_output, rescued_group_output, groups
                        )
                    )
                    grouping[0]["line_segment_rescue"] = line_rescue[0]
                    attempts.append(
                        (
                            f"line_atom_pair_{name}",
                            grouped_proposals,
                            grouped_output,
                            grouping[0],
                            args.atom_pair_group_threshold,
                        )
                    )
                if line_atom_pair_head is not None:
                    existing_pairs = set(groups["atom_pairs"][0])
                    line_only_candidates = tuple(
                        candidate
                        for candidate in (sample_line_candidates or ())
                        if candidate.atom_pair not in existing_pairs
                    )
                    line_features, line_sample_indices = build_line_atom_pair_features(
                        bond_output["object_features"].float(),
                        proposals["atom_features"].float(),
                        proposals["atom_geometry"][..., :2].float(),
                        [line_only_candidates],
                        config=line_atom_pair_head.config,
                    )
                    line_output = line_atom_pair_head(line_features)
                    line_proposals, line_model_output, line_diagnostics = (
                        append_line_atom_pair_proposals(
                            grouped_proposals,
                            grouped_output,
                            line_output,
                            [line_only_candidates],
                            line_sample_indices,
                            line_threshold=float(line_atom_pair_payload["threshold"]),
                        )
                    )
                    line_grouping = deepcopy(grouping[0])
                    line_grouping["line_atom_pair"] = line_diagnostics[0]
                    attempts.insert(
                        len(attempts) - int(args.line_segment_group_rescue),
                        (
                            f"first_class_line_atom_pair_{name}",
                            line_proposals,
                            line_model_output,
                            line_grouping,
                            args.atom_pair_group_threshold,
                        ),
                    )
                attempts.append(
                    (
                        f"atom_pair_{name}",
                        plain_proposals,
                        plain_output,
                        plain_grouping[0],
                        args.atom_pair_group_threshold,
                    )
                )
        attempts.extend(
            (name, proposals, model_output, None, args.bond_threshold)
            for name, model_output in base_outputs
        )
    scene = predicted = diagnostics = None
    fallback_scene = fallback_diagnostics = None
    errors = []
    gt_scene = placeholder_scene("upload", 768, 768)
    joint_config = JointGraphMilpConfig(
        endpoint_topk=2,
        incidence_weight=0.5,
        time_limit_seconds=2.0,
    )
    for name, attempt_proposals, model_output, grouping, bond_threshold in attempts:
        for joint in (True, False):
            try:
                scene, predicted, diagnostics = _attempt_scene(
                    molecule_id="upload",
                    gt_scene=gt_scene,
                    proposals=attempt_proposals,
                    output=model_output,
                    batch_index=0,
                    atom_threshold=args.atom_threshold,
                    bond_threshold=bond_threshold,
                    joint=joint,
                    joint_config=joint_config,
                )
                diagnostics["decode_path"] = f"{name}_{'joint' if joint else 'independent'}"
                if grouping is not None:
                    diagnostics["atom_pair_groups"] = grouping
                break
            except SceneCompilationFailure as error:
                scene = error.scene
                diagnostics = dict(error.diagnostics)
                diagnostics["decode_path"] = (
                    f"{name}_{'joint' if joint else 'independent'}_compile_failed"
                )
                if grouping is not None:
                    diagnostics["atom_pair_groups"] = grouping
                if fallback_scene is None:
                    fallback_scene = deepcopy(scene)
                    fallback_diagnostics = deepcopy(diagnostics)
                errors.append(f"{name}: {error}")
            except Exception as error:
                errors.append(f"{name}: {error}")
        if predicted is not None:
            break
    scene_path = output / "prediction.chemscene.json"
    if predicted is None and fallback_scene is not None:
        scene = fallback_scene
        diagnostics = fallback_diagnostics
    if predicted is None and scene is not None:
        try:
            scene, predicted, demoted = compile_carbon_placeholder(scene)
            diagnostics = dict(diagnostics or {})
            diagnostics["carbon_placeholder_aromatic_demotions"] = demoted
            diagnostics["decode_path"] = "carbon_placeholder_aromatic_fallback"
        except Exception as error:
            errors.append(f"carbon placeholder fallback: {error}")
    if scene is not None:
        scene_path.write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    sdf_path = output / "prediction-repaired.sdf"
    if predicted is None:
        _write_empty_sdf(sdf_path, errors[-1] if errors else "No valid decode path")
    else:
        write_prediction_sdf(predicted, sdf_path, sanitize=True)
    _render_sdf(sdf_path, output / "prediction-repaired.png")
    metrics = {
        "model_version": args.model_version,
        "chemistry_mode": "all_text_atoms_as_carbon_placeholder",
        "inference_seconds": time.perf_counter() - started,
        "vertices": predicted.GetNumAtoms() if predicted is not None else 0,
        "candidate_strokes": int(proposals["bond_mask"][0].sum()),
        "repaired_strokes": predicted.GetNumBonds() if predicted is not None else 0,
        "formula": (
            rdMolDescriptors.CalcMolFormula(predicted) if predicted is not None else None
        ),
        "sdf_valid": predicted is not None,
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
        "atom_pair_one_move_checkpoint": args.atom_pair_one_move_checkpoint,
        "atom_pair_one_move_samples_seen": (
            atom_pair_one_move_payload["samples_seen"]
            if atom_pair_one_move_payload is not None
            else None
        ),
        "atom_pair_one_move_minimum_margin": (
            float(atom_pair_one_move_payload["minimum_margin"])
            if atom_pair_one_move_payload is not None
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
        "diagnostics": diagnostics,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
