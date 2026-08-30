#!/usr/bin/env python3
"""Run joint Atom/Bond object inference with carbon placeholder chemistry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from rdkit.Chem import rdMolDescriptors
import torch

from chemvector.benchmark_joint_mr_tadf import (
    DEFAULT_FALLBACK,
    DEFAULT_PRIMARY,
    SceneCompilationFailure,
    _attempt_scene,
)
from chemvector.benchmark_mr_tadf import write_prediction_sdf
from chemvector.joint_atom_pair_refinement import build_atom_pair_attempts
from chemvector.joint_component_loader import load_joint_components
from chemvector.joint_graph_milp import JointGraphMilpConfig
from chemvector.placeholder_scene import (
    compile_carbon_placeholder,
    placeholder_scene,
    prepare_image,
)
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
)

# Historical import paths: tests and the Direct entry import these helpers
# from this CLI module. The implementations live in chemvector.placeholder_scene.
__all__ = [
    "compile_carbon_placeholder",
    "placeholder_scene",
    "prepare_image",
    "parse_args",
    "main",
]


DEFAULT_ATOM = (
    "outputs/checkpoints/chemvector-v0967-atom-object-full8967-step2000-rerun/"
    "step-000250.pt"
)


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
    parser.add_argument("--atom-pair-chemical-retype-verifier-checkpoint")
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
    components = load_joint_components(
        grounded_graph_checkpoint=Path(args.grounded_graph_checkpoint),
        atom_checkpoint=Path(args.atom_checkpoint),
        bond_checkpoint=Path(args.bond_checkpoint),
        primary_checkpoint=args.primary_checkpoint,
        fallback_checkpoint=args.fallback_checkpoint,
        atom_pair_group_checkpoint=args.atom_pair_group_checkpoint,
        atom_pair_graph_context_checkpoint=(
            args.atom_pair_graph_context_checkpoint
        ),
        atom_pair_one_move_checkpoint=args.atom_pair_one_move_checkpoint,
        atom_pair_chemical_retype_verifier_checkpoint=(
            args.atom_pair_chemical_retype_verifier_checkpoint
        ),
        line_atom_pair_checkpoint=args.line_atom_pair_checkpoint,
        device=device,
    )
    experts = components.experts
    pixels = _image_tensor(image)[None].to(device)
    proposal_config = PredictedProposalConfig(maximum_atoms=128, maximum_bonds=192)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
    ):
        features = components.vision.forward_stroke_resolution(pixels)
        atom_output = components.atom_head(features)
        bond_output = forward_frozen_bond_head(
            components.vision, components.bond_head, features
        )
        proposals = {
            **decode_predicted_atom_proposals(atom_output, proposal_config),
            **decode_predicted_bond_proposals(bond_output, proposal_config),
        }
        base_outputs = [
            ("primary", components.primary(
                proposals["atom_features"], proposals["atom_geometry"],
                proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"],
                proposals["bond_types"], proposals["bond_scores"],
                proposals["bond_mask"],
            )),
            ("fallback", components.fallback(
                proposals["atom_features"], proposals["atom_geometry"],
                proposals["atom_scores"], proposals["atom_mask"],
                proposals["bond_features"], proposals["bond_endpoints"],
                proposals["bond_types"], proposals["bond_scores"],
                proposals["bond_mask"],
            )),
        ]
        attempts = []
        if experts.group_head is not None:
            for name, model_output in base_outputs:
                attempts.extend(
                    build_atom_pair_attempts(
                        name,
                        model_output,
                        proposals,
                        bond_output,
                        pixels,
                        image,
                        experts,
                        atom_threshold=args.atom_threshold,
                        group_threshold=args.atom_pair_group_threshold,
                        line_segment_group_rescue=args.line_segment_group_rescue,
                        line_rescue_minimum_support=(
                            args.line_rescue_minimum_support
                        ),
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
            experts.group_payload["samples_seen"]
            if experts.group_payload is not None
            else None
        ),
        "atom_pair_graph_context_checkpoint": (
            args.atom_pair_graph_context_checkpoint
        ),
        "atom_pair_graph_context_samples_seen": (
            experts.graph_context_payload["samples_seen"]
            if experts.graph_context_payload is not None
            else None
        ),
        "atom_pair_graph_context_residual_scale": (
            float(experts.graph_context_payload.get("residual_scale", 1.0))
            if experts.graph_context_payload is not None
            else None
        ),
        "atom_pair_graph_context_base_positive_protection_threshold": (
            experts.graph_context_payload.get(
                "base_positive_protection_threshold"
            )
            if experts.graph_context_payload is not None
            else None
        ),
        "atom_pair_one_move_checkpoint": args.atom_pair_one_move_checkpoint,
        "atom_pair_one_move_samples_seen": (
            experts.one_move_payload["samples_seen"]
            if experts.one_move_payload is not None
            else None
        ),
        "atom_pair_one_move_minimum_margin": (
            float(experts.one_move_payload["minimum_margin"])
            if experts.one_move_payload is not None
            else None
        ),
        "atom_pair_chemical_retype_verifier_checkpoint": (
            args.atom_pair_chemical_retype_verifier_checkpoint
        ),
        "atom_pair_chemical_retype_verifier_samples_seen": (
            experts.retype_verifier_payload["samples_seen"]
            if experts.retype_verifier_payload is not None
            else None
        ),
        "atom_pair_chemical_retype_verifier_minimum_margin": (
            float(experts.retype_verifier_payload["gate_threshold"])
            if experts.retype_verifier_payload is not None
            else None
        ),
        "line_segment_group_rescue": args.line_segment_group_rescue,
        "line_rescue_minimum_support": args.line_rescue_minimum_support,
        "line_atom_pair_checkpoint": args.line_atom_pair_checkpoint,
        "line_atom_pair_epoch": (
            experts.line_atom_pair_payload["epoch"]
            if experts.line_atom_pair_payload is not None
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