#!/usr/bin/env python3
"""Run v0.10.3 Direct AtomPair Bond inference with carbon placeholders."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

from rdkit.Chem import rdMolDescriptors
import torch

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.benchmark_mr_tadf import write_prediction_sdf
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings
from chemvector.ctc_ocr import decode_ctc_with_score
from chemvector.direct_atom_role import enumerate_selected_atoms
from chemvector.direct_scene_inference import (
    apply_atom_ocr,
    apply_structural_twin_suppression,
    direct_scene_tensors,
    load_action_ranking_model,
    load_direct_model,
)
from chemvector.evaluate_direct_atom_ocr import crop_tensor, load_bbox, load_ctc, load_role
from chemvector.infer_joint_object_structure import (
    compile_carbon_placeholder,
    placeholder_scene,
    prepare_image,
)
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.shadow_chemscene_v02_oracle import (
    _render_sdf,
    _write_empty_sdf,
    apply_aromatic_consistency,
    build_oracle_chemistry_shadow_scene,
)
from chemvector.train_atom_bond_scene import _image_tensor
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import forward_frozen_bond_head

# Historical import paths: audits and legacy callers import these names from
# the CLI module. The implementations live in chemvector.direct_scene_inference.
__all__ = [
    "apply_atom_ocr",
    "apply_structural_twin_suppression",
    "direct_scene_tensors",
    "load_action_ranking_model",
    "load_direct_model",
    "parse_args",
    "main",
]


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
)
DEFAULT_ROLE_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k/selected.pt"
)
DEFAULT_BBOX_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0108-direct-atom-text-bbox-style10k/selected.pt"
)
DEFAULT_OCR_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k/selected.pt"
)
DEFAULT_ACTION_RANKING_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v01013-safe-atom-action-50k/samples-030000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-version", default="v0.10.3-direct-atom-pair-carbon")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--role-checkpoint", default=DEFAULT_ROLE_CHECKPOINT)
    parser.add_argument("--bbox-checkpoint", default=DEFAULT_BBOX_CHECKPOINT)
    parser.add_argument("--ocr-checkpoint", default=DEFAULT_OCR_CHECKPOINT)
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument("--structural-twin-suppression", action="store_true")
    parser.add_argument(
        "--action-ranking-checkpoint", default=DEFAULT_ACTION_RANKING_CHECKPOINT
    )
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--grounded-graph-checkpoint")
    parser.add_argument("--aromatic-ring-fusion", action="store_true")
    parser.add_argument("--aromatic-mean-threshold", type=float, default=0.40)
    parser.add_argument("--aromatic-minimum-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = prepare_image(Path(args.image))
    image.save(output_dir / "input-prepared.png")
    started = time.perf_counter()

    direct, direct_payload = load_direct_model(args.checkpoint, device)
    action_ranking = None
    if args.structural_twin_suppression:
        action_ranking, _ = load_action_ranking_model(
            args.action_ranking_checkpoint, device
        )
    role = bbox = ctc = None
    role_threshold = None
    if args.enable_ocr:
        role, role_payload = load_role(args.role_checkpoint, device)
        bbox, _ = load_bbox(args.bbox_checkpoint, device)
        ctc, _ = load_ctc(args.ocr_checkpoint, device)
        role_threshold = float(role_payload["validation"]["threshold"])
    training_args = direct_payload["training_args"]
    source_checkpoint = args.source_checkpoint or training_args["source_checkpoint"]
    graph_checkpoint = (
        args.grounded_graph_checkpoint
        or training_args["grounded_graph_checkpoint"]
    )
    vision, atom, bond, set_head, source = load_components(
        source_checkpoint, graph_checkpoint, device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    probability_threshold = float(direct_payload["validation"]["threshold"])
    logit_threshold = math.log(probability_threshold / (1.0 - probability_threshold))
    pixels = _image_tensor(image)[None].to(device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        features = vision.forward_stroke_resolution(pixels)
        atom_output = atom(features)
        bond_output = forward_frozen_bond_head(vision, bond, features)
        atoms = decode_predicted_atom_proposals(atom_output, proposal_config)
        dense_bonds = decode_predicted_bond_proposals(bond_output, proposal_config)
        set_output = set_head(
            atoms["atom_features"],
            atoms["atom_geometry"],
            atoms["atom_scores"],
            atoms["atom_mask"],
            dense_bonds["bond_features"],
            dense_bonds["bond_endpoints"],
            dense_bonds["bond_types"],
            dense_bonds["bond_scores"],
            dense_bonds["bond_mask"],
        )
        atoms["atom_mask"] = atoms["atom_mask"] & (
            set_output["atom_presence_logits"] >= atom_threshold
        )
        pairs, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        direct_output = direct(
            features,
            atoms["atom_features"],
            atoms["atom_geometry"][..., :2],
            atoms["atom_scores"],
            pairs,
            sample_indices,
        )

        suppression_diagnostics = {"enabled": False, "executed": False}
        if action_ranking is not None:
            pairs, direct_output, suppression_diagnostics = (
                apply_structural_twin_suppression(
                    atoms,
                    pairs,
                    direct_output,
                    action_ranking,
                    presence_threshold=probability_threshold,
                )
            )

        ocr_predictions = {}
        if args.enable_ocr:
            atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
            role_probabilities = role(
                pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices
            ).sigmoid()
            text_rows = torch.where(role_probabilities >= role_threshold)[0]
            learned = bbox(
                pixels,
                atoms["atom_geometry"][..., :2],
                atom_indices[text_rows],
                sample_indices[text_rows],
            )
            crops = []
            for center, half_extent in zip(learned["centers"], learned["half_extents"]):
                x, y = (center.float().cpu() * 768).tolist()
                half_width, half_height = (half_extent.float().cpu() * 768).tolist()
                crops.append(
                    crop_tensor(
                        image,
                        (x - half_width, y - half_height, x + half_width, y + half_height),
                    )
                )
            decoded = []
            if crops:
                logits = ctc(torch.stack(crops).to(device))
                decoded = [decode_ctc_with_score(row.float()) for row in logits]
            for index, row in enumerate(text_rows.tolist()):
                candidate = int(atom_indices[row])
                text, score = decoded[index]
                ocr_predictions[candidate] = {
                    "text": text,
                    "score": score,
                    "center": learned["centers"][index].float().cpu().tolist(),
                    "half_extent": learned["half_extents"][index].float().cpu().tolist(),
                }

    ring_diagnostics = None
    chemical_predictions = None
    if args.aromatic_ring_fusion:
        chemical_predictions, ring_diagnostics = promote_complete_aromatic_rings(
            pairs,
            direct_output["presence_logits"].sigmoid(),
            direct_output["chemical_logits"],
            presence_threshold=probability_threshold,
            mean_threshold=args.aromatic_mean_threshold,
            minimum_threshold=args.aromatic_minimum_threshold,
        )
    compiler = direct_scene_tensors(
        atoms, pairs, direct_output, chemical_predictions
    )
    gt_scene = placeholder_scene("upload", 768, 768)
    scene, diagnostics = build_oracle_chemistry_shadow_scene(
        molecule_id="upload",
        gt_scene=gt_scene,
        atom_geometry=atoms["atom_geometry"][0].float().cpu(),
        atom_mask=atoms["atom_mask"][0].cpu(),
        atom_presence_logits=compiler["atom_presence_logits"].float().cpu(),
        bond_types=compiler["bond_types"].float().cpu(),
        bond_mask=compiler["bond_mask"].cpu(),
        bond_presence_logits=compiler["bond_presence_logits"].float().cpu(),
        incidence_logits=compiler["incidence_logits"].float().cpu(),
        atom_threshold=0.0,
        bond_threshold=logit_threshold,
    )
    if args.enable_ocr:
        diagnostics["ocr"] = apply_atom_ocr(
            scene,
            diagnostics["selected_atom_candidate_indices"],
            ocr_predictions,
        )
    scene, aromatic = apply_aromatic_consistency(
        scene, "demote-noncycle-clear-atoms"
    )
    diagnostics["aromatic"] = aromatic
    diagnostics["direct_atom_pairs"] = len(pairs)
    diagnostics["direct_presence_threshold"] = probability_threshold
    diagnostics["aromatic_ring_fusion"] = ring_diagnostics
    diagnostics["structural_twin_suppression"] = suppression_diagnostics
    errors = []
    predicted = None
    decode_path = "direct_atom_pair"
    try:
        predicted = mol_from_chemscene_v0_2(scene)
    except Exception as error:
        errors.append(str(error))
        try:
            scene, predicted, demoted = compile_carbon_placeholder(scene)
            diagnostics["carbon_placeholder_aromatic_demotions"] = demoted
            decode_path = "direct_atom_pair_carbon_fallback"
        except Exception as fallback_error:
            errors.append(str(fallback_error))
            decode_path = "direct_atom_pair_invalid"
    diagnostics["decode_path"] = decode_path
    (output_dir / "prediction.chemscene.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sdf_path = output_dir / "prediction-repaired.sdf"
    if predicted is None:
        _write_empty_sdf(sdf_path, errors[-1] if errors else "Invalid direct graph")
    else:
        write_prediction_sdf(predicted, sdf_path, sanitize=True)
    _render_sdf(sdf_path, output_dir / "prediction-repaired.png")
    selected_bonds = int(
        (direct_output["presence_logits"].sigmoid() >= probability_threshold).sum()
    )
    metrics = {
        "model_version": args.model_version,
        "checkpoint": args.checkpoint,
        "source_checkpoint": source_checkpoint,
        "chemistry_mode": (
            "v0109_open_byte_ctc" if args.enable_ocr else "all_text_atoms_as_carbon_placeholder"
        ),
        "role_checkpoint": args.role_checkpoint if args.enable_ocr else None,
        "bbox_checkpoint": args.bbox_checkpoint if args.enable_ocr else None,
        "ocr_checkpoint": args.ocr_checkpoint if args.enable_ocr else None,
        "structural_twin_suppression": args.structural_twin_suppression,
        "action_ranking_checkpoint": (
            args.action_ranking_checkpoint
            if args.structural_twin_suppression
            else None
        ),
        "inference_seconds": time.perf_counter() - started,
        "vertices": int(atoms["atom_mask"].sum()),
        "candidate_atom_pairs": len(pairs),
        "selected_bonds": selected_bonds,
        "presence_threshold": probability_threshold,
        "aromatic_ring_fusion": args.aromatic_ring_fusion,
        "aromatic_mean_threshold": args.aromatic_mean_threshold,
        "aromatic_minimum_threshold": args.aromatic_minimum_threshold,
        "formula": (
            rdMolDescriptors.CalcMolFormula(predicted) if predicted is not None else None
        ),
        "sdf_valid": predicted is not None,
        "errors": errors,
        "diagnostics": diagnostics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
