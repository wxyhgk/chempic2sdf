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
from Data.sdf.chemscene_v0_2 import (
    QUANTIZATION_MAX,
    _clip_port_to_text_bbox,
    parse_chemscene_v0_2,
)
from chemvector.benchmark_mr_tadf import write_prediction_sdf
from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    enumerate_unordered_atom_pairs,
)
from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings
from chemvector.ctc_ocr import decode_ctc_with_score
from chemvector.direct_atom_role import enumerate_selected_atoms
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
from chemvector.ocr import parse_atom_display_text
from chemvector.safe_atom_action_model import (
    SafeAtomActionConfig,
    SafeAtomActionModel,
    delete_action_features,
)
from chemvector.structural_twin_certificate import (
    StructuralTwinCertificateConfig,
    structural_twin_witnesses,
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


def _q(value: float) -> int:
    return max(0, min(QUANTIZATION_MAX, round(float(value) * QUANTIZATION_MAX)))


def apply_atom_ocr(
    scene: dict,
    candidate_indices: list[int],
    predictions: dict[int, dict],
) -> dict:
    """Write parsed OCR labels into ChemScene and rebuild text-clipped bond ports."""

    if len(candidate_indices) != len(scene["atom_objects"]):
        raise ValueError("Compiled AtomObject/candidate mapping length differs")
    diagnostics = {"predicted_text": 0, "parsed_text": 0, "unparsed_text": 0, "labels": []}
    for atom, candidate in zip(scene["atom_objects"], candidate_indices):
        prediction = predictions.get(int(candidate))
        if prediction is None or not prediction["text"]:
            continue
        text = str(prediction["text"])
        diagnostics["predicted_text"] += 1
        parsed = parse_atom_display_text(text)
        diagnostics["labels"].append(
            {"atom_id": int(atom["id"]), "candidate": int(candidate), "text": text,
             "score": float(prediction["score"]), "parsed": parsed is not None}
        )
        if parsed is None:
            diagnostics["unparsed_text"] += 1
            continue
        diagnostics["parsed_text"] += 1
        center_x, center_y = prediction["center"]
        half_width, half_height = prediction["half_extent"]
        center = [_q(center_x), _q(center_y)]
        bbox = [
            _q(center_x - half_width),
            _q(center_y - half_height),
            _q(center_x + half_width),
            _q(center_y + half_height),
        ]
        atom["role"] = "TEXT"
        atom["chemical"] = {
            "element": parsed["element"],
            "formal_charge": parsed["formal_charge"],
            "isotope": parsed["isotope"],
            "aromatic": False,
            "explicit_h_count": parsed["explicit_h_count"],
            "no_implicit": parsed["explicit_h_count"] > 0,
            "radical_electrons": parsed["radical_electrons"],
            "canonical_chiral_tag": "CHI_UNSPECIFIED",
        }
        atom["text"] = {"value": text, "center": center, "bbox": bbox}

    for atom in scene["atom_objects"]:
        atom["incident_ports"] = []
    for bond in scene["bond_objects"]:
        begin_atom, end_atom = (int(value) for value in bond["atom_refs"])
        begin = scene["atom_objects"][begin_atom]
        end = scene["atom_objects"][end_atom]
        begin_port, begin_source = _clip_port_to_text_bbox(
            tuple(begin["anchor"]),
            tuple(end["anchor"]),
            tuple(begin["text"]["bbox"]) if begin["text"] is not None else None,
        )
        end_port, end_source = _clip_port_to_text_bbox(
            tuple(end["anchor"]),
            tuple(begin["anchor"]),
            tuple(end["text"]["bbox"]) if end["text"] is not None else None,
        )
        bond["ports"] = [list(begin_port), list(end_port)]
        begin["incident_ports"].append(
            {"bond_id": int(bond["id"]), "endpoint": 0,
             "coordinate": list(begin_port), "source": begin_source}
        )
        end["incident_ports"].append(
            {"bond_id": int(bond["id"]), "endpoint": 1,
             "coordinate": list(end_port), "source": end_source}
        )
    parse_chemscene_v0_2(scene)
    return diagnostics


def load_direct_model(
    checkpoint: str, device: torch.device
) -> tuple[DirectAtomPairBond, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "direct-atom-pair-bond-0.1":
        raise ValueError("Checkpoint is not a Direct AtomPair Bond model")
    model = DirectAtomPairBond(
        DirectAtomPairBondConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


def load_action_ranking_model(
    checkpoint: str, device: torch.device
) -> tuple[SafeAtomActionModel, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "safe-atom-action-0.1":
        raise ValueError("Checkpoint is not a safe Atom action ranking model")
    model = SafeAtomActionModel(SafeAtomActionConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload


@torch.no_grad()
def apply_structural_twin_suppression(
    atoms: dict[str, torch.Tensor],
    pairs: torch.Tensor,
    direct_output: dict[str, torch.Tensor],
    ranking: SafeAtomActionModel,
    *,
    presence_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
    """Execute at most one frozen v0.10.15 deletion before OCR/compilation."""

    if atoms["atom_mask"].size(0) != 1:
        raise ValueError("Structural-twin inference currently requires batch size 1")
    if pairs.shape != (len(direct_output["presence_logits"]), 2):
        raise ValueError("Structural-twin pair/output count differs")
    selected = torch.where(atoms["atom_mask"][0])[0]
    remap = torch.full(
        (atoms["atom_mask"].size(1),),
        -1,
        device=pairs.device,
        dtype=torch.long,
    )
    remap[selected] = torch.arange(len(selected), device=pairs.device)
    compact_pairs = remap[pairs]
    if bool((compact_pairs < 0).any()):
        raise ValueError("Structural-twin pair references an unselected AtomObject")
    sample = {
        "anchors": atoms["atom_geometry"][0, selected, :2].float().cpu(),
        "atom_features": atoms["atom_features"][0, selected].float().cpu(),
        "atom_geometry": atoms["atom_geometry"][0, selected].float().cpu(),
        "scores": atoms["atom_scores"][0, selected].float().cpu(),
        "pairs": compact_pairs.cpu(),
        "probabilities": direct_output["presence_logits"].float().sigmoid().cpu(),
        "chemical": direct_output["chemical_logits"].float().argmax(-1).cpu(),
    }
    features, removed_nodes = delete_action_features(
        sample, presence_threshold=presence_threshold
    )
    diagnostics = {
        "enabled": True,
        "candidate_actions": len(removed_nodes),
        "certified_actions": 0,
        "qualified_actions": 0,
        "executed": False,
        "removed_candidate_index": None,
        "removed_compact_index": None,
        "witness_compact_indices": [],
        "ranking_score": None,
        "utility_class": None,
        "atoms_before": int(atoms["atom_mask"].sum()),
        "atoms_after": int(atoms["atom_mask"].sum()),
        "pairs_before": len(pairs),
        "pairs_after": len(pairs),
    }
    if not len(removed_nodes):
        return pairs, direct_output, diagnostics
    certificate = StructuralTwinCertificateConfig()
    witnesses = [
        structural_twin_witnesses(
            sample,
            int(removed),
            presence_threshold=presence_threshold,
            config=certificate,
        )
        for removed in removed_nodes.tolist()
    ]
    certified = torch.tensor([bool(rows) for rows in witnesses], dtype=torch.bool)
    diagnostics["certified_actions"] = int(certified.sum())
    if not bool(certified.any()):
        return pairs, direct_output, diagnostics
    ranking_device = next(ranking.parameters()).device
    output = ranking(features.to(ranking_device))
    utility = output["utility_logits"].argmax(-1).cpu()
    ranking_score = (
        output["rank_logits"]
        + torch.nn.functional.logsigmoid(output["safety_logits"])
    ).float().cpu()
    qualified = torch.where(
        certified & (utility > 0) & (ranking_score > 0.0)
    )[0]
    diagnostics["qualified_actions"] = len(qualified)
    if not len(qualified):
        return pairs, direct_output, diagnostics
    row = int(qualified[ranking_score[qualified].argmax()])
    removed_compact = int(removed_nodes[row])
    removed_candidate = int(selected[removed_compact])
    atoms["atom_mask"][0, removed_candidate] = False
    keep_pairs = (pairs[:, 0] != removed_candidate) & (
        pairs[:, 1] != removed_candidate
    )
    filtered_output = {
        name: (
            value[keep_pairs]
            if isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and len(value) == len(keep_pairs)
            else value
        )
        for name, value in direct_output.items()
    }
    filtered_pairs = pairs[keep_pairs]
    diagnostics.update(
        {
            "executed": True,
            "removed_candidate_index": removed_candidate,
            "removed_compact_index": removed_compact,
            "witness_compact_indices": list(witnesses[row]),
            "ranking_score": float(ranking_score[row]),
            "utility_class": int(utility[row]),
            "atoms_after": int(atoms["atom_mask"].sum()),
            "pairs_after": len(filtered_pairs),
        }
    )
    return filtered_pairs, filtered_output, diagnostics


def direct_scene_tensors(
    atoms: dict[str, torch.Tensor],
    pairs: torch.Tensor,
    output: dict[str, torch.Tensor],
    chemical_predictions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Translate direct AtomPair output into the existing ChemScene compiler API."""

    if atoms["atom_mask"].size(0) != 1:
        raise ValueError("Direct scene compilation currently requires batch size 1")
    if pairs.shape != (len(output["presence_logits"]), 2):
        raise ValueError("Pair count differs from Direct AtomPair output")
    maximum_atoms = atoms["atom_mask"].size(1)
    chemical = output["chemical_logits"].float().softmax(-1)
    if chemical_predictions is not None:
        if chemical_predictions.shape != (len(pairs),):
            raise ValueError("Chemical prediction count differs from AtomPairs")
        changed = chemical_predictions != chemical.argmax(-1)
        chemical[changed] = torch.nn.functional.one_hot(
            chemical_predictions[changed], num_classes=chemical.size(-1)
        ).to(chemical.dtype)
    incidence = chemical.new_full((len(pairs), 2, maximum_atoms), -1e4)
    if len(pairs):
        rows = torch.arange(len(pairs), device=pairs.device)
        incidence[rows, 0, pairs[:, 0]] = 1e4
        incidence[rows, 1, pairs[:, 1]] = 1e4
    atom_logits = torch.where(
        atoms["atom_mask"][0],
        atoms["atom_scores"][0].new_tensor(1e4),
        atoms["atom_scores"][0].new_tensor(-1e4),
    )
    return {
        "atom_presence_logits": atom_logits,
        "bond_presence_logits": output["presence_logits"].float(),
        "bond_types": torch.cat((chemical, chemical), dim=-1),
        "bond_mask": torch.ones(
            len(pairs), device=pairs.device, dtype=torch.bool
        ),
        "incidence_logits": incidence,
    }


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
