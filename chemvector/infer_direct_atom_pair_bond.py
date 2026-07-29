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
from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
    enumerate_unordered_atom_pairs,
)
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


DEFAULT_CHECKPOINT = (
    "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
)


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


def direct_scene_tensors(
    atoms: dict[str, torch.Tensor],
    pairs: torch.Tensor,
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Translate direct AtomPair output into the existing ChemScene compiler API."""

    if atoms["atom_mask"].size(0) != 1:
        raise ValueError("Direct scene compilation currently requires batch size 1")
    if pairs.shape != (len(output["presence_logits"]), 2):
        raise ValueError("Pair count differs from Direct AtomPair output")
    maximum_atoms = atoms["atom_mask"].size(1)
    chemical = output["chemical_logits"].float().softmax(-1)
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
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--grounded-graph-checkpoint")
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

    compiler = direct_scene_tensors(atoms, pairs, direct_output)
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
    scene, aromatic = apply_aromatic_consistency(
        scene, "demote-noncycle-clear-atoms"
    )
    diagnostics["aromatic"] = aromatic
    diagnostics["direct_atom_pairs"] = len(pairs)
    diagnostics["direct_presence_threshold"] = probability_threshold
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
        "chemistry_mode": "all_text_atoms_as_carbon_placeholder",
        "inference_seconds": time.perf_counter() - started,
        "vertices": int(atoms["atom_mask"].sum()),
        "candidate_atom_pairs": len(pairs),
        "selected_bonds": selected_bonds,
        "presence_threshold": probability_threshold,
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
