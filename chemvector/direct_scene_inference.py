"""Direct AtomPair Bond inference components.

``chemvector.infer_direct_atom_pair_bond`` is the pre-registered CLI module
name (it is launched as a subprocess by audits); this module owns the
checkpoint loaders and the scene/OCR/suppression algorithms so the entry
stays a thin orchestration re-exporting them for historical import paths.
"""

from __future__ import annotations

from Data.sdf.chemscene_v0_2 import (
    QUANTIZATION_MAX,
    _clip_port_to_text_bbox,
    parse_chemscene_v0_2,
)
import torch

from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
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