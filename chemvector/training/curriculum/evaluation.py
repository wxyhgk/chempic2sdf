"""Fixed full-molecule validation with frozen OCR and no target-fed inference."""

from copy import deepcopy
import math

from rdkit import Chem
import torch

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings
from chemvector.direct_atom_role import enumerate_selected_atoms
from chemvector.direct_scene_inference import direct_scene_tensors, apply_atom_ocr
from chemvector.ctc_ocr import decode_ctc_with_score
from chemvector.evaluate_direct_atom_ocr import crop_tensor, load_ctc
from chemvector.placeholder_scene import placeholder_scene, compile_carbon_placeholder
from chemvector.shadow_chemscene_v02_oracle import build_oracle_chemistry_shadow_scene, apply_aromatic_consistency
from .checkpoints import file_digest
from .continuation import structure_forward
from .pixels import image_tensor
from .validation_truth import canonical_molblock, expected_identity


class FrozenOCRPredictor:
    def __init__(self, provenance, device):
        record = provenance["checkpoints"]["ocr_frozen_not_loaded"]
        if file_digest(record["path"]) != record["sha256"]:
            raise ValueError("Frozen OCR hash mismatch")
        self.ocr, _ = load_ctc(record["path"], device)
        self.ocr.eval().requires_grad_(False)
        self.provenance = provenance
        for name in ("pair_threshold", "role_threshold"):
            if not 0 < provenance[name] < 1:
                raise ValueError("Fixed inference thresholds must be probabilities")

    @torch.no_grad()
    def __call__(self, model, config, pixels, image):
        # No graph, target label, target count, or GT bounding box is accepted.
        structure = self.predict_structure(model, config, pixels)
        return self.from_structure(structure, model.role, model.bbox, pixels, image)

    @torch.no_grad()
    def predict_structure(self, model, config, pixels):
        """Cache only model predictions, so text-interface audits can hold them fixed."""
        p = self.provenance
        features, _, _, _, _, _, atoms = structure_forward(model, pixels, config, p["atom_threshold"])
        return self.from_predicted_atoms(model.pair, features, atoms)

    @torch.no_grad()
    def from_predicted_atoms(self, pair, features, atoms):
        """Same pair/compiler tensors, accepting only cached model predictions."""
        p = self.provenance
        pairs, rows = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        output = pair(features, atoms["atom_features"], atoms["atom_geometry"][..., :2],
                      atoms["atom_scores"], pairs, rows)
        if any(not bool(torch.isfinite(output[k]).all()) for k in ("presence_logits", "chemical_logits")):
            raise ValueError("Non-finite validation predictions")
        chemistry, _ = promote_complete_aromatic_rings(pairs, output["presence_logits"].sigmoid(),
            output["chemical_logits"], presence_threshold=p["pair_threshold"], mean_threshold=.40, minimum_threshold=.05)
        tensors = direct_scene_tensors(atoms, pairs, output, chemistry)
        return {"atoms": atoms, "pairs": pairs, "output": output, "chemistry": chemistry, "tensors": tensors}

    @torch.no_grad()
    def from_structure(self, structure, role, bbox, pixels, image, *, trace=None):
        """No truth input or structure edits; optional trace does not affect decisions."""
        p = self.provenance
        atoms, tensors = structure["atoms"], structure["tensors"]
        atom_indices, sample_indices = enumerate_selected_atoms(atoms["atom_mask"])
        probabilities = role(pixels, atoms["atom_geometry"][..., :2], atom_indices, sample_indices).sigmoid()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("Non-finite text role predictions")
        text = probabilities >= p["role_threshold"]
        boxes = bbox(pixels, atoms["atom_geometry"][..., :2], atom_indices[text], sample_indices[text])
        if any(not bool(torch.isfinite(boxes[k]).all()) for k in ("centers", "half_extents")):
            raise ValueError("Non-finite text box predictions")
        crops = []
        for center, extent in zip(boxes["centers"], boxes["half_extents"]):
            x, y = (center.cpu() * 768).tolist()
            w, h = (extent.cpu() * 768).tolist()
            crops.append(crop_tensor(image, (x - w, y - h, x + w, y + h)))
        predictions = {}
        if crops:
            decoded = [decode_ctc_with_score(row.float()) for row in self.ocr(torch.stack(crops).to(pixels.device))]
            for i, ((label, score), index) in enumerate(zip(decoded, atom_indices[text].tolist())):
                predictions[index] = {"text": label, "score": score,
                    "center": boxes["centers"][i].cpu().tolist(), "half_extent": boxes["half_extents"][i].cpu().tolist()}
        if trace is not None:
            trace["roles"] = [{"candidate": int(index), "probability": float(probability), "text": bool(is_text)}
                for index, probability, is_text in zip(atom_indices, probabilities, text)]
            trace["ocr"] = [{"candidate": int(index), **value} for index, value in predictions.items()]
        try:
            # Existing compiler API has a historical "oracle" name. Its scene
            # here is EMPTY, exactly as production upload inference; no truth.
            scene, diagnostics = build_oracle_chemistry_shadow_scene(molecule_id="validation-prediction",
                gt_scene=placeholder_scene("validation-prediction", 768, 768),
                atom_geometry=atoms["atom_geometry"][0].float().cpu(), atom_mask=atoms["atom_mask"][0].cpu(),
                atom_presence_logits=tensors["atom_presence_logits"].float().cpu(),
                bond_types=tensors["bond_types"].float().cpu(), bond_mask=tensors["bond_mask"].cpu(),
                bond_presence_logits=tensors["bond_presence_logits"].float().cpu(),
                incidence_logits=tensors["incidence_logits"].float().cpu(), atom_threshold=0.,
                bond_threshold=math.log(p["pair_threshold"] / (1 - p["pair_threshold"])))
            ocr_application = apply_atom_ocr(scene, diagnostics["selected_atom_candidate_indices"], predictions)
            scene, _ = apply_aromatic_consistency(scene, "demote-noncycle-clear-atoms")
            if trace is not None:
                trace["compiled_candidate_indices"] = diagnostics["selected_atom_candidate_indices"]
                trace["ocr_application"] = ocr_application
                trace["atoms_before_fallback"] = deepcopy(scene["atom_objects"])
                trace["scene_before_fallback"] = deepcopy(scene)
            fallback = False
            try:
                molecule = mol_from_chemscene_v0_2(scene)
            except (ValueError, RuntimeError) as error:
                if trace is not None:
                    trace["sanitize_error"] = str(error)
                scene, molecule, _ = compile_carbon_placeholder(scene)
                fallback = True
            if trace is not None:
                trace["atoms_after_fallback"] = deepcopy(scene["atom_objects"])
            block = Chem.MolToMolBlock(molecule)
            return {"molblock": block, "fallback": fallback, "error": None}
        except (ValueError, RuntimeError) as error:
            if trace is not None:
                trace["compile_error"] = str(error)
            return {"molblock": None, "fallback": False, "error": str(error)}


def summarize_predictions(records, baseline=None):
    by_key = {r["key"]: r for r in records}
    if not records or len(by_key) != len(records):
        raise ValueError("Empty/duplicate validation predictions")
    total = len(records)
    result = {"images": total, "exact": sum(r["exact"] for r in records),
              "valid": sum(r["valid"] for r in records), "fallback": sum(r["fallback"] for r in records)}
    result["sdf_exact_rate"] = result["exact"] / total
    if baseline is not None:
        old = {r["key"]: r for r in baseline}
        if len(old) != len(baseline) or old.keys() != by_key.keys():
            raise ValueError("Paired validation sample identities changed")
        result["rescued"] = sum(not old[k]["exact"] and r["exact"] for k, r in by_key.items())
        result["broken"] = sum(old[k]["exact"] and not r["exact"] for k, r in by_key.items())
    return result


@torch.no_grad()
def evaluate(model, config, manifest, record_loader, predictor, *, device):
    previous_mode = model.training
    model.eval()
    results = []
    try:
        for row in manifest["samples"]:
            if row["split"] != "validation":
                continue
            sample = record_loader(manifest, row)
            if sample.get("purpose") != "diagnostic":
                raise ValueError("Validation requires a diagnostic-only sample")
            # No glyph records, GT proposals, query capacity or training targets
            # are required to evaluate an original image. The pixel operations
            # are exactly the same as the training collator's previous formula.
            pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
            predicted = predictor(model, config, pixels, sample["image"])
            canonical = canonical_molblock(predicted["molblock"])
            expected = expected_identity(sample)
            results.append({"key": row["source"] + ":" + row["sample_key"], "domain": row["domain"],
                "valid": canonical is not None, "exact": canonical is not None and canonical == expected,
                "predicted_smiles": canonical, "expected_smiles": expected, **predicted})
    finally:
        model.train(previous_mode)
    return results
