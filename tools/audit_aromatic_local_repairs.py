"""Read-only replay of bounded local aromatic demotion candidates.

The chemical-head safety audit deliberately stores only aggregate scene
summaries.  This audit reconstructs the same fixed development witnesses from
the bound checkpoint and release manifest, then tests every one-edge aromatic
demotion on the final r3 arm.  A candidate is never accepted because RDKit
sanitizes alone: its canonical identity is compared with both the untouched
initial prediction and the released validation identity.

This is an audit tool, not an inference path.  It performs no training, does
not read sealed data, and refuses to overwrite an existing report.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

from rdkit import Chem, RDLogger
import torch

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.components.atoms.chemistry.aromatic_local_verifier import MAX_ATTEMPTS
from chemvector.evaluation.audits.aromatic_saved_replay import scene_digest
from chemvector.training.curriculum.bundle import load_bundle
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.validation_truth import canonical_molblock, expected_identity
from tools.audit_chemical_head_aromatic_safety import (
    INITIAL_SHA,
    MANIFEST_SHA,
    _fixed_cases,
)
from tools.run_pair_scale_trace_diagnostic import compare_prediction


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get(
    "CHEMVECTOR_AROMATIC_REPAIR_RUN",
    str(ROOT / "outputs/experiments/chemvector-chemical-head-probe-20261001-r3"),
))
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
MANIFEST = ROOT / "outputs/datasets/chemvector-expanded-release-20260915/manifest.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_AROMATIC_REPAIR_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-aromatic-local-repair-replay-20261001"),
))
STEPS = (0, 1200)


def _canonical_mol(molecule) -> str:
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _compile_candidate(scene: dict) -> tuple[bool, str | None, str | None]:
    try:
        molecule = mol_from_chemscene_v0_2(deepcopy(scene))
        if molecule is None:
            raise RuntimeError("Compiler returned no molecule")
        return True, _canonical_mol(molecule), None
    except (ValueError, RuntimeError) as error:
        return False, None, str(error)


def _candidate_scores(structure: dict, scene: dict, compiled_indices: list[int]) -> dict[int, float]:
    probabilities = structure["output"]["chemical_logits"].softmax(-1)[:, 3].detach().cpu().tolist()
    pairs = structure["pairs"].detach().cpu().tolist()
    pair_scores = {tuple(sorted(pair)): float(probabilities[row]) for row, pair in enumerate(pairs)}
    scores = {}
    for bond in scene["bond_objects"]:
        if bond["chemical_type"] != "AROMATIC":
            continue
        pair = tuple(sorted(compiled_indices[index] for index in bond["atom_refs"]))
        if pair not in pair_scores:
            raise ValueError("Aromatic scene bond is absent from fixed pair predictions")
        scores[int(bond["id"])] = pair_scores[pair]
    return scores


def _initial_chemical_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.pair.chemical.state_dict().items()}


def _load_chemical_state(model, initial_state: dict[str, torch.Tensor], step: int) -> None:
    if step == 0:
        model.pair.chemical.load_state_dict(initial_state, strict=True)
        return
    payload = torch.load(RUN / f"chemical-{step:06d}.pt", map_location="cpu", weights_only=True)
    if payload.get("initial_sha256") != INITIAL_SHA or payload.get("deployment_authorized") is not False:
        raise ValueError(f"Checkpoint binding changed: {step}")
    model.pair.chemical.load_state_dict(payload["chemical_state"], strict=True)


def _replay_arm(model, config, predictor, loader, manifest, row, saved, device, step: int) -> dict:
    sample = loader(manifest, row)
    if row["split"] != "validation" or sample.get("purpose") != "diagnostic":
        raise ValueError("Repair replay requires fixed diagnostic validation rows")
    pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
    trace = {}
    structure = predictor.predict_structure(model, config, pixels)
    prediction = predictor.from_structure(
        structure, model.role, model.bbox, pixels, sample["image"], trace=trace
    )
    if not compare_prediction(prediction, saved)["matches_saved"]:
        raise ValueError(f"Saved prediction parity failed: {row['source']}:{row['sample_key']}:{step}")
    scene = trace.get("scene_before_fallback")
    if scene is None:
        raise ValueError("Fixed replay did not expose the pre-fallback scene")
    scores = _candidate_scores(structure, scene, trace["compiled_candidate_indices"])
    truth = expected_identity(sample)
    baseline = canonical_molblock(prediction["molblock"])
    candidates = []
    for bond_id, score in sorted(scores.items(), key=lambda item: (item[1], item[0]))[:MAX_ATTEMPTS]:
        candidate = deepcopy(scene)
        bond = next(item for item in candidate["bond_objects"] if int(item["id"]) == bond_id)
        replacement = bond.get("display_type")
        if replacement not in {"SINGLE", "DOUBLE", "TRIPLE"}:
            replacement = "SINGLE"
        bond["chemical_type"] = replacement
        bond["display_type"] = replacement
        valid, identity, error = _compile_candidate(candidate)
        candidates.append({
            "bond_id": bond_id,
            "aromatic_probability": score,
            "replacement": replacement,
            "valid": valid,
            "identity": identity,
            "matches_untouched_arm": valid and baseline is not None and identity == baseline,
            "matches_truth": valid and identity == truth,
            "error": error,
            "changed_aromatic_bonds": 1,
            "candidate_scene_sha256": scene_digest(candidate),
        })
    return {
        "step": step,
        "baseline_valid": baseline is not None,
        "baseline_fallback": bool(prediction["fallback"]),
        "baseline_identity": baseline,
        "expected_identity": truth,
        "aromatic_bonds": len(scores),
        "candidates": candidates,
        "scene_sha256": scene_digest(scene),
        "sanitize_error": trace.get("sanitize_error"),
    }


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    if file_digest(INITIAL) != INITIAL_SHA or file_digest(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Initial checkpoint or release manifest changed")
    baseline, changed = _fixed_cases()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    loader = RecordLoader(manifest)
    rows = {row["source"] + ":" + row["sample_key"]: row for row in manifest["samples"]}
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded replay requires CUDA; no CPU fallback")
    RDLogger.DisableLog("rdApp.error")
    torch.set_num_threads(2)
    torch.manual_seed(20261001)
    torch.cuda.manual_seed_all(20261001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, config, payload = load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    predictor = FrozenOCRPredictor(payload["provenance"], device)
    initial_state = _initial_chemical_state(model)
    results = []
    with torch.no_grad():
        for key in sorted(changed):
            if key not in rows:
                raise ValueError(f"Fixed witness missing from manifest: {key}")
            arms = {}
            for step in STEPS:
                _load_chemical_state(model, initial_state, step)
                saved_path = RUN / ("baseline-summary.json" if step == 0 else f"validation-{step:06d}.json")
                saved = {item["key"]: item for item in json.loads(saved_path.read_text(encoding="utf-8"))["records"]}[key]
                arms[str(step)] = _replay_arm(model, config, predictor, loader, manifest, rows[key], saved, device, step)
            # The final candidate is also compared with the untouched initial
            # arm.  This is intentionally a readout only; it never chooses a
            # candidate or supplies a target to inference.
            initial_identity = arms["0"]["baseline_identity"]
            for candidate in arms["1200"]["candidates"]:
                candidate["matches_initial_arm"] = (
                    candidate["valid"]
                    and initial_identity is not None
                    and candidate["identity"] == initial_identity
                )
            results.append({"key": key, "domain": changed[key]["domain"], "arms": arms})

    final_candidates = [candidate for case in results for candidate in case["arms"]["1200"]["candidates"]]
    truth_matches = [candidate for candidate in final_candidates if candidate["matches_truth"]]
    valid = [candidate for candidate in final_candidates if candidate["valid"]]
    report = {
        "schema_version": "chemvector-aromatic-local-repair-replay-0.1",
        "complete": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_modified": False,
        "scope": {"fixed_changed_development_images": len(results), "steps": list(STEPS), "max_attempts": MAX_ATTEMPTS},
        "source_sha256": {
            "initial.pt": file_digest(INITIAL),
            "manifest.json": file_digest(MANIFEST),
            "chemical_head_safety_report.json": file_digest(ROOT / "outputs/audits/chemvector-chemical-head-aromatic-safety-20261001-r3/report.json"),
        },
        "policy": {
            "sanitize_is_not_identity": True,
            "repair_requires_identity_match": True,
            "max_changed_aromatic_bonds_per_candidate": 1,
            "production_integration": "not_authorized",
        },
        "aggregate": {
            "cases": len(results),
            "final_candidates": len(final_candidates),
            "final_valid_candidates": len(valid),
            "final_truth_matching_candidates": len(truth_matches),
            "final_initial_arm_matching_candidates": sum(candidate["matches_initial_arm"] for candidate in final_candidates),
            "new_fallback_cases": sum(not case["arms"]["0"]["baseline_fallback"] and case["arms"]["1200"]["baseline_fallback"] for case in results),
            "initial_fallback_cases_with_truth_matching_repair": sum(
                case["arms"]["0"]["baseline_fallback"] and any(c["matches_truth"] for c in case["arms"]["1200"]["candidates"])
                for case in results
            ),
        },
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    value = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "cases": value["aggregate"]["cases"]}, ensure_ascii=False))
