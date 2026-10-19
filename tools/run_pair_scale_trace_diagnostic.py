"""Bounded GPU read-only trace for the saved pair-scale regression.

The worker runs the unchanged production prediction path on 15 already exposed
full-molecule development images, using the initial bundle and the saved final
pair delta.  It performs no backward pass, optimizer step, deployment, or
identity query.  The trace is post-hoc diagnosis only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.direct_scene_inference import apply_atom_ocr
from chemvector.evaluation.shadows.shadow_chemscene_v02_oracle import (
    apply_aromatic_consistency,
    build_oracle_chemistry_shadow_scene,
)
from chemvector.placeholder_scene import placeholder_scene
from chemvector.training.curriculum.bundle import load_bundle
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from chemvector.training.curriculum.validation_truth import canonical_molblock
from chemvector.components.atoms.chemistry.direct_atom_pair_chemistry import (
    promote_complete_aromatic_rings,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/experiments/chemvector-pair-scale-20260924"
RELEASE = ROOT / "outputs/datasets/chemvector-expanded-release-20260915"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TRACE_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-pair-scale-trace-20260925"),
))
MANIFEST = RELEASE / "manifest.json"
BASELINE = RUN / "validation-000000.json"
FINAL = RUN / "validation-015443.json"
DELTA = RUN / "pair-015443.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MANIFEST_SHA = "67ccb1307f7af3ca19f16ee6fd900d1a2e8c3c62dd0b9c0ec28704ddd93aa05e"
BASELINE_SHA = "e34db4c34a124ad797f979d2714da26bd3daa3f98d6574b88339bd8dfd69035c"
FINAL_SHA = "e0ef7568960920ac7e37a786ad635f7a9c337af3339ec6f479e48cc78a882a3e"
DELTA_SHA = "a1191c86007440e016f653077a99437938f690318747a700722728f8c8b89573"
MAX_SECONDS = 180.0
PAIR_THRESHOLD = 0.85
ROLE_THRESHOLD = 0.85
ATOM_THRESHOLD = -0.25

BROKEN_KEYS = (
    "fresh_dev:fresh_dev_000120_coordgen",
    "fresh_dev:fresh_dev_000221_rdkit",
    "fresh_dev:fresh_dev_000263_rdkit",
    "fresh_dev:fresh_dev_000314_rdkit",
    "fresh_dev:fresh_dev_000676_stress",
    "fresh_dev:fresh_dev_000746_rdkit",
    "fresh_dev:fresh_dev_000953_stress",
    "fresh_dev:fresh_dev_001091_rdkit",
    "fresh_dev:fresh_dev_001900_coordgen",
)
RESCUED_KEYS = (
    "fresh_dev:fresh_dev_000883_rdkit",
    "fresh_dev:fresh_dev_001800_coordgen",
    "fresh_dev:fresh_dev_001539_coordgen",
    "fresh_dev:fresh_dev_000437_coordgen",
    "fresh_dev:fresh_dev_000054_rdkit",
    "fresh_dev:fresh_dev_001091_coordgen",
)
KEYS = BROKEN_KEYS + RESCUED_KEYS


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scene_digest(scene: dict) -> str:
    return sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scene_summary(scene: dict) -> dict:
    atoms = scene.get("atom_objects", [])
    bonds = scene.get("bond_objects", [])
    aromatic_atoms = sum(bool(a.get("chemical", {}).get("aromatic")) for a in atoms)
    aromatic_bonds = sum(b.get("chemical_type") == "AROMATIC" for b in bonds)
    return {
        "atoms": len(atoms),
        "bonds": len(bonds),
        "aromatic_atoms": aromatic_atoms,
        "aromatic_bonds": aromatic_bonds,
        "bond_types": {kind: sum(b.get("chemical_type") == kind for b in bonds) for kind in BOND_TYPES},
    }


def pair_rows(structure: dict, *, threshold: float = PAIR_THRESHOLD) -> list[dict]:
    pairs = structure["pairs"].detach().cpu()
    presence = structure["output"]["presence_logits"].detach().float().cpu()
    chemical = structure["output"]["chemical_logits"].detach().float().cpu()
    final_chemical = structure["chemistry"].detach().cpu()
    probabilities = presence.sigmoid()
    chemical_probabilities = chemical.softmax(-1)
    rows: list[dict] = []
    for index, pair in enumerate(pairs.tolist()):
        values = chemical_probabilities[index]
        top = torch.topk(values, k=min(2, len(BOND_TYPES))).values.tolist()
        rows.append({
            "row": index,
            "atom_candidates": [int(pair[0]), int(pair[1])],
            "presence_logit": float(presence[index]),
            "presence_probability": float(probabilities[index]),
            "selected_at_pair_threshold": bool(probabilities[index] >= threshold),
            "chemical_logits": [float(value) for value in chemical[index].tolist()],
            "chemical_probabilities": [float(value) for value in values.tolist()],
            "raw_chemical": BOND_TYPES[int(chemical[index].argmax())],
            "final_chemical": BOND_TYPES[int(final_chemical[index])],
            "aromatic_probability": float(values[BOND_TYPES.index("AROMATIC")]),
            "chemical_top2_margin": float(top[0] - top[1]) if len(top) == 2 else None,
        })
    return rows


def promotion_summary(structure: dict) -> dict:
    pairs = structure["pairs"]
    output = structure["output"]
    _, details = promote_complete_aromatic_rings(
        pairs,
        output["presence_logits"].sigmoid(),
        output["chemical_logits"],
        presence_threshold=PAIR_THRESHOLD,
        mean_threshold=.40,
        minimum_threshold=.05,
    )
    raw = output["chemical_logits"].argmax(-1)
    final = structure["chemistry"]
    aromatic_id = BOND_TYPES.index("AROMATIC")
    return {
        **details,
        "selected_pairs": int((output["presence_logits"].sigmoid() >= PAIR_THRESHOLD).sum()),
        "raw_aromatic_selected": int(((output["presence_logits"].sigmoid() >= PAIR_THRESHOLD) & (raw == aromatic_id)).sum()),
        "final_aromatic_selected": int(((output["presence_logits"].sigmoid() >= PAIR_THRESHOLD) & (final == aromatic_id)).sum()),
        "raw_to_final_changed": int((raw != final).sum()),
    }


def compiler_summary(structure: dict, trace: dict) -> dict:
    """Reconstruct only compiler diagnostics; no truth or pixels enter."""
    atoms, tensors = structure["atoms"], structure["tensors"]
    selected = trace["compiled_candidate_indices"]
    predictions = {int(item["candidate"]): {k: v for k, v in item.items() if k != "candidate"}
                   for item in trace.get("ocr", [])}
    scene, diagnostics = build_oracle_chemistry_shadow_scene(
        molecule_id="pair-scale-trace",
        gt_scene=placeholder_scene("pair-scale-trace", 768, 768),
        atom_geometry=atoms["atom_geometry"][0].float().cpu(),
        atom_mask=atoms["atom_mask"][0].cpu(),
        atom_presence_logits=tensors["atom_presence_logits"].float().cpu(),
        bond_types=tensors["bond_types"].float().cpu(),
        bond_mask=tensors["bond_mask"].cpu(),
        bond_presence_logits=tensors["bond_presence_logits"].float().cpu(),
        incidence_logits=tensors["incidence_logits"].float().cpu(),
        atom_threshold=0.0,
        bond_threshold=torch.tensor(PAIR_THRESHOLD).logit().item(),
    )
    if diagnostics["selected_atom_candidate_indices"] != selected:
        raise ValueError("Compiler selected candidates differ from saved production trace")
    ocr_application = apply_atom_ocr(scene, selected, predictions)
    scene, aromatic_consistency = apply_aromatic_consistency(scene, "demote-noncycle-clear-atoms")
    return {
        "diagnostics": diagnostics,
        "ocr_application": ocr_application,
        "aromatic_consistency": aromatic_consistency,
        "scene_before_fallback_sha256": scene_digest(trace["scene_before_fallback"]),
        "reconstructed_scene_before_fallback_sha256": scene_digest(scene),
        "scene_summary": scene_summary(trace["scene_before_fallback"]),
    }


def compare_prediction(prediction: dict, saved: dict) -> dict:
    canonical = canonical_molblock(prediction["molblock"])
    actual = {
        "predicted_smiles": canonical,
        "valid": canonical is not None,
        "exact": canonical is not None and canonical == saved["expected_smiles"],
        "fallback": bool(prediction["fallback"]),
        "error": prediction.get("error"),
        "molblock_sha256": sha256((prediction.get("molblock") or "").encode()).hexdigest(),
    }
    expected = {key: saved[key] for key in ("predicted_smiles", "valid", "exact", "fallback", "error")}
    expected["molblock_sha256"] = sha256((saved.get("molblock") or "").encode()).hexdigest()
    return {"matches_saved": actual == expected, "actual": actual, "saved": expected}


def make_plan() -> dict:
    baseline = read_json(BASELINE)["records"]
    final = read_json(FINAL)["records"]
    old, new = {row["key"]: row for row in baseline}, {row["key"]: row for row in final}
    if set(old) != set(new) or len(old) != len(baseline):
        raise ValueError("Saved validation cohorts are not identical")
    selected = []
    for key in KEYS:
        if key not in old or old[key]["domain"] != "full_molecule":
            raise ValueError("Fixed trace key is missing or not full_molecule: " + key)
        expected_class = "broken" if key in BROKEN_KEYS else "rescued"
        if expected_class == "broken" and not (old[key]["exact"] and not new[key]["exact"]):
            raise ValueError("Broken witness no longer matches saved outcome: " + key)
        if expected_class == "rescued" and not (not old[key]["exact"] and new[key]["exact"]):
            raise ValueError("Rescued witness no longer matches saved outcome: " + key)
        selected.append({"key": key, "class": expected_class,
                         "baseline_exact": old[key]["exact"], "final_exact": new[key]["exact"]})
    return {
        "schema_version": "chemvector-pair-scale-trace-plan-0.1",
        "user_statement": "那就按照你说的干",
        "purpose": "post_hoc_saved_pair_scale_regression_trace",
        "samples": selected,
        "arms": ["initial", "pair-015443"],
        "budget": {"wall_seconds": MAX_SECONDS, "gpu_devices": ["0"], "model_forwards": 30,
                   "optimizer_steps": 0, "backward_passes": 0, "automatic_retries": 0},
        "thresholds": {"atom": ATOM_THRESHOLD, "pair": PAIR_THRESHOLD, "role": ROLE_THRESHOLD,
                       "aromatic_mean": .40, "aromatic_minimum": .05},
        "source_sha256": {
            "manifest.json": MANIFEST_SHA,
            "initial.pt": INITIAL_SHA,
            "validation-000000.json": BASELINE_SHA,
            "validation-015443.json": FINAL_SHA,
            "pair-015443.pt": DELTA_SHA,
        },
        "read_images": True,
        "read_sealed_test": False,
        "new_identity_queries": 0,
        "training_authorized": False,
        "deployment_authorized": False,
    }


def worker():
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Trace worker requires explicitly selected GPU 0")
    if digest(INITIAL) != INITIAL_SHA or digest(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Initial bundle or released manifest changed")
    if digest(BASELINE) != BASELINE_SHA or digest(FINAL) != FINAL_SHA or digest(DELTA) != DELTA_SHA:
        raise ValueError("Saved pair-scale artifacts changed")
    plan = make_plan()
    write_json(OUTPUT / "plan.json", {**plan, "implementation_sha256": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}})
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA probe failed; trace has no CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    environment = {"device": str(device), "name": props.name, "uuid": str(props.uuid),
                   "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                   "dtype": "float32", "tf32": False, "autocast": False}
    manifest = read_json(MANIFEST)
    saved = {"initial": {row["key"]: row for row in read_json(BASELINE)["records"]},
             "pair-015443": {row["key"]: row for row in read_json(FINAL)["records"]}}
    loader = RecordLoader(manifest)
    base_model, config, payload = load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    if payload.get("progress", {}).get("optimizer_steps") != 0:
        raise ValueError("Initial bundle is not the zero-step baseline")
    if any(payload["provenance"][key] != value for key, value in {
        "atom_threshold": ATOM_THRESHOLD, "pair_threshold": PAIR_THRESHOLD,
        "role_threshold": ROLE_THRESHOLD}.items()):
        raise ValueError("Inference thresholds differ from the saved production run")
    delta = torch.load(DELTA, map_location="cpu", weights_only=True)
    if (delta.get("schema_version") != "pair-scale-delta-0.1"
            or delta.get("initial_sha256") != INITIAL_SHA or delta.get("deployment_authorized") is not False):
        raise ValueError("Final pair delta binding is invalid")
    if delta.get("architecture") != asdict(base_model.pair.config):
        raise ValueError("Final pair architecture differs from initial bundle")
    models = {}
    for arm in ("initial", "pair-015443"):
        model, arm_config, arm_payload = load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
        if asdict(arm_config) != asdict(config):
            raise ValueError("Arm proposal configuration changed")
        if arm == "pair-015443":
            model.pair.load_state_dict(delta["pair_state"], strict=True)
        model.to(device).eval().requires_grad_(False)
        models[arm] = (model, FrozenOCRPredictor(arm_payload["provenance"], device))
    results = []
    with torch.inference_mode():
        for item in plan["samples"]:
            if time.monotonic() - started >= MAX_SECONDS - 20:
                raise TimeoutError("Trace budget exhausted before fixed 15-case scope completed")
            key = item["key"]
            row = next(row for row in manifest["samples"] if row["source"] + ":" + row["sample_key"] == key)
            sample = loader(manifest, row)
            if sample.get("purpose") != "diagnostic" or row["split"] != "validation":
                raise ValueError("Trace input is not fixed diagnostic validation")
            case = {"key": key, "class": item["class"], "arms": {}}
            for arm, (model, predictor) in models.items():
                pixels = __import__("chemvector.training.curriculum.pixels", fromlist=["image_tensor"]).image_tensor(sample["image"]).unsqueeze(0).to(device)
                structure = predictor.predict_structure(model, config, pixels)
                trace = {}
                prediction = predictor.from_structure(structure, model.role, model.bbox, pixels, sample["image"], trace=trace)
                parity = compare_prediction(prediction, saved[arm][key])
                if not parity["matches_saved"]:
                    raise ValueError(f"Saved prediction parity failed: {arm} {key}")
                rows = pair_rows(structure)
                case["arms"][arm] = {
                    "prediction_parity": parity,
                    "pair_rows": rows,
                    "promotion": promotion_summary(structure),
                    "compiler": compiler_summary(structure, trace),
                    "trace_summary": {
                        "compiled_candidate_count": len(trace.get("compiled_candidate_indices", [])),
                        "ocr_labels": len(trace.get("ocr", [])),
                        "fallback": bool(prediction["fallback"]),
                        "sanitize_error": trace.get("sanitize_error"),
                        "compile_error": trace.get("compile_error"),
                        "scene_after_fallback": scene_summary({"atom_objects": trace.get("atoms_after_fallback", []),
                                                                "bond_objects": trace.get("scene_before_fallback", {}).get("bond_objects", [])}),
                    },
                }
                del pixels, structure, trace, prediction
            results.append(case)
            print(json.dumps({"completed": len(results), "total": len(plan["samples"]), "key": key}, ensure_ascii=False), flush=True)
    if len(results) != 15 or any(not all(arm["prediction_parity"]["matches_saved"] for arm in case["arms"].values()) for case in results):
        raise ValueError("Incomplete or non-parity trace")
    summary = {
        "cases": len(results),
        "arms": {arm: {"cases": len(results), "prediction_parity": all(case["arms"][arm]["prediction_parity"]["matches_saved"] for case in results)}
                 for arm in ("initial", "pair-015443")},
        "broken": sum(case["class"] == "broken" for case in results),
        "rescued": sum(case["class"] == "rescued" for case in results),
        "presence_selected_delta_cases": sum(case["arms"]["initial"]["promotion"]["selected_pairs"] != case["arms"]["pair-015443"]["promotion"]["selected_pairs"] for case in results),
        "promotion_delta_cases": sum(case["arms"]["initial"]["promotion"]["accepted_cycles"] != case["arms"]["pair-015443"]["promotion"]["accepted_cycles"] for case in results),
        "compiler_bond_count_delta_cases": sum(case["arms"]["initial"]["compiler"]["scene_summary"]["bonds"] != case["arms"]["pair-015443"]["compiler"]["scene_summary"]["bonds"] for case in results),
        "new_fallback_cases": sum(not case["arms"]["initial"]["trace_summary"]["fallback"] and case["arms"]["pair-015443"]["trace_summary"]["fallback"] for case in results),
    }
    write_json(OUTPUT / "report.json", {
        "schema_version": "chemvector-pair-scale-trace-report-0.1", "complete": True,
        "summary": summary, "environment": environment, "results": results,
        "source_sha256": plan["source_sha256"], "model_forwards": 30,
        "optimizer_steps": 0, "backward_passes": 0, "gpu_used": True,
        "training_performed": False, "deployment_authorized": False,
        "sealed_test_read": False, "new_identity_queries": 0,
        "limitations": ["Post-hoc trace of 15 affected development images; not an accuracy estimate.",
                        "Pair outputs identify observable changes but not a historical optimizer causal step.",
                        "No hidden features, images, or full scene payloads are saved in the report."],
        "elapsed_seconds": time.monotonic() - started,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    if OUTPUT.exists():
        raise FileExistsError("Trace output exists; no retry or overwrite")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "tools.run_pair_scale_trace_diagnostic", "--worker"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    receipt = {"schema_version": "chemvector-trace-runtime-0.1", "command": command,
               "wall_budget_seconds": MAX_SECONDS, "gpu_devices": ["0"],
               "training": False, "automatic_retries": 0, "status": "launch_failed"}
    process = None
    try:
        with (OUTPUT / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=MAX_SECONDS)
                receipt.update(status="completed" if code == 0 else "worker_failed", exit_code=code)
            except subprocess.TimeoutExpired:
                receipt.update(status="timeout")
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        receipt["elapsed_seconds"] = time.monotonic() - started
        with (OUTPUT / "runtime.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
