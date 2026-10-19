"""Read-only r3 attribution for the three newly introduced fallback witnesses.

This is deliberately narrower than a training experiment. It reruns only the
fixed initial and 1200-step chemical-head arms, verifies saved prediction
parity, and records whether the regression comes from pair presence, bond
classification, aromatic promotion, or compilation. No gradients, optimizer
steps, deployment changes, or sealed-test reads are allowed.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.bundle import load_bundle
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.validation_truth import canonical_molblock
from tools.audit_chemical_head_aromatic_safety import INITIAL_SHA, MANIFEST_SHA
from tools.run_pair_scale_trace_diagnostic import pair_rows, promotion_summary, compare_prediction


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/experiments/chemvector-chemical-head-probe-20261001-r3"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
MANIFEST = ROOT / "outputs/datasets/chemvector-expanded-release-20260915/manifest.json"
SAFETY = ROOT / "outputs/audits/chemvector-chemical-head-aromatic-safety-20261001-r3/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_AROMATIC_ATTRIBUTION_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-aromatic-upstream-attribution-20261001"),
))


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _state(model, initial, step):
    if step == 0:
        model.pair.chemical.load_state_dict(initial, strict=True)
        return
    payload = torch.load(RUN / f"chemical-{step:06d}.pt", map_location="cpu", weights_only=True)
    if payload.get("initial_sha256") != INITIAL_SHA or payload.get("deployment_authorized") is not False:
        raise ValueError("Invalid r3 chemical checkpoint binding")
    model.pair.chemical.load_state_dict(payload["chemical_state"], strict=True)


def _scene_edges(scene, compiled_indices):
    return {
        tuple(sorted(compiled_indices[index] for index in bond["atom_refs"])): {
            "bond_id": int(bond["id"]),
            "chemical_type": bond["chemical_type"],
            "display_type": bond["display_type"],
        }
        for bond in scene.get("bond_objects", [])
    }


def run(output: Path = OUTPUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite attribution output: {output}")
    if file_digest(INITIAL) != INITIAL_SHA or file_digest(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Initial checkpoint or manifest changed")
    safety = json.loads(SAFETY.read_text(encoding="utf-8"))
    keys = sorted(
        row["key"] for row in safety["results"]
        if not row["arms"]["0"]["fallback"] and row["arms"]["1200"]["fallback"]
    )
    if keys != [
        "fresh_dev:fresh_dev_001401_coordgen",
        "fresh_dev:fresh_dev_001833_original",
        "fresh_dev:fresh_dev_001866_stress",
    ]:
        raise ValueError("r3 new-fallback witness set changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {row["source"] + ":" + row["sample_key"]: row for row in manifest["samples"]}
    saved = {
        0: {row["key"]: row for row in json.loads((RUN / "baseline-summary.json").read_text())["records"]},
        1200: {row["key"]: row for row in json.loads((RUN / "validation-001200.json").read_text())["records"]},
    }
    loader = RecordLoader(manifest)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
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
    initial_chemical = {name: value.detach().cpu().clone() for name, value in model.pair.chemical.state_dict().items()}
    result = []
    with torch.no_grad():
        for key in keys:
            arms = {}
            row = rows[key]
            sample = loader(manifest, row)
            if row["split"] != "validation" or sample.get("purpose") != "diagnostic":
                raise ValueError("Witness is not fixed diagnostic validation")
            for step in (0, 1200):
                _state(model, initial_chemical, step)
                pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
                trace = {}
                structure = predictor.predict_structure(model, config, pixels)
                prediction = predictor.from_structure(
                    structure, model.role, model.bbox, pixels, sample["image"], trace=trace
                )
                if not compare_prediction(prediction, saved[step][key])["matches_saved"]:
                    raise ValueError(f"Saved prediction parity failed: {step}:{key}")
                rows_now = pair_rows(structure)
                selected = {tuple(item["atom_candidates"]): item for item in rows_now if item["selected_at_pair_threshold"]}
                scene = trace["scene_before_fallback"]
                edges = _scene_edges(scene, trace["compiled_candidate_indices"])
                arms[str(step)] = {
                    "fallback": bool(prediction["fallback"]),
                    "valid": canonical_molblock(prediction["molblock"]) is not None,
                    "exact": bool(saved[step][key]["exact"]),
                    "predicted_smiles": canonical_molblock(prediction["molblock"]),
                    "sanitize_error": trace.get("sanitize_error"),
                    "presence_signature": _digest(sorted(selected)),
                    "selected_pairs": len(selected),
                    "promotion": promotion_summary(structure),
                    "selected": selected,
                    "scene_edges": edges,
                    "scene_sha256": _digest(scene),
                }
                del pixels, structure, trace, prediction
            before, after = arms["0"], arms["1200"]
            if before["presence_signature"] != after["presence_signature"]:
                raise ValueError(f"Presence/pair selection changed: {key}")
            changed = []
            for pair in sorted(before["selected"]):
                left, right = before["selected"][pair], after["selected"][pair]
                if left["raw_chemical"] != right["raw_chemical"] or left["final_chemical"] != right["final_chemical"]:
                    changed.append({
                        "pair": list(pair), "row": left["row"],
                        "raw_before": left["raw_chemical"], "raw_after": right["raw_chemical"],
                        "final_before": left["final_chemical"], "final_after": right["final_chemical"],
                        "aromatic_probability_before": left["aromatic_probability"],
                        "aromatic_probability_after": right["aromatic_probability"],
                        "presence_probability_before": left["presence_probability"],
                        "presence_probability_after": right["presence_probability"],
                    })
            changed_edges = []
            for pair in sorted(set(before["scene_edges"]) | set(after["scene_edges"])):
                if before["scene_edges"].get(pair) != after["scene_edges"].get(pair):
                    changed_edges.append({"pair": list(pair), "before": before["scene_edges"].get(pair), "after": after["scene_edges"].get(pair)})
            result.append({"key": key, "arms": {
                "0": {k: v for k, v in before.items() if k not in {"selected", "scene_edges"}},
                "1200": {k: v for k, v in after.items() if k not in {"selected", "scene_edges"}},
            }, "pair_presence_changed": False, "chemical_changes": changed, "scene_edge_changes": changed_edges})
    report = {
        "schema_version": "chemvector-aromatic-upstream-attribution-0.1",
        "complete": True,
        "scope": {"keys": keys, "arms": [0, 1200], "model_forwards": len(keys) * 2, "read_images": True},
        "aggregate": {
            "cases": len(result),
            "presence_regression_cases": sum(item["pair_presence_changed"] for item in result),
            "promotion_delta_cases": sum(item["arms"]["0"]["promotion"]["accepted_cycles"] != item["arms"]["1200"]["promotion"]["accepted_cycles"] for item in result),
            "chemical_change_cases": sum(bool(item["chemical_changes"]) for item in result),
            "scene_edge_change_cases": sum(bool(item["scene_edge_changes"]) for item in result),
        },
        "interpretation": {
            "pair_presence": "not implicated if presence_signature remains equal",
            "aromatic_promotion": "implicated when accepted cycle count or final aromatic edges changes",
            "identity": "not repaired by compiler-side demotion; see local-repair replay",
            "training_authorized": False,
            "deployment_authorized": False,
            "sealed_test_read": False,
        },
        "source_sha256": {"initial.pt": file_digest(INITIAL), "manifest.json": file_digest(MANIFEST), "safety_report.json": file_digest(SAFETY)},
        "results": result,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    value = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "aggregate": value["aggregate"]}, ensure_ascii=False))
