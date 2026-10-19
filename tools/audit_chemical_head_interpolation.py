"""Read-only trust-region replay between initial and r3 chemical heads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from chemvector.training.curriculum.bundle import load_bundle
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.validation_truth import canonical_molblock, expected_identity
from tools.audit_chemical_head_aromatic_safety import INITIAL_SHA, MANIFEST_SHA, _fixed_cases

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/experiments/chemvector-chemical-head-probe-20261001-r3"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
MANIFEST = ROOT / "outputs/datasets/chemvector-expanded-release-20260915/manifest.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_INTERPOLATION_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-chemical-head-interpolation-20261001"),
))
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run(output: Path = OUTPUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite interpolation output: {output}")
    if file_digest(INITIAL) != INITIAL_SHA or file_digest(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Initial checkpoint or manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    baseline, changed = _fixed_cases()
    if os.environ.get("CHEMVECTOR_INTERPOLATION_SCOPE") == "full_dev":
        keys = sorted(
            row["source"] + ":" + row["sample_key"]
            for row in manifest["samples"]
            if row["split"] == "validation"
        )
        if len(keys) != 944:
            raise ValueError("Fixed full development cohort changed")
    else:
        keys = sorted(changed)
    loader = RecordLoader(manifest)
    rows = {row["source"] + ":" + row["sample_key"]: row for row in manifest["samples"]}
    model, config, payload = load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    model.to(device).eval().requires_grad_(False)
    predictor = FrozenOCRPredictor(payload["provenance"], device)
    initial = {name: value.detach().cpu().clone() for name, value in model.pair.chemical.state_dict().items()}
    final_payload = torch.load(RUN / "chemical-001200.pt", map_location="cpu", weights_only=True)
    if final_payload.get("initial_sha256") != INITIAL_SHA or final_payload.get("deployment_authorized") is not False:
        raise ValueError("Invalid r3 checkpoint binding")
    final = {name: value.detach().cpu().clone() for name, value in final_payload["chemical_state"].items()}
    results = []
    with torch.no_grad():
        for alpha in ALPHAS:
            state = {name: initial[name] + float(alpha) * (final[name] - initial[name]) for name in initial}
            model.pair.chemical.load_state_dict(state, strict=True)
            records = []
            for key in keys:
                sample = loader(manifest, rows[key])
                pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
                structure = predictor.predict_structure(model, config, pixels)
                prediction = predictor.from_structure(structure, model.role, model.bbox, pixels, sample["image"])
                canonical = canonical_molblock(prediction["molblock"])
                expected = expected_identity(sample)
                records.append({
                    "key": key,
                    "domain": baseline[key]["domain"],
                    "valid": canonical is not None,
                    "exact": canonical is not None and canonical == expected,
                    "fallback": bool(prediction["fallback"]),
                    "predicted_smiles": canonical,
                })
                del pixels, structure, prediction
            results.append({
                "alpha": alpha,
                "summary": {
                    "images": len(records),
                    "exact": sum(row["exact"] for row in records),
                    "valid": sum(row["valid"] for row in records),
                    "fallback": sum(row["fallback"] for row in records),
                "new_fallback_vs_initial": sum(
                    not baseline[row["key"]]["fallback"] and row["fallback"]
                    for row in records if row["key"] in baseline
                ),
                },
                "records": records,
            })
    report = {
        "schema_version": "chemvector-chemical-head-interpolation-0.1",
        "complete": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_modified": False,
        "scope": {"cohort": "full_dev" if os.environ.get("CHEMVECTOR_INTERPOLATION_SCOPE") == "full_dev" else "changed_witnesses", "images": len(keys), "alphas": list(ALPHAS), "model_forwards": len(keys) * len(ALPHAS)},
        "source_sha256": {"initial.pt": file_digest(INITIAL), "manifest.json": file_digest(MANIFEST), "r3_checkpoint.pt": file_digest(RUN / "chemical-001200.pt")},
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    value = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "summaries": [row["summary"] for row in value["results"]]}, ensure_ascii=False))
