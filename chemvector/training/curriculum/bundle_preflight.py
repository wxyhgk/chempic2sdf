"""CPU-only actual-weight bundle roundtrip, including the frozen OCR predictor."""

import argparse
import json
from pathlib import Path
import sys

from .runtime import ResourceBudget, run_budgeted


def audit(root, key, output):
    import torch
    from .bundle import save_bundle, load_bundle
    from .checkpoints import load_current_components, file_digest, verify_sources
    from .data import load_sample, CurriculumCollator
    from .evaluation import FrozenOCRPredictor, canonical_molblock
    from .gradient_preflight import state_digest

    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    model, config, provenance = load_current_components()
    model.eval()
    sample = load_sample(root, key, purpose="diagnostic")
    pixels = CurriculumCollator(purpose="diagnostic")([sample])["pixel_values"]
    predictor = FrozenOCRPredictor(provenance, torch.device("cpu"))
    before = state_digest(model)
    ocr_before = state_digest(predictor.ocr)
    initial_prediction = predictor(model, config, pixels, sample["image"])
    record = save_bundle(Path(output) / "untrained-roundtrip.pt", model, config, provenance=provenance,
                         progress={"optimizer_steps": 0, "diagnostic_only": True})
    restored, restored_config, _ = load_bundle(record["path"], expected_sha256=record["sha256"])
    restored_prediction = predictor(restored, restored_config, pixels, sample["image"])
    initial_identity = canonical_molblock(initial_prediction["molblock"])
    restored_identity = canonical_molblock(restored_prediction["molblock"])
    if (state_digest(restored) != before or state_digest(model) != before or state_digest(predictor.ocr) != ocr_before
            or restored_config != config or restored_identity != initial_identity
            or initial_prediction["fallback"] != restored_prediction["fallback"]):
        raise ValueError("Actual checkpoint bundle roundtrip changed weights or prediction")
    verify_sources(provenance["checkpoints"])
    report = {"schema_version": "chemvector-continuation-bundle-preflight-0.1", "device": "cpu",
              "sample_key": key, "optimizer_steps": 0, "training_performed": False, "training_authorized": False,
              "gpu_used": False, "state_roundtrip_equal": True, "frozen_ocr_unchanged": True,
              "prediction_identity_equal": True, "before_smiles": initial_identity, "after_smiles": restored_identity,
              "prediction_error": restored_prediction["error"], "bundle": record,
              "state_sha256": before, "checkpoint_provenance": provenance,
              "sample_meta_sha256": file_digest(Path(root) / "samples" / (key + ".meta.json")),
              "source_sha256": {str(p): file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "notes": ["This is an untrained diagnostic copy of current weights, not a new trained model.",
                        "Single-sample roundtrip equality is not a recognition accuracy or training efficacy result."]}
    with (Path(output) / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        audit(args.data_root, args.sample_key, args.output)
        return 0
    command = [sys.executable, "-m", "chemvector.training.curriculum.bundle_preflight", "--worker",
               "--data-root", str(args.data_root), "--sample-key", args.sample_key, "--output", str(args.output)]
    receipt = run_budgeted(command, output=args.output, budget=ResourceBudget(180))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
