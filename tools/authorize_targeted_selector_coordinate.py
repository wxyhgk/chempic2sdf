"""Issue one bounded authorization for the coordinate-refinement shadow."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261006-r1"
RELEASE_AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-release-audit-20261006-r1/report.json"
PREFLIGHT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-candidate-cpu-preflight-20261006-r1/report.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-run-authorization-20261006-r1/receipt.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    admission = json.loads((RELEASE / "admission.json").read_text(encoding="utf-8"))
    audit = json.loads(RELEASE_AUDIT.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    if not audit["gate"].get("readiness") or audit.get("sealed_test_read") is not False:
        raise ValueError("Coordinate release audit is not ready")
    if not preflight["gate"].get("training_ready") or preflight.get("sealed_test_read") is not False:
        raise ValueError("CPU preflight did not pass")
    if admission.get("training_authorized") is not True or admission.get("training_run_authorized") is not False:
        raise ValueError("Admission lifecycle flags are unsafe")
    receipt = {
        "schema_version": "chemvector-targeted-selector-coordinate-run-authorization-0.1",
        "status": "authorized_one_bounded_coordinate_shadow",
        "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "release_admission_sha256": digest(RELEASE / "admission.json"),
        "release_audit_sha256": digest(RELEASE_AUDIT),
        "preflight_sha256": digest(PREFLIGHT),
        "initial_sha256": INITIAL_SHA,
        "parameter_scope": "new_coordinate_refinement_and_refinable_candidate_heads_only",
        "frozen_components": ["vision", "atom_decoder", "bond_decoder", "set_head", "ocr", "old_selected_prefix", "old_selector"],
        "train_targets": "train-refinement-targets.json plus far-candidate negatives from train cache",
        "holdout_targets": "holdout-refinement-targets.json diagnostic_only",
        "holdout_never_fitted": True,
        "near_threshold_px": 10.0,
        "max_steps": 100, "wall_seconds": 1800, "max_gpus": 1, "gpu_index": 0,
        "batch_size": 512, "automatic_resume": False, "automatic_retry": False,
        "refinable_logit_threshold": 0.0,
        "threshold_selection": "fixed_predeclared_no_sweep",
        "seed": 20261007,
        "production_checkpoint_write": False, "deployment_authorized": False,
        "sealed_test_read": False, "optimizer_steps": 0, "run_authorized": True,
        "limitations": [
            "One in-memory coordinate/refinable-candidate shadow only; no production integration.",
            "A failed gate is terminal; no retry, threshold change, or extra steps.",
            "Holdout targets are evaluation-only and never enter gradients.",
            "Near-target candidates are refinement positives/unknown, never selector negatives.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "run_authorized": True, "max_steps": 100, "wall_seconds": 1800}, indent=2))
    return receipt


if __name__ == "__main__":
    run()
