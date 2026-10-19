"""Issue one hash-bound authorization for a bounded visual existence shadow."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-release-audit-20261006-r1/report.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-run-authorization-20261006-r1/receipt.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    admission = json.loads((RELEASE / "admission.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    if not audit["gate"].get("readiness") or audit["gate"].get("training_run_authorized") is not False:
        raise ValueError("Release readiness gate is not satisfied")
    if admission.get("training_authorized") is not True or admission.get("training_run_authorized") is not False:
        raise ValueError("Data admission flags are unsafe")
    if admission.get("strict_positive_train", 0) < 300 or admission.get("strict_positive_holdout", 0) < 250:
        raise ValueError("Strict-positive denominator changed")
    receipt = {
        "schema_version": "chemvector-targeted-selector-visual-pool-run-authorization-0.1",
        "status": "authorized_one_bounded_shadow",
        "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "release_admission_sha256": digest(RELEASE / "admission.json"),
        "release_audit_sha256": digest(AUDIT),
        "initial_sha256": INITIAL_SHA,
        "parameter_scope": "new_visual_group_existence_selector_only",
        "frozen_components": ["vision", "atom_decoder", "bond_decoder", "set_head", "ocr", "old_selected_prefix"],
        "train_cache": "train-group-cache.json",
        "holdout_cache": "holdout-group-cache.json",
        "holdout_never_fitted": True,
        "max_steps": 100,
        "wall_seconds": 1800,
        "max_gpus": 1,
        "gpu_index": 0,
        "batch_size": 1,
        "automatic_resume": False,
        "automatic_retry": False,
        "thresholds": {"existence_logit": 0.0, "winner_logit": 0.0, "margin_logit": 0.0},
        "threshold_selection": "fixed_predeclared_no_sweep",
        "seed": 20261006,
        "production_checkpoint_write": False,
        "deployment_authorized": False,
        "sealed_test_read": False,
        "optimizer_steps": 0,
        "run_authorized": True,
        "limitations": [
            "One in-memory visual group-existence/abstention shadow only; no checkpoint promotion.",
            "A failed gate is terminal; no retry, threshold change, or extra steps.",
            "Holdout labels are evaluation-only and never enter gradients.",
            "Protected baseline prefix is hard locked; unknown members are not negatives.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_authorized": True, "max_steps": 100, "wall_seconds": 1800}, indent=2))
    return receipt


if __name__ == "__main__":
    run()
