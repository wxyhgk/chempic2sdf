"""Bundle the frozen CPU candidate gate with a train-only ridge refiner."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.audit_targeted_selector_coordinate_assignment import fit, collect

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
GATE = ROOT / "outputs/audits/chemvector-targeted-selector-frozen-coordinate-gate-20261007-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-sidecar-20261007-r2"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    gate_report = json.loads((GATE / "report.json").read_text(encoding="utf-8"))
    if not gate_report["gate"].get("gpu_refiner_ready"):
        raise ValueError("Frozen candidate gate did not pass")
    train = collect("train")
    mean, std, beta = fit(train)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.savez(OUTPUT / "coordinate-refiner.npz", mean=mean, std=std, beta=beta,
             near_threshold_px=np.asarray([10.0]), ridge=np.asarray([10.0]))
    refiner_hash = digest(OUTPUT / "coordinate-refiner.npz")
    sidecar = {
        "schema_version": "chemvector-targeted-selector-coordinate-sidecar-0.1",
        "use": "local_shadow_only",
        "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "candidate_gate_report_sha256": digest(GATE / "report.json"),
        "candidate_gate_weights_sha256": digest(GATE / "candidate-head.pt"),
        "coordinate_refiner_sha256": refiner_hash,
        "candidate_gate": {"fixed_logit_threshold": 0.0, "near_threshold_px": 10.0,
                            "train_precision": gate_report["train"]["precision"],
                            "holdout_precision": gate_report["holdout"]["precision"],
                            "holdout_recall": gate_report["holdout"]["recall"]},
        "coordinate_refiner": {"fit_scope": "train_targets_only", "ridge": 10.0,
                               "input_feature_dim": int(beta.shape[0]), "output_dim": 2},
        "strict_selector_status": "not_solved; strict and near_unknown remain separately reported",
        "production_checkpoint_written": False, "deployment_authorized": False,
        "sealed_test_read": False, "training_run_authorized": False,
    }
    (OUTPUT / "sidecar.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sidecar": sidecar}, indent=2, ensure_ascii=False))
    return sidecar


if __name__ == "__main__":
    run()
