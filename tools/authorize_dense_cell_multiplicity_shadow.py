"""Issue one bounded receipt for the isolated dense-cell multiplicity shadow.

The receipt is deliberately separate from the CPU contract audit.  It binds
the exact candidate manifest, contract report, and frozen production checkpoint
and authorizes only a sidecar run; it cannot authorize a production update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/report.json"
MANIFEST = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/manifest.json"
PROTOCOL = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/protocol.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUTPUT = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-shadow-authorization-20261017-r1/receipt.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not contract.get("complete") or not contract["gate"].get("cpu_contract_pass"):
        raise ValueError("CPU multiplicity contract did not pass")
    if contract["gate"].get("gpu_training_authorized") is not False:
        raise ValueError("CPU report must not self-authorize GPU")
    if manifest.get("training_performed") is not False or manifest.get("sealed_test_read") is not False:
        raise ValueError("candidate manifest is not read-only")
    if manifest.get("initial_sha256") != INITIAL_SHA or digest(INITIAL) != INITIAL_SHA:
        raise ValueError("production checkpoint binding failed")
    if contract.get("candidate_manifest_sha256") != digest(MANIFEST):
        raise ValueError("contract is not bound to the candidate manifest")
    if protocol.get("future_gpu_shadow", {}).get("max_steps") != 100:
        raise ValueError("unexpected shadow step budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    receipt = {
        "schema_version": "chemvector-dense-cell-multiplicity-shadow-receipt-0.1",
        "status": "authorized_one_bounded_multiplicity_shadow",
        "contract_sha256": digest(CONTRACT),
        "candidate_manifest_sha256": digest(MANIFEST),
        "protocol_sha256": digest(PROTOCOL),
        "production_checkpoint_sha256": INITIAL_SHA,
        "gpu": {"index": 0, "name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()},
        "model_scope": "new 1x1 extra-heatmap head only",
        "frozen_components": ["initial vision", "base atom heatmap", "base atom offsets", "selector", "OCR", "bond heads"],
        "architecture": {"input": "frozen stride-4 vision features", "output": "one extra heatmap", "capacity": 16},
        "max_steps": 100,
        "max_wall_seconds": 300,
        "seed": 20261017,
        "train_split": "manifest split=train only",
        "holdout_split": "manifest split=holdout only",
        "baseline_prefix_immutable": True,
        "production_checkpoint_write": False,
        "deployment_authorized": False,
        "sealed_test_read": False,
        "automatic_retry": False,
        "threshold_sweep": False,
        "run_authorized": True,
        "limitations": [
            "Sidecar artifact only; no selector, OCR, graph, or production integration.",
            "A failed gate is terminal: no retry, threshold sweep, or extra steps.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "run_authorized": True, "max_steps": 100, "max_wall_seconds": 300}, indent=2))
    return receipt


if __name__ == "__main__":
    run()
