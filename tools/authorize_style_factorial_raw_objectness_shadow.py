"""Issue one bounded in-memory GPU shadow receipt for raw atom objectness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-cpu-preflight-20261010-r1/report.json"
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-shadow-authorization-20261010-r1/receipt.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("sealed_test_read") is not False or not preflight.get("complete"):
        raise ValueError("raw objectness preflight is not complete and sealed-safe")
    gate = preflight.get("gate", {})
    if not gate.get("per_core_conservative_precision_ge_085") or not gate.get("per_core_raw_recall_ge_050"):
        raise ValueError("CPU raw objectness gate did not pass")
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("immutable initial checkpoint changed")
    receipt = {
        "schema_version": "chemvector-style-factorial-raw-objectness-shadow-authorization-0.1",
        "status": "authorized_one_bounded_in_memory_raw_objectness_shadow",
        "preflight_sha256": digest(PREFLIGHT), "candidate_cache_sha256": digest(CACHE),
        "initial_sha256": INITIAL_SHA,
        "parameter_scope": "new_raw_atom_objectness_sidecar_only",
        "frozen_components": ["initial_checkpoint", "vision", "atom_decoder", "bond_decoder", "set_head", "ocr", "production_selector"],
        "positive_labels": ["protected", "strict_positive"],
        "negative_labels": ["hard_negative"],
        "masked_labels": ["near_unknown", "selected_unmatched"],
        "feature_names": ["raw_score", "presence_logit", "x_norm", "y_norm"],
        "max_steps": 100, "wall_seconds": 900, "max_gpus": 1, "gpu_index": 0,
        "batch_size": 2048, "seed": 20261015, "fixed_logit_gate": 0.0,
        "automatic_resume": False, "automatic_retry": False,
        "production_checkpoint_write": False, "deployment_authorized": False,
        "sealed_test_read": False, "optimizer_steps": 0, "run_authorized": True,
        "limitations": [
            "In-memory sidecar shadow only; it cannot replace or promote production.",
            "A failed run is terminal; no threshold sweep, retry, or extra steps.",
            "Stress remains evaluation-only and is not a training negative source.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "run_authorized": True, "max_steps": 100}, indent=2))
    return receipt


if __name__ == "__main__":
    run()
