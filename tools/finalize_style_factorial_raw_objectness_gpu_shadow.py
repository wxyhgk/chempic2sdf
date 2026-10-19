"""Finalize the completed raw-objectness GPU shadow without rerunning it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261010-r1/report.json"
AUTH = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-shadow-authorization-20261010-r1/receipt.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261010-r1/final-audit.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if report.get("sealed_test_read") is not False or report.get("production_checkpoint_write") is not False:
        raise ValueError("shadow report is not sealed-safe")
    if report.get("authorization_sha256") != digest(AUTH) or auth.get("run_authorized") is not True:
        raise ValueError("authorization binding mismatch")
    if report.get("initial_sha256") != INITIAL_SHA or report.get("initial_sha256_after") != INITIAL_SHA or digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint changed")
    core = report["metrics"]["dev_core"]
    checks = {
        "coordgen_precision_ge_085": core["coordgen"]["conservative_precision"] >= .85,
        "original_precision_ge_085": core["original"]["conservative_precision"] >= .85,
        "rdkit_precision_ge_085": core["rdkit"]["conservative_precision"] >= .85,
        "coordgen_recall_ge_050": core["coordgen"]["raw_recall"] >= .50,
        "original_recall_ge_050": core["original"]["raw_recall"] >= .50,
        "rdkit_recall_ge_050": core["rdkit"]["raw_recall"] >= .50,
        "stress_separate": report["gate"].get("stress_separate") is True,
        "initial_unchanged": report["gate"].get("initial_unchanged") is True,
    }
    result = {"schema_version": "chemvector-style-factorial-raw-objectness-gpu-shadow-final-audit-0.1",
              "complete": True, "read_only": True, "training_rerun": False,
              "source_report_sha256": digest(REPORT), "authorization_sha256": digest(AUTH),
              "checks": checks, "pass": all(checks.values()), "metrics": report["metrics"],
              "production_checkpoint_write": False, "deployment_authorized": False,
              "sealed_test_read": False, "initial_sha256": INITIAL_SHA}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "checks": checks, "pass": result["pass"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
