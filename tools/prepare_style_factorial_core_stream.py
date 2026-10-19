"""Derive a non-stress core stream and stress-only evaluation manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "outputs/audits/chemvector-style-factorial-target-pool-20261007-r1"
REPLAY = ROOT / "outputs/audits/chemvector-style-factorial-endpoint-reachability-20261007-r1/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
VARIANTS = {"original", "rdkit", "coordgen"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train = json.loads((POOL / "train-manifest.json").read_text(encoding="utf-8"))
    dev = json.loads((POOL / "dev-manifest.json").read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    train_core = [r for r in train["rows"] if r["variant"] in VARIANTS]
    dev_core = [r for r in dev["rows"] if r["variant"] in VARIANTS]
    stress = [r for r in dev["rows"] if r["variant"] == "stress"]
    if len(train_core) != 4800 or len(dev_core) != 1200 or len(stress) != 400:
        raise ValueError("unexpected four-variant pool shape")
    report = {
        "schema_version": "chemvector-style-factorial-core-stream-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "training_authorized": False, "gpu_used": False, "sealed_test_read": False,
        "source_pool_sha256": digest(POOL / "train-manifest.json"),
        "source_dev_sha256": digest(POOL / "dev-manifest.json"),
        "replay_sha256": digest(REPLAY),
        "core_variants": sorted(VARIANTS),
        "train_core_samples": len(train_core), "dev_core_samples": len(dev_core),
        "stress_dev_samples": len(stress),
        "stress_policy": "held_out_until_prefix_retention_gate_is demonstrated",
        "gate": {"identity_disjoint": len({r["molecule_id"] for r in train_core} & {r["molecule_id"] for r in dev_core}) == 0,
                 "core_stream_replay_required": True, "training_admissible": False,
                 "next": "train-only strict crop/endpoint preflight on core stream; evaluate stress separately"},
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "train-core-manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "split": "train", "rows": train_core}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "dev-core-manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "split": "dev", "rows": dev_core}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "dev-stress-manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "split": "dev_stress", "rows": stress}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "report": report}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
