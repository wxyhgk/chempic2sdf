"""Write the bounded render receipt after the renderer has completed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    plan = json.loads((DATA / "plan.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rejected = [json.loads(line) for line in (DATA / "rejected.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) + len(rejected) != plan["selection"]["planned_images"]:
        raise ValueError("render accounting incomplete")
    report = {
        "schema_version": "chemvector-identity-aligned-visible-text-render-report-0.1",
        "complete": True, "status": summary["status"], "training_authorized": False,
        "gpu_used": False, "model_forwards": 0, "optimizer_steps": 0,
        "sealed_test_read": False, "plan_sha256": sha(DATA / "plan.json"),
        "summary_sha256": sha(DATA / "summary.json"),
        "samples_sha256": sha(DATA / "samples.jsonl"),
        "rejected_sha256": sha(DATA / "rejected.jsonl"),
        "sample_hash_manifest_sha256": sha(DATA / "sample-sha256.json"),
        "planned_images": len(plan["rows"]), "accepted_images": len(rows),
        "rejected_images": len(rejected), "accepted_by_split": summary["accepted_by_split"],
        "accepted_by_profile": summary["accepted_by_profile"],
        "rejection_reasons": summary["rejection_reasons"],
        "molecule_overlap": summary["molecule_overlap"],
        "next": "CPU identity/geometry audit; no training admission",
    }
    output = DATA / "render-report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
