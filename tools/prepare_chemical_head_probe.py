"""Prepare a deterministic train-only manifest; never authorizes training."""

import json
from pathlib import Path

from chemvector.training.curriculum import pair_scale_training as source
from chemvector.training.curriculum.chemical_head_plan import draft_protocol, make_sample_plan
from chemvector.training.curriculum.checkpoints import file_digest


OUTPUT = source.ROOT / "outputs/audits/chemvector-chemical-head-probe-preparation-20260925"


def write_new_json(path: Path, value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return file_digest(path)


def prepare() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("Preparation output is immutable and already exists")
    checked = source.source_checks()
    train_rows = checked["rows"]
    plan = make_sample_plan(train_rows)
    if plan["summary"]["images"] != 1200:
        raise ValueError("Fixed 1,200-image plan was not produced")
    protocol = draft_protocol(
        sample_sha256=plan["samples_sha256"],
        preparation_sha256=checked["preparation_report_sha256"],
        initial_sha256=source.prep.INITIAL_SHA,
        manifest_sha256=source.prep.MANIFEST_SHA,
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sample_sha = write_new_json(OUTPUT / "training-samples.json", plan)
    protocol_sha = write_new_json(OUTPUT / "protocol.json", protocol)
    report = {
        "schema_version": "chemical-head-probe-preparation-report-0.1",
        "complete": True,
        "training_authorized": False,
        "training_performed": False,
        "deployment_authorized": False,
        "model_loaded": False,
        "gpu_used": False,
        "development_media_read": False,
        "sealed_test_read": False,
        "new_identity_queries": 0,
        "source_preparation_sha256": checked["preparation_report_sha256"],
        "release_manifest_sha256": source.prep.MANIFEST_SHA,
        "initial_checkpoint_sha256": source.prep.INITIAL_SHA,
        "sample_sha256": plan["samples_sha256"],
        "protocol_sha256": file_digest(OUTPUT / "protocol.json"),
        "artifact_sha256": {
            "training-samples.json": sample_sha,
            "protocol.json": protocol_sha,
        },
        "summary": plan["summary"],
        "pending": [
            "Independent read-only runtime estimate bound to the fixed sample list",
            "CPU readiness tests bound to this protocol and implementation",
            "A separate exact training authorization before any model/data training run",
        ],
    }
    write_new_json(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True, ensure_ascii=False, indent=2))
