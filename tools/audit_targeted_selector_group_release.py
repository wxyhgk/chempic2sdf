"""Fail-closed exact-byte audit for the targeted group-selector admission."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-group-release-20261006-r2"
PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r5/report.json"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-r3/report.json"
CACHE_AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-group-cache-audit-20261006-r3/report.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-group-release-audit-20261006-r1/report.json"
SUFFIXES = (".meta.json", ".graph.json", ".layers.json", ".text-records.json", ".png", ".svg", ".sdf",
            ".text.png", ".bond.png", ".semantic.png", ".auxiliary.png")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    admission = json.loads((RELEASE / "admission.json").read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cache_audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    if admission.get("training_authorized") is not True or admission.get("training_run_authorized") is not False:
        raise ValueError("Admission lifecycle flags are unsafe")
    if admission.get("coverage_gaps") != [] or admission.get("external_overlap_status") != "checked":
        raise ValueError("Admission has unresolved identity gaps")
    if admission.get("approved_manifest_sha256") != digest(RELEASE / "manifest.json"):
        raise ValueError("Admission does not bind exact manifest bytes")
    if manifest.get("training_run_authorized") is not False or manifest.get("sealed_test_read") is not False:
        raise ValueError("Manifest lifecycle flags are unsafe")
    rows = manifest.get("samples", [])
    if len(rows) != 130 or sum(row["split"] == "train" for row in rows) != 98 or sum(row["split"] == "holdout" for row in rows) != 32:
        raise ValueError("Fixed 98/32 release denominator changed")
    planned = {row["sample_key"]: row for row in plan["rows"]}
    if set(planned) != {row["sample_key"] for row in rows}:
        raise ValueError("Release rows differ from the fixed plan")
    train_ids = {row["molecule_id"] for row in rows if row["split"] == "train"}
    holdout_ids = {row["molecule_id"] for row in rows if row["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("Train/holdout molecule identity overlap")
    source_manifest = {row["sample_key"]: row for row in json.loads((SOURCE / "samples.json").read_text(encoding="utf-8"))}
    preserved = 0
    for row in rows:
        key = row["sample_key"]
        source_meta_path = SOURCE / "samples" / (key + ".meta.json")
        release_meta_path = RELEASE / "samples" / (key + ".meta.json")
        if digest(release_meta_path) != row["released_meta_sha256"] or row["source_meta_sha256"] != digest(source_meta_path):
            raise ValueError("Metadata binding changed")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        release_meta = json.loads(release_meta_path.read_text(encoding="utf-8"))
        if source_meta.get("split") != "quarantine" or source_meta.get("training_authorized") is not False:
            raise ValueError("Source is not quarantine-only")
        expected_split = "train" if row["split"] == "train" else "validation"
        if release_meta.get("split") != expected_split or release_meta.get("training_authorized") is not (row["split"] == "train"):
            raise ValueError("Released lifecycle flags differ from manifest")
        for suffix in SUFFIXES:
            source = SOURCE / "samples" / (key + suffix)
            released = RELEASE / "samples" / (key + suffix)
            if suffix not in {".meta.json", ".graph.json", ".layers.json"}:
                if source.read_bytes() != released.read_bytes():
                    raise ValueError("Non-lifecycle bytes changed")
                preserved += 1
        if source_manifest[key]["molecule_id"] != row["molecule_id"]:
            raise ValueError("Source identity differs")

    train_cache = json.loads((RELEASE / "train-group-cache.json").read_text(encoding="utf-8"))
    holdout_cache = json.loads((RELEASE / "holdout-group-cache.json").read_text(encoding="utf-8"))
    if {group["sample_key"] for group in train_cache["groups"]} - set(planned) or any(group["split"] != "train" for group in train_cache["groups"]):
        raise ValueError("Train cache split mismatch")
    if {group["sample_key"] for group in holdout_cache["groups"]} - set(planned) or any(group["split"] != "holdout" for group in holdout_cache["groups"]):
        raise ValueError("Holdout cache split mismatch")
    if cache_audit["gate"]["large_group_review_required"] or contract["totals"].get("train_strict_positive") != 51 or contract["totals"].get("holdout_strict_positive") != 16:
        raise ValueError("Cache readiness receipt changed")
    result = {
        "schema_version": "chemvector-targeted-selector-group-release-audit-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "source_files_modified": False,
        "release": str(RELEASE.relative_to(ROOT)), "manifest_sha256": digest(RELEASE / "manifest.json"),
        "admission_sha256": digest(RELEASE / "admission.json"), "copied_images": len(rows),
        "byte_identical_non_lifecycle_artifacts": preserved,
        "strict_positive_train": 51, "strict_positive_holdout": 16,
        "gate": {"exact_manifest_binding": True, "identity_split": True,
                 "train_holdout_cache_separation": True,
                 "training_run_authorized": False,
                 "readiness": True,
                 "next": "issue separate 100-step/1800-second run authorization before shadow training"},
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
