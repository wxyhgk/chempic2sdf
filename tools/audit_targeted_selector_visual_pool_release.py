"""Fail-closed exact-byte audit for the 480-image visual-pool admission."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
PLAN = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-plan-20261006-r1/report.json"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-pool-r1/report.json"
COMBINED = ROOT / "outputs/audits/chemvector-targeted-selector-combined-readiness-20261006-r1/report.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-release-audit-20261006-r1/report.json"
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
    combined = json.loads(COMBINED.read_text(encoding="utf-8"))
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    if admission.get("training_authorized") is not True or admission.get("training_run_authorized") is not False:
        raise ValueError("Admission lifecycle flags are unsafe")
    if admission.get("coverage_gaps") != [] or admission.get("external_overlap_status") != "checked":
        raise ValueError("Admission has unresolved identity gaps")
    if admission.get("approved_manifest_sha256") != digest(RELEASE / "manifest.json"):
        raise ValueError("Admission does not bind exact manifest bytes")
    if manifest.get("training_run_authorized") is not False or manifest.get("sealed_test_read") is not False:
        raise ValueError("Manifest lifecycle flags are unsafe")
    rows = manifest.get("samples", [])
    if len(rows) != 480 or sum(r["split"] == "train" for r in rows) != 320 or sum(r["split"] == "holdout" for r in rows) != 160:
        raise ValueError("Fixed 320/160 release denominator changed")
    planned = {r["sample_key"]: r for r in plan["rows"]}
    if set(planned) != {r["sample_key"] for r in rows}:
        raise ValueError("Release rows differ from fixed plan")
    train_ids = {r["molecule_id"] for r in rows if r["split"] == "train"}
    holdout_ids = {r["molecule_id"] for r in rows if r["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("Train/holdout molecule identity overlap")
    source_manifest = {r["sample_key"]: r for r in json.loads((SOURCE / "samples.json").read_text(encoding="utf-8"))}
    preserved = 0
    for row in rows:
        key = row["sample_key"]
        source_meta = SOURCE / "samples" / (key + ".meta.json")
        release_meta = RELEASE / "samples" / (key + ".meta.json")
        if digest(release_meta) != row["released_meta_sha256"] or digest(source_meta) != row["source_meta_sha256"]:
            raise ValueError("Metadata binding changed")
        source_obj = json.loads(source_meta.read_text(encoding="utf-8"))
        release_obj = json.loads(release_meta.read_text(encoding="utf-8"))
        if source_obj.get("split") != "quarantine" or source_obj.get("training_authorized") is not False:
            raise ValueError("Source is not quarantine-only")
        expected_split = "train" if row["split"] == "train" else "validation"
        if release_obj.get("split") != expected_split or release_obj.get("training_authorized") is not (row["split"] == "train"):
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
    train_cache = json.loads((RELEASE / "train-group-cache.json").read_text(encoding="utf-8"))["groups"]
    holdout_cache = json.loads((RELEASE / "holdout-group-cache.json").read_text(encoding="utf-8"))["groups"]
    if any(g["split"] != "train" for g in train_cache) or any(g["split"] != "holdout" for g in holdout_cache):
        raise ValueError("Cache split mismatch")
    totals = contract["totals"]
    if totals.get("train_strict_positive", 0) < 300 or totals.get("holdout_strict_positive", 0) < 250:
        raise ValueError("Strict-positive support changed")
    if combined.get("sealed_test_read") is not False or not combined["gate"].get("holdout_combined_auc_minimum_080"):
        raise ValueError("Combined readiness binding changed")
    result = {
        "schema_version": "chemvector-targeted-selector-visual-pool-release-audit-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "source_files_modified": False,
        "release": str(RELEASE.relative_to(ROOT)), "manifest_sha256": digest(RELEASE / "manifest.json"),
        "admission_sha256": digest(RELEASE / "admission.json"),
        "copied_images": len(rows), "byte_identical_non_lifecycle_artifacts": preserved,
        "strict_positive_train": totals["train_strict_positive"],
        "strict_positive_holdout": totals["holdout_strict_positive"],
        "gate": {"exact_manifest_binding": True, "identity_split": True,
                 "train_holdout_cache_separation": True, "combined_readiness": True,
                 "training_run_authorized": False, "readiness": True,
                 "next": "issue one bounded visual group-existence run authorization"},
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
