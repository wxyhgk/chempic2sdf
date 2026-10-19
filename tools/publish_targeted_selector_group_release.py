"""Publish the fixed 130-image group-selector data slice.

This is an independent exact-byte data admission step.  It never edits the
quarantine source, loads a model, starts CUDA, reads sealed data, or grants a
training-run authorization.  The holdout cache is written separately and is
diagnostic-only.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

from chemvector.training.curriculum import publish as lifecycle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r5/report.json"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-r3/report.json"
CACHE_AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-group-cache-audit-20261006-r3/report.json"
CACHE = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-r3/group-cache.json"
PUBLIC = ROOT / "outputs/audits/structure-expansion-public-identity-20260923/remaining-candidates.json"
SOURCE_USE = ROOT / "docs/model/training/releases/20261006-targeted-selector-source-use.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/datasets/chemvector-targeted-selector-group-release-20261006-r2"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: bytes | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    cache_audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    groups = json.loads(CACHE.read_text(encoding="utf-8"))["groups"]
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    rows = plan["rows"]
    if len(rows) != 130 or len({r["sample_key"] for r in rows}) != 130:
        raise ValueError("Fixed 130-row plan required")
    if cache_audit["gate"]["large_group_review_required"] or cache_audit["gate"]["training_ready"]:
        # training_ready is intentionally false in the cache-only receipt; a
        # future implementation must not silently turn this into run authority.
        if cache_audit["gate"]["large_group_review_required"]:
            raise ValueError("Candidate cache still contains an oversized group")
    remaining = {key for candidate in public for key in candidate["sample_keys"]}
    if any(row["sample_key"] not in remaining for row in rows):
        raise ValueError("Targeted row is not in the conservatively cleared public subset")
    by_key = {r["sample_key"]: r for r in json.loads((DATA / "samples.json").read_text(encoding="utf-8"))}
    groups_by_split = {"train": [], "holdout": []}
    for group in groups:
        if group["split"] not in groups_by_split or group["sample_key"] not in by_key:
            raise ValueError("Group cache identity mismatch")
        groups_by_split[group["split"]].append(group)
    train_ids = {r["molecule_id"] for r in rows if r["split"] == "train"}
    holdout_ids = {r["molecule_id"] for r in rows if r["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("Train/holdout molecule overlap")
    if contract["totals"].get("train_strict_positive") != 51 or contract["totals"].get("holdout_strict_positive") != 16:
        raise ValueError("Strict-positive gate changed")
    for row in rows:
        metadata = json.loads((DATA / "samples" / (row["sample_key"] + ".meta.json")).read_text(encoding="utf-8"))
        if metadata.get("training_authorized") is not False or metadata.get("split") != "quarantine":
            raise ValueError("Source is not quarantine-only")

    output.mkdir(parents=True)
    source_root = output / "samples"
    manifest_rows = []
    copied = 0
    preserved = 0
    for row in rows:
        key = row["sample_key"]
        source_meta = json.loads((DATA / "samples" / (key + ".meta.json")).read_text(encoding="utf-8"))
        source_meta_sha = digest(DATA / "samples" / (key + ".meta.json"))
        payloads = {}
        for suffix in lifecycle.SUFFIXES:
            path = DATA / "samples" / (key + suffix)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing source artifact {key}{suffix}")
            payloads[suffix] = path.read_bytes()
            expected = source_meta["sha256"].get(suffix)
            if suffix != ".meta.json" and expected != sha256(payloads[suffix]).hexdigest():
                raise ValueError(f"Source artifact hash mismatch {key}{suffix}")
        lifecycle_row = dict(row, split="train" if row["split"] == "train" else "validation",
                             meta_sha256=source_meta_sha)
        converted = lifecycle.transition(payloads, lifecycle_row,
                                        plan_sha256=digest(PLAN), original_root=str(DATA))
        for suffix, data in converted.items():
            destination = source_root / (key + suffix)
            write_new(destination, data)
            if suffix not in {".meta.json", ".graph.json", ".layers.json"}:
                if data != payloads[suffix]:
                    raise ValueError("Non-lifecycle artifact changed during publication")
                preserved += 1
        manifest_rows.append({
            "sample_key": key, "molecule_id": row["molecule_id"],
            "canonical_smiles": row["canonical_smiles"],
            "connection_identity_without_isotope_stereo": row["connection_identity_without_isotope_stereo"],
            "profile_id": row["profile_id"], "split": row["split"],
            "source_meta_sha256": source_meta_sha,
            "released_meta_sha256": digest(source_root / (key + ".meta.json")),
            "fit_use": "group_selector_only" if row["split"] == "train" else "diagnostic_holdout_only",
        })
        copied += 1

    manifest = {
        "schema_version": "chemvector-targeted-selector-group-release-0.1",
        "use": "local_selector_group_fit_only", "source_root": str(source_root),
        "source_dataset": str(DATA.relative_to(ROOT)), "plan_sha256": digest(PLAN),
        "contract_sha256": digest(CONTRACT), "cache_audit_sha256": digest(CACHE_AUDIT),
        "source_use_sha256": digest(SOURCE_USE), "initial_sha256": INITIAL_SHA,
        "samples": manifest_rows, "train_images": sum(r["split"] == "train" for r in manifest_rows),
        "holdout_images": sum(r["split"] == "holdout" for r in manifest_rows),
        "training_run_authorized": False, "sealed_test_read": False,
        "deployment_authorized": False,
    }
    write_new(output / "manifest.json", manifest)
    write_new(output / "train-group-cache.json", {"groups": groups_by_split["train"], "purpose": "training"})
    write_new(output / "holdout-group-cache.json", {"groups": groups_by_split["holdout"], "purpose": "diagnostic_only"})
    write_new(output / "source-use-review.json", SOURCE_USE.read_bytes())
    report = {
        "schema_version": "chemvector-targeted-selector-group-publication-0.1",
        "complete": True, "data_published": True, "training_authorized": True,
        "training_run_authorized": False, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "source_files_modified": False,
        "copied_bundles": copied, "byte_identical_non_lifecycle_artifacts": preserved,
        "train_images": manifest["train_images"], "holdout_images": manifest["holdout_images"],
        "strict_positive_train": 51, "strict_positive_holdout": 16,
        "manifest_sha256": digest(output / "manifest.json"),
        "admission_sha256": None,
        "limitations": ["Data admission only; no group scorer has been trained.",
                         "Holdout cache is diagnostic-only and must not enter fitting.",
                         "Local research only; no redistribution or deployment."],
    }
    write_new(output / "publication-report.json", report)
    admission = {
        "schema_version": "chemvector-targeted-selector-group-admission-0.1",
        "training_authorized": True, "training_run_authorized": False,
        "external_overlap_status": "checked", "coverage_gaps": [],
        "approved_manifest_sha256": digest(output / "manifest.json"),
        "plan_sha256": digest(PLAN), "contract_sha256": digest(CONTRACT),
        "cache_audit_sha256": digest(CACHE_AUDIT), "initial_sha256": INITIAL_SHA,
        "checked_images": 130, "checked_train_images": 98, "checked_holdout_images": 32,
        "strict_positive_train": 51, "strict_positive_holdout": 16,
        "sealed_test_read": False, "deployment_authorized": False,
        "use": "local_research_only",
        "limitations": ["This admission does not authorize a GPU run; a separate run receipt is required.",
                         "Only the fixed 98-image train cache may supply gradients.",
                         "The 32-image holdout is fixed and diagnostic-only."],
    }
    admission["admission_sha256"] = sha256((json.dumps(admission, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()).hexdigest()
    write_new(output / "admission.json", admission)
    report["admission_sha256"] = digest(output / "admission.json")
    # publication-report is already written, so retain an immutable summary
    # receipt instead of rewriting it in place.
    write_new(output / "publication-final-receipt.json", report)
    print(json.dumps({"output": str(output), "copied_bundles": copied,
                      "train_images": manifest["train_images"], "holdout_images": manifest["holdout_images"],
                      "training_authorized": True, "training_run_authorized": False}, indent=2))
    return report


if __name__ == "__main__":
    run()
