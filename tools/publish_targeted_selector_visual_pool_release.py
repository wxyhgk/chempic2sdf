"""Publish the fixed 480-image targeted selector visual-pool release.

This is an independent data-admission operation.  It copies quarantine
bundles, changes lifecycle metadata only, and never loads a model, trains,
touches sealed data, or authorizes a GPU run.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from chemvector.training.curriculum import publish as lifecycle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PLAN = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-plan-20261006-r1/report.json"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-pool-r1/report.json"
CONTRACT_DIR = CONTRACT.parent
COMBINED = ROOT / "outputs/audits/chemvector-targeted-selector-combined-readiness-20261006-r1/report.json"
PUBLIC = ROOT / "outputs/audits/structure-expansion-public-identity-20260923/remaining-candidates.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: bytes | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_bytes(value if isinstance(value, bytes) else
                     (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    combined = json.loads(COMBINED.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    rows = plan.get("rows", [])
    if len(rows) != 480 or len({r["sample_key"] for r in rows}) != 480:
        raise ValueError("Fixed 480-row plan required")
    if combined.get("sealed_test_read") is not False or not combined["gate"].get("holdout_combined_auc_minimum_080"):
        raise ValueError("Combined readiness gate is not satisfied")
    totals = contract.get("totals", {})
    if totals.get("train_strict_positive", 0) < 300 or totals.get("holdout_strict_positive", 0) < 250:
        raise ValueError("Strict-positive support is below the fixed pool minimum")
    remaining = {key for candidate in public for key in candidate.get("sample_keys", [])}
    if any(row["sample_key"] not in remaining for row in rows):
        raise ValueError("Targeted row is not in the conservatively cleared public subset")
    source_rows = {r["sample_key"]: r for r in json.loads((SOURCE / "samples.json").read_text(encoding="utf-8"))}
    train_ids = {r["molecule_id"] for r in rows if r["split"] == "train"}
    holdout_ids = {r["molecule_id"] for r in rows if r["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("Train/holdout molecule identity overlap")
    output.mkdir(parents=True)
    destination = output / "samples"
    manifest_rows = []
    preserved = 0
    for row in rows:
        key = row["sample_key"]
        source_meta_path = SOURCE / "samples" / (key + ".meta.json")
        if key not in source_rows or not source_meta_path.is_file():
            raise ValueError(f"Missing source row {key}")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        if source_meta.get("split") != "quarantine" or source_meta.get("training_authorized") is not False:
            raise ValueError("Only quarantine source bundles may be admitted")
        source_meta_sha = digest(source_meta_path)
        payloads = {}
        for suffix in lifecycle.SUFFIXES:
            path = SOURCE / "samples" / (key + suffix)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing source artifact {key}{suffix}")
            data = path.read_bytes()
            if suffix != ".meta.json" and source_meta["sha256"].get(suffix) != sha256(data).hexdigest():
                raise ValueError(f"Source artifact hash mismatch {key}{suffix}")
            payloads[suffix] = data
        lifecycle_row = dict(row, split="train" if row["split"] == "train" else "validation",
                             meta_sha256=source_meta_sha)
        converted = lifecycle.transition(payloads, lifecycle_row,
                                        plan_sha256=digest(PLAN), original_root=str(SOURCE))
        for suffix, data in converted.items():
            write_new(destination / (key + suffix), data)
            if suffix not in {".meta.json", ".graph.json", ".layers.json"}:
                if data != payloads[suffix]:
                    raise ValueError("Non-lifecycle artifact changed during publication")
                preserved += 1
        manifest_rows.append({
            "sample_key": key,
            "molecule_id": row["molecule_id"],
            "canonical_smiles": row["canonical_smiles"],
            "connection_identity_without_isotope_stereo": row["connection_identity_without_isotope_stereo"],
            "profile_id": row["profile_id"],
            "split": row["split"],
            "source_meta_sha256": source_meta_sha,
            "released_meta_sha256": digest(destination / (key + ".meta.json")),
            "fit_use": "group_existence_selector_only" if row["split"] == "train" else "diagnostic_holdout_only",
        })
    manifest = {
        "schema_version": "chemvector-targeted-selector-visual-pool-release-0.1",
        "use": "local_selector_group_existence_fit_only",
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "plan_sha256": digest(PLAN), "contract_sha256": digest(CONTRACT),
        "combined_readiness_sha256": digest(COMBINED), "initial_sha256": INITIAL_SHA,
        "samples": manifest_rows,
        "train_images": sum(r["split"] == "train" for r in manifest_rows),
        "holdout_images": sum(r["split"] == "holdout" for r in manifest_rows),
        "training_run_authorized": False, "sealed_test_read": False,
        "deployment_authorized": False,
    }
    write_new(output / "manifest.json", manifest)
    groups = json.loads((CONTRACT_DIR / "group-cache.json").read_text(encoding="utf-8"))["groups"]
    write_new(output / "train-group-cache.json", {"groups": [g for g in groups if g["split"] == "train"], "purpose": "training"})
    write_new(output / "holdout-group-cache.json", {"groups": [g for g in groups if g["split"] == "holdout"], "purpose": "diagnostic_only"})
    source_use = {
        "schema_version": "chemvector-targeted-selector-source-use-review-0.2",
        "reviewed_on": "2026-10-06", "use": "local_research_only",
        "redistribution_authorized": False,
        "source": {"kind": "deterministic_hand_defined_chemical_templates",
                    "basis": "Fixed 480-image slice copied from the local rendered structure-expansion quarantine batch.",
                    "original_root_is_immutable": True},
        "scope_limits": ["Only this fixed 320/160 train/holdout slice is admitted.",
                         "No sealed identity or private evaluation payload is copied.",
                         "A separate bounded run authorization is required before GPU execution.",
                         "No redistribution or deployment is authorized."],
    }
    write_new(output / "source-use-review.json", source_use)
    admission = {
        "schema_version": "chemvector-targeted-selector-visual-pool-admission-0.1",
        "training_authorized": True, "training_run_authorized": False,
        "external_overlap_status": "checked", "coverage_gaps": [],
        "approved_manifest_sha256": digest(output / "manifest.json"),
        "plan_sha256": digest(PLAN), "contract_sha256": digest(CONTRACT),
        "combined_readiness_sha256": digest(COMBINED), "initial_sha256": INITIAL_SHA,
        "checked_images": 480, "checked_train_images": 320, "checked_holdout_images": 160,
        "strict_positive_train": totals["train_strict_positive"],
        "strict_positive_holdout": totals["holdout_strict_positive"],
        "sealed_test_read": False, "deployment_authorized": False,
        "use": "local_research_only",
        "limitations": ["Admission does not authorize a GPU run; a separate receipt is required.",
                         "Only train-group-cache.json may supply gradients.",
                         "holdout-group-cache.json is diagnostic-only."],
    }
    admission["admission_sha256"] = sha256((json.dumps(admission, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()).hexdigest()
    write_new(output / "admission.json", admission)
    report = {
        "schema_version": "chemvector-targeted-selector-visual-pool-publication-0.1",
        "complete": True, "data_published": True, "training_authorized": True,
        "training_run_authorized": False, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "source_files_modified": False,
        "copied_bundles": len(rows), "byte_identical_non_lifecycle_artifacts": preserved,
        "train_images": manifest["train_images"], "holdout_images": manifest["holdout_images"],
        "manifest_sha256": digest(output / "manifest.json"),
        "admission_sha256": digest(output / "admission.json"),
    }
    write_new(output / "publication-report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
