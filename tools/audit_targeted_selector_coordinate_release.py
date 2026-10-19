"""Fail-closed audit for the coordinate-refinement target release."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-release-audit-20261007-r2/report.json"
SUFFIXES = (".meta.json", ".graph.json", ".layers.json", ".text-records.json", ".png", ".svg", ".sdf",
            ".text.png", ".bond.png", ".semantic.png", ".auxiliary.png")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    admission = json.loads((RELEASE / "admission.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if admission.get("training_authorized") is not True or admission.get("training_run_authorized") is not False:
        raise ValueError("Unsafe admission flags")
    if admission.get("approved_manifest_sha256") != digest(RELEASE / "manifest.json"):
        raise ValueError("Manifest binding changed")
    if manifest.get("sealed_test_read") is not False or manifest.get("training_run_authorized") is not False:
        raise ValueError("Unsafe release flags")
    rows = manifest["samples"]
    source_rows = {r["sample_key"]: r for r in source_manifest["samples"]}
    if len(rows) != 480 or {r["sample_key"] for r in rows} != set(source_rows):
        raise ValueError("Sample identity changed")
    train_ids = {r["molecule_id"] for r in rows if r["split"] == "train"}
    holdout_ids = {r["molecule_id"] for r in rows if r["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("Molecule identity overlap")
    preserved = 0
    for row in rows:
        key = row["sample_key"]
        for suffix in SUFFIXES:
            if (SOURCE / "samples" / (key + suffix)).read_bytes() != (RELEASE / "samples" / (key + suffix)).read_bytes():
                raise ValueError("Source image/render bytes changed")
            if suffix not in {".meta.json", ".graph.json", ".layers.json"}:
                preserved += 1
    target_counts = {}
    for split, name in (("train", "train-refinement-targets.json"), ("holdout", "holdout-refinement-targets.json")):
        payload = json.loads((RELEASE / name).read_text(encoding="utf-8"))
        if payload.get("split") != split or payload.get("purpose") != ("train_only" if split == "train" else "diagnostic_only"):
            raise ValueError("Target split/purpose mismatch")
        targets = payload.get("targets", [])
        if any(t["sample_key"] not in {r["sample_key"] for r in rows if r["split"] == split} for t in targets):
            raise ValueError("Target sample outside split")
        if any(t["status"] not in {"strict_positive", "near_unknown"} or t["distance_px"] > 10.0 for t in targets):
            raise ValueError("Target label policy changed")
        target_counts[split] = {"total": len(targets), "strict_positive": sum(t["status"] == "strict_positive" for t in targets),
                                "near_unknown": sum(t["status"] == "near_unknown" for t in targets)}
    result = {"schema_version": "chemvector-targeted-selector-coordinate-release-audit-0.1",
              "complete": True, "read_only": True, "training_performed": False,
              "gpu_used": False, "optimizer_steps": 0, "sealed_test_read": False,
              "source_files_modified": False, "copied_images": len(rows),
              "byte_identical_non_lifecycle_artifacts": preserved, "target_counts": target_counts,
              "manifest_sha256": digest(RELEASE / "manifest.json"), "admission_sha256": digest(RELEASE / "admission.json"),
              "gate": {"exact_manifest_binding": True, "identity_split": True,
                       "byte_identity": True, "near_unknown_not_negative": True,
                       "training_run_authorized": False,
                       "readiness": True,
                       "next": "run CPU-only refinement preflight before any GPU receipt"}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
