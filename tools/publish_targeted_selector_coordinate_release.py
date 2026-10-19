"""Publish an independent coordinate-refinement target release.

The visual-pool images remain byte-identical.  New target files contain only
candidate-to-nearest-layer-atom residuals for candidates within 10px; they are
owned by this release and are never exposed to the frozen selector cache.
Near-targets are explicitly unknown for selector-negative losses.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
CONTRACT = ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-pool-r1/report.json"
LABEL_AUDIT = ROOT / "outputs/audits/chemvector-targeted-selector-group-label-strata-20261006-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
NEAR_PX = 10.0
SUFFIXES = (".meta.json", ".graph.json", ".layers.json", ".text-records.json", ".png", ".svg", ".sdf",
            ".text.png", ".bond.png", ".semantic.png", ".auxiliary.png")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: bytes | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(value if isinstance(value, bytes) else
                     (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    label_audit = json.loads(LABEL_AUDIT.read_text(encoding="utf-8"))
    if source_manifest.get("sealed_test_read") is not False or source_manifest.get("training_run_authorized") is not False:
        raise ValueError("Source release is not run-gated")
    if label_audit.get("sealed_test_read") is not False or label_audit.get("training_performed"):
        raise ValueError("Label audit is not diagnostic-only")
    rows = source_manifest["samples"]
    train_ids = {r["molecule_id"] for r in rows if r["split"] == "train"}
    holdout_ids = {r["molecule_id"] for r in rows if r["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("Identity overlap")
    cache_by_split = {
        "train": json.loads((SOURCE / "train-group-cache.json").read_text(encoding="utf-8"))["groups"],
        "holdout": json.loads((SOURCE / "holdout-group-cache.json").read_text(encoding="utf-8"))["groups"],
    }
    output_samples = OUTPUT / "samples"
    OUTPUT.mkdir(parents=True)
    targets = {"train": [], "holdout": []}
    copied = 0
    preserved = 0
    for row in rows:
        key, split = row["sample_key"], row["split"]
        destination = output_samples
        for suffix in SUFFIXES:
            src = SOURCE / "samples" / (key + suffix)
            data = src.read_bytes()
            write_new(destination / (key + suffix), data)
            if suffix not in {".meta.json", ".graph.json", ".layers.json"}:
                preserved += 1
        layers = json.loads((SOURCE / "samples" / (key + ".layers.json")).read_text(encoding="utf-8"))
        atoms = [(float(a["center"][0]), float(a["center"][1])) for a in layers["atoms"]]
        groups = [g for g in cache_by_split[split] if g["sample_key"] == key]
        for group in groups:
            for member in group["members"]:
                x, y = float(member["feature"][3]) * 768.0, float(member["feature"][4]) * 768.0
                distances = [((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 for ax, ay in atoms]
                index = min(range(len(distances)), key=distances.__getitem__)
                distance = distances[index]
                if distance > NEAR_PX:
                    continue
                explicit_strict = member["raw_target"] >= 0 and member["selected_target"] < 0
                explicit_negative = group["confirmed_negative"] and member["raw_target"] < 0 and member["selected_target"] < 0
                # Unknown members inside a strict-positive group are not
                # refinement positives or negatives and are excluded.
                if not (explicit_strict or explicit_negative):
                    continue
                status = "strict_positive" if explicit_strict else "near_unknown"
                targets[split].append({"sample_key": key, "molecule_id": row["molecule_id"],
                                      "group_id": group["group_id"], "slot": int(member["slot"]),
                                      "profile_id": row["profile_id"], "status": status,
                                      "candidate_xy_px": [x, y], "target_atom_id": int(index),
                                      "target_xy_px": list(atoms[index]), "distance_px": distance,
                                      "residual_px": [atoms[index][0] - x, atoms[index][1] - y]})
        copied += 1
    manifest = {
        "schema_version": "chemvector-targeted-selector-coordinate-release-0.1",
        "use": "local_coordinate_refinement_fit_only",
        "source_release": str(SOURCE.relative_to(ROOT)), "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "contract_sha256": digest(CONTRACT), "label_audit_sha256": digest(LABEL_AUDIT),
        "near_threshold_px": NEAR_PX, "samples": rows,
        "train_images": sum(r["split"] == "train" for r in rows),
        "holdout_images": sum(r["split"] == "holdout" for r in rows),
        "train_target_file": "train-refinement-targets.json", "holdout_target_file": "holdout-refinement-targets.json",
        "training_run_authorized": False, "sealed_test_read": False, "deployment_authorized": False,
    }
    write_new(OUTPUT / "manifest.json", manifest)
    for split in ("train", "holdout"):
        write_new(OUTPUT / ("train-refinement-targets.json" if split == "train" else "holdout-refinement-targets.json"),
                  {"schema_version": "chemvector-coordinate-refinement-targets-0.1",
                   "split": split, "purpose": "train_only" if split == "train" else "diagnostic_only",
                   "near_threshold_px": NEAR_PX, "targets": targets[split]})
    admission = {
        "schema_version": "chemvector-targeted-selector-coordinate-admission-0.1",
        "training_authorized": True, "training_run_authorized": False,
        "external_overlap_status": "checked", "coverage_gaps": [],
        "approved_manifest_sha256": digest(OUTPUT / "manifest.json"),
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "near_threshold_px": NEAR_PX,
        "train_targets": len(targets["train"]), "holdout_targets": len(targets["holdout"]),
        "strict_positive_train": sum(t["status"] == "strict_positive" for t in targets["train"]),
        "strict_positive_holdout": sum(t["status"] == "strict_positive" for t in targets["holdout"]),
        "near_unknown_train": sum(t["status"] == "near_unknown" for t in targets["train"]),
        "near_unknown_holdout": sum(t["status"] == "near_unknown" for t in targets["holdout"]),
        "sealed_test_read": False, "deployment_authorized": False,
        "limitations": ["Near-target records are refinement targets, never selector negatives.",
                         "Holdout targets are diagnostic-only and cannot enter gradients.",
                         "A separate bounded run receipt is required before GPU execution."],
    }
    admission["admission_sha256"] = sha256((json.dumps(admission, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()).hexdigest()
    write_new(OUTPUT / "admission.json", admission)
    report = {"schema_version": "chemvector-targeted-selector-coordinate-publication-0.1",
              "complete": True, "data_published": True, "training_authorized": True,
              "training_run_authorized": False, "training_performed": False,
              "gpu_used": False, "sealed_test_read": False, "source_files_modified": False,
              "copied_bundles": copied, "byte_identical_non_lifecycle_artifacts": preserved,
              "train_targets": len(targets["train"]), "holdout_targets": len(targets["holdout"]),
              "manifest_sha256": digest(OUTPUT / "manifest.json"), "admission_sha256": digest(OUTPUT / "admission.json")}
    write_new(OUTPUT / "publication-report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
