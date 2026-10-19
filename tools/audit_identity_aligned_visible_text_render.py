#!/usr/bin/env python3
"""Read-only integrity gate for the rendered identity-aligned O/N panel.

This checks only the quarantine renderer outputs.  It does not load a model,
read sealed data, or authorize training.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-visible-text-render-20261017-r4/report.json"
ALLOWED_REJECTION = "Typography clearance would erase a bond segment"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run() -> dict:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    accepted = read_jsonl(DATA / "samples.jsonl")
    rejected = read_jsonl(DATA / "rejected.jsonl")
    if summary.get("complete") is not True or summary.get("training_authorized") is not False:
        raise ValueError("renderer is not a complete quarantine-only artifact")
    if summary.get("sealed_test_read") is not False or summary.get("gpu_used") is not False:
        raise ValueError("renderer contract violation")
    if len(accepted) != summary["accepted_images"] or len(rejected) != summary["rejected_images"]:
        raise ValueError("summary/jsonl accounting mismatch")
    if len(accepted) + len(rejected) != summary["planned_images"]:
        raise ValueError("planned image accounting mismatch")
    if any(row.get("render_status") != "accepted" for row in accepted):
        raise ValueError("accepted list contains non-accepted row")
    if any(row.get("render_status") != "quality_rejected" or row.get("reason") != ALLOWED_REJECTION for row in rejected):
        raise ValueError("unexpected rejection reason")
    train_ids = {row["molecule_id"] for row in accepted if row["split"] == "train"}
    holdout_ids = {row["molecule_id"] for row in accepted if row["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise ValueError("molecule identity overlap")
    if any(not isinstance(row.get("connection_identity"), str) or not row["connection_identity"] for row in accepted):
        raise ValueError("accepted row has empty connection identity")
    train_scaffolds = {row.get("scaffold") for row in accepted if row["split"] == "train"}
    holdout_scaffolds = {row.get("scaffold") for row in accepted if row["split"] == "holdout"}
    if None in train_scaffolds or None in holdout_scaffolds or train_scaffolds & holdout_scaffolds:
        raise ValueError("scaffold identity overlap or missing scaffold")
    samples = DATA / "samples"
    missing = []
    observed_min_target_anchor_separation = float("inf")
    for row in accepted:
        key = row["sample_key"]
        # Each accepted render must retain all source representations and the
        # layer/metadata contracts used by the later adapter audit.
        for suffix in (".png", ".svg", ".sdf", ".layers.json", ".meta.json", ".text-records.json"):
            if not (samples / (key + suffix)).exists():
                missing.append(key + suffix)
        layers = json.loads((samples / (key + ".layers.json")).read_text(encoding="utf-8"))
        visible = [a for a in layers["atoms"] if a.get("element") in {"O", "N"} and a.get("visible_text")]
        all_atoms = layers["atoms"]
        for atom in visible:
            for other in all_atoms:
                if atom["atom_id"] != other["atom_id"]:
                    observed_min_target_anchor_separation = min(
                        observed_min_target_anchor_separation,
                        math.dist(atom["center"], other["center"]),
                    )
    if missing:
        raise ValueError(f"missing accepted artifacts: {missing[:8]}")
    profiles = Counter(row["profile_id"] for row in accepted)
    splits = Counter(row["split"] for row in accepted)
    report = {
        "schema_version": "chemvector-identity-aligned-visible-text-render-audit-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "training_authorized": False,
        "gpu_used": False,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "sealed_test_read": False,
        "data_dir": str(DATA),
        "summary_sha256": sha256(DATA / "summary.json"),
        "accepted_jsonl_sha256": sha256(DATA / "samples.jsonl"),
        "rejected_jsonl_sha256": sha256(DATA / "rejected.jsonl"),
        "planned_images": len(accepted) + len(rejected),
        "accepted_images": len(accepted),
        "rejected_images": len(rejected),
        "accepted_by_split": dict(sorted(splits.items())),
        "accepted_by_profile": dict(sorted(profiles.items())),
        "train_molecules": len(train_ids),
        "holdout_molecules": len(holdout_ids),
        "molecule_overlap": len(train_ids & holdout_ids),
        "missing_artifacts": 0,
        "identity_contract_observed": {
            "declared_target_anchor_separation_px": 48,
            "observed_min_target_anchor_separation_px": observed_min_target_anchor_separation,
            "target_anchor_separation_contract_pass": observed_min_target_anchor_separation >= 48 - 1e-6,
            "candidate_radius_px": 16,
            "other_target_min_px": 20,
        },
        "gate": {
            "accounting_exact": True,
            "accepted_artifacts_complete": True,
            "identity_disjoint": True,
            "quality_rejections_bounded": True,
            "declared_anchor_separation_contract": observed_min_target_anchor_separation >= 48 - 1e-6,
            "unique_ownership_radius_margin_available": observed_min_target_anchor_separation > 20,
            "training_admissible": False,
            "next": "identity-aligned raw reachability replay against frozen initial; no training",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
