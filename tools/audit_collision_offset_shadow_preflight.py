"""CPU-only preflight for the collision candidate score/offset shadow.

This intentionally fails closed when near-miss candidates have no aligned
one-to-one target identity.  Nearest-distance rows alone are not sufficient to
train an offset residual safely under Hungarian collisions.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/manifest.json"
PLAN = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r2/manifest.json"
OUTPUT = ROOT / "outputs/audits/chemvector-collision-offset-shadow-20261017-r1/preflight.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))["rows"]
    if manifest.get("training_performed") is not False or manifest.get("gpu_used") is not False:
        raise ValueError("candidate manifest is not CPU-only")
    if manifest.get("sealed_test_read") is not False or manifest.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("candidate manifest binding failed")
    by_molecule: dict[str, set[str]] = defaultdict(set)
    for row in plan:
        by_molecule[str(row["molecule_id"])].add(str(row["split"]))
    molecule_disjoint = all(len(splits) == 1 for splits in by_molecule.values())
    rows = [r for r in manifest.get("candidate_records", []) if r.get("source") == "raw_proposal"
            and not r.get("protected_old_peak")]
    summary = {}
    for split in ("train", "holdout"):
        subset = [r for r in rows if r.get("split") == split]
        refinable = [r for r in subset if 6.0 < float(r.get("nearest_target_distance_px", 1e9)) <= 16.0]
        aligned = [r for r in refinable if r.get("target_id") is not None]
        strict = [r for r in subset if float(r.get("nearest_target_distance_px", 1e9)) <= 6.0]
        summary[split] = {
            "raw_unprotected_candidates": len(subset),
            "nearest_refinable_6_to_16": len(refinable),
            "aligned_refinable_targets": len(aligned),
            "ambiguous_or_unassigned_refinable": len(refinable) - len(aligned),
            "strict_nearby": len(strict),
        }
    checks = {
        "read_only_cpu_manifest": True,
        "molecule_disjoint": molecule_disjoint,
        "train_refinable_positives_ge_50": summary["train"]["aligned_refinable_targets"] >= 50,
        "holdout_refinable_positives_present": summary["holdout"]["aligned_refinable_targets"] > 0,
        "no_target_identity_ambiguity": all(summary[s]["ambiguous_or_unassigned_refinable"] == 0 for s in ("train", "holdout")),
    }
    result = {
        "schema_version": "chemvector-collision-offset-shadow-preflight-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False,
        "manifest_sha256": digest(MANIFEST), "initial_sha256": INITIAL_SHA,
        "panel_plan_sha256": digest(PLAN), "splits": summary, "checks": checks,
        "gate": {
            "cpu_preflight_pass": all(checks.values()),
            "gpu_training_authorized": False,
            "reason": ("aligned one-to-one refinable labels are unavailable; nearest-distance rows are ambiguous under Hungarian collision"
                        if not checks["no_target_identity_ambiguity"] else "all CPU checks passed; independent GPU receipt still required"),
            "next": "construct an identity-aligned collision residual manifest; do not train or authorize GPU",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "splits": summary, "checks": checks, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
