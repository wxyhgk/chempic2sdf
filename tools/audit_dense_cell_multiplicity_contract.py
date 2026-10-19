"""CPU-only contract audit for a future dense-cell multiplicity shadow.

This audit consumes the read-only pixel/geometry diagnostic and never loads a
model or proposes a new candidate.  It counts collision-positive evidence,
checks the required multi-slot schema, and fails closed when hard negatives or
protected selected-prefix evidence are unavailable.  A passing report would
be a prerequisite for (not an authorization of) a tiny multiplicity-head GPU
shadow.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-pixel-geometry-diagnostic-20261017-r1/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/report.json"
MANIFEST = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/manifest.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

REQUIRED_CANDIDATE_FIELDS = (
    "sample_key", "split", "slot", "cell_id", "x_px", "y_px", "score",
    "source", "protected_old_peak", "selected_baseline",
)
REQUIRED_COLLISION_FIELDS = (
    "sample_key", "cell_id", "target_count", "proposal_count",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(*, diagnostic: Path = DIAGNOSTIC, manifest: Path | None = None) -> dict:
    if manifest is not None:
        manifest = manifest.resolve()
    diagnostic = diagnostic.resolve()
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    records = payload.get("records")
    errors: list[str] = []
    if not isinstance(records, list) or not records:
        errors.append("diagnostic records are missing")
        records = []
    if payload.get("training_performed") is not False:
        errors.append("diagnostic is not read-only")
    if payload.get("sealed_test_read") is not False:
        errors.append("diagnostic accessed sealed data")
    if payload.get("initial_sha256") != INITIAL_SHA or payload.get("initial_sha256_after") != INITIAL_SHA:
        errors.append("diagnostic initial SHA binding is invalid")

    train = [row for row in records if row.get("split") == "train"]
    holdout = [row for row in records if row.get("split") == "holdout"]
    collision_train = [row for row in train if row.get("raw_matched") is False and
                       row.get("nearest_dense_distance_px") is not None and
                       float(row["nearest_dense_distance_px"]) <= 16.0]
    collision_holdout = [row for row in holdout if row.get("raw_matched") is False and
                         row.get("nearest_dense_distance_px") is not None and
                         float(row["nearest_dense_distance_px"]) <= 16.0]
    protected_records = [row for row in records if row.get("raw_matched") is True]
    profiles = Counter(str(row.get("profile_id")) for row in records)
    train_profiles = Counter(str(row.get("profile_id")) for row in collision_train)
    holdout_profiles = Counter(str(row.get("profile_id")) for row in collision_holdout)

    # Candidate evidence is a separate, CPU-only forward manifest.  Keeping it
    # optional preserves the fail-closed target-only audit used by unit tests;
    # the release audit passes --manifest explicitly.
    manifest_payload = None
    if manifest is not None and manifest.exists():
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("training_performed") is not False or manifest_payload.get("gpu_used") is not False:
            errors.append("candidate manifest is not CPU-only read-only evidence")
        if manifest_payload.get("sealed_test_read") is not False:
            errors.append("candidate manifest accessed sealed data")
        if manifest_payload.get("initial_sha256") != INITIAL_SHA or manifest_payload.get("initial_sha256_after") != INITIAL_SHA:
            errors.append("candidate manifest initial SHA binding is invalid")
        if manifest_payload.get("diagnostic_sha256") != digest(diagnostic):
            errors.append("candidate manifest is not bound to this diagnostic")

    candidate_fields_present = set((manifest_payload or {}).get("candidate_fields", ()))
    collision_fields_present = set((manifest_payload or {}).get("collision_fields", ()))
    missing_candidate_fields = sorted(set(REQUIRED_CANDIDATE_FIELDS) - candidate_fields_present)
    missing_collision_fields = sorted(set(REQUIRED_COLLISION_FIELDS) - collision_fields_present)
    hard_negative_count = int((manifest_payload or {}).get("hard_negative_count", 0))
    hard_negative_precision = (manifest_payload or {}).get("hard_negative_precision")
    old_selected_retention = (manifest_payload or {}).get("old_selected_retention")
    all_atom_strata = bool((manifest_payload or {}).get("all_atom_strata"))
    candidate_rows = (manifest_payload or {}).get("candidate_records", [])
    collision_rows = (manifest_payload or {}).get("collision_groups", [])
    candidate_schema_complete = bool(candidate_rows) and all(
        all(field in row for field in REQUIRED_CANDIDATE_FIELDS) for row in candidate_rows
    )
    collision_schema_complete = bool(collision_rows) and all(
        all(field in row for field in REQUIRED_COLLISION_FIELDS) for row in collision_rows
    )
    if candidate_schema_complete:
        missing_candidate_fields = []
    if collision_schema_complete:
        missing_collision_fields = []
    # Recompute safety metrics from candidate rows instead of trusting summary
    # values.  Hard negatives are explicitly defined as top-128 dense peaks
    # farther than the fixed 6 px production radius from every target; the
    # distance is post-hoc audit evidence, never model input.
    hard_negative_rows = [r for r in candidate_rows if r.get("hard_negative") is True]
    hard_negative_count_actual = len(hard_negative_rows)
    hard_negative_true_negative_actual = sum(
        float(r.get("nearest_target_distance_px", float("inf"))) > 6.0
        for r in hard_negative_rows
    )
    hard_negative_precision_actual = (
        hard_negative_true_negative_actual / hard_negative_count_actual
        if hard_negative_count_actual else None
    )
    # Baseline retention is evaluated on raw decoded slots only.  Dense peaks
    # are shadow candidates and can never enlarge the immutable baseline
    # prefix, even if they happen to share a cell with a selected slot.
    protected_rows = [r for r in candidate_rows
                      if r.get("source") == "raw_proposal" and r.get("selected_baseline") is True]
    retained_rows = [r for r in protected_rows if r.get("protected_old_peak") is True]
    old_selected_retention_actual = len(retained_rows) / len(protected_rows) if protected_rows else None
    # A malformed summary cannot silently turn into a passing gate.
    if manifest_payload is not None:
        if int(manifest_payload.get("hard_negative_count", -1)) != hard_negative_count_actual:
            errors.append("candidate manifest hard-negative count does not match rows")
        if int(manifest_payload.get("old_selected_total", -1)) != len(protected_rows):
            errors.append("candidate manifest protected count does not match rows")
    hard_negative_count = hard_negative_count_actual
    hard_negative_precision = hard_negative_precision_actual
    old_selected_retention = old_selected_retention_actual

    checks = {
        "read_only_source": not errors,
        "train_collision_positives_ge_50": len(collision_train) >= 50,
        "holdout_collision_positives_present": len(collision_holdout) > 0,
        "multi_slot_candidate_schema": not missing_candidate_fields and candidate_schema_complete,
        "collision_group_schema": not missing_collision_fields and collision_schema_complete,
        "all_atom_strata_present": all_atom_strata,
        "hard_negative_precision_available": hard_negative_count > 0 and hard_negative_precision is not None,
        "hard_negative_precision_ge_085": hard_negative_precision is not None and float(hard_negative_precision) >= 0.85,
        "protected_old_selected_retention_100": old_selected_retention is not None and float(old_selected_retention) >= 1.0,
    }
    # GPU is authorized only by a future independent receipt after every CPU
    # check passes.  This report itself never grants that receipt.
    result = {
        "schema_version": "chemvector-dense-cell-multiplicity-contract-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "optimizer_steps": 0,
        "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA,
        "diagnostic": str(diagnostic.relative_to(ROOT)),
        "diagnostic_sha256": digest(diagnostic),
        "candidate_manifest": str(manifest.relative_to(ROOT)) if manifest is not None else None,
        "candidate_manifest_sha256": digest(manifest) if manifest is not None and manifest.exists() else None,
        "record_counts": {
            "all": len(records), "train": len(train), "holdout": len(holdout),
            "protected_raw_records": len(protected_records),
            "collision_train": len(collision_train),
            "collision_holdout": len(collision_holdout),
            "collision_train_by_profile": dict(sorted(train_profiles.items())),
            "collision_holdout_by_profile": dict(sorted(holdout_profiles.items())),
            "profiles": dict(sorted(profiles.items())),
            "candidate_records": len(candidate_rows),
            "collision_groups": len(collision_rows),
        },
        "required_candidate_fields": list(REQUIRED_CANDIDATE_FIELDS),
        "required_collision_fields": list(REQUIRED_COLLISION_FIELDS),
        "missing_candidate_fields": missing_candidate_fields,
        "missing_collision_fields": missing_collision_fields,
        "hard_negative_count": hard_negative_count,
        "hard_negative_precision": hard_negative_precision,
        "old_selected_retention": old_selected_retention,
        "checks": checks,
        "gate": {
            "cpu_contract_pass": all(checks.values()),
            "gpu_training_authorized": False,
            "reason": "candidate-level hard negatives, protected selected retention, and complete multi-slot evidence are required",
            "next": ("obtain a separately reviewed GPU shadow receipt; this audit never authorizes GPU"
                     if all(checks.values()) else
                     "prepare an aligned dense candidate/collision manifest; rerun this CPU audit before any GPU receipt"),
        },
        "errors": errors,
        "interpretation": {
            "collision_positive_definition": "raw unmatched target with a frozen dense anchor within 16 px; labels are audit-only",
            "not_selector_rescue": True,
            "protected_prefix_policy": "baseline selected peaks must be immutable in any future sidecar",
            "future_gpu_scope": "new multiplicity/extra-heatmap head only; frozen vision/atom/selector/OCR/bond heads",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="CPU candidate manifest; omitted means target-only fail-closed audit")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(diagnostic=args.diagnostic, manifest=args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "record_counts": result["record_counts"], "checks": result["checks"], "gate": result["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
