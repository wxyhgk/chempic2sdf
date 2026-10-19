"""Read-only compatibility audit for legacy G3 endpoint-field evidence.

P6-E2/E6/E8 are useful evidence about multi-candidate endpoint supervision,
but their caches were produced from a different P6-E1 checkpoint and a
different source manifest.  This report compares lineage and tensor contracts
with the frozen production ``initial.pt``.  It deliberately reads only audit
JSON plus the train/calibration cache headers; the protected hard holdout is
never opened and no model training is started.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.pair_scale_training import source_checks


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
P6E2 = ROOT / "outputs/audits/chemvector-g3-p6e2-candidate-peak-reachability/audit.json"
P6E6 = ROOT / "outputs/audits/chemvector-g3-p6e6-unselected-hard-pool/audit.json"
P6E8 = ROOT / "outputs/audits/chemvector-g3-p6e8-residual-cycle-data/audit.json"
P6E8_TRAIN = ROOT / "outputs/audits/chemvector-g3-p6e8-residual-cycle-data/splits/hard_train.pt"
P6E8_CAL = ROOT / "outputs/audits/chemvector-g3-p6e8-residual-cycle-data/splits/hard_calibration.pt"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-g3-endpoint-compatibility-20261006"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _tensor_header(path: Path) -> dict:
    # Only train/calibration are intentionally opened.  Do not add a holdout
    # path here: its filesystem protection is part of the sealed-data contract.
    cache = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(cache["sample_key"]),
        "tensor_shapes": {key: list(value.shape) for key, value in cache.items() if hasattr(value, "shape")},
        "tensor_dtypes": {key: str(value.dtype) for key, value in cache.items() if hasattr(value, "dtype")},
        "metadata": {
            key: cache[key]
            for key in ("p6e1_checkpoint_sha256", "eligible_manifest_sha256", "vision_checkpoint_sha256")
            if key in cache
        },
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    current_sha = file_digest(INITIAL)
    if current_sha != CURRENT_SHA:
        raise RuntimeError(f"Current production initial changed: {current_sha} != {CURRENT_SHA}")
    p6e2, p6e6, p6e8 = _json(P6E2), _json(P6E6), _json(P6E8)
    train_header, calibration_header = _tensor_header(P6E8_TRAIN), _tensor_header(P6E8_CAL)
    lineage = p6e8["lineage"]

    # Current source manifest is read through the existing source contract;
    # identities are compared only as hashes/counts, never against sealed data.
    current_manifest = source_checks()["manifest"]
    current_keys = {row["sample_key"] for row in current_manifest["samples"]}
    legacy_train_keys = set(torch.load(P6E8_TRAIN, map_location="cpu")["sample_key"])
    legacy_shapes = train_header["tensor_shapes"]
    legacy_is_current_field = (
        legacy_shapes.get("image_crops", [])[2:] == [768, 768]
        and legacy_shapes.get("candidate_offsets", [0, 0])[1] == 128
    )
    legacy_has_selected_identity = "selected_baseline_ids" in legacy_shapes

    checks = {
        "current_initial_sha_matches": current_sha == CURRENT_SHA,
        "p6e1_checkpoint_matches_current_initial": lineage["p6e1_checkpoint_sha256"] == CURRENT_SHA,
        "legacy_vision_lineage_matches_current_initial": lineage["vision_checkpoint_sha256"] == CURRENT_SHA,
        "legacy_train_identity_overlaps_current_manifest": bool(legacy_train_keys & current_keys),
        "train_calibration_headers_match_each_other": train_header["metadata"] == calibration_header["metadata"],
        "legacy_candidate_contract_is_current_192_field": legacy_is_current_field,
        "legacy_production_baseline_is_immutable_selected_set": legacy_has_selected_identity,
    }
    compatibility = all(checks.values())
    result = {
        "schema_version": "chemvector-g3-endpoint-compatibility-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "current_production": {
            "initial_path": str(INITIAL),
            "initial_sha256": current_sha,
            "source_manifest_samples": len(current_manifest["samples"]),
            "atom_field_contract": "768px input -> frozen 192x192 heatmap -> 128 raw/selected proposals -> 6px assignment radius",
        },
        "legacy_evidence": {
            "p6e2": {
                "audit": str(P6E2),
                "decision": p6e2["decision"],
                "lineage": p6e2["lineage"],
                "protocol": p6e2["protocol"],
                "holdout_summary": p6e2["summaries"]["holdout"],
            },
            "p6e6": {
                "audit": str(P6E6),
                "decision": p6e6["decision"],
                "lineage": p6e6["lineage"],
                "capacity": p6e6["capacity"],
                "protocol": p6e6["protocol"],
            },
            "p6e8": {
                "audit": str(P6E8),
                "decision": p6e8["decision"],
                "lineage": lineage,
                "capacity": {key: p6e8["capacity"][key] for key in ("images", "hard", "easy", "molecules", "scaffolds", "hard_splits", "hard_variants", "counts")},
                "protocol": p6e8["protocol"],
                "train_cache": train_header,
                "calibration_cache": calibration_header,
            },
        },
        "checks": checks,
        "compatibility": {
            "direct_bounded_trial_admissible": compatibility,
            "reason": "legacy P6 evidence is not byte-compatible with current initial/manifest/selected baseline contract",
            "reusable_as": [
                "multi-candidate endpoint supervision design",
                "hard/easy residual-cycle sampling and scaffold-isolated splits",
                "train-only cap selection protocol (legacy cap=2.0)",
            ],
            "not_reusable_as": [
                "legacy cached logits/offsets as current model inputs",
                "legacy checkpoint weights or vision features",
                "legacy target IDs as current baseline ownership",
            ],
        },
        "minimal_compatible_direction": {
            "stage_0_cpu": "regenerate current-initial endpoint records on a disjoint train-only manifest; store raw IDs, selected IDs, 192-cell coordinates, residual ownership, and candidate provenance",
            "stage_1_shadow": "freeze selected baseline; train/evaluate a zero-start endpoint sidecar only after forward-only candidate parity has positive labels and old-selected retention is 100%",
            "stage_2_bounded_trial": "at most one short trial, with train positives>=50, train AUPRC>=0.50, train precision>=0.85, holdout new matches>=14, and no baseline loss",
            "stage_3_training": "only then scale hard/easy residual-cycle data regenerated from current initial; never import P6-E8 cache tensors directly",
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "checks": checks, "compatibility": result["compatibility"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
