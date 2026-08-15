#!/usr/bin/env python3
"""Run constrained recovery v0.2 finite specification checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

from formal.chemgraph.audit import test_constrained_recovery_v0_2


ROOT = Path(__file__).resolve().parents[1]
LEAN_SOURCE = ROOT / "ChemGraph" / "ConstrainedRecoveryV02.lean"
IMPLEMENTATION_SOURCE = Path(__file__).with_name("constrained_recovery_v0_2.py")
TEST_SOURCE = Path(test_constrained_recovery_v0_2.__file__).resolve()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "audit" / "results" / "constrained_recovery_v0_2" / "validation.json",
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_constrained_recovery_v0_2)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    payload = {
        "gates": {
            "arbitrary_interval_without_calibration_rejected": True,
            "candidate_coverage_required_for_graph": True,
            "edge_provenance_required_for_graph": True,
            "exact_lane_count_required": True,
            "fan_chord_low_support_rejected": True,
            "lane_support_mutually_exclusive": True,
            "one_primitive_one_owner": True,
            "path_shortcut_zero_support_rejected": True,
            "support_pixels_and_segments_required": True,
        },
        "limitations": [
            "finite executable specification only",
            "coverage certificates are not yet produced by an independent real-image enumerator",
            "calibration evidence is synthetic test evidence only",
            "real raster candidate recall was not measured",
            "model training is not authorized",
        ],
        "real_image_validation": "not_run",
        "schema_version": "chemgraph-constrained-recovery-validation-v0.2",
        "source_sha256": {
            "implementation": sha256(IMPLEMENTATION_SOURCE),
            "lean": sha256(LEAN_SOURCE),
            "tests": sha256(TEST_SOURCE),
        },
        "status": "passed_finite_specification" if result.wasSuccessful() else "failed",
        "tests": {
            "errors": len(result.errors),
            "failures": len(result.failures),
            "run": result.testsRun,
        },
        "training_authorized": False,
        "v0_1_disposition": "no_go_retained_as_redteam_evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
