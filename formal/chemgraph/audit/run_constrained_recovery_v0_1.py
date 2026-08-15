#!/usr/bin/env python3
"""Run the constrained recovery v0.1 finite validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

from formal.chemgraph.audit import test_constrained_recovery_v0_1
from formal.chemgraph.evidence_core_v0_4.candidate.candidate import build_candidate_set


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evidence_core_v0_4" / "candidate" / "fixtures"
LEAN_SOURCE = ROOT / "ChemGraph" / "ConstrainedRecovery.lean"
TEST_SOURCE = Path(test_constrained_recovery_v0_1.__file__).resolve()
IMPLEMENTATION_SOURCE = Path(__file__).with_name("constrained_recovery_v0_1.py")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_decisions() -> dict[str, int]:
    counts = {"Graph": 0, "Ambiguous": 0, "Reject": 0}
    for path in sorted(FIXTURES.glob("*.evidence.json")):
        result = build_candidate_set(json.loads(path.read_bytes()))
        counts[result["decode_kind"]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "audit" / "results" / "constrained_recovery_v0_1" / "validation.json",
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_constrained_recovery_v0_1)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    payload = {
        "claims": {
            "bond_bundle_uses_two_semantic_endpoints": True,
            "equal_visual_scores_remain_ambiguous": True,
            "fan_outer_shortcut_forbidden": True,
            "path_shortcut_forbidden": True,
            "strict_two_epsilon_margin_is_sufficient": True,
            "two_epsilon_boundary_can_tie": True,
        },
        "fixture_decisions": fixture_decisions(),
        "limitations": [
            "finite synthetic fixtures only",
            "does not prove raster primitive extraction coverage",
            "does not prove learned score calibration",
            "does not authorize model training",
        ],
        "schema_version": "chemgraph-constrained-recovery-validation-v0.1",
        "source_sha256": {
            "implementation": sha256(IMPLEMENTATION_SOURCE),
            "lean": sha256(LEAN_SOURCE),
            "tests": sha256(TEST_SOURCE),
        },
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests": {
            "errors": len(result.errors),
            "failures": len(result.failures),
            "run": result.testsRun,
        },
        "training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
