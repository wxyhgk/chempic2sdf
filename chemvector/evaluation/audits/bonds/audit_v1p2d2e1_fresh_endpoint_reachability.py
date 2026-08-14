#!/usr/bin/env python3
"""Decompose fresh endpoint coverage from endpoint-pair ranking reachability."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    DEFAULT_ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
    ENDPOINT_THRESHOLD,
    MAXIMUM_ENDPOINTS,
    ROUTES,
    run_audit,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d2e0_fresh_candidate_oracle import (
    SOURCE_COMMITMENT,
    SAMPLES_CSV_SHA256,
    _fresh_loader,
    load_fresh_split,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p2d2e1-fresh-endpoint-reachability"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P2D2E0_AUDIT = Path(
    "outputs/audits/chemvector-v1p2d2e0-fresh-candidate-oracle/audit.json"
)
P2D2E0_AUDIT_SHA256 = (
    "31c49f8fb9a52aeaabf0a1b9f3092ed28841e8c9637d180ba22753b94df45298"
)
GATES = {
    "candidate_unique_port_recall": 0.97,
    "exhaustive_pair_recall": 0.97,
}


def route_decision(summary: dict) -> str:
    winner = max(
        ROUTES,
        key=lambda route: summary["routes"][route]["full"]["pair_recall"],
    )
    if (
        summary["candidate_unique_port_recall"]
        >= GATES["candidate_unique_port_recall"]
        and summary["routes"][winner]["full"]["pair_recall"]
        >= GATES["exhaustive_pair_recall"]
    ):
        return "partner_selection_blocker"
    return "endpoint_reachability_blocker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-checkpoint", type=Path, default=DEFAULT_ENDPOINT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E1 fresh endpoint audit requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E1 output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E1 endpoint checkpoint SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E1 separator SHA256 differs")
    if file_sha256(P2D2E0_AUDIT) != P2D2E0_AUDIT_SHA256:
        raise ValueError("P2-D2-E1 prerequisite audit SHA256 differs")
    prerequisite = json.loads(P2D2E0_AUDIT.read_text(encoding="utf-8"))
    if prerequisite["decision"] != "fresh_candidate_oracle_no_go":
        raise ValueError("P2-D2-E1 requires the frozen P2-D2-E0 NO-GO result")

    calibration, _, commitment = load_fresh_split()
    device = torch.device("cuda")
    separator_payload = torch.load(DEFAULT_SEPARATOR, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    endpoint_payload = torch.load(args.endpoint_checkpoint, map_location=device)
    detector = SemanticBondEndpointDetector(
        SemanticBondEndpointConfig(**endpoint_payload["config"])
    ).to(device)
    detector.load_state_dict(endpoint_payload["detector"], strict=True)
    detector.eval().requires_grad_(False)

    started = time.perf_counter()
    summary, rows = run_audit(
        separator,
        detector,
        calibration,
        device,
        workers=args.workers,
        log_every=args.log_every,
        loader_factory=_fresh_loader,
    )
    winner = max(
        ROUTES,
        key=lambda route: summary["routes"][route]["full"]["pair_recall"],
    )
    checks = {
        "calibration_samples_exact400": summary["samples"] == 400,
        "candidate_unique_port_recall_min97pct": summary[
            "candidate_unique_port_recall"
        ]
        >= GATES["candidate_unique_port_recall"],
        "exhaustive_pair_recall_min97pct": summary["routes"][winner]["full"][
            "pair_recall"
        ]
        >= GATES["exhaustive_pair_recall"],
        "same_calibration_split_as_p2d2e0": True,
        "source_commitment_matches": commitment["commitment_sha256"]
        == SOURCE_COMMITMENT,
        "separator_endpoint_and_threshold_frozen": True,
        "fresh_holdout_samples_read_zero": True,
        "sealed_test_samples_read_zero": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "schema_version": "chemvector-v1p2d2e1-fresh-endpoint-reachability-0.1",
        "audit_name": AUDIT_NAME,
        "decision": route_decision(summary),
        "checks": checks,
        "gates": GATES,
        "winner": winner,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "maximum_endpoints": MAXIMUM_ENDPOINTS,
        "pair_config": asdict(SemanticEndpointPairConfig()),
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "source_commitment": SOURCE_COMMITMENT,
        "source_samples_csv_sha256": SAMPLES_CSV_SHA256,
        "p2d2e0_audit_sha256": P2D2E0_AUDIT_SHA256,
        "p2d2e0_top8_pair_recall": prerequisite["summary"]["routes"][
            prerequisite["winner"]
        ]["top8"]["pair_recall"],
        "rows_sha256": file_sha256(rows_path),
        "summary": summary,
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
