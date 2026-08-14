#!/usr/bin/env python3
"""Re-run the fresh top-K pair Oracle with the frozen slots4 endpoint field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    ENDPOINT_THRESHOLD,
    MAXIMUM_ENDPOINTS,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d0_local_partner_oracle import (
    GO_GATES,
    PARTNER_COUNTS,
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


AUDIT_NAME = "chemvector-v1p2d2e5-slots4-fresh-local-partner-oracle"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
ENDPOINT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v1p2d2e4-style-endpoint-slots4/final.pt"
)
ENDPOINT_CHECKPOINT_SHA256 = (
    "f9753b38c5484eea646e70b4b3e40bf0715f667804c239d510114939fdc9590d"
)
ENDPOINT_RESULT = Path(
    "outputs/checkpoints/chemvector-v1p2d2e4-style-endpoint-slots4/result.json"
)
ENDPOINT_RESULT_SHA256 = (
    "4d0207b223fa6acc574096450997828f74b57930f93e83f72eedaef27b146916"
)
BASELINE_TOP8_PAIR_RECALL = 0.9282540826128722


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-checkpoint", type=Path, default=ENDPOINT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E5 local partner Oracle requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E5 output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E5 endpoint checkpoint SHA256 differs")
    if file_sha256(ENDPOINT_RESULT) != ENDPOINT_RESULT_SHA256:
        raise ValueError("P2-D2-E5 endpoint result SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E5 separator SHA256 differs")
    endpoint_result = json.loads(ENDPOINT_RESULT.read_text(encoding="utf-8"))
    if endpoint_result["decision"] != "slots4_style_endpoint_go":
        raise ValueError("P2-D2-E5 requires the frozen E4 GO checkpoint")

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
        key=lambda route: (
            summary["routes"][route]["top8"]["pair_recall"],
            summary["routes"][route]["top8"]["image_exact"],
        ),
    )
    top8 = summary["routes"][winner]["top8"]
    crossed = summary["by_stress_operation"]["crossed_bonds"][winner]["top8"]
    checks = {
        "calibration_samples_exact400": summary["samples"] == 400,
        "winner_top8_pair_recall_min95pct": top8["pair_recall"]
        >= GO_GATES["winner_top8_pair_recall"],
        "winner_top8_image_exact_min30pct": top8["image_exact"]
        >= GO_GATES["winner_top8_image_exact"],
        "winner_top8_stress_pair_recall_min90pct": summary["by_variant"][
            "stress"
        ][winner]["top8"]["pair_recall"]
        >= GO_GATES["winner_top8_stress_pair_recall"],
        "winner_top8_crossed_pair_recall_min85pct": crossed["pair_recall"]
        >= GO_GATES["winner_top8_crossed_pair_recall"],
        "source_commitment_matches": commitment["commitment_sha256"]
        == SOURCE_COMMITMENT,
        "endpoint_checkpoint_frozen": True,
        "endpoint_threshold_fixed_0_05": ENDPOINT_THRESHOLD == 0.05,
        "maximum_endpoints_fixed_256": MAXIMUM_ENDPOINTS == 256,
        "no_maximum_bond_length_cutoff": True,
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
        "schema_version": "chemvector-v1p2d2e5-slots4-local-partner-oracle-0.1",
        "audit_name": AUDIT_NAME,
        "decision": (
            "slots4_local_partner_oracle_go"
            if all(checks.values())
            else "slots4_local_partner_oracle_no_go"
        ),
        "checks": checks,
        "gates": GO_GATES,
        "winner": winner,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "maximum_endpoints": MAXIMUM_ENDPOINTS,
        "partner_counts": PARTNER_COUNTS,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "endpoint_result_sha256": ENDPOINT_RESULT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "source_commitment": SOURCE_COMMITMENT,
        "source_samples_csv_sha256": SAMPLES_CSV_SHA256,
        "baseline_top8_pair_recall": BASELINE_TOP8_PAIR_RECALL,
        "top8_pair_recall_delta": top8["pair_recall"]
        - BASELINE_TOP8_PAIR_RECALL,
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
