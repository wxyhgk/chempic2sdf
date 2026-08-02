#!/usr/bin/env python3
"""Measure the OOF ceiling of label-oracle multi-anchor component selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_junction_veto_oof as consensus_helper
import chemvector.train_evidence_carrier_pair_local_oof as helper
import chemvector.train_evidence_carrier_pair_local_veto_oof as veto_helper
from chemvector.build_evidence_carrier_pair_local_oof import file_sha256
from chemvector.evidence_carrier_pair_components import (
    apply_multi_component_label_oracle,
    pair_component_metadata,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_local_residual import (
    EvidenceCarrierPairLocalResidual,
)
from chemvector.evidence_carrier_pair_local_spatial_residual import (
    EvidenceCarrierPairLocalSpatialResidual,
)
from chemvector.evidence_carrier_pair_profile_veto import (
    EvidenceCarrierPairProfileVeto,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


SCHEMA_VERSION = "evidence-carrier-pair-component-oracle-oof-audit-0.1"
STATUS_COMPLETE = "PAIR-COMPONENT-ORACLE-OOF-AUDIT-COMPLETE"
DEFAULT_PROFILE_CHECKPOINT_DIR = (
    "outputs/checkpoints/chemvector-v0923-pair-profile-veto-oof"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/chemvector-v0925-pair-component-oracle-oof/audit.json"
)


def _load_profile(path: Path, device: torch.device) -> EvidenceCarrierPairProfileVeto:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = EvidenceCarrierPairProfileVeto()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    base_index, base_metadata = base.load_oof_index(
        args.base_index, audit_path=args.base_audit
    )
    local_index, local_metadata = helper.load_local_index(args.local_index)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tau = float(SAMPLE_THRESHOLD_BASELINE_TAU)
    profile_rows = []
    component_rows = []
    full_rows = []
    multi_pair_candidates = 0
    multi_component_ids = set()
    checkpoint_references = []
    for fold in range(3):
        records, _ = base.load_oof_fold(base_index, fold)
        crops, _ = helper.load_local_fold(local_index, fold)
        records = helper.merge_pair_local_records(records, crops)
        avg_path = (
            Path(args.avg_checkpoint_dir).expanduser().resolve()
            / f"fold{fold}"
            / "final.pt"
        )
        spatial_path = (
            Path(args.spatial_checkpoint_dir).expanduser().resolve()
            / f"fold{fold}"
            / "final.pt"
        )
        profile_path = (
            Path(args.profile_checkpoint_dir).expanduser().resolve()
            / f"fold{fold}"
            / "final.pt"
        )
        avg_model, avg_sha = consensus_helper._load_teacher(
            EvidenceCarrierPairLocalResidual, avg_path, device
        )
        spatial_model, spatial_sha = consensus_helper._load_teacher(
            EvidenceCarrierPairLocalSpatialResidual, spatial_path, device
        )
        profile_model = _load_profile(profile_path, device)
        checkpoint_references.append(
            {
                "fold": fold,
                "avg": {"path": str(avg_path), "sha256": avg_sha},
                "spatial": {"path": str(spatial_path), "sha256": spatial_sha},
                "profile": {
                    "path": str(profile_path),
                    "sha256": file_sha256(profile_path),
                },
            }
        )
        for start in range(0, len(records), 32):
            current = records[start : start + 32]
            batch = helper._to_device(helper.collate_pair_local_records(current), device)
            avg = helper._forward(avg_model, batch)
            spatial = helper._forward(spatial_model, batch)
            profile_output = veto_helper._forward(profile_model, batch)
            for image_index, record in enumerate(current):
                count = int(batch["pair_count"][image_index])
                consensus = torch.minimum(
                    avg["corrected_gain"][image_index, :count],
                    spatial["corrected_gain"][image_index, :count],
                ).detach().cpu()
                profile_logits = profile_output["veto_logits"][
                    image_index, :count
                ].detach().cpu()
                profile_scores = torch.where(
                    (consensus > tau) & (profile_logits > 0.0),
                    torch.full_like(consensus, tau - 1.0),
                    consensus,
                )
                pairs = torch.as_tensor(
                    record["target_free"]["pair_indices"], dtype=torch.long
                )
                labels = torch.as_tensor(
                    record["target_aware"]["pair_labels"], dtype=torch.long
                )
                metadata = pair_component_metadata(pairs)
                multi = metadata["multi_anchor_component"]
                multi_pair_candidates += int(multi.sum())
                for component_id in metadata["component_ids"][multi].unique().tolist():
                    multi_component_ids.add(
                        (fold, str(record["identity"]["sample_key"]), int(component_id))
                    )
                component_scores = apply_multi_component_label_oracle(
                    profile_scores, pairs, labels, threshold=tau
                )
                full_scores = torch.where(
                    labels == 1,
                    torch.full_like(profile_scores, tau + 1.0),
                    torch.full_like(profile_scores, tau - 1.0),
                )
                target = base._evaluation_target(record)
                profile_rows.append(
                    evaluate_evidence_carrier_pair_sample(
                        target, profile_scores, threshold=tau, verify_permutation=False
                    )
                )
                component_rows.append(
                    evaluate_evidence_carrier_pair_sample(
                        target,
                        component_scores,
                        threshold=tau,
                        verify_permutation=False,
                    )
                )
                full_rows.append(
                    evaluate_evidence_carrier_pair_sample(
                        target, full_scores, threshold=tau, verify_permutation=False
                    )
                )
        del avg_model, spatial_model, profile_model, records
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_COMPLETE,
        "profile_metrics": aggregate_evidence_carrier_pair_evaluations(profile_rows),
        "multi_component_oracle_metrics": aggregate_evidence_carrier_pair_evaluations(
            component_rows
        ),
        "full_label_oracle_metrics": aggregate_evidence_carrier_pair_evaluations(
            full_rows
        ),
        "multi_component_count": len(multi_component_ids),
        "multi_pair_candidate_count": multi_pair_candidates,
        "base_index_metadata": base_metadata,
        "local_index_metadata": local_metadata,
        "checkpoint_references": checkpoint_references,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(output, report)
    return {**report, "output_path": str(output), "output_sha256": file_sha256(output)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", default=base.DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--base-audit", default=base.DEFAULT_AUDIT)
    parser.add_argument("--local-index", default=helper.DEFAULT_LOCAL_INDEX)
    parser.add_argument(
        "--avg-checkpoint-dir", default=consensus_helper.AVG_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--spatial-checkpoint-dir", default=consensus_helper.SPATIAL_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--profile-checkpoint-dir", default=DEFAULT_PROFILE_CHECKPOINT_DIR
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
