#!/usr/bin/env python3
"""Replay the frozen v0.10.15 rule on the completed confirm action cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time

import torch

from chemvector.audit_safe_atom_suppression_oracle import DATA_COMMITMENT, DATA_ROOT
from chemvector.audit_structural_twin_certificate import (
    RANKING_SHA256,
    _sha256_file,
    _summarize,
    evaluate_structural_twin_samples,
)
from chemvector.confirm_structural_twin_certificate import _safe_precision_lower
from chemvector.safe_atom_action_model import SafeAtomActionConfig, SafeAtomActionModel
from chemvector.structural_twin_certificate import StructuralTwinCertificateConfig


DEFAULT_MANIFEST = (
    "outputs/candidates/chemvector-v01015-structural-twin/candidate_manifest.json"
)
DEFAULT_CACHE = (
    "outputs/caches/chemvector-v01015-route-confirm-actions/base-actions.pt"
)
DEFAULT_OUTPUT = (
    "outputs/audits/chemvector-v01015-structural-twin-route-confirm/report.json"
)
FROZEN_MANIFEST_COMMITMENT = (
    "0bff0cf89b05bf26d2f9826ffb5e9b102a93761e4a9c7f39cc7a9a542bed642d"
)
FROZEN_CACHE_SHA256 = (
    "5a270c727efc8bcfbcb0ac68b5bc2f97db244816318240316b41bbcb28dd730e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _verify_frozen_inputs(args) -> tuple[dict, dict, str]:
    manifest_path = Path(args.candidate_manifest)
    manifest_file_sha = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded = manifest.pop("manifest_sha256", None)
    if embedded != FROZEN_MANIFEST_COMMITMENT:
        raise ValueError("Recovery manifest commitment differs")
    if hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest() != embedded:
        raise ValueError("Recovery manifest content differs")
    if manifest.get("certificate") != asdict(StructuralTwinCertificateConfig()):
        raise ValueError("Recovery certificate differs")
    if manifest.get("ranking_checkpoint_sha256") != RANKING_SHA256:
        raise ValueError("Recovery ranking checkpoint differs")
    if _sha256_file(args.cache) != FROZEN_CACHE_SHA256:
        raise ValueError("Recovery confirm cache differs")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("schema_version") != "structural-twin-confirm-cache-0.1":
        raise ValueError("Recovery cache schema differs")
    if cache.get("data_commitment_sha256") != DATA_COMMITMENT:
        raise ValueError("Recovery cache data commitment differs")
    if cache.get("candidate_manifest_sha256") != embedded:
        raise ValueError("Recovery cache references another candidate")
    if cache.get("input_samples") != 42_000:
        raise ValueError("Recovery cache sample count differs")
    if len(cache.get("samples", [])) + len(cache.get("coordinate_collisions", [])) != 42_000:
        raise ValueError("Recovery cache coverage differs")
    if stat.S_IMODE((DATA_ROOT / "route_confirm").stat().st_mode) != 0:
        raise RuntimeError("route-confirm must remain sealed during cache replay")
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise RuntimeError("Tracked worktree must be clean before cache replay")
    return manifest, cache, manifest_file_sha


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Confirm replay output already exists: {output}")
    started = time.perf_counter()
    manifest, cache, manifest_file_sha = _verify_frozen_inputs(args)
    ranking_path = manifest["ranking_checkpoint"]
    if _sha256_file(ranking_path) != RANKING_SHA256:
        raise ValueError("Recovery ranking checkpoint content differs")
    ranking_payload = torch.load(ranking_path, map_location="cpu", weights_only=False)
    ranking = SafeAtomActionModel(SafeAtomActionConfig(**ranking_payload["config"]))
    ranking.load_state_dict(ranking_payload["model"])
    ranking.eval().requires_grad_(False)
    samples = cache["samples"]
    collisions = cache["coordinate_collisions"]
    prepared, certified, decisions = evaluate_structural_twin_samples(
        samples,
        ranking,
        presence_threshold=float(cache["presence_threshold"]),
        config=StructuralTwinCertificateConfig(),
    )
    close_molecules = {
        sample["molecule_id"]
        for sample in [*samples, *collisions]
        if sample["minimum_target_distance_px"] <= 16.0
    }
    all_molecules = {sample["molecule_id"] for sample in samples}
    all_molecules.update(row["molecule_id"] for row in collisions)
    general_molecules = all_molecules - close_molecules
    forced = {
        "all": len(collisions),
        "close_distinct": sum(
            row["molecule_id"] in close_molecules for row in collisions
        ),
        "general": sum(
            row["molecule_id"] in general_molecules for row in collisions
        ),
    }
    metrics = {
        name: _summarize(
            decisions, molecule_ids=molecules, forced_failures=forced[name]
        )
        for name, molecules in (
            ("all", all_molecules),
            ("close_distinct", close_molecules),
            ("general", general_molecules),
        )
    }
    overall = metrics["all"]
    safe_lower = _safe_precision_lower(
        overall["executed_actions"], overall["harmful_actions"]
    )
    zero_loss_fields = (
        "harmful_actions",
        "atom_break",
        "skeleton_break",
        "lost_matched_atom_ids",
        "lost_expressible_bond_ids",
        "lost_covered_bond_ids",
        "lost_typed_bond_ids",
    )
    structure_gate = {
        "passed": overall["executed_actions"] >= 2_995
        and all(overall[name] == 0 for name in zero_loss_fields)
        and all(
            all(metrics[bucket][name] == 0 for name in zero_loss_fields)
            for bucket in ("close_distinct", "general")
        )
        and safe_lower >= 0.999
        and overall["atom_exact_delta"] >= 0.03
        and overall["skeleton_exact_delta"] >= 0.03,
        "minimum_executed_actions": 2_995,
        "minimum_safe_precision_lower_95": 0.999,
        "safe_precision_lower_95": safe_lower,
        "minimum_atom_exact_delta": 0.03,
        "minimum_skeleton_exact_delta": 0.03,
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": "structural-twin-route-confirm-report-0.1",
        "status": (
            "STRUCTURE-CONFIRM-GO"
            if structure_gate["passed"]
            else "STRUCTURE-CONFIRM-NO-GO"
        ),
        "candidate_manifest": {
            "path": args.candidate_manifest,
            "commitment_sha256": FROZEN_MANIFEST_COMMITMENT,
            "file_sha256": manifest_file_sha,
            "original_code_git_revision": manifest["code_git_revision"],
        },
        "recovery": {
            "reason": "float32-cdist-empty-witness-at-16px-boundary",
            "metrics_seen_before_recovery": False,
            "route_confirm_reopened": False,
            "replay_code_git_revision": revision,
            "cache_sha256": FROZEN_CACHE_SHA256,
        },
        "certificate": asdict(StructuralTwinCertificateConfig()),
        "cache": {
            "path": args.cache,
            "sha256": FROZEN_CACHE_SHA256,
            "cached_samples": len(samples),
            "coordinate_collision_failures": len(collisions),
        },
        "action_audit": {
            "candidate_actions": len(prepared["features"]),
            "certified_actions": int(certified.sum()),
            "certified_safe_actions": int(
                (certified & prepared["safe_targets"]).sum()
            ),
            "certified_unsafe_actions": int(
                (certified & ~prepared["safe_targets"]).sum()
            ),
        },
        "metrics": metrics,
        "structure_gate": structure_gate,
        "route_confirm_read": True,
        "route_confirm_mode_restored": "000",
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output, report)
    compact = {
        name: {key: value for key, value in value.items() if key != "cases"}
        for name, value in metrics.items()
    }
    print(
        json.dumps(
            {
                "status": report["status"],
                "action_audit": report["action_audit"],
                "metrics": compact,
                "structure_gate": structure_gate,
                "recovery": report["recovery"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
